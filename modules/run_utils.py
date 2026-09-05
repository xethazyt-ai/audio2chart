"""Shared training entry-point utilities."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO
from omegaconf import DictConfig, OmegaConf

from modules.utils_train import LogGradientNorm


logger = logging.getLogger(__name__)


class DirectCheckpointIO(TorchCheckpointIO):
    """Serialize checkpoints straight to disk instead of through a memory buffer.

    Lightning's `_atomic_save` builds the entire checkpoint in a BytesIO before writing
    it, so saving costs twice the checkpoint's size in RAM. This one is ~2.9 GB (245 M
    parameters plus AdamW's two moments), and the four persistent dataloader workers each
    hold up to 2 GB of cached audio at the same time. Two runs died at exactly this point,
    once with a truncated zip ("unexpected pos ... vs ...") and once with a segfault --
    different symptoms, same cause.

    Writing to a sibling temporary file and renaming keeps the atomicity the buffer was
    there to provide, and the file is checked for a plausible size before the rename so a
    short write surfaces here rather than at resume time.
    """

    MINIMUM_BYTES = 1 << 20

    def save_checkpoint(self, checkpoint: dict[str, Any], path, storage_options=None) -> None:
        if storage_options is not None:
            raise TypeError(
                f"{type(self).__name__} does not accept storage_options={storage_options!r}"
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        try:
            torch.save(checkpoint, temporary)
            written = temporary.stat().st_size
            if written < self.MINIMUM_BYTES:
                raise RuntimeError(f"Checkpoint write produced only {written} bytes")
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        logger.info("Wrote checkpoint %s (%.2f GB)", destination, written / 1e9)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@contextmanager
def experiment_logger(config: DictConfig, run_name: str) -> Iterator[object]:
    """Create an optional W&B logger and always finalize the active run."""
    if not config.tracking.enabled:
        yield CSVLogger(save_dir="lightning_logs", name=run_name)
        return
    wandb.init(
        project=config.tracking.project,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        name=run_name,
        tags=list(config.tracking.tags),
        reinit=True,
    )
    try:
        yield WandbLogger(log_model=False)
    finally:
        wandb.finish()


def build_callbacks(config: DictConfig, monitor: str, mode: str = "max") -> list[object]:
    callbacks: list[object] = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=monitor,
            min_delta=0.0001,
            patience=config.trainer.early_stopping_patience,
            mode=mode,
        ),
        LogGradientNorm(),
    ]
    if config.trainer.save_run:
        callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
            monitor=monitor,
            save_top_k=1,
            mode=mode,
            filename="best-checkpoint",
            # A monitored checkpoint is only written when validation runs. On a corpus
            # this size an epoch is over a day, so without save_last a crash twenty hours
            # in would leave nothing to resume from. last.ckpt is written unconditionally.
            save_last=True,
        ))
    return callbacks


def build_trainer(config: DictConfig, logger: object, monitor: str) -> L.Trainer:
    use_gpu = config.trainer.gpus > 0
    # val_check_interval also sets how often a checkpoint can be written, because the
    # monitored ModelCheckpoint only fires on validation. Left unset, Lightning validates
    # once per epoch, which on this corpus is over a day between save points.
    limits = {
        name: OmegaConf.select(config, f"trainer.{name}")
        for name in ("limit_train_batches", "limit_val_batches", "val_check_interval")
    }
    return L.Trainer(
        max_epochs=config.trainer.max_epochs,
        max_steps=OmegaConf.select(config, "trainer.max_steps", default=-1),
        **{name: value for name, value in limits.items() if value is not None},
        accelerator="gpu" if use_gpu else "cpu",
        devices=config.trainer.gpus if use_gpu else 1,
        plugins=[DirectCheckpointIO()],
        enable_checkpointing=bool(config.trainer.save_run),
        callbacks=build_callbacks(config, monitor),
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=logger,
        precision=config.trainer.precision,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    )
