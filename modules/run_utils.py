"""Shared training entry-point utilities."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import lightning as L
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from modules.utils_train import LogGradientNorm


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
        ))
    return callbacks


def build_trainer(config: DictConfig, logger: object, monitor: str) -> L.Trainer:
    use_gpu = config.trainer.gpus > 0
    return L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=config.trainer.gpus if use_gpu else 1,
        enable_checkpointing=bool(config.trainer.save_run),
        callbacks=build_callbacks(config, monitor),
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=logger,
        precision=config.trainer.precision,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
    )
