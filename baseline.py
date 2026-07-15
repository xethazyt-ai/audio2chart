"""Train the autoregressive chart-only baseline."""

from __future__ import annotations

import random
from datetime import datetime

import hydra
from omegaconf import DictConfig

from dataloader.notes_loader import create_chart_dataloader
from dataloader.utils_dataloader import find_chart_files
from modules.run_utils import build_trainer, configure_logging, experiment_logger
from modules.trainer import NotesTransformer
from modules.utils_train import set_seed_everything, validate_dataset_notes


MONITORED_METRIC = "val/acc_epoch"


def build_run_name(config: DictConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{config.model.name}_seq{config.data.max_length}_"
        f"d{config.model.d_model}_n{config.model.n_layers}_"
        f"lr{config.optimizer.lr}_bs{config.loader.batch_size}_{timestamp}"
    )


def split_charts(paths: list[str], validation_split: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between zero and one")
    if len(paths) < 2:
        raise ValueError("At least two charts are required for training and validation")
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    val_count = min(len(paths) - 1, max(1, round(len(paths) * validation_split)))
    return paths[val_count:], paths[:val_count]


def build_dataloaders(config: DictConfig):
    chart_files = find_chart_files(config.data.root_folder)
    train_charts, val_charts = split_charts(
        chart_files, config.data.validation_split, config.seed
    )
    validation = dict(
        difficulties=list(config.data.difficulties),
        instruments=list(config.data.instruments),
        grid_ms=config.data.grid_ms,
        error_policy=config.data.error_policy,
    )
    train_charts = validate_dataset_notes(train_charts, **validation)
    val_charts = validate_dataset_notes(val_charts, **validation)
    common = dict(
        difficulties=list(config.data.difficulties),
        instruments=list(config.data.instruments),
        max_length=config.data.max_length,
        num_workers=config.loader.num_workers,
        conditional=config.model.conditional,
        is_discrete=config.data.is_discrete,
        grid_ms=config.data.grid_ms,
        window_seconds=config.data.window_seconds,
        error_policy=config.data.error_policy,
    )
    train_loader, vocab = create_chart_dataloader(
        train_charts,
        batch_size=config.loader.batch_size,
        shuffle=True,
        **common,
    )
    val_loader, _ = create_chart_dataloader(
        val_charts,
        batch_size=config.loader.val_batch_size,
        shuffle=False,
        vocab=vocab,
        **common,
    )
    return train_loader, val_loader, vocab


def run(config: DictConfig) -> None:
    configure_logging(config.logging.level)
    set_seed_everything(config.seed)
    train_loader, val_loader, vocab = build_dataloaders(config)
    model = NotesTransformer(
        pad_token_id=vocab["<PAD>"],
        eos_token_id=vocab["<eos>"],
        vocab_size=len(vocab),
        cfg_model=config.model,
        cfg_optimizer=config.optimizer,
        is_discrete=config.data.is_discrete,
    )
    with experiment_logger(config, build_run_name(config)) as logger:
        build_trainer(config, logger, MONITORED_METRIC).fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


@hydra.main(version_base=None, config_path="configs", config_name="text")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
