"""Train the audio-conditioned discrete chart model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from chart.tokenizer import SimpleTokenizerGuitar
from dataloader.audio_loader import create_chunked_audio_chart_dataloader
from dataloader.utils_dataloader import split_json_entries_by_audio_raw
from modules.run_utils import build_trainer, configure_logging, experiment_logger
from modules.trainer import WaveformTransformerDiscrete
from modules.utils_train import set_seed_everything, validate_dataset


MONITORED_METRIC = "val/acc_nonpad_epoch"


def build_run_name(config: DictConfig) -> str:
    transformer = config.model.transformer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{config.model.name}_ws{config.data.window_seconds}_"
        f"grid{config.data.grid_ms}_seq{config.data.max_length}_"
        f"d{transformer.d_model}_n{transformer.n_layers}_"
        f"lr{config.optimizer.lr}_bs{config.loader.batch_size}_{timestamp}"
    )


def _read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(f"Manifest must contain a JSON list: {path}")
    return value


def load_data_splits(config: DictConfig) -> tuple[list[dict], list[dict]]:
    folder = Path(config.data.split_folder or config.data.root_folder)
    if config.data.split_folder is None:
        split_json_entries_by_audio_raw(
            input_json=str(folder / "audio_dataset_with_raw.json"),
            train_json=str(folder / "train.json"),
            val_json=str(folder / "val.json"),
            val_ratio=config.data.validation_split,
            random_seed=config.seed,
        )
    return _read_manifest(folder / "train.json"), _read_manifest(folder / "val.json")


def build_dataloaders(config: DictConfig, train_files: list[dict], val_files: list[dict]):
    tokenizer = SimpleTokenizerGuitar()
    difficulties = list(config.data.difficulties)
    instruments = list(config.data.instruments)
    validation = dict(
        difficulties=difficulties,
        instruments=instruments,
        grid_ms=config.data.grid_ms,
        error_policy=config.data.error_policy,
    )
    train_files = validate_dataset(train_files, **validation)
    val_files = validate_dataset(val_files, **validation)

    common = dict(
        tokenizer=tokenizer,
        window_seconds=config.data.window_seconds,
        sample_rate=config.model.sample_rate,
        difficulties=difficulties,
        instruments=instruments,
        max_length=config.data.max_length,
        num_workers=config.loader.num_workers,
        conditional=config.model.transformer.conditional,
        use_predecoded_raw=True,
        is_discrete=True,
        grid_ms=config.data.grid_ms,
        use_processor=config.model.use_processor,
        processor_checkpoint=OmegaConf.select(
            config, "model.encoder.checkpoint", default="facebook/encodec_24khz"
        ),
        error_policy=config.data.error_policy,
        max_sample_retries=config.loader.max_sample_retries,
    )
    train_loader, vocab = create_chunked_audio_chart_dataloader(
        train_files,
        batch_size=config.loader.batch_size,
        num_pieces=config.loader.train_num_pieces,
        augment=config.loader.augment,
        **common,
    )
    val_loader, _ = create_chunked_audio_chart_dataloader(
        val_files,
        batch_size=config.loader.val_batch_size,
        num_pieces=config.loader.val_num_pieces,
        augment=False,
        shuffle_chunks=False,
        **common,
    )
    return train_loader, val_loader, vocab


def run(config: DictConfig) -> None:
    configure_logging(config.logging.level)
    set_seed_everything(config.seed)
    train_files, val_files = load_data_splits(config)
    train_loader, val_loader, vocab = build_dataloaders(config, train_files, val_files)
    model = WaveformTransformerDiscrete(
        pad_token_id=vocab["<PAD>"],
        eos_token_id=vocab["<eos>"],
        vocab_size=len(vocab),
        cfg_model=config.model,
        cfg_optimizer=config.optimizer,
    )
    with experiment_logger(config, build_run_name(config)) as logger:
        build_trainer(config, logger, MONITORED_METRIC).fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


@hydra.main(version_base=None, config_path="configs", config_name="audio")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
