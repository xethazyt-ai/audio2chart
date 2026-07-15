import json
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from chart.tokenizer import SimpleTokenizerGuitar
from dataloader.audio_loader import (
    create_chunked_audio_chart_dataloader as create_audio_chart_dataloader,
)
from dataloader.utils_dataloader import split_json_entries_by_audio_raw
from modules.trainer import WaveformTransformerDiscrete
from modules.utils_train import LogGradientNorm, set_seed_everything, validate_dataset


MONITORED_METRIC = "val/acc_nonpad_epoch"


def _config_value(config, key, default):
    value = OmegaConf.select(config, key, default=default)
    return default if value is None else value


def build_run_name(config):
    transformer = config.model.transformer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{config.model.name}_ws{config.window_seconds}_grid{config.grid_ms}_"
        f"seq{config.max_length}_d{transformer.d_model}_n{transformer.n_layers}_"
        f"lr{config.optimizer.lr}_bs{config.batch_size}_{timestamp}"
    )


def load_data_splits(config):
    if config.data_split_folder:
        folder = Path(config.data_split_folder)
    else:
        folder = Path(config.root_folder)
        split_json_entries_by_audio_raw(
            input_json=str(folder / "audio_dataset_with_raw.json"),
            train_json=str(folder / "train.json"),
            val_json=str(folder / "val.json"),
            val_ratio=config.validation_split,
        )
    with (folder / "train.json").open(encoding="utf-8") as stream:
        train_files = json.load(stream)
    with (folder / "val.json").open(encoding="utf-8") as stream:
        val_files = json.load(stream)
    return train_files, val_files


def build_dataloaders(config, train_files, val_files):
    tokenizer = SimpleTokenizerGuitar()
    difficulties, instruments = list(config.diff_list), list(config.inst_list)
    train_files = validate_dataset(train_files, difficulties, instruments, config.grid_ms)
    val_files = validate_dataset(val_files, difficulties, instruments, config.grid_ms)
    common = dict(
        window_seconds=config.window_seconds,
        sample_rate=config.model.sample_rate,
        tokenizer=tokenizer,
        difficulties=difficulties,
        instruments=instruments,
        max_length=config.max_length,
        conditional=config.model.transformer.conditional,
        use_predecoded_raw=True,
        is_discrete=config.is_discrete,
        grid_ms=config.grid_ms,
        use_processor=config.model.use_processor,
        num_workers=int(_config_value(config, "num_workers", 8)),
        processor_checkpoint=_config_value(
            config, "model.encoder.checkpoint", "facebook/encodec_24khz"
        ),
    )
    train_loader, vocab = create_audio_chart_dataloader(
        train_files,
        batch_size=config.batch_size,
        num_pieces=int(_config_value(config, "train_num_pieces", 6)),
        augment=config.augment,
        **common,
    )
    val_loader, _ = create_audio_chart_dataloader(
        val_files,
        batch_size=int(_config_value(config, "val_batch_size", 64)),
        num_pieces=int(_config_value(config, "val_num_pieces", 1)),
        augment=False,
        **common,
    )
    return train_loader, val_loader, vocab


def build_callbacks(config):
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=MONITORED_METRIC,
            min_delta=0.0001,
            patience=int(_config_value(config, "early_stopping_patience", 5)),
            verbose=False,
            mode="max",
        ),
        LogGradientNorm(),
    ]
    if config.save_run:
        callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
            monitor=MONITORED_METRIC, save_top_k=1, mode="max",
            filename="best-checkpoint",
        ))
    return callbacks


def build_trainer(config, logger):
    use_gpu = config.gpus > 0
    return L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=config.gpus if use_gpu else 1,
        enable_checkpointing=bool(config.save_run),
        callbacks=build_callbacks(config),
        log_every_n_steps=int(_config_value(config, "log_every_n_steps", 10)),
        logger=logger,
        precision=config.precision,
        num_sanity_val_steps=int(_config_value(config, "num_sanity_val_steps", 0)),
        gradient_clip_val=float(_config_value(config, "gradient_clip_val", 1.0)),
    )


def run(config):
    set_seed_everything(config.seed)
    wandb.init(
        project="audio2chart",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        name=build_run_name(config), tags=config.tags, reinit=True,
    )
    try:
        train_files, val_files = load_data_splits(config)
        train_loader, val_loader, vocab = build_dataloaders(
            config, train_files, val_files
        )
        model = WaveformTransformerDiscrete(
            pad_token_id=vocab["<PAD>"], eos_token_id=vocab["<eos>"],
            vocab_size=len(vocab), cfg_model=config.model,
            cfg_optimizer=config.optimizer,
        )
        build_trainer(config, WandbLogger(log_model=False)).fit(
            model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    finally:
        wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="audio")
def main(config: DictConfig):
    run(config)


if __name__ == "__main__":
    main()
