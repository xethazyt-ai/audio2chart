"""Shared training callbacks, reproducibility, and dataset validation."""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch

from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar


logger = logging.getLogger(__name__)
MAX_NOTES = 5000
ERROR_POLICIES = {"strict", "skip"}


def set_seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    L.seed_everything(seed, workers=True)
    logger.info("Set global random seed to %d", seed)


class LogGradientNorm(L.pytorch.callbacks.Callback):
    """Log the global L2 gradient norm before clipping."""

    def on_before_optimizer_step(self, trainer: L.Trainer, *_: Any, **__: Any) -> None:
        squared_norm = sum(
            parameter.grad.detach().norm(2).item() ** 2
            for parameter in trainer.lightning_module.parameters()
            if parameter.grad is not None
        )
        trainer.lightning_module.log("train/grad_norm", squared_norm ** 0.5)


def _validate_policy(error_policy: str) -> None:
    if error_policy not in ERROR_POLICIES:
        raise ValueError(f"error_policy must be one of {sorted(ERROR_POLICIES)}")


def _validate_encoded_chart(
    processor: ChartProcessor,
    tokenizer: SimpleTokenizerGuitar,
    section: str,
    grid_ms: int,
) -> None:
    if not processor.song_metadata:
        raise ValueError("Chart has no Song metadata")
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    notes = processor.notes[section]
    if not 0 < len(notes) < MAX_NOTES:
        raise ValueError(f"Section {section} has an unsupported note count: {len(notes)}")
    encoded = tokenizer.encode(notes, resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    if len(timed) > 1:
        minimum_delta = min(
            current[0] - previous[0]
            for previous, current in zip(timed, timed[1:])
        )
        if minimum_delta <= grid_ms / 1000.0:
            raise ValueError(
                f"Section {section} contains events closer than the {grid_ms} ms grid"
            )


def validate_dataset(
    data: Iterable[dict[str, Any]],
    difficulties: list[str],
    instruments: list[str],
    grid_ms: int,
    error_policy: str = "strict",
) -> list[dict[str, Any]]:
    """Validate audio manifest entries before constructing a dataset."""
    _validate_policy(error_policy)
    processor = ChartProcessor(difficulties, instruments)
    tokenizer = SimpleTokenizerGuitar()
    valid: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        try:
            chart_path = item["chart_path"]
            section = item["difficulty"]
            if not isinstance(chart_path, str) or not isinstance(section, str):
                raise TypeError("chart_path and difficulty must be strings")
            processor.read_chart(chart_path, target_sections=section)
            _validate_encoded_chart(processor, tokenizer, section, grid_ms)
            valid.append(item)
        except (KeyError, OSError, TypeError, ValueError) as error:
            if error_policy == "strict":
                raise ValueError(f"Invalid manifest entry at index {index}") from error
            logger.warning("Skipping invalid manifest entry %d: %s", index, error)
    if not valid:
        raise ValueError("Dataset contains no valid entries")
    logger.info("Validated %d dataset entries", len(valid))
    return valid


def validate_dataset_notes(
    data: Iterable[str | Path],
    difficulties: list[str],
    instruments: list[str],
    grid_ms: int,
    error_policy: str = "strict",
) -> list[str]:
    """Validate chart files used by the autoregressive baseline."""
    _validate_policy(error_policy)
    processor = ChartProcessor(difficulties, instruments)
    tokenizer = SimpleTokenizerGuitar()
    valid: list[str] = []
    for path_value in data:
        path = str(path_value)
        try:
            processor.read_chart(path)
            if not processor.notes:
                raise ValueError("Chart has no requested note sections")
            for section in processor.notes:
                _validate_encoded_chart(processor, tokenizer, section, grid_ms)
            valid.append(path)
        except (KeyError, OSError, TypeError, ValueError) as error:
            if error_policy == "strict":
                raise ValueError(f"Invalid chart: {path}") from error
            logger.warning("Skipping invalid chart %s: %s", path, error)
    if not valid:
        raise ValueError("Dataset contains no valid charts")
    logger.info("Validated %d chart files", len(valid))
    return valid
