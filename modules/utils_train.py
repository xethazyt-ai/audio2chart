"""Shared training callbacks, reproducibility, and dataset validation."""

from __future__ import annotations

import json
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
# One ceiling, not two. This module used to define its own MAX_NOTES = 5000 while
# discovery used 50000, so every dense chart discovery admitted was silently rejected
# again at training time -- undoing the filter fix without a word in the logs.
from dataloader.utils_dataloader import MAX_NOTES


logger = logging.getLogger(__name__)
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


def usable_window_count(times: list[float], grid_ms: int, window_seconds: float) -> int:
    """How many of the chart's windows can be discretized without a collision.

    The runtime constraint is per window, not per chart: `discretize_time` raises only
    when the window it was handed contains two events closer than one grid step, and the
    dataset windows a song at multiples of `window_seconds` from zero. Judging a whole
    song by its single tightest transition therefore discards four minutes of usable
    chart because of one fast run — measured over this corpus, 17.1% of songs.

    Note the comparison is `<`, matching `discretize_time`; a chart whose events land
    exactly one grid step apart is fine, and quantized charts produce those constantly.
    """
    if not times:
        return 0
    grid_seconds = grid_ms / 1000.0
    windows: dict[int, list[float]] = {}
    for time_value in times:
        windows.setdefault(int(time_value // window_seconds), []).append(time_value)
    usable = 0
    for events in windows.values():
        if len(events) < 2:
            usable += 1
            continue
        events.sort()
        if min(b - a for a, b in zip(events, events[1:])) >= grid_seconds:
            usable += 1
    return usable


def _validate_encoded_chart(
    processor: ChartProcessor,
    tokenizer: SimpleTokenizerGuitar,
    section: str,
    grid_ms: int,
    window_seconds: float | None = None,
    max_notes: int = MAX_NOTES,
) -> None:
    if not processor.song_metadata:
        raise ValueError("Chart has no Song metadata")
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    notes = processor.notes[section]
    if not 0 < len(notes) < max_notes:
        raise ValueError(f"Section {section} has an unsupported note count: {len(notes)}")
    encoded = tokenizer.encode(notes, resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    if len(timed) <= 1:
        return
    times = [event[0] for event in timed]
    if window_seconds:
        if usable_window_count(times, grid_ms, window_seconds) == 0:
            raise ValueError(
                f"Section {section} has no window free of events closer than the "
                f"{grid_ms} ms grid"
            )
        return
    # No window size to work with: fall back to judging the whole section, still using
    # the same `<` boundary the runtime uses.
    minimum_delta = min(b - a for a, b in zip(times, times[1:]))
    if minimum_delta < grid_ms / 1000.0:
        raise ValueError(
            f"Section {section} contains events closer than the {grid_ms} ms grid"
        )


def _cache_signature(
    difficulties: list[str], instruments: list[str], grid_ms: int,
    window_seconds: float | None, max_notes: int,
) -> dict[str, Any]:
    """Everything that changes a verdict. A mismatch discards the whole cache."""
    return {
        "difficulties": sorted(difficulties),
        "instruments": sorted(instruments),
        "grid_ms": grid_ms,
        "window_seconds": window_seconds,
        "max_notes": max_notes,
    }


def _load_cache(path: Path | None, signature: dict[str, Any]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            blob = json.load(stream)
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable validation cache %s: %s", path, error)
        return {}
    if blob.get("signature") != signature:
        logger.info("Validation cache signature changed; revalidating from scratch")
        return {}
    entries = blob.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_cache(path: Path | None, signature: dict[str, Any], entries: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump({"signature": signature, "entries": entries}, stream)
    except OSError as error:
        logger.warning("Could not write validation cache %s: %s", path, error)


def validate_dataset(
    data: Iterable[dict[str, Any]],
    difficulties: list[str],
    instruments: list[str],
    grid_ms: int,
    error_policy: str = "strict",
    window_seconds: float | None = None,
    cache_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Validate audio manifest entries before constructing a dataset.

    Pass `window_seconds` to judge each song by whether it has any usable window, which
    is the constraint the runtime actually imposes. Without it the whole section has to
    clear the grid, which throws away songs the loader would have handled.

    `cache_path` memoizes the verdicts. Parsing a chart costs ~0.5 s, so a 13,720-entry
    corpus spends about two hours here before the first batch, and pays it again after
    every crash or OOM. A verdict is reused only when the chart's mtime and size are
    unchanged and the settings that produced it still match.
    """
    _validate_policy(error_policy)
    processor = ChartProcessor(difficulties, instruments)
    tokenizer = SimpleTokenizerGuitar()
    signature = _cache_signature(difficulties, instruments, grid_ms, window_seconds, MAX_NOTES)
    cache_file = Path(cache_path) if cache_path else None
    cache = _load_cache(cache_file, signature)

    valid: list[dict[str, Any]] = []
    hits = 0
    for index, item in enumerate(data):
        key = f"{item.get('chart_path')}|{item.get('difficulty')}"
        stamp: list[int] | None = None      # Per iteration: never carry one chart's stamp to the next.
        try:
            chart_path = item["chart_path"]
            section = item["difficulty"]
            if not isinstance(chart_path, str) or not isinstance(section, str):
                raise TypeError("chart_path and difficulty must be strings")

            try:
                info = os.stat(chart_path)
                stamp = [info.st_mtime_ns, info.st_size]
            except OSError:
                pass  # Fall through to a real parse, which will report the problem.

            cached = cache.get(key) if stamp is not None else None
            if cached is not None and cached.get("stamp") == stamp:
                hits += 1
                if cached.get("error"):
                    raise ValueError(cached["error"])
                valid.append(item)
                continue

            processor.read_chart(chart_path, target_sections=section)
            _validate_encoded_chart(processor, tokenizer, section, grid_ms, window_seconds)
            if stamp is not None:
                cache[key] = {"stamp": stamp, "error": None}
            valid.append(item)
        except (KeyError, OSError, TypeError, ValueError) as error:
            if stamp is not None:
                cache[key] = {"stamp": stamp, "error": str(error)}
            if error_policy == "strict":
                raise ValueError(f"Invalid manifest entry at index {index}") from error
            logger.warning("Skipping invalid manifest entry %d: %s", index, error)

    _write_cache(cache_file, signature, cache)
    if not valid:
        raise ValueError("Dataset contains no valid entries")
    logger.info("Validated %d dataset entries (%d from cache)", len(valid), hits)
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
