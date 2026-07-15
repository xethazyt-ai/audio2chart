"""Dataset discovery and deterministic grouped splitting utilities."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar


logger = logging.getLogger(__name__)
MAX_NOTES = 5000
AUDIO_CANDIDATES = ("song.opus", "song.ogg", "guitar.opus", "guitar.ogg")


def find_chart_files(root_folder: str | Path) -> list[str]:
    root = Path(root_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Chart root does not exist: {root}")
    return sorted(str(path) for path in root.rglob("*.chart"))


def find_audio_files(
    root: str | Path,
    difficulties: Iterable[str],
    instruments: Iterable[str],
    output_json: str | Path = "results/audio_dataset.json",
    skipped_json: str | Path = "results/audio_skipped.json",
) -> tuple[str, str]:
    """Discover valid chart/audio pairs and write dataset manifests."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    processor = ChartProcessor(list(difficulties), list(instruments))
    tokenizer = SimpleTokenizerGuitar()
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        chart_path = directory / "notes.chart"
        if not chart_path.is_file():
            continue
        audio_path = next(
            (directory / name for name in AUDIO_CANDIDATES if (directory / name).is_file()),
            None,
        )
        if audio_path is None:
            skipped.append({"path": str(directory), "reason": "missing_audio"})
            continue
        try:
            processor.read_chart(str(chart_path))
            if not processor.song_metadata or not processor.synctrack:
                raise ValueError("Chart is missing Song metadata or SyncTrack events")
            resolution = int(processor.song_metadata["Resolution"])
            offset = float(processor.song_metadata["Offset"])
            for section, notes in processor.notes.items():
                if not 0 < len(notes) < MAX_NOTES:
                    continue
                encoded = tokenizer.encode(notes)
                tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
                entries.append({
                    "audio_path": str(audio_path),
                    "chart_path": str(chart_path),
                    "difficulty": section,
                })
        except (KeyError, OSError, TypeError, ValueError) as error:
            skipped.append({
                "path": str(directory),
                "reason": "chart_validation_error",
                "error": str(error),
            })

    _write_json(output_json, entries)
    _write_json(skipped_json, skipped)
    logger.info("Discovered %d entries; skipped %d directories", len(entries), len(skipped))
    return str(output_json), str(skipped_json)


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _grouped_split(
    entries: list[dict[str, Any]],
    group_key: str,
    val_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between zero and one")
    if not entries:
        raise ValueError("Cannot split an empty dataset")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        group = entry.get(group_key)
        difficulty = entry.get("difficulty")
        if not isinstance(group, str) or not group:
            raise ValueError(f"Entry {index} has no valid {group_key!r}")
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError(f"Entry {index} has no valid 'difficulty'")
        groups[group].append(entry)

    group_names = sorted(groups)
    if len(group_names) < 2:
        raise ValueError("At least two distinct audio groups are required for a split")
    random.Random(random_seed).shuffle(group_names)
    val_count = min(len(group_names) - 1, max(1, round(len(group_names) * val_ratio)))
    val_groups = set(group_names[:val_count])
    train_groups = set(group_names[val_count:])
    train = [entry for group in sorted(train_groups) for entry in groups[group]]
    val = [entry for group in sorted(val_groups) for entry in groups[group]]
    return train, val, train_groups, val_groups


def _split_manifest(
    input_json: str | Path,
    train_json: str | Path,
    val_json: str | Path,
    group_key: str,
    val_ratio: float,
    random_seed: int,
) -> tuple[set[str], set[str]]:
    with Path(input_json).open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError("Dataset manifest must contain a JSON list")
    train, val, train_groups, val_groups = _grouped_split(
        entries, group_key, val_ratio, random_seed
    )
    _write_json(train_json, train)
    _write_json(val_json, val)
    logger.info(
        "Split %d groups into %d train and %d validation groups",
        len(train_groups) + len(val_groups), len(train_groups), len(val_groups),
    )
    return train_groups, val_groups


def split_json_entries_by_audio(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[set[str], set[str]]:
    return _split_manifest(
        input_json, train_json, val_json, "audio_path", val_ratio, random_seed
    )


def split_json_entries_by_audio_raw(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[set[str], set[str]]:
    return _split_manifest(
        input_json, train_json, val_json, "raw_path", val_ratio, random_seed
    )
