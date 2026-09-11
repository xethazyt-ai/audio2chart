"""Fit a tier grader from the charts whose setlist folders name a tier.

801 charts in the library sit under folders like "[FH2] TIER 8", spanning tiers 1 to
26. That is a supervised target for the scale Robert means -- difficulty within Expert,
where Through The Fire & Flames is single digits and Schmoo's World is around 24.

Two cautions built into the output rather than left implicit:

*The labels come from different setlist authors*, so tier 8 in one pack need not mean
tier 8 in another. The report breaks error down by pack, because a model that fits well
within packs and badly across them means the scales disagree and need calibrating
against each other, not that the features are wrong.

*Held-out error is the only number worth reading.* Fitting ten features to 801 points
will look excellent in-sample whatever the truth is.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.chart_processor import ChartProcessor
from chart.discover import catalogue_lift
from chart.metrics import profile_from_timed
from chart.patterns import single_note_runs
from chart.tier import FEATURE_NAMES, features
from chart.tokenizer import SimpleTokenizerGuitar


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path(r"G:\a2c_data\tier_labels.json"))
    parser.add_argument("--cache", type=Path, default=Path(r"G:\a2c_data\tier_features.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def extract(chart_path: str, tokenizer, section: str = "ExpertSingle"):
    processor = ChartProcessor(["Expert"], ["Single"])
    processor.read_chart(chart_path, target_sections=section)
    notes = processor.notes.get(section)
    if not notes:
        return None
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    encoded = tokenizer.encode(notes, resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    profile = profile_from_timed(timed, tokenizer)
    if profile.positions < 100:
        return None
    runs = [[fret for _, fret in run] for run in single_note_runs(timed, tokenizer)]
    coverage, floor = catalogue_lift(runs, shuffles=2)
    row = features(profile, [item[0] for item in timed], runs, marker_share=coverage - floor)
    return row


def solve(matrix: list[list[float]], target: list[float], ridge: float = 1e-3):
    """Least squares with a small ridge, in plain Python -- no new dependency."""
    width = len(matrix[0])
    normal = [[sum(row[i] * row[j] for row in matrix) + (ridge if i == j else 0.0)
               for j in range(width)] for i in range(width)]
    rhs = [sum(row[i] * value for row, value in zip(matrix, target)) for i in range(width)]
    for column in range(width):
        pivot = max(range(column, width), key=lambda r: abs(normal[r][column]))
        normal[column], normal[pivot] = normal[pivot], normal[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        if abs(normal[column][column]) < 1e-12:
            continue
        for row in range(width):
            if row == column:
                continue
            factor = normal[row][column] / normal[column][column]
            for col in range(column, width):
                normal[row][col] -= factor * normal[column][col]
            rhs[row] -= factor * rhs[column]
    return [rhs[i] / normal[i][i] if abs(normal[i][i]) > 1e-12 else 0.0 for i in range(width)]


def main():
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.limit:
        labels = labels[:args.limit]

    cache = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    tokenizer = SimpleTokenizerGuitar(expressive=True)
    rows = []
    for index, entry in enumerate(labels, 1):
        path = entry["chart_path"]
        if path in cache:
            row = cache[path]
        else:
            try:
                row = extract(path, tokenizer)
            except Exception:
                row = None
            cache[path] = row
            if index % 25 == 0:
                args.cache.write_text(json.dumps(cache), encoding="utf-8")
                print(f"  extracted {index}/{len(labels)}", flush=True)
        if row:
            rows.append((entry["tier"], row, path))
    args.cache.write_text(json.dumps(cache), encoding="utf-8")

    print(f"\nusable charts: {len(rows)} of {len(labels)}")
    if len(rows) < 50:
        print("too few to fit"); return

    # Correlation of each feature with tier, before any fitting.
    tiers = [t for t, _, _ in rows]
    print(f"\n{'feature':<16}{'corr with tier':>16}")
    for name in FEATURE_NAMES:
        values = [r[name] for _, r, _ in rows]
        if statistics.pstdev(values) == 0:
            print(f"{name:<16}{'constant':>16}"); continue
        mean_v, mean_t = statistics.mean(values), statistics.mean(tiers)
        cov = sum((v - mean_v) * (t - mean_t) for v, t in zip(values, tiers)) / len(values)
        corr = cov / (statistics.pstdev(values) * statistics.pstdev(tiers))
        print(f"{name:<16}{corr:>+16.3f}")

    # Cross-validated fit. In-sample error on 10 features and 801 points means nothing.
    random.Random(0).shuffle(rows)
    folds = args.folds
    errors, baseline_errors = [], []
    for fold in range(folds):
        test = [r for i, r in enumerate(rows) if i % folds == fold]
        train = [r for i, r in enumerate(rows) if i % folds != fold]
        matrix = [[r[n] for n in FEATURE_NAMES] + [1.0] for _, r, _ in train]
        weights = solve(matrix, [t for t, _, _ in train])
        mean_tier = statistics.mean([t for t, _, _ in train])
        for tier, row, _ in test:
            predicted = sum(w * v for w, v in
                            zip(weights, [row[n] for n in FEATURE_NAMES] + [1.0]))
            errors.append(abs(predicted - tier))
            baseline_errors.append(abs(mean_tier - tier))

    print(f"\nheld-out mean absolute error {statistics.mean(errors):.2f} tiers")
    print(f"  predicting the mean instead  {statistics.mean(baseline_errors):.2f} tiers")
    print(f"  within 2 tiers: {100*sum(1 for e in errors if e <= 2)/len(errors):.0f}%"
          f"   within 4: {100*sum(1 for e in errors if e <= 4)/len(errors):.0f}%")

    # Do the packs agree with each other?
    by_pack = defaultdict(list)
    for (tier, row, path), error in zip(rows, errors[:len(rows)]):
        pack = path.split("\\")[4] if len(path.split("\\")) > 5 else "?"
        by_pack[pack].append(error)
    print("\nerror by setlist (do the packs share a scale?)")
    for pack, values in sorted(by_pack.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"  {pack[:44]:<46}{len(values):>4} charts  MAE {statistics.mean(values):.2f}")


if __name__ == "__main__":
    main()
