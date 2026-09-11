"""Score a generated chart against what human charters actually do.

Four independent questions, because "is this a good chart?" is not one measurement:

*What is it made of?* Chord, tap, forced and sustain rates against the per-chart
corpus median. This is where the current model fails hardest -- at the shipped
sampling defaults it emits zero chords, zero taps and almost no sustains despite
predicting all three at near-human rates when teacher-forced.

*Are the shapes real?* Catalogue coverage above the chart's own chance floor. Raw
coverage is not comparable between charts; the floor ranges 13% to 35% depending on
how a chart uses the neck.

*Is it legal?* Universal rule 1, no overlapping sustains. Zero violations in 836,327
human notes, so any violation is a defect -- though a chart with no sustains passes
vacuously, which is worth knowing when reading the result.

*Does it follow the music?* Only with --audio. Note density against the onset
envelope, which is meaningful in aggregate and noisy for one chart (per-song sd 0.284).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.chart_processor import ChartProcessor
from chart.discover import HUMAN_LIFT_MEAN, HUMAN_LIFT_SD, catalogue_lift
from chart.metrics import PER_CHART_MEDIAN, profile_from_timed
from chart.patterns import single_note_runs
from chart.tokenizer import SimpleTokenizerGuitar

NOTE_LINE = re.compile(r"^\s*(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
MAX_FRET = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chart", type=Path)
    parser.add_argument("--section", default="ExpertSingle")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Also score how well note density tracks the audio.")
    return parser.parse_args()


def overlapping_sustains(chart_path: Path) -> tuple[int, int]:
    """Sustains still ringing when the same fret is struck again."""
    per_fret: dict[int, list[tuple[int, int]]] = defaultdict(list)
    inside = False
    with chart_path.open(encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("["):
                inside = stripped == "[ExpertSingle]"
                continue
            if not inside:
                continue
            match = NOTE_LINE.match(line)
            if match:
                tick, fret, length = (int(g) for g in match.groups())
                if fret <= MAX_FRET:
                    per_fret[fret].append((tick, length))
    bad = total = 0
    for items in per_fret.values():
        items.sort()
        for index in range(len(items) - 1):
            tick, length = items[index]
            total += 1
            if length > 0 and tick + length > items[index + 1][0]:
                bad += 1
    return bad, total


def line(name: str, actual: float, target: float, fmt: str = "{:.3f}") -> str:
    delta = actual - target
    flag = "  " if abs(delta) <= max(0.25 * abs(target), 0.02) else " !"
    return (f"  {name:<16}{fmt.format(actual):>10}{fmt.format(target):>10}"
            f"{delta:>+10.3f}{flag}")


def main():
    args = parse_args()
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    processor = ChartProcessor(["Expert"], ["Single"])
    processor.read_chart(str(args.chart), target_sections=args.section)
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    encoded = tokenizer.encode(processor.notes[args.section], resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    profile = profile_from_timed(timed, tokenizer)

    print(f"{args.chart}")
    print(f"  {profile.positions} positions over {profile.duration_seconds:.1f}s\n")
    print(f"  {'metric':<16}{'chart':>10}{'human':>10}{'delta':>10}")
    # Report the chord rate, not the single-note rate. A chart with no chords at all
    # reads as chord_1 1.000 against a human 0.872, and a relative tolerance on a
    # number that close to 1 waves through the most conspicuous defect there is.
    for key, value in (("pct_chord", 1.0 - profile.chord_hist.get(1, 0.0)),
                       ("pct_tap", profile.pct_tap),
                       ("pct_forced", profile.pct_forced),
                       ("pct_sustain", profile.pct_sustain),
                       ("nps", profile.nps)):
        target = (1.0 - PER_CHART_MEDIAN["chord_1"] if key == "pct_chord"
                  else PER_CHART_MEDIAN[key])
        print(line(key, value, target, "{:.2f}" if key == "nps" else "{:.3f}"))

    runs = [[fret for _, fret in run] for run in single_note_runs(timed, tokenizer)]
    coverage, floor = catalogue_lift(runs)
    lift = coverage - floor
    sigmas = (lift - HUMAN_LIFT_MEAN) / HUMAN_LIFT_SD if HUMAN_LIFT_SD else 0.0
    print(f"\n  pattern lift    {lift:>+10.3f}{HUMAN_LIFT_MEAN:>+10.3f}"
          f"{sigmas:>+9.1f}sd")
    print(f"    coverage {coverage:.3f} against a chance floor of {floor:.3f}")

    bad, total = overlapping_sustains(args.chart)
    print(f"\n  overlapping sustains {bad}/{total}"
          + ("  (vacuous: this chart has almost no sustains)"
             if profile.pct_sustain < 0.01 else ""))

    if args.audio:
        import librosa
        from chart.responsiveness import note_density, onset_envelope, responsiveness
        waveform, rate = librosa.load(str(args.audio), sr=24000, mono=True)
        duration = len(waveform) / rate
        score = responsiveness(
            note_density([item[0] for item in timed], duration),
            onset_envelope(waveform, rate))
        print(f"\n  responsiveness  {score:>+10.3f}  "
              f"(human median +0.261; per-song sd 0.284, so one chart says little)")


if __name__ == "__main__":
    main()
