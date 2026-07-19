"""Compatibility/performance benchmark: python tests/benchmark_chart_pipeline.py --json."""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chart.chart_processor import ChartProcessor
from chart.time_conversion import preprocess_bpm_segments, tick_to_seconds


def reference_sections(text, names):
    result = {}
    for name in names:
        match = re.search(rf"\[{re.escape(name)}\]\s*\{{(.*?)\}}", text, re.DOTALL)
        if match:
            result[name] = match.group(1).strip()
    return result


def reference_power_flags(ticks, intervals):
    return [any(start <= tick < end for start, end in intervals) for tick in ticks]


def ordered_power_flags(ticks, intervals):
    intervals, index, flags = sorted(intervals), 0, []
    for tick in ticks:
        while index < len(intervals) and intervals[index][1] <= tick:
            index += 1
        flags.append(index < len(intervals) and intervals[index][0] <= tick < intervals[index][1])
    return flags


def reference_tick_lookup(ticks, segments, resolution):
    return [tick_to_seconds(tick, segments, resolution) for tick in ticks]


def cached_tick_lookup(ticks, segments, resolution):
    segment_ticks = [segment[0] for segment in segments]
    return [tick_to_seconds(tick, segments, resolution, segment_ticks) for tick in ticks]


def synthetic_chart(events):
    notes = "\n".join(f"  {i * 12} = N {i % 5} {i % 97}" for i in range(events))
    return f"[Song]\n{{\n Resolution = 192\n Offset = 0\n}}\n[SyncTrack]\n{{\n 0 = B 120000\n}}\n[ExpertSingle]\n{{\n{notes}\n}}"


def measure(function, *args, repeats=5):
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed)


def run_case(events):
    text = synthetic_chart(events)
    processor = ChartProcessor("Expert", "Single")
    names = processor.sections
    processor.open_chart(None, chart_text=text)
    assert reference_sections(text, names) == processor.extract_sections()
    ticks = list(range(0, events * 12, 12))
    intervals = [(tick, tick + 96) for tick in ticks[::20]]
    assert reference_power_flags(ticks, intervals) == ordered_power_flags(ticks, intervals)
    bpm = [(tick, 90000 + (index % 4) * 30000) for index, tick in enumerate(ticks[::50])]
    segments = preprocess_bpm_segments(bpm or [(0, 120000)], 192)
    assert reference_tick_lookup(ticks, segments, 192) == cached_tick_lookup(ticks, segments, 192)
    timings = {
        "parse": (measure(reference_sections, text, names), measure(processor.extract_sections)),
        "star_power": (measure(reference_power_flags, ticks, intervals), measure(ordered_power_flags, ticks, intervals)),
        "bpm": (measure(reference_tick_lookup, ticks, segments, 192), measure(cached_tick_lookup, ticks, segments, 192)),
    }
    return {name: {"reference_seconds": old, "refactored_seconds": new,
                   "ratio": new / old if old else 0} for name, (old, new) in timings.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-ratio", type=float)
    args = parser.parse_args()
    result = {name: run_case(count) for name, count in
              (("small", 100), ("typical", 5000), ("large", 25000))}
    print(json.dumps(result, indent=2) if args.json else "\n".join(
        f"{size:7} {algorithm:10} ratio={data['ratio']:.3f}"
        for size, case in result.items() for algorithm, data in case.items()))
    if args.max_ratio is not None and any(
            data["ratio"] > args.max_ratio for case in result.values() for data in case.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
