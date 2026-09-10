"""Musical bars, for segmenting a chart the way a charter reads it.

Robert: charters think in beats and bars, bars especially for dense patterns, and patterns
should be looked for bar by bar. The reference's own grid-step rule says the same thing --
"a pattern's rhythmic subdivision must strictly dictate its note count per bar".

That matters because the alternative is a heuristic. `chart.patterns.single_note_runs`
splits on chords and on a fixed 250 ms gap, which has no musical meaning: it can join two
patterns that happen to be close together and split one that straddles a rest. Bars are
where the author actually placed the boundaries.

ChartProcessor keeps only the BPM events from [SyncTrack] and discards the TS lines, so
this module reads them from the file directly.
"""

from __future__ import annotations

import re
from pathlib import Path

TS_LINE = re.compile(r"^\s*(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?\s*$")
DEFAULT_DENOMINATOR_EXPONENT = 2
"""A .chart TS line may omit the denominator; 2 means 2**2 = 4, i.e. x/4 time."""


def parse_time_signatures(chart_path: str | Path) -> list[tuple[int, int, int]]:
    """[(tick, numerator, denominator)], ascending. Always starts at tick 0."""
    events: list[tuple[int, int, int]] = []
    in_sync = False
    with Path(chart_path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("["):
                in_sync = stripped == "[SyncTrack]"
                continue
            if not in_sync:
                continue
            match = TS_LINE.match(line)
            if match:
                tick, numerator, exponent = match.groups()
                power = int(exponent) if exponent is not None else DEFAULT_DENOMINATOR_EXPONENT
                events.append((int(tick), int(numerator), 2 ** power))
    events.sort()
    if not events or events[0][0] != 0:
        events.insert(0, (0, 4, 4))          # Clone Hero's default is 4/4
    return events


def bar_length_ticks(numerator: int, denominator: int, resolution: int) -> int:
    """Ticks in one bar. `resolution` is ticks per quarter note."""
    return int(resolution * 4 * numerator / denominator)


def bar_boundaries(chart_path: str | Path, resolution: int, last_tick: int) -> list[int]:
    """Tick of every bar line from 0 through `last_tick`.

    Time signature changes restart the bar grid at the tick where they occur, which is what
    Moonscraper draws and therefore what the charter was looking at.
    """
    events = parse_time_signatures(chart_path)
    boundaries: list[int] = []
    for index, (tick, numerator, denominator) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else last_tick + 1
        length = bar_length_ticks(numerator, denominator, resolution)
        if length <= 0:
            continue
        position = tick
        while position <= min(end - 1, last_tick):
            boundaries.append(position)
            position += length
    return boundaries


def group_by_bar(encoded, boundaries: list[int]) -> list[list]:
    """Split `tokenizer.encode(...)` output into one list per bar.

    Bars with no notes are kept as empty lists so an index is a bar number.
    """
    if not boundaries:
        return [list(encoded)]
    bars: list[list] = [[] for _ in boundaries]
    index = 0
    for event in encoded:
        tick = event[0]
        while index + 1 < len(boundaries) and tick >= boundaries[index + 1]:
            index += 1
        while index > 0 and tick < boundaries[index]:
            index -= 1
        bars[index].append(event)
    return bars


def bars_for_chart(chart_path: str | Path, encoded, resolution: int) -> list[list]:
    """Convenience: boundaries from the file, then group."""
    if not encoded:
        return []
    last_tick = max(event[0] for event in encoded)
    return group_by_bar(encoded, bar_boundaries(chart_path, resolution, last_tick))
