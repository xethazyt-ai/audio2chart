"""Difficulty features for grading an Expert chart on the community tier scale.

Robert's correction: Easy/Medium/Hard/Expert is the difficulty selector, not the scale
that matters. Expert charts span an enormous range -- Through The Fire & Flames sits in
single digits while Schmoo's World is around 24, and both are Expert. A model that can
be asked for "roughly a grade 16" needs tier as a number, and tier is a property of the
chart, so it can be measured.

The features here are chosen to separate charts that a density average cannot. The
difficulty ladder across Easy..Expert showed notes per second rising only 1.8x while
mean run length quadrupled, so sustained load and hand movement matter more than
average rate:

*Peak sustained density* -- the busiest ten seconds, since what breaks a run is the
hardest passage, not the mean.

*Endurance* -- the longest unbroken stretch above ten notes a second. Schmoo's World
and TTFAF differ here far more than in average density.

*Hand movement* -- mean fret jump and the share of jumps of three frets or more, which
is what forces a hand reposition rather than a finger roll.

*Technique mix* -- tap and chord rates, and the share of pattern instances that are
graded Expert markers.

No model is fitted here; this produces the inputs. `fit_tier.py` fits against the 801
charts whose setlist folders name a tier.
"""

from __future__ import annotations

import statistics

PEAK_WINDOW = 10.0
"""Seconds. Long enough to be sustained, short enough to catch one hard passage."""

FAST_SECONDS = 0.1
"""A gap this short or shorter counts as part of a fast stretch (>= 10 nps)."""

WIDE_JUMP = 3
"""Frets. At this distance the hand repositions rather than rolling fingers."""


def peak_density(times: list[float], window: float = PEAK_WINDOW) -> float:
    """Highest note rate sustained over any `window` seconds."""
    if len(times) < 2:
        return 0.0
    ordered = sorted(times)
    best, start = 0.0, 0
    for index, moment in enumerate(ordered):
        while ordered[start] < moment - window:
            start += 1
        best = max(best, (index - start + 1) / window)
    return best


def longest_fast_stretch(times: list[float], gap: float = FAST_SECONDS) -> float:
    """Longest unbroken run of notes spaced `gap` apart or closer, in seconds."""
    ordered = sorted(set(times))
    best, start = 0.0, None
    for index in range(len(ordered) - 1):
        if ordered[index + 1] - ordered[index] <= gap:
            if start is None:
                start = ordered[index]
        elif start is not None:
            best = max(best, ordered[index] - start)
            start = None
    if start is not None:
        best = max(best, ordered[-1] - start)
    return best


def hand_movement(runs: list[list[int]]) -> tuple[float, float]:
    """(mean fret jump, share of jumps >= WIDE_JUMP) over single-note runs."""
    jumps = [abs(b - a) for run in runs for a, b in zip(run, run[1:])]
    if not jumps:
        return 0.0, 0.0
    wide = sum(1 for jump in jumps if jump >= WIDE_JUMP) / len(jumps)
    return statistics.mean(jumps), wide


def features(profile, times: list[float], runs: list[list[int]],
             marker_share: float = 0.0) -> dict[str, float]:
    """Everything the tier model reads, from one chart."""
    mean_jump, wide_jumps = hand_movement(runs)
    return {
        "nps": profile.nps,
        "peak10": peak_density(times),
        "longest_fast": longest_fast_stretch(times),
        "pct_chord": 1.0 - profile.chord_hist.get(1, 0.0),
        "pct_tap": profile.pct_tap,
        "pct_sustain": profile.pct_sustain,
        "mean_jump": mean_jump,
        "wide_jumps": wide_jumps,
        "marker_share": marker_share,
        "nps_variation": profile.nps_variation,
    }


FEATURE_NAMES = ("nps", "peak10", "longest_fast", "pct_chord", "pct_tap",
                 "pct_sustain", "mean_jump", "wide_jumps", "marker_share",
                 "nps_variation")
