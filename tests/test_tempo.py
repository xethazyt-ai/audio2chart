"""The generated chart has to line up with the audio, and the beat grid has to mean
something. Both come down to the tempo map, so these check the map arithmetic directly:
a beat line lands on the audio's first note, and the lead-in takes exactly as long as
the audio takes to get there.
"""

import pytest

from chart.tempo import beat_aligned_tempo_events
from chart.time_conversion import preprocess_bpm_segments, seconds_to_tick

RESOLUTION = 192


def tick_of(seconds, bpm_events, resolution=RESOLUTION):
    segments = preprocess_bpm_segments(bpm_events, resolution)
    return seconds_to_tick(seconds, segments, resolution, [s[2] for s in segments])


@pytest.mark.parametrize("bpm,phase,onset", [
    (120.0, 0.10, 2.05),
    (174.0, 0.33, 1.02),
    (91.5, 0.48, 7.90),
    (200.0, 0.01, 0.55),
])
def test_first_note_lands_on_the_beat_line(bpm, phase, onset):
    events, anchor_tick, anchor = beat_aligned_tempo_events(bpm, phase, onset, RESOLUTION)
    assert anchor_tick % RESOLUTION == 0, "anchor must be a whole number of beats"
    assert tick_of(anchor, events) == pytest.approx(anchor_tick, abs=1)


@pytest.mark.parametrize("bpm,phase,onset", [(120.0, 0.10, 2.05), (174.0, 0.33, 1.02)])
def test_lead_in_lasts_exactly_as_long_as_the_intro(bpm, phase, onset):
    """Robert's third step: the tick-0 event is whatever makes the lead-in end on time.

    It cannot be exact -- .chart stores BPM as an integer number of milli-BPM, so the
    lead-in is only ever as accurate as that quantisation allows. What matters is that
    the residual is far below the 20 ms token grid, and it is: microseconds.
    """
    events, anchor_tick, anchor = beat_aligned_tempo_events(bpm, phase, onset, RESOLUTION)
    lead_bpm = events[0][1] / 1000.0
    beats = anchor_tick / RESOLUTION
    assert abs(beats * 60.0 / lead_bpm - anchor) < 1e-3


def test_the_grid_stays_locked_to_the_tracked_beat():
    """The anchor is the nearest tracked beat, not the raw onset -- a note played a
    little ahead of the beat must not drag the whole grid with it."""
    bpm, phase, period = 120.0, 0.10, 0.5
    _, _, anchor = beat_aligned_tempo_events(bpm, phase, 2.04, RESOLUTION)
    assert (anchor - phase) % period == pytest.approx(0.0, abs=1e-9)
    assert anchor == pytest.approx(2.10)


def test_song_tempo_holds_after_the_anchor():
    events, anchor_tick, anchor = beat_aligned_tempo_events(174.0, 0.33, 1.02, RESOLUTION)
    assert events[-1][1] == 174000
    one_beat_later = tick_of(anchor + 60.0 / 174.0, events)
    assert one_beat_later == pytest.approx(anchor_tick + RESOLUTION, abs=1)


def test_there_is_always_room_for_a_lead_in():
    """A song whose first note is at t=0 still needs a tick-0 event before it."""
    _, anchor_tick, anchor = beat_aligned_tempo_events(120.0, 0.0, 0.0, RESOLUTION)
    assert anchor_tick >= RESOLUTION
    assert anchor > 0.0
