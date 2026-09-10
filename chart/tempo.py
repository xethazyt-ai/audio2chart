"""Tempo-map helpers: reuse the SyncTrack of an existing chart, or estimate one from audio.

The chart writer only ever emitted a single fixed BPM, so generated charts stayed
aligned to the audio but their beat grid meant nothing. Both entry points here
produce `bpm_events` in the same (tick, bpm * 1000) form that
`preprocess_bpm_segments` already consumes.
"""

import re

DEFAULT_BPM_RANGE = (100.0, 200.0)


def parse_sync_track(chart_path):
    """Lift Resolution and the [SyncTrack] events out of an existing .chart file.

    Returns (bpm_events, ts_events, resolution).
    """
    with open(chart_path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    match = re.search(r"^\s*Resolution\s*=\s*(\d+)", text, re.M)
    if not match:
        raise ValueError(f"no Resolution in {chart_path}")
    resolution = int(match.group(1))

    block = re.search(r"\[SyncTrack\]\s*\{(.*?)\}", text, re.S)
    if not block:
        raise ValueError(f"no [SyncTrack] block in {chart_path}")

    bpm_events, ts_events = [], []
    for tick, kind, rest in re.findall(
        r"^\s*(\d+)\s*=\s*(B|TS)\s+([\d ]+)", block.group(1), re.M
    ):
        values = [int(v) for v in rest.split()]
        if kind == "B":
            bpm_events.append((int(tick), values[0]))
        else:
            ts_events.append((int(tick), tuple(values)))

    if not bpm_events:
        raise ValueError(f"[SyncTrack] in {chart_path} has no B events")

    return sorted(bpm_events), sorted(ts_events), resolution


def detect_tempo(y, sr, bpm_range=DEFAULT_BPM_RANGE, hop_length=256):
    """Estimate (bpm, phase, score) by a phase-locked search over the onset envelope.

    librosa.beat.beat_track reports the tempo of the strongest metrical level, which
    is routinely a half- or double-time octave error and is too imprecise to hold
    sync over a full song. Scoring a dense (bpm, phase) grid directly against onset
    strength is slower but lands within a few thousandths of a BPM.
    """
    import numpy as np
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(env, sr=sr, hop_length=hop_length)
    spread = env.std()
    env = (env - env.mean()) / (spread if spread else 1.0)
    end = times[-1]

    def score(bpm, phase):
        beats = np.arange(phase, end, 60.0 / bpm)
        if beats.size == 0:
            return -np.inf
        idx = np.clip(np.searchsorted(times, beats), 0, len(env) - 1)
        return float(env[idx].mean())

    def best_phase(bpm, step):
        phases = np.arange(0.0, 60.0 / bpm, step)
        scores = [score(bpm, p) for p in phases]
        i = int(np.argmax(scores))
        return float(phases[i]), scores[i]

    low, high = bpm_range
    best = (low, 0.0, -float("inf"))
    for bpm in np.arange(low, high, 0.25):
        phase, value = best_phase(bpm, 0.01)
        if value > best[2]:
            best = (float(bpm), phase, value)

    for bpm in np.arange(best[0] - 0.5, best[0] + 0.5, 0.002):
        phase, value = best_phase(bpm, 0.005)
        if value > best[2]:
            best = (float(bpm), phase, value)

    return best


FIRST_ONSET_FLOOR = 0.25
"""An onset weaker than this fraction of the strongest one is not the song starting."""


def first_onset(y, sr, hop_length=256, floor=FIRST_ONSET_FLOOR):
    """Time in seconds of the first real note in the audio.

    Anchoring a chart on the earliest detected onset is fragile: a fade-in, a room
    tone or a click before the count-in all register, and anchoring on one drags the
    entire chart out of sync. Requiring a fraction of the strongest onset in the
    track skips those. Backtracking then moves the answer from the envelope peak to
    the attack that produced it, which is where a charter would put the note.
    """
    import librosa
    import numpy as np

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    peaks = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop_length, backtrack=False
    )
    if peaks.size == 0:
        return 0.0
    strong = peaks[env[peaks] >= floor * env[peaks].max()]
    if strong.size == 0:
        strong = peaks
    start = librosa.onset.onset_backtrack(strong[:1], env)
    return float(librosa.frames_to_time(start, sr=sr, hop_length=hop_length)[0])


def beat_aligned_tempo_events(bpm, phase, onset, resolution):
    """Tempo map putting a beat line exactly on the audio's first note.

    Robert's rule for syncing a generated chart, in his order: the first note lands
    on a beat line; that line carries a BPM event at the song's real tempo; and the
    tick-0 event -- the one Clone Hero starts every chart with -- is then whatever
    value makes the lead-in last exactly as long as the audio takes to reach that
    note.

    The anchor is the *tracked beat* nearest the first onset rather than the onset
    itself, so the grid stays locked to the music even when the first note is played
    slightly ahead of or behind the beat. Beat tracking never identifies downbeats,
    so a beat line is the strongest claim the analysis actually supports -- which is
    also what Robert specified.

    Returns (bpm_events, anchor_tick, anchor_seconds).
    """
    period = 60.0 / bpm
    anchor = phase + round((onset - phase) / period) * period
    while anchor < period:          # a chart needs at least one beat of lead-in
        anchor += period

    beats = max(1, int(round(anchor / period)))
    lead_bpm = beats * 60.0 / anchor

    events = [
        (0, int(round(lead_bpm * 1000))),
        (beats * resolution, int(round(bpm * 1000))),
    ]
    return events, beats * resolution, anchor
