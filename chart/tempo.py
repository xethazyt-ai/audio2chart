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


def constant_tempo_events(bpm, phase, resolution, lead_in_bpm=120.0):
    """Constant `bpm` whose grid lands on the beat at `phase` seconds.

    Tick 0 has to map to t=0, so the lead-in is absorbed by a first tempo segment
    spanning [0, phase) -- the same trick the human charter used on Sapphire, and it
    keeps Offset at 0. Beat positions repeat every period, so `phase` is pushed
    forward whole periods until the lead-in is long enough to carry a sane BPM.
    """
    period = 60.0 / bpm
    while phase < 0.5:
        phase += period

    beats = max(1, int(round(phase * lead_in_bpm / 60.0)))
    intro_bpm = beats * 60.0 / phase

    return [
        (0, int(round(intro_bpm * 1000))),
        (beats * resolution, int(round(bpm * 1000))),
    ]
