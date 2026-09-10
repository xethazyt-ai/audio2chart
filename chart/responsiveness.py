"""Does a chart follow the music, or just fill time?

This is the measurable core of Robert's complaint that a generated chart "bears no
relation to what you're hearing". Note-level agreement is the obvious test and it
does not work: `chart.alignment` records that matching generated note times against
librosa onsets performs at chance, because a charter does not place a note on every
onset and the ones they skip are not random.

Density is coarser and survives that. A chart gets busier where the music does --
through a chorus, a solo, a drop -- and quieter in a breakdown, and that envelope is
robust to which individual onsets a charter chose. So bin both signals per second and
correlate the two envelopes.

The number only means something against its two bracketing controls:

*The human chart for the same song* is the ceiling. It says how much correlation is
even available given how loosely charts track audio energy.

*The same chart against a different song* is the floor. Any correlation there is an
artefact of two envelopes both being smooth and mostly non-zero.

A generated chart scoring at the floor is unconditioned no matter what its loss says.
"""

from __future__ import annotations

import numpy as np

BIN_SECONDS = 1.0
"""Coarse enough to survive onset-level disagreement, fine enough to see a chorus."""


def onset_envelope(waveform: np.ndarray, sample_rate: int, bin_seconds: float = BIN_SECONDS,
                   hop_length: int = 256) -> np.ndarray:
    """Audio energy per bin, as onset strength rather than raw loudness.

    Loudness would track mastering and instrumentation; onset strength tracks how much
    is *happening*, which is what a charter responds to.
    """
    import librosa

    strength = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(strength, sr=sample_rate, hop_length=hop_length)
    return _bin_by_time(times, strength, bin_seconds)


def note_density(note_times, duration: float, bin_seconds: float = BIN_SECONDS) -> np.ndarray:
    """Notes per bin."""
    if duration <= 0:
        return np.zeros(0)
    bins = int(np.ceil(duration / bin_seconds))
    counts = np.zeros(bins)
    for time in note_times:
        index = int(time / bin_seconds)
        if 0 <= index < bins:
            counts[index] += 1
    return counts


def _bin_by_time(times: np.ndarray, values: np.ndarray, bin_seconds: float) -> np.ndarray:
    if times.size == 0:
        return np.zeros(0)
    bins = int(np.ceil(times[-1] / bin_seconds))
    if bins <= 0:
        return np.zeros(0)
    index = np.clip((times / bin_seconds).astype(int), 0, bins - 1)
    totals = np.bincount(index, weights=values, minlength=bins)
    counts = np.bincount(index, minlength=bins)
    return totals / np.maximum(counts, 1)


def responsiveness(density: np.ndarray, envelope: np.ndarray) -> float:
    """Pearson correlation of the two envelopes over their shared length.

    Returns 0.0 when either side is flat -- a chart of constant density has no
    envelope to correlate, and that is the mechanical failure mode, not an error.
    """
    length = min(len(density), len(envelope))
    if length < 3:
        return float("nan")
    a, b = density[:length], envelope[:length]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
