"""Audio loading and waveform augmentation primitives."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T


MAX_AUDIO_BYTES = 10 * 60 * 24000 * 2


class MusicAugmenter:
    def __init__(self, sample_rate: int, enabled: bool = True) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.enabled = enabled
        self.pitch_shift = T.PitchShift(sample_rate=sample_rate, n_steps=0)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return waveform
        augmentations = [
            self._apply_gain,
            self._apply_noise,
            self._apply_pitch_shift,
            self._apply_polarity_inversion,
        ]
        random.shuffle(augmentations)
        for augmentation in augmentations[:random.randint(0, 2)]:
            waveform = augmentation(waveform)
        peak = waveform.abs().max()
        return waveform / peak if peak > 1.0 else waveform

    @staticmethod
    def _apply_gain(waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            waveform = waveform * (10 ** (random.uniform(-10, 10) / 20))
        return waveform

    @staticmethod
    def _apply_noise(waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.3:
            signal_power = waveform.square().mean()
            noise = torch.randn_like(waveform)
            noise_power = noise.square().mean().clamp_min(1e-12)
            scale = (signal_power / noise_power * 10 ** (-random.uniform(30, 50) / 10)).sqrt()
            waveform = waveform + noise * scale
        return waveform

    def _apply_pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.4:
            self.pitch_shift.n_steps = random.uniform(-2, 2)
            waveform = self.pitch_shift(waveform)
        return waveform

    @staticmethod
    def _apply_polarity_inversion(waveform: torch.Tensor) -> torch.Tensor:
        return -waveform if random.random() < 0.1 else waveform


def load_opus_ffmpeg(
    path: str,
    target_sr: int = 16000,
    timeout_seconds: int = 10,
) -> tuple[torch.Tensor, int]:
    command = [
        "ffmpeg", "-i", path, "-f", "s16le", "-ar", str(target_sr),
        "-ac", "1", "-",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )
    except OSError as error:
        raise RuntimeError(f"Failed to launch FFmpeg for {path}") from error
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise RuntimeError(f"FFmpeg timed out for {path}") from error
    if process.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed for {path}: {message}")
    audio = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0), target_sr


def load_raw_audio(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    source = Path(path)
    try:
        with source.open("rb") as stream:
            buffer = stream.read(MAX_AUDIO_BYTES)
    except OSError as error:
        raise RuntimeError(f"Failed to load raw audio {path}") from error
    if len(buffer) % 2:
        raise ValueError(f"Raw PCM file has an odd byte count: {path}")
    audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0), target_sr
