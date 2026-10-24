"""Render a song dict to a 44.1 kHz stereo WAV."""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import library
from .song import STEPS, validate

SR = 44100


def bar_seconds(bpm: float) -> float:
    return 4 * 60.0 / bpm


@lru_cache(maxsize=256)
def load_sample(path: str) -> np.ndarray:
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = y[:, :2]
    if sr != SR:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T
    return np.ascontiguousarray(y)


def pitch(y: np.ndarray, semitones: float) -> np.ndarray:
    """Repitch by resampling (sampler-style: higher = shorter)."""
    if not semitones:
        return y
    ratio = 2 ** (semitones / 12)
    idx = np.arange(0, len(y) - 1, ratio)
    return np.stack([np.interp(idx, np.arange(len(y)), y[:, c]) for c in range(2)], axis=1).astype(np.float32)


def stretch_to(y: np.ndarray, n: int) -> np.ndarray:
    rate = len(y) / n
    if abs(rate - 1) > 1e-3:
        y = np.stack([librosa.effects.time_stretch(y[:, c], rate=rate) for c in range(2)], axis=1)
    return librosa.util.fix_length(y, size=n, axis=0)


def pan_gain(gain_db: float, pan: float) -> np.ndarray:
    g = 10 ** (gain_db / 20)
    theta = (pan + 1) * np.pi / 4  # constant-power
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * g * np.sqrt(2)


def add_at(buf: np.ndarray, y: np.ndarray, start: int) -> None:
    if start >= len(buf):
        return
    end = min(len(buf), start + len(y))
    buf[start:end] += y[: end - start]


def step_onset(step: int, bar: int, bpm: float, swing: float) -> int:
    """Sample offset of a 16th step. Swing (0..1) delays odd 16ths by up to half a step."""
    step_s = bar_seconds(bpm) / STEPS
    t = bar * bar_seconds(bpm) + step * step_s
    if step % 2 == 1:
        t += swing * step_s * 0.5
    return int(round(t * SR))


def licence_warnings(song: dict, sample_root: Path, db: Path | None) -> list[str]:
    out = []
    for t in song["tracks"]:
        lic = None
        if db and db.exists():
            row = library.lookup(db, t["sample"])
            lic = row["licence"] if row else None
        if lic is None:
            src = (sample_root / t["sample"]).relative_to(sample_root).parts[0]
            lic = library.detect_licence(sample_root / src)
        if lic not in library.RELEASE_SAFE:
            out.append(f"track '{t['name']}': {t['sample']} licence '{lic}' is NOT release-safe")
        elif lic in library.NEEDS_CREDIT:
            out.append(f"track '{t['name']}': {t['sample']} is {lic}, credit the author")
    return out


def render(song: dict, sample_root: str | Path, out_path: str | Path | None = None,
           db: str | Path | None = None, headroom_db: float = -1.0) -> tuple[np.ndarray, list[str]]:
    """Mix the song. Returns (stereo float32 array, licence warnings); writes WAV if out_path."""
    validate(song)
    root = Path(sample_root)
    bpm, bars, swing = float(song["bpm"]), int(song["bars"]), float(song.get("swing", 0))
    total = int(round(bars * bar_seconds(bpm) * SR))
    mix = np.zeros((total, 2), dtype=np.float32)
    for t in song["tracks"]:
        y = load_sample(str(root / t["sample"]))
        g = pan_gain(t.get("gain_db", 0), t.get("pan", 0))
        if t["type"] == "pattern":
            pitches = t.get("pitches") or [0] * STEPS
            for bar in t.get("bars", range(bars)):
                for s, vel in enumerate(t["steps"]):
                    if vel > 0:
                        add_at(mix, pitch(y, pitches[s]) * (vel * g), step_onset(s, bar, bpm, swing))
        else:
            n = int(round(t["loop_bars"] * bar_seconds(bpm) * SR))
            seg = stretch_to(y, n) * g
            for b in t["start_bars"]:
                add_at(mix, seg, int(round(b * bar_seconds(bpm) * SR)))
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix *= (10 ** (headroom_db / 20)) / peak
    warns = licence_warnings(song, root, Path(db) if db else None)
    for w in warns:
        warnings.warn(w)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, mix, SR, subtype="PCM_16")
    return mix, warns
