"""Mastering: LUFS normalisation (pyloudnorm) + oversampled true-peak limiting.

The limiter is ffmpeg's ``alimiter`` run at 4x the sample rate when ffmpeg is on PATH,
otherwise a small numpy look-ahead limiter. Either way the result is re-measured and the
ceiling is tightened until the 4x-oversampled true peak is at or below the target.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

OVERSAMPLE = 4


@dataclass(frozen=True)
class Preset:
    target_lufs: float
    ceiling_dbtp: float


# Loudness targets in the spirit of common streaming / genre mastering presets.
PRESETS: dict[str, Preset] = {
    "streaming": Preset(-14.0, -1.0),
    "balanced": Preset(-14.0, -1.0),
    "warm": Preset(-13.0, -1.0),
    "pop": Preset(-12.5, -1.0),
    "punch": Preset(-12.0, -1.0),
    "rock": Preset(-11.5, -1.0),
    "hiphop": Preset(-11.0, -1.0),
    "edm": Preset(-10.5, -1.0),
}


def _stereo(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return y[:, None] if y.ndim == 1 else y


def lufs(y: np.ndarray, sr: int) -> float:
    return float(pyln.Meter(sr).integrated_loudness(_stereo(y)))


def true_peak_db(y: np.ndarray, oversample: int = OVERSAMPLE) -> float:
    up = resample_poly(_stereo(y), oversample, 1, axis=0)
    return float(20 * np.log10(np.abs(up).max() + 1e-12))


def _limit_numpy(y: np.ndarray, sr: int, ceiling_db: float, release_ms: float = 60.0) -> np.ndarray:
    """Look-ahead peak limiter on the 4x-oversampled envelope (no ringing past the ceiling)."""
    ceil = 10 ** (ceiling_db / 20)
    up = np.abs(resample_poly(y, OVERSAMPLE, 1, axis=0)).max(axis=1)
    up = np.pad(up, (0, len(y) * OVERSAMPLE - len(up)))[: len(y) * OVERSAMPLE]
    peak = up.reshape(len(y), OVERSAMPLE).max(axis=1)
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-12))
    look = max(1, int(0.003 * sr))
    # hold the minimum over the look-ahead window, then release smoothly
    from scipy.ndimage import minimum_filter1d
    g = minimum_filter1d(need, size=2 * look + 1, mode="nearest")
    a = np.exp(-1.0 / (release_ms / 1000 * sr))
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):  # instant attack, exponential release
        cur = v if v < cur else a * cur + (1 - a) * v
        out[i] = cur
    return y * out[:, None]


def _limit_ffmpeg(y: np.ndarray, sr: int, ceiling_db: float) -> np.ndarray:
    limit = 10 ** (ceiling_db / 20)
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        sf.write(src, y, sr, subtype="FLOAT")
        af = (f"aresample={sr * OVERSAMPLE},alimiter=limit={limit:.5f}:attack=3:release=50:level=false,"
              f"aresample={sr}")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, "-c:a", "pcm_f32le", dst],
                       check=True, stdin=subprocess.DEVNULL, capture_output=True)
        out, _ = sf.read(dst, dtype="float64", always_2d=True)
    n = len(y)
    return out[:n] if len(out) >= n else np.pad(out, ((0, n - len(out)), (0, 0)))


def master(y: np.ndarray, sr: int, preset: str | Preset = "edm", engine: str = "auto",
           max_iter: int = 6) -> tuple[np.ndarray, dict]:
    """Normalise to the preset's LUFS, then limit so the 4x true peak <= ceiling.

    Returns (audio, report). ``engine``: 'ffmpeg', 'numpy' or 'auto'.
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset
    y = _stereo(y)
    if engine == "auto":
        engine = "ffmpeg" if shutil.which("ffmpeg") else "numpy"
    limiter = _limit_ffmpeg if engine == "ffmpeg" else _limit_numpy
    gain_db = p.target_lufs - lufs(y, sr)
    ceiling = p.ceiling_dbtp - 0.2  # start just under target; tightened below if needed
    out = y
    for _ in range(max_iter):
        out = limiter(y * 10 ** (gain_db / 20), sr, ceiling)
        tp, loud = true_peak_db(out), lufs(out, sr)
        if tp > p.ceiling_dbtp:
            ceiling -= (tp - p.ceiling_dbtp) + 0.1
            continue
        if loud < p.target_lufs - 0.3:  # limiter ate loudness: push the input a little harder
            gain_db += min(1.5, p.target_lufs - loud)
            continue
        break
    tp, loud = true_peak_db(out), lufs(out, sr)
    if tp > p.ceiling_dbtp:  # last resort: static trim guarantees the ceiling
        out = out * 10 ** ((p.ceiling_dbtp - 0.05 - tp) / 20)
        tp, loud = true_peak_db(out), lufs(out, sr)
    return out.astype(np.float32), {"engine": engine, "lufs": round(loud, 2), "true_peak_dbtp": round(tp, 2),
                                     "target_lufs": p.target_lufs, "ceiling_dbtp": p.ceiling_dbtp}


def master_file(src: str, dst: str, preset: str = "edm", engine: str = "auto") -> dict:
    y, sr = sf.read(src, dtype="float64", always_2d=True)
    out, rep = master(y, sr, preset, engine)
    sf.write(dst, out, sr, subtype="PCM_24")
    return rep
