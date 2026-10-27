"""Arrangement dynamics driven by the song's section map.

* beat-grid sidechain pump (low band ducked harder than the rest, 150 Hz crossover)
* section gain contrast (cuts only, smoothed across bar lines)
* pre-drop low-cut sweep

Section roles are inferred from names: 'drop*' / 'chorus*' hit, 'build*' rises,
'intro*' / 'break*' / 'outro*' are pulled back.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class DynamicsPreset:
    contrast_db: float      # quiet sections end up this far below the loudest drop (pre-master)
    duck_low_db: float
    duck_high_db: float
    release_ms: float
    sweep: float            # 0..1 blend toward the low-cut in the bars before a drop
    sweep_bars: int = 4
    crossover_hz: float = 150.0
    sweep_cut_hz: float = 400.0


DYNAMICS_PRESETS = {
    "subtle": DynamicsPreset(7.5, 3.0, 1.2, 150, 0.5),
    "strong": DynamicsPreset(10.0, 5.0, 2.0, 180, 0.7),
}


def role(name: str) -> str:
    n = name.lower()
    for key, r in (("drop", "drop"), ("chorus", "drop"), ("build", "build"), ("intro", "quiet"),
                   ("break", "quiet"), ("outro", "quiet")):
        if n.startswith(key):
            return r
    return "other"


def bar_seconds(bpm: float) -> float:
    return 240.0 / bpm


def beat_phase(y: np.ndarray, sr: int, bpm: float, hop: int = 256) -> float:
    """Offset (s) of the beat grid that best lines up with detected onsets."""
    mono = y.mean(axis=1) if y.ndim == 2 else y
    env = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=sr, hop_length=hop)
    beat = 60.0 / bpm
    best, best_s = 0.0, -1.0
    for ph in np.arange(0, beat, 0.002):
        idx = np.round(np.arange(ph, len(mono) / sr, beat) * sr / hop).astype(int)
        s = env[idx[idx < len(env)]].mean()
        if s > best_s:
            best, best_s = float(ph), s
    return best


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n < 1:
        return x
    w = np.hanning(2 * n + 1)
    w /= w.sum()
    return np.convolve(np.pad(x, n, mode="edge"), w, mode="valid")


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def section_rms(y: np.ndarray, sr: int, bpm: float, sections: list[dict], phase: float = 0.0) -> dict:
    bs, out = bar_seconds(bpm), {}
    for s in sections:
        a = int((phase + s["start_bar"] * bs) * sr)
        b = min(len(y), int((phase + (s["start_bar"] + s["bars"]) * bs) * sr))
        if b > a:
            out[s["name"]] = round(_rms_db(y[a:b]), 2)
    return out


def section_gain_curve(y, sr, bpm, sections, contrast_db, phase=0.0, ramp_ms=25.0) -> np.ndarray:
    """Per-sample gain (linear). Drops stay at 0 dB; quiet sections are cut to (loudest drop - contrast);
    builds ramp from the previous level up to -1 dB at the bar line."""
    rms = section_rms(y, sr, bpm, sections, phase)
    drops = [rms[s["name"]] for s in sections if role(s["name"]) == "drop" and s["name"] in rms]
    ref = max(drops) if drops else max(rms.values())
    bs, n = bar_seconds(bpm), len(y)
    g = np.zeros(n)
    prev = 0.0
    for s in sorted(sections, key=lambda s: s["start_bar"]):
        a = max(0, int((phase + s["start_bar"] * bs) * sr))
        b = min(n, int((phase + (s["start_bar"] + s["bars"]) * bs) * sr))
        r = role(s["name"])
        if r == "quiet" and s["name"] in rms:
            val = min(0.0, ref - contrast_db - rms[s["name"]])
            g[a:b] = val
        elif r == "build":
            val = -1.0
            g[a:b] = np.linspace(prev, val, max(0, b - a))
        else:
            val = 0.0
            g[a:b] = 0.0
        if s["start_bar"] == min(x["start_bar"] for x in sections):
            g[:a] = val
        prev = val
    return 10 ** (_smooth(g, int(sr * ramp_ms / 1000)) / 20)


def pump_envelopes(n, sr, bpm, sections, duck_low_db, duck_high_db, release_ms, phase=0.0):
    """Two gain envelopes (low band, rest) that duck on every beat of build (second half) and drop sections."""
    beat, bs = 60.0 / bpm, bar_seconds(bpm)
    L = int(beat * sr)
    t = np.arange(L) / sr
    rec = 1 - np.exp(-t / (release_ms / 1000 / 3))  # 0 at the hit, ~1 after release_ms
    env_l, env_h = np.ones(n), np.ones(n)
    dl, dh = 10 ** (-duck_low_db / 20), 10 ** (-duck_high_db / 20)
    for s in sections:
        r = role(s["name"])
        if r not in ("drop", "build"):
            continue
        start = s["start_bar"] + (s["bars"] // 2 if r == "build" else 0)
        for k in range(int((s["start_bar"] + s["bars"] - start) * 4)):
            i = int((phase + start * bs + k * beat) * sr)
            if i >= n:
                break
            m = min(L, n - i)
            env_l[i:i + m] = np.minimum(env_l[i:i + m], dl + (1 - dl) * rec[:m])
            env_h[i:i + m] = np.minimum(env_h[i:i + m], dh + (1 - dh) * rec[:m])
    k = int(0.002 * sr)  # ~2 ms attack smoothing, no clicks
    return _smooth(env_l, k), _smooth(env_h, k)


def apply_dynamics(y: np.ndarray, sr: int, bpm: float, sections: list[dict],
                   preset: str | DynamicsPreset = "subtle", phase: float | None = None) -> tuple[np.ndarray, dict]:
    p = DYNAMICS_PRESETS[preset] if isinstance(preset, str) else preset
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    if phase is None:
        phase = beat_phase(y, sr, bpm)
    n = len(y)
    low = sosfiltfilt(butter(4, p.crossover_hz, "low", fs=sr, output="sos"), y, axis=0)
    high = y - low  # zero-phase split, sums back exactly
    el, eh = pump_envelopes(n, sr, bpm, sections, p.duck_low_db, p.duck_high_db, p.release_ms, phase)
    out = low * el[:, None] + high * eh[:, None]
    w = np.zeros(n)
    bs = bar_seconds(bpm)
    for s in sections:
        if role(s["name"]) == "drop" and s["start_bar"] >= p.sweep_bars:
            a = int((phase + (s["start_bar"] - p.sweep_bars) * bs) * sr)
            b = min(n, int((phase + s["start_bar"] * bs) * sr))
            if b > a:
                w[a:b] = np.linspace(0, p.sweep, b - a) ** 1.5
    w = _smooth(w, int(0.01 * sr))[:, None]
    hp = sosfiltfilt(butter(2, p.sweep_cut_hz, "high", fs=sr, output="sos"), out, axis=0)
    out = out * (1 - w) + hp * w
    out = out * section_gain_curve(y, sr, bpm, sections, p.contrast_db, phase)[:, None]
    return out.astype(np.float32), {"phase_s": round(phase, 4), "preset": p.__dict__,
                                    "section_rms_db": section_rms(out, sr, bpm, sections, phase)}
