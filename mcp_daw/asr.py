"""Optional lyric-intelligibility check with faster-whisper (pip install -r requirements-asr.txt)."""
from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", re.sub(r"\[.*?\]", " ", text.lower()))


def score(transcript: str, lyrics: str) -> dict:
    """precision: share of heard words that are in the lyrics; recall: share of lyric vocabulary heard."""
    heard, vocab = words(transcript), set(words(lyrics))
    hits = [w for w in heard if w in vocab]
    return {"words": len(heard), "precision": round(len(hits) / max(1, len(heard)), 3),
            "recall": round(len(set(hits)) / max(1, len(vocab)), 3)}


def _decode(path: str, start: float, duration: float | None) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to decode audio for ASR")
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(start)] + (["-t", str(duration)] if duration else []) + \
          ["-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)


def intelligibility(path: str, lyrics: str, start: float = 0.0, duration: float | None = None,
                    model: str = "small", device: str = "auto") -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # optional extra
        raise RuntimeError("faster-whisper not installed: pip install -r requirements-asr.txt") from e
    try:
        m = WhisperModel(model, device=device, compute_type="default")
    except Exception:
        m = WhisperModel(model, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(_decode(path, start, duration), language="en", beam_size=5,
                           condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segs)
    return {**score(text, lyrics), "transcript": text}
