"""MCP stdio server for mcp_daw.  Run: python mcp_server.py

Project root = $MCP_DAW_ROOT or this file's folder. All paths in tool I/O are
relative to that root (POSIX style, so they work the same on Windows).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("MCP_DAW_ROOT") or Path(__file__).resolve().parent).resolve()
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_daw import comfy, dynamics, library, master as mastering, render as renderer, song as songlib  # noqa: E402

comfy.load_env(ROOT)

SAMPLES, SONGS, OUT, DB = ROOT / "samples", ROOT / "songs", ROOT / "out", ROOT / "library.db"
mcp = FastMCP("mcp_daw")


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def _db() -> Path:
    if not DB.exists():
        library.scan(SAMPLES, DB)
    return DB


def _song_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError("song name: letters, digits, _ and - only")
    return SONGS / f"{name}.json"


def _load(name: str) -> dict:
    p = _song_path(name)
    if not p.exists():
        raise ValueError(f"no song '{name}'; use new_song or list_songs")
    return songlib.load(p)


def _save(name: str, s: dict) -> None:
    SONGS.mkdir(exist_ok=True)
    songlib.save(s, _song_path(name))


def _check_sample(sample: str) -> str:
    sample = sample.replace("\\", "/").removeprefix("samples/")
    if not (SAMPLES / sample).is_file():
        raise ValueError(f"sample not found: {sample} (use a path from search_samples)")
    return sample


@mcp.tool()
def search_samples(category: str | None = None, kind: str | None = None, bpm: float | None = None,
                   key: str | None = None, note: str | None = None, text: str | None = None,
                   release_safe: bool = True, limit: int = 30) -> list[dict]:
    """Find samples. kind: 'one_shot' or 'loop'. category: see list_categories. text matches the path.
    Returned 'path' values go straight into add_pattern_track / add_loop_track."""
    return library.search(_db(), category=category, kind=kind, bpm=bpm, key=key, note=note,
                          text=text, release_safe=release_safe, limit=limit)


@mcp.tool()
def list_categories() -> list[dict]:
    """Sample counts per category and kind."""
    return [{"category": c, "kind": k, "count": n} for c, k, n in library.counts(_db())]


@mcp.tool()
def new_song(name: str, bpm: float = 120, key: str = "C minor", bars: int = 4, swing: float = 0.0) -> str:
    """Create (or overwrite) songs/<name>.json. swing 0..1 delays odd 16ths."""
    _save(name, songlib.new_song(bpm=bpm, bars=bars, key=key, swing=swing))
    return f"created songs/{name}.json"


@mcp.tool()
def add_pattern_track(song: str, track: str, sample: str, steps: list[float],
                      pitches: list[int] | None = None, gain_db: float = 0.0, pan: float = 0.0,
                      bars: list[int] | None = None) -> str:
    """Add/replace a one-shot track. steps: 16 velocities per bar (0=off, 0..1).
    pitches: optional 16 semitone shifts. bars: 0-based bars to play (default all)."""
    s = _load(song)
    s["tracks"] = [t for t in s["tracks"] if t["name"] != track]
    songlib.add_pattern_track(s, track, _check_sample(sample), steps, pitches, gain_db, pan, bars)
    _save(song, s)
    return songlib.describe(s)


@mcp.tool()
def add_loop_track(song: str, track: str, sample: str, loop_bars: int, start_bars: list[int],
                   gain_db: float = 0.0, pan: float = 0.0) -> str:
    """Add/replace a loop track, time-stretched to loop_bars at song BPM, placed at each 0-based start bar."""
    s = _load(song)
    s["tracks"] = [t for t in s["tracks"] if t["name"] != track]
    songlib.add_loop_track(s, track, _check_sample(sample), loop_bars, start_bars, gain_db, pan)
    _save(song, s)
    return songlib.describe(s)


@mcp.tool()
def set_mix(song: str, track: str, gain_db: float | None = None, pan: float | None = None) -> str:
    """Set a track's gain (dB) and/or pan (-1 left .. 1 right)."""
    s = _load(song)
    t = next((t for t in s["tracks"] if t["name"] == track), None)
    if t is None:
        raise ValueError(f"no track '{track}' in {song}")
    if gain_db is not None:
        t["gain_db"] = gain_db
    if pan is not None:
        t["pan"] = max(-1.0, min(1.0, pan))
    _save(song, s)
    return songlib.describe(s)


@mcp.tool()
def describe_song(song: str) -> str:
    """Text view of the song: tempo, key, and each track's step grid or loop placement."""
    return songlib.describe(_load(song))


@mcp.tool()
def render(song: str, mp3: bool = True) -> dict:
    """Render songs/<song>.json to out/<song>.wav (+ .mp3 if ffmpeg exists)."""
    s = _load(song)
    wav = OUT / f"{song}.wav"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mix, warns = renderer.render(s, SAMPLES, wav, db=_db())
    result = {"wav": rel(wav), "duration_s": round(len(mix) / renderer.SR, 3), "licence_warnings": warns}
    if mp3 and shutil.which("ffmpeg"):
        out = wav.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(out)], check=True, stdin=subprocess.DEVNULL, capture_output=True)
        result["mp3"] = rel(out)
    return result


def _audio_path(path: str, must_exist: bool = True) -> Path:
    p = (ROOT / path.replace("\\", "/")).resolve()
    if ROOT not in p.parents:
        raise ValueError("audio paths must be relative to the project root")
    if must_exist and not p.is_file():
        raise ValueError(f"file not found: {path}")
    return p


def _write_audio(y, sr, out: Path, mp3: bool) -> dict:
    import soundfile as sf
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr, subtype="PCM_24")
    res = {"wav": rel(out)}
    if mp3 and shutil.which("ffmpeg"):
        m = out.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-b:a", "256k", str(m)],
                       check=True, stdin=subprocess.DEVNULL, capture_output=True)
        res["mp3"] = rel(m)
    return res


@mcp.tool()
def master_audio(path: str, preset: str = "edm", mp3: bool = True) -> dict:
    """Master a WAV/MP3 under the project root: LUFS-normalise then 4x-oversampled true-peak limit.
    presets: streaming, balanced, warm, pop, punch, rock, hiphop, edm (-10.5 LUFS / -1 dBTP).
    Writes <name>_master.wav (+ .mp3)."""
    import librosa
    src = _audio_path(path)
    y, sr = librosa.load(str(src), sr=None, mono=False)
    y = y.T if y.ndim == 2 else y
    out, rep = mastering.master(y, sr, preset)
    return {**rep, **_write_audio(out, sr, src.with_name(src.stem + "_master.wav"), mp3)}


@mcp.tool()
def apply_arrangement_dynamics(song: str, path: str, preset: str = "subtle", master_preset: str | None = "edm",
                               mp3: bool = True) -> dict:
    """Use songs/<song>.json's bpm + sections to add a beat-grid sidechain pump, section contrast and a
    pre-drop low-cut sweep to an audio file (e.g. a render or an AI re-take), then optionally master it.
    preset: 'subtle' or 'strong'. Section roles come from names: drop/chorus, build, intro/break/outro."""
    import librosa
    s = _load(song)
    src = _audio_path(path)
    y, sr = librosa.load(str(src), sr=None, mono=False)
    y = y.T if y.ndim == 2 else y
    out, rep = dynamics.apply_dynamics(y, sr, float(s["bpm"]), s.get("sections", []), preset)
    if master_preset:
        out, mrep = mastering.master(out, sr, master_preset)
        rep["master"] = mrep
    return {**rep, **_write_audio(out, sr, src.with_name(f"{src.stem}_{preset}.wav"), mp3)}


@mcp.tool()
def comfy_m2m(path: str, tags: str, lyrics: str = "[instrumental]", denoise: float = 0.5,
              lyrics_strength: float = 1.0, seed: int = 0) -> dict:
    """ACE-Step v1 music-to-music polish of an audio file via ComfyUI (COMFYUI_URL).
    Low denoise (0.3-0.5) keeps the arrangement; 0.7 restyles it. Output: out/<name>_m2m.mp3."""
    src = _audio_path(path)
    c = comfy.ComfyClient()
    wf = comfy.m2m_v1_workflow(c.upload(src), tags, lyrics, lyrics_strength, denoise, seed)
    res = c.run(wf, OUT / f"{src.stem}_m2m.mp3")
    res["out"] = rel(Path(res["out"]))
    return res


@mcp.tool()
def comfy_vocal_pass(song: str, path: str, lyrics: str, tags: str, denoise: float = 0.62, seed: int = 0,
                     language: str = "en") -> dict:
    """ACE-Step 1.5 turbo vocal pass over an (instrumental) audio file via ComfyUI, conditioned on the
    song's bpm/key and the file's duration. Use [verse]/[chorus]/[instrumental] tags in lyrics.
    Do not name real artists in tags. Output: out/<name>_vocal.mp3."""
    import librosa
    s = _load(song)
    src = _audio_path(path)
    dur = round(float(librosa.get_duration(path=str(src))), 2)
    c = comfy.ComfyClient()
    wf = comfy.vocal_v15_workflow(c.upload(src), tags, lyrics, int(round(s["bpm"])), s.get("key", "C minor"),
                                  dur, denoise, seed, language)
    res = c.run(wf, OUT / f"{src.stem}_vocal.mp3")
    res["out"] = rel(Path(res["out"]))
    return res


@mcp.tool()
def check_lyrics(path: str, lyrics: str, start: float = 0.0, duration: float | None = None) -> dict:
    """Optional: transcribe sung audio with faster-whisper and score word overlap with the lyrics
    (precision / recall). Needs: pip install -r requirements-asr.txt"""
    from mcp_daw import asr
    return asr.intelligibility(str(_audio_path(path)), lyrics, start, duration)


@mcp.tool()
def list_songs() -> list[str]:
    """Names of saved songs."""
    return sorted(p.stem for p in SONGS.glob("*.json"))


def _warm():
    """Load librosa's lazy modules and JIT code before serving; on Windows doing this inside a tool call hangs the stdio server."""
    import numpy as np, librosa
    y = np.zeros(2048, dtype=np.float32)
    librosa.resample(y, orig_sr=48000, target_sr=44100)
    librosa.effects.time_stretch(y, rate=1.1)


if __name__ == "__main__":
    _warm()
    mcp.run()  # stdio
