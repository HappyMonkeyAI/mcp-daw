"""Song model: plain JSON the AI edits. Helpers here map 1:1 to future MCP tools."""
from __future__ import annotations

import json
from pathlib import Path

STEPS = 16  # per bar, 16th notes


def new_song(bpm: float = 120, bars: int = 4, key: str = "C minor", swing: float = 0.0) -> dict:
    return {"bpm": bpm, "key": key, "bars": bars, "swing": swing,
            "sections": [{"name": "main", "start_bar": 0, "bars": bars}], "tracks": []}


def add_pattern_track(song: dict, name: str, sample: str, steps: list[float],
                      pitches: list[int] | None = None, gain_db: float = 0.0, pan: float = 0.0,
                      bars: list[int] | None = None) -> dict:
    """steps: 16 velocities (0 = off, 0..1). pitches: semitone shift per step. bars: which bars play (default all)."""
    if len(steps) != STEPS:
        raise ValueError(f"steps must have {STEPS} entries")
    t = {"name": name, "type": "pattern", "sample": sample, "steps": steps,
         "gain_db": gain_db, "pan": pan}
    if pitches:
        t["pitches"] = pitches
    if bars is not None:
        t["bars"] = bars
    song["tracks"].append(t)
    return t


def add_loop_track(song: dict, name: str, sample: str, loop_bars: int, start_bars: list[int],
                   gain_db: float = 0.0, pan: float = 0.0) -> dict:
    t = {"name": name, "type": "loop", "sample": sample, "loop_bars": loop_bars,
         "start_bars": start_bars, "gain_db": gain_db, "pan": pan}
    song["tracks"].append(t)
    return t


def validate(song: dict) -> None:
    for k in ("bpm", "bars", "tracks"):
        if k not in song:
            raise ValueError(f"song missing '{k}'")
    for t in song["tracks"]:
        if t["type"] == "pattern" and len(t["steps"]) != STEPS:
            raise ValueError(f"track {t['name']}: need {STEPS} steps")
        if t["type"] not in ("pattern", "loop"):
            raise ValueError(f"track {t['name']}: unknown type {t['type']}")


def describe(song: dict) -> str:
    lines = [f"{song['bpm']} BPM, {song.get('key', '?')}, {song['bars']} bars, swing {song.get('swing', 0)}"]
    for t in song["tracks"]:
        if t["type"] == "pattern":
            grid = "".join("x" if v >= 0.7 else "o" if v > 0 else "." for v in t["steps"])
            lines.append(f"{t['name']:<8} |{grid}|  {t['sample']}")
        else:
            lines.append(f"{t['name']:<8} loop {t['loop_bars']} bar(s) at {t['start_bars']}  {t['sample']}")
    return "\n".join(lines)


def load(path: str | Path) -> dict:
    s = json.loads(Path(path).read_text())
    validate(s)
    return s


def save(song: dict, path: str | Path) -> None:
    validate(song)
    Path(path).write_text(json.dumps(song, indent=2))
