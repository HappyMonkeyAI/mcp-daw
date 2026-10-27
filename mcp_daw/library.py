"""Sample library: scan folders into SQLite and search them."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXT = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
RELEASE_SAFE = {"CC0-1.0", "Unlicense", "CC-BY-4.0", "CC-BY-3.0"}
NEEDS_CREDIT = {"CC-BY-4.0", "CC-BY-3.0"}
SUSTAINED = {"crash", "ride", "hat", "open_hat", "bass", "tom"}  # ring out; never auto-flag as loops

# token -> category; first match wins, so specific names come first
CATEGORY_RULES = [
    # melodic / fx words first so e.g. "lead_loop" is a lead, not a drum break
    (r"riser|uplift|downlift|impact|sweep|whoosh|reverse crash", "riser"),
    (r"\bfx\b|sfx|\bnoise\b", "fx"),
    (r"vocal|\bvox\b|\bvoc\b|acapella|chant|\bshout\b", "vocal"),
    (r"\barp\b|arpeg", "arp"),
    (r"pluck", "pluck"),
    (r"chord|\bstab\b|stabs", "chord"),
    (r"\bpad\b|pads|atmos|ambient|drone|strings?\b", "pad"),
    (r"\blead\b|leads|melody|\bhook\b|supersaw", "lead"),
    (r"brk|break|amen|loop", "break"),
    (r"kick|\bbd\b|bassdrum", "kick"),
    (r"snare|\bsd\b|\bsn\b", "snare"),
    (r"\bcp\b|clap", "clap"),
    (r"\brim\b", "rim"),
    (r"\boh\b|hat open|open hat|hat sizzling", "open_hat"),
    (r"\bhh\b|hat|hihat", "hat"),
    (r"crash|china|splash|\bcr\b", "crash"),
    (r"ride|\brd\b", "ride"),
    (r"tom|\bht\b|\bmt\b|\blt\b", "tom"),
    (r"bass|meatbass", "bass"),
    (r"cowbell|\bcb\b|bongo|djembe|darbouka|cajon|perc|shaker|\bsh\b|tamb", "perc"),
]
NOTE_RE = re.compile(r"(?:^|_)([a-g](?:b|#)?)(\d)(?:_|$)")
SCHEMA = """CREATE TABLE IF NOT EXISTS samples(
  path TEXT PRIMARY KEY, name TEXT, source TEXT, licence TEXT,
  duration REAL, kind TEXT, category TEXT, bpm REAL, key TEXT, note TEXT, peak REAL, stem INTEGER DEFAULT 0)"""
STEM_SECONDS = 60  # longer files are stems / full bounces, not loops


def detect_licence(root: Path) -> str:
    for f in root.iterdir():
        if f.is_file() and f.stem.lower() in {"license", "licence", "copying"}:
            t = f.read_text(errors="ignore")
            if "CC0" in t:
                return "CC0-1.0"
            if "unencumbered" in t:
                return "Unlicense"
            if "Attribution 4.0" in t:
                return "CC-BY-4.0"
            if "Attribution 3.0" in t:
                return "CC-BY-3.0"
            if "GNU AFFERO" in t:
                return "AGPL-3.0"
            if "GNU GENERAL" in t:
                return "GPL"
    return "unknown"


def tokens(path: Path) -> str:
    return " ".join(re.split(r"[/_\-. ]+", str(path).lower()))


def categorise(rel: Path, y: np.ndarray, sr: int) -> str:
    text = " " + tokens(rel) + " "
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, text):
            return cat
    # fallback: noise-like -> fx, rising envelope -> riser, smooth sustained -> pad, else by brightness
    dur = len(y) / sr
    if dur >= 1.5:
        rms = librosa.feature.rms(y=y)[0]
        q = max(1, len(rms) // 4)
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        if rms[-q:].mean() > 3 * rms[:q].mean() + 1e-6 and flat > 0.05:
            return "riser"
        if flat > 0.3:
            return "fx"
        if dur >= 3 and n_onsets(y, sr) / dur < 0.5:
            return "pad"
    c = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return "low" if c < 800 else "high" if c > 4000 else "mid"


def n_onsets(y: np.ndarray, sr: int) -> int:
    return len(librosa.onset.onset_detect(y=y, sr=sr, backtrack=False))


def estimate_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = max(((np.corrcoef(chroma, np.roll(p, i))[0, 1], f"{names[i]} {m}")
                for p, m in ((major, "major"), (minor, "minor")) for i in range(12)))
    return best[1]


def analyse(path: Path, rel: Path) -> dict:
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = y.mean(axis=1)
    dur = len(mono) / sr
    cat = categorise(rel, mono, sr)
    kind = "loop" if cat == "break" or "loop" in tokens(rel) or (dur >= 3 and cat not in SUSTAINED and n_onsets(mono, sr) >= 8) else "one_shot"
    bpm = key = None
    if kind == "loop":
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        bpm = round(float(np.atleast_1d(tempo)[0]), 1)
        if cat not in {"break", "kick", "snare", "hat"}:
            key = estimate_key(mono, sr)
    m = NOTE_RE.search(path.stem.lower())
    note = (m.group(1).upper() + m.group(2)) if m and cat == "bass" else None
    return dict(duration=round(dur, 4), kind=kind, category=cat, bpm=bpm, key=key,
                note=note, peak=float(np.abs(y).max()), stem=int(dur > STEM_SECONDS))


def connect(db: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    if "stem" not in {r[1] for r in con.execute("PRAGMA table_info(samples)")}:
        con.execute("ALTER TABLE samples ADD COLUMN stem INTEGER DEFAULT 0")  # older DBs
    return con


def scan(sample_root: str | Path, db: str | Path) -> int:
    """Index every audio file under each top-level source folder. Returns count."""
    root = Path(sample_root).resolve()
    con = connect(db)
    n = 0
    for src in sorted(p for p in root.iterdir() if p.is_dir()):
        lic = detect_licence(src)
        for f in sorted(src.rglob("*")):
            if f.suffix.lower() not in AUDIO_EXT:
                continue
            rel = f.relative_to(root)
            info = analyse(f, rel)
            con.execute("INSERT OR REPLACE INTO samples(path,name,source,licence,duration,kind,category,bpm,key,note,peak,stem)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rel.as_posix(), f.stem, src.name, lic, info["duration"], info["kind"],
                         info["category"], info["bpm"], info["key"], info["note"], info["peak"], info["stem"]))
            n += 1
    con.commit()
    con.close()
    return n


def search(db: str | Path, category: str | None = None, kind: str | None = None,
           bpm: float | None = None, bpm_tol: float = 8, key: str | None = None,
           note: str | None = None, licence: str | None = None, release_safe: bool = False,
           text: str | None = None, limit: int = 50) -> list[dict]:
    q, a = ["1=1"], []
    for col, val in (("category", category), ("kind", kind), ("key", key), ("note", note), ("licence", licence)):
        if val:
            q.append(f"{col}=?"); a.append(val)
    if bpm:
        q.append("bpm BETWEEN ? AND ?"); a += [bpm - bpm_tol, bpm + bpm_tol]
    if text:
        q.append("path LIKE ?"); a.append(f"%{text}%")
    if release_safe:
        q.append(f"licence IN ({','.join('?' * len(RELEASE_SAFE))})"); a += sorted(RELEASE_SAFE)
    con = connect(db)
    rows = con.execute(f"SELECT * FROM samples WHERE {' AND '.join(q)} ORDER BY path LIMIT ?", (*a, limit)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def lookup(db: str | Path, path: str) -> dict | None:
    con = connect(db)
    r = con.execute("SELECT * FROM samples WHERE path=?", (path,)).fetchone()
    con.close()
    return dict(r) if r else None


def counts(db: str | Path) -> list[tuple]:
    con = connect(db)
    rows = con.execute("SELECT category, kind, COUNT(*) FROM samples GROUP BY 1,2 ORDER BY 1").fetchall()
    con.close()
    return [tuple(r) for r in rows]
