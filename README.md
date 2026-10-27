# mcp-daw: an MCP music sequencer for AI agents

mcp-daw lets an AI agent make music from loops and one-shot samples over the Model Context Protocol (MCP).
The agent edits a plain JSON **song file** (tempo, key, swing, step patterns, loop placements), and a
renderer turns it into a 44.1 kHz stereo WAV/MP3. Samples are indexed with their licence, and the renderer
warns if a song uses anything that isn't safe to release.

Example: `songs/lofi_full.json` (built by `build_lofi.py`) is a 96-second lo-fi track at 80 BPM.

Core functions (wrapped by the MCP server below):

| MCP tool  | function |
|------------------|----------|
| `search_samples` | `library.search(db, category, kind, bpm, key, note, licence, release_safe, text)` |
| `new_song`       | `song.new_song(bpm, bars, key, swing)` |
| `add_pattern_track` | `song.add_pattern_track(...)` |
| `add_loop_track` | `song.add_loop_track(...)` |
| `describe_song`  | `song.describe(song)` |
| `render`         | `render.render(song, sample_root, out_path, db)` |
## Setup
```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scan.py                                  # index samples/ -> library.db
python render_song.py songs/demo.json out/demo.wav --mp3
pytest -q
```

## Windows setup
```
cd mcp-daw
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scan.py
pytest -q
```

## MCP server
`mcp_server.py` is a FastMCP stdio server (requires `mcp<2`; mcp 2.x renamed FastMCP).
Project root = `MCP_DAW_ROOT` env var, or the folder containing `mcp_server.py`.
Songs live in `songs/<name>.json`, renders in `out/<name>.wav|.mp3`; all paths in tool I/O are relative to the root.
If `library.db` is missing it is built on first use.

Tools: `search_samples`, `list_categories`, `new_song(name,bpm,key,bars,swing)`, `add_pattern_track`,
`add_loop_track`, `set_mix(song,track,gain_db,pan)`, `describe_song`, `render(song, mp3=True)`, `list_songs`,
`master_audio`, `apply_arrangement_dynamics`, `comfy_m2m`, `comfy_vocal_pass`, `check_lyrics` (see below).

Example client config (Claude Desktop / Cursor / LM Studio style). Use the venv's python so deps are found:
```json
{
  "mcpServers": {
    "mcp_daw": {
      "command": "C:\\path\\to\\mcp-daw\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\mcp-daw\\mcp_server.py"],
      "env": { "MCP_DAW_ROOT": "C:\\path\\to\\mcp-daw" }
    }
  }
}
```
(`"command": "python"` also works if that python has the requirements installed.)

## Song file
```json
{"bpm": 94, "key": "C minor", "bars": 4, "swing": 0.35,
 "sections": [{"name": "groove", "start_bar": 0, "bars": 4}],
 "tracks": [
  {"name": "kick", "type": "pattern", "sample": "uzu-drumkit/bd/11_bd_mot4i.wav",
   "steps": [1,0,0,0, 0,0,0,0.6, 0,0,1,0, 0,0,0,0], "gain_db": 0, "pan": 0},
  {"name": "bass", "type": "pattern", "sample": "karoryfer.meatbass/Samples/pizz/c2_vl4_rr1.wav",
   "steps": [...16 velocities...], "pitches": [...16 semitone shifts...]},
  {"name": "break", "type": "loop", "sample": "uzu-drumkit/brk/10_break_amen_pprocessed.wav",
   "loop_bars": 1, "start_bars": [2, 3], "gain_db": -10}
 ]}
```
- `steps`: 16 sixteenth-notes per bar, value = velocity (0 = off). Optional `bars: [0,2]` limits which bars play.
- `swing` 0–1 delays odd 16ths by up to half a step.
- Loops are time-stretched (librosa) to exactly `loop_bars` at the song BPM.
- Sample paths are relative to `samples/`. Mix is normalised to −1 dBFS; tails are cut at the song end.

## Samples (all release-safe)
| folder | licence | source |
|---|---|---|
| `uzu-drumkit` | Unlicense | github.com/tidalcycles/uzu-drumkit |
| `karoryfer.swirly-drums` | CC0 | github.com/sfzinstruments/karoryfer.swirly-drums (trimmed: dry mics, rr1, 4 velocity layers) |
| `karoryfer.meatbass` | CC0 | github.com/sfzinstruments/karoryfer.meatbass (pizz only, rr1) |

Licence is read from each source folder's LICENSE file and stored per sample. The renderer warns on anything
not in `library.RELEASE_SAFE` (CC0, Unlicense, CC-BY; CC-BY gets a "credit the author" note).
Unlicensed sets such as Dirt-Samples should go in their own folder; they'll index as `unknown` and trigger warnings.

## Mastering and arrangement dynamics
`mcp_daw/master.py` normalises to a loudness target with **pyloudnorm** (ITU-R BS.1770) and then limits the
true peak. The limiter is ffmpeg's `alimiter` run at 4x the sample rate (a numpy look-ahead limiter is used if
ffmpeg is missing). The output is re-measured at 4x oversampling and the ceiling is tightened until the true
peak really is at or below the target.

| preset | LUFS | ceiling |
|---|---|---|
| `streaming` / `balanced` | -14 | -1 dBTP |
| `warm` | -13 | -1 dBTP |
| `pop` | -12.5 | -1 dBTP |
| `punch` | -12 | -1 dBTP |
| `rock` | -11.5 | -1 dBTP |
| `hiphop` | -11 | -1 dBTP |
| `edm` | -10.5 | -1 dBTP |

`mcp_daw/dynamics.py` uses the song's `bpm` and `sections` to shape a mix (or an AI re-take of it):
- finds the beat phase from onsets, then ducks on every beat of `build` (second half) and `drop`/`chorus`
  sections. The signal is split at 150 Hz (zero-phase) so the low band ducks harder than the rest.
- section contrast: `intro`/`break`/`outro` are cut relative to the loudest drop; builds ramp up; gain
  changes are smoothed over ~25 ms at bar lines.
- pre-drop sweep: the 4 bars before each drop blend towards a 400 Hz low-cut.

Presets: `subtle` (3 dB / 1.2 dB duck, ~6 dB drop-vs-break after mastering) and `strong` (5 dB / 2 dB, ~8 dB).

```python
from mcp_daw import dynamics, master
out, info = dynamics.apply_dynamics(audio, sr, song["bpm"], song["sections"], "strong")
out, report = master.master(out, sr, "edm")   # report: lufs, true_peak_dbtp
```

## ComfyUI / ACE-Step (optional)
Set `COMFYUI_URL` in the environment or in a `.env` file (see `.env.example`). The workflows are built in
code (`mcp_daw/comfy.py`); ComfyUI needs the ACE-Step checkpoints `ace_step_v1_3.5b.safetensors` and
`ace_step_1.5_turbo_aio.safetensors`.
- `comfy_m2m(path, tags, lyrics, denoise)`: ACE-Step v1 music-to-music. ~0.3-0.5 keeps the arrangement, 0.7 restyles it.
- `comfy_vocal_pass(song, path, lyrics, tags)`: ACE-Step 1.5 turbo over an instrumental, conditioned on the
  song's BPM, key and the file's duration (16 steps, dpmpp_2m, CFG 1.4). Use `[verse]`/`[chorus]`/`[instrumental]`
  tags in the lyrics. Don't name real artists in tags.

Generated audio inherits the licences of whatever you feed in. Keep renders of non-release-safe samples private.

## Lyric intelligibility check (optional)
`pip install -r requirements-asr.txt` (faster-whisper), then `check_lyrics(path, lyrics, start, duration)`
returns `precision` (share of heard words that are in the lyrics), `recall` (share of the lyric vocabulary
heard) and the transcript. It's a rough guide for comparing takes, not a score of the singing.

## Known limits
- Categories come from file/folder names first (kick, snare, hat, bass, lead, pad, vocal, chord/stab, arp,
  pluck, riser, fx, ...), then a simple fallback: noisy -> `fx`, rising envelope -> `riser`, sustained -> `pad`,
  otherwise spectral centroid (`low`/`mid`/`high`). Files over 60 s get `stem = 1` (stems / full bounces).
- Loop detection = name hint (loop/brk/break) or ≥3 s with ≥8 onsets; Swirly's snare rolls/stirs count as loops.
- Pitch shifting is sampler-style resampling (pitch and length change together).

## License
Code: MIT, (c) 2026 Stephen Phillips. See `LICENSE`.
Bundled samples keep their own licences (Unlicense / CC0); see each folder under `samples/`.
