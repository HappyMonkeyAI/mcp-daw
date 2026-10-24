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
`add_loop_track`, `set_mix(song,track,gain_db,pan)`, `describe_song`, `render(song, mp3=True)`, `list_songs`.

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

## Known limits
- Categories come from file/folder names first, then spectral centroid (`low`/`mid`/`high`) as fallback.
- Loop detection = name hint (loop/brk/break) or ≥3 s with ≥8 onsets; Swirly's snare rolls/stirs count as loops.
- Pitch shifting is sampler-style resampling (pitch and length change together).

## License
Code: MIT, (c) 2026 Stephen Phillips. See `LICENSE`.
Bundled samples keep their own licences (Unlicense / CC0); see each folder under `samples/`.
