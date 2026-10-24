from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mcp_daw import library, render, song

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
KICK = "uzu-drumkit/bd/11_bd_mot4i.wav"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    p = tmp_path_factory.mktemp("lib") / "lib.db"
    library.scan(SAMPLES, p)
    return p


def test_index_finds_kicks(db):
    kicks = library.search(db, category="kick", kind="one_shot", release_safe=True)
    assert len(kicks) >= 5
    assert any(k["path"] == KICK and k["licence"] == "Unlicense" for k in kicks)


def test_index_finds_break_loop_with_bpm(db):
    loops = library.search(db, kind="loop", category="break")
    assert loops and loops[0]["bpm"] > 0


@pytest.mark.parametrize("bpm,bars", [(94, 4), (120, 2), (174, 8)])
def test_render_length_matches_bars_and_bpm(bpm, bars):
    s = song.new_song(bpm=bpm, bars=bars)
    song.add_pattern_track(s, "kick", KICK, [1, 0, 0, 0] * 4)
    mix, _ = render.render(s, SAMPLES)
    assert mix.shape == (round(bars * 4 * 60 / bpm * render.SR), 2)


def first_hit(mix, after):
    env = np.abs(mix[:, 0])
    return after + int(np.argmax(env[after:] > 0.05 * env.max()))


def test_swing_delays_offbeats_only():
    bpm, step = 120, int(60 / 120 / 4 * render.SR)
    results = {}
    for sw in (0.0, 0.5):
        s = song.new_song(bpm=bpm, bars=1, swing=sw)
        song.add_pattern_track(s, "hit", KICK, [0, 1] + [0] * 14)  # step 1 = offbeat 16th
        results[sw] = first_hit(render.render(s, SAMPLES)[0], 0)
    assert abs(results[0.0] - step) < 50
    assert abs(results[0.5] - results[0.0] - 0.25 * step) < 50
    assert render.step_onset(2, 0, bpm, 0.5) == render.step_onset(2, 0, bpm, 0.0)


def test_render_writes_normalised_wav_and_flags_unsafe_licence(tmp_path, monkeypatch):
    s = song.new_song(bpm=100, bars=1)
    song.add_pattern_track(s, "kick", KICK, [1, 0, 0, 0] * 4, gain_db=12)
    out = tmp_path / "x.wav"
    _, warns = render.render(s, SAMPLES, out)
    y, sr = sf.read(out)
    assert sr == 44100 and y.shape[1] == 2 and 0 < np.abs(y).max() < 1.0
    assert warns == []
    monkeypatch.setattr(library, "detect_licence", lambda root: "unknown")
    _, warns = render.render(s, SAMPLES)
    assert warns and "NOT release-safe" in warns[0]
