import re
import json

import numpy as np
import pytest

from mcp_daw import asr, comfy, dynamics, library, master

SR = 44100


def synth(bpm=128, bars=16, sr=SR, seed=0):
    """Kick on every beat + a sustained saw-ish pad, stereo."""
    rng = np.random.default_rng(seed)
    n = int(bars * 240 / bpm * sr)
    t = np.arange(n) / sr
    pad = 0.2 * np.sign(np.sin(2 * np.pi * 207.65 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
    kick = np.zeros(n)
    k = np.exp(-np.arange(int(0.2 * sr)) / (0.04 * sr)) * np.sin(2 * np.pi * 55 * np.arange(int(0.2 * sr)) / sr)
    for i in range(0, n, int(60 / bpm * sr)):
        m = min(len(k), n - i)
        kick[i:i + m] += 0.9 * k[:m]
    y = pad + kick + 0.01 * rng.standard_normal(n)
    return np.stack([y, y * 0.98], axis=1)


@pytest.mark.parametrize("engine", ["numpy", "ffmpeg"])
def test_master_hits_lufs_and_true_peak_ceiling(engine):
    if engine == "ffmpeg":
        pytest.importorskip("shutil").which("ffmpeg") or pytest.skip("no ffmpeg")
    y = synth(bars=4) * 0.3
    out, rep = master.master(y, SR, "edm", engine=engine)
    assert rep["true_peak_dbtp"] <= -1.0
    assert master.true_peak_db(out) <= -1.0
    assert abs(rep["lufs"] - (-10.5)) <= 1.0


def test_master_presets_cover_edm():
    assert master.PRESETS["edm"].target_lufs == -10.5 and master.PRESETS["edm"].ceiling_dbtp == -1.0


def test_beat_phase_finds_offset():
    y = synth(bars=8)
    off = int(0.1 * SR)
    shifted = np.vstack([np.zeros((off, 2)), y])[: len(y)]
    assert abs(dynamics.beat_phase(shifted, SR, 128) - 0.1) < 0.012


SECTIONS = [{"name": "intro", "start_bar": 0, "bars": 4}, {"name": "build", "start_bar": 4, "bars": 4},
            {"name": "drop", "start_bar": 8, "bars": 4}, {"name": "break", "start_bar": 12, "bars": 4}]


def test_dynamics_adds_section_contrast_and_pump():
    y = synth(bars=16)
    out, rep = dynamics.apply_dynamics(y, SR, 128, SECTIONS, "strong", phase=0.0)
    r = rep["section_rms_db"]
    assert r["drop"] - r["break"] >= 6 and r["drop"] - r["intro"] >= 6
    assert out.shape == y.shape and np.isfinite(out).all()


def test_pump_envelope_ducks_on_beats_in_drops_only():
    n, beat = int(16 * 240 / 128 * SR), int(60 / 128 * SR)
    el, eh = dynamics.pump_envelopes(n, SR, 128, SECTIONS, 5.0, 2.0, 180)
    hit = 8 * 4 * beat + 4 * beat  # a beat inside the drop
    assert 20 * np.log10(el[hit + int(0.01 * SR)]) < -4 and 20 * np.log10(eh[hit + int(0.01 * SR)]) > -2.5
    assert el[hit + beat - 10] > 0.95
    assert np.allclose(el[: 4 * 4 * beat], 1.0)  # intro untouched


def test_dynamics_has_no_clicks_on_bar_lines():
    y = synth(bars=16)
    out, _ = dynamics.apply_dynamics(y, SR, 128, SECTIONS, "strong", phase=0.0)
    d_in, d_out = np.abs(np.diff(y[:, 0])).max(), np.abs(np.diff(out[:, 0])).max()
    assert d_out <= d_in * 1.05


def test_roles():
    assert [dynamics.role(n) for n in ("drop2", "Chorus", "build", "intro", "break", "outro", "verse")] == \
        ["drop", "drop", "build", "quiet", "quiet", "quiet", "other"]


def test_asr_score():
    s = asr.score("lift me higher through the sky la la", "[chorus]\nLift me higher, through the sky")
    assert s["recall"] == 1.0 and s["precision"] == pytest.approx(6 / 8)


def test_comfy_workflows_are_wired_and_generic():
    v1 = comfy.m2m_v1_workflow("in.mp3", "trance", denoise=0.4, seed=7)
    v15 = comfy.vocal_v15_workflow("in.mp3", "trance, female vocal", "[verse]\nhello", 128, "G# minor", 187.5)
    for g in (v1, v15):
        for node in g.values():
            for v in node["inputs"].values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    assert v[0] in g, v
        assert g["ks"]["inputs"]["latent_image"] == ["enc", 0]
    assert v15["pos"]["inputs"]["bpm"] == 128 and v15["ks"]["inputs"]["steps"] == 16
    blob = json.dumps([v1, v15])
    assert not re.search(r"\d+\.\d+\.\d+\.\d+", blob) and "Users" not in blob


def test_comfy_url_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFYUI_URL", raising=False)
    (tmp_path / ".env").write_text("COMFYUI_URL=http://localhost:8188/\n")
    comfy.load_env(tmp_path)
    assert comfy.comfy_url() == "http://localhost:8188"
    monkeypatch.delenv("COMFYUI_URL")
    with pytest.raises(RuntimeError):
        comfy.comfy_url()


def test_comfy_client_run_downloads_audio(tmp_path, monkeypatch):
    calls = []

    def fake(self, path, data=None, headers=None):
        calls.append(path)
        if path == "/prompt":
            return b'{"prompt_id": "p1"}'
        if path.startswith("/history/"):
            return json.dumps({"p1": {"status": {"status_str": "success", "messages": [
                ["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 3500}]]},
                "outputs": {"9": {"audio": [{"filename": "a.mp3", "subfolder": "audio", "type": "output"}]}}}}).encode()
        if path.startswith("/view?"):
            return b"ID3fake"
        raise AssertionError(path)

    monkeypatch.setattr(comfy.ComfyClient, "_req", fake)
    res = comfy.ComfyClient("http://localhost:8188").run({"x": {}}, tmp_path / "o.mp3", poll=0)
    assert (tmp_path / "o.mp3").read_bytes() == b"ID3fake" and res["seconds"] == 2.5


def test_library_new_categories_and_stem_flag(tmp_path):
    import soundfile as sf
    src = tmp_path / "pack"
    src.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    for name in ("Lead_Loop.wav", "big_pad.wav", "vocal_chop.wav", "riser_up.wav", "pluck_c.wav", "chord_stab.wav"):
        sf.write(src / name, tone, SR)
    db = tmp_path / "l.db"
    library.scan(tmp_path, db)
    cats = {r["name"]: r["category"] for r in library.search(db, limit=50)}
    assert cats == {"Lead_Loop": "lead", "big_pad": "pad", "vocal_chop": "vocal", "riser_up": "riser",
                    "pluck_c": "pluck", "chord_stab": "chord"}
    assert all(r["stem"] == 0 for r in library.search(db, limit=50))


def test_unknown_presets_raise_clear_errors():
    with pytest.raises(ValueError, match="valid"):
        master.master(synth(bars=1), SR, "club")
    with pytest.raises(ValueError, match="valid"):
        dynamics.apply_dynamics(synth(bars=1), SR, 128, SECTIONS, "huge", phase=0.0)


def test_dynamics_without_overlapping_sections_is_passthrough_gain():
    y = synth(bars=2)
    assert np.allclose(dynamics.section_gain_curve(y, SR, 128, [], 8.0), 1.0)
    late = [{"name": "drop", "start_bar": 50, "bars": 4}]
    assert np.allclose(dynamics.section_gain_curve(y, SR, 128, late, 8.0), 1.0)
    out, _ = dynamics.apply_dynamics(y, SR, 128, [], "subtle", phase=0.0)
    assert out.shape == y.shape


def test_upload_name_is_sanitised():
    n = comfy.safe_upload_name('evil\r\nX-Injected: 1".wav')
    assert "\r" not in n and "\n" not in n and '"' not in n and n.endswith(".wav")
    assert comfy.safe_upload_name("../../etc/passwd") == "passwd"
