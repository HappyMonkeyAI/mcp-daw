import asyncio
import json
import shutil
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KICK = "uzu-drumkit/bd/11_bd_mot4i.wav"
SNARE = "uzu-drumkit/sd/11_sd_switchangel_3.wav"
HAT = "uzu-drumkit/hh/11_hh_mot4i.wav"


@pytest.fixture
def srv(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT))
    import mcp_server
    monkeypatch.setattr(mcp_server, "SONGS", tmp_path / "songs")
    monkeypatch.setattr(mcp_server, "OUT", ROOT / "out" / "_test")
    return mcp_server


def test_tools_in_process(srv):
    assert any(c["category"] == "kick" for c in srv.list_categories())
    kicks = srv.search_samples(category="kick", kind="one_shot")
    assert kicks and all(k["licence"] in ("CC0-1.0", "Unlicense") for k in kicks)
    srv.new_song("t1", bpm=100, bars=2, swing=0.2)
    srv.add_pattern_track("t1", "kick", KICK, [1, 0, 0, 0] * 4)
    srv.add_loop_track("t1", "brk", "uzu-drumkit/brk/10_break_amen_pprocessed.wav", 1, [1])
    out = srv.set_mix("t1", "kick", gain_db=-3, pan=2)
    assert "kick" in out and srv._load("t1")["tracks"][0]["pan"] == 1.0
    assert srv.list_songs() == ["t1"]
    r = srv.render("t1", mp3=False)
    assert r["duration_s"] == pytest.approx(2 * 4 * 60 / 100, abs=1e-3)
    assert r["wav"] == "out/_test/t1.wav" and (ROOT / r["wav"]).exists() and r["licence_warnings"] == []


def test_bad_inputs(srv):
    with pytest.raises(ValueError):
        srv.new_song("../evil")
    srv.new_song("t2")
    with pytest.raises(ValueError):
        srv.add_pattern_track("t2", "x", "nope.wav", [0] * 16)


def test_stdio_smoke(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        root = tmp_path / "proj"
        root.mkdir()
        for d in ("samples", "mcp_daw"):
            try:
                os.symlink(ROOT / d, root / d, target_is_directory=True)
            except OSError:  # Windows without symlink rights
                shutil.copytree(ROOT / d, root / d)
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")],
                                       env={**os.environ, "MCP_DAW_ROOT": str(root)})
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert {"search_samples", "list_categories", "new_song", "add_pattern_track", "add_loop_track",
                    "set_mix", "describe_song", "render", "list_songs"} <= names
            call = lambda n, **a: s.call_tool(n, a)
            await call("new_song", name="smoke", bpm=96, bars=4, swing=0.3)
            await call("add_pattern_track", song="smoke", track="kick", sample=KICK, steps=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
            await call("add_pattern_track", song="smoke", track="snare", sample=SNARE, steps=[0, 0, 0, 0, 1, 0, 0, 0] * 2)
            await call("add_pattern_track", song="smoke", track="hats", sample=HAT, steps=[0.7, 0.4] * 8, gain_db=-8)
            res = await call("render", song="smoke", mp3=True)
            assert not res.isError, res
            data = json.loads(res.content[0].text)
            assert data["duration_s"] == pytest.approx(4 * 4 * 60 / 96, abs=1e-3)
            assert (root / data["wav"]).exists()
            assert "smoke" in (await call("list_songs")).content[0].text
    asyncio.run(run())
