"""ComfyUI client + ACE-Step workflows (built in code, no bundled third-party JSON).

Configure with COMFYUI_URL in the environment or a .env file in the project root.
Requires ComfyUI with the ACE-Step checkpoints installed:
  * v1:  ace_step_v1_3.5b.safetensors        (music-to-music polish)
  * 1.5: ace_step_1.5_turbo_aio.safetensors  (vocal pass with bpm/key/duration conditioning)
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

V1_CKPT = "ace_step_v1_3.5b.safetensors"
V15_CKPT = "ace_step_1.5_turbo_aio.safetensors"
# Sampler settings known to give coherent 1.5-turbo vocals (16 steps, low CFG).
V15_PRESET = dict(steps=16, cfg=1.4, sampler_name="dpmpp_2m", scheduler="sgm_uniform", cfg_scale=3.5)
V15_NEGATIVE = "low quality, glitch, distorted, clipping, harsh artifacts, mumbled, slurred vocals"


def load_env(root: str | Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides real environment variables."""
    p = Path(root) / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def comfy_url() -> str:
    url = os.environ.get("COMFYUI_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("COMFYUI_URL is not set (environment or .env)")
    return url


def _common_ace(ckpt: str, audio_name: str) -> dict:
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "load": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "enc": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["load", 0], "vae": ["ckpt", 2]}},
        "dec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}},
    }


def m2m_v1_workflow(audio_name: str, tags: str, lyrics: str = "[instrumental]", lyrics_strength: float = 1.0,
                    denoise: float = 0.5, seed: int = 0, steps: int = 50, cfg: float = 5.0,
                    prefix: str = "audio/mcp_daw_m2m") -> dict:
    g = _common_ace(V1_CKPT, audio_name)
    g.update({
        "pos": {"class_type": "TextEncodeAceStepAudio",
                "inputs": {"clip": ["ckpt", 1], "tags": tags, "lyrics": lyrics, "lyrics_strength": lyrics_strength}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["ckpt", 0], "shift": 5.0}},
        "tone": {"class_type": "LatentOperationTonemapReinhard", "inputs": {"multiplier": 1.0}},
        "cfgop": {"class_type": "LatentApplyOperationCFG", "inputs": {"model": ["shift", 0], "operation": ["tone", 0]}},
        "ks": {"class_type": "KSampler", "inputs": {
            "model": ["cfgop", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["enc", 0],
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
            "denoise": denoise}},
        "save": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["dec", 0], "filename_prefix": prefix, "quality": "V0"}},
    })
    return g


def vocal_v15_workflow(audio_name: str, tags: str, lyrics: str, bpm: int, keyscale: str, duration: float,
                       denoise: float = 0.62, seed: int = 0, language: str = "en", timesignature: str = "4",
                       prefix: str = "audio/mcp_daw_vocal") -> dict:
    p = V15_PRESET
    g = _common_ace(V15_CKPT, audio_name)
    g.update({
        "pos": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["ckpt", 1], "tags": tags, "lyrics": lyrics, "seed": seed, "bpm": int(bpm),
            "duration": float(duration), "timesignature": timesignature, "language": language,
            "keyscale": keyscale, "generate_audio_codes": True, "cfg_scale": p["cfg_scale"],
            "temperature": 0.85, "top_p": 0.9, "top_k": 0, "min_p": 0}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": V15_NEGATIVE}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["ckpt", 0], "shift": 3}},
        "ks": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["enc", 0],
            "seed": seed, "steps": p["steps"], "cfg": p["cfg"], "sampler_name": p["sampler_name"],
            "scheduler": p["scheduler"], "denoise": denoise}},
        "save": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["dec", 0], "filename_prefix": prefix, "quality": "V0"}},
    })
    return g


def safe_upload_name(name: str) -> str:
    """Keep upload filenames to a conservative charset (no CR/LF/quotes/path separators in the multipart header)."""
    import re
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name).lstrip(".")
    return clean or f"mcp_daw_{uuid.uuid4().hex[:8]}"


class ComfyClient:
    def __init__(self, url: str | None = None, timeout: float = 1800):
        self.url = (url or comfy_url()).rstrip("/")
        self.timeout = timeout

    def _req(self, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def upload(self, path: str | Path, name: str | None = None) -> str:
        path = Path(path)
        name = name or f"mcp_daw_{uuid.uuid4().hex[:8]}{path.suffix}"
        name = safe_upload_name(name)
        b = "----mcpdaw" + uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + \
               f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
        res = json.loads(self._req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def run(self, workflow: dict, out_path: str | Path, poll: float = 3.0) -> dict:
        """Queue a workflow, wait for it, download the first audio output to out_path."""
        body = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode()
        pid = json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            h = json.loads(self._req(f"/history/{pid}"))
            if pid in h:
                break
            time.sleep(poll)
        else:
            raise TimeoutError(f"ComfyUI job {pid} did not finish in {self.timeout}s")
        st = h[pid].get("status", {})
        if st.get("status_str") != "success":
            raise RuntimeError(f"ComfyUI job {pid} failed: {st.get('messages', [])[-1:]}")
        files = [a for o in h[pid]["outputs"].values() for a in o.get("audio", [])]
        if not files:
            raise RuntimeError(f"ComfyUI job {pid} produced no audio")
        f = files[0]
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(self._req(f"/view?{q}"))
        ts = [m[1].get("timestamp") for m in st.get("messages", [])
              if m[0] in ("execution_start", "execution_success")]
        return {"prompt_id": pid, "out": str(out_path),
                "seconds": round((ts[-1] - ts[0]) / 1000, 2) if len(ts) > 1 else None}
