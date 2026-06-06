#!/usr/bin/env python3
"""persona-gen - self-serve local image generation UI (FastAPI, Apple Silicon / MPS).

Pick an engine, type a prompt, hit Generate, watch a live preview, browse results with
provenance sidecars. Config-driven (config.yaml): bring your own model checkpoints.

One heavy render at a time via a lock file; a single worker thread drains a queue and keeps
the chosen engine warm. Run:  python -m persona_gen.server --port 7860
"""

import argparse
import io
import ipaddress
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

try:
    from .engines import build_engines
except ImportError:  # running as a script, not a package
    from engines import build_engines

log = logging.getLogger("persona_gen")

# DoS / abuse limits (this is a local single-user app, but bound everything anyway)
MAX_PROMPT = 4000  # chars; longer is silently truncated (CLIP/T5 ignore the tail anyway)
MAX_QUEUE = 64  # pending jobs; reject further /api/generate with 429
MAX_JOBS = 500  # in-memory job records kept; oldest pruned beyond this
MAX_BODY = 64 * 1024  # request body bytes; larger -> 413

# A crafted/huge PNG dropped into the output dir must not be a decompression bomb.
Image.MAX_IMAGE_PIXELS = 64_000_000  # ~8000x8000; PIL raises above this


def _load_config():
    import os as _os

    for c in (
        _os.environ.get("PERSONA_GEN_CONFIG"),
        "config.yaml",
        str(Path.home() / ".config" / "persona-gen" / "config.yaml"),
    ):
        if c and Path(c).expanduser().exists():
            return yaml.safe_load(Path(c).expanduser().read_text() or "{}") or {}
    raise SystemExit(
        "No config found. Copy config.example.yaml -> config.yaml and set your model paths."
    )


CONFIG = _load_config()
OUT = Path(CONFIG.get("output_dir", "./output")).expanduser().resolve()
LOCK = str(Path(CONFIG.get("lock_file", "/tmp/persona-gen.lock")).expanduser())
OUT.mkdir(parents=True, exist_ok=True)


def _safe(name: str):
    """Resolve a flat gallery .png name safely (no traversal). Returns Path or None."""
    try:
        p = (OUT / str(name)).resolve()
    except Exception:
        return None
    return p if (p.parent == OUT.resolve() and p.suffix == ".png") else None


# ---- engine registry (built from config) -----------------------------------
_ENGINES = build_engines(CONFIG)  # {key: {engine, label, steps, cfg}}
ENGINES = {
    k: {"label": v["label"], "steps": v["steps"], "cfg": v["cfg"]} for k, v in _ENGINES.items()
}

_PIPE = {"engine": None, "obj": None}  # warm cache

# Fast latent->RGB preview (ComfyUI-style) so the UI shows a pixelated frame sharpening
# step-by-step. SDXL 4-channel factors; other latent spaces fall back to first-3-channels.
_SDXL_RGB = torch.tensor(
    [
        [0.3651, 0.4232, 0.4341],
        [-0.2533, -0.0042, 0.1068],
        [0.1076, 0.1111, -0.0362],
        [-0.3165, -0.2492, -0.2188],
    ]
)


def _preview_jpeg(latents, max_w: int = 240):
    try:
        x = latents[0].detach().to("cpu", torch.float32)  # [C,h,w]
        if x.shape[0] == 4:
            rgb = torch.einsum("chw,cr->rhw", x, _SDXL_RGB)
        else:
            rgb = x[:3]  # generic fallback (16-ch DiT etc.) - denoising still reads
        rgb = rgb.permute(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-5)
        im = Image.fromarray((rgb * 255).clamp(0, 255).byte().numpy(), "RGB")
        h = int(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.NEAREST)  # NEAREST keeps the blocky/pixelated look
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        return None


def _load(engine: str):
    if _PIPE["engine"] == engine:
        return _PIPE["obj"]
    _PIPE["obj"] = None  # free previous
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    obj = _ENGINES[engine]["engine"]
    obj.load()
    _PIPE["engine"], _PIPE["obj"] = engine, obj
    return obj


def _generate(job: dict):
    """Run one job (engine + prompt + params), writing images + updating progress."""
    eng = job["engine"]
    pipe = _load(eng)
    prompt = job["prompt"].strip()
    neg = job["negative"] or None  # empty -> engine's own default negative
    steps, cfg = int(job["steps"]), float(job["cfg"])
    w, h, n = int(job["width"]), int(job["height"]), int(job["count"])
    job["total_steps"] = steps

    for i in range(n):
        seed = int(job["seed"]) + i
        job["img_index"] = i + 1
        job["step"] = 0

        def _pcb(d):
            if d.get("step") is not None:
                job["step"], job["total_steps"] = d["step"], d.get("total_steps", steps)
            job["stage"] = d.get("stage")
            lat = d.get("latents")
            if lat is not None:
                pv = _preview_jpeg(lat)
                if pv:
                    job["preview"] = pv

        img = pipe.render(
            prompt,
            negative=neg,
            seed=seed,
            width=w,
            height=h,
            steps=steps,
            cfg=cfg,
            progress=_pcb,
            cancel=lambda: job.get("cancel"),
        )
        job["stage"] = None
        fp = OUT / f"{eng}-{job['id'][:8]}-{i:02d}-s{seed}.png"
        img.save(fp)
        # provenance sidecar next to the image (seed/model/params) - viewable in the UI
        meta = {
            "file": fp.name,
            "engine": eng,
            "model": _ENGINES.get(eng, {}).get("label", eng),
            "prompt": prompt,
            "negative": neg,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": w,
            "height": h,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "job_id": job["id"],
        }
        try:  # atomic: write temp then replace, so a gallery scan never reads a partial sidecar
            _tmp = fp.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            _tmp.replace(fp.with_suffix(".json"))
        except Exception:
            pass
        job["results"].append(fp.name)


# ---- worker thread + lock --------------------------------------------------
Q: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE)
JOBS: dict = {}
STATE = {"current": None, "lock": "free"}


def _prune_jobs():
    """Cap the in-memory job log so a long-running server can't grow unboundedly. Keep the
    newest MAX_JOBS by insertion order, but never drop the currently-running job."""
    if len(JOBS) <= MAX_JOBS:
        return
    cur = STATE.get("current")
    for jid in list(JOBS)[: len(JOBS) - MAX_JOBS]:
        if jid != cur:
            JOBS.pop(jid, None)


def _read_lock_pid():
    """Read the pid from the lock file without following a symlink (O_NOFOLLOW)."""
    try:
        fd = os.open(LOCK, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None  # missing, or a symlink someone pre-planted -> treat as not-ours
    try:
        return int(os.read(fd, 32).decode().strip())
    except (ValueError, OSError):
        return None
    finally:
        os.close(fd)


def _write_lock_pid():
    """Claim the lock atomically: O_EXCL|O_NOFOLLOW so a pre-planted symlink/file can't be
    clobbered (CWE-59). A stale file from a dead pid is removed first by the caller."""
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)


def _lock_held_by_other() -> bool:
    if not os.path.lexists(LOCK):
        return False
    pid = _read_lock_pid()
    if pid is None or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
        return True  # alive, someone else
    except Exception:
        return False  # stale


def _worker():
    while True:
        job = Q.get()
        JOBS[job["id"]] = job
        if job.get("cancel"):  # cancelled while still queued -> skip entirely
            job["status"] = "cancelled"
            Q.task_done()
            continue
        # wait for the heavy-job lock to be free (bail out if cancelled meanwhile)
        while _lock_held_by_other():
            if job.get("cancel"):
                break
            STATE["lock"] = "waiting (external batch)"
            job["status"] = "queued (lock held)"
            time.sleep(5)
        if job.get("cancel"):
            job["status"] = "cancelled"
            Q.task_done()
            continue
        # Claim the lock with O_EXCL|O_NOFOLLOW, retrying briefly on contention. We retry
        # IN PLACE rather than requeueing: this thread is the only consumer, so a blocking
        # re-put into a full queue would deadlock it (and reordering jobs is undesirable).
        claimed = False
        for _ in range(5):
            try:
                if os.path.lexists(LOCK) and not _lock_held_by_other():
                    os.remove(LOCK)
                _write_lock_pid()
                claimed = True
                break
            except OSError:  # lost the race / pre-planted symlink -> back off and retry
                time.sleep(1)
        if not claimed:
            job["status"] = "error: could not acquire render lock"
            Q.task_done()
            continue
        STATE["lock"] = f"held by ui pid {os.getpid()}"
        STATE["current"] = job["id"]
        job["status"] = "running"
        t = time.time()
        try:
            _generate(job)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            # log full detail locally; surface only a generic message to clients
            log.exception("job %s failed", job["id"])
            job["status"] = "cancelled" if job.get("cancel") else "error: render failed"
            job["error_detail"] = str(e)
        job["secs"] = round(time.time() - t, 1)
        STATE["current"] = None
        # release lock only if we own it
        try:
            if os.path.lexists(LOCK) and _read_lock_pid() == os.getpid():
                os.remove(LOCK)
        except Exception:
            pass
        STATE["lock"] = "free"
        Q.task_done()


# ---- API -------------------------------------------------------------------
app = FastAPI()
_THUMB: dict = {}

def _host_only(authority: str) -> str:
    """Extract the hostname from a Host/Origin authority, IPv6-bracket aware.
    '[::1]:7860' -> '::1', '127.0.0.1:7860' -> '127.0.0.1', 'localhost' -> 'localhost'."""
    a = (authority or "").strip().lower()
    if a.startswith("//"):
        a = a[2:]
    if a.startswith("["):  # [ipv6] or [ipv6]:port
        return a[1 : a.index("]")] if "]" in a else a[1:]
    return a.rsplit(":", 1)[0] if a.count(":") == 1 else a  # one colon = host:port


def _host_ok(host_header: str) -> bool:
    """DNS-rebinding guard. Accept localhost names and any IP-literal host (rebinding needs a
    *name* that resolves to a private IP; a raw IP host can't be rebound). Reject other names."""
    host = _host_only(host_header)
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(host)  # any literal IPv4/IPv6 -> fine
        return True
    except ValueError:
        return False


class _BodyLimit:
    """Pure-ASGI body-size guard. Buffers a POST body up to MAX_BODY and returns a clean 413
    the moment it is exceeded - for BOTH honest (Content-Length set) and chunked/absent-CL
    clients (the CL header alone is bypassable). Memory is bounded to ~MAX_BODY + one chunk;
    the exact buffered body is then replayed to the handler, so no valid prefix can slip
    through with an oversized suffix silently dropped."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        body = b""
        more = True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                body = None
                break
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
            if len(body) > MAX_BODY:
                return await JSONResponse(
                    {"error": "request too large"}, status_code=413
                )(scope, receive, send)

        sent = False

        async def replay():  # hand the fully-buffered body to the downstream handler
            nonlocal sent
            if sent or body is None:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return await self.app(scope, replay, send)


app.add_middleware(_BodyLimit)


@app.middleware("http")
async def _guard(request: Request, call_next):
    """DNS-rebinding guard (all methods) + cross-origin POST block (CSRF)."""
    if not _host_ok(request.headers.get("host", "")):
        return JSONResponse({"error": "bad host"}, status_code=400)
    if request.method == "POST":
        # A browser sends Origin on cross-site POSTs; same-origin calls match the Host.
        origin = request.headers.get("origin")
        if origin:
            o_host = _host_only(origin.split("://", 1)[-1])
            req_host = _host_only(request.headers.get("host", ""))
            if o_host != req_host and o_host not in ("localhost", "127.0.0.1", "::1"):
                return JSONResponse({"error": "cross-origin POST rejected"}, status_code=403)
    return await call_next(request)


def _num(v, default, lo, hi, integer=True):
    """Bulletproof numeric coercion: letters/empty/null/NaN/inf -> default, then clamp [lo,hi].
    Users WILL type garbage (even letters) in these fields - nothing here may crash the job."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return default
    v = max(lo, min(hi, v))
    return int(v) if integer else round(v, 2)


def _snap(v, default: int) -> int:
    """Idiot-proof dimension: round to a multiple of 16 (Z-Image + SDXL safe), clamp 512-1536.
    Prevents the 'Width must be divisible by 16' crash on free-typed sizes like 1001 or 'abc'."""
    return int(round(_num(v, default, 512, 1536) / 16)) * 16


@app.post("/api/generate")
async def api_generate(body: dict):
    eng = body.get("engine", "pony")
    if eng not in ENGINES:
        return JSONResponse({"error": "unknown engine"}, status_code=400)
    sp = ENGINES[eng]
    job = {
        "id": uuid.uuid4().hex,
        "engine": eng,
        "prompt": str(body.get("prompt", "") or "").strip()[:MAX_PROMPT],
        "negative": str(body.get("negative", "") or "").strip()[:MAX_PROMPT],
        "steps": _num(body.get("steps"), sp["steps"], 1, 80),
        "cfg": _num(body.get("cfg"), sp["cfg"], 0, 20, integer=False),
        "width": _snap(body.get("width"), 832),
        "height": _snap(body.get("height"), 1216),
        "count": _num(body.get("count"), 1, 1, 8),
        "seed": _num(body.get("seed"), 0, 0, 2_000_000_000),
        "status": "queued",
        "step": 0,
        "total_steps": 0,
        "img_index": 0,
        "results": [],
        "created": time.strftime("%H:%M:%S"),
    }
    if not job["prompt"]:
        return JSONResponse(
            {"error": "Type a prompt - an English description of what to generate."},
            status_code=400,
        )
    if not job["seed"]:
        job["seed"] = int(time.time()) % 2_000_000_000
    try:
        Q.put_nowait(job)
    except queue.Full:
        return JSONResponse({"error": "queue full, try again shortly"}, status_code=429)
    JOBS[job["id"]] = job
    _prune_jobs()
    return {"job_id": job["id"], "queue_pos": Q.qsize()}


@app.get("/api/status")
async def api_status():
    recent = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:20]
    cur = JOBS.get(STATE["current"]) if STATE["current"] else None
    # Gallery scans the OUTPUT DIR (persistent) - not the in-memory job list, so images never
    # vanish when error/new jobs scroll past the recent window.
    try:
        files = sorted(OUT.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:80]
        gallery = [p.name for p in files]
    except Exception:
        gallery = []
    errors = [
        {"engine": j.get("engine"), "status": j["status"], "prompt": j.get("prompt", "")}
        for j in recent
        if str(j.get("status", "")).startswith("error")
    ][:4]
    waiting = [
        {
            "id": j["id"],
            "engine": j.get("engine"),
            "count": j.get("count"),
            "prompt": j.get("prompt", ""),
        }
        for j in sorted(JOBS.values(), key=lambda j: j["created"])
        if str(j.get("status", "")).startswith("queued") and j["id"] != STATE["current"]
    ]
    return JSONResponse(
        {
            "lock": STATE["lock"],
            "qsize": Q.qsize(),
            "current": cur
            and {
                k: cur.get(k)
                for k in (
                    "id",
                    "engine",
                    "status",
                    "step",
                    "total_steps",
                    "img_index",
                    "count",
                    "prompt",
                    "stage",
                )
            },
            "queue": waiting,
            "gallery": [_gitem(n) for n in gallery],
            "errors": errors,
            "engines": {
                k: {"label": v.get("label"), "steps": v["steps"], "cfg": v["cfg"]}
                for k, v in ENGINES.items()
            },
        }
    )


@app.get("/img/{rel:path}")
def img(rel: str):
    p = _safe(rel)
    if not p or not p.exists():
        return Response(status_code=404)
    key = (str(p), os.path.getmtime(p))
    if key not in _THUMB:
        try:  # a corrupt or oversized PNG (decompression bomb) must not crash the route
            im = Image.open(p)
            im.load()
            im = im.convert("RGB")
        except Exception:  # noqa: BLE001 - PIL DecompressionBombError, OSError, etc.
            log.warning("could not decode %s", p.name)
            return Response(status_code=415)
        im.thumbnail((512, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        _THUMB[key] = buf.getvalue()
        if len(_THUMB) > 200:
            _THUMB.pop(next(iter(_THUMB)))
    return Response(_THUMB[key], media_type="image/jpeg")


@app.get("/api/preview/{job_id}")
def preview(job_id: str):
    j = JOBS.get(job_id)
    if not j or not j.get("preview"):
        return Response(status_code=404)
    return Response(j["preview"], media_type="image/jpeg")


@app.post("/api/cancel")
async def api_cancel(body: dict):
    """Stop a job. {'job_id': '<id>'} cancels one; {'job_id': 'all'} cancels the running
    job + everything queued. Running job aborts at its next step; queued jobs are skipped."""
    jid = str(body.get("job_id", "")).strip()
    n = 0
    if jid == "all":
        for j in JOBS.values():
            if str(j.get("status", "")).startswith(("queued", "running")):
                j["cancel"] = True
                n += 1
    elif jid in JOBS:
        JOBS[jid]["cancel"] = True
        n = 1
    return {"cancelled": n}


@app.get("/api/meta/{name}")
def meta(name: str):
    p = _safe(name)
    if not p:
        return JSONResponse({"error": "bad path"}, status_code=400)
    sc = p.with_suffix(".json")
    if not sc.exists():
        return JSONResponse({"error": "no sidecar for this image"}, status_code=404)
    try:
        return JSONResponse(json.loads(sc.read_text()))
    except Exception:  # noqa: BLE001
        log.exception("meta read failed for %s", name)
        return JSONResponse({"error": "could not read sidecar"}, status_code=500)


# ---- gallery management: favorites + per-image meta cache + delete + dup ----
_FAV_FILE = OUT / ".favorites.json"
try:
    _FAV = set(json.loads(_FAV_FILE.read_text()))
except Exception:
    _FAV = set()
_GMETA: dict = {}  # name -> {engine, nsfw}; sidecars are immutable so cache once


def _save_fav():
    try:
        _FAV_FILE.write_text(json.dumps(sorted(_FAV)))
    except Exception:
        pass


def _gitem(name: str) -> dict:
    m = _GMETA.get(name)
    if m is None:
        try:
            d = json.loads((OUT / name).with_suffix(".json").read_text())
            m = _GMETA[name] = {"engine": d.get("engine") or name.split("-", 1)[0]}
        except Exception:
            # sidecar missing or being written (PNG lands before JSON) -> DON'T cache;
            # return a filename fallback and retry on the next poll once the JSON is complete.
            return {"name": name, "engine": name.split("-", 1)[0], "fav": name in _FAV}
    return {"name": name, "engine": m["engine"], "fav": name in _FAV}


@app.post("/api/favorite")
async def api_favorite(body: dict):
    name = str(body.get("name", ""))
    p = _safe(name)
    if not p or not p.exists():
        return JSONResponse({"error": "bad path"}, status_code=400)
    _FAV.add(name) if body.get("on") else _FAV.discard(name)
    _save_fav()
    return {"fav": name in _FAV}


@app.post("/api/delete")
async def api_delete(body: dict):
    """Delete one image ({'name': ...}) or many ({'names': [...]}) plus their sidecars."""
    names = body.get("names") or ([body["name"]] if body.get("name") else [])
    deleted = []
    for name in names:
        p = _safe(str(name))
        if not p or not p.exists():
            continue
        try:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
            _FAV.discard(str(name))
            _GMETA.pop(str(name), None)
            deleted.append(str(name))
        except OSError:
            pass
    _save_fav()
    return {"deleted": deleted}


@app.post("/api/queue/dup")
async def api_dup(body: dict):
    """Re-enqueue a copy of a job by id (fresh random seed)."""
    src = JOBS.get(str(body.get("job_id", "")))
    if not src:
        return JSONResponse({"error": "no such job"}, status_code=404)
    job = {
        k: src.get(k)
        for k in ("engine", "prompt", "negative", "steps", "cfg", "width", "height", "count")
    }
    job.update(
        id=uuid.uuid4().hex,
        seed=int(time.time()) % 2_000_000_000,
        status="queued",
        step=0,
        total_steps=0,
        img_index=0,
        results=[],
        created=time.strftime("%H:%M:%S"),
    )
    try:  # respect the queue cap; never block the async handler
        Q.put_nowait(job)
    except queue.Full:
        return JSONResponse({"error": "queue full, try again shortly"}, status_code=429)
    JOBS[job["id"]] = job
    _prune_jobs()
    return {"job_id": job["id"]}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    # Loopback by default: the API has no auth and can generate/delete, so it must NOT be
    # reachable from the LAN unless the user explicitly opts in with --host.
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default 127.0.0.1, localhost only; pass 0.0.0.0 to expose on LAN)",
    )
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Binding to %s exposes this UNAUTHENTICATED app to your network - anyone who can "
            "reach this port can generate and delete images. Use 127.0.0.1 unless you mean to.",
            a.host,
        )
    threading.Thread(target=_worker, daemon=True).start()
    print(f"[gen-ui] http://localhost:{a.port}  (engines: {', '.join(ENGINES)})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
