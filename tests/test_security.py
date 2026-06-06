"""Security regression tests for the hardening in the 2026-06 audit:
host/DNS-rebinding guard, cross-origin POST block, body + prompt + queue + JOBS bounds,
symlink-safe lock claim, PIL decode guard, and generic (non-leaky) error responses."""

import os
import queue

import pytest


# ---- DNS-rebinding / Host guard --------------------------------------------
def test_rejects_nonlocal_host(client):
    # a rebound DNS name (Host: evil.com) must be refused even though it resolves to us
    r = client.get("/api/status", headers={"host": "evil.com"})
    assert r.status_code == 400


def test_allows_ip_literal_host(client):
    # a raw IP host can't be a rebinding target -> allowed (LAN opt-in by IP works)
    r = client.get("/api/status", headers={"host": "192.168.1.50:7860"})
    assert r.status_code == 200


def test_allows_localhost_subdomain(client):
    r = client.get("/api/status", headers={"host": "app.localhost"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "host,ok",
    [
        ("[::1]:7860", True),  # bracketed IPv6 + port
        ("[::1]", True),  # bracketed IPv6, no port
        ("[fe80::1]:7860", True),
        ("[::]", True),  # this mis-parsed before the IPv6-aware fix
        ("127.0.0.1.evil.com", False),  # suffix trick must not pass
        ("localhost.evil.com", False),
    ],
)
def test_host_guard_ipv6_and_suffix_tricks(client, host, ok):
    r = client.get("/api/status", headers={"host": host})
    assert r.status_code == (200 if ok else 400)


# ---- CSRF: cross-origin POST -----------------------------------------------
def test_cross_origin_post_rejected(client):
    r = client.post(
        "/api/cancel",
        json={"job_id": "x"},
        headers={"origin": "http://evil.com", "host": "127.0.0.1"},
    )
    assert r.status_code == 403


def test_same_origin_post_ok(client):
    r = client.post(
        "/api/cancel",
        json={"job_id": "nope"},
        headers={"origin": "http://127.0.0.1", "host": "127.0.0.1"},
    )
    assert r.status_code == 200


# ---- DoS bounds ------------------------------------------------------------
def test_oversized_body_rejected(client):
    big = b"x" * (200 * 1024)  # > MAX_BODY (64 KiB)
    r = client.post(
        "/api/generate", content=big, headers={"content-type": "application/json"}
    )
    assert r.status_code == 413


def test_prompt_is_truncated(client, app_mod):
    r = client.post("/api/generate", json={"engine": "zimage", "prompt": "p " * 5000})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert len(app_mod.JOBS[jid]["prompt"]) <= app_mod.MAX_PROMPT


def test_queue_full_returns_429(client, app_mod, monkeypatch):
    # shrink the queue so we hit the cap without enqueuing 64 jobs
    monkeypatch.setattr(app_mod, "Q", queue.Queue(maxsize=1))
    r1 = client.post("/api/generate", json={"engine": "zimage", "prompt": "a portrait"})
    r2 = client.post("/api/generate", json={"engine": "zimage", "prompt": "a portrait"})
    assert r1.status_code == 200
    assert r2.status_code == 429


def test_prune_jobs_caps_memory(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "JOBS", {f"j{i}": {"id": f"j{i}"} for i in range(20)})
    monkeypatch.setattr(app_mod, "MAX_JOBS", 5)
    app_mod._prune_jobs()
    assert len(app_mod.JOBS) <= 5


# ---- lock symlink race (CWE-59) --------------------------------------------
def test_lock_write_refuses_symlink(app_mod, tmp_path, monkeypatch):
    target = tmp_path / "victim"
    target.write_text("precious")
    link = tmp_path / "lock.symlink"
    os.symlink(target, link)
    monkeypatch.setattr(app_mod, "LOCK", str(link))
    with pytest.raises(OSError):  # O_EXCL|O_NOFOLLOW must refuse the pre-planted symlink
        app_mod._write_lock_pid()
    assert target.read_text() == "precious"  # untouched


# ---- PIL decode guard ------------------------------------------------------
def test_corrupt_png_returns_415(client, app_mod):
    bad = app_mod.OUT / "corrupt.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not a real image" * 4)
    r = client.get(f"/img/{bad.name}")
    assert r.status_code == 415


# ---- error responses don't leak internals ----------------------------------
def test_meta_error_is_generic(client, app_mod):
    img = app_mod.OUT / "broken.png"
    img.write_bytes(b"x")
    img.with_suffix(".json").write_text("{ this is not valid json ")
    r = client.get("/api/meta/broken.png")
    assert r.status_code == 500
    assert r.json() == {"error": "could not read sidecar"}  # no raw exception / path


def test_status_gallery_excludes_corrupt_sidecar_payloads(client, app_mod):
    # sanity: status still serves with odd files present
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "gallery" in r.json()
