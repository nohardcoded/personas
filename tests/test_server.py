"""API + helper tests. The worker thread is not started on import, so /api/generate
enqueues without loading any model."""

import json


def test_num_coerces_garbage(app_mod):
    n = app_mod._num
    for bad in ("abc", "", None, "nan", "inf", "-inf", {}, []):
        assert n(bad, 9, 1, 80) == 9
    assert n(-999, 5, 1, 80) == 1
    assert n(10**9, 5, 1, 80) == 80
    assert n("28", 9, 1, 80) == 28
    assert n("6.5", 0, 0, 20, integer=False) == 6.5


def test_snap_dims(app_mod):
    s = app_mod._snap
    assert s("abc", 832) == 832
    assert s(1001, 832) == 1008
    assert s(100, 832) == 512
    assert s(9999, 832) == 1536
    assert s(1024, 832) % 16 == 0


def test_safe_path_guard(app_mod):
    safe = app_mod._safe
    p = app_mod.OUT / "x.png"
    p.write_bytes(b"x")
    assert safe("x.png") == p.resolve()
    assert safe("../x.png") is None
    assert safe("x.txt") is None
    assert safe("sub/x.png") is None


def test_generate_validation(client):
    assert client.post("/api/generate", json={"engine": "zimage", "prompt": ""}).status_code == 400
    assert (
        client.post("/api/generate", json={"engine": "nope", "prompt": "a cat"}).status_code == 400
    )
    r = client.post(
        "/api/generate",
        json={
            "engine": "zimage",
            "prompt": "a cat on a mat",
            "width": "abc",
            "steps": "xx",
            "count": "99",
            "seed": "q",
        },
    )
    assert r.status_code == 200 and "job_id" in r.json()


def test_status_shape(client):
    d = client.get("/api/status").json()
    assert "zimage" in d["engines"] and "sdxl" in d["engines"]
    assert "gallery" in d and "queue" in d and "lock" in d


def test_meta_and_batch_delete(client, app_mod):
    out = app_mod.OUT
    (out / "zimage-t-00.png").write_bytes(b"png")
    (out / "zimage-t-00.json").write_text(json.dumps({"engine": "zimage", "seed": 1}))
    assert client.get("/api/meta/zimage-t-00.png").json()["engine"] == "zimage"
    r = client.post("/api/delete", json={"names": ["zimage-t-00.png"]})
    assert "zimage-t-00.png" in r.json()["deleted"]
    assert not (out / "zimage-t-00.png").exists()
    assert not (out / "zimage-t-00.json").exists()


def test_delete_traversal_rejected(client):
    assert client.post("/api/delete", json={"name": "../../etc/passwd.png"}).json()["deleted"] == []


def test_favorite_roundtrip_and_guard(client, app_mod):
    (app_mod.OUT / "zimage-f-00.png").write_bytes(b"png")
    assert (
        client.post("/api/favorite", json={"name": "zimage-f-00.png", "on": True}).json()["fav"]
        is True
    )
    assert (
        client.post("/api/favorite", json={"name": "zimage-f-00.png", "on": False}).json()["fav"]
        is False
    )
    assert client.post("/api/favorite", json={"name": "../x.png", "on": True}).status_code == 400


def test_img_and_cancel_and_dup(client):
    assert client.get("/img/nope.png").status_code == 404
    assert client.post("/api/cancel", json={"job_id": "nope"}).status_code == 200
    assert client.post("/api/queue/dup", json={"job_id": "nope"}).status_code == 404
