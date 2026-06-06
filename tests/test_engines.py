"""Engine logic tests that don't load any model."""

import numpy as np
from PIL import Image


def _E():
    import persona_gen.engines as engines

    return engines


def test_build_engines_filters_and_defaults():
    out = _E().build_engines(
        {
            "engines": {
                "a": {"type": "zimage", "model": "/x"},
                "b": {"type": "sdxl", "model": "/y", "label": "B", "steps": 30, "cfg": 7},
                "c": {"type": "unknown", "model": "/z"},
            }
        }
    )
    assert set(out) == {"a", "b"}
    assert out["b"]["label"] == "B" and out["b"]["steps"] == 30 and out["b"]["cfg"] == 7.0
    assert out["a"]["steps"] == 9 and out["a"]["cfg"] == 0.0


def test_exists(tmp_path):
    E = _E()
    assert E._exists(None) is False
    assert E._exists("/nonexistent/x") is False
    p = tmp_path / "f"
    p.write_text("x")
    assert E._exists(str(p)) is True


def test_face_box_square_and_in_bounds():
    kps = np.array([[100, 100], [140, 100], [120, 130], [105, 150], [135, 150]], dtype=np.float32)
    L, T, R, B = _E()._face_box(kps, 512, 768)
    assert (R - L) == (B - T)
    assert 0 <= L and 0 <= T and R <= 512 and B <= 768


def test_feather_mask():
    m = _E()._feather(100, 80)
    assert m.size == (100, 80) and m.mode == "L"


def test_detectors_skip_when_models_absent():
    det = _E()._Detectors({"yoloface": "/nope", "pose_task": "/nope", "hand_task": "/nope"})
    img = Image.new("RGB", (64, 64))
    assert det.face_kps(img) == []
    assert det.foot_boxes(img) == []
    assert det.hand_boxes(img) == []


def test_engine_constructors_do_not_load():
    E = _E()
    z = E.ZImageEngine({"model": "~/x", "max_sequence_length": 512})
    assert z.pipe is None and z.max_seq == 512
    s = E.CuratedSDXLEngine({"model": "~/x", "refiner": "/r"})
    assert s.pony is None and s.i2i is None
