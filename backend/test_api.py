"""End-to-end API tests (fixtures = frontend/samples, oracle = the class key stored in samples.json).

    python backend/test_api.py        (or: python -m pytest backend/test_api.py)
"""
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import CLASS_INFO, FRONTEND, app  # noqa: E402

client = TestClient(app)
SAMPLES = json.loads((FRONTEND / "samples" / "samples.json").read_text(encoding="utf-8"))


def post(data, name="x.jpg", force=False):
    return client.post("/api/analyze" + ("?force=true" if force else ""), files={"file": (name, data, "image/jpeg")})


def png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def sample(s):
    return post((FRONTEND / s["url"].lstrip("/")).read_bytes(), "s.jpg").json()


def test_health_crops_metrics():
    h = client.get("/api/health").json()
    assert h["model_loaded"] and h["classes"] > 100 and h["crops"] >= 30
    crops = client.get("/api/crops").json()
    assert len(crops) == h["crops"] and all(c["conditions"] for c in crops)
    m = client.get("/api/metrics").json()
    assert m["leaf_models"] and m["leaf"]["per_class"] and m["dataset"]["leaf_images"] > 100_000


def test_every_leaf_sample_matches_oracle():
    for s in (s for s in SAMPLES if s["kind"] != "insect"):
        r = sample(s)
        assert r["status"] == "ok" and r["mode"] == "leaf", (s["url"], r["warnings"])
        d, info = r["diagnosis"], CLASS_INFO[s["key"]]
        assert d["key"] == s["key"], (s["url"], d["key"])
        assert r["crop"]["name"] == info["crop"] and r["recommendation"]["fertilizers"] and r["explain"]["segmentation"]
        assert (r["health"]["score"] >= 80) == info["healthy"], (s["url"], r["health"]["score"])
        assert r["disease"]["detected"] == (info["kind"] == "disease") or not r["disease"]["primary"]
        if info["kind"] == "deficiency":                    # learned nutrient finding
            assert r["nutrient"]["status"] == "deficiency_detected" and r["nutrient"]["primary"]["source"] == "model", s["url"]
        if info["kind"] == "pest":                          # learned pest-damage finding
            assert any(f["source"] == "classifier" for f in r["pest"]["findings"]), s["url"]


def test_insect_samples_use_pest_model():
    for s in (s for s in SAMPLES if s["kind"] == "insect"):
        r = sample(s)
        assert r["status"] == "ok" and r["mode"] == "insect" and r["diagnosis"] is None, (s["url"], r.get("mode"))
        assert s["key"] in [c["key"] for c in r["insect"]["candidates"]] and r["insect"]["species"]["control"]
        assert r["insect"]["safety"]          # chemical control advice must always carry the safety note


def test_rejections():
    assert post(b"not an image", "a.txt").status_code == 400
    black = post(png(np.zeros((300, 300, 3), np.uint8))).json()
    assert black["status"] == "rejected" and black["diagnosis"] is None
    forced = post(png(np.zeros((300, 300, 3), np.uint8)), force=True).json()
    assert forced["status"] == "ok" and forced["warnings"]
    portrait = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data", "grace_hopper.jpg")
    r = post(open(portrait, "rb").read()).json()
    assert r["status"] == "rejected" and any(c["name"] == "Plant check" for c in r["quality"]["checks"]), r["status"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
