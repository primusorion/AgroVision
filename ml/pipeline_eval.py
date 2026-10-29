"""End-to-end evaluation of the deployed web pipeline through its real API (in-process, no network).

    python ml/pipeline_eval.py [n_leaf=4000]      -> results/pipeline_eval.json

A class-stratified sample of TEST images goes through POST /api/analyze, exactly as a user's upload would:
routing (leaf / insect / not a plant), gate rejections, diagnosis, learned nutrient + pest findings, health score,
latency. Also: insect test photos, "not a plant" test photos and non-leaf photos found on this machine.
"""
import glob
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from fastapi.testclient import TestClient  # noqa: E402

from app import CLASS_INFO, INSECT, NOT_PLANT, app  # noqa: E402

client = TestClient(app)


def post(path):
    data = open(path, "rb").read()
    t = time.perf_counter()
    r = client.post("/api/analyze", files={"file": (os.path.basename(str(path)), io.BytesIO(data), "image/jpeg")})
    return (r.json() if r.status_code == 200 else {"status": f"http_{r.status_code}"}), (time.perf_counter() - t) * 1000


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    leaf = pd.read_csv(ROOT / "data" / "manifest_leaf.csv")
    te = leaf[leaf.split == "test"]
    real = te[te.kind != "other"]
    per = max(4, n // real.key.nunique())
    sample = real.groupby("key", group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=0))
    rows, lat, timing = [], [], defaultdict(list)
    for i, r in enumerate(sample.itertuples()):
        res, ms = post(ROOT / r.path)
        lat.append(ms)
        row = {"key": r.key, "kind": r.kind, "imagery": r.imagery, "status": res["status"], "mode": res.get("mode")}
        if res["status"] == "ok" and res["mode"] == "leaf":
            d = res["diagnosis"]
            for k, v in res["timing_ms"].items():
                timing[k].append(v)
            row.update(correct=d["key"] == r.key, crop_ok=res["crop"]["name"] == r.crop, low_conf=d["low_confidence"],
                       health=res["health"]["score"], deficiency=res["nutrient"]["status"] != "normal",
                       def_source=(res["nutrient"]["primary"] or {}).get("source"), pest=res["pest"]["detected"],
                       pest_classifier=any(f["source"] == "classifier" for f in res["pest"]["findings"]),
                       urgency=res["recommendation"]["urgency"],
                       warned=bool(res["warnings"]))
        elif res["status"] == "rejected":
            row["reason"] = "; ".join(c["name"] for c in res["quality"]["checks"] if not c["passed"] and c["severity"] == "hard")
        rows.append(row)
        if i % 500 == 0:
            print(i, "/", len(sample), flush=True)
    df = pd.DataFrame(rows)
    ok = df[(df.status == "ok") & (df["mode"] == "leaf")]   # rejected / insect-routed rows leave NaN -> object dtype; ~ needs bool
    ok = ok.astype({c: bool for c in ("correct", "crop_ok", "low_conf", "deficiency", "pest", "pest_classifier", "warned")})
    out = {"n_leaf": len(df), "status": df.status.value_counts().to_dict(), "mode": df["mode"].value_counts(dropna=False).to_dict(),
           "rejection_reasons": Counter(df.reason.dropna()).most_common(8) if "reason" in df else [],
           "rejected_by_imagery": df.assign(rej=df.status == "rejected").groupby("imagery").rej.mean().round(4).to_dict(),
           "accuracy_answered": float(ok.correct.mean()), "crop_accuracy_answered": float(ok.crop_ok.mean()),
           "accuracy_all": float((df.get("correct") == True).mean()),  # noqa: E712  (rejections count as not answered)
           "accuracy_confident": float(ok[~ok.low_conf].correct.mean()), "low_confidence_rate": float(ok.low_conf.mean()),
           "latency_ms": {k: float(np.percentile(lat, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
           "stage_ms_p50": {k: float(np.median(v)) for k, v in timing.items()}, "by_kind": {}}
    for kind, g in ok.groupby("kind"):
        out["by_kind"][kind] = {"n": len(g), "accuracy": float(g.correct.mean()), "health_mean": float(g.health.mean()),
                                "health_p10_p90": [float(g.health.quantile(0.1)), float(g.health.quantile(0.9))],
                                "deficiency_flag_rate": float(g.deficiency.mean()),
                                "deficiency_from_model_rate": float((g.def_source == "model").mean()),
                                "pest_flag_rate": float(g.pest.mean()), "pest_from_classifier_rate": float(g.pest_classifier.mean()),
                                "urgency": g.urgency.value_counts(normalize=True).round(3).to_dict()}

    # insect photos (pest model, official IP102 test split) and negatives, through the same endpoint
    pest = pd.read_csv(ROOT / "data" / "manifest_pest.csv")
    ins = pest[pest.split == "test"].sample(600, random_state=0)
    routed = top1 = top3 = 0
    for r in ins.itertuples():
        res, _ = post(ROOT / r.path)
        if res["status"] == "ok" and res["mode"] == "insect":
            routed += 1
            top1 += res["insect"]["species"]["key"] == r.key
            top3 += r.key in [c["key"] for c in res["insect"]["candidates"]]
    out["insect"] = {"n": len(ins), "routed_to_pest_model": routed / len(ins), "species_top1_of_routed": top1 / max(1, routed),
                     "species_top3_of_routed": top3 / max(1, routed), "species_top1_end_to_end": top1 / len(ins)}
    neg = te[te.key == NOT_PLANT]
    neg = neg.sample(min(400, len(neg)), random_state=0)
    out["not_a_plant_test"] = {"n": len(neg), "rejected": float(np.mean([post(ROOT / p)[0]["status"] == "rejected" for p in neg.path]))}

    mp = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "sample_data")
    files = sorted(glob.glob(r"C:\Windows\Web\Wallpaper\*\*.jpg") + glob.glob(r"C:\Windows\Web\Screen\*.jpg")
                   + glob.glob(r"C:\Windows\Web\4K\Wallpaper\*\*.jpg") + glob.glob(mp + "/*.jpg") + glob.glob(mp + "/*.png"))
    wild = []
    for f in files:                                           # never-seen non-leaf photos (wallpapers, a portrait)
        res, _ = post(f)
        wild.append({"file": os.path.basename(f), "status": res["status"], "mode": res.get("mode"),
                     "prediction": (res.get("diagnosis") or {}).get("key"), "warned": bool(res.get("warnings"))})
    out["non_leaf_photos"] = {"n": len(wild), "rejected": sum(w["status"] == "rejected" for w in wild),
                              "answered_with_warning": sum(w["status"] == "ok" and w["warned"] for w in wild),
                              "answered_silently": sum(w["status"] == "ok" and not w["warned"] for w in wild), "rows": wild}
    (ROOT / "results" / "pipeline_eval.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "non_leaf_photos"}, indent=1, default=float))
    print({k: v for k, v in out["non_leaf_photos"].items() if k != "rows"})


if __name__ == "__main__":
    main()
