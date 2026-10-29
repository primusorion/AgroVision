"""AgroVision AI - FastAPI backend (v2: 39 crops, learned disease / pest / nutrient classes, insect identification).

    python backend/app.py            (from the project root)  ->  http://127.0.0.1:8000

Pipeline per image: decode -> centre square -> photo-quality gate -> leaf model (also routes "not a plant" and
"insect") -> [insect: pest model, 102 species] | [leaf: Grad-CAM, leaf segmentation + measurements, advisory engine:
disease, nutrient deficiency, pests, health score, fertilizer + treatment].
"""
import base64
import io
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import advisory  # noqa: E402
import vision  # noqa: E402
from advisory import CLASS_INFO, CROP_INFO, INSECT, NOT_PLANT, ROUTER  # noqa: E402
from model import Classifier, center_square  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS, RESULTS, FRONTEND = ROOT / "models", ROOT / "results", ROOT / "frontend"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 50_000_000
FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF")  # decoders allowed on uploads (no EPS/Ghostscript, PSD, ...)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # PIL only warns up to 2x this; analyze() rejects anything above 1x
DISCLAIMER = ("Decision-support tool trained on public leaf-image datasets (lab and field photos). It can be "
              "wrong, especially on photos unlike its training data. Confirm with a local agriculture officer / KVK and "
              "a soil test before applying chemicals; always follow the product label.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agrovision")
STATS = {"requests": 0, "ok": 0, "insect": 0, "rejected_quality": 0, "rejected_not_plant": 0, "low_confidence": 0,
         "domain_shift": 0, "last_total_ms": None, "started": time.time()}  # telemetry for /api/health


def load(task):
    """Deployed model of a task (models/deployed.json names the architecture; set by ml/report.py)."""
    cfg = json.loads((MODELS / "deployed.json").read_text()) if (MODELS / "deployed.json").exists() else {}
    f = MODELS / f"{task}_{cfg.get(task, 'convnext_tiny')}.pt"
    if not f.exists():
        log.warning("model %s missing - run ml/train.py --task %s", f.name, task)
        return None
    return Classifier(f)


app = FastAPI(title="AgroVision AI", version="2.0", description="Crop health analysis API (KL University AIML-25)")
leaf_clf, pest_clf = load("leaf"), load("pest")


@app.middleware("http")
async def revalidate(request, call_next):
    """Browsers must revalidate (cheap ETag check) so a retrained model's samples/figures or an updated UI show up."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response
LOW_CONF = 0.80   # on calibrated probabilities; validation trade-off is stored in the model's calibration file
# crops the deployed model can actually recognise (the knowledge base may describe more than the data covers)
CROPS = sorted({CLASS_INFO[k]["crop"] for k in (leaf_clf.classes if leaf_clf else CLASS_INFO) if k in CLASS_INFO})


def ms(t):
    return round((time.perf_counter() - t) * 1000, 1)


def data_url(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def label(key):
    i = CLASS_INFO[key]
    return f"{i['crop']} · {i['name']}"


def brief(key, p):
    i = CLASS_INFO.get(key)
    if i is None:                                           # router classes
        return {"key": key, "crop": "-", "name": key.split("___")[1].replace("_", " "), "kind": "other", "confidence": float(p)}
    return {"key": key, "crop": i["crop"], "name": i["name"], "kind": i["kind"], "confidence": float(p)}


def reject(res, t0, reason, msg=None, tip=None):
    q = res["quality"]
    if msg:                                                 # rejection decided after the quality gate (not a plant)
        q["passed"], q["score"] = False, min(q["score"], 40)
        q["checks"].append({"name": "Plant check", "value": 0, "passed": False, "severity": "hard",
                            "threshold": "photo must show a plant leaf (or an insect pest)", "message": msg})
        q["issues"].insert(0, msg)
        q["tips"].insert(0, tip)
    res["status"] = "rejected"
    res["warnings"].extend(i for i in q["issues"] if i not in res["warnings"])
    res["timing_ms"]["total"] = STATS["last_total_ms"] = ms(t0)
    STATS[reason] += 1
    log.info("req %s %s: %s", res["request_id"], reason, q["issues"])
    return res


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": leaf_clf is not None, "model": leaf_clf.name if leaf_clf else None,
            "classes": len(leaf_clf.classes) if leaf_clf else 0, "crops": len(CROPS),
            "pest_model": pest_clf.name if pest_clf else None, "pest_species": len(pest_clf.classes) if pest_clf else 0,
            "device": "cpu", "gates": {"low_confidence": LOW_CONF,
                                       "domain_shift": leaf_clf.cal["ood_threshold"] if leaf_clf and leaf_clf.cal else None},
            "stats": {**STATS, "uptime_s": round(time.time() - STATS["started"])}}


@app.get("/api/crops")
def crops():
    """Supported crops with their conditions (what the leaf model can recognise)."""
    known = set(leaf_clf.classes) if leaf_clf else set(CLASS_INFO)
    out = []
    for name, c in sorted(CROP_INFO.items()):
        conds = [{"key": k, "name": v["name"], "kind": v["kind"]} for k, v in CLASS_INFO.items() if v["crop"] == name and k in known]
        if conds:
            out.append({"crop": name, "scientific_name": c["scientific_name"], "family": c["family"], "conditions": conds})
    return out


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), force: bool = False):
    if leaf_clf is None:
        raise HTTPException(503, "Model not trained yet - run ml/train.py")
    t0 = time.perf_counter()
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB")
    try:
        im = Image.open(io.BytesIO(data), formats=FORMATS)  # lazy: reads the header only
        if im.width * im.height > MAX_PIXELS:
            raise Image.DecompressionBombError
        img = center_square(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))
    except Image.DecompressionBombError:
        raise HTTPException(413, f"Image is too large (over {MAX_PIXELS // 1_000_000} megapixels)")
    except Exception:  # PIL raises many types (UnidentifiedImageError, OSError, ...)
        raise HTTPException(400, "Not a readable image - upload a JPG, PNG or WebP photo of a leaf")

    STATS["requests"] += 1
    T, warnings = {}, []
    t = time.perf_counter(); q = vision.quality_check(img); T["quality"] = ms(t)
    res = {"status": "ok", "mode": "leaf", "request_id": uuid.uuid4().hex[:8], "warnings": warnings, "quality": q,
           "diagnosis": None, "crop": None, "disease": None, "nutrient": None, "pest": None, "health": None,
           "recommendation": None, "insect": None, "explain": None, "timing_ms": T,
           "model": {"name": leaf_clf.name, "arch": leaf_clf.arch, "classes": len(leaf_clf.classes), "crops": len(CROPS)},
           "disclaimer": DISCLAIMER}
    t = time.perf_counter(); probs, cam, shift = leaf_clf.predict(img); T["classify"] = ms(t)
    class_probs = {k: float(p) for k, p in zip(leaf_clf.classes, probs)}
    order = np.argsort(probs)[::-1]
    key, conf = leaf_clf.classes[order[0]], float(probs[order[0]])

    # ---- route: not a plant / insect photo / leaf
    if key == NOT_PLANT and not force:
        return reject(res, t0, "rejected_not_plant", "This does not look like a plant leaf.",
                      "Photograph a single leaf (or the insect pest), centred and filling most of the frame.")
    hard = [c for c in q["checks"] if not c["passed"] and c["severity"] == "hard"]
    if key == INSECT and pest_clf is not None:
        hard = [c for c in hard if c["name"] != "Leaf coverage"]    # an insect close-up need not show a leaf
        if hard and not force:
            return reject(res, t0, "rejected_quality")
        t = time.perf_counter(); pp, pcam, shift = pest_clf.predict(img); T["pest_model"] = ms(t)
        top = advisory.insect_report({k: float(p) for k, p in zip(pest_clf.classes, pp)})
        if top[0]["confidence"] < 0.5:
            warnings.append(f"Low confidence ({top[0]['confidence']:.0%}) in the species - compare the top candidates; "
                            "many pests look alike at this resolution.")
        # IP102 photos are often wide outdoor scenes, so landscapes can get routed here: warn at the stricter 95th
        # percentile (about 5% of real insect photos get this warning; the leaf path uses the 99th)
        cal = pest_clf.cal or {}
        if shift is not None and shift > cal.get("ood_p95", cal.get("ood_threshold", 9)):
            STATS["domain_shift"] += 1
            warnings.append("This photo looks unlike the insect photos the model was trained on. If there is no insect "
                            "in it, the species below is meaningless - photograph the insect up close.")
        res.update({"mode": "insect", "insect": {"router_confidence": conf, "species": top[0], "candidates": top,
                                                 "model": {"name": pest_clf.name, "species": len(pest_clf.classes)},
                                                 "safety": advisory.SAFETY},
                    "explain": {"gradcam": data_url(pcam) if pcam is not None else None, "segmentation": None}})
        T["total"] = STATS["last_total_ms"] = ms(t0)
        STATS["ok"] += 1; STATS["insect"] += 1
        log.info("req %s insect %s conf=%.3f %.0fms", res["request_id"], top[0]["key"], top[0]["confidence"], T["total"])
        return res
    if hard and not force:
        return reject(res, t0, "rejected_quality")
    if hard:
        warnings.append("Photo failed the quality gate; analysis was forced and may be unreliable.")
    warnings.extend(q["issues"])
    routed = key in ROUTER
    if routed:      # forced past "not a plant", or an insect photo without the pest model: diagnose the best leaf class
        warnings.append("This looks like an insect pest, but the insect-identification model is not installed, so the "
                        "leaf in the photo was analysed instead. Identify the insect by eye or with your local KVK."
                        if key == INSECT else "The model does not see a plant leaf here. Analysis was forced; the "
                        "result is not meaningful for other objects.")
        leaf_mass = max(sum(p for k, p in class_probs.items() if k not in ROUTER), 1e-12)
        class_probs = {k: p / leaf_mass for k, p in class_probs.items() if k not in ROUTER}   # P(class | it is a leaf)
        key = max(class_probs, key=class_probs.get)
        conf = class_probs[key]

    t = time.perf_counter(); leaf = vision.analyze_leaf(img); T["analysis"] = ms(t)
    info = CLASS_INFO[key]
    crop_conf = {c: sum(p for k, p in class_probs.items() if k not in ROUTER and CLASS_INFO[k]["crop"] == c) for c in CROP_INFO}
    ood = shift is not None and shift > leaf_clf.cal["ood_threshold"]
    if ood:
        STATS["domain_shift"] += 1
        warnings.append(f"Domain shift detected: this photo looks different from the training images (feature distance "
                        f"{shift:.2f} > {leaf_clf.cal['ood_threshold']:.2f}). Running in degraded mode - treat the diagnosis "
                        "as indicative and confirm in the field.")
    if conf < LOW_CONF:
        STATS["low_confidence"] += 1
        warnings.append(f"Low model confidence ({conf:.0%}). The crop may not be one of the {len(CROPS)} supported, or "
                        "the photo differs from training conditions - treat results as indicative and consider retaking.")

    t = time.perf_counter()
    nutrient = advisory.assess_nutrients(leaf, class_probs, key, conf)
    pest = advisory.assess_pests(leaf, class_probs, key, conf)
    hs = advisory.health_score(leaf, class_probs, key, conf, nutrient, pest)
    uncertain = conf < LOW_CONF or ood or routed
    rec = advisory.recommend(info["crop"], key, nutrient, pest, hs, conf, uncertain=uncertain)
    T["advisory"] = ms(t)

    # disease view: the primary finding if it is a disease, else the strongest disease class if it is plausible
    dk, dp = (key, conf) if info["kind"] == "disease" else advisory.top_of_kind(class_probs, "disease")
    d = CLASS_INFO[dk] if dk and (info["kind"] == "disease" or dp >= 0.20) else None
    disease = {"detected": d is not None, "primary": info["kind"] == "disease", "affected_area_pct": round(float(leaf["lesion_pct"]), 2)}
    if d:
        disease.update(key=dk, name=d["name"], confidence=dp, pathogen=d["cause"], type=d["type"],
                       description=d["description"], symptoms=d["symptoms"], spread=d["spread"])
    else:
        disease.update(name="No infectious disease detected", confidence=advisory.kind_mass(class_probs)["disease"],
                       description=("The leaf is classified as healthy." if info["healthy"] else
                                    f"The main finding is {info['name']} ({info['kind']}), not an infectious disease."))
    ci = CROP_INFO[info["crop"]]
    res.update({
        "diagnosis": {**brief(key, conf), "healthy": info["healthy"], "cause": info["cause"], "type": info["type"],
                      "description": info["description"], "symptoms": info["symptoms"], "spread": info["spread"],
                      "low_confidence": uncertain, "domain_shift": shift,
                      "top_k": [brief(leaf_clf.classes[i], probs[i]) for i in order[:3]],
                      "kind_probabilities": {k: round(v, 4) for k, v in advisory.kind_mass(class_probs).items()}},
        "crop": {"name": info["crop"], "scientific_name": ci["scientific_name"], "family": ci["family"],
                 "confidence": crop_conf[info["crop"]], "description": ci["description"],
                 "common_varieties": ci["common_varieties"],
                 "alternatives": [{"name": c, "confidence": p} for c, p in sorted(crop_conf.items(), key=lambda x: -x[1])[:4]
                                  if c != info["crop"] and p >= 0.005]},
        "disease": disease, "nutrient": nutrient, "pest": pest, "health": hs, "recommendation": rec,
        "explain": {"gradcam": data_url(cam) if cam is not None else None, "segmentation": data_url(leaf["overlay"])},
    })
    T["total"] = STATS["last_total_ms"] = ms(t0)
    STATS["ok"] += 1
    log.info("req %s %s conf=%.3f shift=%s health=%s nutrient=%s pests=%s %.0fms", res["request_id"], key, conf,
             shift and round(shift, 3), hs["score"], nutrient["status"], pest["detected"], T["total"])
    return res


@app.get("/api/samples")
def samples():
    f = FRONTEND / "samples" / "samples.json"                # written by ml/make_samples.py
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


@app.get("/api/metrics")
def metrics():
    """Everything the Model Performance page needs, read from results/ on each request."""
    def read(p):
        return json.loads((RESULTS / p).read_text(encoding="utf-8")) if (RESULTS / p).exists() else None

    def fig(p):
        return f"/results/{p}" if (RESULTS / p).exists() else None

    deployed = json.loads((MODELS / "deployed.json").read_text()) if (MODELS / "deployed.json").exists() else {}
    slim = lambda m: {k: v for k, v in m.items() if k not in ("per_class", "top_confusions", "by_source", "by_crop")}  # noqa: E731
    out = {"deployed": deployed, "leaf_models": [], "pest_models": [], "report": read("report.json") or {},
           "calibration": {}, "dataset": read("dataset.json") or {}}
    for task in ("leaf", "pest"):
        for d in sorted((RESULTS / task).glob("*/metrics.json")) if (RESULTS / task).exists() else []:
            out[f"{task}_models"].append(slim(json.loads(d.read_text())))
        arch = deployed.get(task)
        m = read(f"{task}/{arch}/metrics.json") if arch else None
        if m:
            out[task] = {"by_crop": m.get("by_crop"), "by_source": m.get("by_source"), "top_confusions": m["top_confusions"],
                         "per_class": [{"key": k, "label": label(k) if k in CLASS_INFO else k.split("___")[-1].replace("_", " "),
                                        "kind": CLASS_INFO[k]["kind"] if k in CLASS_INFO else "other", **v}
                                       for k, v in m["per_class"].items()],
                         "figures": {n: fig(f"{task}/{arch}/{n}.png") for n in ("training_curves", "per_crop_accuracy", "per_class_f1_hist")}}
            cal = MODELS / f"{task}_{arch}_calibration.json"
            if cal.exists():
                out["calibration"][task] = {k: v for k, v in json.loads(cal.read_text()).items() if k != "centroids"}
    out["figures"] = {n: fig(f"{n}.png") for n in ("model_comparison", "robustness", "lab_vs_field", "selective_prediction",
                                                   "dataset_crops", "by_kind")}
    return out


RESULTS.mkdir(exist_ok=True)
(FRONTEND / "samples").mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")
app.mount("/samples", StaticFiles(directory=FRONTEND / "samples"), name="samples")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
