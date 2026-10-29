"""Fill the web app's sample gallery from the held-out TEST split (never seen in training).

    python ml/make_samples.py      -> frontend/samples/*.jpg + samples.json

One correctly-classified test image per crop (rotating through disease / pest / deficiency / healthy so every kind
of finding is represented), plus two insect photos for the pest model. Picking correctly-classified images keeps
the live demo predictable; the honest error rates are on the Model Performance page.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import kb  # noqa: E402
from taxonomy import SOURCES  # noqa: E402

OUT = ROOT / "frontend" / "samples"
WANT = ["pest", "disease", "deficiency", "healthy", "disease", "abiotic"]   # rotation over crops (rarer kinds first)
EXTRA = ["Coffee___Nitrogen_deficiency", "Citrus___Iron_deficiency", "Banana___Potassium_deficiency",
         "Groundnut___Nutrient_deficiency", "Rice___Blast", "Tomato___Late_blight"]


def licence(src):
    """Licence of a source's images; '' when the source states none - those images are never shipped in the repo."""
    lic = SOURCES.get(src, {}).get("license", "CC BY 4.0 (IP102 mirror)" if src == "ip102" else "")
    return "" if "unstated" in lic else lic


def correct(task, arch):
    """Test rows the deployed model gets right, with its confidence."""
    z = np.load(ROOT / "results" / task / arch / "test_probs.npz", allow_pickle=True)
    df = pd.read_csv(ROOT / "data" / f"manifest_{task}.csv").set_index("path").loc[z["paths"]].reset_index()
    p = z["p"].astype(np.float32)
    df["conf"] = p.max(1)
    return df[p.argmax(1) == z["y"]]


def main():
    deployed = json.loads((ROOT / "models" / "deployed.json").read_text())
    classes, _, _, pests = kb.load()
    for f in OUT.glob("*"):
        f.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    leaf = correct("leaf", deployed["leaf"])
    leaf = leaf[(leaf.kind != "other") & (leaf.source.map(licence) != "")]   # only images we may redistribute
    items, rng = [], np.random.default_rng(7)
    for i, (crop, g) in enumerate(sorted(leaf.groupby("crop"))):
        kinds = [k for k in WANT[i % len(WANT):] + WANT if k in set(g.kind)]
        g = g[g.kind == kinds[0]]
        g = g[g.conf >= g.conf.quantile(0.5)]                 # a typical confident example, not the single best
        r = g.iloc[rng.integers(len(g))]
        name = f"{len(items):02d}_{r.key}.jpg"
        shutil.copy(ROOT / r.path, OUT / name)
        c = classes[r.key]
        items.append({"url": f"/samples/{name}", "key": r.key, "kind": c["kind"], "label": f"{crop} · {c['name']}",
                      "imagery": r.imagery, "source": r.source})
    for key in EXTRA:                                         # showcase findings the rotation may not reach
        g = leaf[(leaf.key == key)]
        if len(g) and key not in {i["key"] for i in items}:
            r = g.sort_values("conf").iloc[len(g) // 2]
            name = f"{len(items):02d}_{r.key}.jpg"
            shutil.copy(ROOT / r.path, OUT / name)
            c = classes[r.key]
            items.append({"url": f"/samples/{name}", "key": r.key, "kind": c["kind"], "label": f"{c['crop']} · {c['name']}",
                          "imagery": r.imagery, "source": r.source})
    if "pest" in deployed:
        pest = correct("pest", deployed["pest"])
        pest = pest[pest.conf >= 0.8]
        by = {"".join(ch for ch in k.lower() if ch.isalnum()): v for k, v in pests.items()}
        for r in pest.sample(2, random_state=7).itertuples():
            name = f"{len(items):02d}_insect.jpg"
            shutil.copy(ROOT / r.path, OUT / name)
            e = by.get("".join(ch for ch in r.key.split("___")[-1].lower() if ch.isalnum()), {})
            items.append({"url": f"/samples/{name}", "key": r.key, "kind": "insect", "imagery": "field", "source": r.source,
                          "label": f"Insect · {e.get('name', r.key.split('___')[-1].replace('_', ' '))}"})
    (OUT / "samples.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    credits = ["# Sample image credits", "", "Held-out test images from public datasets, resized to a 256 px short side. "
               "Each image keeps the licence of its source dataset.", "", "| File | Source dataset | Licence |", "|---|---|---|"]
    credits += [f"| {i['url'].rsplit('/', 1)[-1]} | {i['source']} | {licence(i['source'])} |" for i in items]
    (OUT / "ATTRIBUTION.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(len(items), "samples:", pd.Series([i["kind"] for i in items]).value_counts().to_dict())


if __name__ == "__main__":
    main()
