"""Calibrate a trained model's safety gates on TRAIN/VAL data only (the test split is never touched).

    python ml/calibrate.py --task leaf --arch convnext_tiny      -> models/leaf_convnext_tiny_calibration.json

* temperature: scalar T minimising validation NLL of softmax(logits / T)  (confidence calibration; ECE before/after)
* domain shift: cosine distance between an image embedding and the centroid of its predicted class (centroids from up
  to 100 training images per class); threshold = 99th percentile on validation
* confidence gate: coverage / accuracy of calibrated top-1 probability thresholds on validation
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import MODELS, ROOT, GPUBatches, build  # noqa: E402

PER_CLASS = 100


@torch.no_grad()
def embed(model, head, paths):
    feats, logits = [], []
    hook = head.register_forward_pre_hook(lambda m, i: feats.append(i[0].flatten(1).float().cpu()))
    for x, _ in GPUBatches(paths, np.zeros(len(paths), int), 64, root=ROOT):   # small: shares the GPU with training
        with torch.autocast("cuda", dtype=torch.float16):
            logits.append(model(x).float().cpu())
    hook.remove()
    return torch.nn.functional.normalize(torch.cat(feats), dim=1).numpy(), torch.cat(logits).numpy()


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(abs(correct[m].mean() - conf[m].mean()) * m.mean()
                     for m in ((conf > a) & (conf <= b) for a, b in zip(edges[:-1], edges[1:])) if m.any()))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="leaf")
    ap.add_argument("--arch", default="convnext_tiny")
    a = ap.parse_args()
    ck = torch.load(MODELS / f"{a.task}_{a.arch}.pt", map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = build(a.arch, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model = model.cuda().eval()
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]

    df = pd.read_csv(ROOT / "data" / f"manifest_{a.task}.csv")
    idx = {c: i for i, c in enumerate(classes)}
    tr = df[df.split == "train"].groupby("key", group_keys=False).apply(lambda g: g.sample(min(PER_CLASS, len(g)), random_state=42))
    va = df[df.split == "val"]
    ftr, _ = embed(model, head, tr["path"].values)
    ytr = tr["key"].map(idx).values
    cent = np.stack([ftr[ytr == c].mean(0) for c in range(len(classes))])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)

    fv, zv = embed(model, head, va["path"].values)
    yv = va["key"].map(idx).values
    pred = zv.argmax(1)
    nll = lambda t: float(-np.log(softmax(zv / t)[np.arange(len(yv)), yv] + 1e-12).mean())  # noqa: E731
    ts = np.linspace(0.3, 3.0, 136)
    T = float(ts[np.argmin([nll(t) for t in ts])])
    p0, p1 = softmax(zv).max(1), softmax(zv / T).max(1)
    dist = 1 - (fv * cent[pred]).sum(1)
    gate = {f"{t:.2f}": {"coverage": float((p1 >= t).mean()), "accuracy_answered": float((pred == yv)[p1 >= t].mean()),
                         "errors_flagged": int(((pred != yv) & (p1 < t)).sum()), "errors_total": int((pred != yv).sum())}
            for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
    out = {"task": a.task, "arch": a.arch, "temperature": T, "centroids": cent.round(5).tolist(),
           "ood_threshold": float(np.percentile(dist, 99)), "ood_p95": float(np.percentile(dist, 95)),
           "ood_p50": float(np.median(dist)),
           "val_accuracy": float((pred == yv).mean()), "val_nll": nll(1.0), "val_nll_calibrated": nll(T),
           "ece": ece(p0, pred == yv), "ece_calibrated": ece(p1, pred == yv), "confidence_gate_val": gate}
    (MODELS / f"{a.task}_{a.arch}_calibration.json").write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "centroids"}, indent=1))


if __name__ == "__main__":
    main()
