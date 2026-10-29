"""Train + evaluate one classifier on the merged corpus (GPU, mixed precision).

    python ml/train.py --task leaf --arch convnext_tiny --epochs 12
    python ml/train.py --task leaf --arch efficientnet_b0 --epochs 12
    python ml/train.py --task leaf --arch baseline_cnn --epochs 12 --lr 2e-3      (O1 reference, from scratch)
    python ml/train.py --task pest --arch convnext_tiny --epochs 15              (IP102 insect pests)

Reads data/manifest_<task>.csv (ml/build_dataset.py). Writes models/<task>_<arch>.pt and
results/<task>/<arch>/{metrics.json, history.json, test_probs.npz, *.png}.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, top_k_accuracy_score
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import WeightedRandomSampler
from torchvision import models

from gpu_aug import MEAN, SIZE, STD, GPUBatches, eval_tensor  # noqa: F401  (re-exported for backend/model.py)

ROOT = Path(__file__).resolve().parents[1]
MODELS, RESULTS = ROOT / "models", ROOT / "results"
SEED = 42
ARCH_NAMES = {"convnext_tiny": "ConvNeXt-Tiny", "efficientnet_v2_s": "EfficientNetV2-S", "efficientnet_b0": "EfficientNet-B0",
              "mobilenet_v3_large": "MobileNetV3-Large", "vit_b_16": "ViT-B/16", "baseline_cnn": "Baseline CNN"}


# training / evaluation batches come from gpu_aug.GPUBatches (CPU decode threads + batched GPU augmentation);
# eval_tensor is also the web app's preprocessing, so test metrics and live predictions see identical inputs.


class BaselineCNN(nn.Module):
    """O1 reference: small CNN trained from scratch."""
    def __init__(self, n):
        super().__init__()
        def blk(i, o, s=1):
            return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blk(3, 32, 2), nn.MaxPool2d(2), *blk(32, 64), nn.MaxPool2d(2),
                                      *blk(64, 128), nn.MaxPool2d(2), *blk(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.classifier(self.features(x))


def build(arch, n, pretrained=True):
    """ImageNet-pretrained backbone with a new n-way head (all layers trainable)."""
    if arch == "baseline_cnn":
        return BaselineCNN(n)
    m = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    if arch == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, n)
    else:                                                    # torchvision CNNs: the last classifier layer is Linear
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    return m


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            probs.append(torch.softmax(model(x.cuda(non_blocking=True)).float(), 1).cpu())
    return torch.cat(probs).numpy()


def prf(y, pred, labels=None):
    """Accuracy + macro P/R/F1. For a slice, pass labels = the slice's own classes; otherwise classes that only
    appear as wrong predictions inside the slice would enter the macro average with F1 = 0."""
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f)}


def metrics(df, p, classes):
    """df: test rows (key, crop, kind, source, imagery); p: probabilities aligned with classes."""
    y = df["y"].values
    pred = p.argmax(1)
    labels = list(range(len(classes)))
    P, R, F, S = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    rng = np.random.default_rng(SEED)
    boot = np.array([[(pred[ix] == y[ix]).mean(),
                      precision_recall_fscore_support(y[ix], pred[ix], average="macro", zero_division=0)[2]]
                     for ix in (rng.integers(0, len(y), len(y)) for _ in range(500))])
    crop_of = np.array([c.split("___")[0] for c in classes])
    m = {**prf(y, pred),
         "accuracy_ci95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
         "f1_macro_ci95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
         "top3_accuracy": float(top_k_accuracy_score(y, p, k=3, labels=labels)),
         "precision_weighted": float(wp), "recall_weighted": float(wr), "f1_weighted": float(wf),
         "roc_auc_macro_ovr": float(np.mean([roc_auc_score(y == i, p[:, i]) for i in labels if 0 < (y == i).sum() < len(y)])),
         "crop_accuracy": float((crop_of[pred] == crop_of[y]).mean()),
         "per_class": {c: {"precision": float(P[i]), "recall": float(R[i]), "f1": float(F[i]), "support": int(S[i])}
                       for i, c in enumerate(classes)}}
    # slices: every metric restricted to the test images of that slice (macro over the classes present in it)
    for col in ("kind", "imagery", "source", "crop"):
        m[f"by_{col}"] = {str(k): prf(y[ix], pred[ix], sorted(set(y[ix]))) for k, ix in df.groupby(col).indices.items()}
    cm = confusion_matrix(y, pred, labels=labels)
    np.fill_diagonal(cm, 0)
    top = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:25], cm.shape))[0]
    m["top_confusions"] = [{"true": classes[i], "pred": classes[j], "n": int(cm[i, j])} for i, j in top if cm[i, j]]
    return m


def latency_ms(model, trials=50):
    model = model.cpu().float().eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t = []
        for _ in range(trials):
            s = time.perf_counter(); model(x); t.append((time.perf_counter() - s) * 1000)
    return {k: float(np.percentile(t, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))}


def plots(out, hist, m, title):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train"); ax[0].plot(ep, [h["val_loss"] for h in hist], "o-", label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_acc"] for h in hist], "o-", label="val accuracy")
    ax[1].plot(ep, [h["val_f1"] for h in hist], "s--", label="val macro-F1")
    ax[1].plot(ep, [h["val_f1_ema"] for h in hist], "^:", label="val macro-F1 (EMA weights)")
    ax[1].set(title="Validation", xlabel="epoch"); ax[1].legend()
    fig.suptitle(f"{title} - training curves"); fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)

    if len(m["by_crop"]) > 1:
        crops = sorted(m["by_crop"], key=lambda c: m["by_crop"][c]["accuracy"])
        fig, ax = plt.subplots(figsize=(8, max(4, 0.24 * len(crops))))
        ax.barh(crops, [m["by_crop"][c]["accuracy"] for c in crops], color="#2e7d32")
        for i, c in enumerate(crops):
            ax.text(m["by_crop"][c]["accuracy"] + 0.003, i, f"{m['by_crop'][c]['accuracy']:.3f} (n={m['by_crop'][c]['n']})", va="center", fontsize=7)
        ax.set(xlim=(max(0, m["by_crop"][crops[0]]["accuracy"] - 0.1), 1.08), title=f"{title} - test accuracy per crop")
        fig.tight_layout(); fig.savefig(out / "per_crop_accuracy.png", dpi=150); plt.close(fig)

    f1 = np.array([v["f1"] for v in m["per_class"].values()])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(f1, bins=np.linspace(0, 1, 26), color="#2e7d32")
    ax.set(xlabel="per-class F1 (test)", ylabel="classes", title=f"{title} - distribution of per-class F1 (median {np.median(f1):.3f})")
    fig.tight_layout(); fig.savefig(out / "per_class_f1_hist.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="leaf", choices=["leaf", "pest"])
    ap.add_argument("--arch", default="convnext_tiny", choices=list(ARCH_NAMES))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny sample, writes to results/smoke - checks the code path only")
    ap.add_argument("--resume", action="store_true", help="continue from results/<task>/<arch>/resume.pt (saved every epoch)")
    a = ap.parse_args()
    lr = a.lr or {"convnext_tiny": 4e-4, "vit_b_16": 2e-4, "efficientnet_v2_s": 8e-4}.get(a.arch, 1e-3)
    bs = a.bs or {"convnext_tiny": 64, "vit_b_16": 48, "efficientnet_v2_s": 64, "efficientnet_b0": 96}.get(a.arch, 128)
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(2)                  # CPU work happens in the loader threads; avoid oversubscription
    torch.backends.cudnn.benchmark = True     # NB: channels_last is 6x SLOWER for training on this GPU/cuDNN - keep NCHW

    df = pd.read_csv(ROOT / "data" / f"manifest_{a.task}.csv")
    classes = sorted(df["key"].unique())
    df["y"] = df["key"].map({c: i for i, c in enumerate(classes)})
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if a.smoke:
        tr, va, te = (d.sample(n, random_state=0).reset_index(drop=True) for d, n in ((tr, 4000), (va, 1500), (te, 1500)))
    print({"train": len(tr), "val": len(va), "test": len(te), "classes": len(classes)}, flush=True)

    counts = np.bincount(tr["y"], minlength=len(classes)).astype(float)
    weights = 1 / np.sqrt(np.maximum(counts, 1))                    # softened class balance
    sampler = WeightedRandomSampler(torch.tensor(weights[tr["y"].values]), len(tr), replacement=True)
    train_dl = GPUBatches(tr["path"].values, tr["y"].values, bs, train=True, sampler=sampler, root=ROOT)
    val_dl = GPUBatches(va["path"].values, va["y"].values, 256, root=ROOT)

    model = build(a.arch, len(classes), pretrained=a.arch != "baseline_cnn").cuda()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9995), use_buffers=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    decay = 0.05 if a.arch in ("convnext_tiny", "vit_b_16") else 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=a.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    out = RESULTS / ("smoke" if a.smoke else a.task) / a.arch
    out.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    ckpt = (out if a.smoke else MODELS) / f"{a.task}_{a.arch}.pt"
    yv = va["y"].values
    hist, best, t0, start = [], -1, time.time(), 1
    resume_f = out / "resume.pt"            # full training state after the last finished epoch -> safe to kill and resume
    if a.resume and resume_f.exists():
        r = torch.load(resume_f, weights_only=False)
        model.load_state_dict(r["model"]); ema.load_state_dict(r["ema"]); opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"]); scaler.load_state_dict(r["scaler"])
        torch.set_rng_state(r["rng"]); torch.cuda.set_rng_state(r["cuda_rng"]); np.random.set_state(r["np_rng"])
        hist, best, start = r["hist"], r["best"], r["epoch"] + 1
        t0 -= hist[-1]["time_s"]
        print(f"resumed after epoch {r['epoch']} (best val F1 {best:.4f})", flush=True)
    for ep in range(start, a.epochs + 1):
        model.train()
        tl = n = 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update_parameters(model)
            tl += loss.item() * len(y); n += len(y)
            if i % 200 == 0:
                print(f"  ep{ep} step {i}/{len(train_dl)} loss {tl / n:.4f} ({time.time() - t0:.0f}s, {n / max(1e-9, time.time() - t0 - sum(h['val_s'] for h in hist) - sum(h['train_s'] for h in hist)):.0f} img/s)", flush=True)
        t_train = time.time()
        row = {"epoch": ep, "train_loss": tl / n}
        for tag, mdl in (("", model), ("_ema", ema.module)):
            pv = predict(mdl, val_dl)
            row[f"val_f1{tag}"] = float(precision_recall_fscore_support(yv, pv.argmax(1), average="macro", zero_division=0)[2])
            if not tag:
                row["val_loss"] = float(-np.log(pv[np.arange(len(yv)), yv] + 1e-9).mean())
                row["val_acc"] = float((pv.argmax(1) == yv).mean())
            if row[f"val_f1{tag}"] > best:                 # keep whichever weights (raw or EMA) validate best
                best = row[f"val_f1{tag}"]
                torch.save({"task": a.task, "arch": a.arch, "classes": classes, "size": SIZE, "mean": MEAN, "std": STD,
                            "state_dict": mdl.state_dict(), "epoch": ep, "ema": bool(tag), "val_f1": best}, ckpt)
        row.update(train_s=t_train - t0 - sum(h["val_s"] + h["train_s"] for h in hist), time_s=time.time() - t0)
        row["val_s"] = time.time() - t_train
        hist.append(row)
        print(row, flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                    "np_rng": np.random.get_state(), "hist": hist, "best": best, "epoch": ep}, out / "resume.tmp")
        (out / "resume.tmp").replace(resume_f)   # atomic: a kill during the save never leaves a corrupt resume file

    resume_f.unlink(missing_ok=True)             # training finished; the checkpoint below is the result
    ck = torch.load(ckpt, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    pt = predict(model, GPUBatches(te["path"].values, te["y"].values, 256, root=ROOT))
    np.savez_compressed(out / "test_probs.npz", y=te["y"].values, p=pt.astype(np.float16), paths=te["path"].values)
    m = metrics(te, pt.astype(np.float64), classes)
    t = time.perf_counter(); predict(model, val_dl); gpu_ips = len(va) / (time.perf_counter() - t)
    m.update({"task": a.task, "arch": a.arch, "name": ARCH_NAMES[a.arch], "best_epoch": ck["epoch"], "ema": ck["ema"],
              "epochs": a.epochs, "train_time_min": hist[-1]["time_s"] / 60,
              "params_m": sum(p.numel() for p in model.parameters()) / 1e6, "size_mb": ckpt.stat().st_size / 1e6,
              "gpu_images_per_s": gpu_ips, "latency_ms_cpu": latency_ms(model), "classes": len(classes),
              "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)}})
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    plots(out, hist, m, f"{ARCH_NAMES[a.arch]} ({a.task})")
    print(f"TEST {a.task}/{a.arch}: acc {m['accuracy']:.4f} P {m['precision_macro']:.4f} R {m['recall_macro']:.4f} "
          f"F1 {m['f1_macro']:.4f} top3 {m['top3_accuracy']:.4f} AUC {m['roc_auc_macro_ovr']:.4f}", flush=True)
    print({k: round(v["accuracy"], 4) for k, v in m["by_imagery"].items()}, {k: round(v["f1_macro"], 4) for k, v in m["by_kind"].items()}, flush=True)


if __name__ == "__main__":
    main()
