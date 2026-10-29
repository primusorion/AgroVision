"""Recompute metrics.json from a run's saved test predictions (no retraining), keeping the run's other fields.

    python ml/rescore.py leaf efficientnet_b0
"""
import json
import sys

import numpy as np
import pandas as pd

from train import RESULTS, ROOT, metrics


def main(task, arch):
    out = RESULTS / task / arch
    z = np.load(out / "test_probs.npz", allow_pickle=True)
    df = pd.read_csv(ROOT / "data" / f"manifest_{task}.csv")
    classes = sorted(df["key"].unique())
    te = df.set_index("path").loc[z["paths"]].reset_index()
    te["y"] = te["key"].map({c: i for i, c in enumerate(classes)})
    assert (te["y"].values == z["y"]).all(), "manifest / predictions mismatch"
    old = json.loads((out / "metrics.json").read_text())
    new = metrics(te, z["p"].astype(np.float64), classes)
    old.update(new)
    (out / "metrics.json").write_text(json.dumps(old, indent=1))
    print(task, arch, {k: round(v["f1_macro"], 4) for k, v in new["by_kind"].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
