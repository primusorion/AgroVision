"""Look at a downloaded Project-AgML parquet source before trusting it.

    python ml/inspect_source.py <repo> [samples per class]   -> prints label counts / image sizes, writes a contact sheet
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def label_names(schema):
    """ClassLabel names stored by Hugging Face in the parquet schema metadata."""
    feats = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
    return feats["label"].get("names") if "label" in feats else None


def rows(repo, columns=None):
    """Yield dict rows of every parquet file of a source (image as bytes)."""
    for f in sorted((ROOT / "data" / "sources" / "agml" / repo).glob("*.parquet")):
        pf = pq.ParquetFile(f)
        names = label_names(pf.schema_arrow)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=columns).to_pylist():
                if names is not None and isinstance(r.get("label"), int):
                    r["label"] = names[r["label"]]
                yield r


def main():
    repo, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
    extra = [c for c in pq.ParquetFile(next((ROOT / "data" / "sources" / "agml" / repo).glob("*.parquet"))).schema_arrow.names
             if c not in ("image", "label")]
    print("columns besides image/label:", extra)
    counts, sizes, picks = Counter(), Counter(), {}
    rng = np.random.default_rng(0)
    for r in rows(repo):
        key = "/".join(str(r[c]) for c in extra if c in ("crop", "crop_type")) + ("/" if any(c in ("crop", "crop_type") for c in extra) else "") + str(r["label"])
        counts[key] += 1
        seen = counts[key]
        if seen <= k or rng.random() < k / seen:               # reservoir sample k images per class
            im = Image.open(io.BytesIO(r["image"]["bytes"]))
            sizes[im.size] += 1
            slot = picks.setdefault(key, [])
            thumb = np.asarray(im.convert("RGB").resize((160, 160)))
            if len(slot) < k:
                slot.append(thumb)
            else:
                slot[rng.integers(k)] = thumb
        for c in extra:
            if c not in ("crop", "crop_type"):
                counts[f"[{c}={r[c]}]"] += 0 if isinstance(r[c], (dict, list)) else 1
    for key, n in sorted(counts.items()):
        print(f"{n:6}  {key}")
    print("image sizes (sampled):", sizes.most_common(5))
    sheet = np.vstack([np.hstack(v + [np.zeros((160, 160, 3), np.uint8)] * (k - len(v))) for _, v in sorted(picks.items())])
    out = ROOT / "data" / "sources" / "sheets"
    out.mkdir(exist_ok=True)
    Image.fromarray(sheet).save(out / f"{repo}.jpg", quality=80)
    print("rows top-to-bottom:", sorted(picks))
    print("sheet:", out / f"{repo}.jpg")


if __name__ == "__main__":
    main()
