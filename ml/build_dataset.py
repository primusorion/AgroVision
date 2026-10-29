"""Merge the raw sources into one leak-safe corpus.

    python ml/build_dataset.py          -> data/processed/{leaf,pest}/<class>/*.jpg (short side 256 px)
                                           data/manifest_leaf.csv, data/manifest_pest.csv, results/dataset_card.md

Steps: map every source label through ml/taxonomy.py -> resize -> perceptual hash (dHash) -> drop exact duplicates
(and images whose duplicates carry different labels) -> group near-duplicates (256-bit dHash, Hamming <= 10, within a source+class,
plus PlantVillage leaf ids) -> stratified GROUP split 71/14/14, so no near-duplicate is on both sides of a split.
"""
import io
import json
import re
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from taxonomy import INSECT, NOT_PLANT, SOURCES, parse

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data" / "sources", ROOT / "data" / "processed"
SHORT, LONG, SEED, MIN_CLASS, NEAR = 256, 384, 42, 40, 10   # NEAR: Hamming bits out of 256
N_NEG, N_INSECT = 5000, 3000
Image.MAX_IMAGE_PIXELS = None


def leaf_id(fname):
    """PlantVillage: 'xxx___RS_HL 259.1.JPG' / '... Day 13.JPG' are repeat shots of one leaf."""
    b = fname.split("___")[-1].rsplit(".", 1)[0]
    return re.sub(r"(\.\d+)+$", "", re.sub(r"\s*Day\s*\d+$", "", b))


def save(data, dest):
    """Decode, shrink to SHORT px short side (long side centre-cropped to LONG), save JPEG, return 64-bit dHash."""
    im = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    im.draft("RGB", (2 * SHORT, 2 * SHORT))                     # fast DCT-domain downscale for big JPEGs
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = SHORT / min(im.size)
    if s < 1:
        im = im.resize((max(SHORT, round(im.width * s)), max(SHORT, round(im.height * s))), Image.BILINEAR, reducing_gap=2)
    if max(im.size) > LONG:
        l, t = max(0, (im.width - LONG) // 2), max(0, (im.height - LONG) // 2)
        im = im.crop((l, t, l + min(im.width, LONG), t + min(im.height, LONG)))
    im.save(dest, "JPEG", quality=90)
    return hashes(im)


def hashes(im):
    """(64-bit dHash as int - exact-duplicate key, 256-bit dHash as hex - near-duplicate grouping)."""
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), np.int16)
    g16 = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), np.int16)
    return (int(np.packbits((g[:, 1:] > g[:, :-1]).ravel()).view(">u8")[0]),
            np.packbits((g16[:, 1:] > g16[:, :-1]).ravel()).tobytes().hex())


def rec(task, source, label, value, data, uid, gid=None, split=None):
    crop, kind, cond, key = parse(value) if "|" in value else (value.split("___")[0], "other", value.split("___")[1], value)
    key = re.sub(r"[^\w\-(). ]", "_", key)                       # class key doubles as a folder name
    d = OUT / task / key
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{source}-{uid}.jpg"
    try:
        h, h256 = save(data, dest)
    except Exception:
        return None
    return dict(path=str(dest.relative_to(ROOT)).replace("\\", "/"), key=key, crop=crop, kind=kind, cond=cond,
                source=source, imagery=SOURCES.get(source, {}).get("imagery", "other"), hash=h, h256=h256,
                group=f"{source}:{gid}" if gid else None, split=split)


# ---------------------------------------------------------------- per-source workers (run in a process pool)
def do_parquet(job):
    """One row group of a Project-AgML parquet file."""
    task, source, path, fi, rg, labels, crop_col, fixed = job
    pf = pq.ParquetFile(path)
    names = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]["label"].get("names")
    out, miss = [], Counter()
    rows = (r for b in pf.iter_batches(batch_size=16, row_groups=[rg]) for r in b.to_pylist())   # bounded memory
    for i, r in enumerate(rows):
        lab = names[r["label"]] if names and isinstance(r["label"], int) else str(r["label"])
        if fixed:                                       # pest task: keep the source label as the class
            value = f"{fixed}___{lab}".replace(" ", "_")
        else:
            lab = f"{r[crop_col]}/{lab}" if crop_col else lab
            if lab not in labels:
                miss[lab] += 1
                continue
            value = labels[lab]
            if value is None:
                continue
        x = rec(task, source, lab, value, r["image"]["bytes"], f"{fi}_{rg}_{i}", split=r.get("split"))
        if x:
            out.append(x)
    return out, miss


def do_files(job):
    """A chunk of image files on disk: [(path, value, uid, gid)]."""
    task, source, items = job
    out = [rec(task, source, None, v, p, uid, gid) for p, v, uid, gid in items]
    return [x for x in out if x], Counter()


def chunks(items, n=400):
    return [items[i:i + n] for i in range(0, len(items), n)]


def jobs():
    js = []
    # PlantVillage folders
    lab = SOURCES["plantvillage"]["labels"]
    items = [(str(p), lab[d.name], f"{i}_{j}", f"{d.name}/{leaf_id(p.name)}")
             for i, d in enumerate(sorted((SRC / "plantvillage_full" / "raw" / "color").iterdir())) if d.name in lab
             for j, p in enumerate(sorted(d.iterdir()))]
    js += [(do_files, ("leaf", "plantvillage", c)) for c in chunks(items)]
    # Project-AgML parquet sources
    for source, cfg in SOURCES.items():
        for fi, f in enumerate(sorted((SRC / "agml" / source).glob("*.parquet"))):
            js += [(do_parquet, ("leaf", source, str(f), fi, rg, cfg["labels"], cfg.get("crop_col"), None))
                   for rg in range(row_groups(f))]
    # iCassava zip, wheat zip (class = parent folder)
    for source, archive in (("icassava", "cassavaleafdata.zip"), ("wheat_long2023", "wheat_999.zip")):
        z = SRC / source / archive
        if source in SOURCES and z.exists() and z.stat().st_size > 1e6:
            if not (SRC / source / "x").exists():
                zipfile.ZipFile(z).extractall(SRC / source / "x")
            lab = {k.lower(): v for k, v in SOURCES[source]["labels"].items()}
            items = [(str(p), lab[p.parent.name.lower()], str(i), None)
                     for i, p in enumerate(sorted((SRC / source / "x").rglob("*")))
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png") and lab.get(p.parent.name.lower())]
            js += [(do_files, ("leaf", source, c)) for c in chunks(items)]
    # negatives: Imagenette -> "not a plant"
    t = SRC / "imagenette" / "imagenette2-160.tgz"
    if t.exists():
        if not (SRC / "imagenette" / "imagenette2-160").exists():
            tarfile.open(t).extractall(SRC / "imagenette")
        files = sorted((SRC / "imagenette" / "imagenette2-160").rglob("*.JPEG"))
        pick = np.random.default_rng(SEED).choice(len(files), min(N_NEG, len(files)), replace=False)
        js += [(do_files, ("leaf", "imagenette", c)) for c in chunks([(str(files[i]), NOT_PLANT, str(i), None) for i in pick])]
    # insect pests (IP102 mirror): every image for the pest model
    for fi, f in enumerate(sorted((SRC / "agml" / "CT_PestNet_crop_pest_classification").glob("*.parquet"))):
        js += [(do_parquet, ("pest", "ip102", str(f), fi, rg, None, None, "Pest")) for rg in range(row_groups(f))]
    return js


def row_groups(f):
    try:
        return pq.ParquetFile(f).num_row_groups
    except Exception as e:                                   # incomplete download: say so instead of silently shrinking
        print(f"UNREADABLE {f.name}: {e}", flush=True)
        return 0


def run(job):
    return job[0](job[1])


# ---------------------------------------------------------------- de-duplication, grouping, split
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def near_groups(hex256):
    """Union-find over pairs with Hamming distance <= NEAR on the 256-bit dHash.

    64 bits proved too coarse: in datasets shot on one uniform background, distinct leaves chained into a single
    component covering most of a class, so whole classes landed in one split. At 256 bits a re-encoded or resized copy
    differs by a few bits while two different leaves differ by dozens."""
    b = np.frombuffer(bytes.fromhex("".join(hex256)), np.uint8).reshape(-1, 32)
    parent = np.arange(len(b))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(b), 256):
        d = POP[b[s:s + 256, None, :] ^ b[None, :, :]].sum(2)
        for i, j in zip(*np.nonzero(d <= NEAR)):
            if s + i < j:
                parent[find(s + i)] = find(j)
    return np.array([find(i) for i in range(len(b))])


def finalize(df, task):
    n0 = len(df)
    # exact duplicates: keep one; if the copies disagree on the label, drop them all
    per_hash = df.groupby("hash")["key"].nunique()
    conflict = df["hash"].isin(per_hash[per_hash > 1].index)
    df = df[~conflict].drop_duplicates("hash")
    df, stats = group_and_split(df, task)
    stats.update(read=n0, label_conflicts=int(conflict.sum()), duplicates=int(n0 - conflict.sum() - stats.pop("_n")))
    return df, stats


def group_and_split(df, task):
    """Near-duplicate groups (+ source-provided group ids), small-class filter, stratified group split."""
    df = df.reset_index(drop=True)
    gid = np.empty(len(df), object)
    for (src, key), ix in df.groupby(["source", "key"]).indices.items():
        g = near_groups(df["h256"].values[ix])
        hint = df["group"].values[ix]
        first = {}
        for a, (gg, h) in enumerate(zip(g, hint)):          # tie a hinted group to its near-duplicate cluster
            if isinstance(h, str):
                first.setdefault(h, gg)
                g[g == gg] = first[h]
        gid[ix] = [f"{src}|{key}|{x}" for x in g]
    df = df.assign(group=gid)
    counts = df["key"].value_counts()
    small = counts[counts < MIN_CLASS]
    n = len(df)
    df = df[~df["key"].isin(small.index)].reset_index(drop=True)
    sizes = df.groupby(["key", "group"]).size()
    frac = (sizes / sizes.groupby(level=0).transform("sum")).groupby(level=0).max()
    stats = dict(_n=n, small_classes_dropped={k: int(v) for k, v in small.items()}, kept=len(df),
                 groups=int(df["group"].nunique()), largest_group_share_max=round(float(frac.max()), 3),
                 classes_with_group_over_25pct=sorted(frac[frac > 0.25].index))
    if df["split"].notna().all():                           # source ships an official split (IP102)
        df["split"] = df["split"].replace({"validation": "val"})
    else:
        y, g = df["key"].values, df["group"].values
        rest, test = next(StratifiedGroupKFold(7, shuffle=True, random_state=SEED).split(y, y, g))
        tr, va = next(StratifiedGroupKFold(6, shuffle=True, random_state=SEED).split(y[rest], y[rest], g[rest]))
        split = np.empty(len(df), object)
        split[test], split[rest[tr]], split[rest[va]] = "test", "train", "val"
        df["split"] = split
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            assert not set(df.group[df.split == a]) & set(df.group[df.split == b])
        missing = df.groupby("key").split.agg(lambda s: {"train", "val", "test"} - set(s))
        stats["classes_missing_a_split"] = {k: sorted(v) for k, v in missing.items() if v}
    df.drop(columns=["hash", "h256"], errors="ignore").to_csv(ROOT / "data" / f"manifest_{task}.csv", index=False)
    return df, stats


def card(leaf, pest, stats, missing):
    md = ["# AgroVision v2 dataset card (auto-generated by ml/build_dataset.py)\n",
          f"Leaf corpus: **{len(leaf):,} images, {leaf.key.nunique()} classes, {leaf.crop.nunique() - 2} crops** "
          f"(+ 'not a plant' and 'insect' router classes). Pest corpus (IP102): **{len(pest):,} images, {pest.key.nunique()} species**.\n",
          "Split: stratified by class, grouped by near-duplicate cluster (256-bit dHash, Hamming <= 10, within source+class) and PlantVillage "
          f"leaf id; seed {SEED}. Images stored at {SHORT} px short side.\n",
          "## Sources\n", "| Source | Images kept | Classes | Imagery | Licence |", "|---|---|---|---|---|"]
    for s, g in leaf.groupby("source"):
        md.append(f"| {s} | {len(g):,} | {g.key.nunique()} | {SOURCES.get(s, {}).get('imagery', '-')} | "
                  f"{SOURCES.get(s, {}).get('license', 'Apache-2.0 (Imagenette) / CC BY 4.0 (IP102 mirror)')} |")
    md += ["\n## Cleaning\n", "```json", json.dumps(stats, indent=1), "```\n",
           "Labels present in a source but not mapped by ml/taxonomy.py (ignored):\n", "```json", json.dumps(missing, indent=1), "```\n",
           "## Classes by kind\n", "| Kind | Classes | Images |", "|---|---|---|"]
    md += [f"| {k} | {g.key.nunique()} | {len(g):,} |" for k, g in leaf.groupby("kind")]
    md += ["\n## Per-class counts (leaf corpus)\n", "| Class | Kind | Train | Val | Test | Lab | Field/mixed |", "|---|---|---|---|---|---|---|"]
    for k, g in leaf.groupby("key"):
        md.append(f"| {k} | {g.kind.iloc[0]} | {(g.split == 'train').sum()} | {(g.split == 'val').sum()} | "
                  f"{(g.split == 'test').sum()} | {(g.imagery == 'lab').sum()} | {(g.imagery != 'lab').sum()} |")
    (ROOT / "results" / "dataset_card.md").write_text("\n".join(md), encoding="utf-8")


def pv_hints():
    """Processed PlantVillage file stem -> same-leaf group id (mirrors the enumeration in jobs())."""
    lab, out = SOURCES["plantvillage"]["labels"], {}
    for i, d in enumerate(sorted((SRC / "plantvillage_full" / "raw" / "color").iterdir())):
        if d.name in lab:
            for j, p in enumerate(sorted(d.iterdir())):
                out[f"plantvillage-{i}_{j}"] = f"plantvillage:{d.name}/{leaf_id(p.name)}"
    return out


def h256_of(path):
    return hashes(Image.open(ROOT / path))[1]


def resplit():
    """Regroup and re-split the existing processed corpus (no re-decoding of the sources)."""
    leaf = pd.read_csv(ROOT / "data" / "manifest_leaf.csv")
    ins, leaf = leaf[leaf.key == INSECT], leaf[leaf.key != INSECT].copy()
    with ProcessPoolExecutor(20) as ex:
        leaf["h256"] = list(ex.map(h256_of, leaf.path, chunksize=256))
    hints = pv_hints()
    leaf["group"] = [hints.get(Path(p).stem) for p in leaf.path]
    leaf["split"] = None
    leaf, stats = group_and_split(leaf, "leaf")
    stats.pop("_n")
    leaf = pd.concat([leaf.drop(columns=["h256", "hash"], errors="ignore"), ins[[c for c in ins.columns if c != "hash"]]],
                     ignore_index=True)
    leaf.to_csv(ROOT / "data" / "manifest_leaf.csv", index=False)
    card(leaf, pd.read_csv(ROOT / "data" / "manifest_pest.csv"), {"leaf": stats}, {})
    print(json.dumps(stats, indent=1))
    print(leaf.groupby("split").size().to_dict(), "classes", leaf.key.nunique())


def main():
    if "--resplit" in sys.argv:
        return resplit()
    js = jobs()
    if "--smoke" in sys.argv:                                # quick end-to-end check on a slice of the jobs
        js = js[::12]
        globals()["MIN_CLASS"] = 8
    print(len(js), "jobs", flush=True)
    recs, missing = defaultdict(list), defaultdict(Counter)
    with ProcessPoolExecutor(20) as ex:
        for n, (out, miss) in enumerate(ex.map(run, js, chunksize=1)):
            for r in out:
                recs["pest" if "/pest/" in r["path"] else "leaf"].append(r)
            for k, v in miss.items():
                missing[js[n][1][1]][k] += v
            if n % 200 == 0:
                print(n, "/", len(js), {k: len(v) for k, v in recs.items()}, flush=True)
    pest, pstats = finalize(pd.DataFrame(recs["pest"]), "pest")
    leaf = pd.DataFrame(recs["leaf"])
    # router class: a sample of insect photos teaches the leaf model to hand insect pictures to the pest model
    ins = pest.groupby("split", group_keys=False).apply(
        lambda g: g.sample(min(len(g), int(N_INSECT * len(g) / len(pest))), random_state=SEED))
    leaf, lstats = finalize(leaf, "leaf")
    ins = ins.assign(key=INSECT, crop="Insect", kind="other", cond="Insect pest", imagery="other")
    leaf = pd.concat([leaf, ins[leaf.columns]], ignore_index=True).drop(columns=["hash", "h256"])
    leaf.to_csv(ROOT / "data" / "manifest_leaf.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    card(leaf, pest, {"leaf": lstats, "pest": pstats}, {k: dict(v) for k, v in missing.items()})
    print(json.dumps({"leaf": lstats, "pest": pstats}, indent=1))
    print({k: dict(v) for k, v in missing.items()})
    print(leaf.groupby("split").size().to_dict(), "classes", leaf.key.nunique(), "crops", leaf.crop.nunique())


if __name__ == "__main__":
    main()
