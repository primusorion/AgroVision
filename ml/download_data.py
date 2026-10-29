"""Download the open datasets behind the AgroVision v2 corpus (no login needed, resumable).

    python ml/download_data.py [wave]        -> data/sources/<name>/...      (wave: 1 = core, 2 = large extras, all)

Hugging Face mirrors come from the Project-AgML organisation (uniform parquet: image bytes + label); the raw,
non-augmented configuration is always used. Licences and citations are written to results/dataset_card.md by
ml/build_dataset.py.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sources"
HF = "https://huggingface.co"

# name -> Project-AgML repo                                      (wave 1: everything small enough to grab first)
AGML = {
    1: ["crop_pest_disease_classification", "paddy_disease_classification", "sugarcane_leaf_disease_classification",
        "potato_leaf_disease_classification", "rice_leaf_bacterial_fungal", "rice_leaf_disease_classification_india",
        "cotton_leaf_disease_classification_bangladesh_2", "groundnut_leaf_disease_classification_2",
        "COLD_chili_leaf_disease_classification", "COLD_onion_leaf_disease_classification",
        "mango_leaf_disease_classification", "banana_leaf_disease_classification", "BananaLSD_leaf_disease_classification",
        "banana_leaf_nutrient_classification", "PFSD_Musa_banana_disease_classification", "citrusuat_disease_classification",
        "lemon_leaf_disease_classification", "CoLeaf_nutritional_deficiency_classification",
        "maize_nitrogen_deficiency_classification", "TOM2024_disease_classification", "eggplant_disease_classification",
        "BPLD_leaf_disease_classification", "black_gram_disease_classification", "teaLeafBD_disease_classification",
        "grape_leaf_disease_classification", "lentil_disease_classification", "bean_disease_uganda",
        "IDDMSLD_spinach_leaf_disease_classification", "turmeric_disease_classification",
        "date_palm_leaf_disease_classification", "betel_leaf_disease_classification_2",
        "CT_PestNet_crop_pest_classification", "BAU_insectV2_classification", "plant_doc_classification"],
    2: ["papaya_leaf_disease_classification_bangladesh", "MH_SoyaHealthVision_disease_classification_leaf",
        "OkraDiseaseNet_disease_classification", "sunflower_disease_classification"],
}
# name -> (url, file name)
DIRECT = {
    1: {"imagenette": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "imagenette2-160.tgz"),
        "wheat_long2023": ("https://zenodo.org/records/7573133/files/Long%202023%20Plant%20Path%20999%20photos.zip?download=1",
                           "wheat_999.zip"),
        "icassava": ("https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip", "cassavaleafdata.zip")},
    2: {},
}


def hf_parquets(repo):
    """Parquet files of the raw (else default) config of a Project-AgML dataset: [(url, relative path, bytes)]."""
    req = Request(f"{HF}/api/datasets/Project-AgML/{repo}/tree/main?recursive=1", headers={"User-Agent": "agrovision"})
    files = [f for f in json.load(urlopen(req, timeout=60)) if f["path"].endswith(".parquet")]
    tops = {f["path"].split("/")[0] for f in files}
    keep = "raw" if "raw" in tops else "data" if "data" in tops else sorted(tops - {"augmented"})[0]
    return [(f"{HF}/datasets/Project-AgML/{repo}/resolve/main/{f['path']}", f["path"].replace("/", "__"), f["size"])
            for f in files if f["path"].split("/")[0] == keep]


def fetch(job):
    url, dest, size = job
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and size and dest.stat().st_size == size:
        return f"have  {dest.relative_to(OUT)}"
    r = subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "8", "--retry-delay", "5", "-A", "Mozilla/5.0",
                        "-o", str(dest), url], capture_output=True, text=True)
    got = dest.stat().st_size if dest.exists() else 0
    ok = r.returncode == 0 and (not size or got == size)
    return f"{'ok   ' if ok else 'FAIL '} {dest.relative_to(OUT)}  {got / 1e6:.0f} MB {r.stderr.strip()[:120]}"


def main():
    waves = [1, 2] if len(sys.argv) > 1 and sys.argv[1] == "all" else [int(sys.argv[1]) if len(sys.argv) > 1 else 1]
    jobs = []
    for w in waves:
        for repo in AGML[w]:
            try:
                jobs += [(u, OUT / "agml" / repo / name, size) for u, name, size in hf_parquets(repo)]
            except Exception as e:  # keep going; the summary shows what is missing
                print(f"FAIL  listing {repo}: {e}", flush=True)
        jobs += [(u, OUT / name / fname, 0) for name, (u, fname) in DIRECT[w].items()]
    jobs.sort(key=lambda j: j[2])  # small files first: quick wins, early failures
    print(f"{len(jobs)} files, {sum(j[2] for j in jobs) / 1e9:.1f} GB listed (+ direct downloads)", flush=True)
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(fetch, jobs):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
