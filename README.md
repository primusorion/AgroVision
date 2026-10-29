# AgroVision AI v2 — Crop Health Analyzer for 38 crops

KL University Capstone **AIML-25 · On-Device Crop-Disease Detection for Smallholder Farmers**.

Upload a photo of a crop leaf, or of the insect itself. The app returns:

- the **crop** and the **disease, pest damage or nutrient deficiency**;
- the **insect species** for insect photos;
- a **health score (0–100)** and a **fertilizer + treatment plan**;
- Grad-CAM and a leaf-segmentation overlay that show why it decided what it did.

It covers **38 crops and 221 leaf conditions** plus **102 insect pests**, and runs offline on a laptop CPU.

## Quick start

The trained models are included, so there is **no training and no dataset download**: clone, install, run.
A normal laptop CPU is enough; no GPU is needed. Tested with Python 3.10 (3.10–3.12 recommended).

```bash
git clone https://github.com/primusorion/AgroVision.git
cd AgroVision
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`; Linux/macOS: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
python backend/app.py
```

Open http://127.0.0.1:8000 and try one of the sample photos. On Windows, `run.bat` does the last step and opens the
browser. API docs (Swagger, with a file-upload "Try it out"): http://127.0.0.1:8000/docs.

On Linux, `pip install torch` downloads the large CUDA build; for a CPU-only machine install PyTorch first with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then the requirements.

What is in the repo: the code, the deployed models (`models/leaf_efficientnet_v2_s.pt` and
`models/pest_efficientnet_v2_s.pt`, about 80 MB each, plus their calibration files; `models/deployed.json` names them),
the evaluation results that the Performance page shows (`results/`), and 46 sample photos (credits in
`frontend/samples/ATTRIBUTION.md`). The training images are not included: they come from 38 public datasets whose
licences do not all allow redistribution. You only need them to rebuild the corpus or retrain (see Data and Training
below, and `results/dataset_card.md` for every source).

## Capstone objectives → implementation

| Objective | How | Code |
|---|---|---|
| 1. Crop variety identification | One CNN over 223 classes. Crop = class-probability mass per crop (99.2% crop accuracy on test) | `ml/train.py`, `backend/model.py` |
| 2.1 Disease detection | 127 learned disease classes, top-3 + calibrated confidence, Grad-CAM, lesion area % from segmentation | `backend/model.py`, `backend/vision.py` |
| 2.2 Nutrient deficiency | **Learned** for 8 crops with labelled deficiency photos (26 classes: N, P, K, Ca, Mg, S, Fe, Mn, Zn, B). Colour-rule estimate (uniform yellowing → N, purpling → P, marginal scorch → K, interveinal chlorosis → Mg/Fe) for the rest | `backend/advisory.py`, `backend/vision.py` |
| 3.1 Pest detection | **Learned** pest-damage classes (29: armyworm, mites, leaf miner, jassids, …). Insect photos go to a second CNN (IP102, 102 species). Plus disease→vector inference and holes/stippling cues | `backend/advisory.py`, `backend/app.py` |
| 3.2 Fertilizer recommendation | Knowledge base for 39 crops: NPK schedule, corrective fertilizer per deficiency, disease/pest control with doses, organic and cultural options, safety. Reviewed for Indian registration and bee safety | `backend/kb/`, `backend/advisory.py` |
| Health score | Transparent penalty model (diagnosis × severity, lesion area, chlorosis, nutrient stress, pests), with a per-factor breakdown | `backend/advisory.py` |
| Quality gate (problem statement O2) | Resolution, sharpness, brightness, contrast → reject with retake tips. Leaf coverage is a soft warning. "Not a plant" / insect router classes, domain-shift detector, confidence gate | `backend/vision.py`, `backend/app.py` |

## Pipeline

```
photo ─► quality gate ─► leaf CNN (223 classes) ─┬─► "not a plant" ─► rejected with tips
             │ reject                            ├─► "insect" ─────► insect CNN (102 species) ─► host crops, damage, control
             ▼                                   └─► crop + condition (+ Grad-CAM)
        retake tips                                      │
                                                         ▼
                                    leaf segmentation ─► colour profile, lesion %, chlorosis pattern, holes, stippling
                                                         │
                                                         ▼
                                    advisory engine ─► nutrient · pests · health score · fertilizer & treatment
```

## Data

| | Images | Classes | Sources |
|---|---|---|---|
| Leaf corpus | **188,146** (train 134,089 / val 26,703 / test 27,354) | 223 = 221 crop conditions + "not a plant" + "insect" | 38 public datasets (PlantVillage, PlantDoc, iCassava, TOM2024, CCMT, paddy, banana nutrient, CoLeaf, … via Project-AgML) + Imagenette negatives + IP102 sample |
| Insect corpus | **71,965** | 102 species | IP102 (official split) |

The leaf corpus has 127 disease, 36 healthy, 29 pest-damage, 26 deficiency and 3 abiotic classes. 57% of the photos are field photos, 31% are lab photos (plain background) and 8% come from mixed sources; the remaining 4% are the router classes.

How the corpus was built:

- **Labels:** every source label is mapped into one taxonomy, `Crop|kind|Condition` (`ml/taxonomy.py`).
- **Duplicates:** exact duplicates are removed (dHash), and copies with conflicting labels are dropped.
- **Grouped split:** near-duplicates (256-bit dHash, Hamming ≤ 10, within source + class) and PlantVillage leaf ids form groups. The stratified *group* split keeps every group on one side, so no near-duplicate of a test image is in training.
- **Documentation:** the licence of each source and the class counts are in `results/dataset_card.md`.

Rebuild from scratch: `python ml\download_data.py all`, then `python ml\build_dataset.py`.

## Training

```bat
ml\train_all.bat     :: trains/resumes every leaf model, calibrates, picks the deployed one, report + samples + checks
python ml\train.py --task pest --arch efficientnet_v2_s --epochs 12   :: insect model
```

- **Setup:** ImageNet-pretrained backbones, fully fine-tuned on an RTX 5060 Laptop GPU.
- **Optimisation:** mixed precision, EMA weights, label smoothing 0.1, AdamW + OneCycle.
- **Class balance:** a √inverse-frequency sampler.
- **Augmentation (on the GPU):** random resized crop, flips, rotation/shear, colour, blur, JPEG, noise, erasing.
- **Resumable:** `--resume` continues from the last finished epoch.
- **Calibration** (`ml/calibrate.py`, validation split only): temperature scaling, class-centroid domain-shift detector, confidence-gate table.
- **Model selection:** the deployed model is chosen on **validation** macro-F1. The test set is only used for the numbers below.

## Results (held-out test set, 27,354 images)

| Model | Accuracy (95% CI) | Precision | Recall | Macro-F1 (95% CI) | Top-3 | ROC-AUC | Crop acc. | Field acc. | Params | CPU p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **EfficientNetV2-S (deployed)** | **93.38%** (93.06–93.67) | 0.907 | 0.918 | **0.910** (0.901–0.916) | 98.98% | 0.9992 | 99.17% | 90.07% | 20.5 M | 59 ms |
| ConvNeXt-Tiny | 93.28% (92.96–93.59) | 0.908 | 0.916 | 0.910 (0.902–0.915) | 99.13% | 0.9993 | 99.08% | 89.92% | 28.0 M | 59 ms |
| MobileNetV3-Large | 92.95% (92.65–93.26) | 0.906 | 0.910 | 0.905 (0.896–0.911) | 99.02% | 0.9996 | 99.06% | 89.44% | 4.5 M | 10 ms |
| EfficientNet-B0 | 92.78% (92.45–93.09) | 0.899 | 0.907 | 0.901 (0.893–0.906) | 99.07% | 0.9990 | 99.08% | 89.21% | 4.3 M | 17 ms |
| Baseline CNN (from scratch) | 77.30% (76.77–77.82) | 0.736 | 0.766 | 0.739 (0.728–0.745) | 92.26% | 0.9975 | 91.84% | 69.96% | 0.4 M | 4 ms |

Deployed model by kind of finding:

| Kind | Macro-F1 |
|---|---|
| Disease | 0.929 |
| Healthy | 0.965 |
| Pest damage | 0.963 |
| Deficiency | 0.829 |
| Abiotic | 1.000 |
| Router classes ("not a plant" / "insect") | 0.961 |

Deployed model on **lab vs field** photos:

| Imagery | Accuracy |
|---|---|
| Lab | 99.41% |
| Field | 90.07% |

**Insect model** (EfficientNetV2-S, IP102 test split, 102 species):

| Accuracy | Top-3 | Macro-F1 | ROC-AUC |
|---|---|---|---|
| 71.89% | 87.71% | 0.659 | 0.973 |

**Calibration** (temperature 0.76):

- Expected calibration error drops from 0.093 to 0.015.
- At the app's 0.80 confidence gate, the model answers 92.6% of test photos with 96.6% accuracy and flags the rest as low confidence.

**Robustness** (simulated field conditions, test set):

| Condition | Accuracy |
|---|---|
| Low light | 92.9% |
| Sensor noise | 91.4% |
| Over-exposure | 90.9% |
| Defocus blur | 89.9% (the quality gate rejects 99% of these) |
| JPEG compression | 89.4% |
| 25% occlusion | 88.7% |

All tables, slices (per source, per crop, weakest classes, top confusions) and figures are in `results/`. **`results/SUMMARY.md` has everything for the slides.**

### End-to-end (through the real API)

`ml/pipeline_eval.py` sends test photos through `/api/analyze` exactly as the web app does, including the quality gate, router, calibration and advisory. Results are in `results/pipeline_eval.json`.

| Check | Result |
|---|---|
| 3,746 leaf test photos (≈17 per class) | 99.5% answered (0.9% of field photos and 0% of lab photos rejected by the quality gate). Of those answered, 92.8% have the right condition and 99.8% the right crop. The 92% answered with confidence are 96.0% right. |
| 600 IP102 insect test photos | 90.7% routed to the insect model. Species correct for 76.5% of those (top-3: 90.8%). |
| 400 "not a plant" test photos | 99.0% rejected |
| 31 never-seen non-plant photos (Windows wallpapers, a portrait) | 27 rejected, 2 answered with a warning, 2 sent to the insect model without one (see Limitations) |
| Latency (laptop CPU, whole request) | p50 194 ms, p95 264 ms |

## Safety gates

| Gate | Rule |
|---|---|
| Photo quality | Too small, blurry, dark, over-exposed or low-contrast → reject with retake tips. Low leaf coverage → warning only (field photos often show several leaves). |
| Plant check | The leaf model's "not a plant" class (trained on Imagenette scenes and objects) → rejected. |
| Insect routing | The leaf model's "insect" class → the photo goes to the insect model instead of leaf analysis. |
| Domain shift | Cosine distance of the embedding to its class centroid above the validation 99th percentile → warning. |
| Low confidence | Calibrated top-class probability < 0.80 → warning, cautious advice ("confirm before spraying") and top-3 alternatives. |
| Chemical advice | Every recommendation and insect report carries a safety note: label, PPE, pre-harvest interval, bees. Doses are general guidance. |

"Analyze anyway" (`?force=true`) overrides a rejection; the result is then labelled as forced.

## Tests

```bat
python backend\test_api.py     :: every sample vs its class, insect routing + safety note, rejections (non-image, black, portrait)
python backend\kb.py           :: knowledge-base schema + coverage of every class in the taxonomy
python backend\advisory.py     :: advisory engine on all 230 conditions
python backend\vision.py       :: quality gate + segmentation self-check
python backend\model.py        :: deployed models + Grad-CAM + domain shift on test images
python ml\pipeline_eval.py     :: end-to-end evaluation through the API
```

## Layout

```
backend/   app.py (FastAPI) · model.py (classifier, Grad-CAM, domain shift) · vision.py (quality gate, leaf analysis)
           advisory.py (agronomy engine) · kb.py + kb/ (crops/*.json, advice.json, pests.json) · test_api.py
frontend/  index.html · style.css · app.js · samples/ (held-out test photos the model never trained on)
ml/        download_data.py · taxonomy.py · build_dataset.py · gpu_aug.py · train.py · train_all.bat · calibrate.py
           report.py · make_samples.py · pipeline_eval.py · rescore.py · inspect_source.py
models/    deployed leaf + insect models, their calibration files, deployed.json
results/   SUMMARY.md, figures, metrics and training history per model, dataset card, end-to-end evaluation
```

Only the deployed models are in the repo. The other four leaf models of the comparison (ConvNeXt-Tiny, MobileNetV3-L,
EfficientNet-B0, baseline CNN) can be retrained with `ml\train_all.bat`; their metrics are in `results/leaf/`.

## Limitations

- **Field photos:** they are harder than lab photos (90.1% vs 99.4%). Field accuracy is the realistic number for a farmer's phone photo.
- **Weak spots:** classes with few or noisy photos are still weak, for example banana nutrient deficiencies (66.9% for that source), brinjal (74%), okra and PlantDoc web images (63%). See "Weakest classes" in `results/SUMMARY.md`.
- **Rule-based findings:** for crops without labelled deficiency data, nutrient deficiency is a colour-rule estimate. The health score and the fertilizer plan are rule-based. The app labels which findings are learned and which are rules.
- **Insect species:** the insect model reaches 72% top-1 and 88% top-3. Many pests look alike at phone resolution, so the app shows the top 3.
- **Insect router and scenery:** many IP102 photos are wide outdoor scenes, so the router can send a landscape or abstract image to the insect model. A 95th-percentile domain-shift warning catches some of these. The real fix is adding scenery photos to the "not a plant" class and retraining.
- **Wheat:** wheat is in the taxonomy and the knowledge base, but its dataset (Zenodo) could not be downloaded here, so the model does not recognise wheat yet.
- **Crop variety:** "crop variety" means crop species. Cultivar-level identification needs cultivar labels.
- **Doses:** chemical doses are general guidance for Indian conditions. Always follow the product label and the local KVK.

## Licence

The code is released under the MIT licence (`LICENSE`). The model weights were trained on public datasets with mixed
licences, including one non-commercial source (MangoLeafBD, CC BY-NC 3.0); use the weights for research and
non-commercial purposes. The sample photos keep the licences of their source datasets
(`frontend/samples/ATTRIBUTION.md`).
