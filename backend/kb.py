"""Knowledge base: backend/kb/crops/<Crop>.json + kb/advice.json + kb/pests.json.

    python backend/kb.py        validates every file against the class list in ml/taxonomy.py

Crop file:
  {"crop", "scientific_name", "family", "description", "common_varieties": [..],
   "npk": {"N", "P2O5", "K2O", "unit": "kg/ha" | "g/plant/year"}, "npk_split": "...",
   "conditions": {"<Condition as in ml/taxonomy.py>": {
       "name", "kind": healthy|disease|pest|deficiency|abiotic, "cause", "type": one of TYPES,
       "severity_weight": 0..1, "causes_chlorosis": bool, "causes_marginal_necrosis": bool,
       "description", "symptoms": [..], "spread", "action", "manage": <key of advice.json>,
       "treatment": [{"name", "type", "dose", "notes"}],          # condition-specific, shown before the template's
       "nutrient": "N" (deficiency only, optional),  "vector": "<pest name>" (optional)}}}
advice.json:  {"<manage key>": {"treatment": [...], "organic": [...], "cultural": [...], "avoid": [...]}}
pests.json:   {"<IP102 class name>": {"name", "scientific_name", "crops": [..], "damage", "control": [..]}}
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
KINDS = ("healthy", "disease", "pest", "deficiency", "abiotic")
TYPES = ("None", "Fungal", "Oomycete", "Bacterial", "Viral", "Phytoplasma", "Nematode", "Algal", "Insect", "Mite",
         "Nutrient", "Abiotic")
T_TYPES = ("Fungicide", "Bactericide", "Insecticide", "Miticide", "Nematicide", "Fertilizer", "Cultural", "Biological")
COND_KEYS = {"name", "kind", "cause", "type", "severity_weight", "causes_chlorosis", "causes_marginal_necrosis",
             "description", "symptoms", "spread", "action", "manage", "treatment"}
NUTRIENTS = ("N", "P", "K", "Ca", "Mg", "S", "Fe", "Mn", "Zn", "B", "Cu", "Mo")


def key(crop, cond):
    return f"{crop}___{cond}".replace(" ", "_")


def load():
    """-> (CLASS_INFO keyed by class key, CROP_INFO keyed by crop, ADVICE, PESTS)."""
    advice = json.loads((KB / "advice.json").read_text(encoding="utf-8")) if (KB / "advice.json").exists() else {}
    pests = json.loads((KB / "pests.json").read_text(encoding="utf-8")) if (KB / "pests.json").exists() else {}
    classes, crops = {}, {}
    for f in sorted((KB / "crops").glob("*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        crops[c["crop"]] = {k: v for k, v in c.items() if k != "conditions"}
        for cond, v in c["conditions"].items():
            classes[key(c["crop"], cond)] = {"crop": c["crop"], "condition": cond, "healthy": v["kind"] == "healthy", **v}
    return classes, crops, advice, pests


def expected():
    """{(crop, condition): kind} that the dataset taxonomy needs."""
    sys.path.insert(0, str(HERE.parent / "ml"))
    from taxonomy import SOURCES, parse
    out = {}
    for s in SOURCES.values():
        for v in s["labels"].values():
            if v:
                crop, kind, cond, _ = parse(v)
                out[(crop, cond)] = kind
    return out


def validate(only_crops=None):
    """List of human-readable problems (empty = valid). only_crops limits the 'missing' check."""
    bad = []
    try:
        classes, crops, advice, pests = load()
    except Exception as e:
        return [f"cannot load KB: {e!r}"]
    for (crop, cond), kind in sorted(expected().items()):
        if only_crops and crop not in only_crops:
            continue
        c = classes.get(key(crop, cond))
        if c is None:
            bad.append(f"MISSING {crop} / {cond} ({kind})")
        elif c["kind"] != kind:
            bad.append(f"{crop} / {cond}: kind {c['kind']!r} but taxonomy says {kind!r}")
    for k, c in classes.items():
        miss = COND_KEYS - set(c)
        if miss:
            bad.append(f"{k}: missing fields {sorted(miss)}")
            continue
        if c["kind"] not in KINDS or c["type"] not in TYPES:
            bad.append(f"{k}: bad kind/type {c['kind']!r}/{c['type']!r}")
        if not 0 <= c["severity_weight"] <= 1 or (c["kind"] == "healthy") != (c["severity_weight"] == 0):
            bad.append(f"{k}: severity_weight {c['severity_weight']} (0 only for healthy)")
        if advice and c["manage"] not in advice:
            bad.append(f"{k}: manage {c['manage']!r} not in advice.json")
        if not (isinstance(c["symptoms"], list) and c["symptoms"]) or not c["description"] or not c["action"]:
            bad.append(f"{k}: empty symptoms/description/action")
        for t in c["treatment"]:
            if set(t) != {"name", "type", "dose", "notes"} or t["type"] not in T_TYPES:
                bad.append(f"{k}: bad treatment item {t}")
        if c["kind"] == "deficiency" and c.get("nutrient") not in NUTRIENTS + (None,):
            bad.append(f"{k}: nutrient {c.get('nutrient')!r}")
    for name, c in crops.items():
        need = {"crop", "scientific_name", "family", "description", "common_varieties", "npk", "npk_split"}
        if need - set(c) or set(c["npk"]) != {"N", "P2O5", "K2O", "unit"}:
            bad.append(f"crop {name}: needs {sorted(need)} and npk {{N,P2O5,K2O,unit}}")
    for a, v in advice.items():
        if set(v) != {"treatment", "organic", "cultural", "avoid"}:
            bad.append(f"advice {a}: needs treatment/organic/cultural/avoid")
    for p, v in pests.items():
        if set(v) != {"name", "scientific_name", "crops", "damage", "control"}:
            bad.append(f"pest {p}: needs name/scientific_name/crops/damage/control")
    return bad


if __name__ == "__main__":
    problems = validate(set(sys.argv[1:]) or None)
    classes, crops, advice, pests = load()
    print(f"{len(crops)} crops, {len(classes)} conditions, {len(advice)} advice templates, {len(pests)} pests")
    print("\n".join(problems) or "KB valid")
    sys.exit(1 if problems else 0)
