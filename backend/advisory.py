"""Advisory engine: classifier probabilities + leaf measurements -> nutrients, pests, health score, fertilizer plan.

All agronomy text lives in backend/kb/ (see kb.py). Class keys are "<Crop>___<Condition>"; every class has a kind:
healthy | disease | pest | deficiency | abiotic. Deficiency and pest findings come from the classifier where labelled
data exists (learned) and from colour / damage rules otherwise (rules); each finding says which.
"""
import kb

CLASS_INFO, CROP_INFO, ADVICE, PESTS = kb.load()
NOT_PLANT, INSECT = "Not_a_plant___Not_a_plant", "Insect___Insect_pest"
ROUTER = {NOT_PLANT, INSECT}
DEFICIENT, LEARNED_MIN = 0.45, 0.30
SAFETY = ("Read and follow the product label; wear gloves, mask and full-sleeve clothing when mixing and spraying; do not "
          "spray in wind or before rain; observe the pre-harvest interval (PHI) before picking; keep chemicals away from "
          "children, animals and water sources. Doses are general guidance - confirm with a soil test and your local KVK "
          "or agriculture officer before applying.")

NUTRIENTS = {
    "N": ("Nitrogen", "Uniform pale-green to yellow colour starting on older (lower) leaves; stunted, spindly growth."),
    "P": ("Phosphorus", "Dull dark-green leaves with a purple/reddish tint, mainly on older leaves; poor roots, late flowering."),
    "K": ("Potassium", "Yellowing then brown scorching of leaf margins and tips on older leaves while the centre stays green."),
    "Mg": ("Magnesium", "Patchy interveinal chlorosis (yellow between green veins) on older leaves."),
    "Fe": ("Iron", "Interveinal chlorosis on young leaves fading to pale yellow or almost white; veins stay green."),
}
# symbol -> (fertilizer, analysis, dose, method, timing)
CORRECTIVE = {
    "N": ("Urea", "46-0-0", "Soil: 20-25 kg/acre (50-60 kg/ha) top-dressed near roots, then irrigate; or foliar 1% (10 g/L) in cool hours",
          "Soil top-dress or foliar spray", "Now, then continue the split schedule"),
    "P": ("DAP (18-46-0) or SSP (0-16-0)", "18-46-0 / 0-16-0",
          "Soil: DAP 20-25 kg/acre or SSP 50 kg/acre banded near the root zone; quick response: 12:61:0 at 5 g/L foliar",
          "Soil band placement + foliar", "At the next side-dressing; foliar twice 10 days apart"),
    "K": ("MOP (0-0-60) / SOP (0-0-50)", "0-0-60 / 0-0-50",
          "Soil: MOP 20-25 kg/acre top-dress; foliar: SOP 10 g/L (1%) or 13:0:45 at 5 g/L",
          "Soil top-dress + foliar spray", "Now; repeat the foliar spray after 10-15 days"),
    "Mg": ("Magnesium sulphate (MgSO4.7H2O)", "9.6% Mg, 12% S", "Foliar 5 g/L (0.5%); soil 10 kg/acre (25 kg/ha)",
           "Foliar spray (soil for long-term correction)", "2 sprays at 10-15 day interval"),
    "Fe": ("Ferrous sulphate (FeSO4.7H2O) or Fe-EDTA (12% Fe)", "19% Fe / 12% Fe",
           "Foliar: FeSO4 5 g/L + citric acid 1 g/L, or Fe-EDTA 1 g/L; soil: FeSO4 10 kg/acre mixed with FYM",
           "Foliar spray on new growth", "2-3 sprays at 7-10 day interval"),
    "Zn": ("Zinc sulphate (ZnSO4.7H2O)", "21% Zn", "Foliar 5 g/L + lime 2.5 g/L; soil 10 kg/acre (25 kg/ha) once in 2-3 seasons",
           "Foliar spray on new flush; soil basal", "2 sprays 10-15 days apart"),
    "Mn": ("Manganese sulphate (MnSO4)", "30.5% Mn", "Foliar 5 g/L + lime 2.5 g/L", "Foliar spray", "2-3 sprays at 10-day interval"),
    "B": ("Borax / Solubor (20% B)", "10.5% B / 20% B", "Foliar: Solubor 1 g/L or borax 2 g/L; soil: borax 4 kg/acre (10 kg/ha)",
          "Foliar spray; soil basal", "At flowering / new flush; boron is toxic in excess - do not overdose"),
    "Ca": ("Calcium nitrate", "15.5-0-0 + 19% Ca", "Foliar 5 g/L; acid soils: agricultural lime as per soil test",
           "Foliar spray (never tank-mix with phosphate or sulphate fertilizers)", "2-3 sprays at 7-10 day interval"),
    "S": ("Gypsum / wettable sulphur", "18% S / 80% S", "Soil: gypsum 100-200 kg/acre; foliar: wettable sulphur 2-3 g/L",
          "Soil basal or foliar", "At planting or early growth"),
}
NUTRIENT_NAMES = {**{k: v[0] for k, v in NUTRIENTS.items()}, "Zn": "Zinc", "Mn": "Manganese", "B": "Boron", "Ca": "Calcium",
                  "S": "Sulphur"}
K_FOLIAR = ("Potassium nitrate 13:0:45", "13-0-45", "5 g/L foliar", "Foliar spray", "2 sprays 10 days apart, from flowering")
K_REASON = {"Viral": "Potassium and micronutrients help infected plants keep producing; they do not cure the virus.",
            "Mite": "Balanced potassium helps damaged leaves recover; extra N would favour mite build-up.",
            "Insect": "Balanced potassium helps damaged leaves recover; lush, over-fertilised growth attracts sucking pests."}
ACTIVE = ("Fungal", "Oomycete", "Bacterial", "Algal")
CATEGORIES = [(80, "Healthy", "#2e7d32"), (60, "Mild stress", "#f9a825"), (40, "Moderate stress", "#ef6c00"),
              (0, "Severe", "#c62828")]


def ramp(x, lo, hi):
    """0 at lo, 1 at hi, linear in between (works decreasing too when hi < lo)."""
    return min(1.0, max(0.0, (float(x) - lo) / (hi - lo)))


def level(score):
    return "High" if score >= 0.7 else "Moderate" if score >= DEFICIENT else "Low"


def kind_mass(class_probs):
    """Probability mass per kind (router classes excluded)."""
    out = dict.fromkeys(kb.KINDS, 0.0)
    for k, p in class_probs.items():
        if k not in ROUTER:
            out[CLASS_INFO[k]["kind"]] += float(p)
    return out


def top_of_kind(class_probs, kind):
    """Most probable class of a kind -> (key, probability) or (None, 0)."""
    c = [(k, float(p)) for k, p in class_probs.items() if k not in ROUTER and CLASS_INFO[k]["kind"] == kind]
    return max(c, key=lambda x: x[1]) if c else (None, 0.0)


# ---------------------------------------------------------------- nutrients
def assess_nutrients(leaf, class_probs, key, conf):
    """Learned deficiency classes (where labelled data exists) + colour/pattern rules for N, P, K, Mg, Fe."""
    cp, pat = leaf["color_profile"], leaf["pattern"]
    gi = float(leaf["color_stats"]["green_index"])
    chl, pale, purp = float(cp["chlorosis_pct"]), float(cp["pale_pct"]), float(cp["purpling_pct"])
    u, ic = float(pat["chlorosis_uniformity"]), float(pat["interior_chlorosis_pct"])
    md, mnr = float(pat["marginal_damage_pct"]), float(pat["marginal_necrosis_ratio"])
    yellow, mla = chl + pale, float(pat["mean_lesion_area_pct"])
    raw = {  # symbol -> (score, evidence)
        "N": (0.45 * ramp(yellow, 5, 35) + 0.35 * u * ramp(yellow, 3, 15) + 0.2 * ramp(gi, 0.35, 0.10),
              f"{yellow:.0f}% of leaf area yellow/pale, uniformity {u:.2f} (1 = evenly spread), green index {gi:.2f}"),
        "P": (ramp(purp, 2, 12), f"{purp:.1f}% of leaf area has a purple/reddish tint"),
        "K": (ramp(mnr, 1.5, 3.0) * ramp(md, 12, 40) * ramp(mla, 1.0, 4.0),
              f"{md:.0f}% of the leaf margin band chlorotic/necrotic; margin lesions {mnr:.1f}x the interior level; "
              f"mean lesion size {mla:.1f}% of leaf"),
        "Mg": (ramp(ic, 10, 35) * (1 - 0.8 * u), f"{ic:.0f}% of the leaf interior yellow, patchy (uniformity {u:.2f})"),
        "Fe": (ramp(pale, 5, 25) * (0.5 + 0.5 * ramp(ic, 3, 15)), f"{pale:.1f}% of leaf area whitish/pale, interior chlorosis {ic:.0f}%"),
    }
    info, notes, factor = CLASS_INFO[key], [], {}
    if conf >= 0.6 and info["kind"] in ("disease", "pest") and info["causes_chlorosis"]:
        factor.update(N=0.6, Mg=0.6, Fe=0.6)
        notes.append(f"{info['name']} itself causes yellowing, so the colour-rule N/Mg/Fe scores were reduced by 40%.")
    if conf >= 0.6 and info["kind"] in ("disease", "pest") and info["causes_marginal_necrosis"]:
        factor["K"] = 0.7
        notes.append(f"{info['name']} also kills leaf margins, so the colour-rule K score was reduced by 30%.")
    indicators = []
    for sym, (score, evidence) in raw.items():
        score = round(score * factor.get(sym, 1.0), 2)
        if sym in factor:
            evidence += f" (discounted x{factor[sym]} for {info['name']})"
        indicators.append({"nutrient": NUTRIENTS[sym][0], "symbol": sym, "score": score, "level": level(score), "evidence": evidence})
    indicators.sort(key=lambda i: -i["score"])

    learned = sorted(((k, float(p)) for k, p in class_probs.items() if k not in ROUTER and CLASS_INFO[k]["kind"] == "deficiency"
                      and p >= 0.05), key=lambda x: -x[1])[:3]
    learned = [{"key": k, "crop": CLASS_INFO[k]["crop"], "name": CLASS_INFO[k]["name"], "symbol": CLASS_INFO[k].get("nutrient"),
                "probability": round(p, 3)} for k, p in learned]
    primary = None
    # learned finding wins over the colour rules (always when it is the diagnosis itself)
    if learned and (learned[0]["probability"] >= LEARNED_MIN or learned[0]["key"] == key):
        t, c = learned[0], CLASS_INFO[learned[0]["key"]]
        primary = {"nutrient": NUTRIENT_NAMES.get(t["symbol"]) or t["name"].replace(" Deficiency", ""),
                   "symbol": t["symbol"], "score": t["probability"],
                   "level": level(t["probability"]), "symptoms": "; ".join(c["symptoms"]), "source": "model",
                   "evidence": f"Classifier: {t['probability']:.0%} probability of {t['name']} on {t['crop']}"}
    elif indicators[0]["score"] >= DEFICIENT:
        primary = {**indicators[0], "symptoms": NUTRIENTS[indicators[0]["symbol"]][1], "source": "rules"}
    if primary is None:
        notes.insert(0, "No nutrient-deficiency pattern was found.")
    notes.append("Deficiency classes are learned from labelled leaf photos for banana, citrus, coffee, date palm, groundnut, "
                 "chilli, lemon and cotton; for other crops the estimate comes from leaf-colour rules (N, P, K, Mg, Fe). "
                 "Confirm with a soil or leaf-tissue test before correcting.")
    return {"status": "deficiency_detected" if primary else "normal", "primary": primary, "learned": learned,
            "indicators": indicators, "color_profile": {k: round(float(v), 2) for k, v in cp.items()}, "note": " ".join(notes)}


# ---------------------------------------------------------------- pests
def control_lines(manage, extra=()):
    """Short control checklist from condition-specific items + the management template."""
    a = ADVICE[manage]
    lines = [f"{t['name']} {t['dose']}".strip() + (f" - {t['notes']}" if t["notes"] else "") for t in list(extra) + a["treatment"]][:3]
    return lines + a["cultural"][:2]


def finding(name, scientific, source, conf, evidence, control):
    return {"name": name, "scientific_name": scientific, "source": source, "confidence": round(min(1.0, float(conf)), 3),
            "evidence": evidence, "control": control}


def assess_pests(leaf, class_probs, key, conf):
    """Learned pest-damage classes + disease->vector links + damage-pattern cues (holes, stippling)."""
    pc, info = leaf["pest_cues"], CLASS_INFO[key]
    holes, hole_area, stip = int(pc["hole_count"]), float(pc["hole_area_pct"]), float(pc["stippling_index"])
    findings = []
    learned = sorted(((k, float(p)) for k, p in class_probs.items() if k not in ROUTER and CLASS_INFO[k]["kind"] == "pest"
                      and (p >= LEARNED_MIN or k == key)), key=lambda x: -x[1])[:2]   # the diagnosis itself always shows
    for k, p in learned:
        c = CLASS_INFO[k]
        findings.append(finding(c["name"], c["cause"], "classifier", p,
                                f"Classifier: {p:.0%} probability of {c['name']} damage on {c['crop']}.",
                                control_lines(c["manage"], c["treatment"])))
    if info.get("vector") and conf >= 0.5:
        findings.append(finding(f"{info['vector']} (vector of {info['name']})", "", "disease-vector", 0.8 * conf,
                                f"{info['name']} detected with {conf:.0%} confidence. It is spread by: {info['vector']}. "
                                "Check leaf undersides and nearby plants for the vector.", control_lines("sucking_insects")))
    if not learned and holes >= 1 and hole_area >= 1.5:
        c = min(0.8, 0.3 + 0.1 * hole_area)
        ev = f"{holes} hole(s) inside the leaf outline covering {hole_area:.1f}% of the leaf, typical of chewing insects."
        if info["type"] == "Bacterial":
            c, ev = c * 0.5, ev + " Bacterial lesions can also fall out and leave holes, so this is uncertain."
        findings.append(finding("Chewing insects (caterpillars / beetles)", "e.g. Spodoptera, Helicoverpa, flea beetles",
                                "damage-pattern", c, ev, control_lines("caterpillars")))
    if not learned and stip >= 0.5 and info["healthy"]:
        findings.append(finding("Sucking pests (thrips / mites)", "e.g. Thrips, Tetranychus", "damage-pattern",
                                min(0.45, 0.25 + 0.4 * (stip - 0.5)),
                                f"Stippling index {stip:.2f} on an otherwise healthy leaf - fine speckling typical of sucking pests.",
                                control_lines("spider_mites")))
    top = max((f["confidence"] for f in findings), default=0.0)
    risk = "None" if not findings else "High" if top >= 0.7 else "Medium" if top >= 0.4 else "Low"
    note = ("No pest or pest-damage signs detected. " if not findings else "") + \
        ("Pest damage is recognised by the classifier for the pest classes in the training data (e.g. fall armyworm, "
         "jassids, mites, leaf miner, hispa), inferred from virus-vector links, or read from holes and stippling. "
         "For a photo of the insect itself, the pest-identification model (102 species) is used instead.")
    return {"detected": bool(findings), "risk": risk, "findings": findings, "note": note}


# ---------------------------------------------------------------- health score
def health_score(leaf, class_probs, key, conf, nutrient, pest):
    """100 minus transparent penalties (diagnosis 40, lesions 25, chlorosis 15, nutrient 10, pests 10)."""
    mass = kind_mass(class_probs)
    p_problem = sum(v for k, v in mass.items() if k != "healthy")
    sev_mass = sum(float(p) * CLASS_INFO[k]["severity_weight"] for k, p in class_probs.items() if k not in ROUTER)
    sev = sev_mass / p_problem if p_problem > 1e-6 else 0.0
    lesion, chl = float(leaf["lesion_pct"]), float(leaf["color_profile"]["chlorosis_pct"])
    prim, info = nutrient["primary"], CLASS_INFO[key]
    top_pest = max(pest["findings"], key=lambda f: f["confidence"]) if pest["detected"] else None
    breakdown = [
        ("Diagnosis", 40 * sev_mass, f"{p_problem:.0%} total probability of a disease, pest or deficiency x severity {sev:.2f} "
                                     f"(predicted: {info['crop']} {info['name']}); max 40"),
        ("Lesion area", 25 * min(1.0, lesion / 30), f"{lesion:.1f}% of leaf area is lesion tissue (full penalty at 30%); max 25"),
        ("Chlorosis", 15 * min(1.0, chl / 40), f"{chl:.1f}% of leaf area yellow (full penalty at 40%); max 15"),
        ("Nutrient stress", 10 * prim["score"] if prim else 0.0,
         f"{prim['nutrient']} deficiency score {prim['score']:.2f} ({prim['source']}); max 10" if prim else "No deficiency pattern detected"),
        ("Pests", 10 * top_pest["confidence"] if top_pest else 0.0,
         f"{top_pest['name']} (confidence {top_pest['confidence']:.0%}); max 10" if top_pest else "No pests detected"),
    ]
    score = round(min(100.0, max(0.0, 100 - sum(p for _, p, _ in breakdown))))
    if not info["healthy"] and score > 79:                 # never call the leaf "Healthy" next to a diagnosed problem
        score = 79
        breakdown[0] = (*breakdown[0][:2], breakdown[0][2] + "; score capped at 79 because a problem was diagnosed")
    _, category, color = next(c for c in CATEGORIES if score >= c[0])
    return {"score": int(score), "category": category, "color": color,
            "breakdown": [{"factor": f, "penalty": round(p, 1), "detail": d} for f, p, d in breakdown]}


# ---------------------------------------------------------------- recommendations
def _fert(name, npk, dose, method, timing, reason):
    return {"name": name, "npk": npk, "dose": dose, "method": method, "timing": timing, "reason": reason}


def _maintenance(crop):
    """Seasonal N:P2O5:K2O for the crop, as straight fertilizers (Urea + DAP + MOP)."""
    n = CROP_INFO[crop]["npk"]
    dap = n["P2O5"] / 0.46
    urea, mop = max(0.0, n["N"] - 0.18 * dap) / 0.46, n["K2O"] / 0.60
    per = "kg/ha" if n["unit"] == "kg/ha" else n["unit"]
    dose = f"Urea {urea:.0f} + DAP {dap:.0f} + MOP {mop:.0f} {per}"
    if n["unit"] == "kg/ha":
        dose += f" (about {urea / 2.47:.0f} + {dap / 2.47:.0f} + {mop / 2.47:.0f} kg/acre)"
    return _fert("Urea + DAP + MOP (straight fertilizers)", "46-0-0 + 18-46-0 + 0-0-60", dose, "Soil application",
                 "As per the split schedule", f"Supplies the recommended {n['N']}:{n['P2O5']}:{n['K2O']} {n['unit']} N:P2O5:K2O.")


def recommend(crop, key, nutrient, pest, health=None, confidence=1.0, uncertain=None):
    """Fertilizer + treatment plan. When uncertain, the summary asks to confirm first and never says 'Immediate'
    on the strength of the diagnosis alone."""
    info, ci = CLASS_INFO[key], CROP_INFO[crop]
    adv, kind, typ = ADVICE[info["manage"]], info["kind"], info["type"]
    active = typ in ACTIVE

    symbols = []                                            # deficiencies to correct: learned first, then colour rules
    if nutrient["primary"] and nutrient["primary"].get("symbol"):
        symbols.append(nutrient["primary"]["symbol"])
    symbols += [i["symbol"] for i in nutrient["indicators"] if i["score"] >= DEFICIENT and i["symbol"] not in symbols]
    fertilizers = []
    for sym in symbols:
        reason = f"Corrects {NUTRIENT_NAMES[sym].lower()} deficiency."
        if sym == "N" and (active or kind == "pest"):
            reason += f" Use the lower dose and split it - excess N favours {info['name'].lower()}."
        fertilizers.append(_fert(*CORRECTIVE[sym], reason))
    if kind == "deficiency" and not symbols:               # unspecified deficiency class
        fertilizers.append(_fert("Micronutrient mixture (Zn, Fe, Mn, B, Cu) + 19:19:19", "19-19-19 + micronutrients", "5 g/L each, foliar",
                                 "Foliar spray", "2 sprays 10-15 days apart", "Broad correction until a soil / leaf test names the nutrient."))
    if kind in ("healthy", "deficiency", "abiotic"):
        fertilizers.append(_maintenance(crop))
        if kind == "healthy":
            fertilizers.append(_fert("19:19:19 (water-soluble NPK)", "19-19-19", "5 g/L foliar", "Foliar spray",
                                     "Vegetative stage, 2 sprays 15 days apart", "Balanced boost during active growth."))
    else:
        fertilizers.append(_fert(*K_FOLIAR, K_REASON.get(typ, "Potassium strengthens cell walls and improves tolerance; "
                                                               "favour K over extra N while the problem is active.")))
        if active:
            fertilizers.append(_fert("Calcium nitrate", "15.5-0-0 + 19% Ca", "5 g/L foliar",
                                     "Foliar spray (never tank-mix with phosphate or sulphate fertilizers)",
                                     "At first symptoms and at flowering, 2 sprays 10-15 days apart",
                                     "Calcium strengthens cell walls and slows tissue breakdown by the pathogen."))
        if typ in ("Viral", "Phytoplasma"):
            fertilizers.append(_fert("Micronutrient mixture (Zn, Fe, Mn, B, Cu)", "-", "5 g/L foliar", "Foliar spray",
                                     "2-3 sprays at 15-day intervals", "Supports stressed, infected plants; it does not cure the infection."))

    treatment = [dict(t) for t in info["treatment"]]
    treatment += [t for t in adv["treatment"] if t["name"] not in {x["name"] for x in treatment}]

    score = health["score"] if health is not None else 100
    unsure = confidence < 0.6 if uncertain is None else uncertain
    severe = (not unsure and (info["severity_weight"] >= 0.85 or pest["risk"] == "High")) or score < 40
    concern = not info["healthy"] or symbols or pest["detected"] or score < 80 or unsure
    urgency = "Immediate" if severe else "Soon" if concern else "Routine"
    n = ci["npk"]
    if unsure:  # don't tell a farmer to spray (or relax) on a guess
        verdict = "is healthy" if info["healthy"] else f"has {info['name']}"
        summary = (f"Uncertain result: the model suggests this {crop.lower()} leaf {verdict} ({confidence:.0%} confidence), "
                   "but the photo is outside what it can judge reliably. Retake the photo (one leaf, daylight, plain "
                   "background) and confirm with your local KVK or agriculture officer before spraying anything.")
    elif info["healthy"] and score < 80:
        summary = (f"No disease was recognised on this {crop.lower()} leaf, but image analysis found damaged or discoloured "
                   f"tissue (health score {score}/100). Inspect the plant closely and retake the photo in good light if unsure.")
    elif info["healthy"]:
        summary = (f"The {crop.lower()} leaf looks healthy. Continue balanced nutrition ({n['N']}:{n['P2O5']}:{n['K2O']} "
                   f"{n['unit']} N:P2O5:K2O) and weekly scouting.")
    else:
        what = {"disease": f"{typ.lower()} disease", "pest": "pest damage", "deficiency": "nutrient deficiency",
                "abiotic": "non-infectious disorder"}[kind]
        summary = f"{crop} leaf shows {info['name']} ({what}). {info['action']}"
    prim = nutrient["primary"]
    if prim and kind != "deficiency":
        summary += f" {prim['nutrient']} deficiency signs ({prim['level'].lower()}, from {prim['source']})."
    extra = [f["name"] for f in pest["findings"] if f["source"] == "damage-pattern"]
    if extra:
        summary += f" Possible pests: {', '.join(extra)} - see treatment."
    note = "Apply well-decomposed FYM / compost before planting; add micronutrients basally only if a soil test shows deficiency."
    if active or kind == "pest":
        note += " While the problem is active, do not exceed the recommended N and apply it in smaller splits."
    return {"urgency": urgency, "summary": summary, "fertilizers": fertilizers, "treatment": treatment,
            "organic": adv["organic"], "cultural": adv["cultural"], "avoid": adv["avoid"],
            "npk_schedule": {"crop": crop, "N": n["N"], "P2O5": n["P2O5"], "K2O": n["K2O"], "unit": n["unit"],
                             "split": ci["npk_split"], "note": note},
            "safety": SAFETY}


def insect_report(pest_probs, k=3):
    """Top-k species of the pest-identification model with knowledge-base details. pest_probs: {IP102 class: prob}."""
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())   # class keys are folder-safe variants of the names  # noqa: E731
    by_norm = {norm(k): v for k, v in PESTS.items()}
    out = []
    for name, p in sorted(pest_probs.items(), key=lambda x: -x[1])[:k]:
        e = by_norm.get(norm(name.split("___")[-1]), {})
        out.append({"key": name, "name": e.get("name", name.split("___")[-1].replace("_", " ")),
                    "scientific_name": e.get("scientific_name", ""), "crops": e.get("crops", []),
                    "damage": e.get("damage", ""), "control": e.get("control", []), "confidence": round(float(p), 3)})
    return out


if __name__ == "__main__":
    import json

    def fake_leaf(chl=1.0, nec=0.5, gi=0.35, u=0.0, ic=0.5, md=1.0, mnr=0.0, mla=0.3, holes=0, hole_area=0.0, stip=0.05):
        return {"color_profile": {"healthy_green_pct": 100 - chl - nec, "chlorosis_pct": chl, "necrosis_pct": nec,
                                  "dark_lesion_pct": 0.0, "purpling_pct": 0.0, "pale_pct": 0.0, "other_pct": 0.0},
                "lesion_pct": nec, "color_stats": {"green_index": gi},
                "pattern": {"marginal_damage_pct": md, "interior_damage_pct": 1.0, "marginal_necrosis_ratio": mnr,
                            "interior_chlorosis_pct": ic, "chlorosis_uniformity": u, "lesion_count": 1, "mean_lesion_area_pct": mla},
                "pest_cues": {"hole_count": holes, "hole_area_pct": hole_area, "speck_count": 3, "stippling_index": stip}}

    assert not kb.validate(), kb.validate()[:10]
    keys = list(CLASS_INFO)
    for key in keys:                                        # every class: confident prediction on a clean leaf
        probs = {k: (0.95 if k == key else 0.05 / (len(keys) - 1)) for k in keys}
        leaf, info = fake_leaf(), CLASS_INFO[key]
        nut = assess_nutrients(leaf, probs, key, 0.95)
        pst = assess_pests(leaf, probs, key, 0.95)
        hs = health_score(leaf, probs, key, 0.95, nut, pst)
        rec = recommend(info["crop"], key, nut, pst, hs, 0.95)
        json.dumps([nut, pst, hs, rec])
        assert 0 <= hs["score"] <= 100 and rec["fertilizers"] and rec["urgency"] in ("Routine", "Soon", "Immediate"), key
        assert (hs["score"] >= 80) == info["healthy"], (key, hs["score"])
        assert (nut["primary"] is not None and nut["primary"]["source"] == "model") == (info["kind"] == "deficiency"), key
        assert any(f["source"] == "classifier" for f in pst["findings"]) == (info["kind"] == "pest"), key
        assert (info.get("vector") is None) or any(f["source"] == "disease-vector" for f in pst["findings"]), key
    blight = fake_leaf(chl=12, nec=35, md=40, mnr=1.2)
    k = "Potato___Late_blight"
    probs = {x: (0.97 if x == k else 0.03 / (len(keys) - 1)) for x in keys}
    hs = health_score(blight, probs, k, 0.97, assess_nutrients(blight, probs, k, 0.97), assess_pests(blight, probs, k, 0.97))
    assert hs["score"] < 40 and hs["category"] == "Severe", hs
    yellow = fake_leaf(chl=30, gi=0.12, u=0.85, ic=28)        # uniform yellowing on a crop without deficiency classes
    k = "Tomato___healthy"
    probs = {x: (0.9 if x == k else 0.1 / (len(keys) - 1)) for x in keys}
    nut = assess_nutrients(yellow, probs, k, 0.9)
    assert nut["primary"]["symbol"] == "N" and nut["primary"]["source"] == "rules", nut["primary"]
    assert recommend("Tomato", k, nut, assess_pests(yellow, probs, k, 0.9))["fertilizers"][0]["name"] == "Urea"
    k = "Cotton___Leaf_reddening"                           # weak deficiency diagnosis still drives the nutrient view
    probs = {x: (0.25 if x == k else 0.75 / (len(keys) - 1)) for x in keys}
    nut = assess_nutrients(fake_leaf(), probs, k, 0.25)
    assert nut["primary"]["source"] == "model" and nut["primary"]["nutrient"] == "Magnesium", nut["primary"]
    assert health_score(fake_leaf(), probs, k, 0.25, nut, assess_pests(fake_leaf(), probs, k, 0.25))["score"] <= 79
    k = "Maize___Fall_armyworm"                             # unsure: classifier pest risk alone is not "Immediate"
    probs = {x: (0.75 if x == k else 0.25 / (len(keys) - 1)) for x in keys}
    pst = assess_pests(fake_leaf(), probs, k, 0.75)
    assert pst["risk"] == "High" and recommend("Maize", k, assess_nutrients(fake_leaf(), probs, k, 0.75), pst,
                                               {"score": 60}, 0.75, uncertain=True)["urgency"] == "Soon"
    probs = {x: (0.2 if x == k else 0.8 / (len(keys) - 1)) for x in keys}   # weak pest diagnosis still listed
    assert [f["source"] for f in assess_pests(fake_leaf(), probs, k, 0.2)["findings"]] == ["classifier"]
    assert len(insect_report({p: 1 / len(PESTS) for p in PESTS})) == 3
    print(f"advisory self-check OK: {len(CROP_INFO)} crops, {len(CLASS_INFO)} conditions, {len(PESTS)} pests")
