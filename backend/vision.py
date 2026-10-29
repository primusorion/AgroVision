"""Photo quality gate and rule-based leaf measurements (OpenCV + numpy only).

quality_check() decides whether a photo is usable for diagnosis.
analyze_leaf() segments the leaf and measures colour classes, damage patterns and
pest-damage cues that advisory.py turns into nutrient / pest / health decisions.

PlantVillage has no nutrient or pest-box labels, so everything here is hand-tuned colour
and shape analysis (HSV thresholds, OpenCV hue 0..180), calibrated on PlantVillage lab
photos (leaf on grey/black background). The numbers are heuristic evidence, not a model.
"""
import cv2
import numpy as np

SIZE = 256            # analysis resolution
BG_DIST = 15          # Lab distance (L weighted 0.3) below which a pixel matches the background colour
HOLE_DIST = 12        # full Lab distance below which a pixel inside the leaf outline is background (a hole)
HOLE_MIN_PX = 12      # smallest enclosed hole (at 256 px) counted as chewing damage
SPECK_MAX_PX = 12     # largest blob (at 256 px) counted as a stippling speck
SHADOW_MIN = 0.003    # smallest smooth dark blob (fraction of image) that can seed a cast shadow
SHADOW_GROW = 13      # shadow seeds grow this wide (px) through dark pixels to take the shadow's edge rim

CLASS_NAMES = ["healthy_green", "chlorosis", "necrosis", "dark_lesion", "purpling", "pale", "other"]


def _ellipse(d):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def _green(h, s, v):
    return (h >= 34) & (h <= 90) & (s >= 50) & (v >= 40)


def _hsv(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def _fill_blobs(m, min_px):
    """Filled external contours of the largest blob and every blob > 15% of it (and >= min_px)."""
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    out = np.zeros(m.shape, np.uint8)
    if areas:
        big = max(0.15 * max(areas), min_px)
        cv2.drawContours(out, [c for c, a in zip(cnts, areas) if a >= big], -1, 1, cv2.FILLED)
    return out.astype(bool)


def _not_leaf(raw, tissue, h, s, v, border, frame):
    """Tissue-coloured pixels that are really a cast shadow or a purple backdrop (needs a real background).

    core   = filled green / yellow (chlorophyll) leaf body, which neither can mimic.
    shadow = outside the core, much darker than the backdrop and smooth (7x7 grey std < 10; lesions measure ~20+),
             grown 4 px so the textured shadow rim goes too. Textured brown / black margins stay leaf.
    purple = purple blobs outside the core that do not touch it, or that run into the image border (a backdrop).
    """
    core = _fill_blobs(cv2.morphologyEx((tissue & (h >= 22) & (h <= 90) & (s >= 40) & (v >= 40)).astype(np.uint8),
                                        cv2.MORPH_OPEN, _ellipse(5)), 0.01 * h.size)
    g = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY).astype(np.float32)
    std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0))
    dark = ~core & (v < 0.45 * np.median(v[border]))            # black backdrops: nothing is darker, so no-op
    n, cc, st, _ = cv2.connectedComponentsWithStats((dark & (std < 10)).astype(np.uint8), connectivity=8)
    lit = cv2.dilate((~tissue & ~dark).astype(np.uint8), _ellipse(SHADOW_GROW)).astype(bool)   # across the penumbra
    seeds = np.isin(cc, [i for i in np.unique(cc[lit]) if i and st[i, cv2.CC_STAT_AREA] >= SHADOW_MIN * h.size])
    shadow = dark & cv2.dilate(seeds.astype(np.uint8), _ellipse(SHADOW_GROW)).astype(bool)
    n, cc = cv2.connectedComponents((tissue & ~core & (h >= 125) & (h <= 172)).astype(np.uint8), connectivity=8)
    near = cv2.dilate(core.astype(np.uint8), _ellipse(5)).astype(bool)
    bad = np.setdiff1d(np.arange(1, n), np.setdiff1d(cc[near], cc[frame]))
    return shadow | np.isin(cc, bad)


def _segment(rgb):
    """Return (tissue, plant, bg_diff) for an RGB uint8 image.

    tissue  = green / yellow / brown pixels that do not match the background colour, minus shadows / purple backdrop
    plant   = cleaned leaf blobs (largest blob + any blob > 15% of it; blobs < 2% of the image are noise)
    bg_diff = HxWx3 Lab difference to the median non-green border colour (None if the leaf fills the frame)
    """
    raw, rgb = rgb, cv2.medianBlur(rgb, 5)               # suppress JPEG / sensor noise (texture uses the raw image)
    h, s, v = _hsv(rgb)
    green = _green(h, s, v)
    tissue = (((h >= 8) & (h <= 95) & (s >= 40) & (v >= 30))
              | ((h >= 8) & (h <= 35) & (s >= 25) & (v >= 70))      # faded tan / straw dead tissue
              | ((h >= 125) & (h <= 172) & (s >= 90) & (v >= 70)))  # vivid purple (P-deficient) tissue; dull purple = backdrop
    frame = np.ones(h.shape, bool)
    frame[8:-8, 8:-8] = False
    border = frame & ~green
    bg_diff = None
    if border.sum() >= 0.3 * frame.sum():
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_diff = lab - np.median(lab[border], axis=0)
        chroma_dist = np.sqrt(((bg_diff * np.float32([0.3, 1, 1])) ** 2).sum(-1))  # shading changes L most
        tissue &= (chroma_dist >= BG_DIST) | green      # beige / tan backgrounds are not leaf
        tissue &= ~_not_leaf(raw, tissue, h, s, v, border, frame)
    m = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(7))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    plant = np.zeros(h.shape, bool)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        plant = np.isin(cc, 1 + np.flatnonzero(areas >= max(0.15 * areas.max(), 0.02 * h.size)))  # noise blobs are tiny
    return tissue, plant, bg_diff


def _pixel_classes(h, s, v):
    """Mutually exclusive colour classes (boolean maps), assigned in priority order."""
    dark = (v < 50) & ~((h >= 30) & (h <= 95) & (s >= 40))   # dark but green-hued = shaded healthy tissue
    pale = ~dark & (s < 35) & (v >= 140)
    rest = ~dark & ~pale
    purple = rest & (h >= 125) & (h <= 172) & (s >= 70) & (v >= 60)
    rest &= ~purple
    necrosis = rest & (s >= 30) & ((h < 22) | (h > 172) | ((h < 30) & (v < 90)))
    rest &= ~necrosis
    chlorosis = rest & (h >= 22) & (h < 34) & (s >= 50) & (v >= 110)
    rest &= ~chlorosis
    green = rest & (h >= 34) & (h <= 90) & (s >= 40)
    return dict(zip(CLASS_NAMES, [green, chlorosis, necrosis, dark, purple, pale, rest & ~green]))


def _pct(part, whole):
    n = int(whole.sum())
    return round(100.0 * int((part & whole).sum()) / n, 2) if n else 0.0


def quality_check(img):
    """Check resolution, sharpness, exposure, contrast and leaf coverage of an RGB uint8 photo."""
    img = np.ascontiguousarray(img)
    hh, ww = img.shape[:2]
    scale = min(SIZE / min(hh, ww), 4 * SIZE / max(hh, ww))        # short side -> 256 px, long side <= 1024
    small = cv2.resize(img, (max(1, round(ww * scale)), max(1, round(hh * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    _, plant, _ = _segment(small)

    res = float(min(hh, ww))
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    cover = 100.0 * float(plant.mean())
    v = _hsv(small)[2]
    bright = float(v[plant].mean() if cover >= 5 else v.mean())   # judge the leaf, not a dark backdrop
    contrast = float(gray.std())

    def check(name, value, threshold, hard, soft, msg_hard, msg_soft):
        ok = not (hard or soft)
        return {"name": name, "value": round(value, 2), "threshold": threshold, "passed": ok,
                "severity": "hard" if hard else "soft", "message": "OK" if ok else (msg_hard if hard else msg_soft)}

    checks = [
        check("Resolution", res, ">= 64 px short side (>= 128 recommended)", res < 64, res < 128,
              f"Image is too small ({int(res)} px) to analyse.",
              f"Low resolution ({int(res)} px); small lesions may be missed."),
        check("Sharpness", sharp, "variance of Laplacian >= 15 (>= 50 recommended)", sharp < 15, sharp < 50,
              "Image is too blurry to see lesions.", "Image is slightly blurry."),
        check("Brightness", bright, "mean HSV value of the leaf 25-235 (50-215 recommended)",
              bright < 25 or bright > 235,
              bright < 50 or bright > 215,
              "Image is too dark." if bright < 128 else "Image is overexposed.",
              "Image is a little dark." if bright < 128 else "Image is a little bright."),
        check("Contrast", contrast, "grey-level std >= 15", False, contrast < 15, "",
              "Low contrast; leaf details look washed out."),
        # warning only: this colour segmentation assumes a plain backdrop and misses leaves in field photos
        # (up to 14% false rejections per field source); the classifier's "not a plant" class rejects non-plants.
        check("Leaf coverage", cover, ">= 10% of image is leaf (recommended)", False, cover < 10, "",
              "Leaf not clearly separated from the background - fill the frame with one leaf."),
    ]
    tips = {
        "Resolution": "Use the full camera resolution and move closer so the leaf fills most of the frame.",
        "Sharpness": "Hold the phone steady, tap the screen to focus on the leaf, then retake.",
        "Brightness": ("Take the photo in daylight or add light; avoid deep shade." if bright < 128 else
                       "Avoid direct glare; shade the leaf or photograph it in open shade."),
        "Contrast": "Place the leaf on a plain background that differs from the leaf colour.",
        "Leaf coverage": "Photograph a single leaf, centred, filling most of the frame.",
    }
    failed = sorted((c for c in checks if not c["passed"]), key=lambda c: c["severity"] != "hard")
    n_hard = sum(c["severity"] == "hard" for c in failed)
    return {
        "passed": n_hard == 0,
        "score": int(np.clip(100 - 40 * n_hard - 12 * (len(failed) - n_hard), 0, 100)),
        "checks": checks,
        "issues": [c["message"] for c in failed],
        "tips": [tips[c["name"]] for c in failed],
    }


def analyze_leaf(img):
    """Segment the leaf and measure colour classes, damage pattern and pest-damage cues."""
    rgb = cv2.resize(np.ascontiguousarray(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    h, s, v = _hsv(rgb)
    tissue, plant, bg_diff = _segment(rgb)

    # Leaf outline = filled external contours of the plant blobs.
    cnts, _ = cv2.findContours(plant.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((SIZE, SIZE), np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    filled = filled.astype(bool)

    # Holes (chewing damage): non-tissue pixels inside the outline that match the background colour.
    holes, hole_count = np.zeros_like(filled), 0
    if bg_diff is not None:
        bg_like = np.sqrt((bg_diff ** 2).sum(-1)) < HOLE_DIST
        cand = cv2.morphologyEx((filled & ~tissue & bg_like).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3))
        n, cc, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        keep = 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= HOLE_MIN_PX)
        holes, hole_count = np.isin(cc, keep), len(keep)
    leaf = filled & ~holes
    leaf_px = max(int(leaf.sum()), 1)

    cls = {k: m & leaf for k, m in _pixel_classes(h, s, v).items()}
    profile = {f"{k}_pct": _pct(m, leaf) for k, m in cls.items()}
    lesion = cls["necrosis"] | cls["dark_lesion"]
    damage = lesion | cls["chlorosis"]

    # Edge band ~8% of leaf size wide vs interior; the outer 2 px mix leaf and background colour, skip them.
    ring = max(2, round(0.08 * np.sqrt(filled.sum())))
    inner = cv2.erode(filled.astype(np.uint8), _ellipse(2 * ring + 1)).astype(bool)
    band = leaf & ~inner & cv2.erode(filled.astype(np.uint8), _ellipse(5)).astype(bool)
    interior = leaf & inner
    band_nec, int_nec = _pct(lesion, band), _pct(lesion, interior)
    nec_ratio = 0.0 if band_nec < 3 else min(10.0, band_nec / max(int_nec, 2.0))

    # Chlorosis uniformity: 1 - coefficient of variation over a 4x4 grid of the leaf bounding box.
    uniformity = 0.0
    ys, xs = np.nonzero(leaf)
    if len(ys):
        fracs = []
        for ry in np.array_split(np.arange(ys.min(), ys.max() + 1), 4):
            for rx in np.array_split(np.arange(xs.min(), xs.max() + 1), 4):
                cell = leaf[np.ix_(ry, rx)]
                if cell.sum() >= max(20, 0.25 * cell.size):
                    fracs.append(cls["chlorosis"][np.ix_(ry, rx)].sum() / cell.sum())
        if fracs and np.mean(fracs) >= 0.03:
            uniformity = float(np.clip(1 - np.std(fracs) / np.mean(fracs), 0, 1))

    n, _, stats, _ = cv2.connectedComponentsWithStats(lesion.astype(np.uint8), connectivity=8)
    blobs = stats[1:, cv2.CC_STAT_AREA]
    blobs = blobs[blobs >= 4]

    # Stippling: tiny specks brighter and yellower than their 9x9 neighbourhood, on green tissue
    # (s >= 40 drops white glare). Weak cue: PlantVillage healthy leaves score ~0.1-0.35 from hairs/texture.
    hf, vf = h.astype(np.float32), v.astype(np.float32)
    on_green = cv2.blur(cls["healthy_green"].astype(np.float32), (11, 11)) >= 0.6
    specks = (cv2.erode(leaf.astype(np.uint8), _ellipse(7)).astype(bool) & on_green & (s >= 40)
              & (vf - cv2.blur(vf, (9, 9)) >= 20) & (cv2.blur(hf, (9, 9)) - hf >= 3))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(specks.astype(np.uint8), connectivity=8)
    small = 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] <= SPECK_MAX_PX)
    specks = np.isin(cc, small)
    speck_density = len(small) / (leaf_px / 1000)                  # specks per 1000 leaf px

    px = leaf if leaf.any() else np.ones_like(leaf)
    rgbf = rgb.astype(np.float32)
    exg = (2 * rgbf[..., 1] - rgbf[..., 0] - rgbf[..., 2]) / (rgbf.sum(-1) + 1e-6)

    return {
        "mask": leaf.astype(np.uint8) * 255,
        "leaf_coverage_pct": round(100.0 * float(leaf.mean()), 2),
        "color_profile": profile,
        "lesion_pct": round(profile["necrosis_pct"] + profile["dark_lesion_pct"], 2),
        "color_stats": {
            "mean_hue": round(float(h[px].mean()), 2),
            "mean_saturation": round(float(s[px].mean()), 2),
            "mean_value": round(float(v[px].mean()), 2),
            "green_index": round(float(exg[px].mean()), 4),
        },
        "pattern": {
            "marginal_damage_pct": _pct(damage, band),
            "interior_damage_pct": _pct(damage, interior),
            "marginal_necrosis_ratio": round(nec_ratio, 2),
            "interior_chlorosis_pct": _pct(cls["chlorosis"], interior),
            "chlorosis_uniformity": round(uniformity, 3),
            "lesion_count": int(len(blobs)),
            "mean_lesion_area_pct": round(100.0 * float(blobs.mean()) / leaf_px, 3) if len(blobs) else 0.0,
        },
        "pest_cues": {
            "hole_count": int(hole_count),
            "hole_area_pct": _pct(holes, filled),
            "speck_count": int(len(small)),
            "stippling_index": round(float(np.clip(speck_density / 12, 0, 1)), 3),
        },
        "overlay": _overlay(rgb, leaf, filled, holes, cls, specks),
    }


def _overlay(rgb, leaf, filled, holes, cls, specks):
    """Reviewer-friendly picture: dimmed background, tinted damage classes, outlined leaf and holes."""
    out = rgb.astype(np.float32)
    out[~leaf] *= 0.35
    for key, colour, alpha in [("chlorosis", (255, 225, 0), 0.55), ("necrosis", (255, 40, 40), 0.6),
                               ("dark_lesion", (130, 0, 0), 0.75), ("purpling", (170, 0, 255), 0.6)]:
        m = cls[key]
        out[m] = (1 - alpha) * out[m] + alpha * np.float32(colour)
    out[specks] = (0, 255, 255)
    out = out.clip(0, 255).astype(np.uint8)
    for m, colour, width in [(filled, (0, 255, 120), 1), (holes, (255, 0, 255), 2)]:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, colour, width)
    return out


if __name__ == "__main__":
    from pathlib import Path
    data = Path(__file__).resolve().parents[1] / "data" / "PlantVillage"
    load = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    imgs = {c: [load(p) for p in sorted((data / c).glob("*.JPG"))[:2]] for c in ["Tomato_healthy", "Tomato_Late_blight"]}

    keys = {"mask", "leaf_coverage_pct", "color_profile", "lesion_pct", "color_stats", "pattern", "pest_cues", "overlay"}
    sub = {"color_profile": {f"{k}_pct" for k in CLASS_NAMES},
           "color_stats": {"mean_hue", "mean_saturation", "mean_value", "green_index"},
           "pattern": {"marginal_damage_pct", "interior_damage_pct", "marginal_necrosis_ratio", "interior_chlorosis_pct",
                       "chlorosis_uniformity", "lesion_count", "mean_lesion_area_pct"},
           "pest_cues": {"hole_count", "hole_area_pct", "speck_count", "stippling_index"}}
    res = {}
    for c, ims in imgs.items():
        for im in ims:
            q, a = quality_check(im), analyze_leaf(im)
            assert q["passed"] and q["score"] >= 80, (c, q["issues"])
            assert [ch["name"] for ch in q["checks"]] == ["Resolution", "Sharpness", "Brightness", "Contrast", "Leaf coverage"]
            assert set(a) == keys and all(set(a[k]) == v for k, v in sub.items())
            assert a["mask"].shape == (SIZE, SIZE) and a["mask"].dtype == np.uint8 and set(np.unique(a["mask"])) <= {0, 255}
            assert a["overlay"].shape == (SIZE, SIZE, 3) and a["overlay"].dtype == np.uint8
            assert abs(sum(a["color_profile"].values()) - 100) < 0.5 and a["leaf_coverage_pct"] > 10
            res.setdefault(c, []).append(a)
    assert all(a["lesion_pct"] < 5 and a["color_profile"]["chlorosis_pct"] < 5 for a in res["Tomato_healthy"])
    assert all(a["lesion_pct"] > 10 for a in res["Tomato_Late_blight"])

    # Bad photos must fail the gate hard.
    real = imgs["Tomato_healthy"][0]
    noise = np.clip(np.float32([90, 110, 170]) + np.random.default_rng(0).normal(0, 40, (256, 256, 3)), 0, 255)
    bad = {"black": np.zeros((300, 400, 3), np.uint8), "white": np.full((300, 400, 3), 255, np.uint8),
           "blurred": cv2.GaussianBlur(real, (0, 0), 3), "no leaf": noise.astype(np.uint8), "tiny": real[:40, :40]}
    for name, im in bad.items():
        q = quality_check(im)
        if name == "no leaf":                                       # leaf coverage is a warning; the classifier's
            assert q["passed"] and q["issues"], name                # "not a plant" class rejects non-plant photos
        else:
            assert not q["passed"] and q["issues"] and q["tips"], name
        analyze_leaf(im)                                            # must not crash on junk either

    # Synthetic symptoms: background-coloured holes are found; uniform yellowing reads as uniform chlorosis.
    a = res["Tomato_healthy"][0]
    rgb = cv2.resize(real, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    leaf = a["mask"] > 0
    bg = np.median(rgb[~cv2.dilate(a["mask"], _ellipse(9)).astype(bool)], axis=0).astype(int).tolist()
    ys, xs = np.nonzero(cv2.erode(a["mask"], _ellipse(41)))
    holed = rgb.copy()
    for i in np.linspace(0, len(ys) - 1, 3).astype(int):
        cv2.circle(holed, (int(xs[i]), int(ys[i])), 5, bg, -1)
    assert a["pest_cues"]["hole_count"] == 0 and analyze_leaf(holed)["pest_cues"]["hole_count"] >= 2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[leaf, 0] = 27
    hsv[leaf, 2] = np.clip(hsv[leaf, 2].astype(int) + 60, 0, 255)
    y = analyze_leaf(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    assert y["color_profile"]["chlorosis_pct"] > 60 and y["pattern"]["chlorosis_uniformity"] > 0.7

    print("vision self-check OK:", {c: [round(a["lesion_pct"], 1) for a in v] for c, v in res.items()})
