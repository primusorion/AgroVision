"""Image classifiers (leaf conditions, insect pests): class probabilities, Grad-CAM and a domain-shift score."""
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from train import ARCH_NAMES, build, eval_tensor  # same architecture + preprocessing as training


def center_square(img):
    """Centre crop to a square - the region the classifier actually sees (eval_tensor = resize short side + centre crop)."""
    h, w = img.shape[:2]
    s = min(h, w)
    return np.ascontiguousarray(img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s])


class Classifier:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.arch, self.classes, self.task = ck["arch"], ck["classes"], ck.get("task", "leaf")
        self.name = ARCH_NAMES[self.arch]
        self.model = build(self.arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.head = [m for m in self.model.modules() if isinstance(m, nn.Linear)][-1]     # its input = the embedding
        self.target = self.model.features[-1] if hasattr(self.model, "features") else None  # last conv stage -> Grad-CAM
        self.lock = threading.Lock()  # ponytail: one inference at a time; use a model pool if concurrent users matter
        cal = Path(path).with_name(Path(path).stem + "_calibration.json")  # from ml/calibrate.py (train/val only)
        self.cal = json.loads(cal.read_text()) if cal.exists() else None
        self.centroids = np.array(self.cal["centroids"], np.float32) if self.cal else None

    def predict(self, img, cam=True):
        """img: HxWx3 uint8 RGB (square) -> (probs, Grad-CAM overlay 256x256x3 or None, domain-shift score or None).

        Domain shift = cosine distance of the embedding to the centroid of the predicted class."""
        x = eval_tensor(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)).unsqueeze(0)
        acts, want_cam = {}, cam and self.target is not None
        with self.lock:
            hooks = [self.head.register_forward_pre_hook(lambda m, i: acts.update(e=i[0]))]
            if want_cam:
                hooks.append(self.target.register_forward_hook(lambda m, i, o: acts.update(a=o)))
            try:
                with torch.set_grad_enabled(want_cam):
                    logits = self.model(x)
                    grad = torch.autograd.grad(logits[0, logits[0].argmax()], acts["a"])[0] if want_cam else None
            finally:
                for h in hooks:
                    h.remove()
        probs = torch.softmax(logits.detach(), 1)[0].numpy()
        if self.cal:
            probs = probs ** (1 / self.cal.get("temperature", 1.0))      # temperature scaling fitted on validation
            probs /= probs.sum()
        shift = None
        if self.centroids is not None:
            e = torch.nn.functional.normalize(acts["e"].detach().flatten(1), dim=1)[0].numpy()
            shift = float(1 - e @ self.centroids[probs.argmax()])
        overlay = None
        if grad is not None:
            m = torch.relu((grad.mean((2, 3), keepdim=True) * acts["a"]).sum(1))[0].detach().numpy()
            m = cv2.resize(m / (m.max() + 1e-8), (256, 256))
            heat = cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)[:, :, ::-1]
            overlay = (0.55 * cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA) + 0.45 * heat).astype(np.uint8)
        return probs, overlay, shift


if __name__ == "__main__":
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    deployed = json.loads((root / "models" / "deployed.json").read_text())       # the models the app actually serves
    for task in ("leaf", "pest"):
        f = root / "models" / f"{task}_{deployed.get(task)}.pt"
        if not f.exists():
            print(f"skip {task}: no deployed model ({f.name})")
            continue
        clf = Classifier(f)
        df = pd.read_csv(root / "data" / f"manifest_{task}.csv")
        test = df[df.split == "test"].sample(40, random_state=0)
        hit = 0
        for path, key in zip(test.path, test.key):
            p, cam, shift = clf.predict(center_square(np.asarray(Image.open(root / path).convert("RGB"))))
            assert abs(p.sum() - 1) < 1e-4 and (cam is None or (cam.shape == (256, 256, 3) and cam.dtype == np.uint8))
            hit += clf.classes[p.argmax()] == key
        print(f"{task}: {clf.name}, {len(clf.classes)} classes, {hit}/40 sample test images correct")
        assert hit >= (30 if task == "leaf" else 20)
