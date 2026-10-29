"""Batched GPU data pipeline: nvJPEG decode + vectorised augmentation (no per-image CPU work).

Why: on Windows, DataLoader worker processes each load PyTorch's CUDA libraries (~2.6 GB committed memory per
process -> page-file exhaustion), and thread-based Pillow augmentation is throttled by the GIL while the training loop
runs (~180 img/s). Here one thread reads JPEG bytes; everything else is batched tensor work on the GPU.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg, read_file
from torchvision.transforms import RandomResizedCrop

MEAN, STD, SIZE = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224


def augment(x):
    """x: B x 3 x S x S float [0,1] on the GPU -> augmented batch (same geometry/colour ranges as the CPU pipeline).

    The v1 robustness test showed weakness under sensor noise, heavy JPEG compression and occlusion, so those are
    trained for explicitly; colour jitter is mild with no hue shift because colour carries the diagnosis."""
    b, dev = x.shape[0], x.device
    r = lambda *s: torch.rand(*s, device=dev)  # noqa: E731
    # geometry: rotation +-20 deg, random h/v flips, mild shear on 25% of images (perspective-like), one grid_sample
    ang = (r(b) * 40 - 20) * math.pi / 180
    fx, fy = torch.where(r(b) < 0.5, -1.0, 1.0), torch.where(r(b) < 0.5, -1.0, 1.0)
    sh = (r(b, 2) - 0.5) * 0.3 * (r(b, 1) < 0.25)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = ang.cos() * fx, -ang.sin() * fy + sh[:, 0]
    theta[:, 1, 0], theta[:, 1, 1] = ang.sin() * fx + sh[:, 1], ang.cos() * fy
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    # colour: brightness 0.7-1.3, contrast 0.7-1.3, saturation 0.75-1.25
    lum = lambda t: (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)  # noqa: E731
    x = x * (0.7 + 0.6 * r(b, 1, 1, 1))
    m = lum(x).mean((2, 3), keepdim=True)
    x = (x - m) * (0.7 + 0.6 * r(b, 1, 1, 1)) + m
    g = lum(x)
    x = ((x - g) * (0.75 + 0.5 * r(b, 1, 1, 1)) + g).clamp(0, 1)
    # defocus blur on 20% (one sigma per batch keeps it a single conv)
    k = r(b) < 0.2
    if k.any():
        x[k] = TF.gaussian_blur(x[k], [5, 5], [float(0.1 + 2.4 * torch.rand(1))] * 2)
    # JPEG compression on 20% (messaging-app style); GPU encode + decode per image
    for i in torch.nonzero(r(b) < 0.2).flatten().tolist():
        enc = encode_jpeg((x[i] * 255).byte(), quality=int(torch.randint(15, 60, (1,))))
        x[i] = decode_jpeg(enc.cpu(), mode=ImageReadMode.RGB, device=dev).float() / 255
    # sensor noise on 20%
    n = (r(b, 1, 1, 1) < 0.2) * (0.02 + 0.07 * r(b, 1, 1, 1))
    x = (x + torch.randn_like(x) * n).clamp(0, 1)
    x = (x - torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    # occlusion: random erasing on 25% (2-20% of the area, filled with noise)
    for i in torch.nonzero(r(b) < 0.25).flatten().tolist():
        area = float(0.02 + 0.18 * torch.rand(1)) * SIZE * SIZE
        ar = math.exp(float(torch.empty(1).uniform_(math.log(0.3), math.log(3.3))))
        h, w = min(SIZE - 1, int(round(math.sqrt(area * ar)))), min(SIZE - 1, int(round(math.sqrt(area / ar))))
        t, l = int(torch.randint(0, SIZE - h, (1,))), int(torch.randint(0, SIZE - w, (1,)))
        x[i, :, t:t + h, l:l + w] = torch.randn(3, h, w, device=dev)
    return x


def eval_tensor(im):
    """uint8 CHW tensor (CPU or GPU) -> normalised 3 x SIZE x SIZE: short side to SIZE, centre crop.
    The single preprocessing path for validation, test, calibration, reports AND the web app."""
    x = TF.center_crop(TF.resize(im.float() / 255, SIZE, antialias=True), SIZE)
    return (x - torch.tensor(MEAN, device=x.device).view(3, 1, 1)) / torch.tensor(STD, device=x.device).view(3, 1, 1)


class GPUBatches:
    """Mini-batches decoded (nvJPEG) and augmented on the GPU; one background thread reads the JPEG bytes ahead."""
    def __init__(self, paths, labels, batch, train=False, sampler=None, root=None):
        self.paths = [str(root / p) if root else str(p) for p in paths]
        self.labels = torch.as_tensor(labels)
        self.batch, self.train, self.sampler = batch, train, sampler

    def __len__(self):
        n = len(self.sampler) if self.sampler is not None else len(self.paths)
        return n // self.batch if self.train else -(-n // self.batch)

    def __iter__(self):
        order = list(self.sampler) if self.sampler is not None else list(range(len(self.paths)))
        chunks = [order[i:i + self.batch] for i in range(0, len(order), self.batch)][:len(self)]
        # reader threads decode on the CPU (libjpeg-turbo, GIL released) so decoding overlaps with GPU compute
        read = lambda ix: [decode_jpeg(read_file(self.paths[i]), mode=ImageReadMode.RGB).pin_memory() for i in ix]  # noqa: E731
        ahead = 6
        with ThreadPoolExecutor(3) as pool:
            futs = {k: pool.submit(read, chunks[k]) for k in range(min(ahead, len(chunks)))}
            for k, ix in enumerate(chunks):
                imgs = [im.cuda(non_blocking=True) for im in futs.pop(k).result()]
                if k + ahead < len(chunks):
                    futs[k + ahead] = pool.submit(read, chunks[k + ahead])
                if self.train:                       # random resized crop (scale 0.5-1, ratio 0.8-1.25) per image
                    crops = []
                    for im in imgs:
                        i, j, h, w = RandomResizedCrop.get_params(im, (0.5, 1.0), (0.8, 1.25))
                        crops.append(TF.resized_crop(im.float() / 255, i, j, h, w, [SIZE, SIZE], antialias=True))
                    x = augment(torch.stack(crops))
                else:
                    x = torch.stack([eval_tensor(im) for im in imgs])
                yield x, self.labels[ix].cuda(non_blocking=True)
