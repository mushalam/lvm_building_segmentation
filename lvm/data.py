#!/usr/bin/env python3
"""COCO instance-segmentation dataset for torchvision detection models."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util


class BuildingDataset(torch.utils.data.Dataset):
    """One tile, its boxes and its per-instance masks.

    Masks are decoded from RLE at full tile resolution because torchvision's
    Mask R-CNN crops them to each ROI itself. Degenerate boxes are dropped here
    rather than at the loss: a zero-area box produces a NaN in the box
    regression and the failure surfaces far from its cause.
    """

    def __init__(self, root, split, train=True, min_side=2.0, d4=False,
                 scale_jitter=None, copy_paste=0):
        """scale_jitter: (lo, hi) or None. copy_paste: max buildings pasted
        from another random tile, 0 to disable. Both only when train=True;
        see lvm.augment."""
        self.d4 = d4
        self.scale_jitter = scale_jitter if train else None
        self.copy_paste = copy_paste if train else 0
        self.dir = Path(root) / split
        doc = json.loads((self.dir / "_annotations.coco.json").read_text())
        self.images = doc["images"]
        self.train = train
        by_img = {}
        dropped = 0
        for a in doc["annotations"]:
            w, h = a["bbox"][2], a["bbox"][3]
            if w < min_side or h < min_side:
                dropped += 1
                continue
            by_img.setdefault(a["image_id"], []).append(a)
        self.by_img = by_img
        self.dropped = dropped
        # Images with no usable instance would give Mask R-CNN an empty target,
        # which it tolerates but which contributes nothing; keep them out of
        # training and keep them in evaluation, where absence is informative.
        self.index = [i for i in self.images
                      if not train or by_img.get(i["id"])]
        # copy-paste never takes a tile past the densest real one: denser
        # tiles are a scene the data does not contain, and each building costs
        # a full-size float mask inside the model (~17 MB at 2048 px), so an
        # uncapped paste onto a 305-building tile asked for a 5.5 GB block.
        self.max_instances = max((len(v) for v in by_img.values()), default=0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        t, target = self._load(i)
        if self.scale_jitter or self.copy_paste:
            from lvm.augment import scale_jitter, copy_paste
            if self.scale_jitter:
                t, target = scale_jitter(t, target, *self.scale_jitter)
            if self.copy_paste:
                j = int(torch.randint(len(self.index), ()))
                s, st = self._load(j)
                if self.scale_jitter:
                    s, st = scale_jitter(s, st, *self.scale_jitter)
                room = self.max_instances - len(target["masks"])
                if room > 0:
                    t, target = copy_paste(t, target, s, st,
                                           max_paste=min(self.copy_paste, room))
            target.pop("_valid", None)           # augmentation-internal
        if self.d4:
            t, target = apply_d4(t, target, int(torch.randint(8, ())))
        return t, target

    def _load(self, i):
        info = self.index[i]
        img = Image.open(self.dir / info["file_name"]).convert("RGB")
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0

        anns = self.by_img.get(info["id"], [])
        boxes, masks, labels = [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            seg = a["segmentation"]
            if isinstance(seg, dict):
                rle = {"size": seg["size"],
                       "counts": seg["counts"].encode() if isinstance(seg["counts"], str)
                       else seg["counts"]}
            else:
                rle = mask_util.merge(mask_util.frPyObjects(
                    seg, info["height"], info["width"]))
            masks.append(mask_util.decode(rle))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks) if masks else
                                     np.zeros((0, info["height"], info["width"])),
                                     dtype=torch.uint8),
            "image_id": torch.tensor([info["id"]]),
        }
        return t, target


def collate(batch):
    return tuple(zip(*batch))


def apply_d4(img, target, k):
    """One of the 8 dihedral transforms, applied to image and masks together.

    Exact for nadir aerial imagery: a nadir view has no canonical "up", so a
    flipped or quarter-turned tile is an equally valid sample rather than a
    distortion. The same symmetry is why D4 test-time augmentation earned +6%
    AP in the sibling project; this is its train-time counterpart, and it
    multiplies an effective 2,451-image training set by eight.

    Boxes are recomputed from the transformed masks rather than transformed
    themselves. Rotating box corners and re-deriving an axis-aligned extent is
    correct only for multiples of 90 degrees and silently wrong if anyone later
    adds an arbitrary angle; deriving from the mask cannot drift out of
    agreement with it.
    """
    assert 0 <= k < 8, f"k must be 0..7, got {k}"
    if k & 4:
        img = torch.flip(img, dims=[2])
        target["masks"] = torch.flip(target["masks"], dims=[2])
    r = k & 3
    if r:
        img = torch.rot90(img, r, dims=[1, 2])
        target["masks"] = torch.rot90(target["masks"], r, dims=[1, 2])

    m = target["masks"]
    if m.numel() and m.shape[0]:
        boxes = []
        keep = []
        for i in range(m.shape[0]):
            ys, xs = torch.where(m[i] > 0)
            if ys.numel() == 0:
                continue
            boxes.append([xs.min().item(), ys.min().item(),
                          xs.max().item() + 1, ys.max().item() + 1])
            keep.append(i)
        idx = torch.as_tensor(keep, dtype=torch.int64)
        target["masks"] = m[idx]
        target["labels"] = target["labels"][idx]
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return img.contiguous(), target
