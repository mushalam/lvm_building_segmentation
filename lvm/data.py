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

    def __init__(self, root, split, train=True, min_side=2.0):
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

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
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
