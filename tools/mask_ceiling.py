"""Ceiling imposed by the mask head's MxM grid, given a perfect box.

Each ground-truth mask is resampled into its own box at MxM (as the mask
target is built: roi_align of the mask), then pasted back with torchvision's
own paste_masks_in_image and thresholded at 0.5 -- the model's exact output
path. IoU against the original is the best a perfect mask head could score.
"""
import numpy as np, torch
from torchvision.ops import roi_align
from torchvision.models.detection.roi_heads import paste_masks_in_image
from lvm.boundary import sample_ids
from lvm.data import BuildingDataset
from lvm.evaluate import load_gt

V2 = "data/ign_building_v2/building"
gt = load_gt(f"{V2}/test/_annotations.coco.json")
ids = set(sample_ids(gt, 250))
ds = BuildingDataset(V2, "test", train=False)
ds.index = [i for i in ds.index if i["id"] in ids]
res = {28: [], 56: []}
sizes = []
for n in range(len(ds.index)):
    _, t = ds[n]
    m, b = t["masks"].float(), t["boxes"]
    if not len(b):
        continue
    H, W = m.shape[1:]
    rois = torch.cat([torch.arange(len(b))[:, None].float(), b], 1)
    for M in res:
        # as torchvision's project_masks_on_boxes builds the training target
        tgt = roi_align(m[:, None], rois, (M, M), 1.0)[:, 0]
        back = paste_masks_in_image(tgt[:, None], b, (H, W), padding=1)[:, 0] > 0.5
        g = m > 0.5
        inter = (back & g).flatten(1).sum(1).float(); uni = (back | g).flatten(1).sum(1).float()
        res[M] += (inter / uni.clamp(min=1)).tolist()
    sizes += (m.flatten(1).sum(1)).tolist()
sizes = np.array(sizes)
for M, v in res.items():
    v = np.array(v)
    msg = "  ".join(f"{k} {v[s].mean():.3f}" for k, s in
                    [("small", sizes < 32**2), ("medium", (sizes >= 32**2) & (sizes < 96**2)), ("large", sizes >= 96**2)])
    print(f"{M}x{M} mask grid, perfect box: mean IoU {v.mean():.3f}  share >= .75: {(v >= .75).mean():.3f}  "
          f"share >= .9: {(v >= .9).mean():.3f}  |  {msg}   (n={len(v)})")
