"""Scale jitter and copy-paste for aerial building tiles.

Both are the recipe that lets long schedules pay off on COCO instead of
overfitting (Ghiasi et al., "Simple Copy-Paste", CVPR 2021; torchvision's
own maskrcnn_resnet50_fpn_v2 weights were trained with them for 400
epochs). Runs 2-5 overfit after ~12 epochs with neither.

Two adaptations to nadir imagery:

- Scale jitter is narrow (default x0.75-1.33) and crops or pads back to the
  tile size. Ground sampling distance is a real prior here -- a building's
  pixel size means something -- so the wide 0.1-2.0 of COCO LSJ would teach
  sizes the test set never shows. The crop/pad matters: the model's own
  transform resizes every input to --image-size, so a plain resize would be
  undone before the network saw it.
- Copy-paste only pastes onto background. From above, buildings do not
  occlude one another, so a pasted building overlapping an existing one would
  be a scene that cannot occur; those are skipped rather than composited.

Boxes are recomputed from the final masks, as apply_d4 does, so box and mask
cannot disagree, and instances left under 2 px a side are dropped.
"""
import torch
import torch.nn.functional as F


def _boxes_from_masks(masks, min_side=2):
    """(N,H,W) uint8 -> (boxes (K,4) float, keep index (K,)) for non-degenerate masks."""
    if masks.shape[0] == 0:
        return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64)
    m = masks.bool()                 # any() on uint8 stays uint8; where() needs bool
    rows, cols = m.any(2), m.any(1)
    ar_h = torch.arange(masks.shape[1]); ar_w = torch.arange(masks.shape[2])
    big = torch.tensor(masks.shape[1] + masks.shape[2]); neg = torch.tensor(-1)
    x0 = torch.where(cols, ar_w, big).min(1).values
    y0 = torch.where(rows, ar_h, big).min(1).values
    x1 = torch.where(cols, ar_w, neg).max(1).values + 1
    y1 = torch.where(rows, ar_h, neg).max(1).values + 1
    keep = ((x1 - x0) >= min_side) & ((y1 - y0) >= min_side)
    boxes = torch.stack([x0, y0, x1, y1], 1).float()
    return boxes[keep], torch.nonzero(keep)[:, 0]


def _refresh(target):
    boxes, keep = _boxes_from_masks(target["masks"])
    target["masks"] = target["masks"][keep]
    target["labels"] = target["labels"][keep]
    target["boxes"] = boxes
    return target


def scale_jitter(img, target, lo=0.75, hi=1.33, generator=None):
    """Resize by s ~ U(lo, hi), then random-crop or zero-pad back to the input size."""
    _, H, W = img.shape
    s = float(torch.empty(()).uniform_(lo, hi, generator=generator))
    h, w = max(1, round(H * s)), max(1, round(W * s))
    img = F.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    m = target["masks"]
    if m.shape[0]:
        m = F.interpolate(m[None].float(), size=(h, w), mode="nearest")[0].to(torch.uint8)
    else:
        m = torch.zeros((0, h, w), dtype=torch.uint8)
    valid = torch.ones((H, W), dtype=torch.bool)
    if s >= 1:                                   # crop a random H x W window
        y = int(torch.randint(0, h - H + 1, (), generator=generator))
        x = int(torch.randint(0, w - W + 1, (), generator=generator))
        img, m = img[:, y:y + H, x:x + W], m[:, y:y + H, x:x + W]
    else:                                        # pad at a random offset
        y = int(torch.randint(0, H - h + 1, (), generator=generator))
        x = int(torch.randint(0, W - w + 1, (), generator=generator))
        out = torch.zeros((3, H, W)); out[:, y:y + h, x:x + w] = img
        mo = torch.zeros((m.shape[0], H, W), dtype=torch.uint8); mo[:, y:y + h, x:x + w] = m
        img, m = out, mo
        valid[:] = False; valid[y:y + h, x:x + w] = True
    target["masks"] = m.contiguous()
    # Where real imagery is. copy_paste reads it so nothing is pasted onto the
    # zero padding -- a building floating on black is a scene no test tile has.
    # BuildingDataset removes it before the target reaches the model.
    target["_valid"] = valid
    return img.contiguous(), _refresh(target)


def copy_paste(img, target, src_img, src_target, max_paste=30, dilate=2, generator=None):
    """Paste up to max_paste of src's buildings onto img's background, in place.

    A source building is pasted only if it does not touch any building already
    on the tile (existing or pasted), after `dilate` px of dilation, so pasted
    buildings never overlap or abut labelled ones.
    """
    sm = src_target["masks"]
    if sm.shape[0] == 0:
        return img, target
    occ = target["masks"].any(0) if target["masks"].shape[0] else torch.zeros(img.shape[1:], dtype=torch.bool)
    valid = target.get("_valid")
    order = torch.randperm(sm.shape[0], generator=generator)[:max_paste]
    k = 2 * dilate + 1
    added = []
    for i in order.tolist():
        m = sm[i].bool()
        if valid is not None and (m & ~valid).any():
            continue                             # would land on padding
        grown = F.max_pool2d(m[None, None].float(), k, stride=1, padding=dilate)[0, 0].bool()
        if (grown & occ).any():
            continue
        img[:, m] = src_img[:, m]
        occ |= grown
        added.append(sm[i])
    if added:
        target["masks"] = torch.cat([target["masks"], torch.stack(added)])
        target["labels"] = torch.cat([target["labels"],
                                      torch.ones(len(added), dtype=torch.int64)])
        target = _refresh(target)
    return img, target
