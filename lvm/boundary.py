"""Matched mask IoU and boundary IoU, as the SAM 3 incumbent measures them.

Ported from sam3-ft-EOSC's sam3ft/metrics.py (evaluate_file, boundary_iou) so
its 0.7341 mask IoU and 0.1944 boundary IoU can be compared like for like.
Same matching (greedy, highest score first, IoU >= 0.5, each ground truth
claimed once), same 0.05 score floor and 300-detection cap, same dilation (2%
of the image diagonal), and the same seeded 250-image sample.

Both are means over *matched* pairs, so they measure outline quality of what
was found and say nothing about what was missed; boundary_match_rate reports
that side. AP remains the headline metric.
"""
import numpy as np
from pycocotools import mask as mask_util


def sample_ids(coco_gt, max_images=250, seed=0):
    """The incumbent's sample: seeded choice over the sorted ids of the split."""
    ids = sorted(coco_gt.getImgIds())
    if max_images is not None and len(ids) > max_images:
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=max_images, replace=False).tolist())
    return ids


def _band(mask, dilation):
    # Pads with zeros, so a tile edge counts as contour -- the reference
    # implementation's behaviour, kept for comparability.
    import cv2
    binary = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8),
                       iterations=max(1, int(dilation)))[1:-1, 1:-1]
    return binary - (eroded & binary)


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def boundary_iou(gt, pred, dilation_ratio=0.02):
    dilation = max(1, int(round(dilation_ratio * np.hypot(*gt.shape))))
    union = gt | pred
    rows, cols = np.any(union, axis=1), np.any(union, axis=0)
    if not rows.any():
        return 0.0
    # Erosion is local, so a window with dilation+2 of context is exact.
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    m = dilation + 2
    y0, x0 = max(0, y0 - m), max(0, x0 - m)
    y1, x1 = min(gt.shape[0] - 1, y1 + m), min(gt.shape[1] - 1, x1 + m)
    return _iou(_band(gt[y0:y1 + 1, x0:x1 + 1], dilation),
                _band(pred[y0:y1 + 1, x0:x1 + 1], dilation))


def _rle(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        c = seg["counts"]
        return {"size": seg["size"], "counts": c.encode("ascii") if isinstance(c, str) else c}
    return mask_util.merge(mask_util.frPyObjects(seg, h, w))


def boundary_metrics(coco_gt, results, img_ids, dilation_ratio=0.02,
                     match_iou=0.5, score_thresh=0.05, max_dets=300):
    by_img = {}
    for r in results:
        if r["score"] >= score_thresh:
            by_img.setdefault(r["image_id"], []).append(r)
    b_scores, m_scores, total_gt = [], [], 0
    for iid in img_ids:
        info = coco_gt.loadImgs(iid)[0]
        h, w = info["height"], info["width"]
        gts = [_rle(a, h, w) for a in
               coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=iid, iscrowd=False))]
        total_gt += len(gts)
        cands = sorted(by_img.get(iid, []), key=lambda r: -r["score"])[:max_dets]
        if not gts or not cands:
            continue
        dts = [_rle(c, h, w) for c in cands]
        ious = np.asarray(mask_util.iou(dts, gts, [0] * len(gts))).reshape(len(dts), len(gts))
        free = np.ones(len(gts), bool)
        for i in range(len(dts)):
            if not free.any():
                break
            row = np.where(free, ious[i], -1.0)
            j = int(row.argmax())
            if row[j] < match_iou:
                continue
            free[j] = False
            m_scores.append(float(row[j]))
            b_scores.append(boundary_iou(mask_util.decode(gts[j]).astype(bool),
                                         mask_util.decode(dts[i]).astype(bool),
                                         dilation_ratio))
    n = len(b_scores)
    return {"boundary_iou": float(np.mean(b_scores)) if n else 0.0,
            "matched_mask_iou": float(np.mean(m_scores)) if n else 0.0,
            "boundary_matched": n,
            "boundary_match_rate": n / total_gt if total_gt else 0.0,
            "boundary_images": len(img_ids)}
