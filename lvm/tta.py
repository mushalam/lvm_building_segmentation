"""D4 test-time augmentation for torchvision Mask R-CNN.

A port of sam3-ft-EOSC's sam3ft/tta.py, where it earned about +6% relative AP
for SAM 3. A nadir tile has no canonical "up", so its eight quarter-turn and
mirror views are exact -- nothing is resampled. Each view is predicted, its
masks mapped back to the original frame, and matched to the unrotated pass's
detections by mask IoU; matched mask probabilities and scores are averaged.

Same rules as the original: the output refines the base (unrotated) detection
set and never adds to it, each base detection is claimed at most once per
view, highest-scoring view detection first, and boxes are recomputed from the
fused masks rather than averaged.

One change, for speed only: matching IoU is computed on the GPU at 1/4
resolution as a (view x base) matrix, instead of a Python loop over 400x400
full-resolution mask pairs. Accumulation stays at full resolution.
"""
import torch

# (quarter turns counter-clockwise, then mirror). Identity first.
D4_OPS = [(0, False), (1, False), (2, False), (3, False),
          (0, True), (1, True), (2, True), (3, True)]


def forward(img, turns, mirror):
    """(C,H,W) image into a D4 view. np.rot90 semantics: counter-clockwise."""
    if turns:
        img = torch.rot90(img, turns, dims=(1, 2))
    if mirror:
        img = torch.flip(img, dims=(2,))
    return img


def invert(masks, turns, mirror):
    """(N,H,W) masks from a view back to the original frame."""
    if mirror:
        masks = torch.flip(masks, dims=(2,))
    if turns:
        masks = torch.rot90(masks, -turns, dims=(1, 2))
    return masks


def _iou_matrix(a, b, stride=4):
    """IoU between every mask in a (M,H,W) and b (N,H,W), both bool, at 1/stride."""
    a = a[:, ::stride, ::stride].flatten(1).float()
    b = b[:, ::stride, ::stride].flatten(1).float()
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))


def boxes_from_masks(masks):
    """Tight xyxy boxes for (N,H,W) bool masks; empty masks give a zero box."""
    n = masks.shape[0]
    out = torch.zeros((n, 4), device=masks.device)
    if n == 0:
        return out
    rows, cols = masks.any(2), masks.any(1)
    has = rows.any(1)
    ar_h = torch.arange(masks.shape[1], device=masks.device)
    ar_w = torch.arange(masks.shape[2], device=masks.device)
    big = masks.shape[1] + masks.shape[2]
    out[:, 0] = torch.where(cols, ar_w, big).min(1).values
    out[:, 1] = torch.where(rows, ar_h, big).min(1).values
    out[:, 2] = torch.where(cols, ar_w, -1).max(1).values + 1
    out[:, 3] = torch.where(rows, ar_h, -1).max(1).values + 1
    out[~has] = 0
    return out


@torch.no_grad()
def predict_d4(model, img, mask_thresh=0.5, match_iou=0.5, min_votes=1, ops=D4_OPS,
               fuse="masks"):
    """One (C,H,W) image through all views of `ops`, fused.

    fuse="masks" averages matched mask probabilities and scores, as the SAM 3
    original does. fuse="scores" keeps the base pass's masks and averages only
    the scores. Mask R-CNN is not self-consistent across views the way SAM 3
    is -- on a 119-building B3 tile, matched view/base masks overlap at median
    IoU 0.80-0.87 with centroid spread 7-10 px (split vs merged buildings), and
    no systematic offset -- so averaging its masks blends different
    segmentations: -11% AP75 on 250 v2-test tiles. Scores still carry signal
    (AR +1.1%, matched +1.6 points), which fuse="scores" keeps.

    Returns a torchvision-style dict: masks (N,1,H,W) probabilities, scores,
    boxes, labels, votes.
    """
    assert fuse in ("masks", "scores"), fuse
    base = model([img])[0]
    probs = base["masks"][:, 0].float()                       # (N,H,W)
    scores = base["scores"].float()
    n = len(scores)
    if n == 0:
        return base
    acc = probs.clone()
    score_sum = scores.clone()
    votes = torch.ones(n, device=img.device)
    base_bin = probs > mask_thresh

    for turns, mirror in ops[1:]:
        o = model([forward(img, turns, mirror)])[0]
        if len(o["scores"]) == 0:
            continue
        vp = invert(o["masks"][:, 0].float(), turns, mirror)
        order = torch.argsort(o["scores"], descending=True)
        ious = _iou_matrix(vp[order] > mask_thresh, base_bin).cpu()
        claimed = torch.zeros(n, dtype=torch.bool)
        for r, vi in enumerate(order.tolist()):
            row = ious[r].masked_fill(claimed, -1.0)
            j = int(row.argmax())
            if row[j] < match_iou:
                continue
            claimed[j] = True
            if fuse == "masks":
                acc[j] += vp[vi]
            score_sum[j] += o["scores"][vi]
            votes[j] += 1
            if claimed.all():
                break

    fused = acc / votes[:, None, None] if fuse == "masks" else probs
    keep = votes >= min_votes
    fused, fused_scores, votes = fused[keep], (score_sum / votes)[keep], votes[keep]
    order = torch.argsort(fused_scores, descending=True)
    fused, fused_scores, votes = fused[order], fused_scores[order], votes[order]
    return {"masks": fused[:, None], "scores": fused_scores,
            "boxes": boxes_from_masks(fused > mask_thresh),
            "labels": torch.ones(len(fused_scores), dtype=torch.int64, device=img.device),
            "votes": votes}
