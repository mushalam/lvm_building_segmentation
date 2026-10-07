#!/usr/bin/env python3
"""Measure the offset between ground-truth footprints and a model's predicted masks.

Before re-aligning training labels (Align and Segment, arXiv 2607.10841; OMAF,
CVPR 2026), this asks what re-alignment could change. For every ground-truth
building with a confident matching prediction, it searches integer shifts in a
+-R px window for the one that maximises mask IoU between the shifted ground
truth and the prediction (FFT cross-correlation on a crop), and reports:

  - the shift distribution (median and percentiles of its length),
  - IoU at zero shift vs at the best shift: the IoU a perfectly re-aligned
    label would have with today's prediction,
  - how consistent shifts are within a tile: the share of shift variance that
    one per-tile translation explains. High means a per-tile shift (what AnS
    learns) captures the offset; low means it is per building (height-
    dependent lean), which needs per-object alignment,
  - the same split by building size.

Per-building rows go to <out>.npz, so two models' shifts can be compared.

If predictions sit on the labels (shifts ~0), the model has learned the
labels' convention and moving the training labels would move it away from the
unchanged test labels. Reads detection dumps from tools/dump_detections.py
(--keep-rles), so no GPU is needed.

    python tools/label_offset.py --dets runs/rerank1/dets_valid.pt --out offset_valid.json
"""
import argparse, json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util
from scipy.signal import fftconvolve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvm.evaluate import load_gt

R = 24


def best_shift(g, p):
    """g, p: same-size binary crops padded by R. Returns (dy, dx, iou0, iou_best)."""
    inter = fftconvolve(p.astype(np.float32), g[::-1, ::-1].astype(np.float32), mode="same")
    H, W = g.shape
    cy, cx = H // 2, W // 2
    win = np.rint(inter[cy - R:cy + R + 1, cx - R:cx + R + 1])
    union = g.sum() + p.sum() - win
    iou = win / np.maximum(union, 1)
    k = np.unravel_index(np.argmax(iou), iou.shape)
    i0 = iou[R, R]
    return int(k[0] - R), int(k[1] - R), float(i0), float(iou[k])


def work(job):
    iid, h, w, gts, preds = job
    P = [mask_util.decode(r).astype(bool) for r in preds]
    out = []
    if not P:
        return out
    pr = [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in P]
    for aid, g_rle, area in gts:
        G = mask_util.decode(g_rle).astype(bool)
        ious = np.asarray(mask_util.iou(pr, [g_rle], [0])).reshape(-1)
        j = int(np.argmax(ious))
        if ious[j] < 0.1:
            continue
        ys, xs = np.nonzero(G | P[j])
        y0, y1 = max(ys.min() - R, 0), min(ys.max() + R + 1, h)
        x0, x1 = max(xs.min() - R, 0), min(xs.max() + R + 1, w)
        g = np.pad(G[y0:y1, x0:x1], R); p = np.pad(P[j][y0:y1, x0:x1], R)
        dy, dx, i0, ib = best_shift(g, p)
        out.append((iid, aid, area, dy, dx, i0, ib))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="dump with --keep-rles")
    ap.add_argument("--min-score", type=float, default=0.3, help="on ms1's own score")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.dets, weights_only=False)
    ms = d["features"][:, d["names"].index("ms")].numpy()
    gt = load_gt(d["ann_file"])
    by_img = defaultdict(list)
    for i, r, s in zip(d["image_id"].tolist(), d["rles"], ms):
        if s >= args.min_score:
            by_img[i].append(r)
    jobs = []
    for iid in d["img_ids"]:
        info = gt.loadImgs(iid)[0]
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False))
        jobs.append((iid, info["height"], info["width"],
                     [(a["id"], gt.annToRLE(a), a["area"]) for a in anns], by_img.get(iid, [])))
    with Pool(args.workers) as pool:
        rows = [x for part in pool.imap_unordered(work, jobs, chunksize=4) for x in part]

    rows.sort(key=lambda r: r[1])
    a = np.array([r[2:] for r in rows], dtype=np.float64)       # area, dy, dx, iou0, iou_best
    ids = np.array([r[0] for r in rows])
    L = np.hypot(a[:, 1], a[:, 2])

    def summary(sel):
        s = a[sel]; l = L[sel]
        return {"n": int(sel.sum()),
                "shift_px_median": float(np.median(l)),
                "shift_px_p75": float(np.percentile(l, 75)),
                "shift_px_p90": float(np.percentile(l, 90)),
                "share_shift_0": float((l == 0).mean()),
                "share_shift_ge3": float((l >= 3).mean()),
                "iou_at_0_mean": float(s[:, 3].mean()),
                "iou_best_mean": float(s[:, 4].mean()),
                "mean_dy": float(s[:, 1].mean()), "mean_dx": float(s[:, 2].mean())}

    out = {"dets": args.dets, "min_score": args.min_score, "window_px": R,
           "all": summary(np.ones(len(a), bool)),
           "small (<32^2 px)": summary(a[:, 0] < 32 ** 2),
           "medium": summary((a[:, 0] >= 32 ** 2) & (a[:, 0] < 96 ** 2)),
           "large (>=96^2 px)": summary(a[:, 0] >= 96 ** 2)}
    # per-tile consistency: share of shift variance explained by tile means,
    # over tiles with >=5 measured buildings
    keep = np.isin(ids, [i for i, c in zip(*np.unique(ids, return_counts=True)) if c >= 5])
    v = a[keep][:, 1:3]; t = ids[keep]
    tot = ((v - v.mean(0)) ** 2).sum()
    means = {i: v[t == i].mean(0) for i in np.unique(t)}
    within = sum(((v[t == i] - m) ** 2).sum() for i, m in means.items())
    tile_len = np.hypot(*np.array(list(means.values())).T)
    out["per_tile"] = {"tiles": len(means), "buildings": int(keep.sum()),
                       "variance_explained_by_tile_shift": float(1 - within / max(tot, 1e-9)),
                       "tile_mean_shift_px_median": float(np.median(tile_len)),
                       "tile_mean_shift_px_p90": float(np.percentile(tile_len, 90))}
    Path(args.out).write_text(json.dumps(out, indent=2))
    np.savez(Path(args.out).with_suffix(".npz"), ann_id=np.array([r[1] for r in rows]),
             image_id=ids, area=a[:, 0], dy=a[:, 1], dx=a[:, 2], iou0=a[:, 3], iou_best=a[:, 4])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
