#!/usr/bin/env python3
"""Cache a Mask Scoring checkpoint's detections with context features, for a re-ranker.

HYDRA ("Queries Knew More Than We Thought", arXiv 2609.20283) re-ranks a frozen
model's outputs with a small model trained only on cached outputs, using what a
per-RoI head cannot see: how each detection relates to the others and to the
image. This writes, per detection:

  own      class score, predicted mask IoU, their product, log mask/box area,
           aspect, mask fill of its box, mean mask probability, boundary softness
           (share of pixels with 0.2<p<0.8 among those and the mask's), distance to the tile edge, rank
  context  max mask IoU with a higher-ranked / any detection, how much of it a
           higher-ranked one covers, how much of another it covers, overlap
           counts at IoU 0.1/0.3/0.5, the score and predicted IoU of its
           strongest-overlap neighbour
  image    detections above 0.5, mean of the top-50 scores, mean predicted IoU
           above 0.5, fraction of the tile covered by them, channel means/stds

and the target: the true mask IoU with the best-matching ground truth (the
oracle's rule, tools/oracle_rescore.py). With --keep-rles the masks are kept
too, so the re-ranked detections can be scored with lvm.evaluate.

    python tools/dump_detections.py --ckpt runs/ms1_head_only/best.pt \\
        --data data/ign_building_v2/building --split test --keep-rles --out dets_test.pt
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvm.data import BuildingDataset
from lvm.evaluate import load_gt
from lvm.train import build_model

OWN = ["cls", "pred_iou", "ms", "log_area", "log_box_area", "aspect", "fill",
       "mean_prob", "softness", "edge_dist", "rank_frac", "log_rank"]
CTX = ["max_iou_higher", "max_iou_any", "covered_by_higher", "covers_other",
       "n_iou_01", "n_iou_03", "n_iou_05", "nb_ms", "nb_pred_iou", "nb_cls_ratio"]
IMG = ["n_dets_05", "top50_mean", "pred_iou_mean_05", "cover_05", "n_dets",
       "ch_mean0", "ch_mean1", "ch_mean2", "ch_std0", "ch_std1", "ch_std2"]
FEATURES = OWN + CTX + IMG


@torch.no_grad()
def features(o, img):
    """o: one model output (sorted by the model's final score). Returns (N, F)."""
    P = o["masks"][:, 0].float()                                  # N,H,W probabilities
    N, H, W = P.shape
    if N == 0:
        return torch.zeros(0, len(FEATURES))
    B = P > 0.5
    area = B.flatten(1).sum(1).float()
    bx = o["boxes"]
    bw, bh = (bx[:, 2] - bx[:, 0]).clamp(min=1), (bx[:, 3] - bx[:, 1]).clamp(min=1)
    mean_prob = (P * B).flatten(1).sum(1) / area.clamp(min=1)
    unsure = ((P > 0.2) & (P < 0.8)).flatten(1).sum(1).float()
    soft = unsure / (area + unsure).clamp(min=1)                  # 0..1, defined for empty masks
    edge = torch.stack([bx[:, 0], bx[:, 1], W - bx[:, 2], H - bx[:, 3]], 1).min(1).values / W
    cls, piou, ms = o["scores_cls"], o["mask_iou"], o["scores"]
    rank = torch.arange(N, device=P.device, dtype=torch.float32)

    # pairwise overlaps on a 2x-pooled grid (fractional cells keep tiny masks)
    S = F.avg_pool2d(B[:, None].float(), 2)[:, 0].flatten(1)      # N, HW/4
    inter = S @ S.T
    a = S.sum(1)
    iou = inter / (a[:, None] + a[None, :] - inter).clamp(min=1e-6)
    cont = inter / a[:, None].clamp(min=1e-6)                     # cont[i,j]: share of i inside j
    eye = torch.eye(N, dtype=torch.bool, device=P.device)
    iou_o = iou.masked_fill(eye, 0)
    higher = torch.ones(N, N, device=P.device).tril(-1).bool()    # j < i: ranked above i
    max_iou_higher = (iou_o * higher).max(1).values
    max_iou_any = iou_o.max(1).values
    covered_by_higher = (cont.masked_fill(eye, 0) * higher).max(1).values
    covers_other = cont.T.masked_fill(eye, 0).max(1).values
    n01, n03, n05 = [(iou_o > t).sum(1).float() for t in (0.1, 0.3, 0.5)]
    nb = iou_o.argmax(1)
    has_nb = max_iou_any > 0
    nb_ms = torch.where(has_nb, ms[nb], torch.zeros_like(ms))
    nb_piou = torch.where(has_nb, piou[nb], torch.zeros_like(ms))
    nb_ratio = torch.where(has_nb, cls / cls[nb].clamp(min=1e-6), torch.zeros_like(ms))

    good = ms > 0.5
    n05_img = good.sum().float()
    top50 = ms[:50].mean()
    piou05 = piou[good].mean() if good.any() else torch.tensor(0.0, device=P.device)
    cover05 = B[good].any(0).float().mean() if good.any() else torch.tensor(0.0, device=P.device)
    chm, chs = img.flatten(1).mean(1).to(P.device), img.flatten(1).std(1).to(P.device)

    own = [cls, piou, ms, torch.log1p(area), torch.log1p(bw * bh), torch.log(bw / bh),
           area / (bw * bh), mean_prob, soft, edge, rank / N, torch.log1p(rank)]
    ctx = [max_iou_higher, max_iou_any, covered_by_higher, covers_other, n01, n03, n05,
           nb_ms, nb_piou, nb_ratio]
    imgf = [n05_img, top50, piou05, cover05, torch.tensor(float(N), device=P.device),
            *chm, *chs]
    cols = own + ctx + [v.expand(N) for v in imgf]
    return torch.stack([c.float() for c in cols], 1).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-images", type=int, default=0, help="random subset, 0 = all")
    ap.add_argument("--keep-rles", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert ck.get("mask_scoring"), "needs a Mask Scoring checkpoint (predicted IoU is a feature)"
    model = build_model(400, image_size=ck["image_size"], backbone=ck.get("backbone", "resnet50"),
                        scratch_heads=ck.get("scratch_heads", False), mask_scoring=True)
    model.load_state_dict(ck["model"]); model.cuda().eval()
    gt = load_gt(str(Path(args.data) / args.split / "_annotations.coco.json"))
    ds = BuildingDataset(args.data, args.split, train=False)
    order = list(range(len(ds.index)))
    if args.max_images and args.max_images < len(order):
        g = torch.Generator().manual_seed(0)
        order = sorted(torch.randperm(len(order), generator=g)[:args.max_images].tolist())

    feats, target, image_id, rles = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, n in enumerate(order):
            img, _ = ds[n]
            iid = ds.index[n]["id"]
            o = model([img.cuda()])[0]
            feats.append(features(o, img))
            masks = (o["masks"][:, 0] > 0.5).cpu().numpy()
            r = [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
            g_ = [gt.annToRLE(a) for a in gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False))]
            if r and g_:
                best = np.asarray(mask_util.iou(r, g_, [0] * len(g_))).reshape(len(r), len(g_)).max(1)
            else:
                best = np.zeros(len(r))
            target.append(torch.as_tensor(best, dtype=torch.float32))
            image_id += [iid] * len(r)
            if args.keep_rles:
                for x in r:
                    x = dict(x); x["counts"] = x["counts"].decode("ascii"); rles.append(x)
            if k % 200 == 0:
                print(f"  {k}/{len(order)} images  {time.time() - t0:.0f}s", flush=True)

    out = {"features": torch.cat(feats), "names": FEATURES, "target": torch.cat(target),
           "image_id": torch.as_tensor(image_id), "img_ids": [ds.index[n]["id"] for n in order],
           "ckpt": args.ckpt, "data": args.data, "split": args.split,
           "ann_file": str(Path(args.data) / args.split / "_annotations.coco.json")}
    if args.keep_rles:
        out["rles"] = rles
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(order)} images, {len(out['target'])} detections, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
