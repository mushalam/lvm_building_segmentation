#!/usr/bin/env python3
"""Upper bound on what rescoring detections could add (Mask Scoring R-CNN's oracle).

COCO AP depends on how detections are ranked. Mask Scoring R-CNN (Huang et al.,
CVPR 2019, arXiv 1903.00241) re-ranks by score x predicted mask IoU, and its
Table 7 reports the oracle -- the true mask IoU in place of the prediction --
at +2.2-2.6 AP over its learned version on COCO. This measures the same
ceiling for one checkpoint, before any training:

  model       the detector's own scores (the baseline)
  iou         score := true mask IoU with the best-matching ground truth
  score_iou   score := score x true mask IoU (a perfect MaskIoU head)

Masks, boxes and detection set are identical across the three; only the ranking
changes. The same lvm.evaluate.summarise scores all of them, over every image.

    python tools/oracle_rescore.py --ckpt runs/r10_jitter_up/best.pt --out oracle.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvm.data import BuildingDataset
from lvm.evaluate import load_gt, summarise
from lvm.train import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=["model", "raw"], default="model")
    ap.add_argument("--data", default="data/ign_building_v2/building")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key = "model_raw" if args.weights == "raw" else "model"
    model = build_model(400, image_size=ck["image_size"], backbone=ck.get("backbone", "resnet50"),
                        scratch_heads=ck.get("scratch_heads", False),
                        mask_scoring=ck.get("mask_scoring", False))
    model.load_state_dict(ck[key]); model.to(args.device).eval()
    gt = load_gt(str(Path(args.data) / args.split / "_annotations.coco.json"))
    ds = BuildingDataset(args.data, args.split, train=False)

    dets = []          # (image_id, rle, score, true_iou)
    t0 = time.time()
    for n in range(len(ds.index)):
        img, _ = ds[n]
        iid = ds.index[n]["id"]
        o = model([img.to(args.device)])[0]
        masks = (o["masks"][:, 0] > 0.5).cpu().numpy()
        scores = o["scores"].cpu().numpy()
        rles = [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False))
        info = gt.loadImgs(iid)[0]
        g = [gt.annToRLE(a) for a in anns]
        if rles and g:
            best = np.asarray(mask_util.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g)).max(1)
        else:
            best = np.zeros(len(rles))
        for r, s, b in zip(rles, scores, best):
            r = dict(r); r["counts"] = r["counts"].decode("ascii")
            dets.append((iid, r, float(s), float(b)))
        if n % 100 == 0:
            print(f"  {n}/{len(ds.index)} images  {time.time() - t0:.0f}s", flush=True)

    img_ids = [i["id"] for i in ds.index]
    rules = {"model": lambda s, b: s, "iou": lambda s, b: b, "score_iou": lambda s, b: s * b}
    out = {"checkpoint": args.ckpt, "weights": key, "split": args.split,
           "images": len(img_ids), "detections": len(dets), "rules": {}}
    for name, f in rules.items():
        res = [{"image_id": i, "category_id": 1, "segmentation": r, "score": f(s, b)}
               for i, r, s, b in dets]
        m = summarise(gt, res, max_dets=args.max_dets, img_ids=img_ids)
        out["rules"][name] = {k: m[k] for k in ("AP", "AP50", "AP75", "AP_small",
                                                 "AP_medium", "AP_large", "AR")}
        print(f"{name:10s} AP {m['AP']:.4f}  AP50 {m['AP50']:.4f}  AP75 {m['AP75']:.4f}  "
              f"AP_small {m['AP_small']:.4f}  AR {m['AR']:.4f}", flush=True)
    base = out["rules"]["model"]["AP"]
    for name in ("iou", "score_iou"):
        out["rules"][name]["delta_AP"] = out["rules"][name]["AP"] - base
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
