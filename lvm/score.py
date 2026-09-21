#!/usr/bin/env python3
"""Score a checkpoint on a split. This produces the numbers that get quoted.

Deliberately uses the same `summarise` as training validation, so the two cannot
drift apart — the defect that cost the predecessor project 7% AP and a wrong
paragraph in a paper.

No confidence threshold. COCO expects every detection submitted, ranked by
score, with maxDets doing the capping; discarding the low-confidence tail throws
away recall the metric would have credited.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from lvm.data import BuildingDataset, collate
from lvm.evaluate import load_gt, summarise
from lvm.train import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/ign_building_v2/building")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-dets", type=int, default=300,
                    help="300 matches what the comparison target was scored at")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(max(args.max_dets, 400)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    ds = BuildingDataset(args.data, args.split, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, collate_fn=collate)
    gt = load_gt(str(Path(args.data) / args.split / "_annotations.coco.json"))

    results = []
    for n, (imgs, targets) in enumerate(dl):
        outs = model([i.to(device) for i in imgs])
        for t, o in zip(targets, outs):
            iid = int(t["image_id"].item())
            masks = (o["masks"] > 0.5).squeeze(1).cpu().numpy().astype("uint8")
            for m, s in zip(masks, o["scores"].cpu().tolist()):
                rle = mask_util.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                results.append({"image_id": iid, "category_id": 1,
                                "segmentation": rle, "score": float(s)})
        if n % 50 == 0:
            print(f"  {n * args.batch_size}/{len(ds)} images", flush=True)

    m = summarise(gt, results, max_dets=args.max_dets)
    m["checkpoint"] = args.ckpt
    m["split"] = args.split
    m["images"] = len(ds)
    print(json.dumps(m, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
