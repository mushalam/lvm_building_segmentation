#!/usr/bin/env python3
"""Recompute BatchNorm running statistics on frozen weights ("PreciseBN").

Wu & Johnson, "Rethinking 'Batch' in BatchNorm" (arXiv 2105.07576): running
averages lag the weights, and Mask R-CNN heads with BatchNorm trained at
~1 image per GPU lose ~9 mask AP using them. torchvision's v2 Mask R-CNN has
trainable BatchNorm in the backbone, FPN and both heads, and this project
trains at batch 2. In r9 and r10 the raw final-epoch weights collapse on test
(0.170, 0.162) while EMA weights -- which average the buffers too -- do not.

This keeps every weight fixed and re-estimates only the BatchNorm buffers,
as an exact cumulative average (momentum=None) over --n un-augmented
training tiles. By default (--mode train) the forward is the training one, so
the heads' statistics come from the sampled proposals the weights were fitted
to. Writes a checkpoint whose 'model' is the result; score it with lvm.score.

    python tools/precise_bn.py --ckpt runs/r10_jitter_up/last.pt --weights raw \\
        --out runs/r10_jitter_up/last_raw_precisebn.pt
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvm.data import BuildingDataset, collate
from lvm.train import build_model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=["model", "raw"], default="raw")
    ap.add_argument("--data", default="data/ign_building_v2/building")
    ap.add_argument("--n", type=int, default=600, help="training tiles to average over")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["eval", "train"], default="train",
                    help="train: full training-mode forward with targets (no backward), so "
                         "the heads see sampled proposals (512/image, 25%% positive) as in "
                         "training. eval: inference-mode proposals (post-NMS, mostly "
                         "background). eval was tried first on r10 and moved the heads' BN "
                         "statistics off what the weights were fitted to: EMA 0.2144 -> 0.1709")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    key = "model_raw" if args.weights == "raw" else "model"
    model = build_model(400, image_size=ck["image_size"], backbone=ck.get("backbone", "resnet50"),
                        scratch_heads=ck.get("scratch_heads", False))
    model.load_state_dict(ck[key]); model.cuda().eval()

    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = {i: (m.running_mean.clone(), m.running_var.clone()) for i, m in enumerate(bns)}
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()   # exact cumulative mean
    print(f"{len(bns)} BatchNorm layers reset; weights frozen", flush=True)

    ds = BuildingDataset(args.data, "train", train=True)        # no augmentation flags
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[:args.n].tolist()
    t0 = time.time()
    if args.mode == "train":
        model.train()                                           # sampled proposals, losses unused
    for k in range(0, len(idx), args.batch_size):
        batch = [ds[i] for i in idx[k:k + args.batch_size]]
        imgs = [b[0].cuda() for b in batch]
        if args.mode == "train":
            tg = [{q: v.cuda() for q, v in b[1].items()} for b in batch]
            with torch.amp.autocast("cuda"):                    # as in training
                model(imgs, tg)
        else:
            model(imgs)                                         # eval-mode proposals
        if k % 100 == 0:
            print(f"  {k}/{len(idx)} tiles  {time.time() - t0:.0f}s", flush=True)
    model.eval()

    shift = [float((m.running_mean - before[i][0]).abs().mean() /
                   (before[i][0].abs().mean() + 1e-8)) for i, m in enumerate(bns)]
    vr = [float(m.running_var.mean() / (before[i][1].mean() + 1e-8)) for i, m in enumerate(bns)]
    print(f"relative running-mean change: median {sorted(shift)[len(shift)//2]:.3f}, "
          f"max {max(shift):.3f}; running-var ratio new/old: median "
          f"{sorted(vr)[len(vr)//2]:.3f}, min {min(vr):.3f}, max {max(vr):.3f}")
    out = dict(ck); out["model"] = model.state_dict(); out.pop("model_raw", None)
    out["precise_bn"] = {"source": args.ckpt, "weights": key, "tiles": len(idx), "mode": args.mode}
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
