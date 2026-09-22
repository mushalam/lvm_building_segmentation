#!/usr/bin/env python3
"""Fine-tune Mask R-CNN for per-building instance segmentation.

Configuration is driven by the data, not by defaults. Measured on this split
(377,305 instances over 4,903 tiles):

    instances/image   mean 77, p95 185, max 305
    instance side     p5 11.5px, p25 30.2, median 54.1, p75 84.0, p95 175.2

Three defaults would silently cap us and are overridden:

  box_detections_per_img  100 -> 400   images hold up to 305 buildings
  rpn_post_nms_top_n      1000 -> 3000 proposals must outnumber instances
  anchor sizes            (32,64,128,256,512) -> (12,30,54,84,175), from the
                          box statistics above; the default's smallest anchor
                          is larger than our median building

Validation calls lvm.evaluate.summarise, the same function offline scoring uses.
"""
import argparse, json, time
from pathlib import Path

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from pycocotools import mask as mask_util

from lvm.data import BuildingDataset, collate
from lvm.evaluate import load_gt, summarise

ANCHORS = (12, 30, 54, 84, 175)


def anchor_sizes_for(image_size):
    """Anchors are in input-image pixels, so they scale with the input."""
    scale = image_size / 1024.0
    return tuple((round(a * scale),) for a in ANCHORS)


def build_model(detections_per_img=400, trainable_layers=5, image_size=1024,
                backbone="resnet50"):
    if backbone != "resnet50":
        from lvm.backbones import build_timm_maskrcnn
        return build_timm_maskrcnn(
            backbone, anchor_sizes_for(image_size),
            detections_per_img=detections_per_img, image_size=image_size)
    model = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT", weights_backbone=None,
        box_detections_per_img=detections_per_img,
        rpn_post_nms_top_n_train=3000, rpn_post_nms_top_n_test=3000,
        rpn_pre_nms_top_n_train=4000, rpn_pre_nms_top_n_test=4000,
        box_batch_size_per_image=512, rpn_batch_size_per_image=256,
        min_size=image_size, max_size=image_size,
        trainable_backbone_layers=trainable_layers,
    )
    # Anchors sized from the data. The FPN has five levels, so five tuples.
    # Anchors are in input-image pixels, so they scale with the input. Leaving
    # them fixed while upsampling would make every anchor too small by the
    # scale factor -- a silent mismatch that costs recall.
    sizes = anchor_sizes_for(image_size)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))

    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, 2)
    return model


@torch.no_grad()
def validate(model, loader, gt, device, max_dets):
    model.eval()
    results = []
    for imgs, targets in loader:
        outs = model([i.to(device) for i in imgs])
        for t, o in zip(targets, outs):
            iid = int(t["image_id"].item())
            masks = (o["masks"] > 0.5).squeeze(1).cpu().numpy().astype("uint8")
            for m, s in zip(masks, o["scores"].cpu().tolist()):
                rle = mask_util.encode(__import__("numpy").asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                results.append({"image_id": iid, "category_id": 1,
                                "segmentation": rle, "score": float(s)})
    # img_ids defaults to the ids present in `results`, so a subset validation
    # cannot be scored against the full ground truth. Run 1 shipped that bug for
    # its whole training and every metric came out scaled by 200/700.
    return summarise(gt, results, max_dets=max_dets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ign_building_v2/building")
    ap.add_argument("--out", default="runs/maskrcnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-dets", type=int, default=400)
    ap.add_argument("--image-size", type=int, default=1024,
                    help="input size fed to the model. Run 1 showed the binding "
                         "constraint is FEATURE resolution, not mask-head output: "
                         "a 41px building at stride 4 gives ~10x10 features "
                         "whatever the mask grid. Upsampling the input is the "
                         "direct way to give small objects more feature cells")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "dinov3_convnext_tiny",
                             "dinov3_convnext_small", "dinov3_convnext_base"],
                    help="resnet50 carries COCO-pretrained heads; the dinov3 "
                         "paths initialise heads randomly. See lvm/backbones.py")
    ap.add_argument("--val-images", type=int, default=200,
                    help="validation is expensive; a fixed prefix of valid/ is "
                         "enough to track progress. Final numbers come from "
                         "lvm.score on test/, never from this")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = BuildingDataset(args.data, "train", train=True)
    va = BuildingDataset(args.data, "valid", train=False)
    va.index = va.index[:args.val_images]
    print(f"train {len(tr):,} tiles ({tr.dropped:,} slivers dropped)  "
          f"valid {len(va):,} tiles")

    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, collate_fn=collate,
                                     pin_memory=True, drop_last=True,
                                     persistent_workers=args.workers > 0)
    vl = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, collate_fn=collate)

    model = build_model(args.max_dets, image_size=args.image_size,
                        backbone=args.backbone).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"backbone {args.backbone}: {ntr:.1f}M trainable params", flush=True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters, "
          f"{n_tr/1e6:.0f}M trainable")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    gt_valid = load_gt(str(Path(args.data) / "valid" / "_annotations.coco.json"))
    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for step, (imgs, targets) in enumerate(tl):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  ep{epoch} {step}/{len(tl)} loss {loss.item():.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

        m = validate(model, vl, gt_valid, device, args.max_dets)
        m.update(epoch=epoch, train_loss=running / max(1, len(tl)),
                 minutes=(time.time() - t0) / 60)
        history.append(m)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch}: AP {m['AP']:.4f} AP50 {m['AP50']:.4f} "
              f"AP75 {m['AP75']:.4f} AP_small {m['AP_small']:.4f} "
              f"({m['minutes']:.1f} min)", flush=True)
        ckpt = {"model": model.state_dict(), "epoch": epoch,
                "metrics": m, "anchors": ANCHORS,
                "image_size": args.image_size, "backbone": args.backbone}
        # Always keep the newest weights. `best` is chosen on --val-images,
        # a subsample whose epoch-to-epoch spread (+-0.07 AP at 200 images in
        # run 3) is far wider than the differences it is asked to arbitrate,
        # so it can lock onto an early lucky epoch and discard later, better
        # models. last.pt makes that recoverable: score both on the full split.
        torch.save(ckpt, out / "last.pt")
        if m["AP"] > best:
            best = m["AP"]
            torch.save(ckpt, out / "best.pt")
            print(f"  new best, saved", flush=True)
    print(f"done. best AP {best:.4f}")


if __name__ == "__main__":
    main()
