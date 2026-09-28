#!/usr/bin/env python3
"""Segment buildings in images with a trained checkpoint, for looking at.

Writes, per image:
  <name>_overlay.png     the image with each building filled and outlined
  <name>_buildings.json  one polygon per building, in pixel coordinates

    python -m lvm.predict --ckpt runs/stageB4/last.pt --image tile.png \
        --out-dir predictions/

--image takes files or directories. Images larger than --tile are cut into
tile-sized pieces and predicted piece by piece, since the model was trained on
1024 px IGN tiles and resizing a large orthophoto down to one tile would
shrink every building. Pieces do not overlap, so a building crossing a piece
boundary comes out as two halves.

The model only sees pixels, so imagery at a very different ground resolution
from IGN's will look like buildings of the wrong size to it.

--score-thresh is for display only. Scoring (lvm.score) uses no threshold.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from lvm.train import build_model

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_model(ckpt_path, weights, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "model_raw" if weights == "raw" else "model"
    assert key in ck, f"{ckpt_path} has no '{key}' (trained without --ema?)"
    model = build_model(400, image_size=ck.get("image_size", 1024),
                        backbone=ck.get("backbone", "resnet50"),
                        scratch_heads=ck.get("scratch_heads", False))
    model.load_state_dict(ck[key])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_tile(model, rgb, device, score_thresh):
    """Masks (N,H,W bool) and scores for one tile, highest score first."""
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).to(device)
    o = model([t])[0]
    keep = o["scores"] >= score_thresh
    masks = (o["masks"][keep, 0] > 0.5).cpu().numpy()
    return masks, o["scores"][keep].cpu().numpy()


def predict_image(model, rgb, device, score_thresh, tile):
    h, w = rgb.shape[:2]
    out = []                                   # (mask as full-image slice, score, y0, x0)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            piece = rgb[y0:y0 + tile, x0:x0 + tile]
            masks, scores = predict_tile(model, piece, device, score_thresh)
            out += [(m, s, y0, x0) for m, s in zip(masks, scores)]
    return out


def polygon(mask):
    """Largest outer contour of a mask as [[x, y], ...], or None."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c.reshape(-1, 2).tolist() if len(c) >= 3 else None


def render(rgb, dets, alpha=0.45, seed=0):
    rng = np.random.default_rng(seed)
    over = rgb.astype(np.float32).copy()
    edges = []
    for m, _, y0, x0 in dets:
        colour = rng.integers(60, 256, 3).astype(np.float32)
        region = over[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        region[m] = (1 - alpha) * region[m] + alpha * colour
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        edges += [c + np.array([x0, y0]) for c in cs]
    img = over.clip(0, 255).astype(np.uint8)
    cv2.drawContours(img, edges, -1, (255, 255, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", nargs="+", required=True, help="files or directories")
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--weights", choices=["model", "raw"], default="model",
                    help="for an --ema checkpoint: averaged ('model') or raw weights")
    ap.add_argument("--score-thresh", type=float, default=0.5,
                    help="display threshold; lower shows more, and more false positives")
    ap.add_argument("--tile", type=int, default=1024,
                    help="piece size for large images, in input pixels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = []
    for p in map(Path, args.image):
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in EXTS) if p.is_dir() else [p]
    assert paths, "no images found"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    model, ck = load_model(args.ckpt, args.weights, args.device)
    print(f"{args.ckpt} (epoch {ck.get('epoch', '?')}, weights '{args.weights}', "
          f"input {ck.get('image_size', 1024)} px) on {args.device}")

    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        dets = predict_image(model, rgb, args.device, args.score_thresh, args.tile)
        Image.fromarray(render(rgb, dets)).save(out / f"{p.stem}_overlay.png")
        feats = []
        for m, s, y0, x0 in dets:
            poly = polygon(m)
            if poly:
                feats.append({"score": round(float(s), 4), "area_px": int(m.sum()),
                              "polygon": [[x + x0, y + y0] for x, y in poly]})
        (out / f"{p.stem}_buildings.json").write_text(json.dumps(
            {"image": str(p), "width": rgb.shape[1], "height": rgb.shape[0],
             "checkpoint": args.ckpt, "weights": args.weights,
             "score_thresh": args.score_thresh, "buildings": feats}))
        print(f"  {p.name}: {len(feats)} buildings -> {out / (p.stem + '_overlay.png')}")


if __name__ == "__main__":
    main()
