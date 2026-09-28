#!/usr/bin/env python3
"""Inference speed of a checkpoint, measured the same way as tools/bench_sam3.py.

Batch 1 on one GPU, the first --n tiles of the seeded 250-tile v2-test sample
(lvm.boundary.sample_ids), --warmup untimed images first, CUDA synchronised
around every timed region. Two timings per image:

  model       tensor on GPU -> detections with masks pasted at tile size
  end_to_end  PNG on disk -> binary masks on the CPU (load, convert, infer,
              threshold, copy back): what a caller actually waits for

    python -m lvm.bench --ckpt runs/stageB3_long/best.pt --out bench.json
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lvm.boundary import sample_ids
from lvm.evaluate import load_gt
from lvm.train import build_model


def stats(ms):
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p90_ms": float(np.percentile(a, 90)), "images_per_s": float(1000 / a.mean())}


@torch.inference_mode()
def run(model, paths, score_thresh, amp, warmup):
    model.roi_heads.score_thresh = score_thresh
    t_model, t_e2e, n_det = [], [], []
    for k, p in enumerate(paths):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rgb = np.asarray(Image.open(p).convert("RGB"))
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).cuda()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            o = model([x])[0]
        torch.cuda.synchronize(); t2 = time.perf_counter()
        masks = (o["masks"][:, 0] > 0.5).cpu().numpy()
        _ = o["scores"].float().cpu().numpy()
        t3 = time.perf_counter()
        if k >= warmup:
            t_model.append((t2 - t1) * 1000); t_e2e.append((t3 - t0) * 1000)
            n_det.append(len(masks))
    return {"model": stats(t_model), "end_to_end": stats(t_e2e),
            "detections_per_image": float(np.mean(n_det)), "images_timed": len(t_model)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/ign_building_v2/building")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args()

    gt = load_gt(str(Path(args.data) / "test" / "_annotations.coco.json"))
    ids = sample_ids(gt, 250)
    names = {i["id"]: i["file_name"] for i in gt.dataset["images"]}
    paths = [Path(args.data) / "test" / names[i] for i in ids]
    paths = paths[:args.warmup] + paths[:args.n]          # warm up on real tiles

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(400, image_size=ck.get("image_size", 1024),
                        backbone=ck.get("backbone", "resnet50"),
                        scratch_heads=ck.get("scratch_heads", False))
    model.load_state_dict(ck["model"]); model.cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())

    out = {"model": "Mask R-CNN (lvm)", "checkpoint": args.ckpt, "parameters": n_params,
           "input_px": ck.get("image_size", 1024), "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "tiles": args.n, "warmup": args.warmup, "runs": {}}
    for name, thresh, amp in [("eval, fp32, no threshold", 0.0, False),
                              ("deploy, fp32, score >= 0.5", 0.5, False),
                              ("eval, fp16 autocast, no threshold", 0.0, True)]:
        torch.cuda.reset_peak_memory_stats()
        r = run(model, paths, thresh, amp, args.warmup)
        r["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        out["runs"][name] = r
        print(f"{name:36s} model {r['model']['mean_ms']:7.1f} ms  end-to-end "
              f"{r['end_to_end']['mean_ms']:7.1f} ms  ({r['end_to_end']['images_per_s']:.2f} img/s)  "
              f"{r['detections_per_image']:.0f} dets  peak {r['peak_gpu_mem_gb']:.1f} GB", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
