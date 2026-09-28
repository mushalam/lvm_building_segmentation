#!/usr/bin/env python3
"""Inference speed of a SAM 3 fine-tune, measured the same way as lvm.bench.

Run from the sam3-ft-EOSC checkout, in its own environment, so the model is
built and run exactly as that project's evaluation runs it:
sam3ft.predict.load_model + predict_one (bf16 autocast, text prompt
"building"). Batch 1, the same first --n tiles of the same seeded 250-tile
v2-test sample, --warmup untimed, CUDA synchronised around every timed region.

  model       set_image + set_text_prompt: tile -> mask probabilities on GPU
  end_to_end  PNG on disk -> binary masks on the CPU

    cd ~/work/sam3ft && ./.venv/bin/python /path/to/bench_sam3.py \
        --ckpt CKPT --arch ARCH.json --sam3-root sam3 --data V2_ROOT --out bench.json

The sample is recomputed here with the same seeded draw lvm.boundary uses
(numpy default_rng(0), 250 of the sorted test ids), so both benches time the
same tiles without importing lvm.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sample_ids(ann_file, n=250, seed=0):
    ids = sorted(i["id"] for i in json.loads(Path(ann_file).read_text())["images"])
    if len(ids) > n:
        ids = sorted(np.random.default_rng(seed).choice(ids, size=n, replace=False).tolist())
    return ids


def stats(ms):
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p90_ms": float(np.percentile(a, 90)), "images_per_s": float(1000 / a.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", help="arch.json the checkpoint was trained with")
    ap.add_argument("--sam3-root", required=True)
    ap.add_argument("--data", required=True, help="v2 root holding test/_annotations.coco.json")
    ap.add_argument("--concept", default="building")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args()

    from sam3ft.predict import load_model, predict_one
    from sam3.model.sam3_image_processor import Sam3Processor

    ann = Path(args.data) / "test" / "_annotations.coco.json"
    names = {i["id"]: i["file_name"] for i in json.loads(ann.read_text())["images"]}
    paths = [Path(args.data) / "test" / names[i] for i in sample_ids(ann)]
    paths = paths[:args.warmup] + paths[:args.n]

    spec = json.loads(Path(args.arch).read_text()) if args.arch else None
    model = load_model(args.ckpt, args.sam3_root, device="cuda", arch=spec)
    processor = Sam3Processor(model)
    n_params = sum(p.numel() for p in model.parameters())

    out = {"model": "SAM 3 fine-tune (sam3-ft-EOSC)", "checkpoint": args.ckpt,
           "arch": spec, "parameters": n_params, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "tiles": args.n, "warmup": args.warmup, "runs": {}}
    for name, thresh in [("eval, bf16, no threshold", 0.0), ("deploy, bf16, score >= 0.5", 0.5)]:
        torch.cuda.reset_peak_memory_stats()
        t_model, t_e2e, n_det = [], [], []
        for k, p in enumerate(paths):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            image = Image.open(p).convert("RGB"); image.load()
            torch.cuda.synchronize(); t1 = time.perf_counter()
            # predict_one returns numpy, so its tail includes the copy back;
            # "model" here therefore includes the probabilities' transfer.
            probs, _, scores = predict_one(processor, image, args.concept, thresh)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            masks = probs > 0.5
            t3 = time.perf_counter()
            if k >= args.warmup:
                t_model.append((t2 - t1) * 1000); t_e2e.append((t3 - t0) * 1000)
                n_det.append(len(scores))
        r = {"model": stats(t_model), "end_to_end": stats(t_e2e),
             "detections_per_image": float(np.mean(n_det)), "images_timed": len(t_model),
             "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        out["runs"][name] = r
        print(f"{name:36s} model {r['model']['mean_ms']:7.1f} ms  end-to-end "
              f"{r['end_to_end']['mean_ms']:7.1f} ms  ({r['end_to_end']['images_per_s']:.2f} img/s)  "
              f"{r['detections_per_image']:.0f} dets  peak {r['peak_gpu_mem_gb']:.1f} GB", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
