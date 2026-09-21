#!/usr/bin/env python3
"""Instance geometry against candidate output strides. Run before configuring."""
import json, sys, math
import numpy as np

d = json.load(open(sys.argv[1]))
wh = np.array([[a["bbox"][2], a["bbox"][3]] for a in d["annotations"]])
side = np.sqrt(wh[:, 0] * wh[:, 1])
per = {}
for a in d["annotations"]:
    per[a["image_id"]] = per.get(a["image_id"], 0) + 1
counts = np.array([per.get(i["id"], 0) for i in d["images"]])

print(f"{len(d['annotations']):,} instances over {len(d['images']):,} images")
print(f"instances/image: mean {counts.mean():.1f} p95 {np.percentile(counts,95):.0f} max {counts.max()}")
print()
print("equivalent square side (px):")
for q in (5, 25, 50, 75, 95):
    print(f"  p{q:<3d} {np.percentile(side, q):6.1f}")
print()
print("mask pixels available per instance, by architecture:")
print(f"  {'':22s} {'p25':>8s} {'median':>8s} {'p75':>8s}")
for name, f in (("SAM3 global stride 3.5", lambda s: s / 3.5),
                ("Mask R-CNN 28x28 ROI",   lambda s: np.full_like(s, 28.0))):
    v = [f(np.percentile(side, q)) for q in (25, 50, 75)]
    print(f"  {name:22s} {v[0]:7.1f} {v[1]:8.1f} {v[2]:8.1f}")
print()
print("anchor sizes to cover p5..p95 (torchvision expects a tuple per FPN level):")
qs = [np.percentile(side, q) for q in (5, 25, 50, 75, 95)]
print("  ", tuple(int(round(q)) for q in qs))
print(f"aspect ratios: p10 {np.percentile(wh[:,0]/wh[:,1],10):.2f} "
      f"median {np.median(wh[:,0]/wh[:,1]):.2f} p90 {np.percentile(wh[:,0]/wh[:,1],90):.2f}")
