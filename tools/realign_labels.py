#!/usr/bin/env python3
"""Re-align footprint labels per building, where two models agree they are offset.

tools/label_offset.py found that ground-truth masks sit a median 4.5 px from
where models put the building, that the offset is per building (one shift per
tile explains 6% of it, so a per-tile aligner such as Align and Segment, arXiv
2607.10841, cannot fix it), and that two independently trained models (ms1,
ms2) agree on the shift far above chance. This moves each label by the shift
the two models agree on -- OMAF's idea (CVPR 2026: object-level offsets from
self-alignment) with a two-model agreement test instead of a learned regressor.

A building is moved only when, for both models, its best shift is found, the
two shifts lie within --agree px of each other, their mean length is at least
--min-shift px, and the mean IoU gain is at least --min-gain. It is moved by
the rounded mean shift, as a pure translation: shape, parcel splits and
missing buildings are untouched. Masks shifted partly off the tile are cropped;
those left under --min-area px are dropped.

Writes <out-root>/<split>/_annotations.coco.json with the tile images
symlinked (the source tree is never written to), and <out-root>/<split>_realign.json
with the counts.

    python tools/realign_labels.py --data data/ign_building_v2/building --split train \\
        --dets a.pt b.pt --out-root data_local/ign_building_v2_aligned/building
"""
import argparse, json, os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvm.evaluate import load_gt
from tools.label_offset import work


def shifts(dets, gt, min_score, workers):
    d = torch.load(dets, weights_only=False)
    ms = d["features"][:, d["names"].index("ms")].numpy()
    by_img = {}
    for i, r, s in zip(d["image_id"].tolist(), d["rles"], ms):
        if s >= min_score:
            by_img.setdefault(i, []).append(r)
    jobs = []
    for iid in d["img_ids"]:
        info = gt.loadImgs(iid)[0]
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False))
        jobs.append((iid, info["height"], info["width"],
                     [(a["id"], gt.annToRLE(a), a["area"]) for a in anns], by_img.get(iid, [])))
    with Pool(workers) as pool:
        rows = [x for part in pool.imap_unordered(work, jobs, chunksize=4) for x in part]
    return {r[1]: r for r in rows}, set(d["img_ids"])


def translate(m, dy, dx):
    out = np.zeros_like(m)
    H, W = m.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = m[ys, xs]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--dets", nargs=2, required=True, help="two dumps (--keep-rles) of the whole split")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--agree", type=float, default=2.0, help="max px between the two models' shifts")
    ap.add_argument("--min-shift", type=float, default=3.0)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--min-area", type=float, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    src = Path(args.data) / args.split
    ann_file = src / "_annotations.coco.json"
    gt = load_gt(str(ann_file))
    a, ia = shifts(args.dets[0], gt, args.min_score, args.workers)
    b, ib = shifts(args.dets[1], gt, args.min_score, args.workers)
    doc = json.loads(ann_file.read_text())
    assert ia == ib == {i["id"] for i in doc["images"]}, "both dumps must cover the whole split"

    moved, dropped, lens, gains = 0, 0, [], []
    out_anns = []
    for ann in doc["annotations"]:
        ra, rb = a.get(ann["id"]), b.get(ann["id"])
        if ra and rb:
            va, vb = np.array(ra[3:5], float), np.array(rb[3:5], float)
            v = (va + vb) / 2
            gain = ((ra[6] - ra[5]) + (rb[6] - rb[5])) / 2
            if (np.hypot(*(va - vb)) <= args.agree and np.hypot(*v) >= args.min_shift
                    and gain >= args.min_gain):
                dy, dx = int(round(v[0])), int(round(v[1]))
                m = translate(mask_util.decode(gt.annToRLE(ann)), dy, dx)
                if m.sum() < args.min_area:
                    dropped += 1
                    continue
                r = mask_util.encode(np.asfortranarray(m))
                ann = dict(ann, segmentation={"size": list(r["size"]), "counts": r["counts"].decode("ascii")},
                           area=float(mask_util.area(r)), bbox=[float(x) for x in mask_util.toBbox(r)],
                           realign_shift=[dy, dx])
                moved += 1; lens.append(np.hypot(dy, dx)); gains.append(gain)
        out_anns.append(ann)

    dst = Path(args.out_root) / args.split
    dst.mkdir(parents=True, exist_ok=True)
    for im in doc["images"]:
        link = dst / im["file_name"]
        if not link.exists():
            os.symlink(os.path.realpath(src / im["file_name"]), link)
    doc["annotations"] = out_anns
    doc.setdefault("info", {})["realigned"] = {
        "source": str(ann_file), "dets": args.dets, "agree_px": args.agree,
        "min_shift_px": args.min_shift, "min_gain": args.min_gain}
    (dst / "_annotations.coco.json").write_text(json.dumps(doc))
    stats = {"split": args.split, "buildings": len(doc["annotations"]) + dropped,
             "measured_by_both": len(set(a) & set(b)), "moved": moved,
             "moved_share": moved / max(len(out_anns) + dropped, 1), "dropped_off_tile": dropped,
             "shift_px_median": float(np.median(lens)) if lens else 0.0,
             "shift_px_p90": float(np.percentile(lens, 90)) if lens else 0.0,
             "iou_gain_mean_moved": float(np.mean(gains)) if gains else 0.0,
             **{k: v for k, v in vars(args).items() if k != "workers"}}
    (Path(args.out_root) / f"{args.split}_realign.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
