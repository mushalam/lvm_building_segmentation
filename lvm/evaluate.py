#!/usr/bin/env python3
"""COCO instance-segmentation scoring. One implementation, used everywhere.

The project this one forks from lost significant time to evaluation defects that
were invisible because scoring happened in two places with different defaults:

  * three detection caps in series (decoder queries, prediction dumper, COCOeval)
    where only the binding one was observable;
  * an offline scorer defaulting to a 0.5 confidence threshold, worth 7% AP,
    because COCO AP integrates precision over all recall levels and the
    low-confidence tail carries recall;
  * `params.maxDets = [1, 10, 300]` silently making pycocotools report AP = -1,
    since `_summarize` resolves its default of 100 by searching that list.

So: one function, called by training validation and by offline scoring alike,
with the caps and thresholds as explicit arguments and no defaults that differ
by caller.
"""
import copy
import json
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def summarise(coco_gt, results, iou_type="segm", max_dets=300, img_ids=None,
              quiet=True):
    """Score `results` (COCO dicts) against `coco_gt`. Returns a metrics dict.

    `max_dets` belongs to the dataset, not the metric: COCO's default of 100
    suits ~7 instances per image, LVIS raised it to 300 for dense scenes, and
    this data averages 77 with a maximum of 305.

    `img_ids` MUST be given whenever the predictions cover a subset of the
    ground-truth file. COCOeval otherwise evaluates every image in `coco_gt`,
    and the unscored ones contribute ground truth that nothing can match, so AP
    comes out multiplied by roughly (scored / total). Not hypothetical: the
    predecessor project shipped this for four full runs, and this file's own
    author reproduced it on the first validation pass here -- 200 images scored
    against 700 images of ground truth reported AP 0.054 where the true value
    was near 0.19. Defaulting to the ids present in `results` makes it hard to
    repeat; pass it explicitly when the intended set is known.
    """
    if not results:
        return {"AP": -1.0, "note": "no detections"}
    if img_ids is None:
        img_ids = sorted({r["image_id"] for r in results})

    buf = StringIO()
    with redirect_stdout(buf if quiet else None):
        # Deep copy: loadRes adds 'bbox' and 'area' to the dicts it is given, so
        # a second call on the same list takes a different branch and raises.
        # Mutating the caller's data is also just bad manners.
        coco_dt = coco_gt.loadRes(copy.deepcopy(list(results)))
        ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        ev.params.maxDets = [1, 10, int(max_dets)]
        ev.params.imgIds = list(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()

    # stats[0] is built by `_summarize(1)`, whose signature defaults to
    # maxDets=100 and resolves it by searching params.maxDets. With 100 absent
    # the lookup is empty and the mean of nothing is -1. Rebuild it explicitly.
    p = ev.params
    prec = ev.eval["precision"]                       # [T, R, K, A, M]
    aind = p.areaRngLbl.index("all")
    mind = len(p.maxDets) - 1
    sl = prec[:, :, :, aind, mind]
    ap = float(np.mean(sl[sl > -1])) if (sl > -1).any() else -1.0

    def at(iou=None, area="all"):
        a = p.areaRngLbl.index(area)
        s = prec[:, :, :, a, mind]
        if iou is not None:
            s = s[np.argmin(np.abs(p.iouThrs - iou))][None]
        return float(np.mean(s[s > -1])) if (s > -1).any() else -1.0

    rec = ev.eval["recall"]                           # [T, K, A, M]
    r = rec[:, :, aind, mind]
    return {"AP": ap, "AP50": at(0.5), "AP75": at(0.75),
            "AP_small": at(area="small"), "AP_medium": at(area="medium"),
            "AP_large": at(area="large"),
            "AR": float(np.mean(r[r > -1])) if (r > -1).any() else -1.0,
            "max_dets": int(max_dets), "n_results": len(results),
            "n_images_scored": len(img_ids)}


def load_gt(ann_file):
    buf = StringIO()
    with redirect_stdout(buf):
        return COCO(ann_file)


def self_test():
    """Score ground truth against itself: every metric must be 1.0.

    A harness that cannot score a perfect prediction perfectly is not measuring
    what it claims, and this catches cap and threshold mistakes immediately.
    """
    import sys
    gt_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not gt_file:
        print("usage: python -m lvm.evaluate <annotations.json>"); return 1
    gt = load_gt(gt_file)
    results = []
    for ann in gt.dataset["annotations"]:
        results.append({"image_id": ann["image_id"], "category_id": ann["category_id"],
                        "segmentation": ann["segmentation"], "score": 1.0})
    m = summarise(gt, results, max_dets=400)
    print(f"perfect-prediction self-test on {gt_file}")
    for k, v in m.items():
        print(f"  {k:12s} {v}")
    ok = m["AP"] > 0.99 and m["AP75"] > 0.99
    print("  PASS" if ok else "  FAIL — the harness cannot score a perfect prediction")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
