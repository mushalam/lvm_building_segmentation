#!/usr/bin/env python3
"""Train a HYDRA-style context re-ranker on cached detections and score it.

Reads dumps from tools/dump_detections.py. A gradient-boosted regressor learns
each detection's true mask IoU from its own, context and image features
(HYDRA, arXiv 2609.20283, uses an MLP on a frozen model's cached outputs; trees
are the stronger default on tabular features and need no tuning loop). Masks,
boxes and the detection set never change: only the ranking does, as in
tools/oracle_rescore.py.

Every choice -- training source, feature set, scoring rule -- is made on v2
valid. Test is scored once, for the chosen settings and the baselines.

  sources   idf     92/93/94 tiles the detector never trained on
            valid   v2 valid itself, cross-fitted in 5 image folds (each valid
                    detection is scored by a model that never saw its image);
                    for test, a model fitted on all of valid
            both    idf + valid, cross-fitted the same way
  features  own | own+ctx | all (own+ctx+image)
  rules     pred, cls*pred, ms*pred, where pred is the re-ranker's IoU

    python tools/context_rerank.py --train dets_idf.pt --valid dets_valid.pt \\
        --test dets_test.pt --out rerank.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lvm.evaluate import load_gt, summarise
from tools.dump_detections import OWN, CTX, IMG

SETS = {"own": OWN, "own+ctx": OWN + CTX, "all": OWN + CTX + IMG}
KEYS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR")


def load(p):
    d = torch.load(p, weights_only=False)
    d["X"] = d["features"].numpy().astype(np.float32)
    d["y"] = d["target"].numpy()
    d["iid"] = d["image_id"].numpy()
    d["col"] = {n: i for i, n in enumerate(d["names"])}
    return d


def cols(d, names):
    return d["X"][:, [d["col"][n] for n in names]]


def fit(X, y, seed=0):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                      min_samples_leaf=100, l2_regularization=1.0,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=seed)
    return m.fit(X, y)


def ap(d, gt, score):
    res = [{"image_id": int(i), "category_id": 1, "segmentation": r, "score": float(s)}
           for i, r, s in zip(d["iid"], d["rles"], score)]
    m = summarise(gt, res, max_dets=300, img_ids=d["img_ids"])
    return {k: round(float(m[k]), 4) for k in KEYS}


def rules(d, pred):
    c, ms = d["X"][:, d["col"]["cls"]], d["X"][:, d["col"]["ms"]]
    pred = np.clip(pred, 0, 1)
    return {"pred": pred, "cls*pred": c * pred, "ms*pred": ms * pred}


def crossfit(d, names, extra=None, k=5):
    """Out-of-fold predictions on d, folds by image. extra: (X, y) always in training."""
    ids = np.unique(d["iid"]); rng = np.random.default_rng(0); rng.shuffle(ids)
    fold = {i: n % k for n, i in enumerate(ids)}
    f = np.array([fold[i] for i in d["iid"]])
    X, y = cols(d, names), d["y"]
    out = np.zeros(len(y), np.float32)
    for q in range(k):
        tr = f != q
        Xt, yt = X[tr], y[tr]
        if extra is not None:
            Xt, yt = np.concatenate([extra[0], Xt]), np.concatenate([extra[1], yt])
        out[~tr] = fit(Xt, yt).predict(X[~tr])
    return out


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--train", required=True, help="idf dump (held-out tiles)")
    ap_.add_argument("--valid", required=True)
    ap_.add_argument("--test", required=True)
    ap_.add_argument("--out", required=True)
    args = ap_.parse_args()

    tr, va, te = load(args.train), load(args.valid), load(args.test)
    gva, gte = load_gt(va["ann_file"]), load_gt(te["ann_file"])
    print(f"detections: train {len(tr['y'])}, valid {len(va['y'])}, test {len(te['y'])}", flush=True)
    out = {"dumps": {"train": args.train, "valid": args.valid, "test": args.test},
           "valid": {}, "test": {}}

    base_v = ap(va, gva, cols(va, ["ms"])[:, 0])
    out["valid"]["baseline (ms1 score)"] = base_v
    out["valid"]["oracle (ms*true IoU)"] = ap(va, gva, cols(va, ["ms"])[:, 0] * va["y"])
    head_mae = float(np.abs(cols(va, ["pred_iou"])[:, 0] - va["y"]).mean())
    out["valid"]["baseline (ms1 score)"]["iou_mae"] = round(head_mae, 4)
    print(f"valid baseline {base_v['AP']}  oracle {out['valid']['oracle (ms*true IoU)']['AP']}  "
          f"MaskIoU head mae {head_mae:.3f}", flush=True)

    best = None
    for src in ("idf", "valid", "both"):
        for fs, names in SETS.items():
            t0 = time.time()
            if src == "idf":
                pv = fit(cols(tr, names), tr["y"]).predict(cols(va, names))
            elif src == "valid":
                pv = crossfit(va, names)
            else:
                pv = crossfit(va, names, extra=(cols(tr, names), tr["y"]))
            mae = float(np.abs(np.clip(pv, 0, 1) - va["y"]).mean())
            for rule, s in rules(va, pv).items():
                m = ap(va, gva, s)
                key = f"{src} | {fs} | {rule}"
                out["valid"][key] = dict(m, iou_mae=round(mae, 4))
                print(f"  {key:28s} AP {m['AP']:.4f} ({m['AP'] - base_v['AP']:+.4f})  "
                      f"AP75 {m['AP75']:.4f}  AP_small {m['AP_small']:.4f}  "
                      f"mae {mae:.3f}  {time.time() - t0:.0f}s", flush=True)
                if best is None or m["AP"] > best[0]:
                    best = (m["AP"], src, fs, rule)

    _, src, fs, rule = best
    print(f"chosen on valid: {src} | {fs} | {rule}", flush=True)
    names = SETS[fs]
    if src == "idf":
        X, y = cols(tr, names), tr["y"]
    elif src == "valid":
        X, y = cols(va, names), va["y"]
    else:
        X, y = np.concatenate([cols(tr, names), cols(va, names)]), np.concatenate([tr["y"], va["y"]])
    pt = fit(X, y).predict(cols(te, names))
    out["chosen"] = {"source": src, "features": fs, "rule": rule}
    out["test"]["baseline (ms1 score)"] = ap(te, gte, cols(te, ["ms"])[:, 0])
    out["test"]["oracle (ms*true IoU)"] = ap(te, gte, cols(te, ["ms"])[:, 0] * te["y"])
    out["test"][f"chosen: {src} | {fs} | {rule}"] = dict(
        ap(te, gte, rules(te, pt)[rule]),
        iou_mae=round(float(np.abs(np.clip(pt, 0, 1) - te["y"]).mean()), 4))
    for k, v in out["test"].items():
        print(f"test {k:40s} AP {v['AP']:.4f}  AP75 {v['AP75']:.4f}  AP_small {v['AP_small']:.4f}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
