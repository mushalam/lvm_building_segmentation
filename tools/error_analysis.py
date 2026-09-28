"""Where does B3 lose AP? Stage-by-stage error analysis on the seeded
250-image v2-test sample (the one lvm.boundary uses), CPU.

Writes runs/stageB4_merged_ema/error_analysis_B3.json and prints a report.
"""
import collections, json, sys, time
import numpy as np
import torch
from pycocotools import mask as mu
from torchvision.ops import box_iou

from lvm.boundary import sample_ids
from lvm.data import BuildingDataset
from lvm.evaluate import load_gt, summarise
from lvm.train import build_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/stageB3_long/best.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/stageB4_merged_ema/error_analysis_B3.json"
V2 = "data/ign_building_v2/building"
NMS = [0.5, 0.6, 0.7]
CONF = 0.5              # "confident" detections, for error typing
torch.set_grad_enabled(False)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = build_model(400, image_size=ck["image_size"]); model.load_state_dict(ck["model"]); model.eval()
gt = load_gt(f"{V2}/test/_annotations.coco.json")
ids = set(sample_ids(gt, 250))
ds = BuildingDataset(V2, "test", train=False)
ds.index = [i for i in ds.index if i["id"] in ids]


def rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return r


def size_bucket(a):
    return "small" if a < 32 ** 2 else ("medium" if a < 96 ** 2 else "large")


results = {t: [] for t in NMS}
C = collections.Counter()               # global counters
by = collections.defaultdict(collections.Counter)   # breakdowns
tp_mask_iou, tp_box_iou = [], []
t0 = time.time()
for n, info in enumerate(ds.index):
    img, tg = ds[n]
    iid = info["id"]
    H, W = img.shape[1:]
    gmask = tg["masks"].numpy().astype(bool)
    gbox = tg["boxes"]
    G = len(gbox)

    imgs, _ = model.transform([img])
    feats = model.backbone(imgs.tensors)
    props, _ = model.rpn(imgs, feats)
    scale = imgs.image_sizes[0][0] / H
    # 1. proposal recall: does any of the RPN's proposals cover each building?
    if G:
        piou = box_iou(gbox * scale, props[0]).max(1).values
        C["gt"] += G
        C["prop_iou50"] += int((piou >= 0.5).sum()); C["prop_iou70"] += int((piou >= 0.7).sum())

    for t in NMS:
        model.roi_heads.nms_thresh = t
        det, _ = model.roi_heads(feats, props, imgs.image_sizes)
        det = model.transform.postprocess(det, imgs.image_sizes, [(H, W)])[0]
        pm = (det["masks"][:, 0] > 0.5).numpy()
        sc = det["scores"].numpy()
        prles = [rle(m) for m in pm]
        for r, s in zip(prles, sc):
            rr = dict(r); rr["counts"] = rr["counts"].decode()
            results[t].append({"image_id": iid, "category_id": 1, "segmentation": rr, "score": float(s)})
        if t != 0.5 or not G:
            continue

        # ---- error analysis at the default NMS 0.5 ----
        grles = [rle(m) for m in gmask]
        garea = gmask.reshape(G, -1).sum(1).astype(float)
        parea = pm.reshape(len(pm), -1).sum(1).astype(float)
        iou = np.asarray(mu.iou(prles, grles, [0] * G)).reshape(len(prles), G) if len(prles) else np.zeros((0, G))
        inter = iou * (parea[:, None] + garea[None, :]) / (1 + iou)      # |P and G|
        cov_g = inter / np.maximum(garea[None, :], 1)                    # share of G covered by P
        in_p = inter / np.maximum(parea[:, None], 1)                     # share of P inside G

        # 2. ceiling: any detection (of 400) with IoU >= .5, ignoring score/matching
        C["ceiling50"] += int((iou.max(0) >= 0.5).sum()) if len(iou) else 0

        # 3. COCO-style greedy matching at .5, highest score first, top 300
        order = np.argsort(-sc)[:300]
        gfree = np.ones(G, bool); pmatch = {}
        for i in order:
            row = np.where(gfree, iou[i], -1)
            j = int(row.argmax())
            if row[j] >= 0.5:
                gfree[j] = False; pmatch[i] = j
        tp = ~gfree
        C["tp50"] += int(tp.sum())

        # 4. why were the unmatched buildings missed?
        conf = sc >= CONF
        edge = ((gbox[:, 0] <= 2) | (gbox[:, 1] <= 2) | (gbox[:, 2] >= W - 2) | (gbox[:, 3] >= H - 2)).numpy()
        dens = "<30" if G < 30 else ("30-80" if G < 80 else ("80-150" if G < 150 else "150+"))
        for j in range(G):
            sb = size_bucket(garea[j])
            by["size_total"][sb] += 1; by["edge_total"]["edge" if edge[j] else "interior"] += 1
            by["density_total"][dens] += 1
            if tp[j]:
                by["size_tp"][sb] += 1; by["edge_tp"]["edge" if edge[j] else "interior"] += 1
                by["density_tp"][dens] += 1
                continue
            best = iou[:, j].max() if len(iou) else 0
            if best >= 0.3:
                why = "poor outline (best IoU .3-.5)"
            else:
                big = np.where(conf & (cov_g[:, j] >= 0.5))[0]
                if len(big):
                    i = big[np.argmax(cov_g[big, j])]
                    others = int(((cov_g[i] >= 0.5).sum()) - 1)
                    why = "merged with a neighbour" if others >= 1 else "inside an oversized detection"
                else:
                    frags = np.where(conf & (in_p[:, j] >= 0.7))[0]
                    if len(frags) >= 2 and cov_g[frags, j].sum() >= 0.5:
                        why = "split into pieces"
                    elif best >= 0.1:
                        why = "weak overlap only (IoU .1-.3)"
                    else:
                        why = "not detected"
            by["miss_reason"][why] += 1
            by["miss_reason_size"][f"{sb}: {why}"] += 1

        # 5. confident false positives
        for i in np.where(conf)[0]:
            if i in pmatch:
                continue
            C["conf_fp"] += 1
            if (cov_g[i] >= 0.5).sum() >= 2:
                why = "covers 2+ buildings (merge)"
            elif (in_p[i] >= 0.7).any():
                why = "fragment of one building"
            elif iou[i].max() >= 0.1:
                why = "mislocalised"
            else:
                why = "background / unlabelled"
            by["fp_reason"][why] += 1
        C["conf_det"] += int(conf.sum()); C["conf_tp"] += sum(1 for i in pmatch if conf[i])

        # 6. box vs mask quality on true positives
        pb = det["boxes"]
        for i, j in pmatch.items():
            tp_mask_iou.append(float(iou[i, j]))
            tp_box_iou.append(float(box_iou(pb[i:i + 1], gbox[j:j + 1])[0, 0]))

    if n % 25 == 0:
        print(f"  {n}/{len(ds.index)}  {time.time() - t0:.0f}s", flush=True)

model.roi_heads.nms_thresh = 0.5
img_ids = [i["id"] for i in ds.index]
aps = {str(t): summarise(gt, results[t], max_dets=300, img_ids=img_ids) for t in NMS}
tm, tb = np.array(tp_mask_iou), np.array(tp_box_iou)
rep = {
    "checkpoint": CKPT, "images": len(img_ids), "gt": C["gt"],
    "proposal_recall_50": C["prop_iou50"] / C["gt"], "proposal_recall_70": C["prop_iou70"] / C["gt"],
    "detection_ceiling_50": C["ceiling50"] / C["gt"], "matched_recall_50": C["tp50"] / C["gt"],
    "conf_precision": C["conf_tp"] / max(C["conf_det"], 1), "conf_detections": C["conf_det"],
    "miss_reason": dict(by["miss_reason"]), "miss_reason_size": dict(by["miss_reason_size"]),
    "fp_reason": dict(by["fp_reason"]),
    "recall_by_size": {k: by["size_tp"][k] / v for k, v in by["size_total"].items()},
    "count_by_size": dict(by["size_total"]),
    "recall_edge": {k: by["edge_tp"][k] / v for k, v in by["edge_total"].items()},
    "count_edge": dict(by["edge_total"]),
    "recall_by_density": {k: by["density_tp"][k] / v for k, v in by["density_total"].items()},
    "tp_mask_iou_mean": float(tm.mean()), "tp_box_iou_mean": float(tb.mean()),
    "tp_share_mask_iou_ge_75": float((tm >= 0.75).mean()),
    "tp_share_box_iou_ge_75": float((tb >= 0.75).mean()),
    "tp_mask_iou_when_box_iou_ge_85": float(tm[tb >= 0.85].mean()) if (tb >= 0.85).any() else None,
    "tp_share_box_iou_ge_85": float((tb >= 0.85).mean()),
    "ap_by_nms": {t: {k: a[k] for k in ("AP", "AP50", "AP75", "AP_small", "AR")} for t, a in aps.items()},
}
json.dump(rep, open(OUT, "w"), indent=2)
print(json.dumps(rep, indent=2))
