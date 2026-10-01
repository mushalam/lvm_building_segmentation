# Experiments

This document lists every training run: the question it tested, its exact
arguments, and what it showed. Each run's arguments are in `runs/<run>/args.json`
on the team server. Its histories and scores are in `runs/<run>/` or `results/` in
this repo. The commit that recorded each result carries the full reasoning; search
`git log` for the run name.

Common to all runs unless noted:
- torchvision `maskrcnn_resnet50_fpn_v2` with COCO weights
- data-derived anchors (12, 30, 54, 84, 175 px at 1024), 400 detections per tile,
  RPN post-NMS 3000
- SGD with momentum 0.9 and weight decay 1e-4, a OneCycle schedule (10% warm-up)
  and AMP
- training data `data/ign_building_v2/building`

**Two splits appear below.**
- **valid** is the full 700-tile v2 valid split. It is the only split with numbers
  for every run.
- **test** is the 1,402-tile v2 test split. It is the verdict, and was scored for
  the later runs.

All scores come from `lvm/score.py`: segm AP, maxDets 300.
- **Scoring before commit e279fab:** detections scoring under 0.05 were dropped
  inside the model. Fixing that moved run 2 on valid from 0.19476 to 0.1955, so
  older numbers stand.
- **best / last:** `best.pt` is the epoch chosen on the 200-tile validation
  prefix. `last.pt` is the final epoch.

## Summary

| run | dir | question | valid best / last | test | verdict |
|---|---|---|---|---|---|
| 1 | `maskrcnn_v1` | Does a per-ROI 28x28 mask beat SAM 3's global grid? | 0.1789 | | No. Features are the limit at stride 4, not the mask head |
| 2 | `maskrcnn_v2_hires` | Does feature resolution matter? Input 2048 px | 0.1948 | 0.2006 | Yes, +0.016. The strongest simple recipe |
| 3 | `maskrcnn_v3_long` | Longer training (30 epochs, lr 3.5e-3) | 0.1890 | | Void: two variables changed, and it selected a lucky epoch 4 |
| 4 | `maskrcnn_v4` | Run 3 repeated at 20 epochs | 0.1874 / 0.1792 | | Still confounded (lr and epochs) |
| 5 | `maskrcnn_v5_lr5e3` | Run 2 at 20 epochs, the clean test | 0.1896 / 0.1845 | | Longer training overfits (train loss −17%, AP down) |
| 6 | `maskrcnn_v6_dinov3t` | DINOv3 ConvNeXt-T trunk with random heads | 0.1708 / 0.1674 | | Worse, but confounded with head initialisation |
| 7 | `maskrcnn_v7_d4` | Run 5 plus D4 train-time augmentation | 0.1920 / 0.1482 | | Hurts. best.pt is flattered by a noisy run |
| 8 | `maskrcnn_v8_scratch` | Control for 6: ImageNet R50 with random heads | 0.1734 / 0.1477 | | COCO detector heads carry the transfer, not the trunk |
| A → B | `pretrainA_public` → `stageB_ign` | Pretrain on a public building corpus, then run 2 | 0.1932 / 0.1761 | | Faster convergence, same endpoint |
| A2 → B2 | `pretrainA2_dense` → `stageB2_ign` | Same, on a dense-tile subset | 0.1905 / 0.1905 | | Density does not explain the null result |
| A2 → B3 | `pretrainA2_dense` → `stageB3_long` | B2 at 20 epochs (single variable vs run 5) | 0.1954 / 0.1814 | 0.2010 | Pretraining worth about +0.005 |
| merged | `merged_r1_coco` | Run 2 on v2 + D001 (8x the data), 6 epochs | n/a†‡ | 0.1440 / **0.1544** | 90% rural data hurts dense Paris by −0.046 |
| B4 | `stageB4_merged_ema` | Initialise from merged, B3 recipe, EMA | n/a†‡ | 0.2011 / 0.1942 (raw 0.1899) | Ties B3. EMA +0.004. French pretraining no gain |
| r9 | `r9_aug_long` | Run 2 + scale jitter x0.75-1.33 + background copy-paste + EMA, 36 epochs | 0.2146 (ep16) / 0.2114 (ep36), 200-tile subset | **0.2070** / best.pt 0.2016 (raw last 0.1702) | **New best, +0.006.** Augmentation makes the long schedule pay off. AP75 +0.012, matched 62→65.6%, but AP_small 0.034→0.023. EMA essential (+0.037 over raw) |
| r10 | `r10_jitter_up` | r9 with enlarge-only jitter (x1.0-1.33): is the x0.75 shrink what cost AP_small? | 0.2303 (ep30) / 0.2268 (ep36), 200-tile subset | **0.2177** / last 0.2144 (raw last 0.1623) | **New best, +0.011 over r9.** AP_small back to 0.030 (r9 0.023), AP50 +0.026. EMA essential again |
| r11 | `r11_paris_idf` | r10's recipe on v2 + 92/93/94 IRC (13,032 tiles), 14 epochs = matched steps (~91k vs 88k) | *running* | | Does more data with the test set's own label convention help? |

† These runs are initialised from, or trained on, merged data, which contains 492
of v2 valid's 700 tiles. Their v2-valid numbers are inflated and not comparable.

‡ **The merged dataset mixes two imagery products.** v2 and its test split are BD
ORTHO **IRC** (false-colour infrared, vegetation red). D001, which is 90% of merged
train, is BD ORTHO **RVB** (natural colour). The team-share France pipeline
(`create_instance_seg_dataset_france_split_v3.py`, `download_ign_ten_split.py`)
requests `RVB-0M20_JP2-E080`. A side-by-side render of the tiles confirms it, as
do pixel statistics: 15.6% red-dominant pixels in v2 against 3.6% in D001.

This changes how two results should be read:
- **merged's −0.046** is confounded between "rural data" and "a different sensor
  product".
- **B4's "French pretraining gains nothing"** tested pretraining mostly on
  natural-colour imagery, not on IRC.

Neither is evidence against more *IRC* data. That is what the 92/93/94 build uses.


**Reference: SAM 3, full fine-tune (sam3-ft-EOSC).** It scores 0.2383 AP on v2
valid, with mask IoU 0.7341. It has no v2 test score.

## Commands

The commands below use the flags each run recorded. Stages A and B need the
public-corpus data under `data_local/`.

```bash
# run 1
python -m lvm.train --out runs/maskrcnn_v1 --epochs 15 --batch-size 4 --lr 0.005
# run 2  (and run 5 = the same with --epochs 20)
python -m lvm.train --out runs/maskrcnn_v2_hires --epochs 12 --batch-size 2 --lr 0.005 --image-size 2048
# runs 3 / 4
python -m lvm.train --out runs/maskrcnn_v3_long --epochs 30 --lr 0.0035 --image-size 2048
# run 6 / 7 / 8: run 5 plus one of
#   --backbone dinov3_convnext_tiny  |  --d4  |  --scratch-heads
# stage A (public corpus, 1024 px), then B / B2 / B3
python -m lvm.train --data data_local/pretrain_buildings --out runs/pretrainA_public --epochs 10 --batch-size 4
python -m lvm.train --data data_local/pretrain_dense     --out runs/pretrainA2_dense  --epochs 10 --batch-size 4
python -m lvm.train --out runs/stageB3_long --epochs 20 --image-size 2048 --init-from runs/pretrainA2_dense/last.pt
# merged, then B4
python -m lvm.train --data data_local/ign_building_merged/building --out runs/merged_r1_coco --epochs 6 --image-size 2048
python -m lvm.train --out runs/stageB4_merged_ema --epochs 20 --image-size 2048 \
    --init-from runs/merged_r1_coco/last.pt --ema 0.9998
# r9 (launch with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
python -m lvm.train --out runs/r9_aug_long --epochs 36 --image-size 2048 --workers 16 --val-every 2 \
    --ema 0.9998 --scale-jitter 0.75,1.33 --copy-paste 30
```

**The stage A corpus** is a public Roboflow building instance-segmentation export
of humanitarian and conflict-monitoring imagery (Tripoli, Kherson, Donetsk,
Mekele, Mykolaiv, Kharkiv), with a little Inria. The `tools/` scripts prepare it:
1. `remap_category.py` changes its `category_id` from 0 to 1, to match IGN.
2. `filter_empty.py` drops the 12% of tiles with no buildings.
3. `filter_density.py`, with a threshold of at least 20 instances per tile, keeps
   1,465 tiles for A2.

## Diagnostics (not training runs)

**TTA.** This is B3 on the seeded 250-tile v2-test sample, on CPU, with
`--sample-only`.

| setting | AP | AP75 |
|---|---|---|
| plain | 0.1969 | 0.1338 |
| `--tta d4`, masks fused | 0.1920 | 0.1192 |
| `--tta d4-scores`, scores fused | 0.1958 | 0.1309 |

TTA hurts in both forms. Mask R-CNN splits and merges adjacent buildings
differently in each view.

**NMS threshold** (`tools/error_analysis.py`):

| NMS | AP |
|---|---|
| 0.5 | 0.1969 |
| 0.6 | 0.1972 |
| 0.7 | 0.1942 |

Not a lever.

**Error analysis of B3.** From `runs/stageB4_merged_ema/error_analysis_B3.json`,
over 20,669 buildings:
- RPN recall at IoU 0.5 is 80.4%. Detection ceiling 64.5%. Matched 62.7%.
- Misses, as shares of all buildings:
  - poor outline, best IoU 0.3-0.5: 15%
  - not detected, 94% of them small: 10%
  - merged with a neighbour: 8%
- Recall on small buildings is 31%.

**Mask-grid ceiling** (`tools/mask_ceiling.py`). With perfect boxes, the 28x28 grid
reaches IoU 0.928 (small buildings 0.879). A 56x56 grid reaches 0.926.

**Inference speed** (`runs/bench/`). See the README.

## r9 in detail

Run 2's recipe with scale jitter (×0.75–1.33, crop/pad back to the tile),
background-only copy-paste (up to 30 buildings, capped at the densest real
tile), and EMA at 0.9998, over 36 epochs with validation every 2.

**The first attempt was restarted after 10 minutes.** It logged a CUDA OOM
warning on a 4.3 GB block. It recovered, but the relaunch added the density cap
and `expandable_segments`. See `attempt1_train.log` on the server.

**Test (v2, 1,402 tiles):**

| weights | AP | AP75 | AP_small | AP_medium | AP_large | AR | mask IoU | matched |
|---|---|---|---|---|---|---|---|---|
| last.pt, EMA | **0.2070** | **0.1570** | 0.0227 | 0.3134 | 0.3594 | 0.3468 | **0.7249** | 65.6% |
| best.pt (epoch 16), EMA | 0.2016 | 0.1476 | 0.0210 | 0.3177 | 0.3772 | 0.3491 | 0.7212 | 66.4% |
| last.pt, raw | 0.1702 | 0.1256 | 0.0148 | 0.3029 | 0.3563 | 0.3461 | 0.7247 | 65.5% |

**What it shows:**
- **The long schedule pays off.** The final epoch beats the mid-run peak on test
  for the first time in this project. Training loss ended at 1.04, against run
  5's 0.88 after only 20 epochs without augmentation: less memorisation, better
  test.
- **The gains are broad.**
  - Medium +0.03, large +0.02 and recall +0.03.
  - AP75 +0.012 over B4.
  - The share of buildings matched rose ~3 points.

  At one seed a +0.006 AP headline is borderline, but these consistent moves
  make it likely real.
- **Small buildings regressed.** AP_small fell from 0.030–0.039 to 0.023. The
  likely cause is the ×0.75 shrink, which turns 10–20 px buildings into slivers.
  A jitter range of ×1.0–1.33 (enlarge only) is the obvious next test.
- **EMA carries it.** The raw weights swing between 0.12 and 0.23 epoch to epoch
  on validation, and the final raw weights test at 0.170.

## r10 in detail

r9 with one change: scale jitter ×1.0–1.33, enlarging only.

**Test (v2, 1,402 tiles):**

| weights | AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR | mask IoU | matched |
|---|---|---|---|---|---|---|---|---|---|
| best.pt (epoch 30), EMA | **0.2177** | **0.4958** | **0.1637** | 0.0295 | 0.3141 | 0.3654 | 0.3436 | 0.7226 | 64.9% |
| last.pt, EMA | 0.2144 | 0.4898 | 0.1607 | 0.0298 | 0.3088 | 0.3531 | 0.3379 | 0.7220 | 64.0% |
| last.pt, raw | 0.1623 | 0.3781 | 0.1182 | 0.0121 | 0.2944 | 0.3480 | 0.3359 | 0.7218 | 63.8% |

**What it shows:**
- **The shrink half of r9's jitter was the small-building cost.** Removing it
  brings AP_small from 0.023 back to 0.030, level with B4, though still under B3's
  0.039. r9's medium and large gains are kept, and AP50 rises +0.026, so the
  single-variable change pays for itself twice over.
- **Validation selection worked.** The epoch-30 best.pt is also better than the
  final epoch on test. v2 valid is clean for this run (COCO init, v2 only).
- **The raw weights collapse on the final epoch in both r9 and r10** (0.170,
  0.162), though raw validation hit 0.230 at epoch 32. The cause is not yet
  understood. A likely suspect is BatchNorm running statistics taken from
  augmented batches at a learning rate near zero. The EMA weights, which average
  buffers too, are unaffected, and are what every r9/r10 checkpoint loads by
  default.
- **Against SAM 3:** 0.2383 on v2 valid against 0.2177 here on v2 test.
  Different splits, but the gap has narrowed from ~0.037 to ~0.02.

## The 92/93/94 dataset (`data_local/ign_idf`, `data_local/ign_paris_idf`)

This dataset uses the same IGN products and conventions as v2. Imagery is BD ORTHO
**IRC** 20 cm 2024, from the same campaign as Paris. Labels are BD TOPO
2026-06-15, v2's edition. Tiling and rasterisation are v2's: 1024 px,
`all_touched`, at least 4 px, no empty tiles.

**How it was built:**
1. `tools/fetch_ign.py` downloaded the six products from data.geopf.fr. BD
   TOPO's advertised `.md5` files return 404, so integrity rests on the 7z CRCs
   that `py7zr` checks on extraction.
2. `tools/build_ign_dataset.py` made the tiles. Of 62 delivered sheets, 15 were
   duplicate border squares, leaving 47.

**What it produced:**

| | tiles | buildings | per tile (mean / p95 / max) |
|---|---|---|---|
| 92/93/94 | 8,131 | 742,610 | 91.3 / 282 / 546 |
| merged train (v2 + 92/93/94) | 13,034 | 1,119,915 | |

Of the 27,072 tile positions considered:
- 12,566 were outside 92 ∪ 93 ∪ 94, and would carry unlabelled neighbour
  buildings.
- 5,063 touched one of v2's 13 Paris sheets.
- 1,312 had no building.

**Leakage:** `tools/merge_coco.py` MD5-hashes every train tile against every v2
valid/test tile, and found 0 identical out of 13,034. Valid and test are links to
v2's own directories.

**Memory:** the densest pair of tiles (484 + 519 buildings after jitter) peaks at
35.1 GiB allocated, 37.2 GiB reserved, of 44.4 GiB. That fits with
`expandable_segments`.

**A finding about v2 itself.** The 5,063 tiles removed for touching Paris sheets
are inner-suburb areas *inside* those sheets. v2 kept all of them, but labelled
them from D075 (Paris-only) buildings. So v2 tiles near the Paris boundary,
including test tiles, contain unlabelled suburb buildings. This plausibly
accounts for part of the "background / unlabelled" false positives in
`tools/error_analysis.py`.

## The raw-weight collapse: BatchNorm diagnostic

`tools/precise_bn.py` keeps the weights and recomputes only the BatchNorm buffers
over 600 un-augmented training tiles. This tests Wu & Johnson's "stale running
statistics" explanation from the 2026-10-01 literature review. Scored on v2 test,
r10 last.pt:

| weights | stored stats | recomputed, inference-mode forward | recomputed, training-mode forward |
|---|---|---|---|
| raw | 0.1623 | 0.1698 | 0.1764 |
| EMA | **0.2144** | 0.1709 | 0.1767 |

**Reading:**
- **Stale statistics are not the explanation.** Recomputing pulls both models to
  the same ~0.176, so it damages the good EMA model rather than repairing the raw
  one.
- **What fits the evidence:**
  - The weights co-adapted with statistics gathered on *augmented* batches.
  - Raw running statistics, at BN momentum 0.1 and batch 2, reflect only the
    last ~10 batches, so they are noisy. The learning rate is ~1e-6 to 1e-8 over
    the last epochs, yet raw validation AP drops 0.226 → 0.157 in two epochs. The
    weights barely move then; the statistics do.
  - The EMA averages those buffers over ~5,000 steps.
- **Untested:** recompute on *augmented* batches. If raw recovers to ~0.21, the
  cause is noisy BN statistics, and lower BN momentum (or EMA) is the fix.
  Practically, keep using the EMA weights.
