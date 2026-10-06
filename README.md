# lvm_building_segmentation

Per-building instance segmentation of French IGN aerial imagery with Mask R-CNN.
It was built as an independent second attempt against a fully fine-tuned SAM 3
model from the sibling project `sam3-ft-EOSC`, on the same dataset and with the
same scoring protocol.

**Status (6 Oct 2026).**
- **Accuracy:** SAM 3 is still ahead, by 0.009 AP on the same test tiles. The best
  Mask R-CNN scores **0.2307 segm AP** on the 1,402 held-out v2 test tiles. That is
  r10 (enlarge-only scale jitter, copy-paste, EMA, 36 epochs) plus a Mask Scoring
  head that re-ranks detections by predicted mask quality. SAM 3 scores
  **0.2399** on those tiles.
- **Cost:** Mask R-CNN is 18x smaller (46M vs 841M parameters), uses about a third
  less GPU memory at inference, and trains in hours rather than days. At deployment
  the two run at the same speed: ~135 ms per tile on an L40S.

Every experiment, including the failures, is written up in
[`docs/experiments.md`](docs/experiments.md). The commit messages carry the full
reasoning behind each result. The relevant literature and data sources, with a
verdict on each, are logged daily in
[`docs/literature_review.md`](docs/literature_review.md). Read it before
proposing what to try next.

---

## Results

**Test split.** These are the 1,402 v2 test tiles, which no model trained on. The
scoring is segm AP@[.50:.95] with maxDets 300 and no confidence threshold, via
`lvm/score.py`. "Mask IoU" is the mean mask IoU over matched buildings, as defined
in the SAM 3 project and ported in `lvm/boundary.py`.

| model | AP | AP75 | AP_small | mask IoU | buildings matched |
|---|---|---|---|---|---|
| run 2: Mask R-CNN R50-FPN v2, COCO weights, 2048 px, 12 epochs | 0.2006 | 0.141 | 0.034 | 0.716 | 62.7% |
| B3: pretrained on a public building corpus first | 0.2010 | 0.143 | **0.039** | 0.715 | 62.0% |
| B4: pretrained on French IGN merged data first, weight EMA | 0.2011 | 0.145 | 0.030 | 0.719 | 62.4% |
| r9: run 2 + scale jitter (x0.75-1.33) + copy-paste + EMA, 36 epochs | 0.2070 | 0.157 | 0.023 | **0.725** | **65.6%** |
| r10: as r9 with enlarge-only jitter (x1.0-1.33) | 0.2177 | 0.164 | 0.030 | 0.723 | 64.9% |
| r10b: r10 repeated with `--seed 1` (reproducibility check) | 0.2157 | 0.161 | 0.027 | 0.722 | 65.1% |
| **ms1: r10 + Mask Scoring head (head-only training, 39 min)** | **0.2307** | **0.178** | **0.045** | **0.724** | 63.7% |
| r11: r10's recipe on v2 + 92/93/94 (2x the buildings), matched steps | 0.2143 | 0.154 | 0.029 | 0.718 | **65.9%** |
| merged: trained on v2 + D001 (8x the data) directly | 0.1544 | 0.086 | 0.033 | 0.696 | 58.4% |

**Against SAM 3.**

| split | Mask R-CNN | SAM 3 | gap |
|---|---|---|---|
| v2 test, 1,402 tiles (segm AP) | 0.2307 (ms1 = r10 + Mask Scoring head) | **0.2399** | 0.009 |
| v2 valid, 700 tiles (segm AP) | 0.196 (run 2) | **0.2383** | 0.042 |
| v2 valid, 250-tile sample (matched mask IoU) | 0.714 (run 2) | **0.7341** | 0.020 |
| boundary IoU, object-scale band (250-tile sample) | 0.186 (r10, v2 test) | **0.1944** (v2 valid) | ~0.008, different splits |

- **The test comparison** uses the same 1,402 tiles. SAM 3's 0.2399 is as reported
  by the SAM 3 project. Its run record was not found on the servers checked on
  2026-10-01.
- **The valid comparison** uses run 2, because r9 and r10 were not scored on valid.
- **Boundary IoU** uses the object-scale band (median 2 px) since 2026-10-01; see
  *Measurement conventions*. The two boundary figures are on different splits. On
  the same test sample, r10's 0.186 is ahead of B3 (0.180) and run 2 (0.179), a
  difference the old band could not show (`runs/boundary_object/`).

**Inference speed.** Both models ran on the same L40S GPU over the same 200 v2 test
tiles, at batch 1, via `lvm/bench.py` and `tools/bench_sam3.py`. Results are in
`runs/bench/`.

| | Mask R-CNN r10 | SAM 3 `merged_v3` | SAM 3 `v2_frozen` |
|---|---|---|---|
| deployment (score ≥ 0.5), end to end | 138 ms, 7.2 tiles/s | **134 ms, 7.5 tiles/s** | 148 ms, 6.8 tiles/s |
| model step alone, deployment | 103 ms | 104 ms | 117 ms |
| masks returned per tile, deployment | 67 | 32 | 48 |
| scoring (no threshold), end to end | **257 ms** (217 ms fp16) | 439 ms | 437 ms |
| peak GPU memory, deployment | **3.0 GB** | 4.4 GB | 4.4 GB |

- **At deployment the two families are at parity.** The model step is ~103 ms for
  both. End-to-end time follows how many masks come back, since each is
  thresholded and copied to the CPU.
- **Without a threshold, Mask R-CNN is ~2x faster.** That is mostly SAM 3's
  `predict_one` copying 320 float32 probability maps per tile.
- **Mask R-CNN's real efficiency edge is memory and training cost**, not
  deployment speed.

**What the experiments established.** The strongest findings are listed first.
[`docs/experiments.md`](docs/experiments.md) has the evidence for each.

1. **Augmentation broke the ~0.201 plateau.**
   - Scale jitter, copy-paste and EMA over 36 epochs reach 0.207 AP (r9). The long
     schedule pays off for the first time.
   - Enlarging only, never shrinking, reaches **0.218** (r10). It also recovers
     most of r9's small-building loss (AP_small 0.023 → 0.030).
   - Before r9, run 2, B3 and B4 landed within 0.0005 of each other.
   - r10 reproduces: a second seed (r10b) scores 0.2157, within 0.002, for a
     two-seed mean of 0.2167.
2. **Re-ranking by predicted mask quality adds +0.013.** A Mask Scoring head
   trained for 39 minutes on frozen r10 reaches 0.2307 (AP_small +52%). An
   oracle shows a ceiling of +0.096, so most of the headroom remains.
   - More data with v2's own label convention (92/93/94, 2x the buildings) did
     **not** help: r11 scores 0.214 (r10 0.218).
2. **The labels are probably the main limit, for every model.**
   - They are BD TOPO footprints, 96-99% of them derived from the land registry in
     central Paris.
   - They are drawn at the walls and split at property boundaries.
   - They are rasterised onto an orthophoto that is *not* a true ortho, so tall
     roofs lean several metres off their footprints.

   See [Known caveats](#known-caveats).
3. **These made no difference or made things worse:**
   - longer training *without* augmentation (it overfits after ~12-16 epochs)
   - D4 augmentation, both in training and at test time
   - a DINOv3 backbone
   - NMS tuning
   - a finer mask grid
   - adding rural data (-0.046 AP)
   - French pretraining (a tie)
4. **Weight EMA is a small, real gain:** +0.004 AP.
5. **Tight outlines are not held back by the 28x28 mask grid.** With perfect boxes
   the grid alone reaches IoU 0.928. Matched buildings average 0.71.

---

## Quickstart

### 1. Environment

This needs Python 3.12 and an NVIDIA GPU. Every result used an L40S with CUDA 12.6
wheels.

```bash
git clone https://github.com/mushalam/lvm_building_segmentation.git
cd lvm_building_segmentation
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

On the team GPU server (`tvs-gpu-2`), the working copy is `~/work/lvm_EOSC`. It uses
the `sam3ft-fine` project's venv through the symlink `.venv`, which already has all
of the above installed.

### 2. Data

The data is **not in git**. The loader expects COCO-format splits: one folder per
split, holding the images and an `_annotations.coco.json`.

```
data/ign_building_v2/building/{train,valid,test}/_annotations.coco.json + *.png
data_local/ign_building_merged/building/{train,valid,test}/...
```

| dataset | tiles (train/valid/test) | where it lives |
|---|---|---|
| `ign_building_v2`: Paris (D075), the benchmark | 4,903 / 700 / 1,402 | `tvs-gpu-2`: `~/work/sam3ft-fine/data/ign_building_v2`. This repo's `data` is a symlink to `~/work/sam3ft-fine/data` |
| `ign_paris_idf`: v2 + 92/93/94 (IRC), same conventions | 13,034 / 700 / 1,402 (valid and test = v2's) | `tvs-gpu-2`: `~/work/lvm_EOSC/data_local/ign_paris_idf`, built by `tools/build_ign_dataset.py` + `tools/merge_coco.py`; see docs/experiments.md |
| `ign_building_merged`: v2 + D001 (Ain) | 39,631 / 5,662 / 11,323 | `tvs-gpu-2`: `~/work/lvm_EOSC/data_local/ign_building_merged` (a real-file copy). The original is on the `tvs-gpu-1` team share: `LVM_datasets/IGN_MERGED_v3`, which is symlinks into the source folders |

Both are 1024x1024 PNG tiles at 20 cm. The imagery is **false-colour infrared**
(NIR-R-G), so vegetation looks red. There is one class, `building`, with
`category_id` 1.

**Where the data comes from.** On the `tvs-gpu-1` share `CLIMATE-ADAPT4EOSC/Research/LVM_datasets/`,
`create_instance_seg_dataset_v2.py` builds v2 from two IGN sources:
- the 2024 BD ORTHO IRC 20 cm sheets for D075 (13 JP2 files)
- the BATIMENT layer of BD TOPO v3.5, D075 edition of 2026-06-15

Each building polygon is clipped to its tile and rasterised with `all_touched=True`.
Instances under 4 px are dropped. The same share has
`create_instance_seg_dataset_france_split_v3.py` and per-département folders
(`IGN_France_All_Split/D001`, `D007`) for the other regions. Both IGN products are
open data (Licence Ouverte 2.0), so the v2 pipeline can be pointed at more
départements.

**Never write into `~/work/sam3ft-fine/data`.** It belongs to another project. New
data goes in `data_local/`, which is gitignored.

### 3. Check the scorer

The scorer scores ground truth against itself, which must give 1.0. It gives about
0.99 on the merged set, which has 43 empty 1 px sliver masks; the scorer reports
them.

```bash
python -m lvm.evaluate data/ign_building_v2/building/valid/_annotations.coco.json
```

### 4. Train

This is the run 2 recipe, the simplest configuration at the plateau. It takes about
3.5 h on one L40S.

```bash
CUDA_VISIBLE_DEVICES=1 python -m lvm.train \
    --data data/ign_building_v2/building --out runs/my_run \
    --image-size 2048 --epochs 12 --lr 0.005 --batch-size 2 --workers 10
```

Useful flags (`python -m lvm.train --help` lists them all):

| flag | what it does |
|---|---|
| `--ema 0.9998` | keeps an exponential moving average of the weights. Checkpoints then hold both `model` (averaged) and `model_raw` |
| `--init-from <ckpt>` | staged pretraining: starts from another run's weights |
| `--scale-jitter 0.75,1.33` / `--copy-paste 30` | augmentation, see `lvm/augment.py` |
| `--val-every N` | validates every N epochs. `last.pt` is still written every epoch |
| `--backbone dinov3_convnext_tiny` / `--scratch-heads` | backbone experiments (runs 6 and 8) |
| `--d4` | D4 train-time augmentation (run 7) |

Validation during training uses a 200-tile prefix of `valid/`. It is for tracking
progress only; see [Measurement conventions](#measurement-conventions).

### 5. Score

The checkpoint records the image size, backbone and head settings, and `lvm.score`
reads them back. The output includes AP, AP50/75, AP by size, AR, matched mask IoU
and boundary IoU.

```bash
python -m lvm.score --ckpt runs/my_run/best.pt --split test --out runs/my_run/score_test.json
# EMA checkpoints: --weights raw scores the unaveraged weights
```

### 6. Look at predictions

```bash
python -m lvm.predict --ckpt runs/my_run/best.pt --image some_tile.png --out-dir predictions/
```

This writes `<name>_overlay.png` (each building filled and outlined) and
`<name>_buildings.json` (polygons in pixel coordinates). It accepts files or
directories. Images larger than 1024 px are predicted in 1024 px pieces.
`--score-thresh` (default 0.5) only affects what is drawn.

### 7. Time inference

```bash
python -m lvm.bench --ckpt runs/my_run/best.pt --out bench.json
```

For SAM 3, `tools/bench_sam3.py` times the model the same way, through the
`sam3-ft-EOSC` code. Its docstring gives the invocation.

---

## Checkpoints

Checkpoints are not in git. They are ~180 MB each, and `*.pt` is gitignored. They
live on `tvs-gpu-2` under `~/work/lvm_EOSC/runs/<run>/`. Each run folder holds
`best.pt` (the best validated epoch), `last.pt` (the final epoch), `args.json` and
`history.json`.

| use | checkpoint |
|---|---|
| best overall (test 0.2307) | `runs/ms1_head_only/best.pt` (r10 + Mask Scoring head; every tool rebuilds the head from the checkpoint) |
| best without the Mask Scoring head (test 0.2177) | `runs/r10_jitter_up/best.pt` (EMA weights; the raw weights score far lower, so use the default) |
| best on small buildings (AP_small 0.039) | `runs/stageB3_long/best.pt` |
| best trained on v2 alone, simplest | `runs/maskrcnn_v2_hires/best.pt` (run 2, test 0.2006) |

## Running on the team GPU server

- **GPU 0** on `tvs-gpu-2` is used by a vLLM server that holds ~45 GB. **Use GPU 1**,
  by setting `CUDA_VISIBLE_DEVICES=1`. Check it first with `nvidia-smi`.
- Launch long jobs detached, so they survive the SSH session closing:
  `setsid nohup python -u -m lvm.train ... > runs/X/train.log 2>&1 < /dev/null &`.
  Record the PID with `echo $! > runs/X/train.pid`.
- To queue scoring after training, wait on that **PID**, as the
  `runs/*/score_after.sh` scripts do. Do not wait on a `pgrep -f` pattern. A pattern
  matches the command line of the shell that launched it, and it once stalled a
  queue for three hours.
- A 1024 px tile upsampled to 2048 needs about 19-36 GB of GPU memory at batch 2,
  so two training runs will not fit on one GPU.

## Known caveats

- **Labels versus imagery.** v2 labels are BD TOPO building footprints drawn over
  the 2024 BD ORTHO mosaic, which is not a true ortho.
  - Outlines trace walls at ground level and are split at parcel boundaries.
  - Roofs of tall buildings sit visibly off their outlines.
  - Some outlines cover buildings hidden under trees.

  A large share of the measured error is disagreement with this convention (see
  `tools/error_analysis.py`). It caps any model scored against these labels, SAM 3
  included. IGN's ORTHO Express (a true ortho) would align better, but switching to
  it changes the benchmark.
- **The v2 valid split is compromised for models initialised from the merged
  data.** 492 of its 700 tiles are in merged train. For B4-style runs, judge on v2
  test only. The 1,402 v2 test tiles are pinned and byte-identical in both datasets.
- **The merged set mixes imagery products.** Its D001 tiles (90% of merged
  train) are natural-colour RVB orthophotos. v2 and all test tiles are
  false-colour IRC. Results trained on merged data (`merged_r1_coco`, B4) are
  confounded by this. New data must use the IRC product, as `tools/fetch_ign.py`
  does for 92/93/94.
- **The merged set has 15 all-white tiles** (imagery nodata) that carry 59 labels.
  They are left in, being too few to matter.

## Measurement conventions

- **Test is the verdict, and there is one scorer.** `lvm/evaluate.py` is used both
  by training validation and by `lvm/score.py`. There is no confidence threshold:
  the model's own 0.05 floor is disabled, and maxDets does the capping.
  Every image in a split is scored, including images with no detections.
- **Validation is a 200-tile prefix and it is noisy.** Epoch-to-epoch standard
  deviation is 0.02-0.04 AP, which is larger than most differences between runs. Do
  not quote it.
- **Keep both `best.pt` and `last.pt`.** Picking the best of N noisy epochs flatters
  a run by 0.025-0.07 AP, and flatters noisy runs more. Score both on the full
  split.
- **Differences under ~0.005 AP are noise** at a single seed.
- **Boundary IoU uses SAM 3's current object-scale band.**
  - The band is `round(0.02 × sqrt(area))` for each building, clamped to 2–15 px.
    The median is 2 px on this data. The port is verified identical to
    `sam3ft-fine/sam3ft/metrics.py` (commit 9e8f64e) on all six outputs.
  - Boundary IoU figures this repo reported before 2026-10-01 used an older
    image-scale band: 2% of the tile diagonal, which is 29 px. That band erodes most
    buildings away entirely, so boundary IoU came out roughly equal to mask IoU.
    Those figures are not comparable with SAM 3's. Mask IoU, AP and match rate were
    never affected.
- **Detector limits come from the data.**
  - Up to 305 buildings per tile, so `box_detections_per_img` is 400.
  - Anchors are 12-175 px, because the default smallest anchor (32 px) is bigger
    than the median building.
  - `tools/size_stats.py` computes these.

## Repository layout

```
lvm/
  data.py        COCO tile loader (+ D4, scale jitter, copy-paste hooks)
  augment.py     scale jitter and background-only copy-paste
  train.py       training loop, EMA, validation via the shared scorer
  evaluate.py    the one COCO scorer, with a perfect-prediction self-test
  boundary.py    matched mask IoU / boundary IoU, ported from sam3-ft-EOSC
  score.py       score a checkpoint on a split (optionally with --tta)
  tta.py         D4 test-time augmentation (measured to hurt here; kept for record)
  predict.py     segment images: overlay PNG + polygon JSON
  bench.py       inference-speed benchmark
  backbones.py   DINOv3 ConvNeXt trunks behind a torchvision FPN
tools/
  error_analysis.py  where AP is lost: proposals, misses, merges, false positives
  mask_ceiling.py    IoU ceiling of the 28x28 mask grid given perfect boxes
  bench_sam3.py      SAM 3 inference benchmark, run from the sam3-ft-EOSC code
  size_stats.py, filter_empty.py, filter_density.py, remap_category.py  data prep
  daily_litreview_check.sh  SessionStart hook: prompts the daily literature review
  fetch_ign.py       download + unpack IGN products (BD ORTHO, BD TOPO) from data.geopf.fr
  build_ign_dataset.py  BD ORTHO + BD TOPO -> COCO tiles, v2's conventions, no-leak guards
  merge_coco.py      union of train splits, with an MD5 leakage check against valid/test
  precise_bn.py      recompute BatchNorm statistics on frozen weights (diagnostic)
  v2_paris_sheets.txt  the 13 BD ORTHO sheets v2 was cut from
runs/<run>/      per-run args, histories, scores and queue scripts (no checkpoints)
results/         archived histories and scores from runs 3-8 and stages A/B
docs/experiments.md  every run: command, question, result, verdict
docs/literature_review.md  daily log of relevant papers/data, each with a verdict
plan.md          original plan and lab notebook (through run B3)
```
