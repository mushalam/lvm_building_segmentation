# lvm_building_segmentation

Per-building instance segmentation of aerial imagery with Mask R-CNN, run as an
independent second attempt against a fine-tuned SAM 3 baseline (`sam3-ft-EOSC`)
on the IGN building dataset.

**Outcome:** the target was not beaten. After nine training configurations the
best result is **0.1954 segm AP**, against SAM 3's **0.2383**. Every experiment,
including the ones that failed, is recorded in `plan.md` and in the commit
messages, which carry the full reasoning for each run.

## The question

SAM 3 finds buildings but outlines them poorly: AP drops 2.5x from IoU 0.50 to
0.75, and AP on small objects is 0.078 against 0.359 on large ones. The starting
hypothesis was that its global mask grid (stride 3.5) is too coarse for a median
building of about 41 px, and that Mask R-CNN's per-ROI 28x28 masks would fix it.

That hypothesis was refuted by run 1. The 28x28 mask is interpolated from
features pooled at stride 4, so both models are limited by feature resolution,
not by the mask head. The later runs then tested the other plausible causes one
variable at a time.

## Results

All figures are segm AP@[0.50:0.95] on the full 700-image `valid` split,
maxDets 300, no confidence threshold, scored by `lvm/score.py`. Where a run kept
both checkpoints, `best.pt` is the epoch chosen on the 200-image validation
subset and `last.pt` is the final epoch.

| run | change being tested | best.pt | last.pt |
|---|---|---|---|
| 1 | Mask R-CNN R50-FPN v2, COCO weights, 1024 px | 0.1789 | |
| 2 | input 2048 px | 0.1948 | |
| 3 | lr 3.5e-3, 30 epochs (confounded, see below) | 0.1890 | |
| 4 | repeat of run 3 at 20 epochs | 0.1874 | 0.1792 |
| 5 | run 2 at 20 epochs instead of 12 | 0.1896 | 0.1845 |
| 6 | DINOv3 ConvNeXt-Tiny trunk, random heads | 0.1708 | 0.1674 |
| 7 | run 5 + D4 augmentation | 0.1920 | 0.1482 |
| 8 | ImageNet R50 trunk, random heads (control for run 6) | 0.1734 | 0.1477 |
| A+B | pretrain on a public building corpus, then run 2 config | 0.1932 | 0.1761 |
| A2+B2 | as A+B, dense-tile subset of the corpus | 0.1905 | 0.1905 |
| A+B3 | as A+B, run 5 schedule | **0.1954** | 0.1814 |
| — | SAM 3 full fine-tune, 50 epochs (target) | 0.2383 | |

What the runs established, most certain first:

- **Feature resolution matters but is not the main gap.** 1024 → 2048 px gave
  +0.016 AP and closed about 27% of the gap to SAM 3.
- **Longer training overfits.** At 20 epochs training loss keeps falling while
  validation AP drifts down. The model saturates around 12 epochs on this data.
- **COCO detector heads carry the transfer, not the backbone.** Swapping the
  trunk between ImageNet ResNet-50 and DINOv3 changes AP by 0.003–0.020 in
  inconsistent directions. Removing COCO-pretrained FPN/RPN/heads costs
  0.016–0.037.
- **In-domain pretraining buys speed, not much accuracy.** Pretraining on a
  public building corpus reaches 0.20 AP by epoch 5 instead of 8–9, and is worth
  about +0.005 AP at a matched schedule. Neither instance density nor corpus
  size explains why it isn't more. Geography, which could not be tested with
  the data available, is the remaining candidate.
- **D4 augmentation hurts at this schedule.** `best.pt` shows a small gain, but
  that comes from selecting the maximum of a much noisier run. Every other
  estimator shows a loss, including −0.036 on `last.pt`.

Runs 3 and 4 changed learning rate and epoch count together, so their results
can't be attributed to either change. Run 5 is the clean single-variable
version.

The **test** split (1,402 images) has not been scored yet. The comparison
against SAM 3 in `plan.md` is specified on test, so the numbers above are
validation figures.

## Repository layout

```
lvm/
  data.py        COCO-format tile loader, D4 augmentation
  backbones.py   DINOv3 ConvNeXt trunks (via timm) behind a torchvision FPN
  train.py       training loop; validates with the same scorer as score.py
  evaluate.py    the single COCO scorer, with a self-test
  score.py       score a checkpoint on a split
tools/
  size_stats.py      instance-size statistics, used to set anchors and limits
  filter_empty.py    drop images with no annotations
  filter_density.py  keep images at or above a minimum instance count
  remap_category.py  rewrite category_id to match the target dataset
results/         per-run training histories and full-split scores (JSON)
runs/            histories from runs 1–2
plan.md          the plan, the hypotheses, and a write-up of every run
```

## Setup

Requires Python 3 with PyTorch, torchvision, pycocotools, numpy, and timm (timm
only for the DINOv3 backbones). Training was done on a single GPU.

```
pip install torch torchvision pycocotools numpy timm
```

Data is not included. The loader expects COCO-format splits, one folder per
split with images alongside an `_annotations.coco.json`:

```
data/ign_building_v2/building/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
```

`data/` and `data_local/` are gitignored, and so are checkpoints (`*.pt`).

## Usage

Check the scorer first. It scores ground truth against itself and must return
1.0 on every metric, including the case where only some images are scored:

```
python -m lvm.evaluate
```

Train the run 2 configuration (the strongest single-stage setup):

```
python -m lvm.train --image-size 2048 --epochs 12 --lr 0.005 --batch-size 2 \
    --out runs/run2
```

Score a checkpoint on the full split. Image size, backbone and head settings are read
back from the checkpoint:

```
python -m lvm.score --ckpt runs/run2/best.pt --split valid --out results/run2.json
```

Other flags: `--backbone dinov3_convnext_{tiny,small,base}`, `--scratch-heads`
(ImageNet trunk with random heads, the control for the DINOv3 runs), `--d4`, and
`--init-from <ckpt>` for staged pretraining. `python -m lvm.train --help` shows
the rest.

## Measurement notes

- Validation during training uses a 200-image prefix of `valid/`. It is only
  good enough to track progress. Its epoch-to-epoch noise (sd 0.02–0.04) is
  larger than most of the differences between runs, so reported numbers always
  come from `lvm/score.py` on the full split.
- `last.pt` is saved every epoch alongside `best.pt`. Picking the best of 20
  noisy epochs flatters a run by 0.025–0.07 AP, and more for noisier runs. That
  makes `best.pt` comparisons unfair between runs with different variance.
- Detector limits are set from the data, not left at torchvision defaults.
  Images hold up to 305 buildings, and the default's smallest anchor (32 px) is
  larger than the median building. `tools/size_stats.py` produces the
  statistics these settings come from.
