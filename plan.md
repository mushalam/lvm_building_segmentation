# lvm_EOSC — an independent second attempt at building footprint segmentation

**Status:** started 21 September 2026. Independent of `sam3-ft-EOSC`; shares only
the dataset. One day to beat it.

**Target to beat.** `sam3-ft-EOSC`'s best per-building model, measured on the
same held-out images with the same protocol:

| | segm AP | mask IoU | boundary IoU | AP small |
|---|---|---|---|---|
| SAM 3 full fine-tune, 50 epochs | **0.2383** (val) | 0.7341 | 0.1944 | 0.078 |

Comparison will be on the `ign_building_v2` **test** split (1,402 images, never
used for training or selection by either project), maxDets 300, no confidence
threshold. Anything else is not a comparison.

---

## 1. Why a second attempt might win

The incumbent's failure is specific and measured, not diffuse:

    AP @0.50       0.507        AP small   0.078   (35.8% of instances)
    AP @0.75       0.200        AP medium  0.346   (54.8%)
    AP @0.50:0.95  0.238        AP large   0.359   ( 9.4%)

It **finds** buildings and fails to **outline** them: a 2.5x collapse from IoU
0.50 to 0.75, and a 4.4x gap between small and large objects.

**The mechanism is arithmetic.** SAM 3's mask head predicts on a global grid at
288² for a 1008 px input — stride 3.5. The median building here is 41×41 px,
which becomes **11.8 × 11.8 mask pixels**; at p25 it is 6.5 × 6.5. You cannot
place a tight footprint with ~40 pixels, and one pixel of error moves IoU
enormously. That is why boundary IoU sits at 0.19 while interior overlap reaches
0.73.

## 2. The central bet

**Predict masks per-instance, not on a global grid.**

Mask R-CNN [1] crops each detection to an ROI and predicts a 28×28 mask *within
that ROI*. For a 41×41 px building that is **sub-pixel** mask resolution —
roughly 12x finer than a stride-3.5 global grid. The architecture spends
resolution where the object is, rather than uniformly across a tile that is
mostly background.

This is a falsifiable prediction, not a preference: **if the diagnosis is right,
Mask R-CNN should beat SAM 3 on boundary IoU and AP-small by a wide margin, even
if it loses on detection recall.** If it does not, the resolution diagnosis is
wrong and the incumbent's ceiling is elsewhere.

Supporting evidence: Mask R-CNN is the established approach for this task, with
strong published results on building footprints [2, 3], and a PointRend head [4]
— which refines exactly at boundaries — is reported as the best-performing
variant [3].

## 3. Plan

**Phase 1 — harness before model.** Build the evaluator first, as one code path,
and validate it by scoring a released COCO model and reproducing its published
AP. The incumbent project lost weeks to three detection caps in series, a
confidence threshold worth 7% AP, and two scorers in one repository that
disagreed. None of that is interesting and all of it is avoidable by writing the
measurement first.

**Phase 2 — baseline.** torchvision `maskrcnn_resnet50_fpn_v2`, COCO-pretrained,
fine-tuned on `ign_building_v2` train (4,903 images). Configuration driven by
the data rather than defaults:
- `box_detections_per_img = 400` (default 100; our images hold up to 305)
- anchor sizes from the actual box statistics (median 41 px, p25 23 px)
- 1024 px input, no resize, since the tiles are already tiled

**Phase 3 — if time allows.** PointRend-style boundary refinement [4], or
polygon output [5], which is what the deliverable actually consumes.

## 4. What this attempt does differently, by process

1. **Evaluation harness first**, validated against a published number.
2. **One scorer.** Training validation calls the same function as offline
   scoring. The incumbent's 7% discrepancy came from having two.
3. **No confidence threshold at evaluation.** COCO expects all detections
   ranked; thresholding discards recall the metric would credit.
4. **Instance-size histogram against output stride, computed before training.**
   Knowing 36% of instances fall in the failing regime is a day-one fact.
5. **Report on test, select on valid.** The 70/10/20 split already exists and
   is disjoint.

## 5. Honest risks

- **One day, one GPU.** A COCO-pretrained Mask R-CNN fine-tunes in ~2 h on this
  data, so several configurations are reachable — but nothing exotic is.
- **Mask R-CNN may lose on recall.** Its default proposal machinery is tuned for
  ~7 objects per image; ours averages 77. Raising detection and proposal caps is
  necessary and may not be sufficient.
- **The comparison could be unfair in our favour** if the incumbent's numbers
  carry defects we have already fixed. Both must be scored through *this*
  project's harness before any claim is made.
- **A negative result is a real outcome** and will be reported as one. If
  per-ROI masks do not beat a global grid here, that refutes the resolution
  diagnosis and is worth more than a marginal win.

## References

[1] He, K. et al. *Mask R-CNN.* ICCV, 2017.
[2] Zhao, K. et al. *Large-scale building extraction in very high-resolution aerial imagery using Mask R-CNN.* 2019.
[3] NourEldeen et al. *Enhanced building footprint extraction from satellite imagery using Mask R-CNN and PointRend.* Bulletin of EEI.
[4] Kirillov, A. et al. *PointRend: Image Segmentation as Rendering.* CVPR, 2020.
[5] *SAMPolyBuild: Adapting the Segment Anything Model for polygonal building extraction.* ISPRS J., 2024.


---

# Result: the central bet is refuted

**Run 1 (`runs/maskrcnn_v1`)**, Mask R-CNN R50-FPN v2, 15 epochs, batch 4,
anchors from the data, scored on the same 700 validation images as the
incumbent with the same protocol:

| | Mask R-CNN (46M) | SAM 3 v2_full (841M) |
|---|---|---|
| segm AP | **0.1789** | **0.2383** |
| AP@0.50 | 0.4446 | 0.507 |
| AP@0.75 | 0.1099 | 0.200 |
| **AP small** | **0.0286** | **0.078** |
| AP medium | 0.2506 | 0.346 |
| AP large | 0.2905 | 0.359 |
| AR | 0.2987 | 0.333 |

**It loses on every measure, and by the widest margin on AP-small — the metric
the whole design was built to win.** 2.7x worse, where I predicted better.

## Why the reasoning was wrong

I compared SAM 3's mask *output grid* (stride 3.5) with Mask R-CNN's mask output
grid (28x28 per ROI) and concluded the latter gave 3.3x more resolution at p25.

That compares the wrong things. Mask R-CNN's 28x28 is interpolated from features
pooled by ROIAlign off **P2, at stride 4**. A 41 px building yields ~10x10
*features* — essentially the same as SAM 3's 11.8. The output grid was never the
binding constraint; **feature stride is**, and both sit at roughly stride 4. A
mask head can upsample to any resolution it likes and cannot invent detail the
backbone did not encode.

## What this implies, which is the useful part

1. **Boundary quality is limited by feature resolution, not mask-head output.**
   Interventions that refine the output (PointRend, RefineMask) address a
   symptom. Interventions that raise feature resolution — larger input, a
   stride-2 FPN level, a high-resolution stem — address the cause.
2. **The incumbent's `--highres-stem`, never yet run, is better motivated than
   anything attempted here**, and is a one-flag experiment on a model already
   30% ahead.
3. **Scale and pretraining are doing real work.** 841M parameters with
   large-scale pretraining beat 46M COCO-pretrained by 33% on this task, on a
   third the epochs per unit of wall-clock. The gap is not obviously closable by
   a better small architecture.

## What was worth having anyway

- The harness caught its own subset-scoring bug within one validation pass, via
  a perfect-prediction self-test and an arithmetic check (0.19 x 200/700 =
  0.054). The predecessor shipped the same bug for four full runs.
- The comparison is clean: same split, same protocol, no threshold, ids pinned.
- A negative result in one day, costing ~2 GPU-hours, is a cheap way to rule out
  a plausible architecture.

## Honest status against the brief

The brief was to beat the incumbent within a day. **It does not.** 0.1789
against 0.2383. Reporting that rather than looking for a framing that rescues
it; the diagnostic value is in *why*, and that has redirected the incumbent
project's next experiment.


---

# Run 2: higher feature resolution, and what it did not fix

Run 1's refutation said the binding constraint was **feature stride**, not
mask-head output. Run 2 tests that directly: identical architecture and epochs,
input upsampled 1024 -> 2048, anchors scaled with it.

**All three scored on the same 700 validation images, same protocol:**

| | run 1 @1024 | run 2 @2048 | SAM 3 v2_full |
|---|---|---|---|
| trainable | 46M | 46M | 487M |
| **segm AP** | 0.1789 | **0.1947** | **0.2383** |
| AP@0.50 | 0.4446 | 0.4633 | 0.507 |
| AP@0.75 | 0.1099 | 0.1342 | 0.200 |
| AP small | 0.0286 | 0.0342 | 0.078 |
| AP medium | 0.2506 | 0.2733 | 0.346 |
| AP large | 0.2905 | 0.3125 | 0.359 |
| AR | 0.2987 | 0.3102 | 0.333 |

## What run 2 establishes

**Feature resolution matters, and was correctly identified.** +0.0158 AP from
one change, with the largest relative gain on AP@0.75 (+22%), which is the
metric that measures tight localisation. The mechanism is confirmed.

**It is not the dominant factor.** It closed **27% of the gap** to SAM 3, not
the majority. And `AP_small` remains **2.3x worse** (0.034 against 0.078)
despite doubling the feature cells available to every small object. If
resolution were the main driver of that gap, this run should have closed far
more of it.

**So SAM 3's advantage on small objects is mostly scale and pretraining, not
architecture.** That is the honest reading, and it is the opposite of the
premise this project started from.

## A measurement note against ourselves

Mid-run the 200-image validation showed 0.2117 and was reported as closing 55%
of the gap. The full 700-image split gives 0.1947 and 27%. The 200-image subset
flattered the result by 0.017. Nothing was wrong with either number; the error
was characterising a result from the cheap proxy before the proper measurement
existed. Validation subsets are for tracking progress, not for conclusions.

## Status against the brief

**The brief was to beat SAM 3's 0.2383 within a day. Run 2 reaches 0.1947. It
does not.**

What the day produced instead: two falsifiable hypotheses tested, one refuted
outright and one confirmed but shown insufficient; a measurement harness that
caught its own subset-scoring bug within a single validation pass; and a
quantified answer to *why* the incumbent wins, which is scale and pretraining
rather than mask-head design.

## Where a third run would go, and whether it is worth it

Ranked by what the evidence now supports:

1. **Longer training.** SAM 3 had 50 epochs; both runs here had 12-15. The
   incumbent's own arms peaked at their final epoch, so 12 is likely
   under-trained. Cheapest untested variable.
2. **A stronger pretrained backbone.** The gap is attributed to pretraining, so
   this attacks the cause. torchvision offers only COCO weights; a
   remote-sensing or self-supervised backbone would be the real test.
3. **Higher resolution still (3072) or a stride-2 level.** Diminishing: the
   1024 -> 2048 step bought 0.0158 and the next would likely buy less.

Honest assessment: **(2) is the only one likely to close a 0.044 gap**, and it
is not a one-day change. A specialist architecture at this scale appears to be
the wrong tool against a well-pretrained 841M foundation model on this task,
which is itself a useful finding for the project that commissioned it.
