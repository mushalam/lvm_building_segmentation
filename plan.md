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
