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

# Run 3, in flight: longer training, and a selection rule that cannot decide

Run 3 tests option (1) above — 30 epochs at image size 2048, the cheapest
untested variable — and it exposed a defect in this project's own harness
before it produced a result.

## The validation subset is too small for the question asked of it

Through epoch 13, validation AP on the 200-image selection subset reads:

```
ep 0  0.1506    ep 5  0.1367    ep 10  0.1518
ep 1  0.1453    ep 6  0.1434    ep 11  0.1758
ep 2  0.1438    ep 7  0.1370    ep 12  0.1722
ep 3  0.1325    ep 8  0.1710    ep 13  0.2024
ep 4  0.2036    ep 9  0.1502
```

Epochs 3 and 4 are consecutive and differ by **0.0711 AP**. The best-checkpoint
rule is meanwhile deciding between epoch 4 (0.2036) and epoch 13 (0.2024) — a
margin of **0.0012**, about one-sixtieth of the noise floor the same estimator
demonstrates one epoch apart. The rule is not measuring what it claims to.

The components disagree with the scalar, which is the tell. Epoch 13 beats
epoch 4 on AP75 (0.1419 vs 0.1324) and AP_small (0.027 vs 0.025) and loses only
on AP50 (0.4795 vs 0.4890) — the loosest threshold, where duplicate and
near-miss detections are most easily rewarded. Epoch 13 also sits on a rising
shoulder (0.1518, 0.1758, 0.1722, 0.2024) while epoch 4 stands alone above a
0.13–0.15 neighbourhood. On the evidence epoch 13 is the better model, and
`best.pt` still holds epoch 4.

## This is the subset bug again, wearing different clothes

Run 1's harness bug scored 200 predicted images against 700 ground-truth ones
and scaled AP by the ratio. That was caught and fixed. This is the same
mistake's second form: a 200-image estimate trusted for a decision finer than
its precision. The first form corrupted a *number*; this one corrupts a
*choice*, which is harder to notice because the number it reports is correct —
it is merely imprecise, and nothing in the output says so.

Worth recording that **the incumbent does not have this problem.** The SAM 3
arms run `--val-images 700`, the full split. The pipeline being competed
against selects checkpoints on the whole set; this one did not. That asymmetry
is in the incumbent's favour and should be stated whenever run 3 is compared
against it.

## What was changed, and what it does not fix

`lvm/train.py` now writes `last.pt` every epoch alongside `best.pt` (commit
`6392ade`). This does not repair the selection rule — it makes the rule's
mistakes recoverable, since both checkpoints can then be rescored on the full
700-image split where a 0.0012 difference is actually resolvable.

It does not help run 3, which was already in flight when the change landed;
the module was loaded. Epoch 13's weights are gone. If epoch 4's spike survives
to epoch 29, run 3's only artefact will be a checkpoint chosen by a coin-flip,
and the honest report of that outcome is that the run failed to select rather
than that the model failed to learn.

**Correct fix for run 4:** select on the full 700-image split, as the incumbent
does. Validation cost rises ~3.5×, which at ~18 min/epoch is the binding
objection — but selecting on noise makes the cheaper epochs worthless.

## Run 3, result: the run did not test its own hypothesis

Full 700-image `valid` split, maxDets 300, same scorer as runs 1 and 2:

| run | config | epochs | full-split AP | AP50 | AP75 |
|---|---|---|---|---|---|
| 1 | 1024 px | 12 | 0.17891 | 0.4446 | 0.1099 |
| 2 | 2048 px | 15 | **0.19476** | 0.4633 | 0.1342 |
| 3 | 2048 px, lr 3.5e-3 | 30 | 0.18899 | 0.4651 | 0.1196 |
| — | SAM 3, full fine-tune | 50 | **0.2383** | | |

Read naively, run 3 says longer training makes things slightly worse. That
reading is wrong, and the reason matters more than the number.

**The checkpoint scored is epoch 4 of 30.** Every epoch that constituted
"longer training" — the thing under test — was discarded by the selection rule
before it could be measured. 0.18899 is the score of a five-epoch model that
happened to land on a favourable subsample. The hypothesis was never tested.
Eight and a half GPU-hours produced no evidence about the question they were
spent on.

### The subset reading was inflated by 2.9 sigma, confirmed against truth

Epoch 4 scored **0.2036** on the 200-image selection subset and **0.18899** on
the full 700. The subset overstated that specific checkpoint by **0.0146**,
which is 2.9x the standard deviation of the run's own plateau (0.0051, epochs
13-29). This is no longer an argument from trajectory shape — it is the same
weights measured both ways, and most of the spike disappears under the honest
estimator.

It also explains the AP75 column. Run 3 scores 0.1196 there against run 2's
0.1342, despite its plateau epochs reaching AP75 0.1452 on the subset. The
retained checkpoint is simply an earlier, blunter model.

### What the plateau was worth is now unknowable

Epochs 13-29 held a subset mean of 0.1930 with sd 0.0051 and produced the run's
best AP75 (0.1452 at epoch 20) and best AP_small (0.0312, +27% over epoch 4).
None of those weights exist. `last.pt` landed one commit too late to help the
run that motivated it.

## Run 4: the corrected repeat

Launched 07:46 on 22 Sep on tvs-gpu-2 GPU 1, `runs/maskrcnn_v4`, 20 epochs at
2048 px, lr 3.5e-3 — run 3's configuration, truncated to 20 because the plateau
was reached by epoch 13 and the remaining 17 epochs bought nothing measurable.

The change is not to the model. It is that `last.pt` now exists, so the
converged model survives whatever the selection rule decides. **Both `last.pt`
and `best.pt` will be scored on the full 700**, and `last.pt` is the primary
artefact: taking the final epoch applies no max over a noisy sequence and so
carries no selection bias at all. `best.pt` becomes the secondary reading, and
the difference between them measures what the bias was worth.

Expected finish ~13:15. This is what run 3 should have been.

## Run 4, result: selection worked, and my experiment design did not

Full 700-image `valid` split, maxDets 300:

| checkpoint | epoch | subset AP | full-split AP | AP50 | AP75 |
|---|---|---|---|---|---|
| `best.pt` | 14 | 0.2038 | **0.18743** | 0.4529 | 0.1279 |
| `last.pt` | 19 | 0.1938 | **0.17922** | 0.4326 | 0.1241 |

### Correction: "last.pt is the unbiased artefact" was too strong

The subset ranked epoch 14 above epoch 19 by 0.0100. The full split confirms it
by 0.0082 — same direction, comparable size. `best.pt` is genuinely the better
model and the selection rule got it right.

The distinction I blurred: taking the final epoch is unbiased as a
*measurement*, because no maximum is taken over a noisy sequence. It is not
therefore the best *model* — the final epoch of a OneCycle schedule has no
claim to being the peak. Selection is not broken in general. It fails when the
margin it arbitrates is smaller than the estimator's noise, which in run 3 was
0.14 sigma and here was about 2 sigma. At 2 sigma it worked.

Keeping both checkpoints remains correct, because which regime you are in is
only knowable after the fact.

### The real problem: run 3 and run 4 were confounded

| run | image size | batch | lr | epochs | full-split AP |
|---|---|---|---|---|---|
| 1 | 1024 | 4 | 0.005 | 15 | 0.17891 |
| 2 | 2048 | 2 | **0.005** | 12 | **0.19476** |
| 3 | 2048 | 2 | **0.0035** | 30 | 0.18899 (void) |
| 4 | 2048 | 2 | **0.0035** | 20 | 0.18743 |

Between run 2 and run 3 I changed the learning rate from 0.005 to 0.0035 *and*
the epoch count from 12 to 30, then described the result as a test of longer
training. It is not. Runs 3 and 4 measure a different learning rate that also
runs longer, and run 4's 0.0073 loss against run 2 is unattributable between
the two changes.

Run 2's trajectory argues the confound matters. Its last four epochs read
0.2057, 0.2117, 0.1948, 0.2073 — still climbing at epoch 12, never plateaued —
and its subset ceiling of 0.2117 is above anything runs 3 or 4 reached (0.2038
in run 4, 0.2036 in run 3). The case for longer training looks *better* at
lr 0.005 than at the rate I actually tested it with.

Run 4 is still a valid measurement of its own configuration, because its
selection landed at epoch 14, inside the converged region. It is a clean
negative for (2048 px, lr 3.5e-3, 20 epochs). It is not evidence about epochs.

## Run 5: the clean single-variable test

Launched 13:42 on 22 Sep, `runs/maskrcnn_v5_lr5e3` — run 2's exact
configuration with epochs 12 -> 20 and nothing else touched:

    --image-size 2048 --batch-size 2 --lr 0.005 --epochs 20 --val-images 200

Against run 2 this isolates epoch count, which is what run 3 was supposed to do
and did not. Both `last.pt` and `best.pt` will be scored on the full 700.
Expected finish ~19:15.

Prediction, recorded before the result: run 2 was still improving when it
stopped, so 20 epochs at lr 0.005 should exceed 0.19476. If it does not, the
2048 px Mask R-CNN has converged near 0.195 and the remaining 0.043 to SAM 3 is
not reachable by schedule changes at all — which would settle option (1) and
promote the pretrained-backbone question to the only live one.

## Run 5, result: the prediction was wrong, and it closes option (1)

Run 2's exact configuration with epochs 12 -> 20, nothing else changed:

| run | epochs | checkpoint | full-split AP |
|---|---|---|---|
| 2 | 12 | best (ep 9) | **0.19476** |
| 5 | 20 | best (ep 12) | 0.18963 |
| 5 | 20 | last (ep 19) | 0.18453 |

I predicted, in writing and before the result, that 20 epochs would exceed
0.19476 because run 2 was still climbing when it stopped. It did not. Longer
training is worth **-0.0051**, and the prediction is refuted.

### The mechanism is overfitting, and the loss shows it plainly

|  | train_loss | subset AP |
|---|---|---|
| run 2, final epoch (11) | 1.0689 | 0.2073 |
| run 5, final epoch (19) | **0.8842** | 0.1971 |

Run 5 fit the training set 17% better and generalised worse. Over its last
eight epochs train_loss falls monotonically 1.1495 -> 0.8842 while validation
AP drifts 0.2043 -> 0.1971. The extra epochs are spent memorising.

This also explains a pattern visible since run 4: `best.pt` lands mid-schedule
in both runs (epoch 14 of 20, then epoch 12 of 20), never at the end. The late
low-learning-rate phase of a OneCycle schedule is where this model overfits.

**Option (1) from the run-2 postmortem is closed.** The architecture saturates
near 12 epochs on 2,451 images and further epochs actively hurt. No schedule
change will reach SAM 3's 0.2383 from 0.195.

### Why that makes augmentation the obvious next move

The loader applied **no augmentation at all** -- `__getitem__` returned each
tile unmodified. With 2,451 training tiles and a 46M-parameter detector,
saturating at epoch 12 is then unsurprising.

The remedy is handed over by the sibling project. D4 -- the 8 dihedral
transforms -- is *exact* for nadir aerial imagery, because a nadir view has no
canonical "up", so a flipped or quarter-turned tile is a valid sample rather
than a distortion. That symmetry is precisely why D4 test-time augmentation
earned +6% AP over there. This is its train-time counterpart, and it
multiplies the effective training set by eight.

Implemented in `lvm.data.apply_d4`, with boxes recomputed from the transformed
masks rather than transformed themselves -- corner rotation is correct only at
multiples of 90 degrees and fails silently otherwise, whereas a box derived
from its mask cannot drift out of agreement with it. Verified across all 8
transforms on a real 34-instance tile: exact box/mask agreement, mask areas
preserved, all 8 renderings distinct.

## Runs 6 and 7

**Run 6** (launched 19:25, `runs/maskrcnn_v6_dinov3t`): DINOv3 ConvNeXt-Tiny
trunk, otherwise run 5's configuration. Tests pretraining quality at matched
capacity (49.6M vs 45.9M total). Batch 2 fits in 43.2 GB.

**Run 7** (queued): ResNet-50 with `--d4`, otherwise run 5's configuration.
Isolates augmentation as a single variable against run 5's 0.18963.

Between them these separate the two live hypotheses -- better features, and
more effective data -- rather than confounding them as runs 3 and 4 did.

## Run 6, result: DINOv3 loses, and the loss says why the result is ambiguous

DINOv3 ConvNeXt-Tiny trunk, run 5's configuration otherwise. Full 700-image
split:

| run | trunk | checkpoint | full-split AP | AP50 |
|---|---|---|---|---|
| 5 | ResNet-50 (COCO detector) | best, ep 12 | **0.18963** | 0.4626 |
| 6 | DINOv3 ConvNeXt-Tiny | best, ep 16 | 0.17084 | 0.4238 |
| 6 | DINOv3 ConvNeXt-Tiny | last, ep 19 | 0.16735 | 0.4142 |

DINOv3 loses by **0.0188**. Under the rule recorded in `lvm/backbones.py` before
the run started, a win would have been conclusive a fortiori and a loss
ambiguous. This is the loss, so it is ambiguous, and the training loss shows the
ambiguity is real rather than a formality.

### Run 6 underfit where run 5 overfit

| | final train_loss | final val AP |
|---|---|---|
| run 5 (ResNet-50, COCO heads) | **0.8842** | 0.1971 |
| run 6 (DINOv3, random heads) | **1.0750** | 0.1765 |

Run 5 ended 18% *better* fitted on the training set. Run 6 never reached the
regime where run 5 started losing generalisation — it is still underfit at
epoch 19, which is exactly what a detector whose FPN, RPN and both heads began
from random initialisation should look like after 20 epochs on 2,451 images.

So the 0.0188 deficit has two candidate causes that this experiment cannot
separate: DINOv3 features may genuinely be worse for this task, or 20 epochs
may simply be too few to train a detection stack from scratch. The opening
epochs make the second reading concrete — run 6 started at AP 0.0356 against
the baseline's 0.1393, a handicap worth more than the final gap.

### What settles it

`--scratch-heads` (commit `4f34857`, verified to carry ImageNet-V2 weights
318/318 while initialising FPN, RPN and heads randomly) reproduces exactly that
handicap on the ResNet side. An ImageNet-trunk arm with random heads, trained
for the same 20 epochs, differs from run 6 only in which trunk it carries, and
its comparison against run 6 is clean in both directions.

It is queued behind run 7, not ahead of it. Run 7 attacks the overfitting run 5
demonstrated and could actually improve the model; the control only interprets a
result that is already negative. Given a brief to build something better, the
improvement attempt earns the GPU first.

### Honest position on the pretraining hypothesis

`plan.md` ranked "a stronger pretrained backbone" as the only change likely to
close the 0.044 gap to SAM 3. That hypothesis is **not** refuted by run 6 — but
it is not supported either, and one attempt at it has now cost five GPU-hours
and returned an uninterpretable number. The honest summary is that the question
is still open and the experiment that was supposed to answer it needs its
control before it says anything at all.
