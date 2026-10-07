# Literature review log

This is a running record of papers, code, datasets and data releases relevant to
this project. A new dated entry is added each working day and covers what
appeared since the previous entry.

**Read this file before proposing a method or experiment.** A suggestion should
cite the entry it rests on. Don't re-propose anything recorded below as
*measured here* or *skip* unless new evidence appears. The ruled-out list with
measurements is in [`experiments.md`](experiments.md).

## Scope

- building footprint and instance segmentation in aerial imagery
- dense small-object segmentation
- query-based instance segmentation (Mask2Former, Mask DINO)
- SAM, SAM 2 and SAM 3: fine-tuning, adapters, distillation
- label-imagery misalignment and off-nadir footprint offset
- IGN and French open data: BD TOPO, BD ORTHO, ORTHO Express, LiDAR HD, FLAIR

## Entry format

Each entry covers:
- **Searched:** the window it covers and the queries run.
- **Items:** each item gives its title, link, date, the finding (with numbers),
  and a verdict:
  - `act`: worth trying now, with a note on how.
  - `watch`: interesting, not yet actionable.
  - `skip`: not worth trying, with the reason.
  - `measured here`: already tested in this repo, with the result.

  Write **nothing new** when that is the truth.
- **Takeaway:** a sentence or two on what, if anything, changes for the project.

A SessionStart hook (`tools/daily_litreview_check.sh`, wired up in each user's
`.claude/settings.local.json`) reminds Claude when today's entry is missing.

---

## 2026-09-28 (baseline)

This entry sets the baseline. It condenses the deep research done that day by
two research agents, checked against primary sources, together with this repo's
own diagnostics.

### Methods: mask quality, detectors, building-specific

| item | finding | verdict |
|---|---|---|
| Bigger mask grid: DCT-Mask ([2011.09876](https://arxiv.org/abs/2011.09876)), DynaMask ([2303.07868](https://arxiv.org/abs/2303.07868)) | Moving from 28 to 56/64/128 **lowers** COCO AP (35.2 → 34.4 → 32.9) | **measured here:** with perfect boxes the 28x28 grid already reaches IoU 0.928, and 56x56 reaches 0.926 (`tools/mask_ceiling.py`) |
| Mask Scoring R-CNN ([1903.00241](https://arxiv.org/abs/1903.00241)) | Rescores detections by predicted mask IoU: +1.5 AP and +2.1 AP75 on COCO R50. Oracle rescoring adds a further +2.2–2.6 | `act`: our weakest metric is AP75. About 150 lines on torchvision `RoIHeads`. Run an oracle-rescoring diagnostic first |
| DCT-Mask ([2011.09876](https://arxiv.org/abs/2011.09876)) | Frequency-domain mask head: +1.3 AP and +2.1 AP75 on COCO, at the same speed | `watch`: the cheapest boundary head to port. Our mask ceiling suggests limited headroom |
| Mask Transfiner ([2111.13673](https://arxiv.org/abs/2111.13673)), RefineMask ([2104.08569](https://arxiv.org/abs/2104.08569)), BMask ([2007.08921](https://arxiv.org/abs/2007.08921)) | +1.4 to +2.6 AP on COCO, mainly boundary AP. They need detectron2 or mmdet | `watch`: switching framework costs 3–5 days |
| PointRend ([1912.08193](https://arxiv.org/abs/1912.08193)) | +0.9–1.1 AP on COCO, but **AP_S drops 21.1 → 18.8** (Transfiner Table 9) | `skip`: 38% of our buildings are small |
| Cascade Mask R-CNN / HTC ([1906.09756](https://arxiv.org/abs/1906.09756)) | +1.2–1.3 AP on COCO. Its own paper finds a detection AP90 gain of +8.7 shrinks to +1.8 for masks | `watch`: better boxes do not fix masks |
| Mask2Former ([2112.01527](https://arxiv.org/abs/2112.01527)), Mask DINO ([2206.02777](https://arxiv.org/abs/2206.02777)) | On the WHU building benchmark, Mask2Former scores 69.2 AP vs Mask R-CNN's 65.6 (RSPrompter Table I). Mask2Former degrades at 300 queries | `watch`: the SAM 3-like family. Needs ≥ 400 queries. About a week of work |
| ViTDet, Co-DETR, RTMDet-Ins | +4–15 AP on COCO with large backbones | `skip` for now: cost approaches SAM 3's |
| HiSup, Frame Field Learning, PolyWorld, GCP | CrowdAI SOTA, but ~93% of CrowdAI val images leak into train (HiSup 79.4 → 65.4 AP on unseen images). HiSup and PolyWorld cannot represent shared walls | `skip`: poor fit for mask AP and for shared walls |
| Frozen-SAM box refinement ("SAM-det", RSPrompter [2306.16269](https://arxiv.org/abs/2306.16269)) | 61.8 vs 65.6 AP for Mask R-CNN on WHU, so it is **worse** | `skip` |
| Soft-NMS / Matrix NMS | +0.3–1.1 AP on COCO | **measured here:** box NMS 0.6 +0.0004, 0.7 −0.003 |
| Simple Copy-Paste + large-scale jitter ([2012.07177](https://arxiv.org/abs/2012.07177)) | +0.5–1.1 mask AP on COCO, far more at low data | **measuring:** run r9 (narrow jitter ×0.75–1.33, background-only paste) |
| EMA / SWA of weights | SWA: 34.7 → 35.5 AP (Mask R-CNN R50) | **measured here:** EMA +0.004 AP (B4) |
| SAHI sliced inference ([2202.06934](https://arxiv.org/abs/2202.06934)) | +6.8 AP on VisDrone (large frames, tiny objects) | `skip`: our 2x upsampling already acts like slicing |
| D4 test-time augmentation (sam3-ft-EOSC `tta.py`) | +6% relative AP for SAM 3 | **measured here:** −0.005 AP with mask fusion, −0.001 with score fusion. Mask R-CNN is not rotation-consistent |
| BONAI / LOFT ([2204.13637](https://arxiv.org/abs/2204.13637)) | Models the roof-to-footprint offset: +3.37 F1 over Cascade Mask R-CNN | `watch`: directly relevant to our label misalignment. Needs offset labels |
| Self-training with a stronger teacher (Copy-Paste Table 2) | +1.3 mask AP, or +2.3 combined with copy-paste | `watch`: SAM 3 as teacher on unlabelled same-region IGN tiles |

### Data and weights

| item | finding | verdict |
|---|---|---|
| **BD TOPO `batiment`** (IGN, Licence Ouverte 2.0) | 96–99% of central Paris, Lyon and Marseille buildings are cadastre-derived (the `origine_du_batiment` attribute): wall footprints, split per parcel. v2's labels come from this layer | `act`: generate 92/93/94 with `create_instance_seg_dataset_v2.py`. Same label convention as the test set |
| **BD ORTHO IRC 20 cm** | Free per département. RGB and IRC editions from the same flight | `act`: same-year imagery for 92/93/94 |
| **ORTHO Express** (true ortho, IGN) | Buildings are "not overturned" and less offset than BD ORTHO. Available as WMTS tiles only | `watch`: fixes the label alignment but changes the benchmark (the user must decide) |
| **FLAIR-HUB `LC-A_IR` Swin-B** (IGNF, Etalab 2.0) | Trained on 152k IGN 20 cm patches in NIR-R-G order `[4,1,2]`. Building IoU 83.86 | `watch`: a domain-matched backbone, but our DINOv3 result says a new trunk with random heads loses |
| FLAIR #1 / FLAIR-HUB datasets | Semantic labels only (19 classes) | `skip` as instance data |
| RGB vs NIR-R-G | IGN's own Swin-B: building IoU 83.77 (RGB) vs 83.86 (IRC) | `skip`: channel order does not matter once fine-tuned |
| SpaceNet 2 Paris (AOI_3), original S3 GeoJSON | Instance polygons, WorldView-3 at 0.3 m, suburban | `skip`: wrong sensor and density |
| Netherlands PDOK CIR + BAG; NRW DOP + ALKIS | Open footprints plus CIR orthophotos | `skip` for now: cross-border shift. D001 showed that off-distribution data hurts (−0.046) |
| WHU aerial, CrowdAI | RGB, low density | `skip`: the public-corpus pretraining gave +0.005 |
| Microsoft Global ML Building Footprints | 7.4M French footprints, IoU 65.1% on Europe | `skip`: worse than BD TOPO |

### Diagnostics from this repo (context for everything above)

These come from `tools/error_analysis.py` on B3, over 20,669 v2-test buildings:
- **Pipeline:** RPN proposals cover 80.4% of buildings, the detection ceiling is
  64.5%, and 62.7% are matched.
- **Misses:** poor outline 15%, not detected 10% (mostly small), merged with a
  neighbour 8%.
- **Small buildings:** recall is 31%.
- **Label convention:** the ground-truth overlays confirm the misalignment and
  the parcel splits.

**Takeaway.** The largest lever is data with the test set's own label
convention, meaning more Paris-region départements. The cheapest model-side
bet is mask-quality rescoring aimed at AP75. Label alignment caps every model;
whether to rebuild the benchmark on ORTHO Express is the user's decision.

---

## 2026-09-29

**Searched.** The window is ~2026-08-29 to 2026-09-29. I queried the arXiv API
sorted by submission date, ran web searches, and read the primary sources: arXiv
abstract pages, GitHub repos, data.gouv.fr, cartes.gouv.fr and the OSM-FR forum.
I read abstracts and READMEs, not full PDFs, so the numbers are as those pages
state them. The queries are listed at the end of this entry.

### Building / dense small-object instance segmentation

| item | date | finding | verdict |
|---|---|---|---|
| [PolyTopoBench](https://arxiv.org/abs/2609.32856), NeurIPS 2026 D&B | 09-26 | A polygon-topology benchmark: 11 methods (Mask R-CNN+poly, SAM2+poly, HiSup, FFL, GCP, PolyWorld and others) on Inria (France, 254k instances) and Deventer. Methods "degrade substantially on polygons with holes". Code and data are released | `watch`: a Mask R-CNN vs SAM2 comparison on French aerial imagery, but the metric is polygon topology, not mask AP |
| [LACE](https://arxiv.org/abs/2609.26549), box-supervised tree crowns | 09-22 | Trained on 900 boxes with a frozen DINOv3 ViT-L, it reaches mask AP50 0.663 vs 0.626 for mask-supervised Mask R-CNN on OAM-TCD. No code yet | `watch`: box-only supervision could sidestep offset and parcel-split masks, but it is trees, AP50 only, and has no code |
| [Urban building instance seg.](https://arxiv.org/abs/2609.19631) | 09-17 | Point clouds | `skip`: 3D data |
| [MariSat](https://arxiv.org/abs/2608.29852) | 08-30 | Vessel dataset, fine-tunes SAM 3 and YOLO11 | `skip`: not buildings |

### SAM 3 / SAM 2 fine-tuning, adapters, distillation

| item | date | finding | verdict |
|---|---|---|---|
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | 09-18 | No new release. The latest is still SAM 3.1 (2026-03-27); the window's commits are video-tracker fixes | `skip` |
| [SAM3-LoRA, structural defects](https://arxiv.org/abs/2609.00469) | 08-31 | LoRA on 0.12–1.3% of parameters. Failure mode: the presence head decouples from the prompt when trained only on positive tiles, fixed by hard-negative prompting. No code | `watch`: this matters to the SAM 3 baseline only if it was trained on building-positive tiles alone, and `building` is its only class |
| [SRPR-Net](https://arxiv.org/abs/2609.24226) | 09-21 | Refines detector boxes before SAM decodes them. [Code](https://github.com/JeremyXSC/SRPR-Net). No benchmarks or numbers in the abstract | `watch`: box-prompted *frozen* SAM was worse than Mask R-CNN on WHU (2026-09-28 entry). Needs numbers before trying |
| [WireSeg-32K](https://arxiv.org/abs/2609.03102), [VPRef](https://arxiv.org/abs/2609.16486) | 09-02, 09-15 | SAM 3 fine-tunes for wires and for referring segmentation | `skip`: different tasks |

No new work on distilling SAM 3 into a small building detector.

### Footprint/roof alignment, misaligned labels, true ortho

| item | date | finding | verdict |
|---|---|---|---|
| [Align and Segment (AnS)](https://arxiv.org/abs/2607.10841), [code](https://github.com/venkanna37/align-and-segment) (MIT) | v1 07-12. The repo now reads *accepted at ECCV 2026* and has new DINOv3 weights on Hugging Face (date of change not shown) | A spatial transformer learns a per-label affine shift to align misaligned building labels with the imagery, self-supervised, with no clean ground truth needed | `act`: the closest match to our roof-vs-cadastre offset. Re-align the BD TOPO training masks with AnS, then retrain Mask R-CNN. It fixes offset, not parcel splits. Scoring still uses the original test labels, so gains on the benchmark are not guaranteed |
| [ObliCity / DragRoof](https://arxiv.org/abs/2607.25210) | 07-28 | A roof-to-ground displacement benchmark. The data is available only by request (email/WeChat/Baidu) | `watch`: not openly available |

### Datasets / IGN releases

| item | date | finding | verdict |
|---|---|---|---|
| [IGN Ortho-Express](https://www.data.gouv.fr/datasets/ortho-express) (true ortho, correlation DSM, 20 cm, **IRC** and RVB) | updated 09-15 | The 2026 départements are being published progressively (Rhône/Ain 08-23 through Eure-et-Loir/Loiret 09-11). **No Paris (75) or Île-de-France listing was found** | `act`, conditional: check data.geopf.fr for a D075 tile. If it exists, BD TOPO rasterised onto it removes most of the roof/outline offset. This changes the benchmark (the user must decide) |
| LiDAR HD MNS/MNT | 09-06 | 507,791 tiles published, 78.4% of communal area, 50 cm GeoTIFF, etalab-2.0. Paris coverage not confirmed | `watch`: a DSM input channel, or DSM-based label re-projection |
| BD TOPO | — | No release in the window. The next edition (263) is due in October 2026 | `skip` for now |
| [Global building datasets compared](https://arxiv.org/abs/2609.28154) | 09-23 | 7 products over 135 areas; Overture has the best vector F1 at 0.786 | `skip`: continental products |
| FLAIR | — | No new release | — |

### COCO/LVIS instance segmentation (small objects, boundary)

Nothing new with released code and AP_S, AP75 or boundary gains.
[iFAN](https://arxiv.org/abs/2608.03216) (v2 08-07) gives +1.30 AP for mask
transformers: `skip`, since it doesn't apply to Mask R-CNN.

**Takeaway.** Two new leads both target the diagnosed label misalignment, the
largest limit found:
1. **Align and Segment**, which re-aligns the training labels to the imagery.
2. **Ortho-Express**, IGN's true ortho, if Paris becomes available.

Neither changes the recommendation to generate 92/93/94 data first. Both are
cheap to check:
- AnS has code and weights.
- Ortho-Express needs one catalogue lookup for D075.

**Queries.**
- arXiv API, sorted by submission date:
  - building AND (segmentation OR footprint)
  - cs.CV AND building AND (aerial | satellite | remote sensing)
  - cs.CV AND "instance segmentation"
  - (SAM | segment anything) AND (remote sensing | aerial | satellite)
  - cs.CV AND (footprint | roof | off-nadir | orthophoto)
  - cs.CV AND (noisy labels | label noise | misaligned) AND segmentation
  - (IGN | FLAIR | BD TOPO | LiDAR HD | France) AND (building | aerial)
  - cs.CV AND (boundary AP | AP75 | small objects | mask quality) AND COCO
- Web searches on SAM 3 releases, IGN news for September 2026, Ortho-Express
  coverage and LiDAR HD diffusion.

---

## 2026-09-30

**Searched.** The window is 2026-09-29 to 2026-09-30. The arXiv API feed was last
updated at 02:10Z, so this covers 09-29 submissions and revisions only. The
searches ran the same arXiv API and web queries as yesterday, plus a new one:
"scale augmentation that preserves small objects", prompted by r9's AP_small
regression.

| item | date | finding | verdict |
|---|---|---|---|
| [UniBuild](https://arxiv.org/abs/2609.37031) | 09-29 | One DINOv3-B + HR-DPT model trained on 10 high-resolution sets plus Planet and Sentinel-2. It adds a structure-tensor direction loss and a "saddle-aware" loss against false positives in the narrow gaps between buildings. INRIA: IoU 83.29, Boundary-IoU 70.86. The saddle loss raises Planet mIoU from 45.77 to 47.80. Semantic output, RGB only, inference-only repo with no licence | `watch`: not our instance CIR task. The gap loss is an idea for our 8% of buildings merged with a neighbour (2026-09-28 diagnostics) |
| [HyperSAM](https://arxiv.org/abs/2609.37340), IEEE GRSM | 09-29 | SAM 3 RGB branch frozen, plus a trainable spectral side encoder with zero-init ControlNet-style injection, trained on confidence-weighted SAM 3 pseudo-masks. No numbers in the abstract | `watch`: a clean way to feed NIR into SAM 3. Relevant to the SAM 3 project, not ours |
| facebookresearch/sam3, efficientsam3 | — | No commits since 09-18, no release since v0.4.0 (06-11) | `skip` |
| Footprint/roof alignment, noisy labels | — | **Nothing new** | — |
| IGN Ortho-Express 2026 | 09-30 | Jura (39) added. **Still no Paris or Île-de-France** ([OSM-FR thread](https://forum.openstreetmap.fr/t/ign-ortho-express-2026/41923?page=4)) | `watch` |
| **Géoportail closes 2026-09-30**, redirecting to cartes.gouv.fr ([source](https://www.banquedesterritoires.fr/cartesgouvfr-veut-conquerir-le-grand-public-le-geoportail-ferme-en-septembre-2026)) | 09-30 | Old links must be converted by 2026-12-31 | **checked:** this repo has no IGN endpoints. The dataset scripts on the team share (`download_bdtopo.py`, `download_ign_ten_split*.py`) use `data.geopf.fr`, the Géoplateforme download API, not geoportail.gouv.fr. `data.geopf.fr/telechargement/resource/BDTOPO?zone=D075` answered HTTP 200 on 09-30. No action needed |
| LiDAR HD MNS coverage of Paris | — | Not verifiable: the progress map needs a login and the public pages give no percentage | `watch` |
| BD ORTHO | — | No September update found | — |
| COCO/LVIS small-object or boundary methods; scale augmentation that keeps AP_small | — | **Nothing new** | — |

**Takeaway.** Nothing here changes what to try next. The data pipeline is not
affected by the Géoportail closure. The only new idea worth keeping is
UniBuild's gap loss for merged neighbours.

---

## 2026-10-01

**Searched.** The window is 2026-09-30 to 2026-10-01. The arXiv feed was last
updated at 2026-10-01T08:37Z; the newest cs.CV submission it lists is from
09-30 17:59Z, so 10-01 papers fall to the next entry. The searches ran the
usual arXiv API and web queries, plus direct checks of the Géoplateforme BD
TOPO listings, the Ortho-Express WMS and the torchvision source. One new
question was added: why r9/r10's raw weights collapse on the final epoch while
the EMA weights don't.

| item | date | finding | verdict |
|---|---|---|---|
| [COBICount](https://arxiv.org/abs/2609.39366) | 09-30 | Building/vehicle counting by density map, suppressing false responses on repeated structures such as roof edges and parking grids. No masks | `skip`: counting |
| [UAV urbanised-area recognition](https://arxiv.org/abs/2609.40212) | 09-30 | Pixel classification at 10–15 mm | `skip` |
| [AdvPCS](https://arxiv.org/abs/2609.39265), NeurIPS 2026 | 09-30 | A universal adversarial perturbation that transfers across SAM 3 prompt types | `skip`: robustness |
| [DCM-SAM](https://arxiv.org/abs/2609.38811) | 09-30 | Frozen SAM plus per-class Conv-LoRA experts (4.4% trainable). 64.2% IoU on few-pixel CT pores; code released | `watch`: a small-object adapter recipe, far domain |
| sam3 / sam2 repos | — | No pushes since 09-18 | — |
| Alignment / noisy labels / true ortho | — | **Nothing new** | — |
| **BD TOPO edition 263** (2026-09-15) | 09-29/30 | **Released regionally.** Île-de-France `R11` GPKG posted 09-30 (1.27 GB, HTTP 200), France-wide `FRA` posted 09-30. **Paris `D075` is not posted yet**; its latest is still 06-15. The change log (v3.6) lists TAAF, transport and parking changes, nothing on buildings | `act`: diff R11's buildings against our 06-15 labels. Footprint changes are probably small |
| Ortho-Express | — | The WMS has an `IRC-EXPRESS.2026` layer, but a map request over central Paris returns an empty image while BD ORTHO IRC returns a full one. Read as no Paris coverage yet (inferred) | `watch` |
| BD ORTHO | — | Last updated 09-07 | — |
| [Hard-Region Supervision](https://arxiv.org/abs/2609.38714) | 09-30 | A training-only extra head and loss on the baseline's hard regions. #1 on Waymo video panoptic (wSTQ +2.4 over DVIS++), no inference cost. No COCO result, no code | `watch`: transferable to small buildings, unproven |

**The raw-vs-EMA final-epoch collapse.** These sources are older than the
window but answer an open question from r9/r10.
- **What the architecture has:** torchvision `maskrcnn_resnet50_fpn_v2` uses
  trainable `nn.BatchNorm2d` in the backbone/FPN, the 4-conv box head and the
  mask head (verified in the source).
- **[Wu & Johnson, "Rethinking 'Batch' in BatchNorm"](https://arxiv.org/abs/2105.07576) (2021).**
  - Mask R-CNN heads with BatchNorm trained at one image per GPU score
    30.7 box / 27.9 mask AP using their running statistics, against 41.5 / 37.0
    using per-image proposal statistics.
  - Running averages lag the weights; the fix is to recompute them on frozen
    weights (*PreciseBN*).
  - `act`: recompute BatchNorm statistics for r10's raw `last.pt` on
    un-augmented training tiles and re-score. If it recovers, the statistics are
    the cause. Our batch is 2 tiles, close to their failing case.
- **[Morales-Brotons et al., "EMA of Weights"](https://arxiv.org/abs/2411.18704) (TMLR 2024).**
  - BatchNorm statistics, not the weights, limit EMA; recomputing them restores
    performance.
  - Separately, with ~40% label noise the EMA model peaks around learning rate
    0.4 and degrades once the rate is decayed far enough to memorise noisy labels.
  - `watch`: the alternative explanation, memorising our misaligned BD TOPO
    labels at the bottom of OneCycle. If PreciseBN doesn't fix it, try a higher
    final learning rate.

**Takeaway.** There is a cheap, well-founded test for the raw-weight collapse:
recompute BatchNorm statistics on the raw checkpoint, about an hour of GPU. If
it works, the final raw weights become usable and the EMA's role is better
understood. BD TOPO's new edition for Île-de-France is out, which matters for
the 92/93/94 data plan. Use edition 263 for any new départements, and check
whether Paris's labels change. Nothing changes the priority order otherwise.

---

## 2026-10-02

**Searched.** The window is since the 10-01 entry. The cs.CV feed was last updated
2026-10-02T08:31Z, newest submission 10-01 17:59 UTC, so this covers late 09-30
and all of 10-01. The searches ran the usual arXiv API queries, IGN Atom/WMTS
probes and GitHub activity checks, plus one new topic: late memorisation of noisy
masks, prompted by the raw-weight collapse in r9/r10 but not r11.

| item | date | finding | verdict |
|---|---|---|---|
| Building / dense small-object instance segmentation | — | **Nothing new**. Only off-task aerial items (2609.40212, 2610.01870, 2610.00693) | — |
| [SAM3-ASH / Generalized Presence Token](https://arxiv.org/abs/2610.01022) | 10-01 | Batches N text prompts so SAM 3 encodes the image once. MOTS20 SOTA zero-shot, peak memory under 25 GB | `skip`: video/tracking inference; we use one prompt |
| [MoSA](https://arxiv.org/abs/2609.39785) | 09-30 | Unsupervised segment-anything from motion pseudo-labels, "comparable to supervised SAM" zero-shot | `skip` |
| sam3 / sam2 / detectron2 repos | — | No commits or releases since 09-29 (detectron2: CI only) | — |
| Alignment, noisy labels, true ortho (new work) | — | **Nothing new** | — |
| **BD TOPO 263 for D075** | posted 10-01 | `BDTOPO_3-5_TOUSTHEMES_GPKG_LAMB93_D075_2026-09-15` (210.6 MB .7z) is now on data.geopf.fr. SHP not yet | `act`, optional: diff against v2's 06-15 labels to measure label drift. It cannot change the benchmark, since test labels are fixed |
| Ortho-Express | 10-02 | `IRC-EXPRESS.2026` tiles return 404 over central Paris, the 11e and Saint-Denis, while Rennes returns imagery. Sampled points only | `watch`: still no Paris |
| [RankSEG with spatial dependence](https://arxiv.org/abs/2609.38930) | 09-30 | Inference-time Dice/IoU-optimal decoding in O(d log d), no retraining; largest gains claimed on small/low-contrast objects; code released. Semantic segmentation only, no figures in the abstract | `watch`: applying it to per-instance mask logits would be our own untested extension |
| [GRACE](https://arxiv.org/abs/2610.01409) | 10-01 | Post-hoc box uncertainty under adversarial attack | `skip` |

**Late memorisation of noisy masks.** These sources are older than the window.
- **[Benchmarking Label Noise in Instance Segmentation (COCO-N)](https://arxiv.org/abs/2406.10891) (2024).**
  - Mask R-CNN R50 mask AP falls from 34.6 to 31.8 / 30.3 / 28.4 under easy, medium
    and hard *spatial* mask noise. Boundary AP falls 20.6 → 16.3.
  - Box AP drops less than mask AP.
  - A symmetric cross-entropy loss recovers only +0.5 mAP.
  - `act`, as a reference point: it is a direct benchmark of our failure mode. It
    predicts that off-the-shelf noise-robust losses will barely help, so they are
    not worth a run on their own.
- **[ADELE](https://arxiv.org/abs/2110.03740) (CVPR 2022, [code](https://github.com/Kangningthu/ADELE)).**
  - Fits a curve to each class's *training* IoU. When its slope has fallen past a
    threshold, it starts correcting labels with the model's own predictions, plus a
    multi-scale consistency term.
  - VOC weakly supervised: 71.6 / 72.0 mIoU.
  - `watch`: its training-IoU slope is a cheap detector of memorisation onset, and
    could set early stopping or a learning-rate floor for long runs. Transfer to
    instance masks is our inference.
- [ELR](https://arxiv.org/abs/2007.00151): the base early-learning regulariser
  ADELE builds on. Background.

**Takeaway.**
- **Nothing new changes the plan.**
- **The memorisation literature explains r9/r10's late raw collapse,** with ADELE's
  slope test as a way to detect it. COCO-N says not to expect much from a
  noise-robust loss alone.
- **The fresh D075 labels give an optional measurement:** how much do Paris's
  footprints move between editions?

---

## 2026-10-05

**Searched.** The window is 2026-10-02 to 10-05. The feed was last updated
2026-10-05T08:17Z, newest submission 10-02 17:58Z. Friday-afternoon and weekend
submissions arrive in Tuesday's batch, so **the next entry should re-check 10-02
to 10-05**. The arXiv API was queried by submission date. The boolean queries
misparsed, so the newest 600 cs.CV/eess.IV submissions and the 600 most recently
updated cs.CV papers were also keyword-filtered. Web, IGN and GitHub checks as
usual. New this time: evidence on the four candidate next steps.

| item | date | finding | verdict |
|---|---|---|---|
| Building / dense small-object instance segmentation | — | **Nothing new** (nearest: [SigLIP2 aerial fire-risk classification](https://arxiv.org/abs/2610.03689), classification only) | — |
| [When Predicting Nothing Beats SAM 3](https://arxiv.org/abs/2610.02946) | 10-02 | An empty-mask predictor beats SAM 3 on VOS J&F when targets are rarely visible; proposes Volumetric J&F | `skip`: video evaluation |
| [ViTok multi-teacher distillation](https://arxiv.org/abs/2610.02903) | 10-02 | SigLIP2 + DINOv3-L teachers; PHI-S balancing recovers ADE20K 46.5 → 48.5 mIoU | `skip`: backbone pretraining |
| [3D dendrite instance segmentation](https://arxiv.org/abs/2610.03332) | 10-02 | YOLO prompts SAM, nnU-Net refines (Dice 0.93); dense regions are the main failure | `skip`: microscopy |
| sam3 repo | — | No commits since 09-18; SAM 3.1 is still the latest | — |
| Alignment / noisy labels / true ortho (new work) | — | **Nothing new** | — |
| IGN Ortho-Express 2026 ([OSM-FR thread](https://forum.openstreetmap.fr/t/ign-ortho-express-2026/41923), pp. 4–5) | 09-30 → 10-02 | Added 22, 25, 61 (complete) and 69. Aircraft reportedly not flown since 08-23. **Still no Paris, 92, 93 or 94** | `watch` |
| LiDAR HD, BD ORTHO | — | Nothing dated in the window | — |
| COCO/LVIS small-object or boundary methods | — | **Nothing new** | — |

**Evidence on the candidate next steps.** These sources are older than the window.
- **[RSPrompter](https://arxiv.org/abs/2306.16269) Table I, WHU aerial buildings**
  (read from the PDF).

  | model | mask AP | AP75 |
  |---|---|---|
  | Mask R-CNN | 65.6 | 76.7 |
  | **Mask Scoring R-CNN** | 66.9 (+1.3) | 77.5 (+0.8) |
  | Mask2Former | 69.2 (+3.6) | 79.3 |
  | **frozen SAM backbone + Mask R-CNN heads** | 70.1 (+4.5) | 81.0 |
  | RSPrompter-query | 72.5 | 82.9 |

  On the much smaller NWPU dataset, Mask2Former was *worse* than Mask R-CNN (58.8
  against 59.7).

  `watch`: this roughly ranks the candidates *on buildings*.
  - Mask-IoU rescoring is a small gain, below its COCO figure.
  - A query-based detector helps on sparse 0.3 m imagery but is not robust on small
    data.
  - SAM features carry the most, which is indirect support for distilling from
    SAM 3.

  WHU is sparse and clean, unlike dense Paris.
- **[OMAF: weakly supervised object-level offset correction for misaligned building labels](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Revisiting_the_Necessity_of_Full_Accuracy_Weakly_Supervised_Object-Level_Offset_CVPR_2026_paper.html) (CVPR 2026).**
  - Per-building offset correction of footprint labels on non-orthorectified
    imagery, with under 1% of the data hand-annotated.
  - Gains of up to +40.6 mIoU: UNetFormer 35.8 → 76.4 on Islahiye; DeepLabV3+
    +17.9.
  - Semantic mIoU only, not instance AP. The numbers come from a notes page and the
    CVF snippet, not the full PDF, and the cited repo returned 404.
  - `watch`: the strongest published evidence that re-aligning footprint labels
    pays. The caveat from 09-29 stands: our test labels stay misaligned.

**Takeaway.** It supports the order already proposed:
1. Gate Mask Scoring R-CNN on an oracle-rescoring check. On buildings its gain is
   ~+1.3 AP, not COCO's +2.1 AP75.
2. Rank SAM 3 distillation above a query-based detector. SAM features gave the
   largest gain on WHU, and Mask2Former was unreliable on small data.

---

## 2026-10-06

**Searched.** The window is 2026-10-02 to 10-06. It **re-checks the Friday-afternoon
and weekend span flagged on 10-05, now announced**:
- cs.CV RSS build 10-06 04:10Z, up to 2610.06852.
- The newest API submission is 10-05 17:59Z.
- 781 cs.CV/eess.IV entries were parsed and keyword-filtered: 10-02 131, 10-03 60,
  10-04 74, 10-05 116.
- Revised (v2+) versions inside the window are not covered.

Plus a new topic: learned mask-quality scoring, now that ms1 has measured it on
this data.

| item | date | finding | verdict |
|---|---|---|---|
| Building / dense small-object instance segmentation | — | **Nothing new** | — |
| [LoDEOT: offset tokens for footprint extraction from off-nadir imagery](https://arxiv.org/abs/2610.05899) | 10-05 | Query-based model with a 5-D roof-to-footprint offset token. BONAI FAP50 54.58, mEPE 5.23 px, +7.56 to +16.85 pp over the end-to-end baselines tested; five datasets; no code | `watch`: the most on-topic paper on roof-vs-footprint offset, but off-nadir, DETR-style, no code |
| [PAR: prompt and refinement for noisy-label infrared small targets](https://arxiv.org/abs/2610.05918) | 10-05 | SAM and a detector correct each other's masks; no numbers in the abstract | `skip` |
| [Training-dynamics detection of noisy keypoint labels](https://arxiv.org/abs/2610.06423) | 10-05 | 91.9% F1 at flagging noisy keypoints; filtering them gives up to +7.4 AP on COCO pose | `skip`: keypoints. The same idea as ADELE/ELR, already logged |
| SAM 3 / SAM 2 | — | **Nothing new** (GitHub rate-limited; web search found only older items) | — |
| IGN Ortho-Express | — | Nothing new; still no Paris (inferred from the latest forum post, 09-30) | `watch` |
| COCO/LVIS small-object or boundary methods | — | **Nothing new** | — |

**Learned mask-quality scoring.** These sources are older than the window.
- **[HYDRA: "Queries Knew More Than We Thought"](https://arxiv.org/abs/2609.20283) (v1 2026-07-30).**
  - A ~1.6M-parameter MLP trained only on a frozen model's cached outputs. It
    re-picks among existing masks using objectness, logit moments,
    agreement/exclusivity between overlapping masks, and image-level entropy.
  - It recovers **43.5% of the oracle gap on Mask2Former** (54.4% on ADE20k),
    27.5% on Mask DINO, and **32.5–78.3% on SAM 3**.
  - No Mask R-CNN or instance-AP results, no code.
  - `act`, as an idea: our MaskIoU head recovers ~14% of its oracle gap
    (0.013 of 0.096). HYDRA suggests 30–50% is reachable for a frozen-model
    re-ranker given richer context features. The candidates are agreement with
    overlapping detections and image-level statistics, beyond the single RoI
    features the head uses now.
- **[iFAN](https://arxiv.org/abs/2608.03216) (v2 2026-08-07, logged 09-29 as `skip`).**
  - Its "adjusted probability-mask ranking" aligns query scores with mask quality
    during training, for +1.30 AP.
  - `watch`: it is evidence that aligning scores with mask quality *during
    training* pays, which supports the joint run ms2. It is DETR-only.

**Takeaway.**
- **The two queued runs are what this literature supports:** ms1b, a longer
  head-only run, and ms2, joint training.
- **HYDRA sets a target and a design for the next step if ms1b plateaus:** a
  re-ranker that sees each detection's context, not just its own RoI, could plausibly
  take the +0.013 to +0.03–0.05.

## 2026-10-07

**Searched.** The window is 2026-10-05 18:00Z to 10-07. It picks up where 10-06 stopped:
- arXiv API, cs.CV OR eess.IV by submittedDate. 136 entries: 10-05 22, 10-06 114.
  The newest is 10-06 17:59Z; 10-06 evening and 10-07 are not announced yet.
- Keyword filter over titles and abstracts (building, footprint, roof, instance
  segmentation, mask scoring/quality/IoU, rescoring, SAM, aerial, remote sensing,
  off-nadir, cadastre, label noise, small object, boundary, Mask R-CNN, LiDAR).
- Web: SAM 3 releases, IGN ORTHO Express 2026 (OSM France forum thread, page 4),
  mask-quality re-ranking for frozen detectors, misaligned footprint labels.

Revised (v2+) versions inside the window are not covered.

| item | date | finding | verdict |
|---|---|---|---|
| [RBMatch: class rebalancing for semi-supervised building footprint extraction](https://arxiv.org/abs/2610.07698) | 10-06 | Self-training with class-specific thresholds, loss reweighting and distribution alignment. Semantic segmentation on WHU/INRIA/Massachusetts at 1–10% labels; best IoU, +1.37 IoU on Massachusetts at 1%. No code | `skip`: semantic, low-label regime. We are fully labelled; our limit is label alignment, not label count |
| [Fine-tuning nuclei segmentation models on pseudo-labels by difficulty](https://arxiv.org/abs/2610.07711) | 10-06 | Dense instance segmentation. Expert labels on the hard/medium cases beat 5,901 easy pseudo-labels; best composition is model-dependent | `skip`: pathology. The general point (a few hundred corrected hard cases beat many easy ones) is a reminder for any relabelling effort |
| [How many independent samples does a satellite image contain?](https://arxiv.org/abs/2610.08227) (TGRS) | 10-06 | Effective sample size of an n×n image with correlation range r is Θ(n²/r²); random holdout understates confidence-interval width by a factor ∝ r | `watch`: our ±0.002 noise estimate comes from two seeds on a fixed split. It says nothing about split-to-split variance on spatially correlated Paris tiles, so small gains (< ~0.005) should be read cautiously |
| SAM 3 / SAM 2 | — | **Nothing new**. Latest is SAM 3.1 (March 2026, video multiplexing; image model unchanged) | — |
| IGN ORTHO Express 2026 | — | Service live on the Géoplateforme since 09-28. The forum's latest post (09-30, Jura) still lists no Paris or 92/93/94 | `watch` |
| Mask-quality re-ranking, misaligned footprint labels | — | **Nothing new**. Hits were older work already logged (Align and Segment 09-29, OMAF 10-05) or pre-2024 (Mask Frozen-DETR, which also uses a mask-scoring head on a frozen detector) | — |
| Building / dense small-object instance segmentation (instance-level, aerial) | — | **Nothing new** | — |

**Measured here since the last entry** (details in [experiments.md](experiments.md)):
- **ms1b** (head-only Mask Scoring, 12 epochs): test 0.2298, no gain over ms1's
  0.2307. A head on frozen r10 RoI features saturates at ~14% of the oracle gap.
- **ms2** (Mask Scoring trained jointly from COCO, the paper's setup): test 0.2310,
  ties ms1. iFAN's case (10-06) that aligning scores with mask quality *during
  training* pays does not carry over here; joint training is now `measured here`.

**Takeaway.**
- The Mask Scoring line is worth +0.013 and has stopped giving more from RoI
  features, whether trained alone or jointly.
- The remaining leads are both from earlier entries:
  1. **A HYDRA-style context re-ranker** (10-06, `act`): a small model on cached
     r10 + MaskIoU outputs that adds what the RoI head cannot see (overlap and
     agreement with neighbouring detections, image-level statistics). It is cheap
     to try because it trains on cached detections, not images.
  2. **Label realignment** (Align and Segment 09-29, OMAF 10-05): it attacks the
     cause rather than the ranking, but our test labels stay misaligned, so test AP
     may not show the gain.
- Nothing published today changes that ordering.
