#!/bin/bash
# al1: r10's recipe on per-building realigned training labels, then ms1's Mask
# Scoring head on top. Labels move only where ms1 and ms2 agree on the shift
# (tools/realign_labels.py; evidence in runs/label_offset). Model selection uses
# the ORIGINAL v2 valid labels; the benchmark is the ORIGINAL v2 test labels. A
# realigned copy of valid/test is scored as a secondary measure only.
cd ~/work/lvm_EOSC || exit 1
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python; R=runs/al1_aligned; H=runs/al1_ms; D=$R/dets
V2=data/ign_building_v2/building
A=data_local/ign_building_v2_aligned/building          # train realigned; valid/test original
E=data_local/ign_building_v2_aligned_eval/building     # valid/test realigned (secondary)
mkdir -p $D $H
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)  $(date '+%F %T') ==="

echo "=== 1. detections  $(date +%T) ==="
for job in ms1:runs/ms1_head_only/best.pt:train ms2:runs/ms2_joint/best.pt:train ms2:runs/ms2_joint/best.pt:test; do
  IFS=: read n c s <<< "$job"
  [ -f $D/${s}_$n.pt ] && continue
  $P tools/dump_detections.py --ckpt $c --data $V2 --split $s --keep-rles --out $D/${s}_$n.pt 2>&1 \
    | grep -v -i warn | tail -1
  [ -f $D/${s}_$n.pt ] || { echo "dump $s $n failed"; exit 1; }
done
ln -sf ../../rerank1/dets_test.pt $D/test_ms1.pt
ln -sf ../../rerank1/dets_valid.pt $D/valid_ms1.pt
ln -sf ../../label_offset/dets_valid_ms2.pt $D/valid_ms2.pt

echo "=== 2. realign  $(date +%T) ==="
$P tools/realign_labels.py --data $V2 --split train --dets $D/train_ms1.pt $D/train_ms2.pt --out-root $A || exit 1
for s in valid test; do
  $P tools/realign_labels.py --data $V2 --split $s --dets $D/${s}_ms1.pt $D/${s}_ms2.pt --out-root $E || exit 1
done
for s in valid test; do [ -e $A/$s ] || ln -s "$(readlink -f $V2/$s)" $A/$s; done
ls -la $A; $P - <<PY || exit 1
import json
for root, s in [("$A", "train"), ("$A", "valid"), ("$A", "test"), ("$E", "valid"), ("$E", "test")]:
    d = json.load(open(f"{root}/{s}/_annotations.coco.json"))
    n = sum(1 for a in d["annotations"] if "realign_shift" in a)
    print(f"{root}/{s}: {len(d['images'])} images, {len(d['annotations'])} buildings, {n} moved")
PY

echo "=== 3. train al1 (r10 recipe)  $(date +%T) ==="
$P -u -m lvm.train --data $A --out $R --epochs 36 --batch-size 2 --lr 0.005 --workers 16 \
  --max-dets 400 --image-size 2048 --val-images 200 --val-every 2 --ema 0.9998 \
  --scale-jitter 1.0,1.33 --copy-paste 30 > $R/train.log 2>&1 < /dev/null
echo "=== training exited $(date '+%F %T') ==="; tail -2 $R/train.log
[ -f $R/best.pt ] || { echo "no best.pt"; exit 1; }

echo "=== 4. Mask Scoring head (ms1 recipe)  $(date +%T) ==="
$P -u -m lvm.train --data $A --out $H --seed 1 --init-from $R/best.pt --mask-scoring \
  --freeze-except-maskiou --epochs 4 --batch-size 2 --lr 0.005 --workers 16 --max-dets 400 \
  --image-size 2048 --val-images 200 --val-every 2 --scale-jitter 1.0,1.33 --copy-paste 30 \
  > $H/train.log 2>&1 < /dev/null
echo "=== head exited $(date '+%F %T') ==="; tail -2 $H/train.log

echo "=== 5. scoring  $(date +%T) ==="
score() {  # ckpt data tag
  echo "=== $1 on $3  $(date +%T) ==="
  $P -m lvm.score --ckpt $1 --data $2 --split test --max-dets 300 --out $4 2>&1 \
    | grep -E '"(AP|AP75|AP_small|matched_mask_iou|boundary_iou)"|Traceback'
}
score $R/best.pt $V2 "original test" $R/score_v2test_best.json
[ -f $H/best.pt ] && score $H/best.pt $V2 "original test" $H/score_v2test_best.json
score $R/best.pt $E "realigned test" $R/score_alignedtest_best.json
[ -f $H/best.pt ] && score $H/best.pt $E "realigned test" $H/score_alignedtest_best.json
score runs/r10_jitter_up/best.pt $E "realigned test (r10 baseline)" $R/score_alignedtest_r10.json
score runs/ms1_head_only/best.pt $E "realigned test (ms1 baseline)" $R/score_alignedtest_ms1.json
echo "=== all done $(date '+%F %T') ==="
