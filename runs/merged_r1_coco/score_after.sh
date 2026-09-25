#!/bin/bash
# Queued behind runs/merged_r1_coco: GPU 1 is the only free GPU, and scoring
# alongside training risks an OOM in a 10-hour run. All scoring uses the
# fixed scorer (no in-model score threshold, every image counted).
#
# v2 test is the 1,402 pinned images, byte-identical in the merged set and
# never used for training by any run -- the only split where the merged model
# and the v2-trained baselines can be compared. v2 valid is NOT usable for the
# merged model: 492 of its 700 images are in merged train.
cd /home/ankur/work/lvm_EOSC || exit 1
R=runs/merged_r1_coco
V2=data/ign_building_v2/building
MG=data_local/ign_building_merged/building
export CUDA_VISIBLE_DEVICES=1

while pgrep -f "out $R " >/dev/null; do sleep 60; done
sleep 30
echo "=== training exited $(date '+%F %H:%M') ==="
tail -3 $R/train.log
[ -f $R/last.pt ] || { echo "no last.pt -- training failed, not scoring"; exit 1; }

score() {  # ckpt data split out
  echo "=== $1 on $2/$3 -> $4  $(date +%H:%M) ==="
  ./.venv/bin/python -m lvm.score --ckpt "$1" --data "$2" --split "$3" \
    --max-dets 300 --out "$4" 2>&1 | grep -vE "Warning|warn" | tail -14
}

# the new model
score $R/best.pt $V2 test  $R/score_v2test_best.json
score $R/last.pt $V2 test  $R/score_v2test_last.json
score $R/best.pt $MG test  $R/score_mergedtest_best.json
# baselines on the same pinned test set, fixed scorer
score runs/maskrcnn_v2_hires/best.pt $V2 test  $R/baseline_run2_v2test.json
score runs/stageB3_long/best.pt      $V2 test  $R/baseline_B3_v2test.json
# what the scorer fixes alone are worth: run 2 was 0.19476 here before them
score runs/maskrcnn_v2_hires/best.pt $V2 valid $R/baseline_run2_v2valid_fixed.json
echo "=== all scoring done $(date '+%F %H:%M') ==="
