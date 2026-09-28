#!/bin/bash
# Queued behind stageB4_merged_ema on GPU 1. Waits on the training PID in
# train.pid -- not a pgrep pattern, which matched the launching shell's own
# command line and stalled merged_r1_coco's queue for three hours.
#
# Scored on the 1,402 pinned v2 test images only. v2 valid is compromised for
# this model (492 of its 700 images were in merged train), so best.pt, chosen
# on it, is secondary; last.pt is the primary artefact.
cd /home/ankur/work/lvm_EOSC || exit 1
R=runs/stageB4_merged_ema
V2=data/ign_building_v2/building
export CUDA_VISIBLE_DEVICES=1

pid=$(cat $R/train.pid)
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
sleep 30
echo "=== training exited $(date '+%F %H:%M') ==="
tail -3 $R/train.log
[ -f $R/last.pt ] || { echo "no last.pt -- training failed, not scoring"; exit 1; }

score() {  # ckpt weights out
  echo "=== $1 [$2] on v2 test -> $3  $(date +%H:%M) ==="
  ./.venv/bin/python -m lvm.score --ckpt "$1" --weights "$2" --data $V2 \
    --split test --max-dets 300 --out "$3" 2>&1 | grep -vE "Warning|warn" | tail -22
}

score $R/last.pt model $R/score_v2test_last_ema.json   # primary
score $R/last.pt raw   $R/score_v2test_last_raw.json   # EMA's effect, same run
score $R/best.pt model $R/score_v2test_best_ema.json
echo "=== all scoring done $(date '+%F %H:%M') ==="
