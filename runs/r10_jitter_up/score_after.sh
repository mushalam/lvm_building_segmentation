#!/bin/bash
# Queued behind r10_jitter_up on GPU 1; waits on the training PID in train.pid.
# v2 test (1,402 pinned tiles) is the verdict. This run trains on v2 alone
# from COCO weights, so v2 valid is clean for it and best.pt is a fair pick.
cd /home/ankur/work/lvm_EOSC || exit 1
R=runs/r10_jitter_up
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
score $R/best.pt model $R/score_v2test_best_ema.json
score $R/last.pt model $R/score_v2test_last_ema.json
score $R/last.pt raw   $R/score_v2test_last_raw.json
echo "=== all scoring done $(date '+%F %H:%M') ==="
