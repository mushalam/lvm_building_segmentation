#!/bin/bash
# D4 TTA on the full 1,402-image v2 test split, GPU 1. Runs after
# score_after.sh has finished (it waits on that queue's PID), so the two
# never share the GPU with training.
cd /home/ankur/work/lvm_EOSC || exit 1
R=runs/stageB4_merged_ema
V2=data/ign_building_v2/building
export CUDA_VISIBLE_DEVICES=1

qpid=$(pgrep -f "$R/score_after.sh" | head -1)
[ -n "$qpid" ] && while kill -0 "$qpid" 2>/dev/null; do sleep 60; done
echo "=== starting TTA scoring $(date '+%F %H:%M') ==="

tta() {  # ckpt weights out
  echo "=== $1 [$2] D4 TTA on v2 test -> $3  $(date +%H:%M) ==="
  ./.venv/bin/python -m lvm.score --ckpt "$1" --weights "$2" --data $V2 \
    --split test --max-dets 300 --tta d4 --batch-size 1 --out "$3" 2>&1 \
    | grep -vE "Warning|warn" | tail -22
}

# B3 with TTA against its plain 0.2010: TTA's effect on a known model
tta runs/stageB3_long/best.pt model $R/tta_B3_v2test.json
# B4, the candidate
[ -f $R/last.pt ] && tta $R/last.pt model $R/tta_v2test_last_ema.json
echo "=== all TTA scoring done $(date '+%F %H:%M') ==="
