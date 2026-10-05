#!/bin/bash
# Mask Scoring, step 1: train ONLY a MaskIoU head on frozen r10 (best.pt, EMA
# weights), so detections are exactly r10's and only their ranking changes --
# directly comparable to r10's 0.2177 and to the oracle ceiling 0.3135
# (runs/oracle/). Queued behind r10b's scoring queue (PID in r10b_seed1/
# queue.pid) so the two never share GPU 1.
cd /home/ankur/work/lvm_EOSC || exit 1
R=runs/ms1_head_only
q=$(cat runs/r10b_seed1/queue.pid 2>/dev/null)
[ -n "$q" ] && while kill -0 "$q" 2>/dev/null; do sleep 60; done
sleep 30
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)  $(date '+%F %H:%M') ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u -m lvm.train \
  --data data/ign_building_v2/building --out $R --seed 1 \
  --init-from runs/r10_jitter_up/best.pt --mask-scoring --freeze-except-maskiou \
  --epochs 4 --batch-size 2 --lr 0.005 --workers 16 --max-dets 400 \
  --image-size 2048 --val-images 200 --val-every 2 \
  --scale-jitter 1.0,1.33 --copy-paste 30 > $R/train.log 2>&1 < /dev/null
echo "=== training exited $(date '+%F %H:%M') ==="; tail -3 $R/train.log
[ -f $R/last.pt ] || { echo "no last.pt -- not scoring"; exit 1; }
for c in best last; do
  echo "=== $c.pt on v2 test  $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -m lvm.score --ckpt $R/$c.pt --data data/ign_building_v2/building \
    --split test --max-dets 300 --out $R/score_v2test_$c.json 2>&1 | grep -vE "Warning|warn" | tail -24
done
echo "=== all done $(date '+%F %H:%M') ==="
