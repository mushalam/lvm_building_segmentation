#!/bin/bash
# Mask Scoring, step 2: r10's recipe from COCO weights with the MaskIoU head
# trained jointly from the start (the paper's setup), so the rest of the model
# can adapt to quality scoring too. Queued behind ms1b's chain (PID in
# ms1b_head_long/chain.pid) so the two never share GPU 1.
cd /home/ankur/work/lvm_EOSC || exit 1
R=runs/ms2_joint
q=$(cat runs/ms1b_head_long/chain.pid 2>/dev/null)
[ -n "$q" ] && while kill -0 "$q" 2>/dev/null; do sleep 60; done
sleep 30
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)  $(date '+%F %H:%M') ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u -m lvm.train \
  --data data/ign_building_v2/building --out $R --seed 1 --mask-scoring \
  --epochs 36 --batch-size 2 --lr 0.005 --workers 16 --max-dets 400 \
  --image-size 2048 --val-images 200 --val-every 2 \
  --ema 0.9998 --scale-jitter 1.0,1.33 --copy-paste 30 > $R/train.log 2>&1 < /dev/null
echo "=== training exited $(date '+%F %H:%M') ==="; tail -3 $R/train.log
[ -f $R/last.pt ] || { echo "no last.pt -- not scoring"; exit 1; }
score() {
  echo "=== $1 [$2] on v2 test  $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -m lvm.score --ckpt $R/$1 --weights $2 --data data/ign_building_v2/building \
    --split test --max-dets 300 --out $R/score_v2test_${1%.pt}_$2.json 2>&1 | grep -vE "Warning|warn" | tail -24
}
score best.pt model; score last.pt model; score last.pt raw
echo "=== all done $(date '+%F %H:%M') ==="
