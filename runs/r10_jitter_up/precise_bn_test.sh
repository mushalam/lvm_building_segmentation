#!/bin/bash
# PreciseBN diagnostic for the raw final-epoch collapse (lit review 2026-10-01).
cd /home/ankur/work/lvm_EOSC; R=runs/r10_jitter_up; export CUDA_VISIBLE_DEVICES=1
for w in raw model; do
  echo "=== recompute BN stats: last.pt [$w]  $(date +%H:%M) ==="
  ./.venv/bin/python -u tools/precise_bn.py --ckpt $R/last.pt --weights $w --n 600 --out $R/last_${w}_precisebn.pt 2>&1 | grep -v -i warn
  echo "=== score last_${w}_precisebn on v2 test  $(date +%H:%M) ==="
  ./.venv/bin/python -m lvm.score --ckpt $R/last_${w}_precisebn.pt --data data/ign_building_v2/building --split test --max-dets 300 --out $R/score_v2test_last_${w}_precisebn.json 2>&1 | grep -E '"AP"|AP75|AP_small|matched_mask_iou'
done
echo "=== done $(date +%H:%M) ==="
