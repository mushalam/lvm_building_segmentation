#!/bin/bash
# PreciseBN, training-mode forward (sampled proposals), raw then EMA control.
cd /home/ankur/work/lvm_EOSC; R=runs/r10_jitter_up; export CUDA_VISIBLE_DEVICES=1
for w in raw model; do
  echo "=== recompute BN stats (train mode): last.pt [$w]  $(date +%H:%M) ==="
  ./.venv/bin/python -u tools/precise_bn.py --mode train --ckpt $R/last.pt --weights $w --n 600 --out $R/last_${w}_precisebn_train.pt 2>&1 | grep --line-buffered -v -i warn
  echo "=== score on v2 test  $(date +%H:%M) ==="
  ./.venv/bin/python -m lvm.score --ckpt $R/last_${w}_precisebn_train.pt --data data/ign_building_v2/building --split test --max-dets 300 --out $R/score_v2test_last_${w}_precisebn_train.json 2>&1 | grep --line-buffered -E '"AP"|AP75|AP_small|matched_mask_iou'
done
echo "=== done $(date +%H:%M) ==="
