#!/bin/bash
# Boundary IoU with the object-scale band (lvm/boundary.py, 2026-10-01) for the
# main checkpoints, on the seeded 250-tile v2-test sample, CPU (GPU 1 is training).
cd /home/ankur/work/lvm_EOSC; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=40
for c in r10_jitter_up/best.pt stageB3_long/best.pt maskrcnn_v2_hires/best.pt; do
  n=$(echo $c | cut -d/ -f1)
  echo "=== $c  $(date +%H:%M) ==="
  ./.venv/bin/python -m lvm.score --ckpt runs/$c --data data/ign_building_v2/building --split test \
    --sample-only --batch-size 1 --workers 4 --out runs/boundary_object/${n}_sample250.json 2>&1 \
    | grep -E '"AP"|boundary_iou|matched_mask_iou|boundary_dilation_px|boundary_match_rate'
done
echo "=== done $(date +%H:%M) ==="
