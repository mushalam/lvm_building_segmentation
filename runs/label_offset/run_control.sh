#!/bin/bash
# Control for the label-offset measurement: do two independently trained models
# (ms1 = r10 + head, ms2 = joint from COCO) put the same building at the same shift?
cd ~/work/lvm_EOSC
export CUDA_VISIBLE_DEVICES=1
P=.venv/bin/python; O=runs/label_offset
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)  $(date '+%F %T') ==="
$P tools/label_offset.py --dets runs/rerank1/dets_valid.pt --workers 16 --out $O/offset_valid_ms1.json > $O/ms1.log 2>&1 || exit 1
$P tools/dump_detections.py --ckpt runs/ms2_joint/best.pt --data data/ign_building_v2/building --split valid \
  --keep-rles --out $O/dets_valid_ms2.pt 2>&1 | grep -v -i warn | tail -1 || exit 1
$P tools/label_offset.py --dets $O/dets_valid_ms2.pt --workers 16 --out $O/offset_valid_ms2.json > $O/ms2.log 2>&1 || exit 1
echo "=== all done $(date '+%F %T') ==="
