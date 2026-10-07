#!/bin/bash
# HYDRA-style context re-ranker on ms1: cache detections (v2 valid, v2 test, all
# 8,131 held-out 92/93/94 tiles), then train and score tools/context_rerank.py.
cd ~/work/lvm_EOSC
export CUDA_VISIBLE_DEVICES=1
P=.venv/bin/python; C=runs/ms1_head_only/best.pt; O=runs/rerank1
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)  $(date '+%F %T') ==="
for s in valid test; do
  echo "=== dump v2 $s  $(date +%T) ==="
  $P tools/dump_detections.py --ckpt $C --data data/ign_building_v2/building --split $s \
    --keep-rles --out $O/dets_$s.pt 2>&1 | grep -v -i warn || exit 1
done
echo "=== dump 92/93/94 (held out)  $(date +%T) ==="
$P tools/dump_detections.py --ckpt $C --data data_local/ign_idf/building --split train \
  --out $O/dets_idf.pt 2>&1 | grep -v -i warn || exit 1
echo "=== re-ranker  $(date +%T) ==="
$P tools/context_rerank.py --train $O/dets_idf.pt --valid $O/dets_valid.pt --test $O/dets_test.pt \
  --out $O/rerank.json 2>&1 | grep -v -i warn
echo "=== all done $(date '+%F %T') ==="
