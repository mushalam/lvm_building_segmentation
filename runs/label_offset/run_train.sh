#!/bin/bash
# Did ms1/ms2 memorise the label offsets on their own training tiles? Same
# measurement and same two models on a 700-tile v2 train sample.
cd ~/work/lvm_EOSC
export CUDA_VISIBLE_DEVICES=1
P=.venv/bin/python; O=runs/label_offset
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)  $(date '+%F %T') ==="
for m in ms1:runs/ms1_head_only/best.pt ms2:runs/ms2_joint/best.pt; do
  n=${m%%:*}; c=${m#*:}
  $P tools/dump_detections.py --ckpt $c --data data/ign_building_v2/building --split train --max-images 700 \
    --keep-rles --out $O/dets_train700_$n.pt 2>&1 | grep -v -i warn | tail -1 || exit 1
  $P tools/label_offset.py --dets $O/dets_train700_$n.pt --workers 16 --out $O/offset_train700_$n.json > $O/train_$n.log 2>&1 || exit 1
done
echo "=== all done $(date '+%F %T') ==="
