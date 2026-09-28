#!/bin/bash
# Inference-speed comparison, Mask R-CNN vs SAM 3, on the same GPU (tvs-gpu-2
# GPU 1, L40S) and the same 200 v2-test tiles, once nothing else uses it:
# waits on stageB4_merged_ema's scoring queue (by PID, never a pgrep pattern).
#
# SAM 3 runs from the sam3-ft-EOSC code at commit 1b063f8 in its own venv,
# via its own load_model/predict_one; the checkpoint is v2_frozen's inference
# export (num_queries 320, the architecture of the full fine-tune and of
# merged_v3 -- freezing changes training, not inference).
cd /home/ankur/work/lvm_EOSC || exit 1
B=runs/bench
export CUDA_VISIBLE_DEVICES=1

qpid=$(cat $B/wait.pid 2>/dev/null)
[ -n "$qpid" ] && while kill -0 "$qpid" 2>/dev/null; do sleep 60; done
sleep 30
echo "=== GPU 1 before: $(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)  $(date '+%F %H:%M') ==="

echo "=== Mask R-CNN: B3 (v2 test 0.2010) ==="
./.venv/bin/python -m lvm.bench --ckpt runs/stageB3_long/best.pt --out $B/lvm_B3.json 2>&1 | grep -v -i warn
echo "=== Mask R-CNN: B4 last.pt ==="
[ -f runs/stageB4_merged_ema/last.pt ] && ./.venv/bin/python -m lvm.bench \
  --ckpt runs/stageB4_merged_ema/last.pt --out $B/lvm_B4.json 2>&1 | grep -v -i warn

echo "=== SAM 3 (num_queries 320) ==="
S=/home/ankur/work/bench_sam3
(cd $S/code && PYTHONPATH=$S/code /home/ankur/work/sam3ft/.venv/bin/python \
  /home/ankur/work/lvm_EOSC/tools/bench_sam3.py \
  --ckpt $S/ckpt_v2_frozen/sam3_finetuned.pt --arch $S/ckpt_v2_frozen/arch.json \
  --sam3-root /home/ankur/work/sam3ft/sam3 \
  --data /home/ankur/work/lvm_EOSC/data/ign_building_v2/building \
  --out /home/ankur/work/lvm_EOSC/$B/sam3_v2_frozen.json) 2>&1 | grep -v -i warn | tail -25
echo "=== done $(date '+%F %H:%M') ==="
