#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/HemoData/Vascular_PreTrain}"
SAVE_NAME="${SAVE_NAME:-hemopt_dict_smoketest_real_subset}"
EPOCHS="${EPOCHS:-15}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NTRAIN="${NTRAIN:-40}"
NTEST="${NTEST:-10}"

cd "$ROOT"
mkdir -p results/pretrain_training
mkdir -p training_logs

echo "=========================================================================="
echo " Starting Pretrain Smoke Test on Real Vascular Subset (40 train, 10 test)"
echo " GPU: $GPU | Epochs: $EPOCHS | Batch size: $BATCH_SIZE"
echo " Active Routing Gradients: YES | FCPW 20-View Compatible: YES"
echo "=========================================================================="

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

"$PYTHON" -u run.py \
  --gpu "$GPU" \
  --device cuda \
  --task vascular_pretrain \
  --loader VascularPretrain \
  --data_path "$DATA_PATH" \
  --ntrain "$NTRAIN" \
  --ntest "$NTEST" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr 1e-3 \
  --optimizer AdamW \
  --scheduler StepLR \
  --step_size 5 \
  --gamma 0.5 \
  --space_dim 3 \
  --fun_dim 8 \
  --out_dim 13 \
  --model Transolver \
  --n_hidden 64 \
  --n_layers 2 \
  --n_heads 4 \
  --slice_num 8 \
  --geotype unstructured \
  --n_random_walks 20 \
  --base_walks 20 \
  --vascular_physics_proxy true \
  --vascular_physics_proxy_mode learnable_generalized_flow_compact \
  --dict_loss_weight_entropy 0.01 \
  --dict_loss_weight_noslip 0.1 \
  --dict_loss_weight_div 0.05 \
  --vascular_physics_weight 0.25 \
  --vascular_wall_mask_prob 0.25 \
  --vascular_wall_mask_mode all \
  --num_workers 2 \
  --pin_memory true \
  --prefetch_factor 2 \
  --eval 0 \
  --save_name "$SAVE_NAME" \
  --seed 2026

echo "Smoke test completed successfully!"
