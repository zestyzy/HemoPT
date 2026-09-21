#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/.smoke_data/Vascular_PreTrain}"
SAVE_NAME="${SAVE_NAME:-hemopt_dynamic_flow_dict_smoke}"
EPOCHS="${EPOCHS:-15}"
GPU="${GPU:-0}"

cd "$ROOT"

if [ ! -d "$DATA_PATH" ]; then
  "$PYTHON" scripts/pretrain/generate_synthetic_vascular_pretrain.py \
    --out_dir "$DATA_PATH" \
    --num_samples 4 \
    --num_points 64 \
    --n_random_walks 20 \
    --base_walks 20 \
    --seed 2026
fi

echo "Running Pretrain for ${EPOCHS} epochs with fixed 8-mode analytic compact flow dictionary on GPU ${GPU}..."
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" -u run.py \
  --task vascular_pretrain \
  --loader VascularPretrain \
  --data_path "$DATA_PATH" \
  --ntrain 3 \
  --ntest 1 \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --lr 1e-3 \
  --optimizer AdamW \
  --scheduler StepLR \
  --step_size 5 \
  --gamma 0.5 \
  --space_dim 3 \
  --fun_dim 8 \
  --out_dim 13 \
  --model Transolver \
  --n_hidden 16 \
  --n_layers 1 \
  --n_heads 2 \
  --slice_num 4 \
  --geotype unstructured \
  --n_random_walks 20 \
  --base_walks 20 \
  --vascular_physics_proxy true \
  --vascular_physics_proxy_mode learnable_generalized_flow_compact \
  --dict_loss_weight_entropy 0.01 \
  --dict_loss_weight_noslip 0.1 \
  --dict_loss_weight_div 0.05 \
  --vascular_physics_weight 1.0 \
  --vascular_wall_mask_prob 0.0 \
  --num_workers 0 \
  --pin_memory false \
  --device cuda \
  --eval 0 \
  --save_name "$SAVE_NAME" \
  --seed 2026

echo "SUCCESS: Completed ${EPOCHS} epochs of fixed 8-mode analytic compact flow training!"
