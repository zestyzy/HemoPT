#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/.smoke_data/Vascular_PreTrain}"
SAVE_NAME="${SAVE_NAME:-hemopt_smoke_test}"
GPU="${GPU:-0}"

cd "$ROOT"

echo "=========================================================================="
echo " 1. Generating synthetic vascular pretraining dataset..."
echo "=========================================================================="
"$PYTHON" scripts/generate_synthetic_data.py \
  --out_dir "$DATA_PATH" \
  --num_samples 4 \
  --num_points 64 \
  --n_random_walks 20 \
  --base_walks 20 \
  --seed 2026

echo "=========================================================================="
echo " 2. Running 15-epoch Pretraining Smoke Test with 8-mode Dynamic Flow Dict..."
echo "=========================================================================="
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" run.py \
  --task vascular_pretrain \
  --loader VascularPretrain \
  --data_path "$DATA_PATH" \
  --space_dim 3 \
  --fun_dim 8 \
  --out_dim 13 \
  --ntrain 3 \
  --ntest 1 \
  --epochs 15 \
  --batch-size 1 \
  --lr 1e-3 \
  --optimizer AdamW \
  --scheduler StepLR \
  --step_size 5 \
  --gamma 0.5 \
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
  --save_name "$SAVE_NAME" \
  --eval 0

echo "=========================================================================="
echo " Smoke test completed successfully! Checkpoints are in ./checkpoints/"
echo "=========================================================================="
