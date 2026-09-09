#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/.smoke_data/Vascular_PreTrain}"
SAVE_NAME="${SAVE_NAME:-hemopt_synthetic_smoke}"

cd "$ROOT"

"$PYTHON" scripts/pretrain/generate_synthetic_vascular_pretrain.py \
  --out_dir "$DATA_PATH" \
  --num_samples 4 \
  --num_points 64 \
  --n_random_walks 2 \
  --seed 2026

"$PYTHON" run.py \
  --task vascular_pretrain \
  --loader VascularPretrain \
  --data_path "$DATA_PATH" \
  --ntrain 3 \
  --ntest 1 \
  --epochs 1 \
  --batch-size 1 \
  --lr 1e-3 \
  --optimizer AdamW \
  --scheduler StepLR \
  --step_size 1 \
  --gamma 1.0 \
  --space_dim 3 \
  --fun_dim 8 \
  --out_dim 13 \
  --model Transolver \
  --n_hidden 16 \
  --n_layers 1 \
  --n_heads 2 \
  --slice_num 4 \
  --geotype unstructured \
  --n_random_walks 2 \
  --vascular_physics_proxy true \
  --vascular_physics_proxy_mode conditioned_generalized_flow_compact \
  --vascular_physics_weight 1.0 \
  --vascular_wall_mask_prob 0.0 \
  --num_workers 0 \
  --pin_memory false \
  --device "${DEVICE:-auto}" \
  --eval 0 \
  --save_name "$SAVE_NAME" \
  --seed 2026

echo "Smoke pretraining completed. Checkpoints are under $ROOT/checkpoints/."
