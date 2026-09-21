#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/HemoData/Vascular_PreTrain}"
SAVE_NAME="${SAVE_NAME:-hemopt_dynamic_flow_dict_1000cases_200ep}"
EPOCHS="${EPOCHS:-200}"
GPU="${GPU:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NTRAIN="${NTRAIN:-900}"
NTEST="${NTEST:-100}"
LR="${LR:-1e-3}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.25}"
WALL_MASK_PROB="${WALL_MASK_PROB:-0.25}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"

cd "$ROOT"
mkdir -p results/pretrain_training
mkdir -p training_logs
mkdir -p checkpoints

echo "=========================================================================="
echo " Starting HemoPT Pretraining on 1,000 Shuffled Real Vascular Geometries"
echo " Time: $(date)"
echo " GPU: $GPU | Epochs: $EPOCHS | Batch size: $BATCH_SIZE"
echo " Train cases: $NTRAIN | Test cases: $NTEST"
echo " Save name: $SAVE_NAME"
echo " Model: Transolver (n_hidden=256, n_layers=8, n_heads=8, slice_num=32)"
echo " Analytic Compact Flow Dict: fixed 8 modes, learned 8-way routing, divergence & no-slip"
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
  --lr "$LR" \
  --optimizer AdamW \
  --scheduler CosineAnnealingLR \
  --warmup_epochs 10 \
  --space_dim 3 \
  --fun_dim 8 \
  --out_dim 13 \
  --normalize 0 \
  --model Transolver \
  --n_hidden 256 \
  --n_layers 8 \
  --n_heads 8 \
  --mlp_ratio 2 \
  --slice_num 32 \
  --geotype unstructured \
  --n_random_walks 20 \
  --base_walks 20 \
  --vascular_physics_proxy true \
  --vascular_physics_proxy_mode learnable_generalized_flow_compact \
  --dict_loss_weight_entropy 0.01 \
  --dict_loss_weight_noslip 0.1 \
  --dict_loss_weight_div 0.05 \
  --vascular_physics_weight "$PHYSICS_WEIGHT" \
  --vascular_wall_mask_prob "$WALL_MASK_PROB" \
  --vascular_wall_mask_mode all \
  --early_stop 1 \
  --patience 50 \
  --min_delta 1e-4 \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --num_workers 2 \
  --pin_memory 1 \
  --prefetch_factor 2 \
  --eval 0 \
  --save_name "$SAVE_NAME" \
  --seed 2026

echo "Pretraining completed at: $(date)"
