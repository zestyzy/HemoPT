#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/HemoData/Vascular_PreTrain}"
SAVE_NAME="${SAVE_NAME:-hemopt_pretrain_dynamic_dict}"
EPOCHS="${EPOCHS:-200}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NTRAIN="${NTRAIN:-900}"
NTEST="${NTEST:-100}"
LR="${LR:-1e-3}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.25}"
WALL_MASK_PROB="${WALL_MASK_PROB:-0.25}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"

cd "$ROOT"
mkdir -p checkpoints training_logs

echo "=========================================================================="
echo " Launching Canonical HemoPT Pretraining (Dynamic Flow Dictionary, out_dim=13)"
echo " Data: $DATA_PATH | Train: $NTRAIN, Test: $NTEST | Epochs: $EPOCHS"
echo "=========================================================================="

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

"$PYTHON" -u run.py \
  --task vascular_pretrain \
  --loader VascularPretrain \
  --data_path "$DATA_PATH" \
  --ntrain "$NTRAIN" \
  --ntest "$NTEST" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --optimizer AdamW \
  --scheduler StepLR \
  --step_size 50 \
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
  --dict_loss_weight_energy 0.1 \
  --dict_loss_weight_ortho 0.05 \
  --vascular_physics_weight "$PHYSICS_WEIGHT" \
  --vascular_wall_mask_prob "$WALL_MASK_PROB" \
  --vascular_wall_mask_mode all \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --num_workers 4 \
  --pin_memory true \
  --prefetch_factor 2 \
  --save_name "$SAVE_NAME" \
  --seed 2026

echo "Pretraining job completed: $(date)"
