#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-$ROOT/HemoData/VMR_CFD}"
LOADER="${LOADER:-VMRCFD}"
PRETRAINED="${PRETRAINED:-hemopt_pretrain_dynamic_dict}"
SAVE_NAME="${SAVE_NAME:-hemopt_finetune_vmr}"
EPOCHS="${EPOCHS:-100}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"

cd "$ROOT"
mkdir -p checkpoints training_logs

echo "=========================================================================="
echo " Launching Downstream Hemodynamics CFD Fine-Tuning"
echo " Task: hemo_cfd_finetune | Loader: $LOADER | Pretrained: $PRETRAINED"
echo "=========================================================================="

export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" -u run.py \
  --task hemo_cfd_finetune \
  --loader "$LOADER" \
  --data_path "$DATA_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --optimizer AdamW \
  --scheduler CosineAnnealingLR \
  --space_dim 3 \
  --fun_dim 8 \
  --out_dim 4 \
  --model Transolver \
  --n_hidden 64 \
  --n_layers 2 \
  --n_heads 4 \
  --slice_num 8 \
  --geotype unstructured \
  --finetune 1 \
  --finetune_name "$PRETRAINED" \
  --save_name "$SAVE_NAME" \
  --seed 2026

echo "Fine-tuning job completed: $(date)"
