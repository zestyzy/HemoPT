#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
LOG_DIR=${LOG_DIR:-"$ROOT/results/pretrain_training"}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-"$LOG_DIR/pretrain_03_rw_conditioned_generalized_flow_compact_outdim13_${STAMP}.log"}

GPU=${GPU:-2}
DEVICE=${DEVICE:-auto}
DATA_PATH=${DATA_PATH:-"$ROOT/HemoData/Vascular_PreTrain"}
SAVE_NAME=${SAVE_NAME:-HemoPT_rw_conditioned_generalized_flow_compact_outdim13}
PHYSICS_WEIGHT=${PHYSICS_WEIGHT:-0.25}
WALL_MASK_PROB=${WALL_MASK_PROB:-0.25}
WALL_MASK_MODE=${WALL_MASK_MODE:-all}
EPOCHS=${EPOCHS:-250}
EARLY_STOP=${EARLY_STOP:-1}
PATIENCE=${PATIENCE:-75}
MIN_DELTA=${MIN_DELTA:-1e-4}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-25}
BATCH_SIZE=${BATCH_SIZE:-8}
NTRAIN=${NTRAIN:-100000}
NTEST=${NTEST:-266}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}

mkdir -p "$LOG_DIR"
cd "$ROOT"

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp}
export PYTHONUNBUFFERED=1

exec > >(tee "$LOG_FILE") 2>&1

echo "Writing log to: $LOG_FILE"
echo "Experiment: RW + conditioned generalized_flow_compact, out_dim=13"
echo "Pretrain data: $DATA_PATH"
echo "Checkpoint: $ROOT/checkpoints/${SAVE_NAME}.pt"
echo "Best checkpoint mirror: $ROOT/checkpoints/${SAVE_NAME}_best.pt"
echo "Physics weight: $PHYSICS_WEIGHT"
echo "Wall mask: prob=$WALL_MASK_PROB mode=$WALL_MASK_MODE"
echo "Epochs/early stop: max=$EPOCHS early_stop=$EARLY_STOP patience=$PATIENCE min_delta=$MIN_DELTA"
echo "Periodic checkpoint interval: $CHECKPOINT_INTERVAL"
echo "GPU: $GPU"
echo "Run started: $(date)"

PYTHON=${PYTHON:-python}

"$PYTHON" -u run.py \
--gpu "$GPU" \
--device "$DEVICE" \
--data_path "$DATA_PATH" \
--loader VascularPretrain \
--task vascular_pretrain \
--geotype unstructured \
--space_dim 3 \
--fun_dim 8 \
--out_dim 13 \
--normalize 0 \
--model Transolver \
--n_hidden 256 \
--n_heads 8 \
--n_layers 8 \
--mlp_ratio 2 \
--slice_num 32 \
--ntrain "$NTRAIN" \
--ntest "$NTEST" \
--batch-size "$BATCH_SIZE" \
--epochs "$EPOCHS" \
--early_stop "$EARLY_STOP" \
--patience "$PATIENCE" \
--min_delta "$MIN_DELTA" \
--checkpoint_interval "$CHECKPOINT_INTERVAL" \
--scheduler CosineAnnealingLR \
--warmup_epochs 10 \
--num_workers "$NUM_WORKERS" \
--pin_memory "$PIN_MEMORY" \
--prefetch_factor "$PREFETCH_FACTOR" \
--eval 0 \
--save_name "$SAVE_NAME" \
--n_random_walks 20 \
--vascular_physics_proxy true \
--vascular_physics_proxy_mode conditioned_generalized_flow_compact \
--vascular_physics_weight "$PHYSICS_WEIGHT" \
--vascular_wall_mask_prob "$WALL_MASK_PROB" \
--vascular_wall_mask_mode "$WALL_MASK_MODE"

echo "Run finished: $(date)"
