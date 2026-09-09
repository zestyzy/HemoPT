#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
LOG_DIR=${LOG_DIR:-"$ROOT/results/pretrain_training"}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-"$LOG_DIR/pretrain_01_rw_only_outdim9_${STAMP}.log"}

GPU=${GPU:-0}
DEVICE=${DEVICE:-auto}
DATA_PATH=${DATA_PATH:-"$ROOT/HemoData/Vascular_PreTrain"}
SAVE_NAME=${SAVE_NAME:-HemoPT_rw_only_outdim9}
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
echo "Experiment: RW only, out_dim=9"
echo "Pretrain data: $DATA_PATH"
echo "Checkpoint: $ROOT/checkpoints/${SAVE_NAME}.pt"
echo "Best checkpoint mirror: $ROOT/checkpoints/${SAVE_NAME}_best.pt"
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
--out_dim 9 \
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
--n_random_walks 10

echo "Run finished: $(date)"
