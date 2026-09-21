#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SAVE_ROOT="$ROOT/HemoData/Vascular_PreTrain_400_R25_T5"
LOG_DIR="$ROOT/training_logs"
mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_ROOT"

echo "=========================================================================="
echo " Launching 400-Case Data Preprocessing (R=25 views, T=5 steps, float32)"
echo " Root: $ROOT"
echo " Save destination: $SAVE_ROOT"
echo " Start time: $(date)"
echo "=========================================================================="

"$PYTHON" -u scripts/data_generation/batch_generate_400cases_r25_t5.py \
  --stl_root "$ROOT/HemoData/Vascular_STL" \
  --save_root "$SAVE_ROOT" \
  --qc_manifest "$ROOT/HemoData/Vascular_STL_QC/qc_manifest.jsonl" \
  --vmr_reserved "$ROOT/HemoData/VMR_CFD_Splits/vmr_cfd_split.json" \
  --aneumo_reserved "$ROOT/HemoData/Aneumo_CFD_Splits/aneumo_cfd_split.json" \
  --num_cases 400 \
  --seed 2026 \
  --num_workers 32

echo "Preprocessing job completed at: $(date)"
