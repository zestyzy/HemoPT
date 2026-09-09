# HemoPT

This repository contains the code used for HemoPT, a proxy-supervised
vascular pretraining framework for geometry-to-hemodynamics prediction.

HemoPT builds on a Transolver backbone and extends geometry-conditioned
pretraining with compact hemodynamic proxy supervision. The main pretraining
target contains a 9D random-walk spatial target and a 4D compact flow proxy.
The compact flow proxy is generated from local morphology and probe conditions
using `conditioned_generalized_flow_compact`.

## Repository Layout

```text
data_generation/      Vascular pretraining data generation utilities.
data_preprocess/      Mask-to-STL, STL QC, and dataset preprocessing utilities.
data_provider/        PyTorch data loaders.
exp/                  Training loops for pretraining and downstream tasks.
layers/               Physics-attention layers.
models/               Transolver model definition.
scripts/pretrain/     Reproducible pretraining and smoke-test scripts.
utils/                Losses, normalization, optimization, and visualization.
run.py                Main training entry point.
```

## Data Policy

This code release does not include patient data, CFD labels, STL meshes,
pretraining arrays, downstream arrays, checkpoints, logs, or paper results.
Paths in scripts are repository-relative by default. Users should provide their
own processed data paths through command-line arguments or environment
variables.

The HemoPT vascular preprocessing pipeline is organized as:

```text
segmentation masks (.nii/.nii.gz)
  -> vascular STL meshes
  -> STL QC / optional capping and component cleaning
  -> VascularPreTrain arrays
  -> VascularPretrain loader
```

For segmentation masks, convert masks to STL meshes with:

```bash
python data_preprocess/mask_to_stl.py \
  --src_root /path/to/segmentation_masks \
  --dst_root HemoData/Vascular_STL \
  --dataset_name MyDataset \
  --recursive
```

Then create an STL quality-control manifest:

```bash
python data_preprocess/vascular_stl_qc.py \
  --root HemoData/Vascular_STL \
  --out HemoData/Vascular_STL_QC
```

Optional capping and component-cleaning utilities are provided in
`data_preprocess/vascular_cap_meshes.py` and
`data_preprocess/clean_disconnected_vascular_stl.py`.

After STL preprocessing, generate HemoPT pretraining arrays with:

```bash
python data_generation/Vascular_PreTraining_Data.py \
  --save_root HemoData/Vascular_PreTrain \
  --qc_manifest HemoData/Vascular_STL_QC/qc_manifest.jsonl \
  --allow_missing_qc_manifest
```

The VascularPretrain loader expects the following processed layout:

```text
<data_path>/<dataset>/<sample>/
  x.npy
  condition_0.npy
  supervise_0.npy
  ...
  meta.json
```

For HemoPT compact pretraining:

- `x.npy` stores point coordinates and local wall morphology.
- `condition_j.npy` stores the probe condition for random-walk view `j`.
- `supervise_j.npy` stores the 9D random-walk spatial target.
- The 4D compact flow proxy is constructed online by the loader when
  `--vascular_physics_proxy true` and
  `--vascular_physics_proxy_mode conditioned_generalized_flow_compact` are set.

## Installation

Create a Python environment with PyTorch, then install the required packages:

```bash
pip install -r requirements.txt
```

The code is compatible with CPU for the synthetic smoke test. Full experiments
should be run on GPUs.

## Synthetic Smoke Test

The smoke test generates a tiny synthetic vascular pretraining dataset and runs
one epoch of compact HemoPT pretraining. It does not use real STL, patient, or
CFD data.

```bash
bash scripts/pretrain/smoke_pretrain_synthetic.sh
```

Optional environment variables:

```bash
PYTHON=python DEVICE=cpu bash scripts/pretrain/smoke_pretrain_synthetic.sh
PYTHON=python DEVICE=cuda GPU=0 bash scripts/pretrain/smoke_pretrain_synthetic.sh
```

The smoke test writes temporary outputs under `.smoke_data/`, `checkpoints/`,
and `training_logs/`. These directories are ignored by git.

## Main HemoPT Pretraining

After preparing a processed vascular pretraining dataset, run:

```bash
DATA_PATH=/path/to/Vascular_PreTrain \
PYTHON=python \
DEVICE=cuda \
GPU=0 \
bash scripts/pretrain/pretrain_03_rw_generalized_flow_compact_outdim13.sh
```

This script uses:

- `--loader VascularPretrain`
- `--task vascular_pretrain`
- `--fun_dim 8`
- `--out_dim 13`
- `--vascular_physics_proxy true`
- `--vascular_physics_proxy_mode conditioned_generalized_flow_compact`

For a geometry-only random-walk pretraining baseline:

```bash
DATA_PATH=/path/to/Vascular_PreTrain \
PYTHON=python \
DEVICE=cuda \
GPU=0 \
bash scripts/pretrain/pretrain_01_rw_only_outdim9.sh
```

## Leakage-Safe Release Notes

This submission package was cleaned to avoid data leakage:

- no real datasets are included;
- no checkpoint files are included;
- no generated `.npy`, `.npz`, `.pkl`, `.h5`, `.stl`, or `.vtk` files are included;
- no server-specific absolute paths are required by default;
- no credentials or API keys are included.

## Acknowledgements

This implementation uses the Transolver architecture with HemoPT-specific
vascular random-walk pretraining and compact hemodynamic proxy supervision.
