# HemoPT Pretraining Data Generation

This folder contains utilities for constructing lifted vascular pretraining
arrays from STL meshes.

## Input

The main vascular generator expects STL meshes under:

```text
HemoData/Vascular_STL/<dataset>/*.stl
```

or, when available, capped meshes under:

```text
HemoData/Vascular_STL_capped/<dataset>/*.stl
```

If `Vascular_STL_capped` exists, the generator uses it by default.

## Output

`Vascular_PreTraining_Data.py` writes:

```text
HemoData/Vascular_PreTrain/<dataset>/<sample>/
  x.npy
  condition_0.npy
  supervise_0.npy
  ...
  meta.json
```

The arrays encode:

- `x.npy`: point coordinates plus wall-distance and wall-direction features;
- `condition_j.npy`: random-walk probe direction and step length for view `j`;
- `supervise_j.npy`: 9D random-walk spatial target;
- `meta.json`: source mesh, QC status, generation parameters, and processing status.

The compact hemodynamic proxy is not saved as a separate file. It is constructed
online by `data_provider/vascular_pretrain_loader.py` when compact HemoPT
pretraining is enabled.

## Usage

The default training-view configuration uses 20 base probes. Each base probe
is perturbed once, and only the resulting 20 perturbed views are saved as
`condition_0..19` and `supervise_0..19`; base probes are not training views.

Run STL QC first:

```bash
python data_preprocess/vascular_stl_qc.py \
  --root HemoData/Vascular_STL \
  --out HemoData/Vascular_STL_QC
```

Then generate vascular pretraining arrays:

```bash
python data_generation/Vascular_PreTraining_Data.py \
  --save_root HemoData/Vascular_PreTrain \
  --qc_manifest HemoData/Vascular_STL_QC/qc_manifest.jsonl \
  --qc_statuses pass warn
```

For debugging without a QC manifest:

```bash
python data_generation/Vascular_PreTraining_Data.py \
  --save_root HemoData/Vascular_PreTrain \
  --allow_missing_qc_manifest
```

## HemoPT Vascular Generator

The HemoPT vascular pipeline uses `Vascular_PreTraining_Data.py` as the
pretraining-array generator. Generic non-vascular pretraining utilities are not
included in this submission package.
