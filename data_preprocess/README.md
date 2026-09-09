# HemoPT Data Preprocessing

This folder contains preprocessing utilities for preparing vascular geometries
before HemoPT pretraining and downstream evaluation. The release does not
include data. Users provide their own segmentation masks, STL meshes, or
downstream CFD arrays.

## Vascular STL Pretraining Pipeline

HemoPT uses vascular STL geometries as the entry point for pretraining-data
construction. When public datasets provide segmentation masks rather than STL
meshes, the preprocessing flow is:

```text
segmentation mask (.nii/.nii.gz)
  -> STL mesh
  -> STL quality-control manifest
  -> optional capping / connected-component cleaning
  -> Vascular_PreTrain arrays
```

### 1. Convert Segmentation Masks to STL

Use `mask_to_stl.py` for generic NIfTI masks:

```bash
python data_preprocess/mask_to_stl.py \
  --src_root /path/to/masks \
  --dst_root HemoData/Vascular_STL \
  --dataset_name MyDataset \
  --recursive
```

By default, all voxels with value greater than zero are treated as foreground.
For multi-label masks, specify foreground labels explicitly:

```bash
python data_preprocess/mask_to_stl.py \
  --src_root /path/to/masks \
  --dst_root HemoData/Vascular_STL \
  --dataset_name MyDataset \
  --labels 1 2 \
  --recursive
```

The output layout is:

```text
HemoData/Vascular_STL/
  MyDataset/
    case_001.stl
    case_002.stl
```

The script also writes a JSONL conversion manifest.

### 2. Run STL Quality Control

```bash
python data_preprocess/vascular_stl_qc.py \
  --root HemoData/Vascular_STL \
  --out HemoData/Vascular_STL_QC
```

This writes:

```text
HemoData/Vascular_STL_QC/qc_manifest.jsonl
HemoData/Vascular_STL_QC/qc_manifest.csv
HemoData/Vascular_STL_QC/qc_summary.json
```

The QC manifest records mesh loading status, watertightness, boundary edges,
non-manifold edges, component statistics, surface area, volume, and geometry
extent.

### 3. Optional Mesh Repair

For open-boundary vascular meshes:

```bash
python data_preprocess/vascular_cap_meshes.py
```

For fragmented meshes:

```bash
python data_preprocess/clean_disconnected_vascular_stl.py
```

These utilities preserve the dataset/file layout and write processed STL copies
under HemoData subdirectories.

### 4. Generate HemoPT Pretraining Arrays

After STL conversion and QC, run:

```bash
python data_generation/Vascular_PreTraining_Data.py \
  --save_root HemoData/Vascular_PreTrain \
  --qc_manifest HemoData/Vascular_STL_QC/qc_manifest.jsonl \
  --qc_statuses pass warn
```

This generates:

```text
HemoData/Vascular_PreTrain/<dataset>/<sample>/
  x.npy
  condition_0.npy
  supervise_0.npy
  ...
  meta.json
```

The 4D compact hemodynamic proxy is constructed online by the
`VascularPretrain` loader during pretraining.

## Dataset-Specific Utilities

The remaining scripts support downstream datasets and dataset-specific
conversions used by the paper experiments, including Aneumo, VMR, 4DFlow, and
HemoPT CFD-style arrays. They are provided as references for users who have
obtained the corresponding datasets and agreements.
