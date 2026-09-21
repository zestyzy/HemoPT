# HemoPT Data Preprocessing

This directory contains standalone preprocessing utilities for preparing vascular geometries and downstream hemodynamics CFD datasets.

---

## 1. Vascular Geometry Preprocessing Pipeline

For raw medical segmentation masks (NIfTI format):

```bash
python data_preprocess/mask_to_stl.py   --src_root /path/to/segmentation_masks   --dst_root HemoData/Vascular_STL   --dataset_name MyDataset   --recursive
```

Then execute STL Quality Control (QC) to verify watertightness, manifold edges, and boundary geometry:

```bash
python data_preprocess/vascular_stl_qc.py   --root HemoData/Vascular_STL   --out HemoData/Vascular_STL_QC
```

Pretraining arrays can then be generated via:

```bash
python data_generation/Vascular_PreTraining_Data.py   --save_root HemoData/Vascular_PreTrain   --qc_manifest HemoData/Vascular_STL_QC/qc_manifest.jsonl
```

---

## 2. Downstream Hemodynamics CFD Processors

To convert raw CFD simulation results into standard HemoPT point cloud representations:

* **VMR CFD Benchmark**:
  ```bash
  python data_preprocess/VMR_CFD_process.py --raw_dir /path/to/raw_vmr --save_dir HemoData/VMR_CFD
  ```

* **Aneumo CFD Benchmark**:
  ```bash
  python data_preprocess/Aneumo_CFD_process.py --raw_dir /path/to/raw_aneumo --save_dir HemoData/Aneumo_CFD
  ```

* **4D Flow / HemoPT Benchmark**:
  ```bash
  python data_preprocess/HemoPT_process.py --raw_dir /path/to/raw_hemo --save_dir HemoData/Hemo_CFD
  ```

Shared wall-normal and boundary distance calculations are modularized in `data_preprocess/wall_features.py`.
