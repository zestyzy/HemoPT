# HemoPT: Self-Supervised Pretraining with Dynamic Flow Dictionary for Hemodynamics

This repository contains the clean, reproducible implementation of **HemoPT**, a self-supervised vascular pretraining framework that couples geometry-conditioned random-walk probes with a learnable dynamic hemodynamic flow dictionary.

---

## 📁 Repository Layout

```text
├── data_generation/      Vascular random-walk probe generation routines.
├── data_preprocess/      Vascular STL QC, capping, and CFD benchmark dataset processors.
├── data_provider/        PyTorch datasets and dataloaders for pretraining and downstream CFD.
├── exp/                  Training loops (vascular pretraining & downstream CFD fine-tuning).
├── layers/               Physics-attention layers.
├── models/               Transolver backbone and 8-mode Dynamic Flow Dictionary.
├── scripts/              Standard execution scripts (smoke test, pretraining, fine-tuning, eval).
├── tests/                Unit regression tests for dictionary, gradients, and contracts.
├── utils/                Loss functions, normalizers, and optimization utilities.
└── run.py                Main training entry point.
```

---

## 🛠️ Installation

Create a Python 3.9+ environment with PyTorch (>=1.13.0), then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start: 1-Minute Smoke Test

We provide a self-contained smoke test that automatically generates a tiny synthetic vascular geometry dataset, executes 15 epochs of HemoPT pretraining with the 8-mode dynamic flow dictionary, tracks routing gate entropy, and verifies checkpoint synchronization without needing any external data:

```bash
bash scripts/smoke_test.sh
```

Or run the unit regression test suite:

```bash
pytest tests
```

---

## 🔬 Pretraining & Downstream Fine-Tuning

### 1. Self-Supervised Pretraining
To train HemoPT on a processed vascular geometry dataset:

```bash
DATA_PATH=/path/to/Vascular_PreTrain GPU=0 bash scripts/run_pretrain.sh
```

This runs:
- Task: `vascular_pretrain`
- Backbone: `Transolver`
- Flow dictionary: 8-mode compact dynamic flow bank with learnable routing gate
- Auxiliary losses: Wall no-slip, divergence penalty, kinetic energy scale, and gate entropy regularization.

### 2. Downstream Hemodynamics CFD Fine-Tuning
To fine-tune a pretrained checkpoint on downstream hemodynamic CFD datasets (e.g., VMR, Aneumo):

```bash
DATA_PATH=/path/to/VMR_CFD LOADER=VMRCFD PRETRAINED=hemopt_pretrain_dynamic_dict GPU=0 bash scripts/run_finetune.sh
```

### 3. Metric Evaluation
To evaluate directional alignment ($C_\text{dir}$), magnitude relative error ($C_\text{mag}$), and gate entropy on trained models:

```bash
python scripts/eval_alignment.py --ckpt checkpoints/your_checkpoint.pt --device cuda:0
```

---

## 🛡️ Double-Blind Compliance & Data Policy

This submission package adheres strictly to double-blind conference guidelines:
- **Zero personal or institutional identifiers** (usernames, hostnames, private IPs, credentials).
- **No proprietary binary data or checkpoints** included in the repository.
- All file paths default to repository-relative conventions.
