import os

import numpy as np
import torch

from data_provider.hemo_loader import HEMO_CONFIG_TO_ID
from data_provider.vmr_cfd_loader import _legacy_hemo_prompt, _vmr_prompt
from utils.normalizer import UnitGaussianNormalizer, UnitTransformer


class HemoVMRCFD(object):
    """Train on HemoPT CFD plus VMR CFD, validate/test only on VMR CFD."""

    def __init__(self, args):
        self.vmr_data_path = args.data_path
        self.hemo_data_path = getattr(args, "hemo_data_path", "./hemo_npys")
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.hemo_ntrain = getattr(args, "hemo_ntrain", 100000)
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.eval_split = getattr(args, "vmr_eval_split", "test")
        self.num_workers = getattr(args, "num_workers", 0)
        self.pin_memory = getattr(args, "pin_memory", False)
        self.prefetch_factor = getattr(args, "prefetch_factor", 2)
        self.hemo_configs = getattr(args, "hemo_configs", ["C1", "ICA_norm", "ICA_ste"])
        self.norm_scope = getattr(args, "hemo_vmr_norm_scope", "vmr_train")
        self.hemo_target = getattr(args, "hemo_target", "full")

        unknown = sorted(set(self.hemo_configs) - set(HEMO_CONFIG_TO_ID))
        if unknown:
            raise ValueError(f"Unsupported hemo_configs={unknown}. Supported: {sorted(HEMO_CONFIG_TO_ID)}")
        if self.hemo_target not in ("full", "velocity"):
            raise ValueError("--hemo_target must be full or velocity")
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")
        if self.norm_scope not in ("vmr_train", "mixed_train"):
            raise ValueError("--hemo_vmr_norm_scope must be vmr_train or mixed_train")

    def _vmr_indices(self):
        split_path = os.path.join(self.vmr_data_path, "global_split_info.npy")
        if os.path.exists(split_path):
            info = np.load(split_path, allow_pickle=True).item()
            train = list(info["train_indices"])[:self.ntrain]
            key = "val_indices" if self.eval_split == "val" else "test_indices"
            eval_indices = list(info.get(key, []))
            if not eval_indices and self.eval_split == "val":
                eval_indices = list(info.get("test_indices", []))
            return train, eval_indices[:self.ntest]

        xs = sorted(int(f[2:-4]) for f in os.listdir(self.vmr_data_path)
                    if f.startswith("x_") and f.endswith(".npy"))
        return xs[:self.ntrain], xs[-self.ntest:]

    def _hemo_train_indices(self):
        split_path = os.path.join(self.hemo_data_path, "global_split_info.npy")
        if os.path.exists(split_path):
            info = np.load(split_path, allow_pickle=True).item()
            candidates = list(info["train_indices"])
        else:
            candidates = sorted(int(f[2:-4]) for f in os.listdir(self.hemo_data_path)
                                if f.startswith("x_") and f.endswith(".npy"))

        allowed = {HEMO_CONFIG_TO_ID[name] for name in self.hemo_configs}
        kept = []
        for idx in candidates:
            cond = np.load(os.path.join(self.hemo_data_path, f"cond_{idx}.npy"))
            if int(round(float(cond[0]))) in allowed:
                kept.append(idx)
        return kept[:self.hemo_ntrain]

    def _load_vmr(self, indices):
        xs, ys, conds = [], [], []
        for idx in indices:
            xs.append(np.load(os.path.join(self.vmr_data_path, f"x_{idx}.npy")))
            ys.append(np.load(os.path.join(self.vmr_data_path, f"y_{idx}.npy")))
            conds.append(np.load(os.path.join(self.vmr_data_path, f"cond_{idx}.npy")))
        x = torch.tensor(np.array(xs), dtype=torch.float)
        y = torch.tensor(np.array(ys), dtype=torch.float)
        cond = torch.tensor(np.array(conds), dtype=torch.float)[:, None, :]
        return x[:, :, :3], x[:, :, 3:], cond, y

    def _load_hemo(self, indices):
        xs, ys, conds = [], [], []
        for idx in indices:
            xs.append(np.load(os.path.join(self.hemo_data_path, f"x_{idx}.npy")))
            ys.append(np.load(os.path.join(self.hemo_data_path, f"y_{idx}.npy")))
            conds.append(np.load(os.path.join(self.hemo_data_path, f"cond_{idx}.npy")))

        x = torch.tensor(np.array(xs), dtype=torch.float)
        y = torch.tensor(np.array(ys), dtype=torch.float)
        cond3 = torch.tensor(np.array(conds), dtype=torch.float)

        fx = x[:, :, 3:]
        if fx.shape[-1] < 4:
            pad = torch.zeros(*fx.shape[:-1], 4 - fx.shape[-1], dtype=fx.dtype)
            fx = torch.cat([fx, pad], dim=-1)
        elif fx.shape[-1] > 4:
            fx = fx[:, :, :4]

        cond = torch.zeros(cond3.shape[0], 1, 12, dtype=cond3.dtype)
        cond[:, :, :3] = cond3[:, None, :]
        cond[:, :, 11] = -1.0
        return x[:, :, :3], fx, cond, y

    def _make_normalizer(self, reference_y):
        if self.norm_type == "UnitTransformer":
            return UnitTransformer(reference_y)
        return UnitGaussianNormalizer(reference_y)

    def get_loader(self, full_mesh=True):
        vmr_train_idx, vmr_eval_idx = self._vmr_indices()
        hemo_train_idx = self._hemo_train_indices()
        if not vmr_train_idx or not vmr_eval_idx:
            raise RuntimeError(f"HemoVMRCFD requires non-empty VMR train/eval indices under {self.vmr_data_path}")
        if not hemo_train_idx:
            raise RuntimeError(f"HemoVMRCFD found no selected HemoPT train samples under {self.hemo_data_path}")

        vmr_train_pos, vmr_train_fx, vmr_train_cond, vmr_train_y = self._load_vmr(vmr_train_idx)
        vmr_eval_pos, vmr_eval_fx, vmr_eval_cond, vmr_eval_y = self._load_vmr(vmr_eval_idx)
        hemo_pos, hemo_fx, hemo_cond, hemo_y = self._load_hemo(hemo_train_idx)

        if self.hemo_target == "velocity":
            vmr_train_y = vmr_train_y[:, :, 1:4]
            vmr_eval_y = vmr_eval_y[:, :, 1:4]
            hemo_y = hemo_y[:, :, 1:4]
            print("  hemo_target=velocity: target y is [u,v,w] before normalization")
        else:
            print("  hemo_target=full: target y is [p,u,v,w]")

        train_pos = torch.cat([hemo_pos, vmr_train_pos], dim=0)
        train_fx = torch.cat([hemo_fx, vmr_train_fx], dim=0)
        train_cond = torch.cat([hemo_cond, vmr_train_cond], dim=0)
        train_y = torch.cat([hemo_y, vmr_train_y], dim=0)

        if self.normalize:
            reference_y = vmr_train_y if self.norm_scope == "vmr_train" else train_y
            self.y_normalizer = self._make_normalizer(reference_y)
            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_fx, train_cond, train_y),
            batch_size=self.batch_size, shuffle=True, **loader_kwargs)
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(vmr_eval_pos, vmr_eval_fx, vmr_eval_cond, vmr_eval_y),
            batch_size=self.batch_size, shuffle=False, **loader_kwargs)

        print("HemoVMR CFD dataloading is over.")
        print(f"  Hemo train configs={self.hemo_configs} samples={len(hemo_train_idx)}")
        print(f"  VMR train samples={len(vmr_train_idx)} eval_split={self.eval_split} eval_samples={len(vmr_eval_idx)}")
        print(f"  train: pos={train_pos.shape} fx={train_fx.shape} cond={train_cond.shape} y={train_y.shape}")
        print(f"  eval:  pos={vmr_eval_pos.shape} fx={vmr_eval_fx.shape} cond={vmr_eval_cond.shape} y={vmr_eval_y.shape}")
        print(f"  normalize={self.normalize} norm_type={self.norm_type} norm_scope={self.norm_scope} target={self.hemo_target}")
        return train_loader, eval_loader, [train_y.shape[1]]

    @staticmethod
    def build_prompt(pos, cond, fx=None):
        hemo_mask = cond[:, :, 11:12] < 0.0
        hemo_prompt = _legacy_hemo_prompt(pos, cond)
        vmr_prompt = _vmr_prompt(pos, fx, cond)
        return torch.where(hemo_mask, hemo_prompt, vmr_prompt)
