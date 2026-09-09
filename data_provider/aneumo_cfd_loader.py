import os

import numpy as np
import torch

from data_provider.vmr_cfd_loader import _principal_axis
from utils.normalizer import UnitGaussianNormalizer, UnitTransformer


def _aneumo_prompt(pos, fx, cond):
    b, n, _ = pos.shape
    axis = _principal_axis(pos)

    if fx is not None and fx.shape[-1] >= 4:
        dist = torch.clamp(fx[:, :, 0:1], min=0.0)
        dist_scale = dist.amax(dim=1, keepdim=True).clamp_min(1e-6)
        profile = torch.clamp(dist / dist_scale, 0.0, 1.0)
    else:
        profile = torch.ones(b, n, 1, device=pos.device, dtype=pos.dtype)

    if cond.shape[-1] >= 5:
        inlet_mean = cond[:, :, 3:4].clamp_min(0.0)
        inlet_max = cond[:, :, 4:5].clamp_min(0.0)
        speed = torch.maximum(inlet_mean, 0.25 * inlet_max).repeat(1, n, 1)
    else:
        speed = torch.ones(b, n, 1, device=pos.device, dtype=pos.dtype)

    flow_vec = axis[:, None, :].repeat(1, n, 1) * speed * profile
    scalar = speed * profile
    return torch.cat([flow_vec, scalar], dim=-1)


class AneumoCFD(object):
    """Loader for processed aneumo CFD interior point-cloud samples."""

    def __init__(self, args):
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.eval_split = getattr(args, "aneumo_eval_split", "test")
        self.cv_split_info = getattr(args, "cv_split_info", None)
        self.cv_fold = getattr(args, "fold", 0)
        self.hemo_target = getattr(args, "hemo_target", "full")
        self.wall_mode = getattr(args, "hemo_wall_mode", "keep")
        self.num_workers = getattr(args, "num_workers", 0)
        self.pin_memory = getattr(args, "pin_memory", False)
        self.prefetch_factor = getattr(args, "prefetch_factor", 2)
        if self.wall_mode not in ("keep", "zero"):
            raise ValueError("--hemo_wall_mode must be keep or zero")
        if self.hemo_target not in ("full", "velocity"):
            raise ValueError("--hemo_target must be full or velocity")
        if self.eval_split not in ("val", "test"):
            raise ValueError("--aneumo_eval_split must be val or test")
        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def _indices(self):
        if self.cv_split_info:
            info = np.load(self.cv_split_info, allow_pickle=True).item()
            folds = info.get("folds", [])
            if not 0 <= self.cv_fold < len(folds):
                raise ValueError(f"fold must be in [0, {len(folds) - 1}], got {self.cv_fold}")
            fold_info = folds[self.cv_fold]
            train = list(fold_info["train_indices"])[:self.ntrain]
            if self.eval_split == "val":
                eval_indices = list(fold_info["val_indices"])
            else:
                eval_indices = list(info["test_indices"])
            print(f"  Using strict CV split: path={self.cv_split_info} fold={self.cv_fold} "
                  f"train={len(train)} eval_split={self.eval_split} eval={len(eval_indices)}")
            return train, eval_indices[:self.ntest]

        split_path = os.path.join(self.data_path, "global_split_info.npy")
        if os.path.exists(split_path):
            info = np.load(split_path, allow_pickle=True).item()
            train = list(info["train_indices"])[:self.ntrain]
            key = "val_indices" if self.eval_split == "val" else "test_indices"
            eval_indices = list(info.get(key, []))
            if not eval_indices and self.eval_split == "val":
                eval_indices = list(info.get("test_indices", []))
            return train, eval_indices[:self.ntest]
        xs = sorted(int(f[2:-4]) for f in os.listdir(self.data_path)
                    if f.startswith("x_") and f.endswith(".npy"))
        return xs[:self.ntrain], xs[-self.ntest:]

    def _load(self, indices):
        xs, ys, conds = [], [], []
        for i in indices:
            xs.append(np.load(os.path.join(self.data_path, f"x_{i}.npy")))
            ys.append(np.load(os.path.join(self.data_path, f"y_{i}.npy")))
            conds.append(np.load(os.path.join(self.data_path, f"cond_{i}.npy")))
        x = torch.tensor(np.array(xs), dtype=torch.float)
        y = torch.tensor(np.array(ys), dtype=torch.float)
        cond = torch.tensor(np.array(conds), dtype=torch.float)[:, None, :]
        return x[:, :, :3], x[:, :, 3:], cond, y

    def get_loader(self, full_mesh=True):
        train_idx, eval_idx = self._indices()
        if not train_idx or not eval_idx:
            raise RuntimeError(f"AneumoCFD requires non-empty train/eval indices under {self.data_path}")
        train_pos, train_fx, train_cond, train_y = self._load(train_idx)
        eval_pos, eval_fx, eval_cond, eval_y = self._load(eval_idx)

        if self.wall_mode == "zero":
            train_fx = torch.zeros_like(train_fx)
            eval_fx = torch.zeros_like(eval_fx)
            print("  hemo_wall_mode=zero: wall distance/direction channels are zeroed")
        else:
            print("  hemo_wall_mode=keep: wall distance/direction channels are used")

        if self.hemo_target == "velocity":
            train_y = train_y[:, :, 1:4]
            eval_y = eval_y[:, :, 1:4]
            print("  hemo_target=velocity: target y is [u,v,w] before normalization")
        else:
            print("  hemo_target=full: target y is [p,u,v,w]")

        if self.normalize:
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(train_y)
            else:
                self.y_normalizer = UnitGaussianNormalizer(train_y)
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
            torch.utils.data.TensorDataset(eval_pos, eval_fx, eval_cond, eval_y),
            batch_size=self.batch_size, shuffle=False, **loader_kwargs)
        print("Aneumo CFD dataloading is over.")
        print(f"  train indices={len(train_idx)} eval_split={self.eval_split} eval indices={len(eval_idx)}")
        print(f"  train: pos={train_pos.shape} fx={train_fx.shape} cond={train_cond.shape} y={train_y.shape}")
        print(f"  eval:  pos={eval_pos.shape} fx={eval_fx.shape} cond={eval_cond.shape} y={eval_y.shape}")
        return train_loader, eval_loader, [train_y.shape[1]]

    @staticmethod
    def build_prompt(pos, cond, fx=None):
        return _aneumo_prompt(pos, fx, cond)
