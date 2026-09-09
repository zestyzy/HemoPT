import os
import numpy as np
import torch
from utils.normalizer import UnitTransformer, UnitGaussianNormalizer


HEMO_CONFIG_TO_ID = {
    "C1": 0,
    "ICA_norm": 1,
    "ICA_ste": 2,
}


class HemoPT(object):
    """
    Data loader for intravascular blood flow CFD data.

    Data format (from HemoPT_process.py):
      x_i.npy:    (N, 6) = [x, y, z, 0, 0, 0] — coordinates + placeholder features
      y_i.npy:    (N, 4) = [p, u, v, w] — pressure + 3D velocity
      cond_i.npy: (3,)   = [vessel_type, inlet_velocity, viscosity]

    Total: 133 samples (C1:30, ICA_norm:62, ICA_ste:41)
           train: 107 (C1:24, ICA_norm:50, ICA_ste:33)
           test:   26 (C1:6,  ICA_norm:12, ICA_ste:8)

    Data is organized sequentially: all C1, then all ICA_norm, then all ICA_ste.
    Within each config: train files first (sorted), then test files (sorted).
    """

    def __init__(self, args):
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.num_workers = getattr(args, "num_workers", 0)
        self.pin_memory = getattr(args, "pin_memory", False)
        self.prefetch_factor = getattr(args, "prefetch_factor", 2)
        self.kfold = getattr(args, "kfold", 0)
        self.fold = getattr(args, "fold", 0)
        self.fold_seed = getattr(args, "fold_seed", 2026)
        self.eval_split = getattr(args, "hemo_eval_split", "test")
        self.split_info = getattr(args, "hemo_split_info", None)
        self.cv_split_info = getattr(args, "cv_split_info", None)
        self.hemo_configs = getattr(args, "hemo_configs", ["C1", "ICA_norm", "ICA_ste"])
        self.hemo_target = getattr(args, "hemo_target", "full")
        if self.hemo_target not in ("full", "velocity"):
            raise ValueError("--hemo_target must be full or velocity")
        unknown = sorted(set(self.hemo_configs) - set(HEMO_CONFIG_TO_ID))
        if unknown:
            raise ValueError(f"Unsupported hemo_configs={unknown}. Supported: {sorted(HEMO_CONFIG_TO_ID)}")
        self.allowed_vessel_types = {HEMO_CONFIG_TO_ID[name] for name in self.hemo_configs}

        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def get_loader(self, full_mesh=True):
        print("loading HemoPT dataset...")

        if self.cv_split_info:
            split_info = np.load(self.cv_split_info, allow_pickle=True).item()
            folds = split_info.get("folds", [])
            if not 0 <= self.fold < len(folds):
                raise ValueError(f"fold must be in [0, {len(folds) - 1}], got {self.fold}")
            fold_info = folds[self.fold]
            train_indices = self._filter_indices(list(fold_info["train_indices"]))[:self.ntrain]
            eval_indices = list(fold_info["val_indices"]) if self.eval_split == "val" else list(split_info["test_indices"])
            test_indices = self._filter_indices(eval_indices)[:self.ntest]
            print(f"  Using strict CV split: path={self.cv_split_info} fold={self.fold} "
                  f"train={len(train_indices)}, eval_split={self.eval_split} eval={len(test_indices)}")
            print(f"  hemo_configs={self.hemo_configs}")
        elif self.kfold and self.kfold > 1:
            indices = self._all_filtered_indices()
            train_indices, test_indices = self._make_fold_indices(indices)
            train_indices = train_indices[:self.ntrain]
            test_indices = test_indices[:self.ntest]
            print(f"  Using {self.kfold}-fold split: fold={self.fold}, "
                  f"train={len(train_indices)}, test={len(test_indices)}, seed={self.fold_seed}")
            print(f"  hemo_configs={self.hemo_configs}")
        else:
            # Load global split info if available.
            split_path = self.split_info or os.path.join(self.data_path, "global_split_info.npy")
            if os.path.exists(split_path):
                split_info = np.load(split_path, allow_pickle=True).item()
                train_indices = self._filter_indices(list(split_info["train_indices"]))[:self.ntrain]
                eval_key = "val_indices" if self.eval_split == "val" else "test_indices"
                if eval_key not in split_info:
                    if self.eval_split == "val" and "test_indices" in split_info:
                        eval_key = "test_indices"
                        print("  Warning: requested hemo_eval_split=val but split has no val_indices; using test_indices.")
                    else:
                        raise KeyError(f"{split_path} does not contain {eval_key}")
                test_indices = self._filter_indices(list(split_info[eval_key]))[:self.ntest]
                print(f"  Using saved split: path={split_path} train={len(train_indices)}, "
                      f"eval_split={self.eval_split} eval={len(test_indices)}")
                print(f"  hemo_configs={self.hemo_configs}")
            else:
                # Fallback: sequential split.
                all_x = sorted([f for f in os.listdir(self.data_path) if f.startswith("x_") and f.endswith(".npy")])
                n_total = len(all_x)
                train_indices = list(range(self.ntrain))
                test_indices = list(range(n_total - self.ntest, n_total))
                print(f"  Found {n_total} samples, using first {self.ntrain} for train, last {self.ntest} for test")

        def load_samples(indices):
            xs, ys, conds = [], [], []
            for i in indices:
                xs.append(np.load(os.path.join(self.data_path, f"x_{i}.npy")))
                ys.append(np.load(os.path.join(self.data_path, f"y_{i}.npy")))
                conds.append(np.load(os.path.join(self.data_path, f"cond_{i}.npy")))
            return xs, ys, conds

        train_x, train_y, train_cond = load_samples(train_indices)
        test_x, test_y, test_cond = load_samples(test_indices)

        # Stack into tensors with separate spatial positions and feature channels.
        train_x = torch.tensor(np.array(train_x), dtype=torch.float)
        train_y = torch.tensor(np.array(train_y), dtype=torch.float)
        train_cond = torch.tensor(np.array(train_cond), dtype=torch.float)[:, None, :]

        test_x = torch.tensor(np.array(test_x), dtype=torch.float)
        test_y = torch.tensor(np.array(test_y), dtype=torch.float)
        test_cond = torch.tensor(np.array(test_cond), dtype=torch.float)[:, None, :]

        if self.hemo_target == "velocity":
            train_y = train_y[:, :, 1:4]
            test_y = test_y[:, :, 1:4]
            print("  hemo_target=velocity: target y is [u,v,w] before normalization")
        else:
            print("  hemo_target=full: target y is [p,u,v,w]")

        train_pos = train_x[:, :, :3]
        train_fx = train_x[:, :, 3:]
        test_pos = test_x[:, :, :3]
        test_fx = test_x[:, :, 3:]
        if train_fx.shape[-1] == 3:
            train_fx = torch.cat([train_fx, torch.zeros_like(train_fx[:, :, :1])], dim=-1)
            test_fx = torch.cat([test_fx, torch.zeros_like(test_fx[:, :, :1])], dim=-1)

        print(f"  train: pos={train_pos.shape} fx={train_fx.shape} y={train_y.shape} cond={train_cond.shape}")
        print(f"  test:  pos={test_pos.shape} fx={test_fx.shape} y={test_y.shape} cond={test_cond.shape}")

        if self.normalize:
            if self.norm_type == 'UnitTransformer':
                self.y_normalizer = UnitTransformer(train_y)
            elif self.norm_type == 'UnitGaussianNormalizer':
                self.y_normalizer = UnitGaussianNormalizer(train_y)
            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        # Loader convention: TensorDataset(pos, fx, cond, y).
        # hemo_cfd_finetune appends a 4-channel hemo prompt, so fun_dim=8.
        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_fx, train_cond, train_y),
            batch_size=self.batch_size, shuffle=True, **loader_kwargs)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_pos, test_fx, test_cond, test_y),
            batch_size=self.batch_size, shuffle=False, **loader_kwargs)

        print("HemoPT dataloading is over.")
        return train_loader, test_loader, [train_y.shape[1]]

    def _filter_indices(self, indices):
        if self.allowed_vessel_types == set(HEMO_CONFIG_TO_ID.values()):
            return indices
        kept = []
        for i in indices:
            cond = np.load(os.path.join(self.data_path, f"cond_{i}.npy"))
            if int(round(float(cond[0]))) in self.allowed_vessel_types:
                kept.append(i)
        return kept

    def _all_filtered_indices(self):
        files = [f for f in os.listdir(self.data_path) if f.startswith("cond_") and f.endswith(".npy")]
        indices = sorted(int(f[5:-4]) for f in files)
        return self._filter_indices(indices)

    def _make_fold_indices(self, indices):
        if not 0 <= self.fold < self.kfold:
            raise ValueError(f"fold must be in [0, {self.kfold - 1}], got {self.fold}")
        indices = np.array(indices, dtype=np.int64)
        rng = np.random.default_rng(self.fold_seed)
        perm = rng.permutation(indices)
        folds = np.array_split(perm, self.kfold)
        test_indices = np.sort(folds[self.fold]).astype(int).tolist()
        train_indices = np.sort(np.concatenate([folds[i] for i in range(self.kfold) if i != self.fold])).astype(int).tolist()
        return train_indices, test_indices

    @staticmethod
    def build_prompt(pos, cond, fx=None):
        b, n, _ = pos.shape
        inlet_vel = cond[:, :, 1:2]
        centered = pos - pos.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / max(pos.shape[1] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        axis = eigvecs[:, :, -1]
        signs = torch.sign((centered * axis[:, None, :]).sum(dim=-1).mean(dim=1, keepdim=True))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        axis = axis * signs
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        speed = inlet_vel.repeat(1, n, 1)
        return torch.cat([axis[:, None, :].repeat(1, n, 1) * speed, speed], dim=-1)
