import os

import numpy as np
import torch

from data_provider.hemo_loader import HemoPT
from utils.normalizer import UnitGaussianNormalizer, UnitTransformer


def _principal_axis(pos):
    centered = pos - pos.mean(dim=1, keepdim=True)
    cov = torch.matmul(centered.transpose(1, 2), centered) / max(pos.shape[1] - 1, 1)
    _, eigvecs = torch.linalg.eigh(cov)
    axis = eigvecs[:, :, -1]
    signs = torch.sign((centered * axis[:, None, :]).sum(dim=-1).mean(dim=1, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    axis = axis * signs
    return axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)


def _legacy_hemo_prompt(pos, cond):
    b, n, _ = pos.shape
    inlet_vel = cond[:, :, 1:2]
    axis = _principal_axis(pos)
    speed = inlet_vel.repeat(1, n, 1)
    return torch.cat([axis[:, None, :].repeat(1, n, 1) * speed, speed], dim=-1)


def _vmr_prompt(pos, fx, cond):
    b, n, _ = pos.shape
    axis = _principal_axis(pos)

    if fx is not None and fx.shape[-1] >= 4:
        dist = torch.clamp(fx[:, :, 0:1], min=0.0)
        direction_to_wall = fx[:, :, 1:4]
        radial_inward = -direction_to_wall
        dist_scale = dist.amax(dim=1, keepdim=True).clamp_min(1e-6)
        wall_weight = torch.clamp(dist / dist_scale, 0.0, 1.0)
    else:
        centered = pos - pos.mean(dim=1, keepdim=True)
        radial = centered - (centered * axis[:, None, :]).sum(dim=-1, keepdim=True) * axis[:, None, :]
        radial_inward = -radial / (radial.norm(dim=-1, keepdim=True) + 1e-8)
        wall_weight = torch.ones(b, n, 1, device=pos.device, dtype=pos.dtype)

    axial = (pos * axis[:, None, :]).sum(dim=-1, keepdim=True)
    axial_min = axial.amin(dim=1, keepdim=True)
    axial_max = axial.amax(dim=1, keepdim=True)
    axial_norm = (axial - axial_min) / (axial_max - axial_min + 1e-8)

    if cond.shape[-1] >= 12:
        flow_rms = cond[:, :, 2:3].clamp_min(1e-4)
        flow_peak = torch.maximum(cond[:, :, 3:4], flow_rms)
        pulsatility = cond[:, :, 4:5].clamp(0.0, 1.0)
        resistance = cond[:, :, 8:9].clamp_min(1e-4)
        length = cond[:, :, 10:11].clamp_min(1e-4)
        speed = (flow_rms + 0.25 * flow_peak * pulsatility).repeat(1, n, 1)
        pressure_drop_proxy = (flow_rms * resistance / (length + 1e-4)).repeat(1, n, 1)
    else:
        speed = cond[:, :, 1:2].repeat(1, n, 1)
        pressure_drop_proxy = torch.zeros(b, n, 1, device=pos.device, dtype=pos.dtype)

    profile = torch.clamp(wall_weight, 0.0, 1.0)
    flow_vec = axis[:, None, :].repeat(1, n, 1) * speed * profile
    wall_decay = (1.0 - profile) + 0.1 * radial_inward.norm(dim=-1, keepdim=True)
    scalar = speed * profile + 0.1 * pressure_drop_proxy * (1.0 - axial_norm) - 0.05 * wall_decay
    return torch.cat([flow_vec, scalar], dim=-1)


class VMRCFD(object):
    """Loader for processed VMR CFD interior-node samples."""

    def __init__(self, args):
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.eval_split = getattr(args, "vmr_eval_split", "test")
        self.cv_split_info = getattr(args, "cv_split_info", None)
        self.kfold = getattr(args, "kfold", 0)
        self.fold = getattr(args, "fold", 0)
        self.fold_seed = getattr(args, "fold_seed", 2026)
        self.fold_val_frac = getattr(args, "vmr_fold_val_frac", 0.15)
        self.kfold_strategy = getattr(args, "vmr_kfold_strategy", "group")
        self.stratify_bins = getattr(args, "vmr_stratify_bins", 3)
        self.common_families = self._normalize_filter(getattr(args, "vmr_families", None))
        self.common_territories = self._normalize_filter(getattr(args, "vmr_territories", None))
        self.train_families = self._normalize_filter(getattr(args, "vmr_train_families", None))
        self.train_territories = self._normalize_filter(getattr(args, "vmr_train_territories", None))
        self.eval_families = self._normalize_filter(getattr(args, "vmr_eval_families", None))
        self.eval_territories = self._normalize_filter(getattr(args, "vmr_eval_territories", None))
        self.hemo_target = getattr(args, "hemo_target", "full")
        self.wall_mode = getattr(args, "hemo_wall_mode", "keep")
        if self.wall_mode not in ("keep", "zero"):
            raise ValueError("--hemo_wall_mode must be keep or zero")
        if self.hemo_target not in ("full", "velocity"):
            raise ValueError("--hemo_target must be full or velocity")

        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(
                f"Unsupported norm_type: {self.norm_type}. Must be 'UnitTransformer' or 'UnitGaussianNormalizer'.")

    def _indices(self):
        if self.cv_split_info:
            info = np.load(self.cv_split_info, allow_pickle=True).item()
            folds = info.get("folds", [])
            if not 0 <= self.fold < len(folds):
                raise ValueError(f"fold must be in [0, {len(folds) - 1}], got {self.fold}")
            fold_info = folds[self.fold]
            train = self._filter_indices(list(fold_info["train_indices"]), mode="train")[:self.ntrain]
            eval_indices = list(fold_info["val_indices"]) if self.eval_split == "val" else list(info["test_indices"])
            test = self._filter_indices(eval_indices, mode="eval")[:self.ntest]
            print(f"  Using strict CV split: path={self.cv_split_info} fold={self.fold} "
                  f"train={len(train)} eval_split={self.eval_split} eval={len(test)}")
            if self._has_any_filter():
                print(f"  VMR filters: train={self._filter_text('train')} eval={self._filter_text('eval')}")
            return train, test

        if self.kfold and self.kfold > 1:
            train, val, test = self._make_fold_indices()
            train = self._filter_indices(train, mode="train")
            val = self._filter_indices(val, mode="eval")
            test = self._filter_indices(test, mode="eval")
            eval_indices = val if self.eval_split == "val" else test
            print(f"  Using VMR {self.kfold}-fold group split: fold={self.fold}, "
                  f"train={len(train)}, val={len(val)}, test={len(test)}, "
                  f"eval_split={self.eval_split}, seed={self.fold_seed}, "
                  f"strategy={self.kfold_strategy}")
            if self._has_any_filter():
                print(f"  VMR filters: train={self._filter_text('train')} eval={self._filter_text('eval')}")
            return train[:self.ntrain], eval_indices[:self.ntest]

        split_path = os.path.join(self.data_path, "global_split_info.npy")
        if os.path.exists(split_path):
            info = np.load(split_path, allow_pickle=True).item()
            train = self._filter_indices(list(info["train_indices"]), mode="train")[:self.ntrain]
            key = "val_indices" if self.eval_split == "val" else "test_indices"
            eval_indices = list(info.get(key, []))
            if not eval_indices and self.eval_split == "val":
                eval_indices = list(info.get("test_indices", []))
            test = self._filter_indices(eval_indices, mode="eval")[:self.ntest]
            if self._has_any_filter():
                print(f"  VMR filters: train={self._filter_text('train')} eval={self._filter_text('eval')}")
            return train, test
        xs = sorted(int(f[2:-4]) for f in os.listdir(self.data_path)
                    if f.startswith("x_") and f.endswith(".npy"))
        return xs[:self.ntrain], xs[-self.ntest:]

    def _all_indices_and_groups(self):
        split_path = os.path.join(self.data_path, "global_split_info.npy")
        if os.path.exists(split_path):
            info = np.load(split_path, allow_pickle=True).item()
            metadata = info.get("metadata", [])
            if metadata:
                all_indices = [int(row.get("index")) for row in metadata]
                allowed = set(self._filter_indices(all_indices, mode="fold"))
                indices, groups = [], {}
                for row in metadata:
                    idx = int(row.get("index"))
                    if idx not in allowed:
                        continue
                    group = str(row.get("case_id") or idx)
                    indices.append(idx)
                    groups.setdefault(group, []).append(idx)
                return sorted(indices), groups

        indices = sorted(int(f[2:-4]) for f in os.listdir(self.data_path)
                         if f.startswith("x_") and f.endswith(".npy"))
        indices = self._filter_indices(indices, mode="fold")
        groups = {str(i): [i] for i in indices}
        return indices, groups

    @staticmethod
    def _normalize_filter(values):
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        out = []
        for value in values:
            if value is None:
                continue
            for item in str(value).replace(",", " ").split():
                item = item.strip()
                if item:
                    out.append(item)
        if not out or any(item.lower() == "all" for item in out):
            return None
        return set(out)

    def _has_any_filter(self):
        return any(value is not None for value in [
            self.common_families, self.common_territories,
            self.train_families, self.train_territories,
            self.eval_families, self.eval_territories,
        ])

    def _filter_text(self, mode):
        families, territories = self._filters_for(mode)
        fam = "all" if families is None else ",".join(sorted(families))
        terr = "all" if territories is None else ",".join(sorted(territories))
        return f"families={fam}; territories={terr}"

    def _filters_for(self, mode):
        families = self.common_families
        territories = self.common_territories
        if mode == "train":
            families = self._combine_filters(families, self.train_families)
            territories = self._combine_filters(territories, self.train_territories)
        elif mode == "eval":
            families = self._combine_filters(families, self.eval_families)
            territories = self._combine_filters(territories, self.eval_territories)
        return families, territories

    @staticmethod
    def _combine_filters(common, specific):
        if common is None:
            return specific
        if specific is None:
            return common
        return set(common).intersection(specific)

    def _metadata_by_index(self):
        split_path = os.path.join(self.data_path, "global_split_info.npy")
        if not os.path.exists(split_path):
            return {}
        info = np.load(split_path, allow_pickle=True).item()
        return {int(row.get("index")): row for row in info.get("metadata", []) if "index" in row}

    def _filter_indices(self, indices, mode):
        families, territories = self._filters_for(mode)
        if families is None and territories is None:
            return list(indices)
        row_by_idx = self._metadata_by_index()
        if not row_by_idx:
            return list(indices)

        kept = []
        for idx in indices:
            row = row_by_idx.get(int(idx))
            if row is None:
                continue
            family = str(row.get("meta_territory_family") or row.get("territory", "NA")).split("_")[0]
            territory = str(row.get("territory", "NA"))
            if families is not None and family not in families:
                continue
            if territories is not None and territory not in territories:
                continue
            kept.append(int(idx))
        return kept

    def _make_fold_indices(self):
        if not 0 <= self.fold < self.kfold:
            raise ValueError(f"fold must be in [0, {self.kfold - 1}], got {self.fold}")

        indices, groups = self._all_indices_and_groups()
        if not indices:
            raise RuntimeError(f"VMRCFD found no samples under {self.data_path}")

        if self.kfold_strategy == "stratified_group":
            folds = self._make_stratified_group_folds(groups)
        elif self.kfold_strategy == "group":
            folds = self._make_balanced_group_folds(groups)
        else:
            raise ValueError("--vmr_kfold_strategy must be group or stratified_group")

        test_groups = set(folds[self.fold])
        remaining_groups = [g for i, fold in enumerate(folds) if i != self.fold for g in fold]

        val_target = max(1, int(round(len(indices) * float(self.fold_val_frac))))
        val_groups = self._select_val_groups(remaining_groups, groups, val_target)
        train_groups = [g for g in remaining_groups if g not in val_groups]

        def expand(selected_groups):
            out = []
            for group in selected_groups:
                out.extend(groups[group])
            return sorted(out)

        train = expand(train_groups)
        val = expand(val_groups)
        test = expand(test_groups)
        if not train or not val or not test:
            raise RuntimeError(
                f"Invalid VMR kfold split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def _make_balanced_group_folds(self, groups):
        rng = np.random.default_rng(self.fold_seed)
        group_names = np.array(sorted(groups), dtype=object)
        rng.shuffle(group_names)

        # Greedy balanced group folds keep all simulations from the same case_id together.
        folds = [[] for _ in range(self.kfold)]
        fold_sizes = [0 for _ in range(self.kfold)]
        for group in sorted(group_names, key=lambda name: len(groups[name]), reverse=True):
            target = int(np.argmin(fold_sizes))
            folds[target].append(group)
            fold_sizes[target] += len(groups[group])
        return folds

    def _make_stratified_group_folds(self, groups):
        split_path = os.path.join(self.data_path, "global_split_info.npy")
        metadata = []
        if os.path.exists(split_path):
            info = np.load(split_path, allow_pickle=True).item()
            metadata = info.get("metadata", [])
        row_by_idx = {int(row.get("index")): row for row in metadata if "index" in row}

        group_stats = {}
        speeds = []
        for group, group_indices in groups.items():
            rows = [row_by_idx.get(int(i), {}) for i in group_indices]
            territories = [str(row.get("meta_territory_family") or row.get("territory", "NA")).split("_")[0]
                           for row in rows]
            territory = max(set(territories), key=territories.count) if territories else "NA"
            vmax_values = [float(row.get("vmax", 0.0) or 0.0) for row in rows]
            speed = float(np.nanmean(vmax_values)) if vmax_values else 0.0
            group_stats[group] = {"territory": territory, "speed": speed}
            speeds.append(speed)

        finite_speeds = np.array([s for s in speeds if np.isfinite(s)], dtype=np.float64)
        if finite_speeds.size == 0:
            return self._make_balanced_group_folds(groups)
        n_bins = max(1, int(self.stratify_bins))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.unique(np.quantile(finite_speeds, quantiles)) if quantiles.size else np.array([])

        strata = {}
        for group, stat in group_stats.items():
            speed_bin = int(np.searchsorted(edges, stat["speed"], side="right")) if edges.size else 0
            key = (stat["territory"], speed_bin)
            strata.setdefault(key, []).append(group)

        folds = [[] for _ in range(self.kfold)]
        fold_sizes = [0 for _ in range(self.kfold)]
        rng = np.random.default_rng(self.fold_seed)
        for key in sorted(strata):
            group_names = np.array(strata[key], dtype=object)
            rng.shuffle(group_names)
            for group in sorted(group_names, key=lambda name: len(groups[name]), reverse=True):
                target = int(np.argmin(fold_sizes))
                folds[target].append(group)
                fold_sizes[target] += len(groups[group])
        return folds

    def _select_val_groups(self, remaining_groups, groups, val_target):
        val_rng = np.random.default_rng(self.fold_seed + 1009 + self.fold)
        remaining_groups = list(remaining_groups)
        val_rng.shuffle(remaining_groups)

        val_groups = set()
        val_count = 0
        for group in remaining_groups:
            if len(remaining_groups) - len(val_groups) <= 1:
                break
            val_groups.add(group)
            val_count += len(groups[group])
            if val_count >= val_target:
                break
        return val_groups

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

    def _select_target(self, y):
        if self.hemo_target == "velocity":
            if y.shape[-1] == 4:
                print("  hemo_target=velocity: target y is [u,v,w] sliced from [p,u,v,w] before normalization")
                return y[:, :, 1:4]
            if y.shape[-1] == 3:
                print("  hemo_target=velocity: target y is native [u,v,w] before normalization")
                return y
            raise ValueError(f"velocity target expects y with 3 or 4 channels, got {y.shape[-1]}")
        if y.shape[-1] != 4:
            raise ValueError(f"full target expects y=[p,u,v,w] with 4 channels, got {y.shape[-1]}")
        return y

    def get_loader(self, full_mesh=True):
        train_idx, test_idx = self._indices()
        if not train_idx or not test_idx:
            raise RuntimeError(f"VMRCFD requires non-empty train/test indices under {self.data_path}")
        train_pos, train_fx, train_cond, train_y = self._load(train_idx)
        test_pos, test_fx, test_cond, test_y = self._load(test_idx)
        if self.wall_mode == "zero":
            train_fx = torch.zeros_like(train_fx)
            test_fx = torch.zeros_like(test_fx)
            print("  hemo_wall_mode=zero: wall distance/direction channels are zeroed")
        else:
            print("  hemo_wall_mode=keep: wall distance/direction channels are used")

        train_y = self._select_target(train_y)
        test_y = self._select_target(test_y)
        if self.hemo_target == "full":
            print("  hemo_target=full: target y is [p,u,v,w]")

        if self.normalize:
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(train_y)
            else:
                self.y_normalizer = UnitGaussianNormalizer(train_y)
            train_y = self.y_normalizer.encode(train_y)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_pos, train_fx, train_cond, train_y),
            batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_pos, test_fx, test_cond, test_y),
            batch_size=self.batch_size, shuffle=False)
        print("VMR CFD dataloading is over.")
        print(f"  train: pos={train_pos.shape} fx={train_fx.shape} y={train_y.shape}")
        print(f"  test:  pos={test_pos.shape} fx={test_fx.shape} y={test_y.shape}")
        return train_loader, test_loader, [train_y.shape[1]]

    def build_prompt(self, pos, cond, fx=None):
        if cond.shape[-1] >= 12:
            prompt_fx = None if self.wall_mode == "zero" else fx
            return _vmr_prompt(pos, prompt_fx, cond)
        return _legacy_hemo_prompt(pos, cond)
