import glob
import json
import os
import random

import numpy as np
import torch


DEFAULT_VASCULAR_QC_MANIFEST = ""
GEOMETRIC_TARGET_DIM = 9
PHYSICS_PROXY_DIM = 7
VELOCITY_PROXY_DIM = 4
GENERALIZED_FLOW_COMPACT_DIM = 4
GENERALIZED_FLOW_BANK_DIM = 32
PHYSICS_PROXY_DIMS = {
    "full": PHYSICS_PROXY_DIM,
    "velocity_only": VELOCITY_PROXY_DIM,
    "conditioned_velocity": VELOCITY_PROXY_DIM,
    "generalized_flow_compact": GENERALIZED_FLOW_COMPACT_DIM,
    "conditioned_generalized_flow_compact": GENERALIZED_FLOW_COMPACT_DIM,
    "generalized_flow_bank": GENERALIZED_FLOW_BANK_DIM,
}


def _normalize_statuses(statuses):
    result = []
    for item in statuses:
        for status in str(item).split(","):
            status = status.strip().lower()
            if status:
                result.append(status)
    return set(result)


def _load_qc_manifest(path):
    if not path or not os.path.exists(path):
        return {}

    rows = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = row.get("dataset")
            filename = row.get("filename")
            if not dataset or not filename:
                continue
            stem = os.path.splitext(filename)[0]
            rows[(dataset, stem)] = row
    return rows


class _VascularPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, sample_dirs, n_random_walks=100, random_walk=True,
                 physics_proxy=False, physics_proxy_mode="full",
                 wall_mask_prob=0.0, wall_mask_mode="all"):
        self.sample_dirs = list(sample_dirs)
        self.n_random_walks = n_random_walks
        self.random_walk = random_walk
        self.physics_proxy = physics_proxy
        self.physics_proxy_mode = physics_proxy_mode
        self.wall_mask_prob = float(wall_mask_prob)
        self.wall_mask_mode = wall_mask_mode
        self._proxy_meta_cache = {}

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        walk_idx = random.randrange(self.n_random_walks) if self.random_walk else 0

        x = np.load(os.path.join(sample_dir, "x.npy")).astype(np.float32)
        condition = np.load(os.path.join(sample_dir, f"condition_{walk_idx}.npy")).astype(np.float32)
        supervise = np.load(os.path.join(sample_dir, f"supervise_{walk_idx}.npy")).astype(np.float32)

        pos = x[:, :3]
        geom = x[:, 3:7]
        if self.random_walk and self.wall_mask_prob > 0.0 and random.random() < self.wall_mask_prob:
            geom = geom.copy()
            if self.wall_mask_mode == "distance":
                geom[:, 0] = 0.0
            elif self.wall_mask_mode == "direction":
                geom[:, 1:4] = 0.0
            else:
                geom[:, :] = 0.0
        fx = np.concatenate([geom, condition], axis=-1)
        if self.physics_proxy:
            physics_proxy = self._build_physics_proxy(x, condition, sample_dir)
            supervise = np.concatenate([supervise, physics_proxy], axis=-1)
        dummy_cond = np.zeros((1,), dtype=np.float32)

        return (torch.from_numpy(pos),
                torch.from_numpy(fx),
                torch.from_numpy(dummy_cond),
                torch.from_numpy(supervise))

    def _build_physics_proxy(self, x, condition, sample_dir):
        pos = x[:, :3]
        dist = np.maximum(x[:, 3], 0.0)
        center, axis, side_axis, normal_axis, axial_min, axial_span, dist_scale = self._proxy_meta(
            sample_dir, pos, dist)

        axial = ((pos - center) @ axis - axial_min) / axial_span
        axial = np.clip(axial, 0.0, 1.0).astype(np.float32)

        dist_norm = np.clip(dist / dist_scale, 0.0, 1.0).astype(np.float32)
        speed = (2.0 * dist_norm - dist_norm * dist_norm).astype(np.float32)
        flow = speed[:, None] * axis[None, :].astype(np.float32)
        if self.physics_proxy_mode == "conditioned_velocity":
            return self._build_conditioned_velocity_proxy(
                x=x,
                condition=condition,
                center=center,
                axis=axis,
                side_axis=side_axis,
                normal_axis=normal_axis,
                axial=axial,
                dist_norm=dist_norm,
                compact=False,
            )
        if self.physics_proxy_mode == "velocity_only":
            return np.concatenate([
                flow,
                speed[:, None],
            ], axis=-1).astype(np.float32)
        if self.physics_proxy_mode == "conditioned_generalized_flow_compact":
            return self._build_conditioned_velocity_proxy(
                x=x,
                condition=condition,
                center=center,
                axis=axis,
                side_axis=side_axis,
                normal_axis=normal_axis,
                axial=axial,
                dist_norm=dist_norm,
                compact=True,
            )
        if self.physics_proxy_mode in ("generalized_flow_compact", "generalized_flow_bank"):
            return self._build_generalized_flow_proxy(
                x=x,
                center=center,
                axis=axis,
                side_axis=side_axis,
                normal_axis=normal_axis,
                axial=axial,
                dist_norm=dist_norm,
            )

        pressure = (1.0 - axial).astype(np.float32)
        no_slip_weight = (1.0 - dist_norm).astype(np.float32)

        return np.concatenate([
            flow,
            speed[:, None],
            pressure[:, None],
            no_slip_weight[:, None],
            axial[:, None],
        ], axis=-1).astype(np.float32)

    def _condition_features(self, condition, axis, fallback_dir):
        cond = np.asarray(condition, dtype=np.float32)
        walk_dir = self._normalize_vectors(cond[:, :3], axis)
        axis = axis.astype(np.float32)

        alignment = np.clip(walk_dir @ axis, -1.0, 1.0).astype(np.float32)
        transverse = walk_dir - alignment[:, None] * axis[None, :]
        transverse_dir = self._normalize_vectors(transverse, fallback_dir)

        step = np.abs(cond[:, 3]).astype(np.float32) if cond.shape[-1] > 3 else np.ones(cond.shape[0], dtype=np.float32)
        finite_step = step[np.isfinite(step) & (step > 0.0)]
        if finite_step.size == 0:
            step_scale = 1.0
        else:
            step_scale = float(np.percentile(finite_step, 95))
            if not np.isfinite(step_scale) or step_scale < 1e-8:
                step_scale = float(np.max(finite_step))
            if not np.isfinite(step_scale) or step_scale < 1e-8:
                step_scale = 1.0
        step_norm = np.clip(step / step_scale, 0.0, 1.0).astype(np.float32)
        return walk_dir, alignment, transverse_dir, step_norm

    def _build_conditioned_velocity_proxy(self, x, condition, center, axis, side_axis,
                                          normal_axis, axial, dist_norm, compact=False):
        pos = x[:, :3].astype(np.float32)
        axis = axis.astype(np.float32)
        side_axis = side_axis.astype(np.float32)
        normal_axis = normal_axis.astype(np.float32)

        centered = pos - center[None, :].astype(np.float32)
        axial_offset = (centered @ axis)[:, None] * axis[None, :]
        radial = centered - axial_offset
        radial_dir = self._normalize_vectors(radial, side_axis)
        swirl_dir = self._normalize_vectors(np.cross(axis[None, :], radial_dir), normal_axis)

        walk_dir, alignment, transverse_dir, step_norm = self._condition_features(
            condition=condition,
            axis=axis,
            fallback_dir=side_axis,
        )

        rho = np.clip(dist_norm.astype(np.float32), 0.0, 1.0)
        wall = 1.0 - rho
        axial_phase = np.sin(np.pi * axial).astype(np.float32)
        lateral_walk = np.clip(np.sum(radial_dir * walk_dir, axis=-1), -1.0, 1.0).astype(np.float32)

        parabolic_speed = (2.0 * rho - rho * rho).astype(np.float32)
        plug_speed = (1.0 - wall ** 6).astype(np.float32)
        skew_gain = np.clip(1.0 + 0.35 * lateral_walk, 0.55, 1.45).astype(np.float32)

        plug_mix = np.clip(0.25 + 0.35 * step_norm + 0.20 * np.maximum(alignment, 0.0), 0.05, 0.85)
        pulse_gain = np.clip(1.0 + 0.22 * alignment * axial_phase, 0.65, 1.30).astype(np.float32)
        base_speed = ((1.0 - plug_mix) * parabolic_speed + plug_mix * plug_speed).astype(np.float32)
        speed = np.clip(base_speed * skew_gain * pulse_gain, 0.0, 1.35).astype(np.float32)

        transverse_strength = (0.20 + 0.20 * step_norm)[:, None]
        swirl_strength = (0.10 * (1.0 - np.abs(alignment)) * rho * wall)[:, None]
        flow_dir = self._normalize_vectors(
            axis[None, :] + transverse_strength * transverse_dir + swirl_strength * swirl_dir,
            axis,
        )
        flow = speed[:, None] * flow_dir

        if not compact:
            return self._pack_flow(flow)

        reverse = np.clip(np.maximum(-alignment, 0.0) * axial_phase * wall, 0.0, 1.0).astype(np.float32)
        dean = parabolic_speed[:, None] * axis[None, :] + (
            0.22 * rho * wall * axial_phase * lateral_walk
        )[:, None] * normal_axis[None, :]
        helical = parabolic_speed[:, None] * axis[None, :] + (
            0.20 * rho * wall * (0.5 + 0.5 * step_norm)
        )[:, None] * swirl_dir
        recirc = (parabolic_speed * (1.0 - 0.55 * reverse) - 0.18 * reverse)[:, None] * axis[None, :]
        recirc += (0.10 * reverse)[:, None] * radial_dir

        w_conditioned = np.clip(0.45 + 0.25 * step_norm, 0.25, 0.75)
        w_dean = np.clip(0.12 + 0.18 * np.abs(lateral_walk), 0.05, 0.35)
        w_helical = np.clip(0.10 + 0.20 * (1.0 - np.abs(alignment)), 0.05, 0.35)
        w_recirc = np.clip(0.05 + 0.25 * np.maximum(-alignment, 0.0), 0.02, 0.30)
        w_sum = w_conditioned + w_dean + w_helical + w_recirc

        compact_flow = (
            (w_conditioned / w_sum)[:, None] * flow
            + (w_dean / w_sum)[:, None] * dean
            + (w_helical / w_sum)[:, None] * helical
            + (w_recirc / w_sum)[:, None] * recirc
        ).astype(np.float32)
        return self._pack_flow(compact_flow)

    @staticmethod
    def _normalize_vectors(vec, fallback):
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        out = vec / np.maximum(norm, 1e-8)
        bad = (~np.isfinite(out).all(axis=-1)) | (norm[:, 0] < 1e-8)
        if np.any(bad):
            out[bad] = fallback
        return out.astype(np.float32)

    @staticmethod
    def _pack_flow(flow):
        speed = np.linalg.norm(flow, axis=-1, keepdims=True)
        return np.concatenate([flow, speed], axis=-1).astype(np.float32)

    def _build_generalized_flow_proxy(self, x, center, axis, side_axis, normal_axis, axial, dist_norm):
        pos = x[:, :3].astype(np.float32)
        axis = axis.astype(np.float32)
        side_axis = side_axis.astype(np.float32)
        normal_axis = normal_axis.astype(np.float32)

        centered = pos - center[None, :].astype(np.float32)
        axial_offset = (centered @ axis)[:, None] * axis[None, :]
        radial = centered - axial_offset
        radial_dir = self._normalize_vectors(radial, side_axis)
        swirl_dir = self._normalize_vectors(np.cross(axis[None, :], radial_dir), normal_axis)

        rho = np.clip(dist_norm.astype(np.float32), 0.0, 1.0)
        wall = 1.0 - rho
        axial_phase = np.sin(np.pi * axial).astype(np.float32)
        lateral = np.clip(radial_dir @ side_axis, -1.0, 1.0).astype(np.float32)

        parabolic_speed = (2.0 * rho - rho * rho).astype(np.float32)
        plug_speed = (1.0 - wall ** 6).astype(np.float32)
        mixed_speed = (0.65 * parabolic_speed + 0.35 * plug_speed).astype(np.float32)

        parabolic = parabolic_speed[:, None] * axis[None, :]
        plug = plug_speed[:, None] * axis[None, :]

        accel_speed = np.clip(mixed_speed * (0.85 + 0.30 * axial_phase), 0.0, 1.25).astype(np.float32)
        pulsatile_accel = accel_speed[:, None] * axis[None, :]

        reversal = (0.30 * wall * axial_phase).astype(np.float32)
        decel_speed = (0.70 * parabolic_speed - reversal).astype(np.float32)
        pulsatile_decel = decel_speed[:, None] * axis[None, :]

        skew_gain = np.clip(1.0 + 0.40 * lateral, 0.50, 1.50).astype(np.float32)
        skew_dir = self._normalize_vectors(axis[None, :] + 0.18 * side_axis[None, :], axis)
        skewed = (parabolic_speed * skew_gain)[:, None] * skew_dir

        secondary_amp = (0.28 * rho * wall * axial_phase * lateral).astype(np.float32)
        dean = parabolic + secondary_amp[:, None] * normal_axis[None, :]

        swirl_amp = (0.22 * rho * wall * (0.5 + 0.5 * axial_phase)).astype(np.float32)
        helical = parabolic + swirl_amp[:, None] * swirl_dir

        pocket = (np.exp(-((axial - 0.65) / 0.22) ** 2).astype(np.float32) * wall).astype(np.float32)
        recirc = (parabolic_speed * (1.0 - 0.70 * pocket) - 0.22 * pocket)[:, None] * axis[None, :]
        recirc += (0.10 * pocket)[:, None] * radial_dir

        modes = [
            parabolic.astype(np.float32),
            plug.astype(np.float32),
            pulsatile_accel.astype(np.float32),
            pulsatile_decel.astype(np.float32),
            skewed.astype(np.float32),
            dean.astype(np.float32),
            helical.astype(np.float32),
            recirc.astype(np.float32),
        ]

        if self.physics_proxy_mode == "generalized_flow_bank":
            return np.concatenate([self._pack_flow(mode) for mode in modes], axis=-1).astype(np.float32)

        weights = np.array([0.24, 0.18, 0.16, 0.10, 0.14, 0.08, 0.06, 0.04], dtype=np.float32)
        compact = np.zeros_like(modes[0], dtype=np.float32)
        for weight, mode in zip(weights, modes):
            compact += weight * mode
        return self._pack_flow(compact)

    def _proxy_meta(self, sample_dir, pos, dist):
        cached = self._proxy_meta_cache.get(sample_dir)
        if cached is not None:
            return cached

        pos64 = np.asarray(pos, dtype=np.float64)
        center = np.nanmean(pos64, axis=0)
        if not np.all(np.isfinite(center)):
            center = np.zeros((3,), dtype=np.float64)
        centered = pos64 - center
        if not np.all(np.isfinite(centered)):
            centered = np.nan_to_num(centered, copy=False)

        if centered.shape[0] > 1:
            cov = centered.T @ centered / float(centered.shape[0] - 1)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)
                axis = eigvecs[:, int(order[-1])]
                side_axis = eigvecs[:, int(order[-2])]
                normal_axis = eigvecs[:, int(order[0])]
            except np.linalg.LinAlgError:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                side_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                normal_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            side_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            normal_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        axis_norm = np.linalg.norm(axis)
        if not np.isfinite(axis_norm) or axis_norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis

        side_axis = side_axis - axis * float(np.dot(side_axis, axis))
        side_norm = np.linalg.norm(side_axis)
        if not np.isfinite(side_norm) or side_norm < 1e-8:
            side_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            side_axis = side_axis - axis * float(np.dot(side_axis, axis))
            side_norm = np.linalg.norm(side_axis)
            if not np.isfinite(side_norm) or side_norm < 1e-8:
                side_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                side_axis = side_axis - axis * float(np.dot(side_axis, axis))
                side_norm = np.linalg.norm(side_axis)
        side_axis = side_axis / max(side_norm, 1e-8)
        side_dominant = int(np.argmax(np.abs(side_axis)))
        if side_axis[side_dominant] < 0:
            side_axis = -side_axis

        normal_axis = np.cross(axis, side_axis)
        normal_norm = np.linalg.norm(normal_axis)
        if not np.isfinite(normal_norm) or normal_norm < 1e-8:
            normal_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normal_axis = normal_axis / normal_norm

        axial_raw = centered @ axis
        finite_axial = axial_raw[np.isfinite(axial_raw)]
        if finite_axial.size == 0:
            axial_min = 0.0
            axial_span = 1.0
        else:
            axial_min = float(np.min(finite_axial))
            axial_span = float(np.max(finite_axial) - axial_min)
            if not np.isfinite(axial_span) or axial_span < 1e-8:
                axial_span = 1.0

        finite_dist = dist[np.isfinite(dist) & (dist > 0.0)]
        if finite_dist.size == 0:
            dist_scale = 1.0
        else:
            dist_scale = float(np.percentile(finite_dist, 95))
            if not np.isfinite(dist_scale) or dist_scale < 1e-8:
                dist_scale = float(np.max(finite_dist))
            if not np.isfinite(dist_scale) or dist_scale < 1e-8:
                dist_scale = 1.0

        meta = (
            center.astype(np.float32),
            axis.astype(np.float32),
            side_axis.astype(np.float32),
            normal_axis.astype(np.float32),
            np.float32(axial_min),
            np.float32(axial_span),
            np.float32(dist_scale),
        )
        self._proxy_meta_cache[sample_dir] = meta
        return meta


class VascularPretrain(object):
    """
    Loader for vascular lifted geometric pre-training data.

    Expected sample layout:
      sample_dir/x.npy:           (N, 7)  [xyz, dist_to_wall, direction_or_normal]
      sample_dir/condition_j.npy: (N, 4)  [dx, dy, dz, step_length]
      sample_dir/supervise_j.npy: (N, 9)  3-step wall-vector trajectory

    Model input convention:
      pos = x[:, :3]
      fx  = concat(x[:, 3:7], condition_j), so fun_dim=8 and out_dim=9.

    Optional full physics proxy mode appends 7 deterministic proxy targets:
      [flow_x, flow_y, flow_z, speed, pressure_proxy, no_slip_weight, axial_coord].
      Use out_dim=16 when enabled.

    Optional velocity_only proxy mode appends 4 deterministic proxy targets:
      [flow_x, flow_y, flow_z, speed].
      Use out_dim=13 when enabled.

    Optional conditioned_velocity proxy mode appends 4 random-walk-conditioned
    velocity proxy targets:
      [flow_x, flow_y, flow_z, speed].
      Use out_dim=13 when enabled.

    Optional generalized_flow_compact mode appends 4 deterministic proxy targets:
      [flow_x, flow_y, flow_z, speed] from a weighted canonical flow basis.
      Use out_dim=13 when enabled.

    Optional conditioned_generalized_flow_compact mode appends 4 random-walk-
    conditioned proxy targets from a compact flow mixture.
      [flow_x, flow_y, flow_z, speed].
      Use out_dim=13 when enabled.

    Optional generalized_flow_bank mode appends 32 deterministic proxy targets:
      8 canonical flow modes, each [flow_x, flow_y, flow_z, speed].
      Use out_dim=41 when enabled.
    """

    def __init__(self, args):
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.n_random_walks = getattr(args, "n_random_walks", 100)
        self.num_workers = getattr(args, "num_workers", 0)
        self.pin_memory = getattr(args, "pin_memory", False)
        self.prefetch_factor = getattr(args, "prefetch_factor", 2)
        self.qc_statuses = _normalize_statuses(
            getattr(args, "vascular_qc_statuses", ["pass", "warn"]))
        self.qc_manifest = getattr(args, "vascular_qc_manifest", DEFAULT_VASCULAR_QC_MANIFEST)
        self.qc_by_sample = _load_qc_manifest(self.qc_manifest)
        self.physics_proxy = getattr(args, "vascular_physics_proxy", False)
        self.physics_proxy_mode = getattr(args, "vascular_physics_proxy_mode", "full")
        if self.physics_proxy_mode not in PHYSICS_PROXY_DIMS:
            raise ValueError(
                "--vascular_physics_proxy_mode must be one of: "
                + ", ".join(sorted(PHYSICS_PROXY_DIMS.keys())))
        self.wall_mask_prob = getattr(args, "vascular_wall_mask_prob", 0.0)
        self.wall_mask_mode = getattr(args, "vascular_wall_mask_mode", "all")
        if self.wall_mask_mode not in ("all", "distance", "direction"):
            raise ValueError("--vascular_wall_mask_mode must be one of: all, distance, direction")
        if not 0.0 <= float(self.wall_mask_prob) <= 1.0:
            raise ValueError("--vascular_wall_mask_prob must be in [0, 1]")
        expected_out_dim = GEOMETRIC_TARGET_DIM + PHYSICS_PROXY_DIMS[self.physics_proxy_mode]
        if self.physics_proxy and getattr(args, "out_dim", expected_out_dim) != expected_out_dim:
            raise ValueError(
                f"--vascular_physics_proxy mode={self.physics_proxy_mode} requires --out_dim {expected_out_dim}; "
                f"got {getattr(args, 'out_dim', None)}")

    def _passes_meta_qc(self, sample_dir):
        dataset = os.path.basename(os.path.dirname(sample_dir))
        sample_name = os.path.basename(sample_dir)
        meta_path = os.path.join(sample_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                return False
            if meta.get("success") is False:
                return False

        qc_status = meta.get("qc_status")
        if qc_status is None:
            qc_row = self.qc_by_sample.get((dataset, sample_name))
            if qc_row:
                qc_status = qc_row.get("status")
        if qc_status is None:
            return True
        return str(qc_status).lower() in self.qc_statuses

    def _discover_samples(self):
        pattern = os.path.join(self.data_path, "*", "*", "x.npy")
        candidates = sorted(glob.glob(pattern))
        sample_dirs = []
        for x_path in candidates:
            sample_dir = os.path.dirname(x_path)
            if not self._passes_meta_qc(sample_dir):
                continue
            complete = True
            for j in range(self.n_random_walks):
                if not (os.path.exists(os.path.join(sample_dir, f"condition_{j}.npy")) and
                        os.path.exists(os.path.join(sample_dir, f"supervise_{j}.npy"))):
                    complete = False
                    break
            if complete:
                sample_dirs.append(sample_dir)
        return sample_dirs

    def get_loader(self, full_mesh=False):
        sample_dirs = self._discover_samples()
        if len(sample_dirs) == 0:
            raise RuntimeError(f"No complete vascular pretrain samples found under {self.data_path}")

        ntest = min(self.ntest, max(1, len(sample_dirs) // 10))
        ntrain = min(self.ntrain, max(0, len(sample_dirs) - ntest))
        if ntrain <= 0:
            ntrain = len(sample_dirs)
            ntest = 0

        train_dirs = sample_dirs[:ntrain]
        test_dirs = sample_dirs[ntrain:ntrain + ntest]
        if not test_dirs:
            test_dirs = train_dirs[:min(1, len(train_dirs))]

        train_set = _VascularPretrainDataset(
            train_dirs, self.n_random_walks, random_walk=True,
            physics_proxy=self.physics_proxy,
            physics_proxy_mode=self.physics_proxy_mode,
            wall_mask_prob=self.wall_mask_prob,
            wall_mask_mode=self.wall_mask_mode)
        test_set = _VascularPretrainDataset(
            test_dirs, self.n_random_walks, random_walk=False,
            physics_proxy=self.physics_proxy,
            physics_proxy_mode=self.physics_proxy_mode,
            wall_mask_prob=0.0,
            wall_mask_mode=self.wall_mask_mode)

        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
            loader_kwargs["persistent_workers"] = True

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, **loader_kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, **loader_kwargs)

        print("VascularPretrain dataloading is over.")
        print(f"  complete samples={len(sample_dirs)} train={len(train_dirs)} test={len(test_dirs)}")
        print(f"  num_workers={self.num_workers} pin_memory={self.pin_memory}")
        if self.physics_proxy:
            proxy_dim = PHYSICS_PROXY_DIMS[self.physics_proxy_mode]
            print(f"  physics_proxy=true mode={self.physics_proxy_mode} targets={GEOMETRIC_TARGET_DIM}+{proxy_dim}")
        if self.wall_mask_prob > 0.0:
            print(f"  train wall_mask_prob={self.wall_mask_prob} mode={self.wall_mask_mode}")
        return train_loader, test_loader, [36864]
