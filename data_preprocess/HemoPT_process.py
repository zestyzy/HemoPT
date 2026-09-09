#!/usr/bin/env python3
"""
Convert intravascular CFD flow field data (txt) to HemoPT npy format.

Input:  txt files with columns x y z p u v w (no header, space/tab separated)
        These are flow-domain interior points (NOT surface geometry).
Output: x_i.npy (N,6) = xyz + dummy_0 + zeros (no real normals)
        y_i.npy (N,4) = p u v w
        cond_i.npy (3,) = [vessel_type, inlet_velocity, viscosity]

Key preprocessing choices:
  - No geometric surface normals (data is flow domain, not wall surface)
  - x format: (N, 6) = [x, y, z, 0, 0, 0] — coordinates + zero padding
  - y format: (N, 4) = [p, u, v, w] — pressure + 3D velocity
  - fun_dim should be set to account for coord dim + padding + dynamics field
"""

import argparse
import os
import glob
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_preprocess.wall_features import (
    WallFeatureLocator,
    apply_geometry_normalization,
    geometry_normalization_stats,
    str2bool,
)


def normalize_geometry(points: np.ndarray, target_length: float = 5.0) -> np.ndarray:
    """
    Normalize point cloud:
      - Scale so longest axis = target_length
      - Center at origin
    """
    scale, center = geometry_normalization_stats(points, target_length)
    return apply_geometry_normalization(points, scale, center)


def find_wall_surface(root, config_name, split, fpath, pattern):
    if not root:
        return None
    stem = os.path.splitext(os.path.basename(fpath))[0]
    basename = os.path.basename(fpath)
    rel = pattern.format(
        config=config_name,
        split=split,
        stem=stem,
        basename=basename,
    )
    candidates = [os.path.join(root, rel)]
    if "{ext}" not in pattern:
        base = os.path.join(root, rel)
        no_ext = os.path.splitext(base)[0]
        candidates.extend(no_ext + ext for ext in [".vtp", ".stl", ".vtu", ".vtk"])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def process_dataset(data_dir: str, outdir: str, config_name: str,
                    vessel_type: int, inlet_velocity: float, viscosity: float,
                    target_length: float, dtype: str, start_index: int = 0,
                    wall_features: bool = False, wall_surface_root: str = None,
                    wall_surface_pattern: str = "{config}/{split}/{stem}.vtp",
                    strict_wall: bool = True):
    """Process one vessel configuration (C1/ICA_norm/ICA_ste)."""
    np_dtype = getattr(np, dtype)
    os.makedirs(outdir, exist_ok=True)

    splits = {}
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            files = sorted(glob.glob(os.path.join(split_dir, "*.txt")))
            splits[split] = files

    if not splits:
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        splits["train"] = files
        splits["test"] = []

    global_index = start_index
    index_map = {"train": [], "test": []}

    for split, files in splits.items():
        for fpath in files:
            data = np.loadtxt(fpath)
            xyz = data[:, :3]   # (N, 3) spatial coordinates
            puvw = data[:, 3:]  # (N, 4) p, u, v, w

            # Normalize geometry coordinates
            scale, center = geometry_normalization_stats(xyz, target_length)
            xyz_norm = apply_geometry_normalization(xyz, scale, center)

            # x: (N, 7) = [x, y, z, dist_to_wall, dir_x, dir_y, dir_z] when
            # wall surfaces are available, otherwise the legacy zero padding.
            if wall_features:
                wall_path = find_wall_surface(
                    wall_surface_root, config_name, split, fpath, wall_surface_pattern)
                if wall_path is None:
                    message = (
                        f"Missing wall surface for {config_name}/{split}/"
                        f"{os.path.basename(fpath)} under {wall_surface_root}"
                    )
                    if strict_wall:
                        raise RuntimeError(message)
                    print(f"[Warn] {message}; using zero wall features")
                    geom_features = np.zeros((xyz.shape[0], 4), dtype=np.float64)
                else:
                    locator = WallFeatureLocator(wall_path)
                    geom_features, wall_stats = locator.features(xyz, scale=scale)
                    print(f"    wall={wall_path} dist_mean={wall_stats['wall_dist_mean']:.6g}")
            else:
                geom_features = np.zeros((xyz.shape[0], 3), dtype=np.float64)

            x_out = np.concatenate([xyz_norm, geom_features], axis=1).astype(np_dtype)

            # y: (N, 4) = [p, u, v, w]
            y_out = puvw.astype(np_dtype)

            # cond: [vessel_type, inlet_velocity, viscosity]
            cond = np.array([float(vessel_type), float(inlet_velocity), float(viscosity)], dtype=np_dtype)

            np.save(os.path.join(outdir, f"x_{global_index}.npy"), x_out)
            np.save(os.path.join(outdir, f"y_{global_index}.npy"), y_out)
            np.save(os.path.join(outdir, f"cond_{global_index}.npy"), cond)

            index_map[split].append(global_index)
            print(f"  [{split}] {os.path.basename(fpath)} -> index={global_index}  "
                  f"x={x_out.shape} y={y_out.shape} cond={cond.shape}")
            global_index += 1

    split_info = {
        "config": config_name,
        "train_indices": index_map["train"],
        "test_indices": index_map["test"],
    }
    np.save(os.path.join(outdir, f"split_info_{config_name}.npy"), split_info, allow_pickle=True)
    n_written = global_index - start_index
    print(f"\n[{config_name}] Total: {n_written} samples "
          f"(train={len(index_map['train'])}, test={len(index_map['test'])})")
    return index_map, global_index


def main():
    parser = argparse.ArgumentParser(description="Convert intravascular CFD txt data to HemoPT npy format")
    parser.add_argument("--data_root", type=str,
                        default="./external_data/HemoPT",
                        help="Root dir containing C1/, ICA_norm/, ICA_ste/ subdirectories")
    parser.add_argument("--outdir", type=str, default="./hemo_npys",
                        help="Output directory for npy files")
    parser.add_argument("--target_length", type=float, default=5.0,
                        help="Scale geometry so longest axis = target_length")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--wall_features", type=str2bool, default=False,
                        help="Use [dist_to_wall, dir_to_wall] feature channels from wall surfaces.")
    parser.add_argument("--wall_surface_root", type=str, default=None,
                        help="Root containing wall surfaces for HemoPT CFD samples.")
    parser.add_argument("--wall_surface_pattern", type=str, default="{config}/{split}/{stem}.vtp",
                        help="Relative format under --wall_surface_root. Placeholders: config, split, stem, basename.")
    parser.add_argument("--strict_wall", type=str2bool, default=True,
                        help="Fail if --wall_features true and a sample wall surface is missing.")

    args = parser.parse_args()
    if args.wall_features and not args.wall_surface_root:
        raise ValueError("--wall_surface_root is required when --wall_features true")

    # Vessel configurations: (subdir, vessel_type_id, inlet_velocity_m_per_s, viscosity_Pa_s)
    configs = {
        "C1":       ("C1",       0, 0.3, 0.0035),
        "ICA_norm": ("ICA_norm", 1, 0.3, 0.0035),
        "ICA_ste":  ("ICA_ste",  2, 0.3, 0.0035),
    }

    all_train = []
    all_test = []
    global_index = 0

    for config_name, (subdir, vtype, vel, visc) in configs.items():
        data_dir = os.path.join(args.data_root, subdir)
        if not os.path.isdir(data_dir):
            print(f"[Skip] {data_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {config_name} from {data_dir}")
        print(f"{'='*60}")

        index_map, global_index = process_dataset(
            data_dir=data_dir,
            outdir=args.outdir,
            config_name=config_name,
            vessel_type=vtype,
            inlet_velocity=vel,
            viscosity=visc,
            target_length=args.target_length,
            dtype=args.dtype,
            start_index=global_index,
            wall_features=args.wall_features,
            wall_surface_root=args.wall_surface_root,
            wall_surface_pattern=args.wall_surface_pattern,
            strict_wall=args.strict_wall,
        )

        all_train.extend(index_map["train"])
        all_test.extend(index_map["test"])

    global_info = {
        "train_indices": all_train,
        "test_indices": all_test,
    }
    np.save(os.path.join(args.outdir, "global_split_info.npy"), global_info, allow_pickle=True)
    print(f"\n{'='*60}")
    print(f"Done! Total: {global_index} samples (train={len(all_train)}, test={len(all_test)})")
    print(f"Output: {args.outdir}")
    print(f"\nFor HemoPT fine-tuning, set:")
    print(f"  --fun_dim 8   (4 zero feature channels + 4 hemo prompt channels)")
    print(f"  --out_dim 4   (p, u, v, w)")
    print(f"  --space_dim 3")
    print(f"  --loader HemoPT")
    print(f"  --dynamics hemo")
    print(f"  Loader convention: pos=x[:, :, :3], fx=x[:, :, 3:] padded to 4 channels, then concat hemo prompt")


if __name__ == "__main__":
    main()
