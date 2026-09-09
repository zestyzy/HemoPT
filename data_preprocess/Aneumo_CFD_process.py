#!/usr/bin/env python3
"""Convert aneumo CFD arrays to HemoPT/VMR-style supervised npy samples."""
import argparse
import csv
import json
import os
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


ANEUMO_COND_SCHEMA = [
    "dataset_id",
    "m_value",
    "m_value_norm",
    "inlet_speed_mean",
    "inlet_speed_max",
    "inlet_flow_proxy",
    "case_id_norm",
    "n_points_log",
    "raw_length_log",
    "reserved",
    "reserved",
    "domain_flag",
]


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _m_value(m_dir):
    text = str(m_dir)
    if text.startswith("m="):
        text = text[2:]
    return float(text)


def _split_rows(rows):
    return [row for row in rows if row["split"] in {"train", "val", "test"}]


def _solution_stats(y):
    p = y[:, 0]
    v = y[:, 1:4]
    speed = np.linalg.norm(v, axis=1)
    return {
        "y_norm": float(np.linalg.norm(y.reshape(-1))),
        "p_norm": float(np.linalg.norm(p)),
        "v_norm": float(np.linalg.norm(v.reshape(-1))),
        "max_abs": float(np.max(np.abs(y))) if y.size else 0.0,
        "p_std": float(np.std(p)) if len(p) else 0.0,
        "v_std": float(np.std(v)) if len(v) else 0.0,
        "vmax": float(np.max(speed)) if len(speed) else 0.0,
        "speed_mean": float(np.mean(speed)) if len(speed) else 0.0,
    }


def _load_internal(path):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise RuntimeError(f"Expected internal array shape (N, >=7), got {arr.shape}: {path}")
    arr = np.asarray(arr[:, :7], dtype=np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"Non-finite values in {path}")
    return arr[:, :3], arr[:, 3:7]


def _load_boundary_speed(path):
    if not path or not Path(path).exists():
        return 0.0, 0.0, 0.0
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 7 or arr.shape[0] == 0:
        return 0.0, 0.0, 0.0
    vel = np.asarray(arr[:, 4:7], dtype=np.float64)
    speed = np.linalg.norm(vel, axis=1)
    flow_proxy = float(speed.mean() * max(arr.shape[0], 1))
    return float(speed.mean()), float(speed.max()), flow_proxy


def _build_condition(row, args, n_points_raw, scale):
    m_value = _m_value(row["m_dir"])
    inlet_mean, inlet_max, inlet_flow_proxy = _load_boundary_speed(row.get("inlet_npy", ""))
    try:
        case_id_norm = float(row["case_id"]) / 120.0
    except ValueError:
        case_id_norm = 0.0
    raw_length = float(args.target_length) / max(float(scale), 1e-12)
    values = {
        "dataset_id": 4.0,
        "m_value": m_value,
        "m_value_norm": m_value / 0.004,
        "inlet_speed_mean": inlet_mean,
        "inlet_speed_max": inlet_max,
        "inlet_flow_proxy": np.log1p(inlet_flow_proxy) / 12.0,
        "case_id_norm": case_id_norm,
        "n_points_log": np.log1p(n_points_raw) / 12.0,
        "raw_length_log": np.log1p(raw_length) / 10.0,
        "reserved": 0.0,
        "domain_flag": 1.0,
    }
    cond = np.array([
        values["dataset_id"],
        values["m_value"],
        values["m_value_norm"],
        values["inlet_speed_mean"],
        values["inlet_speed_max"],
        values["inlet_flow_proxy"],
        values["case_id_norm"],
        values["n_points_log"],
        values["raw_length_log"],
        0.0,
        0.0,
        values["domain_flag"],
    ], dtype=np.float64)
    return cond, values


def _passes_solution_filter(stats, args):
    return (
        stats["y_norm"] >= args.min_solution_norm
        and stats["v_norm"] >= args.min_velocity_norm
        and stats["max_abs"] >= args.min_max_abs
    )


def main():
    parser = argparse.ArgumentParser(description="Process aneumo CFD arrays to supervised npy samples.")
    parser.add_argument("--split_csv", type=str,
                        default="HemoData/Aneumo_CFD_Splits/aneumo_cfd_split.csv")
    parser.add_argument("--outdir", type=str, default="aneumo_cfd_npys_wall_aligned")
    parser.add_argument("--n_points", type=int, default=4096)
    parser.add_argument("--target_length", type=float, default=5.0)
    parser.add_argument("--normalization_scope", type=str, default="mesh",
                        choices=["mesh", "sample"])
    parser.add_argument("--wall_features", type=str2bool, default=True)
    parser.add_argument("--wall_backend", type=str, default="auto", choices=["auto", "fcpw", "vtk"])
    parser.add_argument("--strict_wall", type=str2bool, default=True)
    parser.add_argument("--min_solution_norm", type=float, default=0.0)
    parser.add_argument("--min_velocity_norm", type=float, default=0.0)
    parser.add_argument("--min_max_abs", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np_dtype = getattr(np, args.dtype)

    rows = _split_rows(_read_rows(args.split_csv))
    split_indices = {"train": [], "val": [], "test": []}
    metadata = []
    index = 0
    for row in rows:
        try:
            xyz, y = _load_internal(row["internal_npy"])
            stats = _solution_stats(y)
            if not _passes_solution_filter(stats, args):
                print(f"[skip] case={row['case_id']} split={row['split']} "
                      f"low-signal y_norm={stats['y_norm']:.6g} v_norm={stats['v_norm']:.6g}")
                continue

            replace = xyz.shape[0] < args.n_points
            sample_idx = rng.choice(xyz.shape[0], size=args.n_points, replace=replace)
            norm_points = xyz if args.normalization_scope == "mesh" else xyz[sample_idx]
            scale, center = geometry_normalization_stats(norm_points, args.target_length)
            xyz_sample = apply_geometry_normalization(xyz[sample_idx], scale, center)
            y_sample = y[sample_idx]

            feature_source = "zeros"
            wall_stats = {}
            wall_path = row.get("wall_vtp", "")
            if args.wall_features:
                if not wall_path or not Path(wall_path).exists():
                    if args.strict_wall:
                        raise RuntimeError(f"missing wall_vtp: {wall_path}")
                    geom_features = np.zeros((args.n_points, 4), dtype=np.float64)
                    feature_source = "zeros_missing_wall"
                else:
                    locator = WallFeatureLocator(wall_path, backend=args.wall_backend)
                    geom_features, wall_stats = locator.features(xyz[sample_idx], scale=scale)
                    feature_source = "wall_vtp"
            else:
                geom_features = np.zeros((args.n_points, 4), dtype=np.float64)

            x_out = np.concatenate([xyz_sample, geom_features], axis=1).astype(np_dtype)
            y_out = y_sample.astype(np_dtype)
            cond, cond_meta = _build_condition(row, args, xyz.shape[0], scale)
            cond = cond.astype(np_dtype)

            np.save(outdir / f"x_{index}.npy", x_out)
            np.save(outdir / f"y_{index}.npy", y_out)
            np.save(outdir / f"cond_{index}.npy", cond)
            split_indices[row["split"]].append(index)

            item = {
                "index": index,
                "split": row["split"],
                "case_id": row["case_id"],
                "m_dir": row["m_dir"],
                "internal_npy": row["internal_npy"],
                "wall_path": wall_path,
                "n_raw_points": int(xyz.shape[0]),
                "sampled_points": int(args.n_points),
                "feature_source": feature_source,
                "geometry_scale": float(scale),
                "normalization_scope": args.normalization_scope,
                "cond_schema": "|".join(ANEUMO_COND_SCHEMA),
            }
            item.update(stats)
            item.update(wall_stats)
            item.update({f"cond_{i}_{name}": float(cond[i])
                         for i, name in enumerate(ANEUMO_COND_SCHEMA)})
            item.update({f"meta_{k}": float(v) for k, v in cond_meta.items()
                         if isinstance(v, (int, float, np.floating))})
            metadata.append(item)
            print(f"[ok] {index} split={row['split']} case={row['case_id']} "
                  f"m={row['m_dir']} y_norm={stats['y_norm']:.6g} "
                  f"vmax={stats['vmax']:.6g} features={feature_source} x={x_out.shape} y={y_out.shape}")
            index += 1
        except Exception as exc:
            print(f"[skip] case={row.get('case_id')} split={row.get('split')}: {exc}")

    info = {
        "train_indices": split_indices["train"],
        "val_indices": split_indices["val"],
        "test_indices": split_indices["test"] or split_indices["val"],
        "source_split_csv": os.path.abspath(args.split_csv),
        "filters": {
            "normalization_scope": args.normalization_scope,
            "wall_features": args.wall_features,
            "wall_backend": args.wall_backend,
            "strict_wall": args.strict_wall,
            "cond_schema": ANEUMO_COND_SCHEMA,
            "target": "[p,u,v,w]",
        },
        "metadata": metadata,
    }
    np.save(outdir / "global_split_info.npy", info, allow_pickle=True)
    if metadata:
        with open(outdir / "metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
            writer.writeheader()
            writer.writerows(metadata)
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. total={index} train={len(split_indices['train'])} "
          f"val={len(split_indices['val'])} test={len(split_indices['test'])} out={outdir}")


if __name__ == "__main__":
    main()
