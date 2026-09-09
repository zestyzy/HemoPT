#!/usr/bin/env python3
"""Convert VMR SimVascular restart results to HemoPT npy format."""

import argparse
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support as nps

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_preprocess.wall_features import (
    WallFeatureLocator,
    apply_geometry_normalization,
    geometry_normalization_stats,
    str2bool,
)


def normalize_geometry(points, target_length):
    scale, center = geometry_normalization_stats(points, target_length)
    return apply_geometry_normalization(points, scale, center)


def read_vtu_points(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    points = nps.vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    gids = grid.GetPointData().GetArray("GlobalNodeID")
    if gids is not None:
        gids = nps.vtk_to_numpy(gids)
        order = np.argsort(gids)
        points = points[order]
    return points


def parse_restart(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    idx = data.find(b"solution")
    if idx < 0:
        raise RuntimeError(f"No solution block in {path}")
    line_end = data.find(b"\n", idx)
    line = data[idx:line_end]
    match = re.search(rb"<\s*(\d+)\s*>\s*(\d+)\s+(\d+)\s+(\d+)", line)
    if not match:
        raise RuntimeError(f"Cannot parse solution header in {path}: {line!r}")
    n_points = int(match.group(2))
    n_vars = int(match.group(3))
    step = int(match.group(4))
    if n_vars < 4:
        raise RuntimeError(f"Expected at least 4 variables in {path}, got {n_vars}")
    flat = np.frombuffer(data, dtype="<f8", count=n_points * n_vars, offset=line_end + 1)
    solution = flat.reshape(n_vars, n_points).T
    return solution[:, :4].astype(np.float64), step


def choose_restart(sim_dir):
    restarts = sorted(Path(sim_dir).glob("restart*"))
    parsed = []
    for path in restarts:
        m = re.search(r"restart\.(\d+)\.", path.name)
        step = int(m.group(1)) if m else -1
        parsed.append((step, path))
    if not parsed:
        return None, -1
    return max(parsed, key=lambda x: x[0])[1], max(parsed, key=lambda x: x[0])[0]


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def solution_stats(puvw):
    p = puvw[:, 0]
    v = puvw[:, 1:4]
    return {
        "y_norm": float(np.linalg.norm(puvw.reshape(-1))),
        "p_norm": float(np.linalg.norm(p)),
        "v_norm": float(np.linalg.norm(v.reshape(-1))),
        "max_abs": float(np.max(np.abs(puvw))),
        "p_std": float(np.std(p)),
        "v_std": float(np.std(v)),
        "vmax": float(np.max(np.linalg.norm(v, axis=1))),
    }


VMR_TERRITORY_FAMILY_TO_ID = {
    "ABAO": 0,
    "AO": 1,
    "CERE": 2,
    "CORO": 3,
    "PULM": 4,
    "PULMFON": 5,
    "PULMGLN": 6,
    "UNKNOWN": 7,
}

VMR_COND_SCHEMA = [
    "territory_family_norm",
    "flow_mean_log",
    "flow_rms_log",
    "flow_peak_log",
    "flow_pulsatility",
    "viscosity_rel",
    "density_rel",
    "rcr_count_norm",
    "rcr_resistance_log",
    "rcr_capacitance_log",
    "raw_length_log",
    "restart_step_log",
]

LEGACY_COND_SCHEMA = ["vessel_type", "inlet_velocity", "viscosity"]


def territory_family(territory):
    if not territory:
        return "UNKNOWN"
    family = territory.split("_")[0]
    return family if family in VMR_TERRITORY_FAMILY_TO_ID else "UNKNOWN"


def parse_solver_inp(path):
    values = {
        "density": 1.06,
        "viscosity": 0.04,
        "time_step_size": 0.0,
        "n_timesteps": 0.0,
    }
    if not path or not Path(path).exists():
        return values
    key_map = {
        "density": "density",
        "viscosity": "viscosity",
        "time step size": "time_step_size",
        "number of timesteps": "n_timesteps",
    }
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key not in key_map:
                continue
            try:
                values[key_map[key]] = float(value.strip().split()[0])
            except (ValueError, IndexError):
                continue
    return values


def summarize_inflow(path):
    summary = {
        "flow_mean_abs": 0.0,
        "flow_rms_abs": 0.0,
        "flow_peak_abs": 0.0,
        "flow_std_abs": 0.0,
        "flow_pulsatility": 0.0,
        "flow_n": 0,
    }
    if not path or not Path(path).exists():
        return summary
    try:
        arr = np.loadtxt(path)
    except Exception:
        return summary
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2 or arr.shape[0] == 0:
        return summary
    flow = arr[:, 1]
    abs_flow = np.abs(flow)
    mean_abs = float(abs_flow.mean())
    rms_abs = float(np.sqrt(np.mean(flow ** 2)))
    peak_abs = float(abs_flow.max())
    std_abs = float(flow.std())
    summary.update({
        "flow_mean_abs": mean_abs,
        "flow_rms_abs": rms_abs,
        "flow_peak_abs": peak_abs,
        "flow_std_abs": std_abs,
        "flow_pulsatility": float(std_abs / (mean_abs + 1e-8)),
        "flow_n": int(arr.shape[0]),
    })
    return summary


def _float_tokens(line):
    values = []
    for token in line.split():
        try:
            values.append(float(token))
        except ValueError:
            return []
    return values


def summarize_rcrt(path):
    summary = {
        "rcr_count": 0,
        "rcr_resistance_mean": 0.0,
        "rcr_resistance_std": 0.0,
        "rcr_capacitance_mean": 0.0,
    }
    if not path or not Path(path).exists():
        return summary
    with open(path) as f:
        rows = [_float_tokens(line) for line in f if line.strip()]
    rows = [row for row in rows if row]
    if not rows:
        return summary

    blocks = []
    i = 1 if len(rows[0]) == 1 and rows[0][0] >= 1 else 0
    while i < len(rows):
        marker = rows[i]
        if len(marker) != 1:
            i += 1
            continue
        n_curve = int(round(marker[0]))
        has_block = (
            n_curve >= 1
            and i + 3 + n_curve < len(rows)
            and len(rows[i + 1]) == len(rows[i + 2]) == len(rows[i + 3]) == 1
            and all(len(rows[i + 4 + j]) >= 2 for j in range(n_curve))
        )
        if not has_block:
            i += 1
            continue
        rp = float(rows[i + 1][0])
        cap = float(rows[i + 2][0])
        rd = float(rows[i + 3][0])
        if rp > 0 and rd > 0 and cap > 0:
            blocks.append((rp, cap, rd))
        i += 4 + n_curve

    if not blocks:
        return summary
    resistances = np.asarray([rp + rd for rp, _, rd in blocks], dtype=np.float64)
    capacitances = np.asarray([cap for _, cap, _ in blocks], dtype=np.float64)
    summary.update({
        "rcr_count": int(len(blocks)),
        "rcr_resistance_mean": float(resistances.mean()),
        "rcr_resistance_std": float(resistances.std()),
        "rcr_capacitance_mean": float(capacitances.mean()),
    })
    return summary


def build_condition(row, args, scale, restart_step):
    if args.cond_mode == "legacy":
        cond = np.array([3.0, 0.3, 0.0035], dtype=np.float64)
        return cond, dict(zip(LEGACY_COND_SCHEMA, cond))

    solver = parse_solver_inp(row.get("solver_inp", ""))
    inflow = summarize_inflow(row.get("inflow_flow", ""))
    rcrt = summarize_rcrt(row.get("rcrt_dat", ""))

    family = territory_family(row.get("territory", ""))
    family_id = VMR_TERRITORY_FAMILY_TO_ID[family]
    family_norm = family_id / max(VMR_TERRITORY_FAMILY_TO_ID.values())
    raw_length = float(args.target_length) / max(float(scale), 1e-12)
    pulsatility = min(inflow["flow_pulsatility"], 5.0) / 5.0
    cap = max(rcrt["rcr_capacitance_mean"], 1e-12)

    values = {
        "territory_family_norm": family_norm,
        "flow_mean_log": np.log1p(inflow["flow_mean_abs"]) / 6.0,
        "flow_rms_log": np.log1p(inflow["flow_rms_abs"]) / 6.0,
        "flow_peak_log": np.log1p(inflow["flow_peak_abs"]) / 7.0,
        "flow_pulsatility": pulsatility,
        "viscosity_rel": solver["viscosity"] / 0.04,
        "density_rel": solver["density"] / 1.06,
        "rcr_count_norm": rcrt["rcr_count"] / 10.0,
        "rcr_resistance_log": np.log1p(rcrt["rcr_resistance_mean"]) / 12.0,
        "rcr_capacitance_log": -np.log10(cap) / 12.0,
        "raw_length_log": np.log1p(raw_length) / 10.0,
        "restart_step_log": np.log1p(max(float(restart_step), 0.0)) / 10.0,
    }
    cond = np.array([values[name] for name in VMR_COND_SCHEMA], dtype=np.float64)
    values.update({
        "territory_family": family,
        "density": solver["density"],
        "viscosity": solver["viscosity"],
        **inflow,
        **rcrt,
    })
    return cond, values


def passes_solution_filter(stats, args):
    return (
        stats["y_norm"] >= args.min_solution_norm
        and stats["p_norm"] >= args.min_pressure_norm
        and stats["p_std"] >= args.min_pressure_std
        and stats["max_abs"] >= args.min_max_abs
    )


def main():
    parser = argparse.ArgumentParser(description="Process VMR CFD restart files to npy samples.")
    parser.add_argument("--split_csv", type=str,
                        default="HemoData/VMR_CFD_Splits/vmr_cfd_split.csv")
    parser.add_argument("--outdir", type=str, default="vmr_cfd_npys")
    parser.add_argument("--n_points", type=int, default=4096)
    parser.add_argument("--target_length", type=float, default=5.0)
    parser.add_argument("--normalization_scope", type=str, default="mesh",
                        choices=["mesh", "sample"],
                        help="Use the full mesh or only sampled points for geometry normalization.")
    parser.add_argument("--min_restart_step", type=int, default=1)
    parser.add_argument("--min_solution_norm", type=float, default=0.0,
                        help="Skip samples whose full [p,u,v,w] norm is below this value.")
    parser.add_argument("--min_pressure_norm", type=float, default=0.0,
                        help="Skip samples whose pressure norm is below this value.")
    parser.add_argument("--min_pressure_std", type=float, default=0.0,
                        help="Skip samples whose pressure standard deviation is below this value.")
    parser.add_argument("--min_max_abs", type=float, default=0.0,
                        help="Skip samples whose max absolute solution value is below this value.")
    parser.add_argument("--wall_features", type=str2bool, default=False,
                        help="Use [dist_to_wall, dir_to_wall] instead of zero geometry feature channels.")
    parser.add_argument("--wall_source", type=str, default="walls",
                        choices=["walls", "exterior"],
                        help="VMR split column used for wall feature computation.")
    parser.add_argument("--wall_backend", type=str, default="auto",
                        choices=["auto", "fcpw", "vtk"],
                        help="Closest-wall query backend.")
    parser.add_argument("--strict_wall", type=str2bool, default=True,
                        help="Skip samples without a usable wall surface when --wall_features true.")
    parser.add_argument("--cond_mode", type=str, default="vmr", choices=["vmr", "legacy"],
                        help="VMR uses case-level inflow/RCR/solver metadata; legacy writes [3,0.3,0.0035].")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np_dtype = getattr(np, args.dtype)

    split_indices = {"train": [], "val": [], "test": []}
    metadata = []
    rows = read_rows(args.split_csv)
    index = 0
    for row in rows:
        sim_dir = Path(row["sim_dir"])
        restart_path, restart_step = choose_restart(sim_dir)
        if restart_path is None or restart_step < args.min_restart_step:
            print(f"[skip] {row['case_id']} {row['sim_name']} restart_step={restart_step}")
            continue
        mesh_vtu = Path(row["mesh_vtu"])
        if not mesh_vtu.exists():
            print(f"[skip] missing mesh_vtu: {mesh_vtu}")
            continue

        try:
            xyz = read_vtu_points(mesh_vtu)
            puvw, step = parse_restart(restart_path)
            if xyz.shape[0] != puvw.shape[0]:
                raise RuntimeError(f"points/solution mismatch: {xyz.shape[0]} vs {puvw.shape[0]}")
            if not np.isfinite(puvw).all() or np.max(np.abs(puvw)) < 1e-12:
                raise RuntimeError("non-finite or near-zero solution")
            stats = solution_stats(puvw)
            if not passes_solution_filter(stats, args):
                print(f"[skip] {row['case_id']} {row['sim_name']} low-signal "
                      f"step={step} y_norm={stats['y_norm']:.6g} "
                      f"p_norm={stats['p_norm']:.6g} p_std={stats['p_std']:.6g} "
                      f"max_abs={stats['max_abs']:.6g}")
                continue
            replace = xyz.shape[0] < args.n_points
            sample_idx = rng.choice(xyz.shape[0], size=args.n_points, replace=replace)
            norm_points = xyz if args.normalization_scope == "mesh" else xyz[sample_idx]
            scale, center = geometry_normalization_stats(norm_points, args.target_length)
            xyz_sample = apply_geometry_normalization(xyz[sample_idx], scale, center)
            y_sample = puvw[sample_idx]
            feature_source = "zeros"
            wall_path = ""
            wall_stats = {}
            if args.wall_features:
                wall_col = "walls_vtp" if args.wall_source == "walls" else "exterior_vtp"
                wall_path = row.get(wall_col, "")
                if not wall_path or not Path(wall_path).exists():
                    if args.strict_wall:
                        raise RuntimeError(f"missing {wall_col}: {wall_path}")
                    geom_features = np.zeros((args.n_points, 4), dtype=np.float64)
                    feature_source = "zeros_missing_wall"
                else:
                    locator = WallFeatureLocator(wall_path, backend=args.wall_backend)
                    geom_features, wall_stats = locator.features(xyz[sample_idx], scale=scale)
                    feature_source = args.wall_source
            else:
                geom_features = np.zeros((args.n_points, 4), dtype=np.float64)
            x_out = np.concatenate([
                xyz_sample,
                geom_features,
            ], axis=1).astype(np_dtype)
            y_out = y_sample.astype(np_dtype)
            cond, cond_meta = build_condition(row, args, scale, step)
            cond = cond.astype(np_dtype)

            np.save(outdir / f"x_{index}.npy", x_out)
            np.save(outdir / f"y_{index}.npy", y_out)
            np.save(outdir / f"cond_{index}.npy", cond)
            split_indices[row["split"]].append(index)
            item = {
                "index": index,
                "split": row["split"],
                "case_id": row["case_id"],
                "territory": row.get("territory", ""),
                "sim_name": row["sim_name"],
                "sim_dir": str(sim_dir),
                "mesh_vtu": str(mesh_vtu),
                "restart_path": str(restart_path),
                "restart_step": int(step),
                "n_mesh_points": int(xyz.shape[0]),
                "sampled_points": int(args.n_points),
                "feature_source": feature_source,
                "wall_path": wall_path,
                "geometry_scale": float(scale),
                "normalization_scope": args.normalization_scope,
                "cond_mode": args.cond_mode,
            }
            item.update(stats)
            item.update(wall_stats)
            item.update({f"cond_{name}": float(cond[i]) for i, name in enumerate(
                VMR_COND_SCHEMA if args.cond_mode == "vmr" else LEGACY_COND_SCHEMA)})
            item.update({f"meta_{k}": v for k, v in cond_meta.items()
                         if isinstance(v, (int, float, str))})
            metadata.append(item)
            print(f"[ok] {index} split={row['split']} case={row['case_id']} "
                  f"sim={row['sim_name']} step={step} y_norm={stats['y_norm']:.6g} "
                  f"p_std={stats['p_std']:.6g} features={feature_source} "
                  f"x={x_out.shape} y={y_out.shape}")
            index += 1
        except Exception as exc:
            print(f"[skip] {row['case_id']} {row['sim_name']}: {exc}")

    train = split_indices["train"]
    test = split_indices["test"] or split_indices["val"]
    info = {
        "train_indices": train,
        "test_indices": test,
        "val_indices": split_indices["val"],
        "source_split_csv": os.path.abspath(args.split_csv),
        "filters": {
            "min_restart_step": args.min_restart_step,
            "normalization_scope": args.normalization_scope,
            "min_solution_norm": args.min_solution_norm,
            "min_pressure_norm": args.min_pressure_norm,
            "min_pressure_std": args.min_pressure_std,
            "min_max_abs": args.min_max_abs,
            "wall_features": args.wall_features,
            "wall_source": args.wall_source,
            "wall_backend": args.wall_backend,
            "strict_wall": args.strict_wall,
            "cond_mode": args.cond_mode,
            "cond_schema": VMR_COND_SCHEMA if args.cond_mode == "vmr" else LEGACY_COND_SCHEMA,
        },
        "metadata": metadata,
    }
    np.save(outdir / "global_split_info.npy", info, allow_pickle=True)
    metadata_path = outdir / "metadata.csv"
    if metadata:
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
            writer.writeheader()
            writer.writerows(metadata)
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. total={index} train={len(train)} val={len(split_indices['val'])} test={len(test)} out={outdir}")


if __name__ == "__main__":
    main()
