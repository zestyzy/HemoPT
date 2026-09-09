#!/usr/bin/env python3
"""Create patient-level train/val/test splits for VMR cases with CFD results.

The split is based on top-level VMR case IDs, not individual STL files, to avoid
leakage between model/mesh variants from the same patient/case.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def _case_inner_dir(case_dir: Path) -> Path:
    nested = case_dir / case_dir.name
    return nested if nested.is_dir() else case_dir


def _territory(case_id: str) -> str:
    parts = case_id.split("_")
    return "_".join(parts[2:]) if len(parts) >= 3 else "UNKNOWN"


def _discover_cases(vmr_root: Path):
    cases = []
    for case_dir in sorted(vmr_root.iterdir()):
        if not case_dir.is_dir():
            continue

        inner = _case_inner_dir(case_dir)
        sim_root = inner / "Simulations"
        if not sim_root.is_dir():
            continue

        simulations = []
        for sim_dir in sorted(sim_root.iterdir()):
            if not sim_dir.is_dir():
                continue

            mesh_complete = sim_dir / "mesh-complete"
            mesh_vtu = mesh_complete / "mesh-complete.mesh.vtu"
            exterior_vtp = mesh_complete / "mesh-complete.exterior.vtp"
            walls_vtp = mesh_complete / "walls_combined.vtp"
            restart_files = sorted(p for p in sim_dir.iterdir() if p.name.startswith("restart"))

            # Treat restart files as the marker that solver results are present.
            if not restart_files:
                continue
            if not (mesh_vtu.exists() or exterior_vtp.exists()):
                continue

            simulations.append({
                "sim_name": sim_dir.name,
                "sim_dir": str(sim_dir),
                "mesh_vtu": str(mesh_vtu) if mesh_vtu.exists() else "",
                "exterior_vtp": str(exterior_vtp) if exterior_vtp.exists() else "",
                "walls_vtp": str(walls_vtp) if walls_vtp.exists() else "",
                "restart_count": len(restart_files),
                "solver_inp": str(sim_dir / "solver.inp") if (sim_dir / "solver.inp").exists() else "",
                "inflow_flow": str(sim_dir / "inflow.flow") if (sim_dir / "inflow.flow").exists() else "",
                "rcrt_dat": str(sim_dir / "rcrt.dat") if (sim_dir / "rcrt.dat").exists() else "",
            })

        if simulations:
            case_id = case_dir.name
            cases.append({
                "case_id": case_id,
                "territory": _territory(case_id),
                "case_dir": str(case_dir),
                "inner_dir": str(inner),
                "simulations": simulations,
            })
    return cases


def _split_cases(cases, seed: int):
    rng = random.Random(seed)
    by_territory = defaultdict(list)
    for case in cases:
        by_territory[case["territory"]].append(case)

    split = {"train": [], "val": [], "test": []}
    for territory, group in sorted(by_territory.items()):
        group = list(group)
        rng.shuffle(group)
        n = len(group)

        if n == 1:
            n_test = 0
            n_val = 0
        elif n == 2:
            n_test = 1
            n_val = 0
        elif n == 3:
            n_test = 1
            n_val = 0
        else:
            n_test = max(1, round(0.15 * n))
            n_val = max(1, round(0.15 * n))

        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)

        test_cases = group[:n_test]
        val_cases = group[n_test:n_test + n_val]
        train_cases = group[n_test + n_val:]

        split["train"].extend(train_cases)
        split["val"].extend(val_cases)
        split["test"].extend(test_cases)

    for key in split:
        split[key] = sorted(split[key], key=lambda x: x["case_id"])
    return split


def _write_csv(path: Path, split):
    rows = []
    for split_name, cases in split.items():
        for case in cases:
            for sim in case["simulations"]:
                rows.append({
                    "split": split_name,
                    "case_id": case["case_id"],
                    "territory": case["territory"],
                    "case_dir": case["case_dir"],
                    "inner_dir": case["inner_dir"],
                    **sim,
                })

    fieldnames = [
        "split", "case_id", "territory", "case_dir", "inner_dir",
        "sim_name", "sim_dir", "mesh_vtu", "exterior_vtp", "walls_vtp",
        "restart_count", "solver_inp", "inflow_flow", "rcrt_dat",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create VMR CFD train/val/test split manifest")
    parser.add_argument("--vmr_root", type=str,
                        default="./HemoData/0511/VMR")
    parser.add_argument("--out_dir", type=str,
                        default="./HemoData/VMR_CFD_Splits")
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    vmr_root = Path(args.vmr_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _discover_cases(vmr_root)
    split = _split_cases(cases, seed=args.seed)

    all_reserved = sorted(case["case_id"] for part in split.values() for case in part)
    manifest = {
        "vmr_root": str(vmr_root),
        "seed": args.seed,
        "definition": "VMR cases with restart.* files and mesh-complete mesh/exterior files are reserved for CFD fine-tuning/evaluation.",
        "reserved_case_ids": all_reserved,
        "counts": {
            "total_cases": len(all_reserved),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
            "territory": dict(Counter(case["territory"] for case in cases)),
        },
        "splits": split,
    }

    json_path = out_dir / "vmr_cfd_split.json"
    csv_path = out_dir / "vmr_cfd_split.csv"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _write_csv(csv_path, split)

    print(f"Discovered reserved VMR CFD cases: {len(all_reserved)}")
    print(f"  train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
