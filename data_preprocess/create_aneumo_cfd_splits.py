#!/usr/bin/env python3
"""Create case-level aneumo CFD splits.

The split reserves CFD-labelled aneumo cases for supervised fine-tuning and
keeps the remaining STL-only cases available for geometric pre-training.
"""
import argparse
import csv
import json
import random
from pathlib import Path


def _case_sort_key(path):
    name = path.name
    return int(name) if name.isdigit() else name


def _case_paths(root: Path, case_id: str, m_dir: str):
    case_dir = root / case_id
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "m_dir": m_dir,
        "internal_npy": str(case_dir / "npy" / m_dir / f"array_internal_{case_id}.npy"),
        "inlet_npy": str(case_dir / "npy" / m_dir / f"array_inlet_{case_id}.npy"),
        "outlet_npy": str(case_dir / "npy" / m_dir / f"array_outlet_{case_id}.npy"),
        "wall_npy": str(case_dir / "npy" / m_dir / f"array_wall_{case_id}.npy"),
        "internal_vtu": str(case_dir / "VTK" / m_dir / "internal.vtu"),
        "wall_vtp": str(case_dir / "VTK" / m_dir / "wall.vtp"),
        "stl_path": str(case_dir / "Stl" / f"{case_id}.stl"),
    }


def discover_cases(root: Path, m_dir: str):
    cases = []
    for case_dir in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
                           key=_case_sort_key):
        paths = _case_paths(root, case_dir.name, m_dir)
        required = ["internal_npy", "inlet_npy", "outlet_npy", "wall_vtp", "stl_path"]
        if all(Path(paths[key]).exists() for key in required):
            cases.append(paths)
    return cases


def split_cases(cases, n_reserved, n_train, n_val, n_test, seed):
    if n_train + n_val + n_test != n_reserved:
        raise ValueError("n_train + n_val + n_test must equal n_reserved")
    if len(cases) < n_reserved:
        raise ValueError(f"Only {len(cases)} usable cases found, cannot reserve {n_reserved}")

    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    reserved = sorted(shuffled[:n_reserved], key=lambda row: int(row["case_id"]))
    pretrain_stl = sorted(shuffled[n_reserved:], key=lambda row: int(row["case_id"]))

    reserved_for_split = list(reserved)
    rng.shuffle(reserved_for_split)
    train = sorted(reserved_for_split[:n_train], key=lambda row: int(row["case_id"]))
    val = sorted(reserved_for_split[n_train:n_train + n_val], key=lambda row: int(row["case_id"]))
    test = sorted(reserved_for_split[n_train + n_val:n_train + n_val + n_test],
                  key=lambda row: int(row["case_id"]))
    return {"train": train, "val": val, "test": test, "pretrain_stl_only": pretrain_stl}


def write_csv(path: Path, split):
    fieldnames = [
        "split", "case_id", "case_dir", "m_dir", "internal_npy", "inlet_npy",
        "outlet_npy", "wall_npy", "internal_vtu", "wall_vtp", "stl_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, rows in split.items():
            for row in rows:
                out = dict(row)
                out["split"] = split_name
                writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Create aneumo CFD train/val/test split manifest")
    parser.add_argument("--aneumo_root", type=str,
                        default="./totaldata/unpacked/dataset_2/aneumo")
    parser.add_argument("--out_dir", type=str,
                        default="./HemoData/Aneumo_CFD_Splits")
    parser.add_argument("--m_dir", type=str, default="m=0.004",
                        help="aneumo CFD setting used for downstream fine-tuning; user requested t=0.04")
    parser.add_argument("--n_reserved", type=int, default=100,
                        help="CFD-labelled cases reserved for supervised downstream splits")
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_val", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    root = Path(args.aneumo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(root, args.m_dir)
    split = split_cases(cases, args.n_reserved, args.n_train, args.n_val, args.n_test, args.seed)
    reserved = sorted(
        [row["case_id"] for name in ("train", "val", "test") for row in split[name]],
        key=int,
    )
    pretrain_stl = sorted([row["case_id"] for row in split["pretrain_stl_only"]], key=int)

    manifest = {
        "aneumo_root": str(root),
        "m_dir": args.m_dir,
        "seed": args.seed,
        "definition": (
            "aneumo CFD fields are reserved for supervised fine-tuning/evaluation. "
            "Only non-reserved aneumo STL geometries may be used for STL-only pre-training."
        ),
        "reserved_case_ids": reserved,
        "pretrain_stl_case_ids": pretrain_stl,
        "counts": {
            "usable_cases": len(cases),
            "reserved_total": len(reserved),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
            "pretrain_stl_only": len(split["pretrain_stl_only"]),
        },
        "splits": split,
    }

    json_path = out_dir / "aneumo_cfd_split.json"
    csv_path = out_dir / "aneumo_cfd_split.csv"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    write_csv(csv_path, split)

    print(f"Discovered usable aneumo cases: {len(cases)}")
    print(f"Reserved CFD cases: {len(reserved)} train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])}")
    print(f"STL-only pretrain aneumo cases: {len(pretrain_stl)}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
