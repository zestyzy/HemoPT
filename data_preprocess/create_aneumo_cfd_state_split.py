#!/usr/bin/env python3
"""Create an aneumo split CSV for a different m state using an existing case split."""
import argparse
import csv
import json
from pathlib import Path


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


def _write_csv(path: Path, split_rows):
    fieldnames = [
        "split", "case_id", "case_dir", "m_dir", "internal_npy", "inlet_npy",
        "outlet_npy", "wall_npy", "internal_vtu", "wall_vtp", "stl_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for row in split_rows[split_name]:
                out = dict(row)
                out["split"] = split_name
                writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_split_json", type=str,
                        default="HemoData/Aneumo_CFD_Splits/aneumo_cfd_split.json")
    parser.add_argument("--aneumo_root", type=str,
                        default="./totaldata/unpacked/dataset_2/aneumo")
    parser.add_argument("--m_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.aneumo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = json.load(open(args.base_split_json))
    split_rows = {}
    for split_name in ("train", "val", "test"):
        rows = []
        for row in base["splits"][split_name]:
            paths = _case_paths(root, row["case_id"], args.m_dir)
            required = ["internal_npy", "inlet_npy", "outlet_npy", "wall_vtp", "stl_path"]
            missing = [key for key in required if not Path(paths[key]).exists()]
            if missing:
                raise FileNotFoundError(f"case={row['case_id']} m_dir={args.m_dir} missing {missing}")
            rows.append(paths)
        split_rows[split_name] = rows

    manifest = {
        "base_split_json": str(Path(args.base_split_json).resolve()),
        "aneumo_root": str(root),
        "m_dir": args.m_dir,
        "definition": "Same aneumo case split as base manifest; CFD state m_dir is replaced.",
        "reserved_case_ids": base["reserved_case_ids"],
        "pretrain_stl_case_ids": base.get("pretrain_stl_case_ids", []),
        "counts": {
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
        "splits": split_rows,
    }
    safe = args.m_dir.replace("=", "").replace(".", "p")
    json_path = out_dir / f"aneumo_cfd_split_{safe}.json"
    csv_path = out_dir / f"aneumo_cfd_split_{safe}.csv"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _write_csv(csv_path, split_rows)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
