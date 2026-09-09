#!/usr/bin/env python3
"""Scan local vascular datasets for likely CFD simulation data.

The scanner is intentionally conservative: it does not parse large binaries.
It searches directory/file names and extensions for CFD-specific evidence such
as velocity, pressure, flow, WSS, OpenFOAM, VTU/VTK fields, and solver outputs.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


ROOT = Path(".")
DEFAULT_OUT = ROOT / "results" / "full_pretrain" / "cfd_availability"

DATASET_ROOTS = {
    "4TCTA_AAA": [ROOT / "HemoData" / "0511" / "4TCTA_AAA"],
    "AneuRisk": [ROOT / "HemoData" / "AneuRisk"],
    "Aneux": [ROOT / "HemoData" / "Aneux"],
    "CMHA": [ROOT / "HemoData" / "0511" / "CMHA"],
    "CereVessMRA": [ROOT / "totaldata" / "unpacked" / "dataset_2" / "CereVessMRA"],
    "ImageCAS": [ROOT / "totaldata" / "unpacked" / "ImageCAS-mask"],
    "InterMask": [ROOT / "totaldata" / "unpacked" / "inter-mask"],
    "IntrA": [ROOT / "HemoData" / "0511" / "IntrA", ROOT / "HemoData" / "0511" / "intra"],
    "PARSE2022": [ROOT / "totaldata" / "unpacked" / "dataset_2" / "PARSE 2022"],
    "Totalsegmentator": [ROOT / "HemoData" / "Totalsegmentator_dataset"],
    "VMR": [ROOT / "HemoData" / "0511" / "VMR"],
    "aneumo": [ROOT / "totaldata" / "unpacked" / "dataset_2" / "aneumo"],
    "imageTBAD": [
        ROOT / "totaldata" / "unpacked" / "dataset_2" / "imageTBAD",
        ROOT / "totaldata" / "unpacked" / "dataset_2" / "imageTBAD_data",
    ],
}

CFD_KEYWORDS = {
    "cfd",
    "simulation",
    "sim",
    "solution",
    "velocity",
    "vel",
    "pressure",
    "press",
    "wss",
    "wall_shear",
    "shear",
    "flow",
    "flowrate",
    "inlet",
    "outlet",
    "centerline",
    "streamline",
    "hemodynamic",
    "hemodynamics",
    "openfoam",
    "foam",
    "fluent",
    "ansys",
    "fsi",
    "navier",
    "stokes",
    "vtu",
}

CFD_EXTENSIONS = {
    ".vtu",
    ".vtk",
    ".vtp",
    ".foam",
    ".case",
    ".cas",
    ".dat",
    ".csv",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".mat",
    ".pvd",
    ".pvtu",
    ".ensight",
}

GEOMETRY_ONLY_EXTENSIONS = {".stl", ".obj", ".ply", ".nii", ".gz", ".mha", ".mhd", ".nrrd", ".dcm"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max_examples", type=int, default=80)
    parser.add_argument("--max_files_per_root", type=int, default=200000)
    return parser.parse_args()


def lower_tokens(path: Path) -> str:
    return str(path).lower().replace("-", "_").replace(" ", "_")


def classify_path(path: Path):
    text = lower_tokens(path)
    suffixes = [s.lower() for s in path.suffixes]
    ext = "".join(suffixes[-2:]) if suffixes[-2:] == [".nii", ".gz"] else path.suffix.lower()
    keyword_hits = sorted([kw for kw in CFD_KEYWORDS if kw in text])
    ext_hit = path.suffix.lower() in CFD_EXTENSIONS or ext in CFD_EXTENSIONS
    geometry_only = path.suffix.lower() in GEOMETRY_ONLY_EXTENSIONS or ext in GEOMETRY_ONLY_EXTENSIONS
    return keyword_hits, ext_hit, geometry_only


def scan_dataset(dataset: str, roots, max_examples: int, max_files_per_root: int):
    examples = []
    ext_counts = defaultdict(int)
    keyword_counts = defaultdict(int)
    file_count = 0
    dir_count = 0
    cfd_score = 0
    strong_hits = 0
    weak_hits = 0

    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dir_count += 1
            dir_path = Path(dirpath)
            dir_keywords, _, _ = classify_path(dir_path)
            for kw in dir_keywords:
                keyword_counts[kw] += 1
            if dir_keywords:
                weak_hits += 1
                cfd_score += min(3, len(dir_keywords))
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "kind": "directory",
                            "path": str(dir_path),
                            "keywords": dir_keywords,
                            "extension": "",
                            "strong": False,
                        }
                    )

            for name in filenames:
                file_count += 1
                path = dir_path / name
                suffix = path.suffix.lower()
                ext_counts[suffix or "<none>"] += 1
                keywords, ext_hit, geometry_only = classify_path(path)
                for kw in keywords:
                    keyword_counts[kw] += 1

                strong = bool(ext_hit and keywords) or any(k in {"velocity", "pressure", "wss", "wall_shear", "openfoam", "fluent", "ansys"} for k in keywords)
                weak = bool(keywords or ext_hit)
                if strong:
                    strong_hits += 1
                    cfd_score += 8 + min(5, len(keywords))
                elif weak and not geometry_only:
                    weak_hits += 1
                    cfd_score += 1 + min(2, len(keywords))

                if weak and len(examples) < max_examples:
                    examples.append(
                        {
                            "kind": "file",
                            "path": str(path),
                            "keywords": keywords,
                            "extension": suffix,
                            "strong": strong,
                        }
                    )

                if file_count >= max_files_per_root:
                    break
            if file_count >= max_files_per_root:
                break

    if strong_hits > 0:
        availability = "likely"
    elif weak_hits > 0:
        availability = "possible"
    else:
        availability = "not_found"

    return {
        "dataset": dataset,
        "roots": [str(r) for r in roots if r.exists()],
        "missing_roots": [str(r) for r in roots if not r.exists()],
        "availability": availability,
        "score": int(cfd_score),
        "strong_hits": int(strong_hits),
        "weak_hits": int(weak_hits),
        "file_count_scanned": int(file_count),
        "dir_count_scanned": int(dir_count),
        "top_extensions": sorted(ext_counts.items(), key=lambda kv: kv[1], reverse=True)[:20],
        "keyword_counts": dict(sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "examples": examples,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, roots in tqdm(sorted(DATASET_ROOTS.items()), desc="scan datasets"):
        rows.append(scan_dataset(dataset, roots, args.max_examples, args.max_files_per_root))

    json_path = args.out_dir / "cfd_availability_scan.json"
    jsonl_path = args.out_dir / "cfd_availability_scan.jsonl"
    csv_path = args.out_dir / "cfd_availability_summary.csv"

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "dataset",
        "availability",
        "score",
        "strong_hits",
        "weak_hits",
        "file_count_scanned",
        "dir_count_scanned",
        "roots",
        "top_extensions",
        "keyword_counts",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["roots"] = ";".join(row["roots"])
            out["top_extensions"] = json.dumps(row["top_extensions"], ensure_ascii=False)
            out["keyword_counts"] = json.dumps(row["keyword_counts"], ensure_ascii=False)
            writer.writerow({k: out.get(k, "") for k in fields})

    print(f"Wrote {json_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"{row['dataset']}: {row['availability']} "
            f"strong={row['strong_hits']} weak={row['weak_hits']} files={row['file_count_scanned']}"
        )


if __name__ == "__main__":
    main()
