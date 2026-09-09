#!/usr/bin/env python3
"""QC and clean fragmented vascular STL files by connected components.

This script is intentionally non-destructive by default. It reads STL files
from HemoData/Vascular_STL and writes cleaned copies to
HemoData/Vascular_STL_component_cleaned, preserving the dataset/file layout.

The connected-component calculation uses a sparse vertex graph, which is much
faster than materializing each component through trimesh.split() for large
intracranial meshes.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


ROOT = Path(".")
DEFAULT_SRC = ROOT / "HemoData" / "Vascular_STL"
DEFAULT_DST = ROOT / "HemoData" / "Vascular_STL_component_cleaned"
DEFAULT_REPORT = ROOT / "results" / "vascular_stl_component_cleaning"


DATASET_DEFAULTS = {
    # CereVessMRA masks are heavily fragmented. For pretraining geometry, the
    # largest connected vascular tree is cleaner than hundreds of isolated bits.
    "CereVessMRA": {
        "keep": "largest",
        "min_faces": 2000,
        "min_fraction_of_largest": 0.05,
    },
    # InterMask and IntrA can contain a few meaningful major components, so keep
    # large components and discard obvious debris.
    "InterMask": {
        "keep": "major",
        "min_faces": 1500,
        "min_fraction_of_largest": 0.05,
    },
    "IntrA": {
        "keep": "major",
        "min_faces": 1500,
        "min_fraction_of_largest": 0.05,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst_root", type=Path, default=DEFAULT_DST)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--datasets", nargs="+", default=["CereVessMRA", "InterMask", "IntrA"])
    parser.add_argument("--copy_unchanged", action="store_true", help="Copy already-clean files too.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qc_only", action="store_true")
    parser.add_argument("--cerevess_keep", choices=["largest", "major"], default="largest")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.vertices) and len(g.faces)]
        if not geometries:
            raise ValueError("empty scene")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("empty mesh")
    # STL stores triangles independently in many files, so shared vertices must
    # be merged before component analysis. Without this, each face can look like
    # an isolated component even when the surface is topologically connected.
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def component_labels(mesh: trimesh.Trimesh):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_vertices = int(len(mesh.vertices))
    if n_vertices == 0 or len(faces) == 0:
        return np.zeros(n_vertices, dtype=np.int32), np.zeros(len(faces), dtype=np.int32)

    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.uint8)
    graph = coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    face_labels = labels[faces[:, 0]]
    return labels.astype(np.int32), face_labels.astype(np.int32)


def component_stats(mesh: trimesh.Trimesh, face_labels: np.ndarray):
    n_components = int(face_labels.max() + 1) if len(face_labels) else 0
    face_counts = np.bincount(face_labels, minlength=n_components).astype(np.int64)
    used_labels = np.flatnonzero(face_counts)
    if len(used_labels) == 0:
        return {
            "n_components": 0,
            "largest_faces": 0,
            "largest_fraction": 0.0,
            "top_faces": [],
        }
    face_counts = face_counts[used_labels]
    order = np.argsort(face_counts)[::-1]
    sorted_counts = face_counts[order]
    return {
        "n_components": int(len(used_labels)),
        "largest_faces": int(sorted_counts[0]),
        "largest_fraction": float(sorted_counts[0] / max(len(mesh.faces), 1)),
        "top_faces": [int(x) for x in sorted_counts[:20]],
        "used_labels": used_labels,
        "sorted_labels": used_labels[order],
        "face_counts_by_label": np.bincount(face_labels, minlength=int(face_labels.max() + 1) if len(face_labels) else 0),
    }


def select_labels(dataset: str, stats: dict, args):
    cfg = dict(DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["IntrA"]))
    if dataset == "CereVessMRA":
        cfg["keep"] = args.cerevess_keep

    labels = np.asarray(stats["sorted_labels"], dtype=np.int64)
    if len(labels) == 0:
        return labels, cfg

    if cfg["keep"] == "largest":
        return labels[:1], cfg

    counts = stats["face_counts_by_label"][labels]
    largest = int(counts[0])
    threshold = max(int(cfg["min_faces"]), int(np.ceil(largest * float(cfg["min_fraction_of_largest"]))))
    keep = labels[counts >= threshold]
    if len(keep) == 0:
        keep = labels[:1]
    return keep, cfg


def clean_mesh(mesh: trimesh.Trimesh, face_labels: np.ndarray, keep_labels: np.ndarray) -> trimesh.Trimesh:
    keep_mask = np.isin(face_labels, keep_labels)
    faces = np.asarray(mesh.faces)[keep_mask]
    cleaned = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces, process=False)
    if hasattr(cleaned, "remove_duplicate_faces"):
        cleaned.remove_duplicate_faces()
    if hasattr(cleaned, "remove_degenerate_faces"):
        cleaned.remove_degenerate_faces()
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned.fix_normals()
    return cleaned


def process_one(path: Path, dataset: str, args):
    dst_path = args.dst_root / dataset / path.name
    row = {
        "dataset": dataset,
        "source": str(path),
        "target": str(dst_path),
        "status": "fail",
        "reason": "",
    }
    try:
        mesh = load_mesh(path)
        labels, face_labels = component_labels(mesh)
        stats = component_stats(mesh, face_labels)
        keep_labels, cfg = select_labels(dataset, stats, args)
        keep_faces = int(np.isin(face_labels, keep_labels).sum()) if len(face_labels) else 0

        row.update(
            {
                "status": "ok",
                "n_vertices": int(len(mesh.vertices)),
                "n_faces": int(len(mesh.faces)),
                "is_watertight": bool(mesh.is_watertight),
                "n_components": int(stats["n_components"]),
                "largest_faces": int(stats["largest_faces"]),
                "largest_fraction": float(stats["largest_fraction"]),
                "top_faces": stats["top_faces"],
                "keep_mode": cfg["keep"],
                "keep_labels": [int(x) for x in keep_labels],
                "kept_components": int(len(keep_labels)),
                "kept_faces": keep_faces,
                "kept_face_fraction": float(keep_faces / max(len(mesh.faces), 1)),
                "removed_faces": int(len(mesh.faces) - keep_faces),
            }
        )

        if args.qc_only:
            row["write_status"] = "qc_only"
            return row

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists() and not args.overwrite:
            row["write_status"] = "exists"
            return row

        needs_cleaning = stats["n_components"] > len(keep_labels) or keep_faces != len(mesh.faces)
        if needs_cleaning:
            cleaned = clean_mesh(mesh, face_labels, keep_labels)
            row.update(
                {
                    "cleaned_vertices": int(len(cleaned.vertices)),
                    "cleaned_faces": int(len(cleaned.faces)),
                    "cleaned_is_watertight": bool(cleaned.is_watertight),
                }
            )
            tmp = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)
            if tmp.exists():
                tmp.unlink()
            cleaned.export(tmp)
            tmp.replace(dst_path)
            row["write_status"] = "cleaned"
        elif args.copy_unchanged:
            shutil.copy2(path, dst_path)
            row["write_status"] = "copied_unchanged"
        else:
            row["write_status"] = "unchanged_not_copied"
        return row
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return row


def write_reports(rows, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / "component_cleaning_manifest.jsonl"
    csv_path = report_dir / "component_cleaning_manifest.csv"
    summary_path = report_dir / "component_cleaning_summary.json"

    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "dataset",
        "source",
        "target",
        "status",
        "reason",
        "write_status",
        "n_vertices",
        "n_faces",
        "is_watertight",
        "n_components",
        "largest_faces",
        "largest_fraction",
        "kept_components",
        "kept_faces",
        "kept_face_fraction",
        "removed_faces",
        "cleaned_vertices",
        "cleaned_faces",
        "cleaned_is_watertight",
        "keep_mode",
        "top_faces",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        ok = [r for r in ds_rows if r["status"] == "ok"]
        summary[dataset] = {
            "total": len(ds_rows),
            "ok": len(ok),
            "failed": len(ds_rows) - len(ok),
            "cleaned": sum(r.get("write_status") == "cleaned" for r in ok),
            "copied_unchanged": sum(r.get("write_status") == "copied_unchanged" for r in ok),
            "unchanged_not_copied": sum(r.get("write_status") == "unchanged_not_copied" for r in ok),
            "max_components": max([r.get("n_components", 0) for r in ok], default=0),
            "median_components": float(np.median([r.get("n_components", 0) for r in ok])) if ok else 0.0,
            "median_kept_face_fraction": float(np.median([r.get("kept_face_fraction", 0.0) for r in ok])) if ok else 0.0,
            "min_kept_face_fraction": float(min([r.get("kept_face_fraction", 0.0) for r in ok], default=0.0)),
        }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return jsonl_path, csv_path, summary_path, summary


def main():
    args = parse_args()
    rows = []
    for dataset in args.datasets:
        files = sorted((args.src_root / dataset).glob("*.stl"))
        for path in tqdm(files, desc=dataset):
            rows.append(process_one(path, dataset, args))

    jsonl_path, csv_path, summary_path, summary = write_reports(rows, args.report_dir)
    print(f"Wrote manifest: {jsonl_path}")
    print(f"Wrote csv: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
