#!/usr/bin/env python3
"""Quality-control manifest for vascular STL pre-training data."""
import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import trimesh


DEFAULT_ROOT = "./HemoData/Vascular_STL"
DEFAULT_OUT = "./HemoData/Vascular_STL_QC"
QC_FIELDS = [
    "path",
    "relative_path",
    "dataset",
    "filename",
    "file_size_bytes",
    "load_ok",
    "status",
    "reasons",
    "n_vertices",
    "n_vertices_merged",
    "n_faces",
    "is_watertight",
    "euler_number",
    "boundary_edges",
    "nonmanifold_edges",
    "n_components",
    "largest_component_faces",
    "largest_component_fraction",
    "surface_area",
    "volume",
    "min_extent",
    "max_extent",
    "aspect_ratio",
    "finite_vertices",
    "degenerate_faces",
]


def _edge_counts(mesh):
    if len(mesh.faces) == 0 or len(mesh.edges_unique) == 0:
        return 0, 0
    counts = np.bincount(mesh.faces_unique_edges.reshape(-1),
                         minlength=len(mesh.edges_unique))
    return int((counts == 1).sum()), int((counts > 2).sum())


def _component_stats(mesh):
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        return 0, 0, 0.0

    if len(components) == 0:
        return 0, 0, 0.0
    faces = [len(c.faces) for c in components]
    largest = max(faces)
    frac = largest / max(len(mesh.faces), 1)
    return int(len(components)), int(largest), float(frac)


def _status_and_reasons(row, args):
    reasons = []
    status = "pass"

    def fail(reason):
        nonlocal status
        status = "fail"
        reasons.append(reason)

    def warn(reason):
        nonlocal status
        if status != "fail":
            status = "warn"
        reasons.append(reason)

    if not row["load_ok"]:
        fail("load_error")
        return status, reasons

    if row["file_size_bytes"] < args.min_file_size:
        fail("tiny_file")
    if row["n_vertices"] < args.min_vertices:
        fail("too_few_vertices")
    if row["n_faces"] < args.min_faces:
        fail("too_few_faces")
    if not row["finite_vertices"]:
        fail("nonfinite_vertices")
    if row["surface_area"] <= args.min_area:
        fail("zero_area")
    if row["max_extent"] <= args.min_extent:
        fail("degenerate_extent")
    if row["largest_component_fraction"] < args.min_largest_component_fraction:
        fail("fragmented_components")

    if not row["is_watertight"]:
        warn("not_watertight")
    if row["boundary_edges"] > 0:
        warn("open_boundary")
    if row["nonmanifold_edges"] > 0:
        warn("nonmanifold_edges")
    if row["n_components"] > 1:
        warn("multiple_components")
    if row["aspect_ratio"] is not None and row["aspect_ratio"] > args.max_aspect_ratio:
        warn("extreme_aspect_ratio")
    if row["degenerate_faces"] > 0:
        warn("degenerate_faces")

    return status, reasons


def qc_one(path, root, args):
    dataset = os.path.basename(os.path.dirname(path))
    rel_path = os.path.relpath(path, root)
    row = {
        "path": os.path.abspath(path),
        "relative_path": rel_path,
        "dataset": dataset,
        "filename": os.path.basename(path),
        "file_size_bytes": int(os.path.getsize(path)),
        "load_ok": False,
        "status": "fail",
        "reasons": "load_error",
        "n_vertices": 0,
        "n_vertices_merged": 0,
        "n_faces": 0,
        "is_watertight": False,
        "euler_number": None,
        "boundary_edges": 0,
        "nonmanifold_edges": 0,
        "n_components": 0,
        "largest_component_faces": 0,
        "largest_component_fraction": 0.0,
        "surface_area": 0.0,
        "volume": None,
        "min_extent": 0.0,
        "max_extent": 0.0,
        "aspect_ratio": None,
        "finite_vertices": False,
        "degenerate_faces": 0,
    }

    try:
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values()
                     if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
            if geoms:
                mesh = trimesh.util.concatenate(geoms)

        if not isinstance(mesh, trimesh.Trimesh):
            row["reasons"] = "not_trimesh"
            return row

        topo_mesh = mesh.copy()
        try:
            topo_mesh.merge_vertices()
        except Exception:
            topo_mesh = mesh

        vertices = np.asarray(mesh.vertices)
        extents = np.ptp(vertices, axis=0) if len(vertices) else np.zeros(3)
        positive_extents = extents[extents > 0]
        min_extent = float(positive_extents.min()) if len(positive_extents) else 0.0
        max_extent = float(extents.max()) if len(extents) else 0.0
        aspect = float(max_extent / min_extent) if min_extent > 0 else None
        boundary_edges, nonmanifold_edges = _edge_counts(topo_mesh)
        n_components, largest_faces, largest_frac = _component_stats(topo_mesh)
        degenerate_faces = int((mesh.area_faces <= args.min_face_area).sum()) if len(mesh.faces) else 0

        row.update({
            "load_ok": True,
            "n_vertices": int(len(mesh.vertices)),
            "n_vertices_merged": int(len(topo_mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
            "is_watertight": bool(topo_mesh.is_watertight),
            "euler_number": int(topo_mesh.euler_number),
            "boundary_edges": boundary_edges,
            "nonmanifold_edges": nonmanifold_edges,
            "n_components": n_components,
            "largest_component_faces": largest_faces,
            "largest_component_fraction": largest_frac,
            "surface_area": float(mesh.area),
            "volume": float(topo_mesh.volume) if topo_mesh.is_watertight else None,
            "min_extent": min_extent,
            "max_extent": max_extent,
            "aspect_ratio": aspect,
            "finite_vertices": bool(np.isfinite(vertices).all()) if len(vertices) else False,
            "degenerate_faces": degenerate_faces,
        })

        status, reasons = _status_and_reasons(row, args)
        row["status"] = status
        row["reasons"] = "|".join(reasons)
        return row
    except Exception as exc:
        row["reasons"] = f"load_error:{str(exc)[:160]}"
        return row


def _write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "qc_manifest.jsonl")
    csv_path = os.path.join(out_dir, "qc_manifest.csv")
    summary_path = os.path.join(out_dir, "qc_summary.json")

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    fieldnames = list(rows[0].keys()) if rows else QC_FIELDS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_dataset = defaultdict(Counter)
    reason_counts = Counter()
    for row in rows:
        by_dataset[row["dataset"]][row["status"]] += 1
        for reason in row["reasons"].split("|"):
            if reason:
                reason_counts[reason] += 1

    summary = {
        "total": len(rows),
        "status": dict(Counter(row["status"] for row in rows)),
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "top_reasons": dict(reason_counts.most_common(30)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return jsonl_path, csv_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(description="Create vascular STL quality-control manifest")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional dataset subdirectories to scan")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max files per dataset for smoke tests; 0 means all")
    parser.add_argument("--min_file_size", type=int, default=1024)
    parser.add_argument("--min_vertices", type=int, default=20)
    parser.add_argument("--min_faces", type=int, default=20)
    parser.add_argument("--min_area", type=float, default=1e-10)
    parser.add_argument("--min_extent", type=float, default=1e-8)
    parser.add_argument("--min_face_area", type=float, default=1e-12)
    parser.add_argument("--max_aspect_ratio", type=float, default=250.0)
    parser.add_argument("--min_largest_component_fraction", type=float, default=0.5)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    datasets = args.datasets
    if datasets is None:
        datasets = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)))

    rows = []
    for dataset in datasets:
        pattern = os.path.join(root, dataset, "*.stl")
        files = sorted(glob.glob(pattern))
        if args.max_files > 0:
            files = files[:args.max_files]
        print(f"[{dataset}] QC {len(files)} STL files")
        for i, path in enumerate(files, start=1):
            rows.append(qc_one(path, root, args))
            if i % 100 == 0 or i == len(files):
                print(f"  [{dataset}] {i}/{len(files)}")

    jsonl_path, csv_path, summary_path, summary = _write_outputs(rows, args.out_dir)
    print("=" * 60)
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")
    print(f"SUM:   {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
