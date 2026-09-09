#!/usr/bin/env python3
"""Visual smoke test for vascular STL transformations before pretraining.

This script is read-only with respect to the real pretraining data directory.
It samples a few STL cases per dataset, applies the same lightweight geometry
steps used by Vascular_PreTraining_Data.py, and writes PNG visualizations plus a
manifest to a separate output directory.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_generation.Vascular_PreTraining_Data import (
    DEFAULT_QC_MANIFEST,
    TARGET_LENGTH,
    VTKDistanceField,
    build_geometry_query,
    filter_files_by_qc,
    get_wall_distance_and_direction,
    load_qc_manifest,
    multi_step_constrained_walk_inside,
    qc_row_for_file,
    transform_mesh,
)


DEFAULT_ROOT = "./HemoData/Vascular_STL"
DEFAULT_OUT = "./results/Vascular_PreTrain_VisualSmoke"
PRETRAIN_DEFAULT_DATASETS = {
    "4TCTA_AAA",
    "CMHA",
    "IntrA",
    "AneuRisk",
    "Aneux",
    "Totalsegmentator",
}


def _safe_name(name):
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(keep)


def _mesh_stats(mesh):
    extents = np.ptp(np.asarray(mesh.vertices), axis=0) if len(mesh.vertices) else np.zeros(3)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "bounds_min": np.asarray(mesh.bounds[0]).astype(float).tolist() if len(mesh.vertices) else [0, 0, 0],
        "bounds_max": np.asarray(mesh.bounds[1]).astype(float).tolist() if len(mesh.vertices) else [0, 0, 0],
        "extents": extents.astype(float).tolist(),
        "max_extent": float(extents.max()) if len(extents) else 0.0,
    }


def _load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values()
                 if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not geoms:
            raise ValueError("Scene has no triangle meshes")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Not a valid Trimesh")
    return mesh


def _sample_surface_points(mesh, n):
    n = max(1, int(n))
    try:
        points, face_idx = mesh.sample(n, return_index=True)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        return points.astype(np.float32), normals
    except Exception:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if len(vertices) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        idx = np.random.choice(len(vertices), size=min(n, len(vertices)), replace=len(vertices) < n)
        return vertices[idx], np.zeros((len(idx), 3), dtype=np.float32)


def _sample_volume_inside_mesh_light(vtk_dist, mesh, n, batch_size=12000, max_iter=25):
    bounds = mesh.bounds
    collected = []
    total = 0
    for _ in range(max_iter):
        if total >= n:
            break
        pts = np.random.uniform(bounds[0], bounds[1], (batch_size, 3)).astype(np.float32)
        inside = vtk_dist.contains_batch(pts)
        new_pts = pts[inside]
        if len(new_pts):
            collected.append(new_pts)
            total += len(new_pts)

    if not collected:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(collected, axis=0)[:n].astype(np.float32)


def _pad_points(points, n):
    if len(points) == 0:
        return points, 1.0, False
    if len(points) >= n:
        return points[:n], 0.0, False
    idx = np.random.choice(len(points), n - len(points), replace=True)
    padded = np.concatenate([points, points[idx]], axis=0)
    return padded.astype(np.float32), (n - len(points)) / n, True


def _set_equal_axes(ax, points):
    points = np.asarray(points)
    if points.size == 0:
        return
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 1e-6)
    radius *= 1.08
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _setup_3d(ax, title, points):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=18, azim=-62)
    _set_equal_axes(ax, points)


def _save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scatter(ax, points, color="#3366cc", size=1.0, alpha=0.75, **kwargs):
    points = np.asarray(points)
    if len(points) == 0:
        return None
    return ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      s=size, c=color, alpha=alpha, linewidths=0, **kwargs)


def _plot_raw_mesh(raw_points, stats, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, raw_points, color="#4c78a8", size=1.0, alpha=0.75)
    title = f"00 raw STL | V={stats['vertices']} F={stats['faces']} watertight={stats['watertight']}"
    _setup_3d(ax, title, raw_points)
    _save_figure(fig, out_path)


def _plot_components(components, kept_mesh, out_path, n_points):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    for i, comp in enumerate(components[:8]):
        pts, _ = _sample_surface_points(comp, max(80, n_points // max(len(components[:8]), 1)))
        all_points.append(pts)
        is_kept = i == 0
        color = "#1b9e77" if is_kept else "#cccccc"
        size = 1.4 if is_kept else 0.8
        alpha = 0.9 if is_kept else 0.35
        _scatter(ax, pts, color=color, size=size, alpha=alpha)
    plot_points = np.vstack(all_points) if all_points else np.asarray(kept_mesh.vertices)
    title = f"01 components | total={len(components)} kept_faces={len(kept_mesh.faces)}"
    _setup_3d(ax, title, plot_points)
    _save_figure(fig, out_path)


def _plot_normalized(norm_points, scale, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, norm_points, color="#1f77b4", size=1.0, alpha=0.75)
    _setup_3d(ax, f"02 normalized mesh | max_extent={TARGET_LENGTH:g} scale={scale:.4g}", norm_points)
    _save_figure(fig, out_path)


def _plot_surface_samples(mesh_points, surf_pts, surf_normals, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, mesh_points, color="#dddddd", size=0.7, alpha=0.22)
    _scatter(ax, surf_pts, color="#e45756", size=3.0, alpha=0.85)
    if len(surf_pts) and np.linalg.norm(surf_normals, axis=1).max() > 0:
        count = min(80, len(surf_pts))
        idx = np.linspace(0, len(surf_pts) - 1, count).astype(int)
        ax.quiver(surf_pts[idx, 0], surf_pts[idx, 1], surf_pts[idx, 2],
                  surf_normals[idx, 0], surf_normals[idx, 1], surf_normals[idx, 2],
                  length=0.12, color="#111111", linewidth=0.4, alpha=0.7)
    points = np.vstack([mesh_points, surf_pts]) if len(surf_pts) else mesh_points
    _setup_3d(ax, f"03 surface samples | N={len(surf_pts)} normals shown", points)
    _save_figure(fig, out_path)


def _plot_volume_samples(mesh_points, vol_pts, padded_ratio, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, mesh_points, color="#bbbbbb", size=0.6, alpha=0.18)
    _scatter(ax, vol_pts, color="#54a24b", size=2.2, alpha=0.70)
    points = np.vstack([mesh_points, vol_pts]) if len(vol_pts) else mesh_points
    _setup_3d(ax, f"04 volume samples | N={len(vol_pts)} padded_ratio={padded_ratio:.3f}", points)
    _save_figure(fig, out_path)


def _plot_unified_x(vol_pts, surf_pts, dist, out_path):
    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    if len(vol_pts):
        sc = ax.scatter(vol_pts[:, 0], vol_pts[:, 1], vol_pts[:, 2],
                        s=2.5, c=dist, cmap="viridis", alpha=0.80, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.02)
        cbar.set_label("dist_to_wall", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    _scatter(ax, surf_pts, color="#e45756", size=2.0, alpha=0.45)
    points = np.vstack([vol_pts, surf_pts]) if len(surf_pts) else vol_pts
    title = f"05 unified x preview | x=({len(points)}, 7), fx=geom4 + condition4"
    _setup_3d(ax, title, points)
    _save_figure(fig, out_path)


def _plot_walk(vol_pts, surf_pts, condition, supervise, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, surf_pts, color="#cccccc", size=1.2, alpha=0.25)
    if len(vol_pts):
        count = min(160, len(vol_pts))
        idx = np.linspace(0, len(vol_pts) - 1, count).astype(int)
        start = vol_pts[idx]
        dirs = condition[idx, :3]
        steps = condition[idx, 3:4]
        to_wall = -supervise[idx, :3]
        _scatter(ax, start, color="#4c78a8", size=4.0, alpha=0.85)
        ax.quiver(start[:, 0], start[:, 1], start[:, 2],
                  dirs[:, 0] * steps[:, 0], dirs[:, 1] * steps[:, 0], dirs[:, 2] * steps[:, 0],
                  length=0.22, normalize=False, color="#f58518", linewidth=0.55, alpha=0.65)
        ax.quiver(start[:, 0], start[:, 1], start[:, 2],
                  to_wall[:, 0], to_wall[:, 1], to_wall[:, 2],
                  length=1.0, normalize=False, color="#e45756", linewidth=0.45, alpha=0.45)
    points = np.vstack([vol_pts, surf_pts]) if len(surf_pts) else vol_pts
    _setup_3d(ax, "06 walk prompt | orange=random step, red=to wall target", points)
    _save_figure(fig, out_path)


def _make_montage(image_paths, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes = axes.reshape(-1)
    for ax, path in zip(axes, image_paths):
        ax.imshow(plt.imread(path))
        ax.set_axis_off()
        ax.set_title(os.path.basename(path).replace(".png", ""), fontsize=9)
    for ax in axes[len(image_paths):]:
        ax.set_axis_off()
    _save_figure(fig, out_path)


def _write_case_readme(case_dir, row):
    path = os.path.join(case_dir, "README.md")
    rel_images = [os.path.basename(p) for p in row["images"]]
    lines = [
        f"# {row['dataset']} / {row['case_name']}",
        "",
        f"- STL: `{row['stl_path']}`",
        f"- QC: `{row.get('qc_status')}` `{row.get('qc_reasons')}`",
        f"- Pretrain default dataset: `{row['pretrain_default_dataset']}`",
        f"- Original faces: `{row['raw_stats']['faces']}`",
        f"- Processed faces: `{row['processed_stats']['faces']}`",
        f"- Scale to normalized max extent 5: `{row['scale']}`",
        f"- Visual x preview shape: `{row['x_preview_shape']}`",
        f"- Visual condition shape: `{row['condition_preview_shape']}`",
        f"- Visual supervise shape: `{row['supervise_preview_shape']}`",
        f"- Padded ratio in visual volume sample: `{row['visual_padded_ratio']}`",
        "",
        "## Montage",
        "",
        "![montage](case_montage.png)",
        "",
        "## Stages",
        "",
    ]
    for image in rel_images:
        lines.extend([f"### {image}", "", f"![{image}]({image})", ""])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _process_case(path, dataset, out_dir, root, qc_by_key, args):
    case_name = os.path.splitext(os.path.basename(path))[0]
    case_dir = os.path.join(out_dir, dataset, _safe_name(case_name))
    os.makedirs(case_dir, exist_ok=True)

    stable = hashlib.md5(f"{dataset}/{case_name}".encode()).hexdigest()
    np.random.seed(args.seed + int(stable[:8], 16) % 100000)
    t0 = time.time()

    raw_mesh = _load_mesh(path)
    raw_stats = _mesh_stats(raw_mesh)
    raw_points, _ = _sample_surface_points(raw_mesh, args.plot_surface_points)

    components = raw_mesh.split(only_watertight=False)
    components = list(components) if len(components) else [raw_mesh]
    components.sort(key=lambda m: len(m.faces), reverse=True)
    kept_mesh = components[0].copy()
    kept_largest_component = len(components) > 1

    norm_mesh = kept_mesh.copy()
    norm_mesh, scale = transform_mesh(norm_mesh)
    processed_stats = _mesh_stats(norm_mesh)
    norm_points, _ = _sample_surface_points(norm_mesh, args.plot_surface_points)

    vtk_dist = VTKDistanceField(norm_mesh)
    geometry_query = build_geometry_query(norm_mesh, args.geometry_backend)

    surf_pts, surf_normals = _sample_surface_points(norm_mesh, args.surface_points)
    vol_sampled = _sample_volume_inside_mesh_light(
        vtk_dist, norm_mesh, args.volume_points,
        batch_size=args.volume_batch_size,
        max_iter=args.volume_max_iter)
    sampled_volume_points = int(len(vol_sampled))
    vol_pts, padded_ratio, padded_volume_points = _pad_points(vol_sampled, args.volume_points)

    if len(vol_pts):
        vol_dist, vol_dir = get_wall_distance_and_direction(geometry_query, vol_pts)
    else:
        vol_dist = np.zeros((0,), dtype=np.float32)
        vol_dir = np.zeros((0, 3), dtype=np.float32)

    vol_data = np.hstack([vol_pts, vol_dist.reshape(-1, 1), vol_dir]).astype(np.float32) if len(vol_pts) else np.zeros((0, 7), dtype=np.float32)
    surf_data = np.hstack([surf_pts,
                           np.zeros((len(surf_pts), 1), dtype=np.float32),
                           surf_normals]).astype(np.float32) if len(surf_pts) else np.zeros((0, 7), dtype=np.float32)
    x_preview = np.vstack([vol_data, surf_data])

    if len(vol_pts) and len(surf_pts):
        walk = multi_step_constrained_walk_inside(
            geometry_query, vtk_dist, vol_pts, surf_pts,
            collision_backend=args.collision_backend)
        condition = walk["condition"].astype(np.float32)
        supervise = walk["supervise"].astype(np.float32)
    else:
        condition = np.zeros((len(x_preview), 4), dtype=np.float32)
        supervise = np.zeros((len(x_preview), 9), dtype=np.float32)

    image_paths = [
        os.path.join(case_dir, "00_raw_mesh.png"),
        os.path.join(case_dir, "01_largest_component.png"),
        os.path.join(case_dir, "02_normalized_mesh.png"),
        os.path.join(case_dir, "03_surface_samples.png"),
        os.path.join(case_dir, "04_volume_samples.png"),
        os.path.join(case_dir, "05_unified_x_features.png"),
        os.path.join(case_dir, "06_walk_prompt_supervision.png"),
    ]
    _plot_raw_mesh(raw_points, raw_stats, image_paths[0])
    _plot_components(components, kept_mesh, image_paths[1], args.plot_surface_points)
    _plot_normalized(norm_points, scale, image_paths[2])
    _plot_surface_samples(norm_points, surf_pts, surf_normals, image_paths[3])
    _plot_volume_samples(norm_points, vol_pts, padded_ratio, image_paths[4])
    _plot_unified_x(vol_pts, surf_pts, vol_dist, image_paths[5])
    _plot_walk(vol_pts, surf_pts, condition, supervise, image_paths[6])
    montage_path = os.path.join(case_dir, "case_montage.png")
    _make_montage(image_paths, montage_path)

    qc_row = qc_row_for_file(path, root, qc_by_key) if qc_by_key else None
    row = {
        "dataset": dataset,
        "case_name": case_name,
        "stl_path": os.path.abspath(path),
        "case_dir": os.path.abspath(case_dir),
        "pretrain_default_dataset": dataset in PRETRAIN_DEFAULT_DATASETS,
        "qc_status": qc_row.get("status") if qc_row else None,
        "qc_reasons": qc_row.get("reasons") if qc_row else None,
        "raw_stats": raw_stats,
        "n_components": int(len(components)),
        "kept_largest_component": bool(kept_largest_component),
        "processed_stats": processed_stats,
        "scale": float(scale),
        "geometry_backend_effective": geometry_query.backend,
        "collision_backend": args.collision_backend,
        "visual_surface_points": int(len(surf_pts)),
        "visual_volume_points_requested": int(args.volume_points),
        "visual_volume_points_sampled": sampled_volume_points,
        "visual_padded_volume_points": bool(padded_volume_points),
        "visual_padded_ratio": float(padded_ratio),
        "x_preview_shape": list(x_preview.shape),
        "condition_preview_shape": list(condition.shape),
        "supervise_preview_shape": list(supervise.shape),
        "dist_to_wall_min": float(vol_dist.min()) if len(vol_dist) else None,
        "dist_to_wall_mean": float(vol_dist.mean()) if len(vol_dist) else None,
        "dist_to_wall_max": float(vol_dist.max()) if len(vol_dist) else None,
        "images": [os.path.abspath(p) for p in image_paths],
        "montage": os.path.abspath(montage_path),
        "elapsed_sec": float(time.time() - t0),
    }

    with open(os.path.join(case_dir, "case_manifest.json"), "w") as f:
        json.dump(row, f, indent=2)
    _write_case_readme(case_dir, row)
    return row


def _write_report(out_dir, rows):
    report_path = os.path.join(out_dir, "README.md")
    lines = [
        "# Vascular Transform Visual Smoke Test",
        "",
        "This output is for visual inspection only. It does not create training checkpoints and does not write to the real pretraining data directory.",
        "",
        "Stages per case:",
        "",
        "1. `00_raw_mesh.png`: original STL surface point preview.",
        "2. `01_largest_component.png`: component split and largest component kept by the pipeline.",
        "3. `02_normalized_mesh.png`: geometry after centering and max-extent scaling to 5.",
        "4. `03_surface_samples.png`: sampled wall points and normals.",
        "5. `04_volume_samples.png`: sampled interior points used as vessel-domain points.",
        "6. `05_unified_x_features.png`: preview of unified `x[:, :7]` with distance-to-wall coloring.",
        "7. `06_walk_prompt_supervision.png`: random prompt directions and nearest-wall supervision vectors.",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        rel_case = os.path.relpath(row["case_dir"], out_dir)
        rel_montage = os.path.join(rel_case, "case_montage.png")
        lines.extend([
            f"### {row['dataset']} / {row['case_name']}",
            "",
            f"- QC: `{row.get('qc_status')}` `{row.get('qc_reasons')}`",
            f"- Components: `{row['n_components']}`, kept largest: `{row['kept_largest_component']}`",
            f"- Raw faces -> processed faces: `{row['raw_stats']['faces']}` -> `{row['processed_stats']['faces']}`",
            f"- Scale: `{row['scale']:.6g}`",
            f"- x preview: `{row['x_preview_shape']}`, condition: `{row['condition_preview_shape']}`, supervise: `{row['supervise_preview_shape']}`",
            f"- Volume sampled/requested: `{row['visual_volume_points_sampled']}/{row['visual_volume_points_requested']}`, padded ratio: `{row['visual_padded_ratio']:.3f}`",
            f"- Case folder: `{rel_case}`",
            "",
            f"![{row['dataset']} {row['case_name']}]({rel_montage})",
            "",
        ])
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Visual smoke test for vascular STL transform stages")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset subdirectories. Default: all subdirectories under root.")
    parser.add_argument("--samples_per_dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--qc_manifest", type=str, default=DEFAULT_QC_MANIFEST)
    parser.add_argument("--qc_statuses", nargs="+", default=["pass", "warn"])
    parser.add_argument("--ignore_qc", action="store_true")
    parser.add_argument("--geometry_backend", type=str, default="auto",
                        choices=["auto", "fcpw", "trimesh"])
    parser.add_argument("--collision_backend", type=str, default="fcpw_ray",
                        choices=["fcpw_ray", "vtk"])
    parser.add_argument("--plot_surface_points", type=int, default=1800)
    parser.add_argument("--surface_points", type=int, default=512)
    parser.add_argument("--volume_points", type=int, default=1536)
    parser.add_argument("--volume_batch_size", type=int, default=12000)
    parser.add_argument("--volume_max_iter", type=int, default=25)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = args.datasets
    if datasets is None:
        datasets = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    qc_by_key = {}
    allowed_statuses = set(args.qc_statuses)
    if not args.ignore_qc:
        qc_by_key, qc_counts = load_qc_manifest(args.qc_manifest)
        print(f"Loaded QC manifest: {args.qc_manifest} counts={qc_counts}")

    rng = random.Random(args.seed)
    selected = []
    selection = {}
    for dataset in datasets:
        files = sorted(glob.glob(os.path.join(root, dataset, "*.stl")))
        if qc_by_key:
            files, skipped, unmatched, skipped_by_status = filter_files_by_qc(
                files, root, qc_by_key, allowed_statuses, require_match=False)
            print(f"[{dataset}] QC kept={len(files)} skipped={len(skipped)} unmatched={unmatched} skipped_by_status={skipped_by_status}")
        if len(files) > args.samples_per_dataset:
            files = rng.sample(files, args.samples_per_dataset)
            files = sorted(files)
        selection[dataset] = [os.path.basename(p) for p in files]
        selected.extend((dataset, p) for p in files)

    with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
        json.dump({
            "root": root,
            "seed": args.seed,
            "samples_per_dataset": args.samples_per_dataset,
            "datasets": datasets,
            "selection": selection,
        }, f, indent=2)

    rows = []
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as manifest:
        for idx, (dataset, path) in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {dataset}/{os.path.basename(path)}")
            try:
                row = _process_case(path, dataset, args.out_dir, root, qc_by_key, args)
                rows.append(row)
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                print(f"  OK {row['elapsed_sec']:.1f}s -> {row['case_dir']}")
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "stl_path": os.path.abspath(path),
                    "case_name": os.path.splitext(os.path.basename(path))[0],
                    "success": False,
                    "error": str(exc)[:500],
                }
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                print(f"  FAIL: {exc}")

    report_path = _write_report(args.out_dir, rows)
    print("=" * 60)
    print(f"Output:   {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_path}")
    print(f"Cases:    {len(rows)} ok / {len(selected)} selected")


if __name__ == "__main__":
    main()
