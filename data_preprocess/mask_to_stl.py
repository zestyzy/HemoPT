#!/usr/bin/env python3
"""Convert vascular segmentation masks to STL surfaces for HemoPT.

This script is the generic mask-to-STL entry point used before STL quality
control and HemoPT pretraining-data construction. It accepts binary or
multi-label NIfTI masks, extracts vascular foreground voxels, converts them to
physical-coordinate surfaces with marching cubes, and writes one STL per mask
under the HemoPT `Vascular_STL` layout:

    HemoData/Vascular_STL/<dataset>/<case>.stl

No image intensities or CFD labels are used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from tqdm import tqdm


MASK_SUFFIXES = (".nii.gz", ".nii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NIfTI vascular segmentation masks to STL meshes."
    )
    parser.add_argument(
        "--src_root",
        type=Path,
        required=True,
        help="Input mask file or directory containing .nii/.nii.gz masks.",
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path("HemoData/Vascular_STL"),
        help="Output STL root. Files are written to dst_root/dataset/*.stl.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for flat input folders. If omitted, parent folders are used.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search src_root for NIfTI masks.",
    )
    parser.add_argument(
        "--labels",
        type=float,
        nargs="+",
        default=None,
        help="Foreground label values. Default: all voxels > 0.",
    )
    parser.add_argument(
        "--min_component_voxels",
        type=int,
        default=32,
        help="Minimum connected-component size to keep.",
    )
    parser.add_argument(
        "--min_component_fraction",
        type=float,
        default=1e-4,
        help="Also keep components at least this fraction of the largest component.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Marching-cubes step size. Larger values are faster but coarser.",
    )
    parser.add_argument(
        "--smooth_iterations",
        type=int,
        default=0,
        help="Optional Laplacian smoothing iterations after mesh extraction.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip outputs that already exist and are non-empty.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("HemoData/Vascular_STL/mask_to_stl_manifest.jsonl"),
        help="JSONL conversion manifest path.",
    )
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    name = path.name
    for suffix in MASK_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"[^A-Za-z0-9._=-]+", "_", name)
    return name.strip("._-") or "case"


def iter_masks(src_root: Path, recursive: bool) -> list[Path]:
    if src_root.is_file():
        if src_root.name.endswith(MASK_SUFFIXES):
            return [src_root]
        raise ValueError(f"Unsupported mask file: {src_root}")

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in src_root.glob(pattern)
        if path.is_file() and path.name.endswith(MASK_SUFFIXES)
    ]
    return sorted(files, key=lambda p: str(p))


def dataset_for_path(path: Path, src_root: Path, dataset_name: str | None) -> str:
    if dataset_name:
        return re.sub(r"[^A-Za-z0-9._=-]+", "_", dataset_name)
    if src_root.is_file():
        return re.sub(r"[^A-Za-z0-9._=-]+", "_", path.parent.name or "dataset")
    rel = path.relative_to(src_root)
    if len(rel.parts) > 1:
        return re.sub(r"[^A-Za-z0-9._=-]+", "_", rel.parts[0])
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", src_root.name or "dataset")


def make_foreground(data: np.ndarray, labels: list[float] | None) -> np.ndarray:
    if labels is None:
        return np.asarray(data > 0)
    mask = np.zeros(data.shape, dtype=bool)
    for label in labels:
        mask |= np.isclose(data, label)
    return mask


def crop_foreground(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None, None
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, np.asarray(mask.shape))
    slices = tuple(slice(int(lo[i]), int(hi[i])) for i in range(mask.ndim))
    return mask[slices], lo.astype(np.float64)


def filter_components(
    mask: np.ndarray,
    min_component_voxels: int,
    min_component_fraction: float,
) -> tuple[np.ndarray, dict[str, int]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return mask, {
            "n_components_raw": 0,
            "n_components_kept": 0,
            "largest_component_voxels": 0,
            "kept_voxels": 0,
        }

    counts = np.bincount(labeled.ravel())[1:]
    largest = int(counts.max()) if counts.size else 0
    threshold = max(
        int(min_component_voxels),
        int(np.ceil(float(largest) * float(min_component_fraction))),
    )
    keep = np.where(counts >= threshold)[0] + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(counts)) + 1], dtype=np.int64)

    filtered = np.isin(labeled, keep)
    return filtered, {
        "n_components_raw": int(n_components),
        "n_components_kept": int(keep.size),
        "largest_component_voxels": int(largest),
        "kept_voxels": int(filtered.sum()),
        "component_threshold_voxels": int(threshold),
    }


def mask_to_mesh(
    mask: np.ndarray,
    affine: np.ndarray,
    crop_origin: np.ndarray,
    step_size: int,
    smooth_iterations: int,
) -> trimesh.Trimesh:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        step_size=step_size,
        allow_degenerate=False,
    )

    voxel_verts = verts + crop_origin[None, :] - 1.0
    homogeneous = np.c_[voxel_verts, np.ones(len(voxel_verts), dtype=np.float64)]
    world_verts = (affine @ homogeneous.T).T[:, :3]

    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces, process=False)
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
    return mesh


def temp_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def write_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def convert_one(mask_path: Path, args: argparse.Namespace, src_root: Path) -> dict:
    dataset = dataset_for_path(mask_path, src_root, args.dataset_name)
    out_path = args.dst_root / dataset / f"{safe_stem(mask_path)}.stl"
    row = {
        "source": str(mask_path),
        "dataset": dataset,
        "target": str(out_path),
        "status": "fail",
        "reason": "",
    }

    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 1024:
        row["status"] = "skip_existing"
        return row

    tmp = temp_path(out_path)
    try:
        image = nib.load(str(mask_path))
        data = np.asanyarray(image.dataobj)
        if data.ndim != 3:
            row["reason"] = f"expected_3d_mask_got_{data.ndim}d"
            return row

        foreground = make_foreground(data, args.labels)
        row["shape"] = [int(x) for x in foreground.shape]
        row["raw_voxels"] = int(foreground.sum())
        row["zooms"] = [float(x) for x in image.header.get_zooms()[:3]]
        if row["raw_voxels"] == 0:
            row["reason"] = "empty_mask"
            return row

        cropped, crop_origin = crop_foreground(foreground)
        if cropped is None or crop_origin is None:
            row["reason"] = "empty_after_crop"
            return row

        filtered, component_stats = filter_components(
            cropped,
            min_component_voxels=args.min_component_voxels,
            min_component_fraction=args.min_component_fraction,
        )
        row.update(component_stats)
        if int(filtered.sum()) == 0:
            row["reason"] = "empty_after_component_filter"
            return row

        mesh = mask_to_mesh(
            filtered,
            affine=np.asarray(image.affine, dtype=np.float64),
            crop_origin=crop_origin,
            step_size=args.step_size,
            smooth_iterations=args.smooth_iterations,
        )
        row["n_vertices"] = int(len(mesh.vertices))
        row["n_faces"] = int(len(mesh.faces))
        row["is_watertight"] = bool(mesh.is_watertight)
        if len(mesh.faces) < 20:
            row["reason"] = "too_few_faces"
            return row

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            tmp.unlink()
        mesh.export(tmp)
        os.replace(tmp, out_path)
        row["status"] = "ok"
        return row
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["traceback"] = traceback.format_exc(limit=4)
        return row
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    src_root = args.src_root
    masks = iter_masks(src_root, recursive=args.recursive)
    if not masks:
        raise RuntimeError(f"No NIfTI masks found under {src_root}")

    ok = failed = skipped = 0
    for mask_path in tqdm(masks, desc="mask_to_stl", unit="case"):
        row = convert_one(mask_path, args, src_root)
        write_manifest_row(args.manifest, row)
        if row["status"] == "ok":
            ok += 1
        elif row["status"] == "skip_existing":
            skipped += 1
        else:
            failed += 1

    print(f"Converted masks: ok={ok} skipped={skipped} failed={failed}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
