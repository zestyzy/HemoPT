#!/usr/bin/env python3
"""Convert newly unpacked vascular datasets under totaldata/ to Vascular_STL.

This extends the existing Vascular_STL convention:
- one dataset per subdirectory under HemoData/Vascular_STL
- one full vessel surface per STL when possible
- already-surface datasets are copied into the same naming style
- NIfTI masks are converted with marching cubes while preserving multiple
  meaningful connected branches, which is important for coronary and
  intracranial vessels.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from tqdm import tqdm


ROOT = Path(".")
DEFAULT_SRC = ROOT / "totaldata" / "unpacked"
DEFAULT_DST = ROOT / "HemoData" / "Vascular_STL"
DEFAULT_MANIFEST = ROOT / "results" / "new_totaldata_stl" / "stl_manifest.jsonl"


DATASET_CONFIGS = {
    "ImageCAS": {
        "min_component_voxels": 30,
        "min_component_fraction": 1e-4,
        "description": "coronary artery masks; preserve multi-branch components",
    },
    "InterMask": {
        "min_component_voxels": 20,
        "min_component_fraction": 5e-5,
        "description": "intracranial/interventional masks; preserve thin branches",
    },
    "CereVessMRA": {
        "min_component_voxels": 12,
        "min_component_fraction": 5e-5,
        "description": "manual intracranial MRA vessel masks; preserve thin branches",
    },
    "imageTBAD": {
        "min_component_voxels": 80,
        "min_component_fraction": 1e-4,
        "description": "thoracic aorta dissection labels",
    },
    "PARSE2022": {
        "min_component_voxels": 80,
        "min_component_fraction": 1e-4,
        "description": "PARSE 2022 training labels",
    },
    "AVT": {
        "min_component_voxels": 80,
        "min_component_fraction": 1e-4,
        "description": "Aortic Vessel Tree masks if labels are present",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def temp_target(path: Path) -> Path:
    return path.with_name(path.stem + ".tmp" + path.suffix)


def safe_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = path.stem
    return "".join(c if c.isalnum() or c in "._=-" else "_" for c in name)


def load_inter_mapping(mapping_csv: Path) -> dict[str, str]:
    if not mapping_csv.exists():
        return {}
    mapping = {}
    with mapping_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            anon = row.get("anonymous_name", "")
            orig = row.get("original_name", "")
            if anon and orig:
                mapping[anon] = safe_stem(Path(orig))
    return mapping


def crop_foreground(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None, None
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, np.asarray(mask.shape))
    slices = tuple(slice(int(lo[i]), int(hi[i])) for i in range(mask.ndim))
    return mask[slices], lo


def filter_components(
    mask: np.ndarray,
    min_component_voxels: int,
    min_component_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Keep meaningful connected components without collapsing to one branch."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n_components = ndimage.label(mask, structure=structure)
    if n_components == 0:
        return mask, {
            "n_components_raw": 0,
            "n_components_kept": 0,
            "largest_component_voxels": 0,
            "kept_voxels": 0,
        }

    counts = np.bincount(labeled.ravel())
    component_counts = counts[1:]
    largest = int(component_counts.max()) if len(component_counts) else 0
    threshold = max(int(min_component_voxels), int(np.ceil(largest * min_component_fraction)))
    keep_labels = np.where(component_counts >= threshold)[0] + 1

    if keep_labels.size == 0:
        keep_labels = np.array([int(component_counts.argmax()) + 1], dtype=np.int64)

    filtered = np.isin(labeled, keep_labels)
    return filtered, {
        "n_components_raw": int(n_components),
        "n_components_kept": int(keep_labels.size),
        "largest_component_voxels": largest,
        "kept_voxels": int(filtered.sum()),
        "component_threshold_voxels": int(threshold),
    }


def mesh_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    origin_index: np.ndarray,
    step_size: int,
) -> trimesh.Trimesh:
    """Convert a cropped binary mask to a physical-coordinate trimesh surface."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        step_size=step_size,
        allow_degenerate=False,
    )

    verts = verts + origin_index.astype(np.float64) - 1.0
    verts_h = np.c_[verts, np.ones(len(verts), dtype=np.float64)]
    world = (affine @ verts_h.T).T[:, :3]

    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def convert_mask_to_stl(
    src_path: Path,
    dst_path: Path,
    dataset: str,
    min_component_voxels: int,
    min_component_fraction: float,
    step_size: int,
    skip_existing: bool,
) -> dict:
    row = {
        "dataset": dataset,
        "source": str(src_path),
        "target": str(dst_path),
        "status": "fail",
        "reason": "",
    }
    if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1024:
        row["status"] = "skip_existing"
        return row

    try:
        img = nib.load(str(src_path))
        data = np.asanyarray(img.dataobj)
        mask = data > 0
        row["shape"] = list(mask.shape)
        row["zooms"] = [float(x) for x in img.header.get_zooms()[:3]]
        row["raw_voxels"] = int(mask.sum())
        if row["raw_voxels"] == 0:
            row["reason"] = "empty_mask"
            return row

        cropped, origin = crop_foreground(mask)
        if cropped is None:
            row["reason"] = "empty_after_crop"
            return row

        filtered, comp_stats = filter_components(
            cropped,
            min_component_voxels=min_component_voxels,
            min_component_fraction=min_component_fraction,
        )
        row.update(comp_stats)
        if filtered.sum() == 0:
            row["reason"] = "empty_after_component_filter"
            return row

        mesh = mesh_from_mask(filtered, img.affine, origin, step_size=step_size)
        row["n_vertices"] = int(len(mesh.vertices))
        row["n_faces"] = int(len(mesh.faces))
        row["is_watertight"] = bool(mesh.is_watertight)
        if len(mesh.faces) < 20:
            row["reason"] = "too_few_faces"
            return row

        ensure_dir(dst_path.parent)
        tmp_path = temp_target(dst_path)
        if tmp_path.exists():
            tmp_path.unlink()
        mesh.export(tmp_path)
        os.replace(tmp_path, dst_path)
        row["status"] = "ok"
        return row
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["traceback"] = traceback.format_exc(limit=4)
        tmp_path = temp_target(dst_path)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return row


def copy_stl(src_path: Path, dst_path: Path, dataset: str, skip_existing: bool) -> dict:
    row = {
        "dataset": dataset,
        "source": str(src_path),
        "target": str(dst_path),
        "status": "fail",
        "reason": "",
    }
    try:
        if skip_existing and dst_path.exists() and dst_path.stat().st_size > 1024:
            row["status"] = "skip_existing"
            return row
        ensure_dir(dst_path.parent)
        tmp_path = temp_target(dst_path)
        if tmp_path.exists():
            tmp_path.unlink()
        shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        row["status"] = "ok"
        return row
    except Exception as exc:
        row["reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        tmp_path = temp_target(dst_path)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return row


def write_row(manifest_path: Path, row: dict) -> None:
    ensure_dir(manifest_path.parent)
    with manifest_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def iter_imagecas(src_root: Path, dst_root: Path):
    for p in sorted((src_root / "ImageCAS-mask").glob("*.nii.gz"), key=lambda x: safe_stem(x)):
        case = safe_stem(p).replace(".label", "")
        yield p, dst_root / "ImageCAS" / f"ImageCAS_{case}.stl"


def iter_intermask(src_root: Path, dst_root: Path):
    base = src_root / "inter-mask"
    mapping = load_inter_mapping(base / "mapping.csv")
    for p in sorted(base.glob("mask_*.nii.gz"), key=lambda x: x.name):
        orig = mapping.get(p.name, "")
        suffix = f"_{orig}" if orig else ""
        yield p, dst_root / "InterMask" / f"InterMask_{safe_stem(p)}{suffix}.stl"


def iter_cerevess(src_root: Path, dst_root: Path):
    base = src_root / "dataset_2" / "CereVessMRA" / "manual-mra-data" / "gt"
    for p in sorted(base.glob("*.nii.gz"), key=lambda x: x.name):
        yield p, dst_root / "CereVessMRA" / f"CereVessMRA_{safe_stem(p)}.stl"


def iter_imagetbad(src_root: Path, dst_root: Path):
    base = src_root / "dataset_2" / "imageTBAD_data" / "imageTBAD"
    for p in sorted(base.glob("*_label.nii.gz"), key=lambda x: x.name):
        yield p, dst_root / "imageTBAD" / f"imageTBAD_{safe_stem(p).replace('_label', '')}.stl"


def iter_parse2022(src_root: Path, dst_root: Path):
    base = src_root / "dataset_2" / "PARSE 2022" / "train"
    for p in sorted(base.glob("*/label/*.nii.gz"), key=lambda x: str(x)):
        case = p.parents[1].name
        yield p, dst_root / "PARSE2022" / f"PARSE2022_{case}.stl"


def iter_avt_masks(src_root: Path, dst_root: Path):
    base = src_root / "dataset_2" / "Aortic Vessel Tree (AVT) CTA Datasets and Se"
    patterns = ["*label*.nii.gz", "*Label*.nii.gz", "*mask*.nii.gz", "*Mask*.nii.gz", "*seg*.nii.gz", "*Seg*.nii.gz"]
    seen = set()
    for pattern in patterns:
        for p in base.rglob(pattern):
            if p in seen:
                continue
            seen.add(p)
            rel = safe_stem(Path("_".join(p.relative_to(base).parts)))
            yield p, dst_root / "AVT" / f"AVT_{rel}.stl"


def iter_aneumo_stl(src_root: Path, dst_root: Path):
    base = src_root / "dataset_2" / "aneumo"
    for p in sorted(base.glob("*/Stl/*.stl"), key=lambda x: str(x)):
        case = p.parents[1].name
        name = safe_stem(p)
        yield p, dst_root / "aneumo" / f"aneumo_{case}_{name}.stl"


MASK_ITERATORS = {
    "ImageCAS": iter_imagecas,
    "InterMask": iter_intermask,
    "CereVessMRA": iter_cerevess,
    "imageTBAD": iter_imagetbad,
    "PARSE2022": iter_parse2022,
    "AVT": iter_avt_masks,
}

COPY_ITERATORS = {
    "aneumo": iter_aneumo_stl,
}


def process_mask_dataset(args, dataset: str) -> tuple[int, int, int]:
    cfg = DATASET_CONFIGS[dataset]
    pairs = list(MASK_ITERATORS[dataset](args.src_root, args.dst_root))
    if args.max_cases:
        pairs = pairs[: args.max_cases]

    ok = failed = skipped = 0
    progress = tqdm(pairs, desc=dataset, unit="case")
    for src_path, dst_path in progress:
        row = convert_mask_to_stl(
            src_path=src_path,
            dst_path=dst_path,
            dataset=dataset,
            min_component_voxels=args.min_component_voxels or cfg["min_component_voxels"],
            min_component_fraction=args.min_component_fraction if args.min_component_fraction is not None else cfg["min_component_fraction"],
            step_size=args.step_size,
            skip_existing=args.skip_existing,
        )
        write_row(args.manifest, row)
        if row["status"] == "ok":
            ok += 1
        elif row["status"] == "skip_existing":
            skipped += 1
        else:
            failed += 1
        progress.set_postfix(ok=ok, skip=skipped, fail=failed)
    print(f"[{dataset}] ok={ok} skipped={skipped} failed={failed} total={len(pairs)}")
    return ok, skipped, failed


def process_copy_dataset(args, dataset: str) -> tuple[int, int, int]:
    pairs = list(COPY_ITERATORS[dataset](args.src_root, args.dst_root))
    if args.max_cases:
        pairs = pairs[: args.max_cases]

    ok = failed = skipped = 0
    progress = tqdm(pairs, desc=dataset, unit="mesh")
    for src_path, dst_path in progress:
        row = copy_stl(src_path, dst_path, dataset=dataset, skip_existing=args.skip_existing)
        write_row(args.manifest, row)
        if row["status"] == "ok":
            ok += 1
        elif row["status"] == "skip_existing":
            skipped += 1
        else:
            failed += 1
        progress.set_postfix(ok=ok, skip=skipped, fail=failed)
    print(f"[{dataset}] ok={ok} skipped={skipped} failed={failed} total={len(pairs)}")
    return ok, skipped, failed


def parse_args():
    parser = argparse.ArgumentParser(description="Convert newly unpacked totaldata vascular masks/meshes to Vascular_STL")
    parser.add_argument("--src_root", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst_root", type=Path, default=DEFAULT_DST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ImageCAS", "InterMask", "CereVessMRA", "imageTBAD", "PARSE2022", "AVT", "aneumo"],
        choices=list(MASK_ITERATORS.keys()) + list(COPY_ITERATORS.keys()),
    )
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_false", dest="skip_existing")
    parser.add_argument("--max_cases", type=int, default=0, help="Smoke-test limit per dataset; 0 means all")
    parser.add_argument("--step_size", type=int, default=1, help="Marching-cubes step size; keep 1 for thin vessels")
    parser.add_argument("--min_component_voxels", type=int, default=0, help="Override per-dataset component voxel threshold")
    parser.add_argument("--min_component_fraction", type=float, default=None, help="Override per-dataset component fraction threshold")
    parser.add_argument("--reset_manifest", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.dst_root)
    ensure_dir(args.manifest.parent)
    if args.reset_manifest and args.manifest.exists():
        args.manifest.unlink()

    totals = {"ok": 0, "skipped": 0, "failed": 0}
    print(f"Source: {args.src_root}")
    print(f"Output: {args.dst_root}")
    print(f"Manifest: {args.manifest}")
    print(f"Datasets: {' '.join(args.datasets)}")

    for dataset in args.datasets:
        if dataset in MASK_ITERATORS:
            ok, skipped, failed = process_mask_dataset(args, dataset)
        else:
            ok, skipped, failed = process_copy_dataset(args, dataset)
        totals["ok"] += ok
        totals["skipped"] += skipped
        totals["failed"] += failed

    print("=" * 60)
    print(f"Total ok={totals['ok']} skipped={totals['skipped']} failed={totals['failed']}")


if __name__ == "__main__":
    main()
