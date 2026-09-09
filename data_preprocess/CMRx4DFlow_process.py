"""CMRx4DFlow full-kspace data -> HemoPT velocity point clouds.

Output format matches the existing VMRCFD/Hemo CFD fine-tuning loader:

  x_i.npy    (N, 7) = [x, y, z, dist_to_wall, dir_to_wall_x, dir_to_wall_y, dir_to_wall_z]
  y_i.npy    (N, 4) = [0, u, v, w]
  cond_i.npy (12,)  = VMR-style prompt metadata, with velocity-scale entries filled

The pressure channel is a dummy zero so that downstream runs can use
--hemo_target velocity and train only on [u, v, w].
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import time
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import fft, ndimage
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


COND_SCHEMA = [
    "territory_family_norm",
    "speed_mean",
    "speed_rms",
    "speed_peak",
    "speed_pulsatility",
    "viscosity_rel",
    "density_rel",
    "rcr_count_norm",
    "rcr_resistance_proxy",
    "rcr_capacitance_proxy",
    "raw_length_log",
    "frame_index_norm",
]

CANONICAL_POSITIVE = {
    "LR": ("x", 1.0),
    "RL": ("x", -1.0),
    "AP": ("y", 1.0),
    "PA": ("y", -1.0),
    "FH": ("z", 1.0),
    "HF": ("z", -1.0),
    "SI": ("z", 1.0),
    "IS": ("z", -1.0),
}
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _split_semicolon(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(";") if x.strip()]


def read_params_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"Empty params.csv: {path}")

    out: dict[str, Any] = {}
    for key, value in row.items():
        text = "" if value is None else str(value).strip()
        if text == "":
            out[key] = None
        elif key in {"FOV", "resolution", "VENC"}:
            out[key] = [float(x) for x in _split_semicolon(text)]
        elif key == "matrix_size":
            out[key] = [int(float(x)) for x in _split_semicolon(text)]
        elif key in {"spatial_order", "venc_order", "VENC_order"}:
            out[key] = _split_semicolon(text)
        elif key in {"RR", "FA", "TE", "TR", "field_strength"}:
            out[key] = float(text)
        else:
            out[key] = text
    return out


def _read_mat_array(path: Path, key: str, idx=()) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{key!r} not found in {path}; keys={list(f.keys())}")
        arr = f[key][idx]
    if arr.dtype.fields is not None and "real" in arr.dtype.fields and "imag" in arr.dtype.fields:
        arr = arr["real"] + 1j * arr["imag"]
    return arr


def load_coo_npz(path: Path, as_dense: bool = True):
    z = np.load(path)
    coords = z["coords"]
    data = z["data"]
    shape = tuple(int(x) for x in z["shape"])
    if not as_dense:
        return coords, data, shape
    out = np.zeros(shape, dtype=data.dtype)
    if coords.size:
        out[tuple(coords.T)] = data
    return out


def choose_frame_index(nt: int, strategy: str, frame_index: int, phase_fraction: float) -> int:
    if strategy == "fixed":
        return max(0, min(int(frame_index), nt - 1))
    if strategy == "middle":
        return nt // 2
    if strategy == "phase":
        phase = max(0.0, min(float(phase_fraction), 1.0))
        return max(0, min(int(round(phase * max(nt - 1, 0))), nt - 1))
    raise ValueError(f"Unsupported non-peak frame strategy: {strategy}")


def load_cached_frame(
    path: Path,
    frame_index: int,
    strategy: str = "fixed",
    phase_fraction: float = 0.5,
) -> tuple[np.ndarray, int, int]:
    """Load one time frame from a COO-compressed img_gt.npz cache.

    The cache stores complex image data as (Nv, Nt, SPE, PE, FE). Loading only
    the requested frame keeps memory lower than densifying all cardiac phases.
    """

    coords, data, shape = load_coo_npz(path, as_dense=False)
    if len(shape) != 5:
        raise ValueError(f"Expected cached image shape (Nv,Nt,SPE,PE,FE), got {shape} from {path}")
    nv, nt, spe, pe, fe = shape
    frame = choose_frame_index(nt, strategy, frame_index, phase_fraction)
    if coords.size == 0:
        return np.zeros((nv, spe, pe, fe), dtype=data.dtype), nt, frame
    keep = coords[:, 1] == frame
    frame_coords = coords[keep]
    frame_data = data[keep]
    out = np.zeros((nv, spe, pe, fe), dtype=data.dtype)
    if frame_coords.size:
        out[(frame_coords[:, 0], frame_coords[:, 2], frame_coords[:, 3], frame_coords[:, 4])] = frame_data
    return out, nt, frame


def _ifftn_ortho(kspace: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    return fft.fftshift(
        fft.ifftn(
            fft.ifftshift(kspace, axes=axes),
            axes=axes,
            norm="ortho",
            workers=-1,
        ),
        axes=axes,
    )


def kspace_frame_to_image(kspace_frame: np.ndarray, coilmap: np.ndarray) -> np.ndarray:
    """Return coil-combined image for one time frame.

    kspace_frame: (Nv, Nc, SPE, PE, FE)
    coilmap:      (Nc, SPE, PE, FE)
    returns:      (Nv, SPE, PE, FE)
    """

    coil_images = _ifftn_ortho(kspace_frame, axes=(-1, -2, -3))
    return np.sum(coil_images * np.conj(coilmap)[None, ...], axis=1).astype(np.complex64)


def velocity_from_image(img: np.ndarray, venc: np.ndarray) -> np.ndarray:
    """Return velocity in VENC units, shape (3, SPE, PE, FE)."""

    phase = np.angle(img[1:] * np.conj(img[0:1])).astype(np.float32)
    scale_shape = (phase.shape[0],) + (1,) * (phase.ndim - 1)
    return (phase / np.pi * venc[: phase.shape[0]].reshape(scale_shape)).astype(np.float32)


def complex_from_split_ri(ri: np.ndarray, nv: int | None = None) -> np.ndarray:
    """Convert FVUNet split real/imag channels back to complex image channels."""

    if ri.ndim != 4:
        raise ValueError(f"Expected split RI shape (2*Nv,SPE,PE,FE), got {ri.shape}")
    if nv is None:
        if ri.shape[0] % 2 != 0:
            raise ValueError(f"Expected an even number of RI channels, got {ri.shape[0]}")
        nv = ri.shape[0] // 2
    if ri.shape[0] < 2 * nv:
        raise ValueError(f"RI channels {ri.shape[0]} are insufficient for Nv={nv}")
    return (ri[:nv].astype(np.float32, copy=False) + 1j * ri[nv:2 * nv].astype(np.float32, copy=False)).astype(np.complex64)


def _read_npy_header_at(file_obj, offset: int) -> tuple[tuple[int, ...], np.dtype, bool, int]:
    file_obj.seek(offset)
    version = np.lib.format.read_magic(file_obj)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file_obj)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file_obj)
    else:
        shape, fortran_order, dtype = np.lib.format._read_array_header(file_obj, version)
    return tuple(shape), np.dtype(dtype), bool(fortran_order), file_obj.tell()


def memmap_stored_npz_member(path: Path, key: str) -> np.ndarray | None:
    """Memory-map an uncompressed .npy member inside an .npz file."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    try:
        with zipfile.ZipFile(path) as zf:
            info = zf.getinfo(member)
            if info.compress_type != zipfile.ZIP_STORED:
                return None
            header_offset = info.header_offset

        with path.open("rb") as f:
            f.seek(header_offset)
            header = f.read(30)
            fields = struct.unpack("<4s5H3I2H", header)
            signature, _, _, compression, _, _, _, _, _, name_len, extra_len = fields
            if signature != b"PK\x03\x04" or compression != zipfile.ZIP_STORED:
                return None
            npy_offset = header_offset + 30 + name_len + extra_len
            shape, dtype, fortran_order, data_offset = _read_npy_header_at(f, npy_offset)
        order = "F" if fortran_order else "C"
        return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=data_offset, order=order)
    except Exception:
        return None


def load_single_frame_cache(path: Path) -> tuple[np.ndarray, int, int]:
    """Load one FVUNet prebuilt single-frame cache sample.

    These caches store target_ri as split real/imag channels for one cardiac
    frame, so they avoid decompressing the much larger full-cycle COO img_gt.npz.
    """

    target_ri = memmap_stored_npz_member(path, "target_ri")
    with np.load(path) as z:
        if target_ri is None:
            target_ri = z["target_ri"]
        nv = int(z["Nv"]) if "Nv" in z.files else target_ri.shape[0] // 2
        nt = int(z["Nt"]) if "Nt" in z.files else 0
        frame = int(z["t"]) if "t" in z.files else 0
    return complex_from_split_ri(target_ri, nv=nv), nt, frame


def single_frame_cache_meta(path: Path) -> tuple[int, int, int]:
    if path.is_dir():
        meta_path = path / "meta.npz"
        target_path = path / "target_ri.npy"
        if not meta_path.is_file():
            siblings = sorted(path.parent.parent.glob(f"R*/{path.name}/meta.npz"))
            meta_path = siblings[0] if siblings else meta_path
        if meta_path.is_file():
            with np.load(meta_path) as z:
                nv = int(z["Nv"]) if "Nv" in z.files else int(np.load(target_path, mmap_mode="r").shape[0] // 2)
                nt = int(z["Nt"]) if "Nt" in z.files else 0
                frame = int(z["t"]) if "t" in z.files else int(path.name.lstrip("t"))
            return nv, nt, frame
        target_ri = np.load(target_path, mmap_mode="r")
        return int(target_ri.shape[0] // 2), 0, int(path.name.lstrip("t"))

    with np.load(path) as z:
        nv = int(z["Nv"]) if "Nv" in z.files else int(z["target_ri"].shape[0] // 2)
        nt = int(z["Nt"]) if "Nt" in z.files else 0
        frame = int(z["t"]) if "t" in z.files else 0
    return nv, nt, frame


def canonical_velocity_samples(velocity: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Map encoded velocity samples to canonical [x, y, z], input shape (C, N)."""

    venc_order = params.get("venc_order", params.get("VENC_order"))
    if venc_order is None or len(venc_order) < velocity.shape[0]:
        raise ValueError("params.csv must provide venc_order/VENC_order with at least three entries")

    out = np.zeros((velocity.shape[1], 3), dtype=np.float32)
    for comp, label in enumerate(venc_order[: velocity.shape[0]]):
        axis, sign = _label_to_axis_sign(label)
        out[:, axis] = velocity[comp] * sign
    return out


def velocity_samples_from_single_frame_cache(
    path: Path,
    flat_indices: np.ndarray,
    mask_shape: tuple[int, int, int],
    params: dict[str, Any],
    velocity_unit_scale: float,
) -> np.ndarray:
    if path.is_dir():
        target_ri = np.load(path / "target_ri.npy", mmap_mode="r")
        nv, _nt, _frame = single_frame_cache_meta(path)
    else:
        target_ri = memmap_stored_npz_member(path, "target_ri")
        with np.load(path) as z:
            if target_ri is None:
                target_ri = z["target_ri"]
            nv = int(z["Nv"]) if "Nv" in z.files else target_ri.shape[0] // 2

    if tuple(target_ri.shape[1:]) != tuple(mask_shape):
        raise ValueError(
            f"Single-frame cache shape {target_ri.shape[1:]} does not match mask shape {mask_shape}: {path}"
        )

    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    order = np.argsort(flat_indices)
    sorted_indices = flat_indices[order]
    channels = target_ri.reshape(target_ri.shape[0], -1)
    sampled_sorted = np.asarray(channels[:, sorted_indices], dtype=np.float32)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    sampled = sampled_sorted[:, inverse]

    real = sampled[:nv]
    imag = sampled[nv:2 * nv]
    cross_real = real[1:] * real[:1] + imag[1:] * imag[:1]
    cross_imag = imag[1:] * real[:1] - real[1:] * imag[:1]
    phase = np.arctan2(cross_imag, cross_real).astype(np.float32)
    venc = np.asarray(params["VENC"], dtype=np.float32)
    scale_shape = (phase.shape[0], 1)
    velocity = phase / np.pi * venc[: phase.shape[0]].reshape(scale_shape) * velocity_unit_scale
    return canonical_velocity_samples(velocity.astype(np.float32, copy=False), params)


def _label_to_axis_sign(label: str) -> tuple[int, float]:
    text = label.strip().upper()
    if text not in CANONICAL_POSITIVE:
        raise ValueError(f"Unsupported anatomical direction label {label!r}")
    axis, sign = CANONICAL_POSITIVE[text]
    return AXIS_TO_INDEX[axis], sign


def canonical_grid(mask_shape: tuple[int, int, int], params: dict[str, Any]) -> np.ndarray:
    """Build canonical physical coordinates for segmask voxels.

    segmask order is (SPE, PE, FE). params['resolution'] and spatial_order are
    listed in (FE, PE, SPE), so both are reversed for array-axis alignment.
    """

    resolution = params.get("resolution")
    spatial_order = params.get("spatial_order")
    if resolution is None or spatial_order is None or len(resolution) < 3 or len(spatial_order) < 3:
        raise ValueError("params.csv must provide resolution and spatial_order with three entries")

    array_spacings = [float(resolution[2]), float(resolution[1]), float(resolution[0])]
    array_labels = [spatial_order[2], spatial_order[1], spatial_order[0]]

    axes = [(np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * spacing
            for n, spacing in zip(mask_shape, array_spacings)]
    mesh = np.meshgrid(*axes, indexing="ij")

    coords = np.zeros(mask_shape + (3,), dtype=np.float32)
    for dim, label in enumerate(array_labels):
        axis, sign = _label_to_axis_sign(label)
        coords[..., axis] = mesh[dim] * sign
    return coords


def canonical_coords_from_indices(
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> np.ndarray:
    """Build canonical physical coordinates only for selected array indices."""

    resolution = params.get("resolution")
    spatial_order = params.get("spatial_order")
    if resolution is None or spatial_order is None or len(resolution) < 3 or len(spatial_order) < 3:
        raise ValueError("params.csv must provide resolution and spatial_order with three entries")

    array_spacings = [float(resolution[2]), float(resolution[1]), float(resolution[0])]
    array_labels = [spatial_order[2], spatial_order[1], spatial_order[0]]
    n_points = int(np.asarray(indices[0]).size)
    coords = np.zeros((n_points, 3), dtype=np.float32)

    for dim, label in enumerate(array_labels):
        axis, sign = _label_to_axis_sign(label)
        vals = (np.asarray(indices[dim], dtype=np.float32) - (mask_shape[dim] - 1) / 2.0) * array_spacings[dim]
        coords[:, axis] = vals * sign
    return coords


def canonical_velocity(velocity: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Map velocity encodings to canonical [x, y, z] components."""

    venc_order = params.get("venc_order", params.get("VENC_order"))
    if venc_order is None or len(venc_order) < velocity.shape[0]:
        raise ValueError("params.csv must provide venc_order/VENC_order with at least three entries")

    out = np.zeros(velocity.shape[1:] + (3,), dtype=np.float32)
    for comp, label in enumerate(venc_order[: velocity.shape[0]]):
        axis, sign = _label_to_axis_sign(label)
        out[..., axis] = velocity[comp] * sign
    return out


def geometry_normalization_stats(points: np.ndarray, target_length: float) -> tuple[float, np.ndarray]:
    bound_min = points.min(axis=0)
    bound_max = points.max(axis=0)
    length = float(np.max(bound_max - bound_min))
    if length < 1e-12:
        raise RuntimeError("Degenerate 4DFlow mask geometry.")
    scale = float(target_length) / length
    center = (points * scale).mean(axis=0)
    return scale, center.astype(np.float32)


def mask_wall_features(mask: np.ndarray, coords: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    resolution = params["resolution"]
    sampling = [float(resolution[2]), float(resolution[1]), float(resolution[0])]
    dist, nearest = ndimage.distance_transform_edt(mask, sampling=sampling, return_indices=True)

    nearest_coords = coords[nearest[0], nearest[1], nearest[2]]
    diff = nearest_coords - coords
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    direction = np.divide(diff, norm, out=np.zeros_like(diff), where=norm > 1e-8)
    return dist.astype(np.float32), direction.astype(np.float32)


def sampled_wall_features(
    mask: np.ndarray,
    sample_points: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wall distance/direction only for sampled mask points."""

    outside_shell = ndimage.binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool)) & ~mask
    wall_points = canonical_coords_from_indices(np.nonzero(outside_shell), mask.shape, params)
    if wall_points.size == 0:
        return (
            np.zeros(sample_points.shape[0], dtype=np.float32),
            np.zeros((sample_points.shape[0], 3), dtype=np.float32),
        )

    tree = cKDTree(wall_points.reshape(-1, 3))
    try:
        dist, nn = tree.query(sample_points, k=1, workers=-1)
    except TypeError:
        dist, nn = tree.query(sample_points, k=1)
    nearest = wall_points.reshape(-1, 3)[nn]
    diff = nearest - sample_points
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    direction = np.divide(diff, norm, out=np.zeros_like(diff), where=norm > 1e-8)
    return dist.astype(np.float32), direction.astype(np.float32)


def discover_cases(root: Path, limit: int = 0) -> list[Path]:
    cases = []
    for kpath in root.rglob("kdata_full.mat"):
        case_dir = kpath.parent
        required = ["coilmap.mat", "segmask.mat", "params.csv"]
        if all((case_dir / name).is_file() for name in required):
            cases.append(case_dir)
            if limit > 0 and len(cases) >= limit:
                break
    return sorted(cases)


def select_frame(
    case_dir: Path,
    coilmap: np.ndarray,
    mask: np.ndarray,
    params: dict[str, Any],
    strategy: str,
    frame_index: int,
    phase_fraction: float,
    velocity_unit_scale: float,
) -> tuple[int, dict[str, float]]:
    with h5py.File(case_dir / "kdata_full.mat", "r") as f:
        nt = int(f["kdata_full"].shape[1])

    if strategy in {"fixed", "middle", "phase"}:
        frame = choose_frame_index(nt, strategy, frame_index, phase_fraction)
        img = reconstruct_frame(case_dir, coilmap, frame)
        vel = velocity_from_image(img, np.asarray(params["VENC"], dtype=np.float32)) * velocity_unit_scale
        vel = canonical_velocity(vel, params)
        speed = np.linalg.norm(vel[mask > 0], axis=-1)
        return frame, speed_stats(speed, nt, frame)

    if strategy != "peak":
        raise ValueError(f"Unsupported frame strategy: {strategy}")

    best_frame = 0
    best_mean = -np.inf
    best_stats: dict[str, float] = {}
    venc = np.asarray(params["VENC"], dtype=np.float32)
    for t in range(nt):
        img = reconstruct_frame(case_dir, coilmap, t)
        vel = canonical_velocity(velocity_from_image(img, venc) * velocity_unit_scale, params)
        speed = np.linalg.norm(vel[mask > 0], axis=-1)
        stats = speed_stats(speed, nt, t)
        if stats["speed_mean"] > best_mean:
            best_mean = stats["speed_mean"]
            best_frame = t
            best_stats = stats
    return best_frame, best_stats


def reconstruct_frame(case_dir: Path, coilmap: np.ndarray, frame_index: int) -> np.ndarray:
    kspace = _read_mat_array(
        case_dir / "kdata_full.mat",
        "kdata_full",
        idx=(slice(None), slice(frame_index, frame_index + 1), slice(None), slice(None), slice(None), slice(None)),
    )
    kspace = np.asarray(kspace[:, 0], dtype=np.complex64)
    return kspace_frame_to_image(kspace, coilmap)


def cache_path_for_case(case_dir: Path, data_root: Path | None, cache_root: Path | None) -> Path | None:
    if cache_root is None:
        return None
    try:
        rel = case_dir.relative_to(data_root) if data_root is not None else case_dir.name
    except ValueError:
        rel = Path(*case_dir.parts[-4:])
    path = cache_root / rel / "img_gt.npz"
    return path if path.is_file() else None


def _parse_single_frame_cache(path: Path) -> tuple[int, int] | None:
    try:
        if path.parent.name == "target":
            r = 0
        else:
            r = int(path.parent.name.lstrip("R"))
        t = int(path.stem.lstrip("t"))
    except ValueError:
        return None
    return r, t


def single_frame_cache_path_for_case(
    case_dir: Path,
    data_root: Path | None,
    cache_root: Path | None,
    frame_index: int | None,
    prefer_r: int,
    phase_fraction: float = 0.5,
) -> Path | None:
    if cache_root is None:
        return None
    try:
        rel = case_dir.relative_to(data_root) if data_root is not None else case_dir.name
    except ValueError:
        rel = Path(*case_dir.parts[-4:])

    case_cache = cache_root / rel
    if not case_cache.is_dir():
        return None

    parsed_candidates: list[tuple[int, int, Path]] = []
    for path in case_cache.glob("R*/t*.npz"):
        parsed = _parse_single_frame_cache(path)
        if parsed is None:
            continue
        r, t = parsed
        parsed_candidates.append((r, t, path))
    for path in case_cache.glob("target/t*"):
        if not path.is_dir() or not (path / "target_ri.npy").is_file():
            continue
        parsed = _parse_single_frame_cache(path)
        if parsed is None:
            continue
        r, t = parsed
        parsed_candidates.append((r, t, path))
    if not parsed_candidates:
        return None
    if frame_index is None:
        times = sorted({item[1] for item in parsed_candidates})
        phase = max(0.0, min(float(phase_fraction), 1.0))
        frame_index = times[int(round(phase * max(len(times) - 1, 0)))]
    candidates = [
        (abs(t - frame_index), 0 if r == 0 else abs(r - prefer_r) + 1, path)
        for r, t, path in parsed_candidates
    ]
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[0][2]


def speed_stats(speed: np.ndarray, nt: int, frame_index: int) -> dict[str, float]:
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        return {
            "speed_mean": 0.0,
            "speed_rms": 0.0,
            "speed_peak": 0.0,
            "speed_pulsatility": 0.0,
            "frame_index_norm": float(frame_index) / max(nt - 1, 1),
        }
    mean = float(speed.mean())
    rms = float(np.sqrt(np.mean(speed ** 2)))
    peak = float(np.percentile(speed, 95))
    pulsatility = float((peak - mean) / (peak + 1e-8))
    return {
        "speed_mean": mean,
        "speed_rms": rms,
        "speed_peak": peak,
        "speed_pulsatility": max(0.0, min(1.0, pulsatility)),
        "frame_index_norm": float(frame_index) / max(nt - 1, 1),
    }


def build_cond(stats: dict[str, float], raw_length: float) -> np.ndarray:
    cond = np.zeros(12, dtype=np.float32)
    cond[0] = 1.0  # Aorta-like vascular territory.
    cond[1] = stats["speed_mean"]
    cond[2] = stats["speed_rms"]
    cond[3] = stats["speed_peak"]
    cond[4] = stats["speed_pulsatility"]
    cond[5] = 1.0
    cond[6] = 1.0
    cond[7] = 0.0
    cond[8] = 1.0
    cond[9] = 0.0
    cond[10] = float(np.log1p(max(raw_length, 0.0)))
    cond[11] = stats["frame_index_norm"]
    return cond


def split_indices(ids: list[int], train_frac: float, val_frac: float, seed: int) -> dict[str, list[int]]:
    n = len(ids)
    rng = np.random.default_rng(seed)
    ids_arr = np.asarray(ids, dtype=np.int64)
    perm = rng.permutation(ids_arr)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 1 if n >= 3 else n), n)
    n_val = min(max(n_val, 1 if n - n_train >= 2 else 0), max(n - n_train, 0))
    train = sorted(perm[:n_train].astype(int).tolist())
    val = sorted(perm[n_train:n_train + n_val].astype(int).tolist())
    test = sorted(perm[n_train + n_val:].astype(int).tolist())
    if not test and val:
        test = [val.pop()]
    return {"train_indices": train, "val_indices": val, "test_indices": test}


def parse_case_metadata(case_dir: Path) -> dict[str, str]:
    parts = case_dir.parts
    if len(parts) >= 4:
        return {
            "territory": parts[-4],
            "center": parts[-3],
            "scanner": parts[-2],
            "case_id": parts[-1],
        }
    return {
        "territory": "",
        "center": "",
        "scanner": "",
        "case_id": case_dir.name,
    }


def assign_speed_bins(manifest: list[dict[str, Any]], n_bins: int) -> list[float]:
    speeds = np.asarray([float(item.get("speed_mean", 0.0)) for item in manifest], dtype=np.float64)
    if speeds.size == 0 or n_bins <= 1 or np.nanmax(speeds) <= np.nanmin(speeds):
        for item in manifest:
            item["speed_bin"] = 0
        return [float(np.nanmin(speeds)) if speeds.size else 0.0, float(np.nanmax(speeds)) if speeds.size else 0.0]

    edges = np.quantile(speeds, np.linspace(0.0, 1.0, n_bins + 1))
    inner_edges = np.unique(edges[1:-1])
    bins = np.searchsorted(inner_edges, speeds, side="right")
    for item, bin_id in zip(manifest, bins):
        item["speed_bin"] = int(bin_id)
    return [float(x) for x in edges]


def _target_split_counts(n: int, train_frac: float, val_frac: float) -> dict[str, int]:
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 1 if n >= 3 else n), n)
    n_val = min(max(n_val, 1 if n - n_train >= 2 else 0), max(n - n_train, 0))
    n_test = n - n_train - n_val
    if n >= 3 and n_test == 0 and n_val > 0:
        n_val -= 1
        n_test = 1
    return {"train": n_train, "val": n_val, "test": n_test}


def split_indices_stratified(
    manifest: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, Any]:
    n = len(manifest)
    targets = _target_split_counts(n, train_frac, val_frac)
    rng = np.random.default_rng(seed)

    groups: dict[str, list[int]] = {}
    for item in manifest:
        key = f"{item.get('center', '')}|{item.get('scanner', '')}|speed{item.get('speed_bin', 0)}"
        groups.setdefault(key, []).append(int(item["index"]))

    assignments = {"train": [], "val": [], "test": []}
    current = {"train": 0, "val": 0, "test": 0}
    split_fracs = {
        "train": float(train_frac),
        "val": float(val_frac),
        "test": max(0.0, 1.0 - float(train_frac) - float(val_frac)),
    }

    for key, group_ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ids = np.asarray(group_ids, dtype=np.int64)
        ids = rng.permutation(ids).astype(int).tolist()
        local = {"train": 0, "val": 0, "test": 0}
        desired = {name: len(ids) * split_fracs[name] for name in local}
        local_assignment = {"train": [], "val": [], "test": []}

        for case_index in ids:
            available = [name for name in ("train", "val", "test") if current[name] < targets[name]]
            if not available:
                available = ["train", "val", "test"]
            split = max(
                available,
                key=lambda name: (
                    desired[name] - local[name],
                    (targets[name] - current[name]) / max(targets[name], 1),
                    1 if name == "train" else 0,
                ),
            )
            local_assignment[split].append(case_index)
            local[split] += 1
            current[split] += 1

        for name in assignments:
            assignments[name].extend(local_assignment[name])

    return {
        "train_indices": sorted(assignments["train"]),
        "val_indices": sorted(assignments["val"]),
        "test_indices": sorted(assignments["test"]),
        "split_strategy": "stratified",
        "stratify_fields": ["center", "scanner", "speed_bin"],
    }


def split_diagnostics(manifest: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    by_index = {int(item["index"]): item for item in manifest}

    def counts_for(indices: list[int], field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for idx in indices:
            item = by_index[int(idx)]
            value = str(item.get(field, ""))
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    def center_scanner_counts(indices: list[int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for idx in indices:
            item = by_index[int(idx)]
            value = f"{item.get('center', '')}|{item.get('scanner', '')}"
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    diagnostics = {}
    for split_name, key in (("train", "train_indices"), ("val", "val_indices"), ("test", "test_indices")):
        indices = [int(x) for x in split.get(key, [])]
        speeds = np.asarray([float(by_index[idx].get("speed_mean", 0.0)) for idx in indices], dtype=np.float64)
        diagnostics[split_name] = {
            "n": len(indices),
            "center_scanner": center_scanner_counts(indices),
            "speed_bin": counts_for(indices, "speed_bin"),
            "speed_mean_avg": float(speeds.mean()) if speeds.size else 0.0,
            "speed_mean_min": float(speeds.min()) if speeds.size else 0.0,
            "speed_mean_max": float(speeds.max()) if speeds.size else 0.0,
        }
    return diagnostics


def process_case(case_dir: Path, out_dir: Path, index: int, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    def stage(name: str) -> None:
        if getattr(args, "verbose_case", False):
            print(f"[case {index:04d}] {name} elapsed={time.time() - t0:.1f}s case={case_dir}", flush=True)

    stage("start")
    params = read_params_csv(case_dir / "params.csv")
    if params.get("VENC") is None:
        raise ValueError(f"Missing VENC in {case_dir / 'params.csv'}")

    stage("read mask/coil")
    mask = _read_mat_array(case_dir / "segmask.mat", "segmask").astype(bool)
    if mask.sum() == 0:
        raise RuntimeError("Empty segmask")

    cache_path = cache_path_for_case(
        case_dir,
        Path(args.data_root).resolve() if getattr(args, "data_root", None) else None,
        Path(args.image_cache_root).resolve() if getattr(args, "image_cache_root", "") else None,
    )
    single_cache_path = single_frame_cache_path_for_case(
        case_dir,
        Path(args.data_root).resolve() if getattr(args, "data_root", None) else None,
        Path(args.single_frame_cache_root).resolve() if getattr(args, "single_frame_cache_root", "") else None,
        None if args.frame_strategy in {"middle", "phase"} else args.frame_index,
        args.single_frame_cache_r,
        args.phase_fraction,
    ) if args.frame_strategy in ("fixed", "middle", "phase") else None
    coilmap = None if (single_cache_path is not None or cache_path is not None) else _read_mat_array(
        case_dir / "coilmap.mat", "coilmap").astype(np.complex64)
    image_source = str(single_cache_path or cache_path or (case_dir / "kdata_full.mat"))

    velocity = None
    sampled_velocity = None
    if args.frame_strategy in ("fixed", "middle", "phase"):
        if single_cache_path is not None:
            stage(f"read single-frame cache meta strategy={args.frame_strategy}")
            _nv, nt, frame = single_frame_cache_meta(single_cache_path)
        elif cache_path is not None:
            stage(f"load cached frame strategy={args.frame_strategy}")
            img, nt, frame = load_cached_frame(
                cache_path,
                args.frame_index,
                strategy=args.frame_strategy,
                phase_fraction=args.phase_fraction,
            )
        else:
            with h5py.File(case_dir / "kdata_full.mat", "r") as f:
                nt = int(f["kdata_full"].shape[1])
            frame = choose_frame_index(nt, args.frame_strategy, args.frame_index, args.phase_fraction)
            stage(f"reconstruct frame={frame}")
            img = reconstruct_frame(case_dir, coilmap, frame)
        if single_cache_path is None:
            stage("velocity convert")
            velocity = velocity_from_image(img, np.asarray(params["VENC"], dtype=np.float32)) * args.velocity_unit_scale
            velocity = canonical_velocity(velocity, params)
            stats = speed_stats(np.linalg.norm(velocity[mask > 0], axis=-1), nt, frame)
        else:
            stats = {
                "speed_mean": 0.0,
                "speed_rms": 0.0,
                "speed_peak": 0.0,
                "speed_pulsatility": 0.0,
                "frame_index_norm": float(frame) / max(nt - 1, 1),
            }
    else:
        if cache_path is not None:
            raise ValueError("--frame_strategy peak is not supported with --image_cache_root; use fixed or middle")
        stage("select peak frame")
        frame, stats = select_frame(
            case_dir,
            coilmap,
            mask,
            params,
            args.frame_strategy,
            args.frame_index,
            args.phase_fraction,
            args.velocity_unit_scale,
        )
        stage(f"reconstruct selected frame={frame}")
        img = reconstruct_frame(case_dir, coilmap, frame)
        stage("velocity convert")
        velocity = velocity_from_image(img, np.asarray(params["VENC"], dtype=np.float32)) * args.velocity_unit_scale
        velocity = canonical_velocity(velocity, params)

    stage("sample points")
    mask_indices = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(args.seed + index)
    replace = mask_indices.size < args.n_points
    chosen = rng.choice(mask_indices, size=args.n_points, replace=replace)
    selected = np.unravel_index(chosen, mask.shape)

    stage("build sampled coords")
    raw_points = canonical_coords_from_indices(selected, mask.shape, params)

    stage("sample wall features")
    wall_dist, wall_dir = sampled_wall_features(mask, raw_points, params)

    if single_cache_path is not None:
        stage("sample velocity from single-frame cache")
        sampled_velocity = velocity_samples_from_single_frame_cache(
            single_cache_path, chosen, mask.shape, params, args.velocity_unit_scale)

    raw_length = float(np.max(raw_points.max(axis=0) - raw_points.min(axis=0)))
    scale, center = geometry_normalization_stats(raw_points, args.target_length)
    points = raw_points * scale - center

    geom = np.concatenate([
        wall_dist[:, None] * scale,
        wall_dir,
    ], axis=1).astype(np.float32)

    if sampled_velocity is not None:
        vel = sampled_velocity.astype(np.float32, copy=False)
        stats = speed_stats(np.linalg.norm(vel, axis=-1), nt, frame)
    else:
        vel = velocity[selected].astype(np.float32)
    if args.clip_speed_percentile > 0:
        speed = np.linalg.norm(vel, axis=1)
        thr = float(np.percentile(speed, args.clip_speed_percentile))
        scale_down = np.minimum(1.0, thr / (speed + 1e-8))
        vel = vel * scale_down[:, None]

    x_out = np.concatenate([points, geom], axis=1).astype(args.dtype)
    y_out = np.concatenate([np.zeros((args.n_points, 1), dtype=np.float32), vel], axis=1).astype(args.dtype)
    cond = build_cond(stats, raw_length).astype(args.dtype)

    np.save(out_dir / f"x_{index}.npy", x_out)
    np.save(out_dir / f"y_{index}.npy", y_out)
    np.save(out_dir / f"cond_{index}.npy", cond)
    stage("done")

    speed = np.linalg.norm(vel, axis=1)
    case_meta = parse_case_metadata(case_dir)
    return {
        "index": index,
        "case_dir": str(case_dir),
        "image_source": image_source,
        **case_meta,
        "frame_strategy": args.frame_strategy,
        "frame_index": int(frame),
        "phase_fraction": float(args.phase_fraction),
        "n_mask_voxels": int(mask_indices.size),
        "sampled_points": int(args.n_points),
        "sampled_with_replacement": bool(replace),
        "geometry_scale": float(scale),
        "raw_length": raw_length,
        "velocity_unit_scale": float(args.velocity_unit_scale),
        "speed_mean": float(speed.mean()),
        "speed_rms": float(np.sqrt(np.mean(speed ** 2))),
        "speed_peak": float(np.percentile(speed, 95)),
        "x_shape": list(x_out.shape),
        "y_shape": list(y_out.shape),
        "cond_schema": COND_SCHEMA,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root containing CMRx4DFlow case folders with kdata_full.mat.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--image_cache_root", type=str, default="",
                        help="Optional root containing relpath/img_gt.npz caches. If present, avoids k-space IFFT.")
    parser.add_argument("--single_frame_cache_root", type=str, default="",
                        help="Optional FVUNet cache root containing relpath/R*/t*.npz single-frame samples.")
    parser.add_argument("--single_frame_cache_r", type=int, default=10,
                        help="Preferred acceleration folder R value for --single_frame_cache_root.")
    parser.add_argument("--n_points", type=int, default=30000)
    parser.add_argument("--target_length", type=float, default=5.0)
    parser.add_argument("--frame_strategy", type=str, default="fixed",
                        choices=["peak", "fixed", "middle", "phase"])
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Used only when --frame_strategy fixed.")
    parser.add_argument("--phase_fraction", type=float, default=0.5,
                        help="Normalized cardiac phase in [0,1] used when --frame_strategy phase.")
    parser.add_argument("--velocity_unit_scale", type=float, default=0.01,
                        help="Scale applied after VENC conversion. Use 0.01 when VENC is cm/s and target is m/s.")
    parser.add_argument("--clip_speed_percentile", type=float, default=99.5,
                        help="Clip per-sample velocity vectors by speed percentile; <=0 disables clipping.")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Process at most this many cases; <=0 means all cases.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--split_strategy", type=str, default="random",
                        choices=["random", "stratified"])
    parser.add_argument("--speed_bins", type=int, default=3,
                        help="Number of quantile bins for speed stratification.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--verbose_case", action="store_true",
                        help="Print per-case preprocessing stages for long full-kspace conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(data_root, limit=args.max_cases)
    if not cases:
        raise RuntimeError(f"No CMRx4DFlow cases found under {data_root}")

    manifest = []
    case_iter = enumerate(cases)
    if tqdm is not None:
        case_iter = tqdm(
            case_iter,
            total=len(cases),
            desc="CMRx4DFlow pointcloud",
            unit="case",
            dynamic_ncols=True,
        )

    def log(message: str) -> None:
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    for index, case_dir in case_iter:
        try:
            item = process_case(case_dir, out_dir, index, args)
            manifest.append(item)
            if tqdm is not None:
                case_iter.set_postfix(
                    frame=item["frame_index"],
                    speed_mean=f"{item['speed_mean']:.4g}",
                    refresh=False,
                )
            log(f"[OK] {index:04d} frame={item['frame_index']} "
                f"speed_mean={item['speed_mean']:.6g} case={case_dir}")
        except Exception as exc:
            log(f"[ERROR] {case_dir}: {exc}")

    speed_bin_edges = assign_speed_bins(manifest, args.speed_bins)
    if args.split_strategy == "stratified":
        split = split_indices_stratified(manifest, args.train_frac, args.val_frac, args.seed)
    else:
        split = split_indices([int(item["index"]) for item in manifest], args.train_frac, args.val_frac, args.seed)
        split["split_strategy"] = "random"
    split["diagnostics"] = split_diagnostics(manifest, split)
    split["speed_bin_edges"] = speed_bin_edges

    np.save(out_dir / "global_split_info.npy", split)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "data_root": str(data_root),
            "out_dir": str(out_dir),
            "n_cases": len(manifest),
            "n_points": args.n_points,
            "frame_strategy": args.frame_strategy,
            "phase_fraction": args.phase_fraction,
            "velocity_unit_scale": args.velocity_unit_scale,
            "split_strategy": args.split_strategy,
            "speed_bins": args.speed_bins,
            "speed_bin_edges": speed_bin_edges,
            "split": split,
            "cond_schema": COND_SCHEMA,
        }, f, indent=2)
    print(f"[DONE] processed={len(manifest)} out_dir={out_dir}")
    print(f"[DONE] split={split}")


if __name__ == "__main__":
    main()
