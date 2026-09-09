#!/usr/bin/env python3
"""Create a strict 4DFlow zero-wall, velocity-only view for downstream tests.

Input is an existing CMRx4DFlow npy directory produced by
CMRx4DFlow_process.py. The aligned output keeps the same split and metadata
but rewrites:

  x_i.npy: (N, 7) = [x, y, z, 0, 0, 0, 0]
  y_i.npy: (N, 3) = [u, v, w]

The four zero feature channels are retained to preserve the HemoPT downstream
input dimensionality. They do not encode wall information.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _indices(src: Path) -> list[int]:
    out = []
    for path in src.glob("x_*.npy"):
        try:
            out.append(int(path.stem.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(out)


def _copy_if_exists(src: Path, dst: Path, name: str) -> None:
    sp = src / name
    if sp.exists():
        shutil.copy2(sp, dst / name)


def align_case(src: Path, dst: Path, index: int, dtype: np.dtype) -> dict[str, object]:
    x = np.load(src / f"x_{index}.npy")
    y = np.load(src / f"y_{index}.npy")
    cond = np.load(src / f"cond_{index}.npy")

    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError(f"x_{index}.npy must be (N,C>=3), got {x.shape}")
    if y.ndim != 2 or y.shape[1] not in (3, 4):
        raise ValueError(f"y_{index}.npy must be (N,3) or (N,4), got {y.shape}")

    x_out = np.zeros((x.shape[0], 7), dtype=dtype)
    x_out[:, :3] = x[:, :3].astype(dtype, copy=False)
    if y.shape[1] == 4:
        y_out = y[:, 1:4]
        source_y_format = "dummy_p_uvw"
    else:
        y_out = y
        source_y_format = "uvw"
    y_out = y_out.astype(dtype, copy=False)

    np.save(dst / f"x_{index}.npy", x_out)
    np.save(dst / f"y_{index}.npy", y_out)
    np.save(dst / f"cond_{index}.npy", cond.astype(dtype, copy=False))

    speed = np.linalg.norm(y_out, axis=1)
    return {
        "index": index,
        "n_points": int(x_out.shape[0]),
        "x_shape": list(x_out.shape),
        "y_shape": list(y_out.shape),
        "source_y_format": source_y_format,
        "speed_mean": float(speed.mean()) if speed.size else 0.0,
        "speed_max": float(speed.max()) if speed.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source 4DFlow npy directory.")
    parser.add_argument("--out", required=True, help="Output aligned npy directory.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise FileNotFoundError(src)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is non-empty: {out}. Use --overwrite to replace files.")
    out.mkdir(parents=True, exist_ok=True)

    for old in out.glob("x_*.npy"):
        old.unlink()
    for old in out.glob("y_*.npy"):
        old.unlink()
    for old in out.glob("cond_*.npy"):
        old.unlink()

    dtype = np.dtype(args.dtype)
    indices = _indices(src)
    if not indices:
        raise RuntimeError(f"No x_*.npy files found in {src}")

    manifest = []
    for index in tqdm(indices, desc="align 4DFlow zero-uvw", unit="case", dynamic_ncols=True):
        manifest.append(align_case(src, out, index, dtype))

    for name in ("global_split_info.npy", "manifest.jsonl"):
        _copy_if_exists(src, out, name)

    source_summary = {}
    if (src / "summary.json").exists():
        source_summary = json.loads((src / "summary.json").read_text())
    split_info = np.load(out / "global_split_info.npy", allow_pickle=True).item() if (out / "global_split_info.npy").exists() else {}
    summary = {
        "source_dir": str(src),
        "out_dir": str(out),
        "alignment": "zero_wall_native_uvw",
        "x_format": "[x,y,z,0,0,0,0]",
        "y_format": "[u,v,w]",
        "n_cases": len(manifest),
        "n_points": int(manifest[0]["n_points"]),
        "dtype": str(dtype),
        "split": {
            key: [int(x) for x in split_info.get(key, [])]
            for key in ("train_indices", "val_indices", "test_indices")
            if key in split_info
        },
        "source_summary": source_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    (out / "alignment_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=True) + "\n" for item in manifest)
    )
    print(f"[DONE] aligned {len(manifest)} cases from {src} to {out}")


if __name__ == "__main__":
    main()
