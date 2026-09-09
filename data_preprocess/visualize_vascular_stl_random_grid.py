#!/usr/bin/env python3
"""Render a random STL grid for every dataset in HemoData/Vascular_STL.

The figure uses a lightweight off-screen point rendering path instead of
matplotlib 3D polygons, which keeps memory and rendering time practical for a
large 13 x 10 overview image.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


ROOT = Path(".")
DEFAULT_STL_ROOT = ROOT / "HemoData" / "Vascular_STL"
DEFAULT_OUT_DIR = ROOT / "results" / "vascular_stl_visualization"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl_root", type=Path, default=DEFAULT_STL_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples_per_dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--tile", type=int, default=260)
    parser.add_argument("--points", type=int, default=14000)
    parser.add_argument("--image_name", default="vascular_stl_random10_grid_seed20260525.png")
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def list_datasets(stl_root: Path):
    datasets = []
    for dataset_dir in stl_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        stls = sorted(dataset_dir.glob("*.stl"))
        if stls:
            datasets.append((dataset_dir.name, stls))
    return sorted(datasets, key=lambda item: item[0].casefold())


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.vertices) and len(g.faces)]
        if not geometries:
            raise ValueError("empty scene")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("empty mesh")
    return mesh


def sample_points(mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) <= n_points:
        points = vertices
    else:
        idx = rng.choice(len(vertices), size=n_points, replace=False)
        points = vertices[idx]

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 4:
        raise ValueError("not enough finite vertices")
    return points


def pca_oriented(points: np.ndarray) -> np.ndarray:
    center = np.median(points, axis=0)
    centered = points - center
    try:
        _, _, vh = np.linalg.svd(centered[:: max(1, len(centered) // 5000)], full_matrices=False)
        oriented = centered @ vh.T
    except np.linalg.LinAlgError:
        oriented = centered

    for axis in range(3):
        if np.percentile(oriented[:, axis], 95) + np.percentile(oriented[:, axis], 5) < 0:
            oriented[:, axis] *= -1.0
    return oriented


def rotate_points(points: np.ndarray) -> np.ndarray:
    rz = math.radians(-32.0)
    rx = math.radians(58.0)
    ry = math.radians(12.0)

    rz_mat = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0.0],
            [math.sin(rz), math.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rx), -math.sin(rx)],
            [0.0, math.sin(rx), math.cos(rx)],
        ]
    )
    ry_mat = np.array(
        [
            [math.cos(ry), 0.0, math.sin(ry)],
            [0.0, 1.0, 0.0],
            [-math.sin(ry), 0.0, math.cos(ry)],
        ]
    )
    return points @ (rz_mat @ rx_mat @ ry_mat).T


def fit_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if font.getlength(text) <= max_width:
        return text
    ellipsis = "..."
    for i in range(len(text), 0, -1):
        candidate = text[:i] + ellipsis
        if font.getlength(candidate) <= max_width:
            return candidate
    return ellipsis


def render_tile(path: Path, tile: int, n_points: int, rng: np.random.Generator, font_small) -> Image.Image:
    img = Image.new("RGB", (tile, tile), (248, 250, 252))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle((4, 4, tile - 5, tile - 5), radius=8, outline=(203, 213, 225), width=1, fill=(248, 250, 252))

    try:
        mesh = load_mesh(path)
        points = sample_points(mesh, n_points=n_points, rng=rng)
        points = rotate_points(pca_oriented(points))

        xy = points[:, :2]
        depth = points[:, 2]
        span = np.ptp(xy, axis=0)
        max_span = float(max(span.max(), 1e-6))
        scale = (tile - 48) / max_span
        xy = xy * scale
        xy -= xy.mean(axis=0)
        xy[:, 0] += tile / 2.0
        xy[:, 1] = tile / 2.0 - xy[:, 1]

        z0, z1 = np.percentile(depth, [2, 98])
        zn = np.clip((depth - z0) / max(z1 - z0, 1e-6), 0.0, 1.0)
        order = np.argsort(zn)
        if len(order) > n_points:
            order = order[:n_points]

        for idx in order:
            x, y = xy[idx]
            if x < 8 or x >= tile - 8 or y < 24 or y >= tile - 28:
                continue
            shade = float(zn[idx])
            r = int(35 + 45 * shade)
            g = int(82 + 90 * shade)
            b = int(124 + 90 * shade)
            radius = 1 if shade < 0.65 else 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(r, g, b, 190))

        title = fit_text(path.stem, font_small, tile - 18)
        draw.rectangle((5, tile - 24, tile - 5, tile - 5), fill=(248, 250, 252, 235))
        draw.text((9, tile - 22), title, fill=(51, 65, 85), font=font_small)
    except Exception as exc:
        draw.text((14, tile // 2 - 10), "render failed", fill=(185, 28, 28), font=font_small)
        draw.text((14, tile // 2 + 10), fit_text(str(exc), font_small, tile - 28), fill=(185, 28, 28), font=font_small)

    return img


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list_datasets(args.stl_root)
    if not datasets:
        raise SystemExit(f"No STL files found under {args.stl_root}")

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    selections = []
    for dataset, stls in datasets:
        selected = py_rng.sample(stls, k=min(args.samples_per_dataset, len(stls)))
        selections.append((dataset, selected, len(stls)))

    tile = args.tile
    label_w = 230
    header_h = 58
    row_h = tile + 18
    width = label_w + args.samples_per_dataset * tile
    height = header_h + len(selections) * row_h

    font_title = load_font(24, bold=True)
    font_ds = load_font(18, bold=True)
    font_small = load_font(12)
    font_col = load_font(13, bold=True)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, width, header_h), fill=(241, 245, 249))
    draw.text((18, 16), f"Vascular_STL random {args.samples_per_dataset} STL per dataset", fill=(15, 23, 42), font=font_title)
    for col in range(args.samples_per_dataset):
        x = label_w + col * tile
        draw.text((x + 10, header_h - 24), f"sample {col + 1}", fill=(71, 85, 105), font=font_col)

    manifest_rows = []
    for row_idx, (dataset, selected, total) in enumerate(tqdm(selections, desc="render datasets")):
        y0 = header_h + row_idx * row_h
        fill = (255, 255, 255) if row_idx % 2 == 0 else (248, 250, 252)
        draw.rectangle((0, y0, width, y0 + row_h), fill=fill)
        draw.line((0, y0, width, y0), fill=(226, 232, 240), width=1)
        draw.text((18, y0 + 18), dataset, fill=(15, 23, 42), font=font_ds)
        draw.text((18, y0 + 44), f"{total} STL", fill=(100, 116, 139), font=font_small)

        for col, path in enumerate(selected):
            x0 = label_w + col * tile
            tile_img = render_tile(path, tile=tile, n_points=args.points, rng=np_rng, font_small=font_small)
            canvas.paste(tile_img, (x0, y0 + 8))
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "dataset_total_stl": total,
                    "sample_index": col + 1,
                    "path": str(path),
                    "file": path.name,
                }
            )

    output_png = args.out_dir / args.image_name
    output_jsonl = args.out_dir / f"{Path(args.image_name).stem}_manifest.jsonl"
    canvas.save(output_png, quality=95)
    with output_jsonl.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote image: {output_png}")
    print(f"Wrote manifest: {output_jsonl}")
    print(f"Datasets: {len(selections)}")
    print(f"Tiles: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
