#!/usr/bin/env python3
"""Visualize before/after STL component cleaning for fragmented datasets."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

from visualize_vascular_stl_random_grid import load_font, render_tile


ROOT = Path(".")
DEFAULT_MANIFEST = ROOT / "results" / "vascular_stl_component_cleaning" / "component_cleaning_manifest.jsonl"
DEFAULT_OUT_DIR = ROOT / "results" / "vascular_stl_component_cleaning"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples_per_dataset", type=int, default=6)
    parser.add_argument("--tile", type=int, default=230)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--image_name", default="component_cleaning_before_after_worst6.png")
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [row for row in rows if row.get("status") == "ok"]


def fit_text(text, font, max_width):
    if font.getlength(text) <= max_width:
        return text
    suffix = "..."
    for n in range(len(text), 0, -1):
        candidate = text[:n] + suffix
        if font.getlength(candidate) <= max_width:
            return candidate
    return suffix


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest)
    datasets = sorted({row["dataset"] for row in rows})

    selected = []
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        ds_rows = sorted(ds_rows, key=lambda r: (r.get("kept_face_fraction", 1.0), -r.get("n_components", 0)))
        selected.append((dataset, ds_rows[: args.samples_per_dataset]))

    tile = args.tile
    label_w = 260
    header_h = 72
    pair_gap = 12
    sample_w = tile * 2 + pair_gap
    row_h = tile + 76
    width = label_w + args.samples_per_dataset * sample_w
    height = header_h + len(selected) * row_h

    font_title = load_font(24, bold=True)
    font_ds = load_font(18, bold=True)
    font_mid = load_font(13, bold=True)
    font_small = load_font(12)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, width, header_h), fill=(241, 245, 249))
    draw.text((18, 16), "Fragmented STL component cleaning: original vs cleaned", fill=(15, 23, 42), font=font_title)
    draw.text((18, 46), "Worst samples selected by removed face fraction", fill=(71, 85, 105), font=font_small)

    import numpy as np

    rng = np.random.default_rng(20260525)
    manifest_rows = []
    for row_idx, (dataset, ds_rows) in enumerate(tqdm(selected, desc="render comparisons")):
        y0 = header_h + row_idx * row_h
        draw.rectangle((0, y0, width, y0 + row_h), fill=(255, 255, 255) if row_idx % 2 == 0 else (248, 250, 252))
        draw.line((0, y0, width, y0), fill=(226, 232, 240), width=1)
        draw.text((18, y0 + 20), dataset, fill=(15, 23, 42), font=font_ds)
        draw.text((18, y0 + 48), f"{len(ds_rows)} worst shown", fill=(100, 116, 139), font=font_small)

        for col, row in enumerate(ds_rows):
            x0 = label_w + col * sample_w
            original = Path(row["source"])
            cleaned = Path(row["target"])
            draw.text((x0 + 6, y0 + 8), "original", fill=(71, 85, 105), font=font_mid)
            draw.text((x0 + tile + pair_gap + 6, y0 + 8), "cleaned", fill=(71, 85, 105), font=font_mid)
            canvas.paste(render_tile(original, tile=tile, n_points=args.points, rng=rng, font_small=font_small), (x0, y0 + 28))
            canvas.paste(render_tile(cleaned, tile=tile, n_points=args.points, rng=rng, font_small=font_small), (x0 + tile + pair_gap, y0 + 28))

            caption = f"keep {row.get('kept_face_fraction', 0.0) * 100:.1f}% | comp {row.get('n_components')}->{row.get('kept_components')}"
            caption = fit_text(caption, font_small, sample_w - 14)
            draw.text((x0 + 6, y0 + tile + 34), caption, fill=(51, 65, 85), font=font_small)
            draw.text((x0 + 6, y0 + tile + 52), fit_text(original.name, font_small, sample_w - 14), fill=(100, 116, 139), font=font_small)

            manifest_rows.append(
                {
                    "dataset": dataset,
                    "source": str(original),
                    "target": str(cleaned),
                    "file": original.name,
                    "n_components": row.get("n_components"),
                    "kept_components": row.get("kept_components"),
                    "kept_face_fraction": row.get("kept_face_fraction"),
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


if __name__ == "__main__":
    main()
