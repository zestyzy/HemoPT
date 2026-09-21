#!/usr/bin/env python3
"""Generate a tiny synthetic VascularPretrain dataset for smoke tests.

The generated files follow the HemoPT pretraining loader contract:

    <out_dir>/<dataset>/<sample>/x.npy
    <out_dir>/<dataset>/<sample>/condition_0.npy
    <out_dir>/<dataset>/<sample>/supervise_0.npy
    ...
    <out_dir>/<dataset>/<sample>/meta.json

This script does not use real patient, STL, or CFD data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path(".smoke_data/Vascular_PreTrain"))
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=64)
    parser.add_argument("--n_random_walks", type=int, default=20)
    parser.add_argument("--base_walks", type=int, default=20)
    parser.add_argument("--perturb_sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, eps)


def make_sample(rng: np.random.Generator, num_points: int, n_random_walks: int, perturb_sigma: float):
    axial = rng.uniform(-1.0, 1.0, size=(num_points, 1)).astype(np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(num_points, 1)).astype(np.float32)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=(num_points, 1))).astype(np.float32)

    x_pos = axial
    y_pos = radius * np.cos(theta)
    z_pos = radius * np.sin(theta)
    pos = np.concatenate([x_pos, y_pos, z_pos], axis=-1).astype(np.float32)

    radial_dir = normalize(np.concatenate([
        np.zeros_like(x_pos),
        y_pos,
        z_pos,
    ], axis=-1).astype(np.float32))
    wall_distance = np.clip(1.0 - radius, 0.0, 1.0).astype(np.float32)

    x = np.concatenate([pos, wall_distance, radial_dir], axis=-1).astype(np.float32)

    conditions = []
    supervises = []
    for _ in range(n_random_walks):
        walk_dir = normalize((rng.normal(size=(num_points, 3)) + perturb_sigma * rng.normal(size=(num_points, 3))).astype(np.float32))
        step = rng.uniform(0.02, 0.12, size=(num_points, 1)).astype(np.float32)
        condition = np.concatenate([walk_dir, step], axis=-1).astype(np.float32)

        offsets = []
        for t in range(3):
            q = pos + float(t) * step * walk_dir
            q_radial = normalize(np.concatenate([
                np.zeros_like(q[:, :1]),
                q[:, 1:2],
                q[:, 2:3],
            ], axis=-1).astype(np.float32))
            q_radius = np.linalg.norm(q[:, 1:3], axis=-1, keepdims=True)
            q_wall_distance = np.clip(1.0 - q_radius, 0.0, 1.0).astype(np.float32)
            offsets.append((-q_wall_distance * q_radial).astype(np.float32))
        supervise = np.concatenate(offsets, axis=-1).astype(np.float32)

        conditions.append(condition)
        supervises.append(supervise)

    return x, conditions, supervises


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset_dir = args.out_dir / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(args.num_samples):
        sample_dir = dataset_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        x, conditions, supervises = make_sample(
            rng=rng,
            num_points=args.num_points,
            n_random_walks=args.n_random_walks,
            perturb_sigma=args.perturb_sigma,
        )
        np.save(sample_dir / "x.npy", x)
        for walk_idx, (condition, supervise) in enumerate(zip(conditions, supervises)):
            np.save(sample_dir / f"condition_{walk_idx}.npy", condition)
            np.save(sample_dir / f"supervise_{walk_idx}.npy", supervise)
        meta = {
            "source": "synthetic_smoke_test",
            "qc_status": "pass",
            "num_points": int(args.num_points),
            "n_random_walks": int(args.n_random_walks),
            "base_walks": int(args.base_walks),
            "perturb_sigma": float(args.perturb_sigma),
            "walk_steps": 3,
            "contains_real_patient_data": False,
            "contains_cfd_labels": False,
        }
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"Wrote synthetic smoke dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
