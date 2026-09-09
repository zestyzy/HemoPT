#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np


CONFIG_TO_ID = {
    "C1": 0,
    "ICA_norm": 1,
    "ICA_ste": 2,
}

ID_TO_CONFIG = {v: k for k, v in CONFIG_TO_ID.items()}


def _load_indices(data_path):
    indices = []
    for name in os.listdir(data_path):
        if name.startswith("cond_") and name.endswith(".npy"):
            indices.append(int(name[5:-4]))
    return sorted(indices)


def _stratified_split(indices, labels, train_frac, val_frac, seed):
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for idx in indices:
        by_label[int(labels[idx])].append(idx)

    train, val, test = [], [], []
    for label in sorted(by_label):
        group = np.array(sorted(by_label[label]), dtype=np.int64)
        group = rng.permutation(group)
        n = len(group)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
        train.extend(group[:n_train].astype(int).tolist())
        val.extend(group[n_train:n_train + n_val].astype(int).tolist())
        test.extend(group[n_train + n_val:].astype(int).tolist())

    return sorted(train), sorted(val), sorted(test)


def _counts(indices, labels):
    counter = Counter(ID_TO_CONFIG[int(labels[idx])] for idx in indices)
    return {name: int(counter.get(name, 0)) for name in CONFIG_TO_ID}


def main():
    parser = argparse.ArgumentParser("Create a strict HemoPT train/val/test split.")
    parser.add_argument("--data_path", default="hemo_npys")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    data_path = args.data_path
    out = args.out or os.path.join(data_path, "hemo_strict_split_info.npy")
    indices = _load_indices(data_path)
    if not indices:
        raise RuntimeError(f"No cond_*.npy files found under {data_path}")

    labels = {}
    for idx in indices:
        cond = np.load(os.path.join(data_path, f"cond_{idx}.npy"))
        labels[idx] = int(round(float(cond[0])))

    train, val, test = _stratified_split(indices, labels, args.train_frac, args.val_frac, args.seed)
    split_info = {
        "train_indices": train,
        "val_indices": val,
        "test_indices": test,
        "seed": args.seed,
        "strategy": "stratified_by_hemo_config",
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "counts": {
            "train": _counts(train, labels),
            "val": _counts(val, labels),
            "test": _counts(test, labels),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out, split_info, allow_pickle=True)
    print(f"Wrote {out}")
    print(json.dumps({
        "total": len(indices),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "counts": split_info["counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
