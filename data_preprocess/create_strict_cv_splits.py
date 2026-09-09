#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _load_jsonl(path):
    rows = {}
    if path is None or not Path(path).exists():
        return rows
    with Path(path).open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _speed_bins(values, n_bins):
    values = np.array([_safe_float(v) for v in values], dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.int64), []
    quantiles = np.linspace(0.0, 1.0, max(1, n_bins) + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles)) if quantiles.size else np.array([])
    bins = np.searchsorted(edges, values, side="right").astype(np.int64) if edges.size else np.zeros_like(values, dtype=np.int64)
    return bins, [float(x) for x in edges.tolist()]


def _metadata_by_index(split_info):
    metadata = split_info.get("metadata") or []
    return {int(row["index"]): row for row in metadata if isinstance(row, dict) and "index" in row}


def _hemo_vessel_type(data_dir, idx):
    cond = np.load(Path(data_dir) / f"cond_{int(idx)}.npy")
    return int(round(float(cond[0])))


def _build_group_rows(dataset, data_dir, split_info, manifest_rows, pool_indices, speed_bins):
    meta = _metadata_by_index(split_info)
    pool_indices = [int(i) for i in pool_indices]

    speed_values = []
    for idx in pool_indices:
        row = meta.get(idx, {})
        manifest = manifest_rows.get(idx, {})
        if dataset == "vmr":
            speed = row.get("qc_speed_mean", row.get("speed_mean", row.get("vmax", 0.0)))
        elif dataset == "aneumo":
            speed = row.get("speed_mean", row.get("meta_inlet_speed_mean", row.get("vmax", 0.0)))
        elif dataset == "4dflow":
            speed = manifest.get("speed_mean", 0.0)
        elif dataset == "hemopt":
            speed = _hemo_vessel_type(data_dir, idx)
        else:
            speed = row.get("speed_mean", manifest.get("speed_mean", 0.0))
        speed_values.append(speed)
    bins, edges = _speed_bins(speed_values, speed_bins)

    groups = {}
    for idx, speed_bin in zip(pool_indices, bins):
        row = meta.get(idx, {})
        manifest = manifest_rows.get(idx, {})
        if dataset == "vmr":
            group = str(row.get("case_id") or idx)
            family = str(row.get("meta_territory_family") or row.get("territory", "NA")).split("_")[0]
            label = f"{family}|speed{int(speed_bin)}"
        elif dataset == "aneumo":
            group = str(row.get("case_id") or idx)
            m_value = row.get("meta_m_value", row.get("m_value", row.get("m_dir", "NA")))
            label = f"m{m_value}|speed{int(speed_bin)}"
        elif dataset == "4dflow":
            group = "|".join(str(manifest.get(key, "")) for key in ("territory", "center", "scanner", "case_id"))
            label = "|".join([
                str(manifest.get("territory", "NA")),
                str(manifest.get("center", "NA")),
                str(manifest.get("scanner", "NA")),
                f"speed{int(speed_bin)}",
            ])
        elif dataset == "hemopt":
            vessel_type = _hemo_vessel_type(data_dir, idx)
            group = str(idx)
            label = f"vessel{vessel_type}"
        else:
            group = str(row.get("case_id") or manifest.get("case_id") or idx)
            label = f"speed{int(speed_bin)}"

        entry = groups.setdefault(group, {"indices": [], "labels": Counter()})
        entry["indices"].append(idx)
        entry["labels"][label] += 1

    group_rows = []
    for group, entry in groups.items():
        label = entry["labels"].most_common(1)[0][0]
        group_rows.append({
            "group": group,
            "indices": sorted(entry["indices"]),
            "label": label,
            "size": len(entry["indices"]),
        })
    return group_rows, edges


def _make_folds(group_rows, n_folds, seed):
    if len(group_rows) < n_folds:
        raise ValueError(f"Need at least {n_folds} groups, got {len(group_rows)}")

    by_label = defaultdict(list)
    for row in group_rows:
        by_label[row["label"]].append(row)

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]
    fold_label_counts = [Counter() for _ in range(n_folds)]

    labels = sorted(by_label, key=lambda key: (-len(by_label[key]), key))
    for label in labels:
        rows = list(by_label[label])
        order = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in order]
        rows.sort(key=lambda row: row["size"], reverse=True)
        for row in rows:
            target = min(
                range(n_folds),
                key=lambda i: (fold_label_counts[i][label], fold_sizes[i], i),
            )
            folds[target].append(row)
            fold_sizes[target] += row["size"]
            fold_label_counts[target][label] += row["size"]

    return folds


def _expand(rows):
    out = []
    groups = []
    labels = []
    for row in rows:
        out.extend(row["indices"])
        groups.append(row["group"])
        labels.append(row["label"])
    return sorted(int(i) for i in out), sorted(groups), Counter(labels)


def _assert_no_overlap(cv_info):
    test = set(map(int, cv_info["test_indices"]))
    pool = set(map(int, cv_info["pool_indices"]))
    if pool & test:
        raise RuntimeError(f"Pool/test overlap detected: {len(pool & test)} indices")
    for fold in cv_info["folds"]:
        train = set(map(int, fold["train_indices"]))
        val = set(map(int, fold["val_indices"]))
        if train & val:
            raise RuntimeError(f"Train/val overlap in fold {fold['fold']}")
        if train & test or val & test:
            raise RuntimeError(f"Independent test leakage in fold {fold['fold']}")


def create(args):
    split_info = np.load(args.split_info, allow_pickle=True).item()
    train_indices = [int(i) for i in split_info["train_indices"]]
    val_indices = [int(i) for i in split_info.get("val_indices", [])]
    test_indices = [int(i) for i in split_info["test_indices"]]
    pool_indices = sorted(train_indices + val_indices)

    manifest_rows = _load_jsonl(args.manifest)
    group_rows, speed_edges = _build_group_rows(
        args.dataset, args.data_path, split_info, manifest_rows, pool_indices, args.speed_bins)
    folds = _make_folds(group_rows, args.n_folds, args.seed)

    cv_folds = []
    fold_summaries = []
    all_pool_groups = {row["group"]: row for row in group_rows}
    for fold_idx, val_rows in enumerate(folds):
        val_groups = {row["group"] for row in val_rows}
        train_rows = [row for group, row in all_pool_groups.items() if group not in val_groups]
        train_out, train_groups, train_labels = _expand(train_rows)
        val_out, val_group_names, val_labels = _expand(val_rows)
        cv_folds.append({
            "fold": fold_idx,
            "train_indices": train_out,
            "val_indices": val_out,
            "train_groups": train_groups,
            "val_groups": val_group_names,
            "train_label_counts": dict(sorted(train_labels.items())),
            "val_label_counts": dict(sorted(val_labels.items())),
        })
        fold_summaries.append({
            "fold": fold_idx,
            "train_n": len(train_out),
            "val_n": len(val_out),
            "train_groups": len(train_groups),
            "val_groups": len(val_group_names),
            "val_label_counts": dict(sorted(val_labels.items())),
        })

    cv_info = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()),
        "source_split_info": str(Path(args.split_info).resolve()),
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "seed": int(args.seed),
        "n_folds": int(args.n_folds),
        "strategy": "strict_independent_test_group_stratified",
        "pool_definition": "source train_indices + source val_indices",
        "pool_indices": pool_indices,
        "test_indices": test_indices,
        "source_train_indices": train_indices,
        "source_val_indices": val_indices,
        "speed_bin_edges": speed_edges,
        "folds": cv_folds,
        "diagnostics": {
            "pool_n": len(pool_indices),
            "test_n": len(test_indices),
            "group_n": len(group_rows),
            "test_overlap_n": len(set(pool_indices) & set(test_indices)),
            "fold_summaries": fold_summaries,
        },
    }
    _assert_no_overlap(cv_info)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, cv_info, allow_pickle=True)
    summary = out.with_suffix(".json")
    summary.write_text(json.dumps(cv_info["diagnostics"], indent=2, ensure_ascii=False) + "\n")

    print(f"Wrote {out}")
    print(f"Wrote {summary}")
    print(json.dumps(cv_info["diagnostics"], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["aneumo", "vmr", "hemopt", "4dflow"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split_info", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--speed_bins", type=int, default=3)
    create(parser.parse_args())


if __name__ == "__main__":
    main()
