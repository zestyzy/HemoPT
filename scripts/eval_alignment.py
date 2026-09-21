import os, sys, glob, json, argparse, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.dynamic_flow_dict import AnalyticCompactFlowDictionary
from data_provider.vascular_pretrain_loader import _VascularPretrainDataset

def compute_metrics(u_pred, u_gt):
    """
    Compute Directional Cosine Alignment (C_dir) and Magnitude Relative Error (C_mag).
    """
    dot = (u_pred * u_gt).sum(axis=-1)
    norm_p = np.linalg.norm(u_pred, axis=-1)
    norm_g = np.linalg.norm(u_gt, axis=-1)
    cos = dot / (norm_p * norm_g + 1e-6)
    cdir = float(np.mean(cos))
    cmag = float(np.sum(np.abs(norm_p - norm_g)) / (np.sum(norm_g) + 1e-6))
    return cdir, cmag

def evaluate_dataset(name, data_dir, split_file, flow_dict, ds_helper, device, eval_all=False, aneumo_max_case=540):
    print(f"\nEvaluating dataset: {name} from {data_dir}...")
    if not os.path.exists(data_dir):
        print(f"  [ERROR] Directory not found: {data_dir}")
        return None

    # Load split info
    if (not eval_all) and os.path.exists(split_file):
        info = np.load(split_file, allow_pickle=True).item()
        test_indices = list(info.get("test_indices", []))
        # For Aneumo, filter to official 540 benchmark cases if metadata is present
        if name == "Aneumo" and aneumo_max_case:
            meta_file = os.path.join(data_dir, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                valid_indices = set(m["index"] for m in meta if int(m.get("case_id", 9999)) <= aneumo_max_case)
                test_indices = [idx for idx in test_indices if idx in valid_indices]
    else:
        if name == "Aneumo" and aneumo_max_case:
            meta_file = os.path.join(data_dir, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                test_indices = [m["index"] for m in meta if int(m.get("case_id", 9999)) <= aneumo_max_case]
            else:
                test_indices = sorted([int(f.replace("x_", "").replace(".npy", ""))
                                       for f in os.listdir(data_dir) if f.startswith("x_") and f.endswith(".npy")])[:aneumo_max_case]
        else:
            test_indices = sorted([int(f.replace("x_", "").replace(".npy", ""))
                                   for f in os.listdir(data_dir) if f.startswith("x_") and f.endswith(".npy")])

    print(f"  Found {len(test_indices)} cases to evaluate (eval_all={eval_all}).")

    metrics = {
        "poiseuille": {"cdir": [], "cmag": []},
        "uniform": {"cdir": [], "cmag": []},
        "hemopt": {"cdir": [], "cmag": []},
    }

    evaluated_count = 0
    for idx in test_indices:
        x_path = os.path.join(data_dir, f"x_{idx}.npy")
        y_path = os.path.join(data_dir, f"y_{idx}.npy")
        cond_path = os.path.join(data_dir, f"cond_{idx}.npy")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            continue

        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)
        cond = np.load(cond_path).astype(np.float32) if os.path.exists(cond_path) else np.array([0.0, 0.3, 0.0035], dtype=np.float32)

        pos = x[:, :3]
        if y.ndim == 2 and y.shape[1] == 3:
            u_cfd = y[:, :3]
        else:
            u_cfd = y[:, 1:4] # (N, 3)

        # Ensure x_feat has 7 dims: [pos (3), dist (1), normals (3)]
        if x.shape[1] >= 7:
            x_feat = x[:, :7]
        else:
            centered = pos - pos.mean(axis=0, keepdims=True)
            r = np.linalg.norm(centered, axis=-1)
            r_max = r.max() + 1e-6
            dist = (r_max - r).astype(np.float32)
            normals = -centered / (r[:, None] + 1e-8)
            x_feat = np.hstack([pos, dist[:, None], normals]).astype(np.float32)

        # Build 8-mode canonical bank
        bank = ds_helper._build_generalized_flow_bank(x_feat, f"{name}_{idx}") # (N, 8, 3)

        # Check alignment of mode 0 with mean flow orientation
        u_mean = u_cfd.mean(axis=0)
        bank_mean = bank[:, 0, :].mean(axis=0)
        if np.dot(bank_mean, u_mean) < 0:
            bank = -bank

        u_poi = bank[:, 0, :] # Mode 1: Static Poiseuille Prior
        u_uni = bank.mean(axis=1) # Uniform 8-mode average

        # Determine characteristic inlet velocity
        if name == "ICA-Sim":
            inlet_v = float(cond[1])
        elif name == "Aneumo":
            inlet_v = float(cond[3]) if len(cond) > 3 else float(cond[1])
        elif name in ["VMR-AOCOA", "4DFlow"]:
            inlet_v = float(cond[2]) if len(cond) > 2 else float(cond[1])
        else:
            inlet_v = float(np.linalg.norm(u_cfd, axis=-1).mean())

        # Synthesize HemoPT proxy
        with torch.no_grad():
            pos_t = torch.from_numpy(pos).float().unsqueeze(0).to(device)
            morph_t = torch.from_numpy(x_feat).float().unsqueeze(0).to(device)
            cond_t = torch.zeros(1, pos.shape[0], 4, device=device)
            cond_t[:, :, 0] = inlet_v
            basis_t = torch.from_numpy(bank).float().unsqueeze(0).to(device)
            traj_t = torch.zeros(1, pos.shape[0], 9, device=device)

            compact, _, weights = flow_dict(pos_t, morph_t, cond_t, basis_t, rw_trajectory=traj_t)
            u_hemopt = compact[0, :, :3].cpu().numpy()

        cd_p, cm_p = compute_metrics(u_poi, u_cfd)
        cd_u, cm_u = compute_metrics(u_uni, u_cfd)
        cd_h, cm_h = compute_metrics(u_hemopt, u_cfd)

        metrics["poiseuille"]["cdir"].append(cd_p)
        metrics["poiseuille"]["cmag"].append(cm_p)
        metrics["uniform"]["cdir"].append(cd_u)
        metrics["uniform"]["cmag"].append(cm_u)
        metrics["hemopt"]["cdir"].append(cd_h)
        metrics["hemopt"]["cmag"].append(cm_h)
        evaluated_count += 1

    summary = {}
    for m in ["poiseuille", "uniform", "hemopt"]:
        cd_arr = np.array(metrics[m]["cdir"])
        cm_arr = np.array(metrics[m]["cmag"])
        summary[m] = {
            "cdir_mean": float(np.mean(cd_arr)) if len(cd_arr) else 0.0,
            "cdir_std": float(np.std(cd_arr)) if len(cd_arr) else 0.0,
            "cmag_mean": float(np.mean(cm_arr)) if len(cm_arr) else 0.0,
            "cmag_std": float(np.std(cm_arr)) if len(cm_arr) else 0.0,
            "n_samples": evaluated_count,
        }

    # Relative magnitude error reduction over Poiseuille
    poi_cmag = summary["poiseuille"]["cmag_mean"]
    hem_cmag = summary["hemopt"]["cmag_mean"]
    rel_red = ((poi_cmag - hem_cmag) / (poi_cmag + 1e-8)) * 100.0
    summary["rel_reduction_pct"] = float(rel_red)

    print(f"  Results for {name} ({evaluated_count} cases):")
    print(f"    Poiseuille : C_dir={summary['poiseuille']['cdir_mean']:.4f} +/- {summary['poiseuille']['cdir_std']:.4f}, C_mag={summary['poiseuille']['cmag_mean']:.4f} +/- {summary['poiseuille']['cmag_std']:.4f}")
    print(f"    Uniform    : C_dir={summary['uniform']['cdir_mean']:.4f} +/- {summary['uniform']['cdir_std']:.4f}, C_mag={summary['uniform']['cmag_mean']:.4f} +/- {summary['uniform']['cmag_std']:.4f}")
    print(f"    HemoPT     : C_dir={summary['hemopt']['cdir_mean']:.4f} +/- {summary['hemopt']['cdir_std']:.4f}, C_mag={summary['hemopt']['cmag_mean']:.4f} +/- {summary['hemopt']['cmag_std']:.4f}")
    print(f"    Rel Error Reduction: {rel_red:+.2f}%")
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/rw_encoder_routing_probe_15ep_best.pt")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate all available cases in each cohort rather than just test split")
    parser.add_argument("--aneumo_max_case", type=int, default=540, help="Official benchmark limit for Aneumo cases (default 540)")
    parser.add_argument("--out_json", type=str, default="results/proxy_alignment_results_all4.json")
    args = parser.parse_args()

    print("==========================================================================")
    print(" Cross-Dataset Physical Flow Proxy Alignment Assessment (All 4 Cohorts)")
    print(f" Checkpoint: {args.ckpt}")
    print(f" Device: {args.device}")
    print(f" Mode: {'ALL CASES' if args.eval_all else 'TEST SPLIT'} (Aneumo limit: {args.aneumo_max_case})")
    print("==========================================================================")

    # 1. Load trained dictionary
    flow_dict = AnalyticCompactFlowDictionary(
        morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64,
        rw_encoder_hidden_dim=16, rw_feature_dim=16
    ).to(args.device)

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        state_dict = ckpt["flow_dict"] if "flow_dict" in ckpt else ckpt
        flow_dict.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {args.ckpt}")
    else:
        print(f"[WARN] Checkpoint {args.ckpt} not found, using initialized weights!")

    flow_dict.eval()
    ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode="learnable_generalized_flow_compact")

    datasets_config = {
        "ICA-Sim": {
            "data_dir": "../GeoPT-main/hemo_npys",
            "split_file": "../GeoPT-main/hemo_npys/global_split_info.npy",
        },
        "Aneumo": {
            "data_dir": "../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004",
            "split_file": "../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004/global_split_info.npy",
        },
        "VMR-AOCOA": {
            "data_dir": "../GeoPT-main/vmr_cfd_npys_quality_filtered",
            "split_file": "../GeoPT-main/vmr_cfd_npys_quality_filtered/global_split_info.npy",
        },
        "4DFlow": {
            "data_dir": "../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw",
            "split_file": "../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw/global_split_info.npy",
        },
    }

    all_results = {}
    for name, cfg in datasets_config.items():
        res = evaluate_dataset(
            name=name,
            data_dir=cfg["data_dir"],
            split_file=cfg["split_file"],
            flow_dict=flow_dict,
            ds_helper=ds_helper,
            device=args.device,
            eval_all=args.eval_all,
            aneumo_max_case=args.aneumo_max_case
        )
        if res:
            all_results[name] = res

    # Compute Macro Average
    macro = {"poiseuille": {}, "uniform": {}, "hemopt": {}}
    for m in ["poiseuille", "uniform", "hemopt"]:
        macro[m]["cdir_mean"] = float(np.mean([all_results[k][m]["cdir_mean"] for k in all_results]))
        macro[m]["cmag_mean"] = float(np.mean([all_results[k][m]["cmag_mean"] for k in all_results]))
    macro_red = ((macro["poiseuille"]["cmag_mean"] - macro["hemopt"]["cmag_mean"]) / macro["poiseuille"]["cmag_mean"]) * 100.0
    macro["rel_reduction_pct"] = float(macro_red)
    all_results["Macro_Average"] = macro

    print("\n==========================================================================")
    print("                    CROSS-DATASET ALIGNMENT SUMMARY")
    print("==========================================================================")
    print(f"{'Benchmark':<12} | {'Method':<24} | {'C_dir (Direction) ^':<22} | {'C_mag (Magnitude) v':<22} | {'Rel. Red. (%)'}")
    print("-" * 95)
    for name in list(datasets_config.keys()) + ["Macro_Average"]:
        if name not in all_results:
            continue
        d = all_results[name]
        for m, label in [("poiseuille", "Classical Poiseuille"), ("uniform", "Uniform 8-Mode Mix"), ("hemopt", "HemoPT Dynamic Proxy")]:
            cd_str = f"{d[m]['cdir_mean']:.4f}" + (f" +/- {d[m]['cdir_std']:.4f}" if "cdir_std" in d[m] else "")
            cm_str = f"{d[m]['cmag_mean']:.4f}" + (f" +/- {d[m]['cmag_std']:.4f}" if "cmag_std" in d[m] else "")
            red_str = f"{d['rel_reduction_pct']:+.2f}%" if m == "hemopt" else ("---" if m == "poiseuille" else "")
            print(f"{name:<12} | {label:<24} | {cd_str:<22} | {cm_str:<22} | {red_str}")
        print("-" * 95)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.out_json}")

if __name__ == "__main__":
    main()
