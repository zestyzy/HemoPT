import os, sys, glob, json, argparse, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.dynamic_flow_dict import AnalyticCompactFlowDictionary
from data_provider.vascular_pretrain_loader import _VascularPretrainDataset

def compute_metrics(u_pred, u_gt):
    dot = (u_pred * u_gt).sum(axis=-1)
    norm_p = np.linalg.norm(u_pred, axis=-1)
    norm_g = np.linalg.norm(u_gt, axis=-1)
    cos = dot / (norm_p * norm_g + 1e-6)
    cdir = float(np.mean(cos))
    cmag = float(np.sum(np.abs(norm_p - norm_g)) / (np.sum(norm_g) + 1e-6))
    return cdir, cmag

def audit_model(flow_dict, ds_helper, datasets_config, device, aneumo_max_case=540):
    all_entropies = []
    all_top1_probs = []
    all_top1_modes = []
    all_weights = []
    cohort_results = {}

    for dname, dpath in datasets_config.items():
        if dname == "Aneumo" and aneumo_max_case:
            meta_file = os.path.join(dpath, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                test_indices = [m["index"] for m in meta if int(m.get("case_id", 9999)) <= aneumo_max_case]
            else:
                test_indices = sorted([int(f.replace("x_", "").replace(".npy", "")) 
                                       for f in os.listdir(dpath) if f.startswith("x_") and f.endswith(".npy")])[:aneumo_max_case]
        else:
            test_indices = sorted([int(f.replace("x_", "").replace(".npy", "")) 
                                   for f in os.listdir(dpath) if f.startswith("x_") and f.endswith(".npy")])

        cdir_list, cmag_list = [], []
        c_entropies, c_top1_probs = [], []

        for idx in test_indices:
            x_path = os.path.join(dpath, f"x_{idx}.npy")
            y_path = os.path.join(dpath, f"y_{idx}.npy")
            cond_path = os.path.join(dpath, f"cond_{idx}.npy")
            if not (os.path.exists(x_path) and os.path.exists(y_path)):
                continue
            x = np.load(x_path).astype(np.float32)
            y = np.load(y_path).astype(np.float32)
            cond = np.load(cond_path).astype(np.float32) if os.path.exists(cond_path) else np.array([0.0, 0.3, 0.0035], dtype=np.float32)
            pos = x[:, :3]
            u_cfd = y[:, :3] if (y.ndim == 2 and y.shape[1] == 3) else y[:, 1:4]

            if x.shape[1] >= 7:
                x_feat = x[:, :7]
            else:
                centered = pos - pos.mean(axis=0, keepdims=True)
                r = np.linalg.norm(centered, axis=-1)
                dist = (r.max() + 1e-6 - r).astype(np.float32)
                normals = -centered / (r[:, None] + 1e-8)
                x_feat = np.hstack([pos, dist[:, None], normals]).astype(np.float32)

            bank = ds_helper._build_generalized_flow_bank(x_feat, f"{dname}_{idx}")
            if np.dot(bank[:, 0, :].mean(axis=0), u_cfd.mean(axis=0)) < 0:
                bank = -bank

            if dname == "ICA-Sim":
                inlet_v = float(cond[1])
            elif dname == "Aneumo":
                inlet_v = float(cond[3]) if len(cond) > 3 else float(cond[1])
            elif dname in ["VMR-AOCOA", "4DFlow"]:
                inlet_v = float(cond[2]) if len(cond) > 2 else float(cond[1])
            else:
                inlet_v = float(np.linalg.norm(u_cfd, axis=-1).mean())

            with torch.no_grad():
                pos_t = torch.from_numpy(pos).float().unsqueeze(0).to(device)
                morph_t = torch.from_numpy(x_feat).float().unsqueeze(0).to(device)
                cond_t = torch.zeros(1, pos.shape[0], 4, device=device)
                cond_t[:, :, 0] = inlet_v
                basis_t = torch.from_numpy(bank).float().unsqueeze(0).to(device)
                traj_t = torch.zeros(1, pos.shape[0], 9, device=device)
                compact, _, weights = flow_dict(pos_t, morph_t, cond_t, basis_t, rw_trajectory=traj_t)
                u_pred = compact[0, :, :3].cpu().numpy()

            w = weights[0].cpu().numpy() # (N, 8)
            # Point-wise Shannon entropy
            ent = -np.sum(w * np.log(np.clip(w, 1e-8, 1.0)), axis=-1) # (N,)
            top1_p = np.max(w, axis=-1) # (N,)
            top1_m = np.argmax(w, axis=-1) # (N,)

            cd, cm = compute_metrics(u_pred, u_cfd)
            cdir_list.append(cd)
            cmag_list.append(cm)

            all_entropies.append(ent)
            all_top1_probs.append(top1_p)
            all_top1_modes.append(top1_m)
            all_weights.append(w.mean(axis=0)) # mean weights for this case

            c_entropies.append(np.mean(ent))
            c_top1_probs.append(np.mean(top1_p))

        cohort_results[dname] = {
            "cdir_mean": float(np.mean(cdir_list)),
            "cmag_mean": float(np.mean(cmag_list)),
            "entropy_mean": float(np.mean(c_entropies)),
            "top1_prob_mean": float(np.mean(c_top1_probs)),
            "n_samples": len(cdir_list),
        }

    flat_ent = np.concatenate(all_entropies)
    flat_top1_p = np.concatenate(all_top1_probs)
    flat_top1_m = np.concatenate(all_top1_modes)
    mean_weights_per_case = np.vstack(all_weights)

    # Histogram of chosen mode
    mode_counts = np.bincount(flat_top1_m, minlength=8)
    mode_freq = (mode_counts / len(flat_top1_m)).tolist()

    quantiles = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    ent_quantiles = {f"p{int(q*100)}": float(np.quantile(flat_ent, q)) for q in quantiles}
    top1_quantiles = {f"p{int(q*100)}": float(np.quantile(flat_top1_p, q)) for q in quantiles}

    return {
        "cohorts": cohort_results,
        "global": {
            "n_total_points": int(len(flat_ent)),
            "entropy_mean": float(np.mean(flat_ent)),
            "entropy_std": float(np.std(flat_ent)),
            "entropy_quantiles": ent_quantiles,
            "top1_prob_mean": float(np.mean(flat_top1_p)),
            "top1_prob_std": float(np.std(flat_top1_p)),
            "top1_quantiles": top1_quantiles,
            "primitive_weight_mean": float(np.mean(mean_weights_per_case, axis=0)).tolist() if False else np.mean(mean_weights_per_case, axis=0).tolist(),
            "primitive_weight_std": np.std(mean_weights_per_case, axis=0).tolist(),
            "mode_selection_frequency": mode_freq,
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoints = {
        "Full_HemoPT_Baseline": "checkpoints/hemopt_ablation_5fold_f0_exp01_baseline_full_best.pt",
        "No_Entropy_Regularization": "checkpoints/hemopt_ablation_5fold_f0_exp04_loss_no_entropy_best.pt",
    }

    datasets_config = {
        "ICA-Sim": "../GeoPT-main/hemo_npys",
        "Aneumo": "../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004",
        "VMR-AOCOA": "../GeoPT-main/vmr_cfd_npys_quality_filtered",
        "4DFlow": "../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw",
    }

    ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode="learnable_generalized_flow_compact")

    summary = {}
    for name, ckpt_path in checkpoints.items():
        print(f"\n=======================================================")
        print(f" Auditing: {name}")
        print(f" Checkpoint: {ckpt_path}")
        print(f"=======================================================")
        flow_dict = AnalyticCompactFlowDictionary(
            morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64,
            rw_encoder_hidden_dim=16, rw_feature_dim=16
        ).to(args.device)

        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        flow_dict.load_state_dict(ckpt["flow_dict"] if "flow_dict" in ckpt else ckpt)
        flow_dict.eval()

        res = audit_model(flow_dict, ds_helper, datasets_config, args.device, aneumo_max_case=540)
        summary[name] = res
        print(f"  Entropy Mean: {res['global']['entropy_mean']:.4f} +- {res['global']['entropy_std']:.4f}")
        print(f"  Entropy p25/p50/p75: {res['global']['entropy_quantiles']['p25']:.4f} / {res['global']['entropy_quantiles']['p50']:.4f} / {res['global']['entropy_quantiles']['p75']:.4f}")
        print(f"  Top-1 Prob Mean: {res['global']['top1_prob_mean']:.4f} +- {res['global']['top1_prob_std']:.4f}")
        print(f"  Primitive Weight Means: {[round(x, 4) for x in res['global']['primitive_weight_mean']]}")
        print(f"  Top-1 Mode Selection Freq: {[round(x, 4) for x in res['global']['mode_selection_frequency']]}")

    out_path = "results/ablation_results/entropy_deep_audit.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved deep audit to {out_path}")

if __name__ == "__main__":
    main()
