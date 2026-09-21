import os, sys, json, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.dynamic_flow_dict import AnalyticCompactFlowDictionary
from data_provider.vascular_pretrain_loader import _VascularPretrainDataset

def audit_cohort(flow_dict, ds_helper, datasets_config, device):
    all_ent = []
    all_top1_p = []
    all_top1_m = []
    all_w = []

    for dname, (dpath, max_n) in datasets_config.items():
        if dname == "Aneumo":
            meta_file = os.path.join(dpath, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                indices = [m["index"] for m in meta if int(m.get("case_id", 9999)) <= max_n]
            else:
                indices = sorted([int(f.replace("x_", "").replace(".npy", "")) 
                                  for f in os.listdir(dpath) if f.startswith("x_") and f.endswith(".npy")])[:max_n]
        else:
            indices = sorted([int(f.replace("x_", "").replace(".npy", "")) 
                              for f in os.listdir(dpath) if f.startswith("x_") and f.endswith(".npy")])[:max_n]

        for idx in indices:
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

            with torch.no_grad():
                pos_t = torch.from_numpy(pos).float().unsqueeze(0).to(device)
                morph_t = torch.from_numpy(x_feat).float().unsqueeze(0).to(device)
                cond_t = torch.zeros(1, pos.shape[0], 4, device=device)
                cond_t[:, :, 0] = float(np.linalg.norm(u_cfd, axis=-1).mean())
                basis_t = torch.from_numpy(bank).float().unsqueeze(0).to(device)
                traj_t = torch.zeros(1, pos.shape[0], 9, device=device)
                _, _, weights = flow_dict(pos_t, morph_t, cond_t, basis_t, rw_trajectory=traj_t)

            w = weights[0].cpu().numpy() # (N, 8)
            ent = -np.sum(w * np.log(np.clip(w, 1e-8, 1.0)), axis=-1)
            top1_p = np.max(w, axis=-1)
            top1_m = np.argmax(w, axis=-1)

            all_ent.append(ent)
            all_top1_p.append(top1_p)
            all_top1_m.append(top1_m)
            all_w.append(w)

    flat_ent = np.concatenate(all_ent)
    flat_top1_p = np.concatenate(all_top1_p)
    flat_top1_m = np.concatenate(all_top1_m)
    flat_w = np.vstack(all_w)

    mode_counts = np.bincount(flat_top1_m, minlength=8)
    mode_freq = (mode_counts / len(flat_top1_m)).tolist()

    q_keys = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    ent_q = {f"p{int(q*100)}": float(np.quantile(flat_ent, q)) for q in q_keys}
    top1_q = {f"p{int(q*100)}": float(np.quantile(flat_top1_p, q)) for q in q_keys}

    return {
        "n_points": int(len(flat_ent)),
        "ent_mean": float(np.mean(flat_ent)),
        "ent_std": float(np.std(flat_ent)),
        "ent_quantiles": ent_q,
        "top1_prob_mean": float(np.mean(flat_top1_p)),
        "top1_prob_std": float(np.std(flat_top1_p)),
        "top1_quantiles": top1_q,
        "mode_selection_frequency": mode_freq,
        "primitive_weights_mean": [float(x) for x in np.mean(flat_w, axis=0)],
        "primitive_weights_std": [float(x) for x in np.std(flat_w, axis=0)],
    }

def main():
    device = "cuda:2"
    datasets_config = {
        "ICA-Sim": ("../GeoPT-main/hemo_npys", 133),
        "4DFlow": ("../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw", 138),
        "VMR-AOCOA": ("../GeoPT-main/vmr_cfd_npys_quality_filtered", 48),
        "Aneumo": ("../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004", 540),
    }
    ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode="learnable_generalized_flow_compact")

    checkpoints = {
        "Full_HemoPT_Fold0": "checkpoints/hemopt_ablation_5fold_f0_exp01_baseline_full_best.pt",
        "No_Entropy_Fold0": "checkpoints/hemopt_ablation_5fold_f0_exp04_loss_no_entropy_best.pt",
    }

    out = {}
    for name, ckpt_path in checkpoints.items():
        print(f"Auditing {name}...")
        flow_dict = AnalyticCompactFlowDictionary(morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64, rw_encoder_hidden_dim=16, rw_feature_dim=16).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        flow_dict.load_state_dict(ckpt["flow_dict"] if "flow_dict" in ckpt else ckpt)
        flow_dict.eval()
        res = audit_cohort(flow_dict, ds_helper, datasets_config, device)
        out[name] = res
        print(f"  Points: {res['n_points']}")
        print(f"  Entropy: mean={res['ent_mean']:.4f}, std={res['ent_std']:.4f}, p25={res['ent_quantiles']['p25']:.4f}, p50={res['ent_quantiles']['p50']:.4f}, p75={res['ent_quantiles']['p75']:.4f}")
        print(f"  Top1: mean={res['top1_prob_mean']:.4f}, std={res['top1_prob_std']:.4f}, p50={res['top1_quantiles']['p50']:.4f}")
        print(f"  Mode Weights Mean: {[round(x, 4) for x in res['primitive_weights_mean']]}")
        print(f"  Mode Selection Freq: {[round(x, 4) for x in res['mode_selection_frequency']]}")

    out_file = "results/ablation_results/exact_859_entropy_audit.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved exact audit to {out_file}")

if __name__ == "__main__":
    main()
