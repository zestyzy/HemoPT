import os, sys, json, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.dynamic_flow_dict import AnalyticCompactFlowDictionary
from data_provider.vascular_pretrain_loader import _VascularPretrainDataset

def audit_checkpoint(flow_dict, ds_helper, datasets_config, device, aneumo_max_case=540):
    all_entropies = []
    all_top1_probs = []
    all_top1_modes = []
    all_weights = []

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

            w = weights[0].cpu().numpy() # (N, 8)
            ent = -np.sum(w * np.log(np.clip(w, 1e-8, 1.0)), axis=-1)
            top1_p = np.max(w, axis=-1)
            top1_m = np.argmax(w, axis=-1)

            all_entropies.append(ent)
            all_top1_probs.append(top1_p)
            all_top1_modes.append(top1_m)
            all_weights.append(w.mean(axis=0))

    flat_ent = np.concatenate(all_entropies)
    flat_top1_p = np.concatenate(all_top1_probs)
    flat_top1_m = np.concatenate(all_top1_modes)
    mean_w = np.vstack(all_weights).mean(axis=0)

    mode_counts = np.bincount(flat_top1_m, minlength=8)
    mode_freq = (mode_counts / len(flat_top1_m)).tolist()

    return {
        "ent_mean": float(np.mean(flat_ent)),
        "ent_std": float(np.std(flat_ent)),
        "ent_p25": float(np.quantile(flat_ent, 0.25)),
        "ent_p50": float(np.quantile(flat_ent, 0.50)),
        "ent_p75": float(np.quantile(flat_ent, 0.75)),
        "top1_prob_mean": float(np.mean(flat_top1_p)),
        "top1_prob_std": float(np.std(flat_top1_p)),
        "weights_mean": mean_w.tolist(),
        "mode_selection_freq": mode_freq,
    }

def main():
    device = "cuda:2"
    datasets_config = {
        "ICA-Sim": "../GeoPT-main/hemo_npys",
        "Aneumo": "../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004",
        "VMR-AOCOA": "../GeoPT-main/vmr_cfd_npys_quality_filtered",
        "4DFlow": "../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw",
    }
    ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode="learnable_generalized_flow_compact")

    results = {"Full_HemoPT": [], "No_Entropy": []}

    for fold in range(5):
        print(f"\nEvaluating Fold {fold}...")
        for name, exp_id in [("Full_HemoPT", "exp01_baseline_full"), ("No_Entropy", "exp04_loss_no_entropy")]:
            ckpt_path = f"checkpoints/hemopt_ablation_5fold_f{fold}_{exp_id}_best.pt"
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_path}, not found")
                continue
            flow_dict = AnalyticCompactFlowDictionary(morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64, rw_encoder_hidden_dim=16, rw_feature_dim=16).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            flow_dict.load_state_dict(ckpt["flow_dict"] if "flow_dict" in ckpt else ckpt)
            flow_dict.eval()
            res = audit_checkpoint(flow_dict, ds_helper, datasets_config, device, aneumo_max_case=540)
            res["fold"] = fold
            results[name].append(res)
            print(f"  [{name} Fold {fold}] ent_mean={res['ent_mean']:.4f}, ent_med={res['ent_p50']:.4f}, top1={res['top1_prob_mean']:.4f}")

    with open("results/ablation_results/entropy_5fold_deep_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved 5-fold entropy audit to results/ablation_results/entropy_5fold_deep_audit.json")

if __name__ == "__main__":
    main()
