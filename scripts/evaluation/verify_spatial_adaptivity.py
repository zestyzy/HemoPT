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
    scale = np.mean(norm_g) / (np.mean(norm_p) + 1e-6)
    u_scaled = u_pred * scale
    norm_s = np.linalg.norm(u_scaled, axis=-1)
    cmag = float(np.sum(np.abs(norm_s - norm_g)) / (np.sum(norm_g) + 1e-6))
    return cdir, cmag

def main():
    device = "cuda:2"
    ckpt_path = "checkpoints/hemopt_ablation_5fold_f0_exp01_baseline_full_best.pt"
    flow_dict = AnalyticCompactFlowDictionary(morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64, rw_encoder_hidden_dim=16, rw_feature_dim=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    flow_dict.load_state_dict(ckpt["flow_dict"] if "flow_dict" in ckpt else ckpt)
    flow_dict.eval()

    ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode="learnable_generalized_flow_compact")

    # Strictly use the 859 benchmark cohort configuration:
    # ICA-Sim: 133, 4DFlow: 138, Aneumo: 540, VMR-AOCOA: 48
    datasets_config = {
        "ICA-Sim": ("../GeoPT-main/hemo_npys", 133),
        "4DFlow": ("../GeoPT-main/4dflow_npys_phase4096_strat", 138),
        "VMR-AOCOA": ("../GeoPT-main/vmr_cfd_npys_quality_filtered", 48),
        "Aneumo": ("../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004", 540),
    }

    # Pass 1: Collect predictions, true CFD, and compute the dataset-wide global mean weight vector
    all_weights = []
    cases_data = []

    print("Pass 1: Running inference across all 859 cases...")
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
            all_weights.append(w)
            cases_data.append({
                "dname": dname,
                "bank": bank,
                "u_cfd": u_cfd,
                "weights": w,
            })

    print(f"Total benchmark cases loaded: {len(cases_data)}")
    flat_weights = np.vstack(all_weights) # (Total_Points, 8)
    global_mean_w = np.mean(flat_weights, axis=0) # (8,)
    print(f"Global mean weights: {[round(float(x), 4) for x in global_mean_w]}")

    # Spatial variance within case: compute mean point-to-point std for each mode
    within_case_stds = [np.std(c["weights"], axis=0) for c in cases_data]
    mean_within_case_std = np.mean(within_case_stds, axis=0)
    print(f"Mean spatial within-case std of weights across points: {[round(float(x), 4) for x in mean_within_case_std]}")

    # Pass 2: Evaluate 4 routing regimes on identical cases:
    # 1. Learned Dynamic Routing: w_i = w_i
    # 2. Shuffled Routing: w_i = w_pi(i) (scramble spatial correspondence)
    # 3. Fixed Global Mean Routing: w_i = global_mean_w
    # 4. Uniform Routing: w_i = 1/8
    # 5. Pure Poiseuille: w_i = [1, 0, 0, 0, 0, 0, 0, 0]

    eval_modes = [
        "Dynamic Learned Routing",
        "Spatial Shuffled Routing",
        "Fixed Global Mean Routing",
        "Uniform 8-Mode Routing",
        "Classical Poiseuille Prior",
    ]

    results = {m: {d: {"cdir": [], "cmag": []} for d in datasets_config} for m in eval_modes}

    np.random.seed(42)
    for c in cases_data:
        dname = c["dname"]
        bank = c["bank"]
        u_cfd = c["u_cfd"]
        w = c["weights"]
        N = w.shape[0]

        # 1. Dynamic
        u_dynamic = np.sum(w[:, :, None] * bank, axis=1)

        # 2. Shuffled
        perm = np.random.permutation(N)
        w_shuffled = w[perm]
        u_shuffled = np.sum(w_shuffled[:, :, None] * bank, axis=1)

        # 3. Fixed Global Mean
        u_fixed = np.sum(global_mean_w[None, :, None] * bank, axis=1)

        # 4. Uniform
        u_uniform = np.mean(bank, axis=1)

        # 5. Poiseuille
        u_poiseuille = bank[:, 0, :]

        preds = {
            "Dynamic Learned Routing": u_dynamic,
            "Spatial Shuffled Routing": u_shuffled,
            "Fixed Global Mean Routing": u_fixed,
            "Uniform 8-Mode Routing": u_uniform,
            "Classical Poiseuille Prior": u_poiseuille,
        }

        for mname, u_p in preds.items():
            cd, cm = compute_metrics(u_p, u_cfd)
            results[mname][dname]["cdir"].append(cd)
            results[mname][dname]["cmag"].append(cm)

    summary = {}
    for mname in eval_modes:
        summary[mname] = {}
        macro_cd = []
        macro_cm = []
        for dname in datasets_config:
            cd_m = float(np.mean(results[mname][dname]["cdir"]))
            cm_m = float(np.mean(results[mname][dname]["cmag"]))
            summary[mname][dname] = {"cdir": cd_m, "cmag": cm_m}
            macro_cd.append(cd_m)
            macro_cm.append(cm_m)
        summary[mname]["Macro_Average"] = {
            "cdir": float(np.mean(macro_cd)),
            "cmag": float(np.mean(macro_cm)),
        }

    pois_cm = summary["Classical Poiseuille Prior"]["Macro_Average"]["cmag"]
    for mname in eval_modes:
        cm = summary[mname]["Macro_Average"]["cmag"]
        red = ((pois_cm - cm) / pois_cm) * 100.0
        summary[mname]["Macro_Average"]["rel_err_red_pct"] = float(red)

    out_file = "results/ablation_results/spatial_adaptivity_audit.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved spatial adaptivity audit to {out_file}")

    print("\n" + "="*80)
    print(f"{'Condition':<30} | {'Macro C_dir':<12} | {'Macro C_mag':<12} | {'Rel Err Red vs Poiseuille'}")
    print("-" * 80)
    for mname in eval_modes:
        s = summary[mname]["Macro_Average"]
        print(f"{mname:<30} | {s['cdir']:<12.4f} | {s['cmag']:<12.4f} | {s['rel_err_red_pct']:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
