import os, sys, glob, json, argparse, torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from models.dynamic_flow_dict import AnalyticCompactFlowDictionary
from data_provider.vascular_pretrain_loader import _VascularPretrainDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/rw_encoder_routing_probe_15ep_best.pt")
    parser.add_argument("--hemo_dir", type=str, default="HemoData/hemo_npys")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default="results/proxy_alignment_results.json")
    args = parser.parse_args()

    print("==========================================================================")
    print(f" Flow Proxy Alignment Assessment (C_dir, C_mag)")
    print(f" Checkpoint: {args.ckpt}")
    print(f" Ground Truth CFD Data: {args.hemo_dir}")
    print(f" Device: {args.device}")
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

    # 2. Gather paired CFD cases
    x_files = sorted(glob.glob(os.path.join(args.hemo_dir, "x_*.npy")), key=lambda p: int(os.path.basename(p).replace("x_", "").replace(".npy", "")))
    print(f"Found {len(x_files)} candidate CFD cases.")

    categories = ["C1", "ICA_norm", "ICA_ste", "Overall"]
    cat_metrics = {cat: {m: {"cdir": [], "cmag": []} for m in ["poiseuille", "uniform", "hemopt"]} for cat in categories}

    for x_path in x_files:
        idx_str = os.path.basename(x_path).replace("x_", "").replace(".npy", "")
        idx = int(idx_str)
        y_path = os.path.join(args.hemo_dir, f"y_{idx}.npy")
        cond_path = os.path.join(args.hemo_dir, f"cond_{idx}.npy")
        if not (os.path.exists(y_path) and os.path.exists(cond_path)):
            continue

        x = np.load(x_path) # (N, 6)
        y = np.load(y_path) # (N, 4) [p, u, v, w]
        cond = np.load(cond_path) # (3,) [vessel_type, inlet_velocity, viscosity]

        # Determine category
        v_type = int(cond[0])
        cat = "C1" if v_type == 0 else ("ICA_norm" if v_type == 1 else "ICA_ste")

        pos = x[:, :3]
        u_cfd = y[:, 1:4] # (N, 3)

        # Approximate wall distance and normals from coordinates
        centered = pos - pos.mean(axis=0, keepdims=True)
        r = np.linalg.norm(centered, axis=-1)
        r_max = r.max() + 1e-6
        dist = (r_max - r).astype(np.float32)
        normals = -centered / (r[:, None] + 1e-8)
        x_feat = np.hstack([pos, dist[:, None], normals]).astype(np.float32) # (N, 7)

        # Build 8-mode canonical bank
        bank = ds_helper._build_generalized_flow_bank(x_feat, f"sample_{idx}") # (N, 8, 3)
        u_poi = bank[:, 0, :] # Mode 1: Static Poiseuille Prior
        u_uni = bank.mean(axis=1) # Uniform 8-mode average

        # Synthesize HemoPT proxy
        with torch.no_grad():
            pos_t = torch.from_numpy(pos).float().unsqueeze(0).to(args.device)
            morph_t = torch.from_numpy(x_feat).float().unsqueeze(0).to(args.device)
            cond_t = torch.zeros(1, pos.shape[0], 4, device=args.device)
            cond_t[:, :, 0] = float(cond[1]) # inlet velocity
            basis_t = torch.from_numpy(bank).float().unsqueeze(0).to(args.device)
            traj_t = torch.zeros(1, pos.shape[0], 9, device=args.device)

            compact, _, weights = flow_dict(pos_t, morph_t, cond_t, basis_t, rw_trajectory=traj_t)
            u_hemopt = compact[0, :, :3].cpu().numpy()

        def compute_cdir_cmag(u_pred, u_gt):
            dot = (u_pred * u_gt).sum(axis=-1)
            norm_p = np.linalg.norm(u_pred, axis=-1)
            norm_g = np.linalg.norm(u_gt, axis=-1)
            cos = dot / (norm_p * norm_g + 1e-6)
            cdir = float(np.mean(cos))
            cmag = float(np.sum(np.abs(norm_p - norm_g)) / (np.sum(norm_g) + 1e-6))
            return cdir, cmag

        cp_p, cm_p = compute_cdir_cmag(u_poi, u_cfd)
        cp_u, cm_u = compute_cdir_cmag(u_uni, u_cfd)
        cp_h, cm_h = compute_cdir_cmag(u_hemopt, u_cfd)

        for m_name, (cd, cm) in [("poiseuille", (cp_p, cm_p)), ("uniform", (cp_u, cm_u)), ("hemopt", (cp_h, cm_h))]:
            cat_metrics[cat][m_name]["cdir"].append(cd)
            cat_metrics[cat][m_name]["cmag"].append(cm)
            cat_metrics["Overall"][m_name]["cdir"].append(cd)
            cat_metrics["Overall"][m_name]["cmag"].append(cm)

    # Summarize results
    summary = {}
    print("\n==========================================================================")
    print("                    QUANTITATIVE ALIGNMENT RESULTS")
    print("==========================================================================")
    print(f"{'Category':<12} | {'Method':<24} | {'C_dir (Directional) ^':<22} | {'C_mag (Magnitude) v':<22}")
    print("-" * 88)

    for cat in ["C1", "ICA_norm", "ICA_ste", "Overall"]:
        summary[cat] = {}
        for m_name, label in [("poiseuille", "Static Poiseuille (U1)"), ("uniform", "Uniform 8-Mode Avg"), ("hemopt", "HemoPT (Dynamic Proxy)")]:
            cd_arr = np.array(cat_metrics[cat][m_name]["cdir"])
            cm_arr = np.array(cat_metrics[cat][m_name]["cmag"])
            cd_mean, cd_std = float(np.mean(cd_arr)), float(np.std(cd_arr))
            cm_mean, cm_std = float(np.mean(cm_arr)), float(np.std(cm_arr))
            summary[cat][m_name] = {
                "cdir_mean": cd_mean, "cdir_std": cd_std,
                "cmag_mean": cm_mean, "cmag_std": cm_std,
                "n_samples": len(cd_arr)
            }
            print(f"{cat:<12} | {label:<24} | {cd_mean:.4f} +/- {cd_std:.4f}         | {cm_mean:.4f} +/- {cm_std:.4f}")
        print("-" * 88)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Assessment complete. Results saved to {args.out_json}")

if __name__ == "__main__":
    main()
