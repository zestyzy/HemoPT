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

def evaluate_single_cohort(name, data_dir, flow_dict, ds_helper, device, aneumo_max_case=540):
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

    cdir_list, cmag_list = [], []
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
        u_cfd = y[:, :3] if (y.ndim == 2 and y.shape[1] == 3) else y[:, 1:4]

        if x.shape[1] >= 7:
            x_feat = x[:, :7]
        else:
            centered = pos - pos.mean(axis=0, keepdims=True)
            r = np.linalg.norm(centered, axis=-1)
            dist = (r.max() + 1e-6 - r).astype(np.float32)
            normals = -centered / (r[:, None] + 1e-8)
            x_feat = np.hstack([pos, dist[:, None], normals]).astype(np.float32)

        bank = ds_helper._build_generalized_flow_bank(x_feat, f"{name}_{idx}")
        if np.dot(bank[:, 0, :].mean(axis=0), u_cfd.mean(axis=0)) < 0:
            bank = -bank

        if name == "ICA-Sim":
            inlet_v = float(cond[1])
        elif name == "Aneumo":
            inlet_v = float(cond[3]) if len(cond) > 3 else float(cond[1])
        elif name in ["VMR-AOCOA", "4DFlow"]:
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
            compact, _, _ = flow_dict(pos_t, morph_t, cond_t, basis_t, rw_trajectory=traj_t)
            u_pred = compact[0, :, :3].cpu().numpy()

        cd, cm = compute_metrics(u_pred, u_cfd)
        cdir_list.append(cd)
        cmag_list.append(cm)

    return {
        "cdir_mean": float(np.mean(cdir_list)),
        "cdir_std": float(np.std(cdir_list)),
        "cmag_mean": float(np.mean(cmag_list)),
        "cmag_std": float(np.std(cmag_list)),
        "n_samples": len(cdir_list),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default="results/ablation_results/physics_loss_fullcohort_alignment.json")
    args = parser.parse_args()

    checkpoints = {
        "Full HemoPT (Baseline)": "checkpoints/hemopt_ablation_5fold_f0_exp01_baseline_full_best.pt",
        "w/o No-Slip (L_noslip=0)": "checkpoints/hemopt_ablation_5fold_f0_exp02_loss_no_noslip_best.pt",
        "w/o Incompressibility (L_div=0)": "checkpoints/hemopt_ablation_5fold_f0_exp03_loss_no_div_best.pt",
        "w/o Gating Entropy (L_entropy=0)": "checkpoints/hemopt_ablation_5fold_f0_exp04_loss_no_entropy_best.pt",
    }

    datasets_config = {
        "ICA-Sim": "../GeoPT-main/hemo_npys",
        "Aneumo": "../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004",
        "VMR-AOCOA": "../GeoPT-main/vmr_cfd_npys_quality_filtered",
        "4DFlow": "../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw",
    }

    ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode="learnable_generalized_flow_compact")

    results = {}
    for exp_name, ckpt_path in checkpoints.items():
        print(f"\n=======================================================")
        print(f" Evaluating: {exp_name}")
        print(f" Checkpoint: {ckpt_path}")
        print(f"=======================================================")
        flow_dict = AnalyticCompactFlowDictionary(
            morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64,
            rw_encoder_hidden_dim=16, rw_feature_dim=16
        ).to(args.device)

        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        flow_dict.load_state_dict(ckpt["flow_dict"] if "flow_dict" in ckpt else ckpt)
        flow_dict.eval()

        exp_res = {}
        for dname, dpath in datasets_config.items():
            print(f"  --> Cohort {dname}...")
            r = evaluate_single_cohort(dname, dpath, flow_dict, ds_helper, args.device, aneumo_max_case=540)
            exp_res[dname] = r
            print(f"      C_dir: {r['cdir_mean']:.4f}, C_mag: {r['cmag_mean']:.4f} (N={r['n_samples']})")

        # Macro average
        macro_cdir = float(np.mean([exp_res[d]['cdir_mean'] for d in datasets_config]))
        macro_cmag = float(np.mean([exp_res[d]['cmag_mean'] for d in datasets_config]))
        exp_res["Macro_Average"] = {
            "cdir_mean": macro_cdir,
            "cmag_mean": macro_cmag,
        }
        print(f"  ==> Macro Average: C_dir={macro_cdir:.4f}, C_mag={macro_cmag:.4f}")
        results[exp_name] = exp_res

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_json}")

if __name__ == "__main__":
    main()
