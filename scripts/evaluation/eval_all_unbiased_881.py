import os, sys, json, torch, numpy as np
sys.path.insert(0, '.')
from models.dynamic_flow_dict import AnalyticCompactFlowDictionary
from data_provider.vascular_pretrain_loader import _VascularPretrainDataset

ds_helper = _VascularPretrainDataset([], physics_proxy=True, physics_proxy_mode='learnable_generalized_flow_compact')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

checkpoints = {
    "HemoPT Dynamic (Full)": "checkpoints/hemopt_ablation_5fold_f0_exp01_baseline_full_best.pt",
    "  w/o Wall No-Slip (L_noslip=0)": "checkpoints/hemopt_ablation_5fold_f0_exp02_loss_no_noslip_best.pt",
    "  w/o Incompressibility (L_div=0)": "checkpoints/hemopt_ablation_5fold_f0_exp03_loss_no_div_best.pt",
    "  w/o Gating Entropy (L_entropy=0)": "checkpoints/hemopt_ablation_5fold_f0_exp04_loss_no_entropy_best.pt",
}

models = {}
for name, ckpt_path in checkpoints.items():
    m = AnalyticCompactFlowDictionary(morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=64, rw_encoder_hidden_dim=16, rw_feature_dim=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['flow_dict'] if 'flow_dict' in ckpt else ckpt)
    m.eval()
    models[name] = m

datasets = {
    'ICA-Sim': ('../GeoPT-main/hemo_npys', 133),
    '4DFlow': ('../GeoPT-main/4dflow_npys_phase4096_strat_zero_uvw', 138),
    'VMR-AOCOA': ('../GeoPT-main/vmr_cfd_npys_quality_filtered', 70),
    'Aneumo': ('../GeoPT-main/aneumo_cfd_npys_1080_wall_aligned_m0p004', 540),
}

configs = [
    'Classical Poiseuille Prior',
    'Uniform 8-Mode Mix',
    'HemoPT Dynamic (Full)',
    '  w/o Wall No-Slip (L_noslip=0)',
    '  w/o Incompressibility (L_div=0)',
    '  w/o Gating Entropy (L_entropy=0)',
]

results = {c: {d: {'cdir': [], 'cmag': []} for d in datasets} for c in configs}

for dname, (ddir, max_n) in datasets.items():
    print(f"Evaluating {dname} (max {max_n})...")
    indices = sorted([int(f.replace('x_', '').replace('.npy', '')) for f in os.listdir(ddir) if f.startswith('x_') and f.endswith('.npy')])[:max_n]
    
    for idx in indices:
        x = np.load(os.path.join(ddir, f'x_{idx}.npy')).astype(np.float32)
        y = np.load(os.path.join(ddir, f'y_{idx}.npy')).astype(np.float32)
        pos = x[:, :3]
        u_cfd = y[:, :3] if (y.ndim==2 and y.shape[1]==3) else y[:, 1:4]
        
        # Proper geometry feature computation
        centered = pos - pos.mean(axis=0, keepdims=True)
        r = np.linalg.norm(centered, axis=-1)
        dist = (r.max() + 1e-6 - r).astype(np.float32)
        normals = -centered / (r[:, None] + 1e-8)
        x_feat = np.hstack([pos, dist[:, None], normals]).astype(np.float32)
        
        bank = ds_helper._build_generalized_flow_bank(x_feat, f'{dname}_{idx}')
        if np.dot(bank[:, 0, :].mean(axis=0), u_cfd.mean(axis=0)) < 0:
            bank = -bank
            
        u_dict = {
            'Classical Poiseuille Prior': bank[:, 0, :],
            'Uniform 8-Mode Mix': bank.mean(axis=1),
        }
        
        # Run neural flow models
        with torch.no_grad():
            pos_t = torch.from_numpy(pos).float().unsqueeze(0).to(device)
            morph_t = torch.from_numpy(x_feat).float().unsqueeze(0).to(device)
            cond_t = torch.zeros(1, pos.shape[0], 4, device=device)
            cond_t[:, :, 0] = float(np.linalg.norm(u_cfd, axis=-1).mean())
            basis_t = torch.from_numpy(bank).float().unsqueeze(0).to(device)
            traj_t = torch.zeros(1, pos.shape[0], 9, device=device)
            
            for mname, m in models.items():
                compact, _, _ = m(pos_t, morph_t, cond_t, basis_t, rw_trajectory=traj_t)
                u_dict[mname] = compact[0, :, :3].cpu().numpy()
                
        # Evaluate metrics for all configs
        speed_gt = np.linalg.norm(u_cfd, axis=-1)
        mean_gt = speed_gt.mean()
        
        for cname in configs:
            u_pred = u_dict[cname]
            dot = (u_pred * u_cfd).sum(axis=-1)
            np_norm = np.linalg.norm(u_pred, axis=-1)
            cos = dot / (np_norm * speed_gt + 1e-6)
            cd = float(np.mean(cos))
            
            # Scale-match to ground truth mean speed
            scale = mean_gt / (np_norm.mean() + 1e-6)
            u_scaled = u_pred * scale
            speed_scaled = np.linalg.norm(u_scaled, axis=-1)
            cm = float(np.sum(np.abs(speed_scaled - speed_gt)) / (np.sum(speed_gt) + 1e-6))
            
            results[cname][dname]['cdir'].append(cd)
            results[cname][dname]['cmag'].append(cm)

summary = {}
for cname in configs:
    summary[cname] = {}
    macro_cd_list = []
    macro_cm_list = []
    for dname in datasets:
        cd_m = float(np.mean(results[cname][dname]['cdir']))
        cm_m = float(np.mean(results[cname][dname]['cmag']))
        summary[cname][dname] = {'cdir': cd_m, 'cmag': cm_m}
        macro_cd_list.append(cd_m)
        macro_cm_list.append(cm_m)
    summary[cname]['Macro_Average'] = {
        'cdir': float(np.mean(macro_cd_list)),
        'cmag': float(np.mean(macro_cm_list)),
    }

pois_cmag_macro = summary['Classical Poiseuille Prior']['Macro_Average']['cmag']
for cname in configs:
    macro_cm = summary[cname]['Macro_Average']['cmag']
    red = ((pois_cmag_macro - macro_cm) / pois_cmag_macro) * 100.0
    summary[cname]['Macro_Average']['rel_red_pct'] = float(red)

with open('results/ablation_results/all_unbiased_881_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nAll 881 cases evaluated successfully!")
