import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser('Fine-Tuning Neural Simulators')

## training
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='maximum epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
parser.add_argument('--pct_start', type=float, default=0.3, help='oncycle lr schedule')
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--device', type=str, default='auto',
                    help='compute device: auto, cuda, cuda:0, or cpu')
parser.add_argument('--max_grad_norm', type=float, default=None, help='make the training stable')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type, select from Adam, AdamW')
parser.add_argument('--scheduler', type=str, default='OneCycleLR',
                    help='learning rate scheduler, select from [OneCycleLR, CosineAnnealingLR, StepLR]')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='warmup epochs for CosineAnnealingLR in vascular pre-training')
parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay parameter for StepLR scheduler')
parser.add_argument('--early_stop', type=str2bool, default=False, help='enable validation early stopping')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience in epochs')
parser.add_argument('--min_delta', type=float, default=0.0, help='minimum validation improvement for early stopping')
parser.add_argument('--checkpoint_interval', type=int, default=0,
                    help='save an extra checkpoint every N epochs when > 0')

## data
parser.add_argument('--data_path', type=str, default='./data', help='data folder, should change accordingly')
parser.add_argument('--loader', type=str, default='airfoil', help='type of data loader')
parser.add_argument('--ntrain', type=int, default=1000, help='training data numbers')
parser.add_argument('--ntest', type=int, default=200, help='test data numbers')
parser.add_argument('--normalize', type=str2bool, default=False, help='make normalization to output')
parser.add_argument('--norm_type', type=str, default='UnitTransformer',
                    help='dataset normalize type. select from [UnitTransformer, UnitGaussianNormalizer]')
parser.add_argument('--geotype', type=str, default='unstructured',
                    help='select from [unstructured, structured_1D, structured_2D, structured_3D]')
parser.add_argument('--space_dim', type=int, default=2, help='position information dimension')
parser.add_argument('--fun_dim', type=int, default=0, help='input observation dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output observation dimension')

## task
parser.add_argument('--task', type=str, required=True,
                    choices=['steady_cond', 'vascular_pretrain', 'hemo_cfd_finetune'],
                    help='select from [steady_cond, vascular_pretrain, hemo_cfd_finetune]')
parser.add_argument('--dynamics', type=str, default='hull',
                    help='select from [hull, craft, drivAerml, nasa, crash]')
parser.add_argument('--n_random_walks', type=int, default=20,
                    help='number of random-walk probe views per vascular geometry')
parser.add_argument('--base_walks', type=int, default=20,
                    help='number of base random walks before perturbation')
parser.add_argument('--walk_steps', type=int, default=3,
                    help='number of random-walk trajectory steps per probe (default: 3)')
parser.add_argument('--vascular_qc_statuses', type=str, nargs='+', default=['pass', 'warn'],
                    help='meta.json qc_status values allowed by the VascularPretrain loader')
parser.add_argument('--vascular_qc_manifest', type=str,
                    default='',
                    help='QC manifest used by VascularPretrain loader for old samples without meta qc_status')
parser.add_argument('--vascular_physics_proxy', type=str2bool, default=False,
                    help='append physics proxy targets to VascularPretrain supervision')
parser.add_argument('--vascular_physics_proxy_mode', type=str, default='full',
                    choices=[
                        'full',
                        'velocity_only',
                        'conditioned_velocity',
                        'generalized_flow_compact',
                        'conditioned_generalized_flow_compact',
                        'generalized_flow_bank',
                        'learnable_dynamic_flow_dict',
                        'learnable_generalized_flow_compact',
                    ],
                    help='full appends scalar proxies; velocity modes append flow/speed targets; '
                         'learnable_generalized_flow_compact uses a fixed 8-mode analytic bank with an 8-way gate')
parser.add_argument('--vascular_physics_weight', type=float, default=0.25,
                    help='loss weight for VascularPretrain physics proxy targets')
parser.add_argument('--vascular_wall_mask_prob', type=float, default=0.0,
                    help='per-sample probability of masking wall geometry channels during VascularPretrain training')
parser.add_argument('--dict_num_modes', type=int, default=4,
                    help='number of learnable flow modes in the legacy NeuralFlowDictionary; '
                         'learnable_generalized_flow_compact always uses 8 analytic modes')
parser.add_argument('--dict_loss_weight_noslip', type=float, default=0.1,
                    help='loss weight for wall no-slip condition in flow dictionary')
parser.add_argument('--dict_loss_weight_div', type=float, default=0.05,
                    help='loss weight for divergence penalty in flow dictionary')
parser.add_argument('--dict_loss_weight_energy', type=float, default=0.1,
                    help='loss weight for kinetic energy scale constraint in flow dictionary')
parser.add_argument('--dict_loss_weight_entropy', type=float, default=0.01,
                    help='loss weight for routing gate entropy regularizer in flow dictionary')
parser.add_argument('--dict_loss_weight_ortho', type=float, default=0.05,
                    help='loss weight for mode orthogonality regularizer in flow dictionary')
parser.add_argument('--rw_encoder_hidden_dim', type=int, default=16,
                    help='hidden width of the three-step random-walk trajectory encoder')
parser.add_argument('--rw_encoder_feature_dim', type=int, default=16,
                    help='point-wise feature width emitted by the random-walk trajectory encoder')
parser.add_argument('--vascular_wall_mask_mode', type=str, default='all',
                    choices=['all', 'distance', 'direction'],
                    help='which wall geometry channels to mask when vascular_wall_mask_prob triggers')
parser.add_argument('--num_workers', type=int, default=0,
                    help='DataLoader worker processes for loaders that support it')
parser.add_argument('--pin_memory', type=str2bool, default=False,
                    help='pin DataLoader memory for faster host-to-GPU transfer')
parser.add_argument('--prefetch_factor', type=int, default=2,
                    help='DataLoader prefetch factor when num_workers > 0')
parser.add_argument('--seed', type=int, default=2026,
                    help='random seed for model initialization and dataloader shuffling')
parser.add_argument('--deterministic', type=str2bool, default=False,
                    help='enable deterministic torch/cuDNN algorithms when possible')
parser.add_argument('--hemo_configs', type=str, nargs='+', default=['C1', 'ICA_norm', 'ICA_ste'],
                    help='HemoPT CFD subsets to use, selected from [C1, ICA_norm, ICA_ste]')
parser.add_argument('--hemo_data_path', type=str, default='./hemo_npys',
                    help='HemoPT npy folder used by mixed HemoVMRCFD training')
parser.add_argument('--hemo_ntrain', type=int, default=100000,
                    help='maximum HemoPT training samples used by mixed HemoVMRCFD training')
parser.add_argument('--hemo_vmr_norm_scope', type=str, default='vmr_train',
                    choices=['vmr_train', 'mixed_train'],
                    help='normalizer fit scope for mixed HemoVMRCFD training')
parser.add_argument('--vmr_eval_split', type=str, default='test', choices=['val', 'test'],
                    help='VMR CFD split used by the evaluation loader')
parser.add_argument('--vmr_fold_val_frac', type=float, default=0.15,
                    help='validation fraction carved from non-test VMR folds when --kfold > 1')
parser.add_argument('--vmr_kfold_strategy', type=str, default='group',
                    choices=['group', 'stratified_group'],
                    help='VMR k-fold split strategy when --kfold > 1')
parser.add_argument('--vmr_stratify_bins', type=int, default=3,
                    help='number of VMR speed bins for --vmr_kfold_strategy stratified_group')
parser.add_argument('--vmr_families', type=str, nargs='+', default=None,
                    help='optional VMR family filter applied to both train and eval, e.g. AO ABAO PULM')
parser.add_argument('--vmr_territories', type=str, nargs='+', default=None,
                    help='optional VMR territory filter applied to both train and eval, e.g. AO_COA ABAO_AAA')
parser.add_argument('--vmr_train_families', type=str, nargs='+', default=None,
                    help='optional VMR family filter applied only to training split')
parser.add_argument('--vmr_train_territories', type=str, nargs='+', default=None,
                    help='optional VMR territory filter applied only to training split')
parser.add_argument('--vmr_eval_families', type=str, nargs='+', default=None,
                    help='optional VMR family filter applied only to eval split for per-family breakdown')
parser.add_argument('--vmr_eval_territories', type=str, nargs='+', default=None,
                    help='optional VMR territory filter applied only to eval split for per-territory breakdown')
parser.add_argument('--aneumo_eval_split', type=str, default='test', choices=['val', 'test'],
                    help='aneumo CFD split used by the evaluation loader')
parser.add_argument('--hemo_eval_split', type=str, default='test', choices=['val', 'test'],
                    help='HemoPT CFD split used by the evaluation loader when split info has val/test indices')
parser.add_argument('--hemo_split_info', type=str, default=None,
                    help='optional HemoPT split info .npy with train_indices, val_indices, and test_indices')
parser.add_argument('--cv_split_info', type=str, default=None,
                    help='optional strict CV split .npy. Folds are drawn from the original train/val pool, while test_indices stay independent')
parser.add_argument('--kfold', type=int, default=0,
                    help='number of folds for HemoPT CFD validation; disabled when <= 1')
parser.add_argument('--fold', type=int, default=0,
                    help='fold index for HemoPT CFD validation')
parser.add_argument('--fold_seed', type=int, default=2026,
                    help='random seed used to create HemoPT CFD folds')
parser.add_argument('--hemo_loss_mode', type=str, default='global_rel',
                    choices=['global_rel', 'channel_rel', 'physics_rel', 'velocity_rel'],
                    help='loss for Hemo/VMR CFD fine-tuning; velocity_rel ignores pressure and optimizes decoded velocity')
parser.add_argument('--hemo_target', type=str, default='full',
                    choices=['full', 'velocity'],
                    help='target channels for Hemo/VMR CFD fine-tuning; velocity uses [u,v,w] only before normalization')
parser.add_argument('--hemo_select_metric', type=str, default='rel_l2',
                    choices=['rel_l2', 'balanced_rel_l2', 'weighted_balanced_rel_l2',
                             'vel_rel_l2', 'vel_vector_rel_l2', 'vel_mag_rel_l2',
                             'velocity_balanced_rel_l2', 'physics_balanced_rel_l2'],
                    help='validation metric used to save best Hemo/VMR CFD checkpoint')
parser.add_argument('--hemo_channel_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0],
                    help='channel weights for Hemo/VMR CFD channel_rel loss and weighted balanced metric: p u v w, or u v w for velocity target')
parser.add_argument('--hemo_pressure_weight', type=float, default=1.0,
                    help='physics_rel loss/select weight for absolute pressure relative error')
parser.add_argument('--hemo_pressure_center_weight', type=float, default=1.0,
                    help='physics_rel loss/select weight for mean-free pressure shape error')
parser.add_argument('--hemo_velocity_weight', type=float, default=1.0,
                    help='physics_rel loss/select weight for vector velocity relative error')
parser.add_argument('--hemo_velocity_mag_weight', type=float, default=0.5,
                    help='physics_rel loss/select weight for velocity magnitude relative error')
parser.add_argument('--hemo_wall_mode', type=str, default='keep', choices=['keep', 'zero'],
                    help='for Hemo/VMR CFD fine-tuning, keep wall geometry channels or zero them for no-wall ablation')
parser.add_argument('--hemo_prompt_mode', type=str, default='flow_geo',
                    choices=['none', 'geometry', 'flow_geo'],
                    help='downstream prompt for Hemo/VMR/Aneumo/4DFlow CFD fine-tuning: none, geometry-only, or hemodynamic flow/geometry')
## models
parser.add_argument('--model', type=str, default='Transolver')
parser.add_argument('--n_hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n_layers', type=int, default=3, help='layers')
parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--mlp_ratio', type=int, default=1, help='mlp ratio for feedforward layers')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--checkpoint', type=int, default=0, help='using gradient checkpoint or not')

## model specific configuration
parser.add_argument('--slice_num', type=int, default=32, help='number of physical states for Transolver')

## eval
parser.add_argument('--eval', type=int, default=0, help='evaluation or not')
parser.add_argument('--save_name', type=str, default='Transolver_check', help='name of folders')
parser.add_argument('--vis_num', type=int, default=10, help='number of visualization cases')
parser.add_argument('--vis_bound', type=int, nargs='+', default=None, help='size of region for visualization, in list')
parser.add_argument('--visualize', type=str2bool, default=False, help='save hemo CFD prediction visualizations')

## finetune
parser.add_argument('--finetune', type=int, default=0, help='finetune or not')
parser.add_argument('--finetune_name', type=str, default='Transolver_check', help='name of folders')
parser.add_argument('--freeze_pretrained_epochs', type=int, default=0,
                    help='for hemo CFD fine-tuning, freeze non-head pretrained layers for this many epochs')
parser.add_argument('--backbone_lr', type=float, default=None,
                    help='optional learning rate for non-head parameters during hemo CFD fine-tuning')
parser.add_argument('--head_lr', type=float, default=None,
                    help='optional learning rate for output-head parameters during hemo CFD fine-tuning')

args = parser.parse_args()
if args.hemo_target == 'velocity' and args.out_dim != 3:
    parser.error("--hemo_target velocity requires --out_dim 3 so the model head predicts [u,v,w] only")
if args.hemo_target == 'velocity' and args.hemo_loss_mode == 'physics_rel':
    parser.error("--hemo_loss_mode physics_rel requires pressure; use velocity_rel with --hemo_target velocity")
if args.hemo_target == 'velocity' and args.hemo_select_metric == 'physics_balanced_rel_l2':
    parser.error("--hemo_select_metric physics_balanced_rel_l2 requires pressure; use velocity metrics with --hemo_target velocity")
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(args.seed, args.deterministic)


def main():

    if args.task == 'steady_cond':
        from exp.steady_cond import Exp_Steady
        exp = Exp_Steady(args)
    elif args.task == 'vascular_pretrain':
        from exp.vascular_pretrain import Exp_VascularPretrain
        exp = Exp_VascularPretrain(args)
    elif args.task == 'hemo_cfd_finetune':
        from exp.hemo_cfd_finetune import Exp_HemoCFDFinetune
        exp = Exp_HemoCFDFinetune(args)
    else:
        raise ValueError('task not supported')

    if eval:
        exp.test()
        exp.test_full_mesh()
    else:
        exp.train()
        exp.test()
        exp.test_full_mesh()

if __name__ == "__main__":
    main()
