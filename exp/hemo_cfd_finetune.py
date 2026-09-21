import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from utils.loss import L2Loss


class Exp_HemoCFDFinetune(Exp_Basic):
    """CFD fine-tuning for interior blood-flow point clouds."""

    def _channel_weights(self, device, dtype, n_channels=None):
        weights = getattr(self.args, "hemo_channel_weights", [1.0, 1.0, 1.0, 1.0])
        w = torch.tensor(weights, device=device, dtype=dtype)
        if n_channels is None:
            n_channels = 3 if getattr(self.args, "hemo_target", "full") == "velocity" else 4
        if w.numel() == n_channels:
            return w.view(1, n_channels)
        if n_channels == 3 and w.numel() == 4:
            return w[1:4].view(1, 3)
        raise ValueError(
            f"--hemo_channel_weights has {w.numel()} values, but target has {n_channels} channels")

    def _physics_weights(self, device, dtype):
        weights = [
            getattr(self.args, "hemo_pressure_weight", 1.0),
            getattr(self.args, "hemo_pressure_center_weight", 1.0),
            getattr(self.args, "hemo_velocity_weight", 1.0),
            getattr(self.args, "hemo_velocity_mag_weight", 0.5),
        ]
        return torch.tensor(weights, device=device, dtype=dtype).view(1, 4)

    @staticmethod
    def _velocity_slice(tensor):
        if tensor.shape[-1] == 3:
            return tensor
        if tensor.shape[-1] == 4:
            return tensor[:, :, 1:4]
        raise ValueError(f"Hemo/VMR CFD output must have 3 velocity channels or 4 full channels, got {tensor.shape[-1]}")

    @staticmethod
    def _has_pressure(tensor):
        return tensor.shape[-1] == 4

    @staticmethod
    def _split_label(n_channels):
        if n_channels == 3:
            return "u,v,w"
        if n_channels == 4:
            return "p,u,v,w"
        return ",".join(f"c{i}" for i in range(n_channels))

    def _load_pretrained_with_filter(self, pretrained_name):
        path = pretrained_name
        if not os.path.exists(path):
            path = os.path.join("./checkpoints", pretrained_name + ".pt")
        pretrained = torch.load(path, map_location="cpu")
        if isinstance(pretrained, dict):
            for candidate_key in ("model", "state_dict", "model_state"):
                if candidate_key in pretrained and isinstance(pretrained[candidate_key], dict):
                    pretrained = pretrained[candidate_key]
                    break
        model_state = self.model.state_dict()
        filtered = {
            k: v for k, v in pretrained.items()
            if k in model_state
            and model_state[k].shape == v.shape
            and "mlp2" not in k
            and "ln_3" not in k
        }
        if len(filtered) == 0:
            print(f"[Pretrain WARNING] 0 compatible parameters were loaded from {path}! Check the checkpoint structure or keys.")
        model_state.update(filtered)
        self.model.load_state_dict(model_state)
        print(f"[Pretrain] Loaded {len(filtered)}/{len(pretrained)} compatible parameters from {path}")

    @staticmethod
    def _geometry_prompt(pos):
        centered = pos - pos.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / max(pos.shape[1] - 1, 1)
        _, eigvecs = torch.linalg.eigh(cov)
        axis = eigvecs[:, :, -1]
        signs = torch.sign((centered * axis[:, None, :]).sum(dim=-1).mean(dim=1, keepdim=True))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        axis = axis * signs
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        return axis[:, None, :].repeat(1, pos.shape[1], 1)

    def _prompt(self, pos, fx, cond):
        mode = getattr(self.args, "hemo_prompt_mode", "flow_geo")
        if mode == "none":
            return None
        if mode == "geometry":
            return self._geometry_prompt(pos)
        if mode == "flow_geo":
            if not hasattr(self.dataset, "build_prompt"):
                raise ValueError(f"loader={self.args.loader} does not implement build_prompt for flow_geo mode")
            return self.dataset.build_prompt(pos, cond, fx=fx)
        raise ValueError(f"Unsupported hemo_prompt_mode={mode}")

    def _forward(self, pos, fx, cond):
        prompt = self._prompt(pos, fx, cond)
        if prompt is not None:
            fx = torch.cat([fx, prompt], dim=-1)
        return self.model(pos[:, :, :3], fx)

    def _head_parameter_names(self):
        head_tokens = ("mlp2", "ln_3")
        return {name for name, _ in self.model.named_parameters()
                if any(token in name for token in head_tokens)}

    def _set_backbone_trainable(self, trainable):
        head_names = self._head_parameter_names()
        for name, param in self.model.named_parameters():
            if name in head_names:
                param.requires_grad = True
            else:
                param.requires_grad = bool(trainable)

    def _build_optimizer(self):
        head_names = self._head_parameter_names()
        if self.args.backbone_lr is not None or self.args.head_lr is not None:
            backbone_lr = self.args.backbone_lr if self.args.backbone_lr is not None else self.args.lr
            head_lr = self.args.head_lr if self.args.head_lr is not None else self.args.lr
            groups = [
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if p.requires_grad and n not in head_names],
                    "lr": backbone_lr,
                },
                {
                    "params": [p for n, p in self.model.named_parameters()
                               if p.requires_grad and n in head_names],
                    "lr": head_lr,
                },
            ]
            groups = [group for group in groups if group["params"]]
        else:
            groups = [p for p in self.model.parameters() if p.requires_grad]

        if self.args.optimizer == 'AdamW':
            return torch.optim.AdamW(groups, lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.optimizer == 'Adam':
            return torch.optim.Adam(groups, lr=self.args.lr, weight_decay=self.args.weight_decay)
        raise ValueError('Optimizer only AdamW or Adam')

    def _build_scheduler(self, optimizer, epochs_left=None):
        epochs = self.args.epochs if epochs_left is None else max(1, epochs_left)
        if self.args.scheduler == 'OneCycleLR':
            max_lr = [group["lr"] for group in optimizer.param_groups]
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, epochs=epochs,
                steps_per_epoch=len(self.train_loader), pct_start=self.args.pct_start)
        if self.args.scheduler == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if self.args.scheduler == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        return None

    def _maybe_decode(self, y):
        if self.args.normalize:
            return self.dataset.y_normalizer.decode(y)
        return y

    def _metrics(self, out, y):
        eps = 1e-8
        diff = out - y
        n_channels = diff.shape[-1]
        weights = self._channel_weights(diff.device, diff.dtype, n_channels=n_channels)
        rel = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / (
            torch.linalg.norm(y.reshape(y.shape[0], -1), dim=1) + eps)
        split_rel = torch.linalg.norm(diff, dim=1) / (torch.linalg.norm(y, dim=1) + eps)
        balanced_rel = split_rel.mean(dim=1)
        weighted_balanced_rel = (split_rel * weights).sum(dim=1) / (weights.sum() + eps)
        out_vel = self._velocity_slice(out)
        y_vel = self._velocity_slice(y)
        vel_mag_err = torch.abs(torch.linalg.norm(out_vel, dim=-1) -
                                torch.linalg.norm(y_vel, dim=-1)).mean(dim=1)
        vel_rel = split_rel if n_channels == 3 else split_rel[:, 1:4]
        vel_rel = vel_rel.mean(dim=1)
        vel_vector_diff = out_vel - y_vel
        vel_vector_rel = torch.linalg.norm(vel_vector_diff.reshape(vel_vector_diff.shape[0], -1), dim=1) / (
            torch.linalg.norm(y_vel.reshape(y.shape[0], -1), dim=1) + eps)
        vel_mag_diff = torch.linalg.norm(out_vel, dim=-1) - torch.linalg.norm(y_vel, dim=-1)
        vel_mag_rel = torch.linalg.norm(vel_mag_diff.reshape(vel_mag_diff.shape[0], -1), dim=1) / (
            torch.linalg.norm(torch.linalg.norm(y_vel, dim=-1).reshape(y.shape[0], -1), dim=1) + eps)
        velocity_balanced_rel = 0.5 * (vel_vector_rel + vel_mag_rel)
        if self._has_pressure(y):
            pressure_rel = split_rel[:, 0]
            pressure_center_rel = torch.linalg.norm(
                (diff[:, :, 0] - diff[:, :, 0].mean(dim=1, keepdim=True)).reshape(diff.shape[0], -1), dim=1
            ) / (
                torch.linalg.norm((y[:, :, 0] - y[:, :, 0].mean(dim=1, keepdim=True)).reshape(y.shape[0], -1), dim=1)
                + eps)
            physics_terms = torch.stack([
                pressure_rel,
                pressure_center_rel,
                vel_rel,
                vel_mag_rel,
            ], dim=1)
            physics_weights = self._physics_weights(diff.device, diff.dtype)
            physics_balanced_rel = (physics_terms * physics_weights).sum(dim=1) / (
                physics_weights.sum() + eps)
        else:
            pressure_rel = torch.full_like(rel, float("nan"))
            pressure_center_rel = torch.full_like(rel, float("nan"))
            physics_balanced_rel = velocity_balanced_rel
        return (rel, split_rel, balanced_rel, weighted_balanced_rel, vel_rel, vel_mag_err,
                pressure_rel, pressure_center_rel, vel_mag_rel, physics_balanced_rel,
                vel_vector_rel, velocity_balanced_rel)

    def _training_loss(self, out, y):
        mode = getattr(self.args, "hemo_loss_mode", "global_rel")
        if mode == "global_rel":
            return L2Loss(size_average=False)(out, y)
        if mode == "channel_rel":
            weights = self._channel_weights(out.device, out.dtype, n_channels=out.shape[-1])
            diff = torch.linalg.norm(out - y, dim=1) / (torch.linalg.norm(y, dim=1) + 1e-8)
            return (diff * weights).sum(dim=1).sum() / (weights.sum() + 1e-8)
        if mode == "physics_rel":
            if out.shape[-1] != 4:
                raise ValueError("physics_rel requires full [p,u,v,w] targets; use velocity_rel for hemo_target=velocity")
            eps = 1e-8
            out_phys = self._maybe_decode(out)
            y_phys = self._maybe_decode(y)
            p_diff = out_phys[:, :, 0] - y_phys[:, :, 0]
            p_true = y_phys[:, :, 0]
            p_rel = torch.linalg.norm(p_diff.reshape(p_diff.shape[0], -1), dim=1) / (
                torch.linalg.norm(p_true.reshape(p_true.shape[0], -1), dim=1) + eps)
            p_center_diff = p_diff - p_diff.mean(dim=1, keepdim=True)
            p_center_true = p_true - p_true.mean(dim=1, keepdim=True)
            p_center_rel = torch.linalg.norm(p_center_diff.reshape(p_center_diff.shape[0], -1), dim=1) / (
                torch.linalg.norm(p_center_true.reshape(p_center_true.shape[0], -1), dim=1) + eps)
            vel_diff = out_phys[:, :, 1:4] - y_phys[:, :, 1:4]
            vel_true = y_phys[:, :, 1:4]
            vel_rel = torch.linalg.norm(vel_diff.reshape(vel_diff.shape[0], -1), dim=1) / (
                torch.linalg.norm(vel_true.reshape(vel_true.shape[0], -1), dim=1) + eps)
            vel_mag_diff = torch.abs(torch.linalg.norm(out_phys[:, :, 1:4], dim=-1) -
                                     torch.linalg.norm(y_phys[:, :, 1:4], dim=-1)).reshape(out.shape[0], -1)
            vel_mag_rel = torch.linalg.norm(vel_mag_diff, dim=1) / (
                torch.linalg.norm(torch.linalg.norm(y_phys[:, :, 1:4], dim=-1).reshape(y.shape[0], -1), dim=1) + eps)
            weights = self._physics_weights(out.device, out.dtype).view(-1)
            return (
                weights[0] * p_rel.mean()
                + weights[1] * p_center_rel.mean()
                + weights[2] * vel_rel.mean()
                + weights[3] * vel_mag_rel.mean()
            ) / (weights.sum() + eps)
        if mode == "velocity_rel":
            eps = 1e-8
            out_phys = self._maybe_decode(out)
            y_phys = self._maybe_decode(y)
            out_vel = self._velocity_slice(out_phys)
            y_vel = self._velocity_slice(y_phys)
            vel_diff = out_vel - y_vel
            vel_true = y_vel
            vel_vector_rel = torch.linalg.norm(vel_diff.reshape(vel_diff.shape[0], -1), dim=1) / (
                torch.linalg.norm(vel_true.reshape(vel_true.shape[0], -1), dim=1) + eps)
            vel_mag_diff = torch.linalg.norm(out_vel, dim=-1) - torch.linalg.norm(y_vel, dim=-1)
            vel_mag_rel = torch.linalg.norm(vel_mag_diff.reshape(vel_mag_diff.shape[0], -1), dim=1) / (
                torch.linalg.norm(torch.linalg.norm(y_vel, dim=-1).reshape(y.shape[0], -1), dim=1) + eps)
            vel_weight = getattr(self.args, "hemo_velocity_weight", 1.0)
            vel_mag_weight = getattr(self.args, "hemo_velocity_mag_weight", 0.5)
            return (
                vel_weight * vel_vector_rel.mean()
                + vel_mag_weight * vel_mag_rel.mean()
            ) / (vel_weight + vel_mag_weight + eps)
        raise ValueError(f"Unsupported hemo_loss_mode={mode}")

    def _select_metric_value(self, metrics):
        mode = getattr(self.args, "hemo_select_metric", "rel_l2")
        if mode == "rel_l2":
            return metrics["rel_l2"]
        if mode == "balanced_rel_l2":
            return metrics["balanced_rel_l2"]
        if mode == "weighted_balanced_rel_l2":
            return metrics["weighted_balanced_rel_l2"]
        if mode == "vel_rel_l2":
            return metrics["vel_rel_l2"]
        if mode == "vel_vector_rel_l2":
            return metrics["vel_vector_rel_l2"]
        if mode == "vel_mag_rel_l2":
            return metrics["vel_mag_rel_l2"]
        if mode == "velocity_balanced_rel_l2":
            return metrics["velocity_balanced_rel_l2"]
        if mode == "physics_balanced_rel_l2":
            return metrics["physics_balanced_rel_l2"]
        raise ValueError(f"Unsupported hemo_select_metric={mode}")

    def vali(self, loader=None, desc="Val"):
        if loader is None:
            loader = self.test_loader
        self.model.eval()
        rel_all, split_all, bal_all, wbal_all, vel_rel_all, vel_all = [], [], [], [], [], []
        p_rel_all, p_center_all, vel_mag_rel_all, physics_bal_all = [], [], [], []
        vel_vector_rel_all, velocity_bal_all = [], []
        myloss = L2Loss(size_average=False)
        loss_sum = 0.0
        n = 0
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc, total=len(loader), dynamic_ncols=True, mininterval=2.0)
            for pos, fx, cond, y in pbar:
                pos, fx, cond, y = pos.to(self.device), fx.to(self.device), cond.to(self.device), y.to(self.device)
                out = self._forward(pos, fx, cond)
                out_dec = self._maybe_decode(out)
                y_dec = y
                loss_sum += myloss(out_dec, y_dec).item()
                n += pos.shape[0]
                (rel, split, bal, wbal, vel_rel, vel, p_rel, p_center_rel, vel_mag_rel,
                 physics_bal, vel_vector_rel, velocity_bal) = self._metrics(out_dec, y_dec)
                rel_all.append(rel.detach().cpu())
                split_all.append(split.detach().cpu())
                bal_all.append(bal.detach().cpu())
                wbal_all.append(wbal.detach().cpu())
                vel_rel_all.append(vel_rel.detach().cpu())
                vel_all.append(vel.detach().cpu())
                p_rel_all.append(p_rel.detach().cpu())
                p_center_all.append(p_center_rel.detach().cpu())
                vel_mag_rel_all.append(vel_mag_rel.detach().cpu())
                physics_bal_all.append(physics_bal.detach().cpu())
                vel_vector_rel_all.append(vel_vector_rel.detach().cpu())
                velocity_bal_all.append(velocity_bal.detach().cpu())
                pbar.set_postfix(rel_l2=loss_sum / max(n, 1))
        rel = torch.cat(rel_all).mean().item()
        split = torch.cat(split_all).mean(dim=0).numpy()
        bal = torch.cat(bal_all).mean().item()
        wbal = torch.cat(wbal_all).mean().item()
        vel_rel = torch.cat(vel_rel_all).mean().item()
        vel = torch.cat(vel_all).mean().item()
        p_rel = torch.cat(p_rel_all).mean().item()
        p_center = torch.cat(p_center_all).mean().item()
        vel_mag_rel = torch.cat(vel_mag_rel_all).mean().item()
        physics_bal = torch.cat(physics_bal_all).mean().item()
        vel_vector_rel = torch.cat(vel_vector_rel_all).mean().item()
        velocity_bal = torch.cat(velocity_bal_all).mean().item()
        return {
            "rel_l2": rel,
            "split_rel_l2": split,
            "balanced_rel_l2": bal,
            "weighted_balanced_rel_l2": wbal,
            "vel_rel_l2": vel_rel,
            "vel_vector_rel_l2": vel_vector_rel,
            "vel_mag_mae": vel,
            "pressure_rel_l2": p_rel,
            "pressure_center_rel_l2": p_center,
            "vel_mag_rel_l2": vel_mag_rel,
            "velocity_balanced_rel_l2": velocity_bal,
            "physics_balanced_rel_l2": physics_bal,
        }

    def train(self):
        print(f"[Prompt] hemo_prompt_mode={getattr(self.args, 'hemo_prompt_mode', 'flow_geo')} "
              f"fun_dim={self.args.fun_dim} hemo_wall_mode={getattr(self.args, 'hemo_wall_mode', 'keep')}")
        if self.args.finetune:
            self._load_pretrained_with_filter(self.args.finetune_name)

        freeze_epochs = self.args.freeze_pretrained_epochs if self.args.finetune else 0
        if freeze_epochs > 0:
            self._set_backbone_trainable(False)
            print(f"[Finetune] Frozen pretrained backbone for first {freeze_epochs} epochs.")

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        myloss = L2Loss(size_average=False)
        train_loss_list, test_loss_list = [], []
        best_val = float("inf")
        best_epoch = -1
        bad_epochs = 0

        for ep in range(self.args.epochs):
            if freeze_epochs > 0 and ep == freeze_epochs:
                self._set_backbone_trainable(True)
                optimizer = self._build_optimizer()
                scheduler = self._build_scheduler(optimizer, epochs_left=self.args.epochs - ep)
                print(f"[Finetune] Unfroze backbone at epoch {ep + 1}; rebuilt optimizer/scheduler.")

            self.model.train()
            train_loss = 0.0
            n = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {ep + 1}/{self.args.epochs} train",
                        total=len(self.train_loader), dynamic_ncols=True, mininterval=2.0)
            for pos, fx, cond, y in pbar:
                pos, fx, cond, y = pos.to(self.device), fx.to(self.device), cond.to(self.device), y.to(self.device)
                out = self._forward(pos, fx, cond)
                loss = self._training_loss(out, y)
                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                if self.args.scheduler == 'OneCycleLR' and scheduler is not None:
                    scheduler.step()
                train_loss += loss.item()
                n += pos.shape[0]
                lr_text = ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
                pbar.set_postfix(loss=train_loss / max(n, 1), lr=lr_text)

            if self.args.scheduler in ('CosineAnnealingLR', 'StepLR') and scheduler is not None:
                scheduler.step()

            train_loss /= max(n, 1)
            metrics = self.vali(desc=f"Epoch {ep + 1}/{self.args.epochs} val")
            train_loss_list.append(train_loss)
            select_metric = self._select_metric_value(metrics)
            test_loss_list.append(select_metric)
            improved = select_metric < best_val - self.args.min_delta
            if improved:
                best_val = select_metric
                best_epoch = ep
                bad_epochs = 0
                self._save_model()
            else:
                bad_epochs += 1
            print(f"Epoch {ep + 1}/{self.args.epochs} train={train_loss:.6f} "
                  f"val_rel_l2={metrics['rel_l2']:.6f} "
                  f"balanced_rel_l2={metrics['balanced_rel_l2']:.6f} "
                  f"weighted_balanced_rel_l2={metrics['weighted_balanced_rel_l2']:.6f} "
                  f"vel_rel_l2={metrics['vel_rel_l2']:.6f} "
                  f"vel_vector_rel_l2={metrics['vel_vector_rel_l2']:.6f} "
                  f"pressure_rel_l2={metrics['pressure_rel_l2']:.6f} "
                  f"pressure_center_rel_l2={metrics['pressure_center_rel_l2']:.6f} "
                  f"vel_mag_rel_l2={metrics['vel_mag_rel_l2']:.6f} "
                  f"velocity_balanced_rel_l2={metrics['velocity_balanced_rel_l2']:.6f} "
                  f"physics_balanced_rel_l2={metrics['physics_balanced_rel_l2']:.6f} "
                  f"split[{self._split_label(len(metrics['split_rel_l2']))}]={metrics['split_rel_l2']} "
                  f"vel_mag_mae={metrics['vel_mag_mae']:.6f} "
                  f"best={best_val:.6f}@{best_epoch + 1}({getattr(self.args, 'hemo_select_metric', 'rel_l2')})", flush=True)
            if ep % 10 == 0:
                self._save_logs(train_loss_list, test_loss_list)
                self._save_curve(train_loss_list, test_loss_list)
            if self.args.early_stop and bad_epochs >= self.args.patience:
                print(f"[EarlyStop] no improvement for {bad_epochs} epochs; "
                      f"best {getattr(self.args, 'hemo_select_metric', 'rel_l2')}={best_val:.6f} at epoch {best_epoch + 1}", flush=True)
                break

        self._save_last_model()
        self._save_logs(train_loss_list, test_loss_list)
        self._save_curve(train_loss_list, test_loss_list)

    def _save_model(self):
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("./checkpoints", self.args.save_name + ".pt"))

    def _save_last_model(self):
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("./checkpoints", self.args.save_name + "_last.pt"))

    def _save_logs(self, train_loss_list, test_loss_list):
        os.makedirs("./training_logs", exist_ok=True)
        np.save(os.path.join("./training_logs", self.args.save_name + "_train_loss.npy"), np.array(train_loss_list))
        np.save(os.path.join("./training_logs", self.args.save_name + "_test_loss.npy"), np.array(test_loss_list))

    def _result_dir(self):
        path = os.path.join("./results", self.args.save_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_curve(self, train_loss_list, test_loss_list):
        if not self.args.visualize:
            return
        out_dir = self._result_dir()
        epochs = np.arange(1, len(train_loss_list) + 1)
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, train_loss_list, label="train rel L2")
        plt.plot(epochs, test_loss_list, label=f"val {getattr(self.args, 'hemo_select_metric', 'rel_l2')}")
        plt.xlabel("Epoch")
        plt.ylabel("Relative L2")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
        plt.close()

    def _save_prediction_visuals(self, max_cases=None):
        if not self.args.visualize:
            return
        if max_cases is None:
            max_cases = self.args.vis_num
        out_dir = self._result_dir()
        self.model.eval()
        saved = 0
        with torch.no_grad():
            for pos, fx, cond, y in self.test_loader:
                pos_dev, fx_dev, cond_dev = pos.to(self.device), fx.to(self.device), cond.to(self.device)
                out = self._maybe_decode(self._forward(pos_dev, fx_dev, cond_dev)).cpu()
                for b in range(pos.shape[0]):
                    self._plot_case(pos[b].numpy(), y[b].numpy(), out[b].numpy(), saved, out_dir)
                    saved += 1
                    if saved >= max_cases:
                        return

    def _plot_case(self, pos, y, out, case_id, out_dir):
        if y.shape[-1] == 3:
            v_true = np.linalg.norm(y, axis=-1)
            v_pred = np.linalg.norm(out, axis=-1)
            fields = [
                ("velocity_mag_true", v_true),
                ("velocity_mag_pred", v_pred),
                ("velocity_mag_abs_error", np.abs(v_pred - v_true)),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        else:
            p_true, p_pred = y[:, 0], out[:, 0]
            v_true = np.linalg.norm(y[:, 1:4], axis=-1)
            v_pred = np.linalg.norm(out[:, 1:4], axis=-1)
            fields = [
                ("pressure_true", p_true),
                ("pressure_pred", p_pred),
                ("velocity_mag_true", v_true),
                ("velocity_mag_pred", v_pred),
            ]
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        xy = pos[:, [0, 1]]
        for ax, (title, values) in zip(np.ravel(axes), fields):
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=2, cmap="viridis")
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(out_dir, f"case_{case_id:03d}_fields.png"), dpi=220)
        plt.close(fig)

    def test(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt", map_location=self.device))
        metrics = self.vali(desc="Test")
        if self.args.loader in ("VMRCFD", "HemoVMRCFD"):
            print(f"eval split:{self.args.vmr_eval_split}")
        elif self.args.loader == "AneumoCFD":
            print(f"eval split:{self.args.aneumo_eval_split}")
        elif self.args.loader == "HemoPT":
            print(f"eval split:{getattr(self.args, 'hemo_eval_split', 'heldout')}")
        else:
            print("eval split:heldout")
        print(f"test rel_l2:{metrics['rel_l2']}")
        print(f"test balanced_rel_l2:{metrics['balanced_rel_l2']}")
        print(f"test weighted_balanced_rel_l2:{metrics['weighted_balanced_rel_l2']}")
        print(f"test vel_rel_l2:{metrics['vel_rel_l2']}")
        print(f"test vel_vector_rel_l2:{metrics['vel_vector_rel_l2']}")
        print(f"test pressure_rel_l2:{metrics['pressure_rel_l2']}")
        print(f"test pressure_center_rel_l2:{metrics['pressure_center_rel_l2']}")
        print(f"test vel_mag_rel_l2:{metrics['vel_mag_rel_l2']}")
        print(f"test velocity_balanced_rel_l2:{metrics['velocity_balanced_rel_l2']}")
        print(f"test physics_balanced_rel_l2:{metrics['physics_balanced_rel_l2']}")
        print(f"test split_rel_l2[{self._split_label(len(metrics['split_rel_l2']))}]:{metrics['split_rel_l2']}")
        print(f"test vel_mag_mae:{metrics['vel_mag_mae']}")
        self._save_prediction_visuals()

    def test_full_mesh(self):
        print("test_full_mesh skipped for hemo CFD fine-tuning.")
