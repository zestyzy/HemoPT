import math
import os

import numpy as np
import torch
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from utils.loss import L2Loss


GEOMETRIC_TARGET_DIM = 9


class Exp_VascularPretrain(Exp_Basic):
    """Lifted geometric pre-training for vascular STL-derived samples."""

    def _use_physics_proxy_loss(self, out, y):
        return (
            getattr(self.args, "vascular_physics_proxy", False)
            and out.shape[-1] > GEOMETRIC_TARGET_DIM
            and y.shape[-1] > GEOMETRIC_TARGET_DIM
        )

    def _loss_components(self, out, y, myloss):
        if self._use_physics_proxy_loss(out, y):
            geom_loss = myloss(out[:, :, :GEOMETRIC_TARGET_DIM].contiguous(),
                               y[:, :, :GEOMETRIC_TARGET_DIM].contiguous())
            phys_loss = myloss(out[:, :, GEOMETRIC_TARGET_DIM:].contiguous(),
                               y[:, :, GEOMETRIC_TARGET_DIM:].contiguous())
            phys_weight = getattr(self.args, "vascular_physics_weight", 0.25)
            total_loss = geom_loss + phys_weight * phys_loss
            return total_loss, geom_loss, phys_loss

        loss = myloss(out, y)
        return loss, loss, None

    def vali(self, epoch=None):
        myloss = L2Loss(size_average=False)
        self.model.eval()
        rel_err = 0.0
        geom_err = 0.0
        phys_err = 0.0
        phys_batches = 0
        n = 0
        desc = "Val"
        if epoch is not None:
            desc = f"Epoch {epoch + 1}/{self.args.epochs} val"
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=desc, total=len(self.test_loader),
                        dynamic_ncols=True, mininterval=2.0)
            for pos, fx, _, y in pbar:
                pos = pos.to(self.device)
                fx = fx.to(self.device)
                y = y.to(self.device)
                out = self.model(pos[:, :, :3], fx)
                loss, geom_loss, phys_loss = self._loss_components(out, y, myloss)
                rel_err += loss.item()
                geom_err += geom_loss.item()
                if phys_loss is not None:
                    phys_err += phys_loss.item()
                    phys_batches += pos.shape[0]
                n += pos.shape[0]
                postfix = {"rel_l2": rel_err / max(n, 1)}
                if phys_loss is not None:
                    postfix["geom_l2"] = geom_err / max(n, 1)
                    postfix["phys_l2"] = phys_err / max(phys_batches, 1)
                pbar.set_postfix(**postfix)
        return rel_err / max(n, 1)

    def train(self):
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Optimizer only AdamW or Adam')

        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                steps_per_epoch=len(self.train_loader), pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
            if warmup_epochs > 0:
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        return float(epoch + 1) / float(max(1, warmup_epochs))
                    progress = float(epoch - warmup_epochs) / float(max(1, self.args.epochs - warmup_epochs))
                    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        else:
            scheduler = None

        myloss = L2Loss(size_average=False)
        train_loss_list = []
        test_loss_list = []
        best_val = float("inf")
        best_epoch = -1
        bad_epochs = 0
        checkpoint_interval = max(0, int(getattr(self.args, "checkpoint_interval", 0)))

        for ep in range(self.args.epochs):
            self.model.train()
            train_loss = 0.0
            train_geom_loss = 0.0
            train_phys_loss = 0.0
            train_phys_batches = 0
            n = 0
            lr = optimizer.param_groups[0]['lr']
            pbar = tqdm(self.train_loader,
                        desc=f"Epoch {ep + 1}/{self.args.epochs} train",
                        total=len(self.train_loader),
                        dynamic_ncols=True,
                        mininterval=2.0)
            for pos, fx, _, y in pbar:
                pos = pos.to(self.device)
                fx = fx.to(self.device)
                y = y.to(self.device)
                out = self.model(pos[:, :, :3], fx)
                loss, geom_loss, phys_loss = self._loss_components(out, y, myloss)

                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                if self.args.scheduler == 'OneCycleLR' and scheduler is not None:
                    scheduler.step()

                train_loss += loss.item()
                train_geom_loss += geom_loss.item()
                if phys_loss is not None:
                    train_phys_loss += phys_loss.item()
                    train_phys_batches += pos.shape[0]
                n += pos.shape[0]
                postfix = {"loss": train_loss / max(n, 1), "lr": lr}
                if phys_loss is not None:
                    postfix["geom_l2"] = train_geom_loss / max(n, 1)
                    postfix["phys_l2"] = train_phys_loss / max(train_phys_batches, 1)
                pbar.set_postfix(**postfix)

            if self.args.scheduler in ('CosineAnnealingLR', 'StepLR') and scheduler is not None:
                scheduler.step()

            train_loss = train_loss / max(n, 1)
            rel_err = self.vali(epoch=ep)
            train_loss_list.append(train_loss)
            test_loss_list.append(rel_err)
            print(f"Epoch {ep + 1}/{self.args.epochs} Train loss: {train_loss:.5f} lr: {optimizer.param_groups[0]['lr']:.6g}", flush=True)
            improved = rel_err < best_val - getattr(self.args, "min_delta", 0.0)
            if improved:
                best_val = rel_err
                best_epoch = ep
                bad_epochs = 0
                self._save_model()
            else:
                bad_epochs += 1
            print(f"Epoch {ep + 1}/{self.args.epochs} val rel_err: {rel_err} "
                  f"best={best_val}@{best_epoch + 1} bad_epochs={bad_epochs}", flush=True)

            if ep % 10 == 0:
                self._save_logs(train_loss_list, test_loss_list)
            if checkpoint_interval > 0 and (ep + 1) % checkpoint_interval == 0:
                self._save_epoch_model(ep + 1)
            if self.args.early_stop and bad_epochs >= self.args.patience:
                print(f"[EarlyStop] no improvement for {bad_epochs} epochs; "
                      f"best val rel_err={best_val} at epoch {best_epoch + 1}", flush=True)
                break

        self._save_last_model()
        self._save_logs(train_loss_list, test_loss_list)

    def _save_model(self):
        os.makedirs('./checkpoints', exist_ok=True)
        state = self.model.state_dict()
        torch.save(state, os.path.join('./checkpoints', self.args.save_name + '.pt'))
        torch.save(state, os.path.join('./checkpoints', self.args.save_name + '_best.pt'))

    def _save_epoch_model(self, epoch):
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(
            './checkpoints', f"{self.args.save_name}_epoch{epoch:03d}.pt"))

    def _save_last_model(self):
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '_last.pt'))

    def _save_logs(self, train_loss_list, test_loss_list):
        os.makedirs('./training_logs', exist_ok=True)
        np.save(os.path.join('./training_logs', self.args.save_name + '_train_loss.npy'),
                np.array(train_loss_list))
        np.save(os.path.join('./training_logs', self.args.save_name + '_test_loss.npy'),
                np.array(test_loss_list))

    def test(self):
        checkpoint = torch.load(
            "./checkpoints/" + self.args.save_name + ".pt",
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint)
        rel_err = self.vali()
        print("test rel_err:{}".format(rel_err))

    def test_full_mesh(self):
        print("test_full_mesh skipped for vascular pre-training.")
