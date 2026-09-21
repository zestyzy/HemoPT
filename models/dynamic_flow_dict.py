import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomWalkTrajectoryEncoder(nn.Module):
    """Encode a short wall-probe trajectory into a point-wise feature.

    The vascular loader stores ``supervise_j`` as three concatenated 3-D
    displacement vectors, i.e. ``(B, N, 9)`` for the default three-step walk.
    This module keeps the step order (through a learned step embedding),
    encodes each displacement with a tiny MLP, and pools the three steps into
    one feature per spatial point.  A ``(B, N, 3, 3)`` input is also accepted
    to make the sequence structure explicit at call sites.
    """

    def __init__(self, num_steps=3, vector_dim=3, hidden_dim=16, output_dim=16):
        super().__init__()
        if num_steps <= 0 or vector_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("RW encoder dimensions must be positive")
        self.num_steps = int(num_steps)
        self.vector_dim = int(vector_dim)
        self.output_dim = int(output_dim)
        self.step_embed = nn.Parameter(torch.zeros(1, self.num_steps, hidden_dim))
        nn.init.normal_(self.step_embed, mean=0.0, std=0.02)
        self.step_mlp = nn.Sequential(
            nn.Linear(self.vector_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, trajectory):
        if trajectory.ndim == 3:
            expected_dim = self.num_steps * self.vector_dim
            if trajectory.shape[-1] != expected_dim:
                raise ValueError(
                    f"RW trajectory with rank 3 must have last dimension {expected_dim}; "
                    f"got {trajectory.shape[-1]}"
                )
            trajectory = trajectory.reshape(
                trajectory.shape[0], trajectory.shape[1], self.num_steps, self.vector_dim
            )
        elif trajectory.ndim == 4:
            if trajectory.shape[-2:] != (self.num_steps, self.vector_dim):
                raise ValueError(
                    "RW trajectory with rank 4 must have shape (..., steps, vector_dim); "
                    f"got {tuple(trajectory.shape)}"
                )
        else:
            raise ValueError(
                "RW trajectory must have shape (B, N, steps*vector_dim) or "
                f"(B, N, {self.num_steps}, {self.vector_dim}); got {tuple(trajectory.shape)}"
            )

        batch, points, steps, vector_dim = trajectory.shape
        step_vectors = trajectory.reshape(batch * points, steps, vector_dim)
        encoded_steps = self.step_mlp(step_vectors)
        encoded_steps = encoded_steps + self.step_embed
        pooled = torch.cat(
            [encoded_steps.mean(dim=1), encoded_steps.amax(dim=1)], dim=-1
        )
        features = self.projection(pooled)
        return features.reshape(batch, points, self.output_dim)


class NeuralFlowBasisGenerator(nn.Module):
    """
    Predicts a bank of M canonical flow modes from local vascular geometry features.
    Input: (B, N, C_in) where C_in usually includes [x, y, z, wall_dist, wall_normal (3)] -> 7 dims.
    Output: (B, N, M, 3) 3D velocity vectors for M orthogonal basis modes.
    """
    def __init__(self, in_dim=7, num_modes=4, hidden_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.num_modes = num_modes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modes * 3)
        )

    def forward(self, morph_features):
        """
        morph_features: (B, N, in_dim)
        Returns: (B, N, M, 3)
        """
        B, N, _ = morph_features.shape
        out = self.net(morph_features)
        return out.view(B, N, self.num_modes, 3)


class DynamicFlowRoutingGate(nn.Module):
    """
    Dynamically routes across the M canonical flow modes conditioned on local morphology,
    random-walk probe conditions, and an encoded random-walk trajectory.
    Input: (B, N, C_cond), assembled from wall distance/normal, probe direction/
    step length, and the encoded three-step wall trajectory.
    Output: (B, N, M) soft routing probabilities (sum to 1 per point).
    """
    def __init__(self, in_dim=8, num_modes=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modes)
        )

    def forward(self, cond_features):
        """
        cond_features: (B, N, in_dim)
        Returns: (B, N, num_modes)
        """
        logits = self.net(cond_features)
        return F.softmax(logits, dim=-1)


class NeuralFlowDictionary(nn.Module):
    """
    Self-Supervised Learnable Dynamic Flow Dictionary with End-to-End Routing Gradients.

    Architecture:
      1. Basis Generator: Maps local geometry to M canonical flow basis modes.
      2. RW Encoder + Routing Gate: Maps (wall morphology + probe condition +
         encoded three-step wall trajectory) to dynamic mixture weights.
      3. Dynamic Flow Synthesis: Combines modes into a probe-conditioned velocity proxy:
         u_proxy = sum_m (weights[:, :, m, None] * basis[:, :, m, :])

    Physics-Informed Self-Supervision Objectives:
      - Applied to Synthesized Flow Field u_proxy (PROVIDING GRADIENTS TO ROUTING GATE):
        1. Synthesized Wall No-Slip Loss: u_proxy must vanish at true vascular wall.
           Forces gate to choose low-velocity combinations near the boundaries.
        2. Synthesized MLS Divergence Loss: div(u_proxy) = Tr(J) must vanish.
           Forces gate to produce spatially consistent, mass-conserving flow mixtures.
      - Applied to Basis Generator (PREVENTING MODE COLLAPSE):
        3. Kinetic Energy Scale Constraint: Each basis mode must maintain non-trivial energy.
        4. Mode Orthogonality Loss: M modes must remain mutually orthogonal.
      - Gate Regularization:
        5. Routing Shannon Entropy Loss: Prevents gate from collapsing into a static single mode.
    """
    def __init__(
        self,
        morph_dim=7,
        cond_dim=4,
        num_modes=4,
        hidden_dim=64,
        loss_weight_noslip=0.1,
        loss_weight_div=0.05,
        loss_weight_ortho=0.05,
        loss_weight_energy=0.1,
        loss_weight_entropy=0.01,
        target_energy=1.0,
        rw_encoder_hidden_dim=16,
        rw_feature_dim=16,
        rw_encoder_num_steps=3,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.target_energy = target_energy
        self.loss_weight_noslip = loss_weight_noslip
        self.loss_weight_div = loss_weight_div
        self.loss_weight_ortho = loss_weight_ortho
        self.loss_weight_energy = loss_weight_energy
        self.loss_weight_entropy = loss_weight_entropy
        self.rw_feature_dim = int(rw_feature_dim)
        self.rw_encoder = RandomWalkTrajectoryEncoder(
            num_steps=int(rw_encoder_num_steps),
            vector_dim=3,
            hidden_dim=int(rw_encoder_hidden_dim),
            output_dim=self.rw_feature_dim,
        )

        self.basis_generator = NeuralFlowBasisGenerator(
            in_dim=morph_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim
        )

        self.routing_gate = DynamicFlowRoutingGate(
            in_dim=morph_dim - 3 + cond_dim + self.rw_feature_dim,
            num_modes=num_modes,
            hidden_dim=hidden_dim // 2
        )

    def forward(self, pos, morph_features, cond_features, rw_trajectory=None):
        """
        Args:
            pos: (B, N, 3) interior point coordinates
            morph_features: (B, N, morph_dim) e.g. [pos, d_w, r_w]
            cond_features: (B, N, cond_dim) e.g. [probe_dir, probe_step]
            rw_trajectory: (B, N, 9) or (B, N, 3, 3) wall-displacement sequence
        Returns:
            flow_proxy: (B, N, 4) [u_x, u_y, u_z, speed]
            basis_fields: (B, N, M, 3)
            routing_weights: (B, N, M)
        """
        basis_fields = self.basis_generator(morph_features)  # (B, N, M, 3)

        # Combine wall morphology (dist, normal) and probe conditions for routing
        if rw_trajectory is None:
            rw_features = cond_features.new_zeros(
                cond_features.shape[0], cond_features.shape[1], self.rw_feature_dim
            )
        else:
            rw_features = self.rw_encoder(rw_trajectory)
        gate_input = torch.cat(
            [morph_features[:, :, 3:], cond_features, rw_features], dim=-1
        )
        weights = self.routing_gate(gate_input)  # (B, N, M)

        # Dynamic soft-mixture across dictionary atoms
        combined_flow = torch.sum(weights.unsqueeze(-1) * basis_fields, dim=2)  # (B, N, 3)
        speed = torch.norm(combined_flow, p=2, dim=-1, keepdim=True)  # (B, N, 1)

        proxy_target = torch.cat([combined_flow, speed], dim=-1)  # (B, N, 4)
        return proxy_target, basis_fields, weights

    def compute_self_supervised_losses(self, pos, basis_fields, weights, true_wall_dist, k_neighbors=8):
        """
        Compute physics-informed self-supervision losses without CFD paired labels.
        CRITICAL: Losses are computed directly on the synthesized flow field u_proxy
        to ensure routing_gate receives strong, direct physical gradients!

        Args:
            pos: (B, N, 3) coordinates
            basis_fields: (B, N, M, 3) generated flow modes
            weights: (B, N, M) dynamic mixture weights (from routing_gate)
            true_wall_dist: (B, N, 1) CLEAN, unmasked true distance to wall
        """
        B, N, M, _ = basis_fields.shape
        device = basis_fields.device

        # Synthesized combined flow field: (B, N, 3)
        # Directly connects basis_fields AND weights into backward graph!
        combined_flow = torch.sum(weights.unsqueeze(-1) * basis_fields, dim=2)

        # -------------------------------------------------------------
        # 1. Kinetic Energy Scale Constraint on Basis Modes
        # Enforces each basis mode m to maintain non-trivial mean kinetic energy ~ target_energy
        # -------------------------------------------------------------
        mode_energies = torch.mean(torch.sum(basis_fields ** 2, dim=-1), dim=1)  # (B, M)
        loss_energy = torch.mean((mode_energies - self.target_energy) ** 2)

        # -------------------------------------------------------------
        # 2. Wall No-Slip Loss on Synthesized Flow Field u_proxy
        # Points near the wall (true_wall_dist -> 0) must have zero velocity
        # Provides gradients to routing_gate to suppress high-speed modes near wall!
        # -------------------------------------------------------------
        wall_proximity = torch.exp(-torch.clamp(true_wall_dist, min=0.0) / 0.05)  # (B, N, 1)
        speed_combined = torch.norm(combined_flow, p=2, dim=-1, keepdim=True)  # (B, N, 1)
        loss_noslip = torch.mean((speed_combined * wall_proximity) ** 2)

        # -------------------------------------------------------------
        # 3. Mode Orthogonality / Diversity Loss on Basis Modes
        # Prevents multi-mode collapse into identical atoms
        # -------------------------------------------------------------
        basis_flat = basis_fields.permute(0, 2, 1, 3).reshape(B, M, -1)
        basis_norm = F.normalize(basis_flat, p=2, dim=-1)  # (B, M, N*3)
        sim_matrix = torch.bmm(basis_norm, basis_norm.transpose(1, 2))  # (B, M, M)
        identity = torch.eye(M, device=device).unsqueeze(0)
        off_diag_sim = (sim_matrix * (1.0 - identity)) ** 2
        loss_ortho = torch.sum(off_diag_sim) / (B * max(1, M * (M - 1)))

        # -------------------------------------------------------------
        # 4. True Local Divergence Loss via Moving Least Squares (MLS) on Synthesized Field
        # Computes div(u_proxy) = Tr(J)
        # Provides gradient to routing_gate to ensure continuous, mass-conserving spatial mixture!
        # -------------------------------------------------------------
        sub_size = min(N, 256)
        sub_idx = torch.randperm(N, device=device)[:sub_size]
        pos_sub = pos[:, sub_idx, :]  # (B, S, 3)
        flow_sub = combined_flow[:, sub_idx, :]  # (B, S, 3)

        dist_mat = torch.cdist(pos_sub, pos_sub)  # (B, S, S)
        k = min(k_neighbors, sub_size - 1)
        topk_dist, nn_idx = torch.topk(dist_mat, k=k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:, :, 1:]  # (B, S, k)
        nn_dist = topk_dist[:, :, 1:]  # (B, S, k)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, sub_size, k)
        nn_pos = pos_sub[batch_idx, nn_idx]  # (B, S, k, 3)
        dp = nn_pos - pos_sub.unsqueeze(2)  # (B, S, k, 3)

        sigma = torch.mean(nn_dist, dim=-1, keepdim=True).clamp(min=1e-4)
        weights_kernel = torch.exp(-nn_dist / sigma)  # (B, S, k)

        dp_col = dp.unsqueeze(-1)  # (B, S, k, 3, 1)
        dp_row = dp.unsqueeze(-2)  # (B, S, k, 1, 3)
        wk = weights_kernel.unsqueeze(-1).unsqueeze(-1)  # (B, S, k, 1, 1)
        outer_dp = wk * (dp_col @ dp_row)  # (B, S, k, 3, 3)
        C = torch.sum(outer_dp, dim=2) + 1e-4 * torch.eye(3, device=device).view(1, 1, 3, 3)  # (B, S, 3, 3)
        C_inv = torch.linalg.inv(C)  # (B, S, 3, 3)

        nn_flow = flow_sub[batch_idx, nn_idx]  # (B, S, k, 3)
        dv = nn_flow - flow_sub.unsqueeze(2)  # (B, S, k, 3)

        dv_col = dv.unsqueeze(-1)  # (B, S, k, 3, 1)
        outer_dv = wk * (dv_col @ dp_row)  # (B, S, k, 3, 3)
        B_mat = torch.sum(outer_dv, dim=2)  # (B, S, 3, 3)

        J = B_mat @ C_inv  # (B, S, 3, 3)
        div_field = J[:, :, 0, 0] + J[:, :, 1, 1] + J[:, :, 2, 2]  # (B, S)
        loss_div = torch.mean(div_field ** 2)

        # -------------------------------------------------------------
        # 5. Routing Shannon Entropy Regularization
        # Keep routing entropy in a non-degenerate band; maximizing entropy alone
        # collapses the gate to uniform weights and removes condition dependence.
        # Target 75% of maximum entropy rather than the uniform maximum.
        # -------------------------------------------------------------
        entropy = - torch.sum(weights * torch.log(weights.clamp(min=1e-8)), dim=-1)  # (B, N)
        target_entropy = 0.75 * torch.log(torch.tensor(float(M), device=device))
        loss_entropy = torch.mean((entropy - target_entropy) ** 2)

        # Total self-supervised regularizer
        loss_dict_total = (
            self.loss_weight_energy * loss_energy
            + self.loss_weight_noslip * loss_noslip
            + self.loss_weight_div * loss_div
            + self.loss_weight_ortho * loss_ortho
            + self.loss_weight_entropy * loss_entropy
        )

        # Compute summary metrics for monitoring
        mean_basis_norm = torch.mean(torch.norm(basis_fields, p=2, dim=-1))
        mean_proxy_speed = torch.mean(speed_combined)

        return {
            "loss_dict_total": loss_dict_total,
            "loss_energy": loss_energy,
            "loss_noslip": loss_noslip,
            "loss_div": loss_div,
            "loss_ortho": loss_ortho,
            "loss_entropy": loss_entropy,
            "basis_norm": mean_basis_norm.detach(),
            "proxy_speed": mean_proxy_speed.detach(),
            "gate_entropy": torch.mean(entropy).detach(),
        }


class AnalyticCompactFlowDictionary(nn.Module):
    """Route a fixed eight-mode analytic bank into a four-channel compact flow.

    The analytic bank is deliberately not a trainable parameter.  The
    eight-way routing gate and its small RW trajectory encoder are learned.
    The fourth compact channel is always recomputed from the mixed three-vector
    velocity, rather than mixing the per-mode speed channels independently.
    """

    NUM_MODES = 8

    def __init__(
        self,
        morph_dim=7,
        cond_dim=4,
        num_modes=8,
        hidden_dim=64,
        loss_weight_noslip=0.1,
        loss_weight_div=0.05,
        loss_weight_entropy=0.01,
        rw_encoder_hidden_dim=16,
        rw_feature_dim=16,
        rw_encoder_num_steps=3,
    ):
        super().__init__()
        if int(num_modes) != self.NUM_MODES:
            raise ValueError(
                "AnalyticCompactFlowDictionary requires exactly 8 analytic modes"
            )
        self.num_modes = self.NUM_MODES
        self.loss_weight_noslip = loss_weight_noslip
        self.loss_weight_div = loss_weight_div
        self.loss_weight_entropy = loss_weight_entropy
        self.rw_feature_dim = int(rw_feature_dim)
        self.rw_encoder = RandomWalkTrajectoryEncoder(
            num_steps=int(rw_encoder_num_steps),
            vector_dim=3,
            hidden_dim=int(rw_encoder_hidden_dim),
            output_dim=self.rw_feature_dim,
        )
        self.routing_gate = DynamicFlowRoutingGate(
            in_dim=morph_dim - 3 + cond_dim + self.rw_feature_dim,
            num_modes=self.NUM_MODES,
            hidden_dim=hidden_dim // 2,
        )

    def forward(
        self, pos, morph_features, cond_features, analytic_basis, rw_trajectory=None
    ):
        if analytic_basis.ndim != 4 or analytic_basis.shape[-2:] != (self.NUM_MODES, 3):
            raise ValueError(
                "analytic_basis must have shape (B, N, 8, 3); "
                f"got {tuple(analytic_basis.shape)}"
            )
        if analytic_basis.shape[:2] != pos.shape[:2]:
            raise ValueError("analytic_basis and pos must share batch and point dimensions")

        if rw_trajectory is None:
            rw_features = cond_features.new_zeros(
                cond_features.shape[0], cond_features.shape[1], self.rw_feature_dim
            )
        else:
            rw_features = self.rw_encoder(rw_trajectory)
        gate_input = torch.cat(
            [morph_features[:, :, 3:], cond_features, rw_features], dim=-1
        )
        weights = self.routing_gate(gate_input)
        combined_flow = torch.sum(weights.unsqueeze(-1) * analytic_basis, dim=2)
        speed = torch.linalg.vector_norm(combined_flow, dim=-1, keepdim=True)
        compact_proxy = torch.cat([combined_flow, speed], dim=-1)
        return compact_proxy, analytic_basis, weights

    def compute_self_supervised_losses(
        self, pos, basis_fields, weights, true_wall_dist, k_neighbors=8
    ):
        """Physics losses for the fixed analytic bank and trainable gate."""
        combined_flow = torch.sum(weights.unsqueeze(-1) * basis_fields, dim=2)
        device = combined_flow.device
        wall_proximity = torch.exp(-torch.clamp(true_wall_dist, min=0.0) / 0.05)
        speed_combined = torch.linalg.vector_norm(combined_flow, dim=-1, keepdim=True)
        loss_noslip = torch.mean((speed_combined * wall_proximity) ** 2)

        B, N, _ = combined_flow.shape
        sub_size = min(N, 256)
        if sub_size < 2:
            loss_div = combined_flow.new_zeros(())
        else:
            sub_idx = torch.randperm(N, device=device)[:sub_size]
            pos_sub = pos[:, sub_idx, :]
            flow_sub = combined_flow[:, sub_idx, :]
            dist_mat = torch.cdist(pos_sub, pos_sub)
            k = min(k_neighbors, sub_size - 1)
            topk_dist, nn_idx = torch.topk(dist_mat, k=k + 1, dim=-1, largest=False)
            nn_idx = nn_idx[:, :, 1:]
            nn_dist = topk_dist[:, :, 1:]
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, sub_size, k)
            nn_pos = pos_sub[batch_idx, nn_idx]
            dp = nn_pos - pos_sub.unsqueeze(2)
            sigma = torch.mean(nn_dist, dim=-1, keepdim=True).clamp(min=1e-4)
            kernel = torch.exp(-nn_dist / sigma).unsqueeze(-1).unsqueeze(-1)
            C = torch.sum(kernel * (dp.unsqueeze(-1) @ dp.unsqueeze(-2)), dim=2)
            C = C + 1e-4 * torch.eye(3, device=device).view(1, 1, 3, 3)
            nn_flow = flow_sub[batch_idx, nn_idx]
            dv = nn_flow - flow_sub.unsqueeze(2)
            B_mat = torch.sum(
                kernel * (dv.unsqueeze(-1) @ dp.unsqueeze(-2)), dim=2
            )
            J = B_mat @ torch.linalg.inv(C)
            div_field = J[:, :, 0, 0] + J[:, :, 1, 1] + J[:, :, 2, 2]
            loss_div = torch.mean(div_field ** 2)

        entropy = -torch.sum(weights * torch.log(weights.clamp(min=1e-8)), dim=-1)
        target_entropy = 0.75 * torch.log(torch.tensor(float(self.num_modes), device=device))
        loss_entropy = torch.mean((entropy - target_entropy) ** 2)
        loss_total = (
            self.loss_weight_noslip * loss_noslip
            + self.loss_weight_div * loss_div
            + self.loss_weight_entropy * loss_entropy
        )
        return {
            "loss_dict_total": loss_total,
            "loss_noslip": loss_noslip,
            "loss_div": loss_div,
            "loss_entropy": loss_entropy,
            "basis_norm": torch.mean(torch.linalg.vector_norm(basis_fields, dim=-1)).detach(),
            "proxy_speed": torch.mean(speed_combined).detach(),
            "gate_entropy": torch.mean(entropy).detach(),
            "mode_weight_mean": torch.mean(weights, dim=(0, 1)).detach(),
            "mode_weight_std": torch.std(weights, dim=(0, 1)).detach(),
        }
