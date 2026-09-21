import torch
import numpy as np
import json
from types import SimpleNamespace

from models.dynamic_flow_dict import (
    AnalyticCompactFlowDictionary,
    RandomWalkTrajectoryEncoder,
)


def test_random_walk_trajectory_encoder_supports_flat_and_sequence_inputs():
    torch.manual_seed(0)
    encoder = RandomWalkTrajectoryEncoder(
        num_steps=3, vector_dim=3, hidden_dim=8, output_dim=6
    )
    trajectory = torch.randn(2, 5, 9)
    flat_features = encoder(trajectory)
    sequence_features = encoder(trajectory.reshape(2, 5, 3, 3))

    assert flat_features.shape == (2, 5, 6)
    assert torch.allclose(flat_features, sequence_features, atol=1e-6)


def test_analytic_dictionary_routes_from_random_walk_trajectory():
    torch.manual_seed(0)
    model = AnalyticCompactFlowDictionary(
        morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=16,
        rw_encoder_hidden_dim=8, rw_feature_dim=6,
    )
    pos = torch.randn(1, 6, 3)
    morph = torch.randn(1, 6, 7)
    cond = torch.randn(1, 6, 4)
    analytic_basis = torch.randn(1, 6, 8, 3)
    trajectory = torch.randn(1, 6, 9, requires_grad=True)

    compact, _, weights = model(
        pos, morph, cond, analytic_basis, rw_trajectory=trajectory
    )
    compact[..., :3].square().mean().backward()

    assert compact.shape == (1, 6, 4)
    assert weights.shape == (1, 6, 8)
    assert trajectory.grad is not None
    assert model.rw_encoder.step_mlp[0].weight.grad is not None
from data_provider.vascular_pretrain_loader import (
    _VascularPretrainDataset,
    PHYSICS_PROXY_DIMS,
    VascularPretrain,
)


def test_analytic_dictionary_mixes_eight_vectors_and_recomputes_speed():
    torch.manual_seed(0)
    model = AnalyticCompactFlowDictionary(
        morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=16
    )
    pos = torch.zeros(2, 5, 3)
    morph = torch.randn(2, 5, 7)
    cond = torch.randn(2, 5, 4)
    analytic_basis = torch.randn(2, 5, 8, 3)

    compact, basis, weights = model(pos, morph, cond, analytic_basis)

    assert compact.shape == (2, 5, 4)
    assert basis.shape == (2, 5, 8, 3)
    assert weights.shape == (2, 5, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.allclose(
        compact[..., 3], torch.linalg.vector_norm(compact[..., :3], dim=-1), atol=1e-6
    )


def test_analytic_dictionary_keeps_gate_differentiable():
    model = AnalyticCompactFlowDictionary(
        morph_dim=7, cond_dim=4, num_modes=8, hidden_dim=16
    )
    pos = torch.randn(1, 6, 3)
    morph = torch.randn(1, 6, 7)
    cond = torch.randn(1, 6, 4)
    analytic_basis = torch.randn(1, 6, 8, 3)

    compact, _, weights = model(pos, morph, cond, analytic_basis)
    loss = compact[..., :3].square().mean()
    loss.backward()

    assert weights.requires_grad
    assert model.routing_gate.net[0].weight.grad is not None


def test_loader_exposes_eight_three_vector_analytic_bank():
    dataset = _VascularPretrainDataset(
        [], physics_proxy=True,
        physics_proxy_mode="learnable_generalized_flow_compact",
    )
    x = np.zeros((32, 7), dtype=np.float32)
    x[:, :3] = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    x[:, 3] = np.linspace(0.01, 1.0, 32, dtype=np.float32)
    bank = dataset._build_generalized_flow_bank(x, "synthetic")

    assert PHYSICS_PROXY_DIMS["learnable_generalized_flow_compact"] == 4
    assert bank.shape == (32, 8, 3)
    assert np.isfinite(bank).all()


def test_dynamic_compact_mode_requires_rw20_base20_positive_perturbation(tmp_path):
    loader = VascularPretrain.__new__(VascularPretrain)
    loader.physics_proxy_mode = "learnable_generalized_flow_compact"
    meta = {
        "n_random_walks": 20,
        "base_walks": 20,
        "perturb_sigma": 0.05,
        "walk_steps": 3,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert loader._passes_walk_contract(str(tmp_path))

    meta["perturb_sigma"] = 0.0
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert not loader._passes_walk_contract(str(tmp_path))


def test_dictionary_loss_is_inside_flow_loss_and_batch_scaled():
    from exp.vascular_pretrain import Exp_VascularPretrain

    exp = object.__new__(Exp_VascularPretrain)
    exp.args = SimpleNamespace(vascular_physics_proxy=True, vascular_physics_weight=0.25)
    out = torch.ones(2, 1, 13)
    target = torch.zeros_like(out)
    unit_loss = lambda pred, truth: pred.sum()

    total, geom, phys = exp._loss_components(
        out, target, unit_loss, dict_loss_val=torch.tensor(5.0)
    )

    # L2Loss(size_average=False) returns batch sums: geom=18, phys=8.
    # dict_loss is a batch mean (5), so it must be multiplied by B=2
    # before joining the flow branch: 18 + .25 * (8 + 10) = 22.5.
    assert torch.allclose(geom, torch.tensor(18.0))
    assert torch.allclose(phys, torch.tensor(8.0))
    assert torch.allclose(total, torch.tensor(22.5))
