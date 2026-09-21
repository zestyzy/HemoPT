import numpy as np
import torch
from pathlib import Path

from data_provider.vascular_pretrain_loader import _VascularPretrainDataset
from utils.loss import L2Loss
from data_generation.Vascular_PreTraining_Data import build_probe_initializers


ROOT = Path(__file__).resolve().parents[1]


def test_pretrain_scripts_do_not_contain_literal_backslash_n():
    scripts = [p for p in (ROOT / "scripts" / "pretrain").glob("*.sh") if not p.name.startswith("._")]
    bad = [str(p) for p in scripts if "\\\\n" in p.read_text()]
    assert not bad, f"literal \\n found in shell scripts: {bad}"


def test_broken_sample_is_not_replaced_by_another_sample(tmp_path):
    valid = tmp_path / "valid"
    broken = tmp_path / "broken"
    valid.mkdir()
    broken.mkdir()
    np.save(valid / "x.npy", np.zeros((4, 7), dtype=np.float32))
    np.save(valid / "condition_0.npy", np.zeros((4, 4), dtype=np.float32))
    np.save(valid / "supervise_0.npy", np.zeros((4, 9), dtype=np.float32))
    dataset = _VascularPretrainDataset([str(valid), str(broken)], n_random_walks=1)
    try:
        dataset[1]
    except FileNotFoundError:
        return
    raise AssertionError("broken sample was silently replaced by another sample")


def test_relative_l2_is_finite_for_zero_target():
    value = L2Loss(size_average=False)(torch.ones(1, 4, 3), torch.zeros(1, 4, 3))
    assert torch.isfinite(value), value


def test_base_walks_are_reused_before_perturbation():
    np.random.seed(2026)
    dirs, steps = build_probe_initializers(
        total_pts=8, n_random_walks=4, base_walks=2, perturb_sigma=0.0
    )
    assert np.array_equal(dirs[0], dirs[2])
    assert np.array_equal(dirs[1], dirs[3])
    assert np.array_equal(steps[0], steps[2])
    assert np.array_equal(steps[1], steps[3])
