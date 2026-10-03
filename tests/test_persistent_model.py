"""Persistent cells must keep the same inductive and gradient contracts."""
import copy

import numpy as np
import pytest
import torch

from tabular_foundation.distributions import regression_loss
from tabular_foundation.model import build_model
from tabular_foundation.optim import parameter_partition


OPTIONS = dict(categorical_encoding="hash_bits", regression_head="quantile",
               regression_bins=1025, regression_quantiles=999,
               length_scaling="logarithmic", trunk_query_kv_heads=2,
               hurdle_target_scale="positive_support")


def test_persistent_geometry_and_optimizer_partition():
    model = build_model("persistent_small", device="meta", model_options=OPTIONS)
    assert model.parameter_count() == 41_089_248
    assert len(model.trunk) == 0
    assert len(model.columns) == len(model.rows) == 12
    assert model.config.cell_width == 256
    groups, names = parameter_partition(model)
    # Every round retains its registered column and row hidden matrices.
    assert len(groups["muon"]) == 12 * 3 * 7
    assert any("columns.11.0.first.q.weight" == n for n in names["muon"])
    assert any("rows.11.0.q.weight" == n for n in names["muon"])


@pytest.mark.parametrize("task", ["multiclass", "regression"])
def test_persistent_cache_isolation_and_support_permutation(task):
    torch.set_num_threads(2)
    torch.manual_seed(37)
    m = build_model("persistent_tiny", model_options=OPTIONS).eval()
    rng = np.random.default_rng(37)
    xs, xq = rng.normal(size=(11, 5)), rng.normal(size=(7, 5))
    xs[:, 2] = np.arange(11) % 3; xq[:, 2] = np.arange(7) % 4
    xs[0, 0] = np.nan; xq[0, 2] = np.nan
    ys = np.arange(11) % 3 if task == "multiclass" else rng.normal(size=11)
    key = "logits" if task == "multiclass" else "mean"
    with torch.no_grad():
        cache = m.prepare_context(xs, ys, [2], task, n_classes=3)
        whole = m.predict_cached(cache, xq)[key]
        parts = torch.cat([m.predict_cached(cache, z)[key] for z in (xq[:2], xq[2:])])
        modified = xq.copy(); modified[1:] = 1e5
        isolated = m.predict_cached(cache, modified)[key]
        order = rng.permutation(len(xs))
        shuffled = m(xs[order], ys[order], xq, [2], task, n_classes=3)[key]
    torch.testing.assert_close(whole, parts, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(whole[0], isolated[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(whole, shuffled, atol=2e-6, rtol=2e-5)
    assert len(cache.frontend) == 3 and cache.trunk == []


def test_persistent_checkpointing_and_cell_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(81)
    model = build_model("persistent_tiny", model_options=OPTIONS)
    checkpointed = copy.deepcopy(model); checkpointed.activation_checkpointing = True
    rng = np.random.default_rng(81)
    x = rng.normal(size=(7, 4)); y = rng.normal(size=7); q = rng.normal(size=(3, 4))
    outputs = [m(x, y, q, [], "regression") for m in (model, checkpointed)]
    torch.testing.assert_close(outputs[0]["mean"], outputs[1]["mean"])
    for out in outputs:
        regression_loss(out, torch.tensor([.3, -.2, .9])).mean().backward()
    for (name, p), (_, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, other.grad, atol=2e-6, rtol=3e-4)
    for stage in model.columns:
        assert stage[0].first.q.weight.grad.abs().sum() > 0
    assert model.early_reg.weight.grad.abs().sum() > 0


def test_persistent_finance_boundary_and_cache_version_guard():
    torch.set_num_threads(2)
    model = build_model("persistent_tiny", model_options=OPTIONS)
    x = np.zeros((5, 3)); q = np.ones((2, 3))
    out = model(x, np.zeros(5), q, [], "regression", hurdle=True,
                finance_metadata=np.zeros(14), source_bits=np.zeros((5, 3)))
    loss = regression_loss(out, torch.tensor([0., 2.])).mean()
    loss.backward()
    assert torch.isfinite(loss) and model.atom_head.weight.grad.abs().sum() > 0
    cache = model.prepare_context(x[:0], np.empty(0), [], "binary")
    assert torch.isfinite(model.predict_cached(cache, q)["logits"]).all()
    with torch.no_grad():
        model.class_head.weight.add_(.01)
    with pytest.raises(RuntimeError, match="stale"):
        model.predict_cached(cache, q)
