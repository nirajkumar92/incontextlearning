import math
import numpy as np
import torch
import pytest

from tabular_foundation.model import build_model
from tabular_foundation.optim import Muon, newton_schulz, parameter_partition, build_optimizers, schedule_multiplier


def reference_ns(g):
    x = np.asarray(g, dtype=np.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x/(np.linalg.norm(x)+1e-7)
    for _ in range(5):
        a = x@x.T
        x = 3.4445*x+(-4.775*a+2.0315*(a@a))@x
    return x.T if transposed else x


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4)])
def test_newton_schulz_matches_independent_reference(shape):
    gradient = torch.randn(shape)
    np.testing.assert_allclose(newton_schulz(gradient).numpy(), reference_ns(gradient.numpy()), atol=2e-5, rtol=2e-5)


def test_muon_momentum_decay_scaling_and_unused_parameters():
    p = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3)/10)
    unused = torch.nn.Parameter(torch.ones(2, 3))
    original = p.detach().clone()
    g = torch.tensor([[.1, .3, -.2], [.5, -.4, .2]])
    p.grad = g.clone()
    opt = Muon([p, unused], lr=.001)
    opt.step()
    expected = original*(1-.001*.01)-.2*.001*math.sqrt(3)*torch.tensor(reference_ns((1+.95)*g.numpy()))
    torch.testing.assert_close(p, expected, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(opt.state[p]["momentum_buffer"], g)
    assert unused not in opt.state
    assert torch.equal(unused, torch.ones_like(unused))
    p.grad = None
    frozen = p.detach().clone()
    opt.step()
    assert torch.equal(p, frozen)


def test_exact_named_partition_and_adamw_control():
    model = build_model("tiny")
    groups, names = parameter_partition(model)
    assert not any("encoder" in name or "inducing" in name for name in names["muon"])
    assert "encoder.log_frequencies" in names["adamw_no_decay"]
    assert "cls" in names["adamw_no_decay"]
    assert "metadata_out.weight" in names["adamw_decay"]
    assert sum(len(x) for x in groups.values()) == len(list(model.parameters()))
    opts = build_optimizers(model)
    assert len(opts) == 2
    assert all("initial_lr" in group for opt in opts for group in opt.param_groups)
    all_params = [p for opt in opts for group in opt.param_groups for p in group["params"]]
    assert len({id(p) for p in all_params}) == len(list(model.parameters()))
    assert len(build_optimizers(model, mode="adamw")) == 1


def test_schedule_boundaries():
    assert schedule_multiplier(1, 100) == .5
    assert schedule_multiplier(2, 100) == 1
    assert schedule_multiplier(100, 100) == pytest.approx(.1)
    with pytest.raises(ValueError):
        schedule_multiplier(0, 100)
