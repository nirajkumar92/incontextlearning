import numpy as np
import pytest
import torch
from scipy.integrate import quad

from tabular_foundation.distributions import ContinuousDistribution, HurdleDistribution


def make_distribution(seed=2, center=0., scale=1.):
    g = torch.Generator().manual_seed(seed)
    return ContinuousDistribution(torch.randn(132, generator=g, dtype=torch.float64), center, scale)


def test_density_and_analytic_mean_against_quadrature():
    dist = make_distribution()
    def density(y):
        return dist.log_prob(torch.tensor(y, dtype=torch.float64)).exp().item()
    masses = [quad(density, -np.inf, -5, epsabs=1e-10)[0],
              sum(quad(density, -5+i/12.8, -5+(i+1)/12.8, epsabs=1e-11)[0] for i in range(128)),
              quad(density, 5, np.inf, epsabs=1e-10)[0]]
    assert sum(masses) == pytest.approx(1., abs=2e-9)
    mean = quad(lambda y: y*density(y), -np.inf, -5)[0]
    mean += sum(quad(lambda y: y*density(y), -5+i/12.8, -5+(i+1)/12.8)[0] for i in range(128))
    mean += quad(lambda y: y*density(y), 5, np.inf)[0]
    assert dist.mean.item() == pytest.approx(mean, abs=2e-8)


@pytest.mark.parametrize("center,scale", [(0., 1.), (10., .3), (-20., .5), (4.999999, 1.)])
def test_positive_truncation_moments(center, scale):
    dist = make_distribution(center=center, scale=scale)
    edges = [0.] + sorted([center + scale*(-5+i/12.8) for i in range(129) if center + scale*(-5+i/12.8) > 0])
    edges += [np.inf]
    density = lambda y: dist.log_prob(torch.tensor(y, dtype=torch.float64)).exp().item()
    mass = sum(quad(density, a, b, epsabs=1e-11)[0] for a, b in zip(edges[:-1], edges[1:]))
    moment = sum(quad(lambda y: y*density(y), a, b, epsabs=1e-10)[0] for a, b in zip(edges[:-1], edges[1:]))
    assert dist.log_survival(0.).exp().item() == pytest.approx(mass, rel=1e-7, abs=1e-10)
    assert dist.mean_above_zero().item() == pytest.approx(moment/mass, rel=2e-7, abs=1e-8)


def test_hurdle_atom_density_and_zero_limit():
    base = make_distribution()
    dist = HurdleDistribution(base, torch.tensor(1.2, dtype=torch.float64))
    p0 = torch.sigmoid(torch.tensor(1.2, dtype=torch.float64)).item()
    assert dist.log_prob(0.).exp().item() == pytest.approx(p0)
    assert dist.cdf(0.).item() == pytest.approx(p0)
    assert dist.cdf(-1.).item() == 0
    assert dist.log_prob(-1.).item() == -torch.inf
    assert dist.mean.item() == pytest.approx((1-p0)*base.mean_above_zero().item())
    almost_zero = HurdleDistribution(base, torch.tensor(60., dtype=torch.float64))
    assert almost_zero.mean.item() < 1e-24


def test_distribution_gradients_are_finite_at_boundaries_and_extreme_threshold():
    params = torch.randn(6, 132, requires_grad=True)
    center = torch.tensor([0., 5., -5., 1e6, -1e6, 0.])
    base = ContinuousDistribution(params, center, 1.)
    atom = torch.zeros(6, requires_grad=True)
    hurdle = HurdleDistribution(base, atom)
    objective = -hurdle.log_prob(torch.tensor([0., 1., 1., 1., 1., 10.])).mean() + hurdle.mean.mean()*1e-6
    objective.backward()
    assert torch.isfinite(params.grad).all()
    assert torch.isfinite(atom.grad).all()


def test_cdf_quantile_and_vector_broadcast():
    dist = make_distribution()
    probabilities = torch.tensor([.0001, .01, .1, .5, .9, .9999], dtype=torch.float64)
    q = dist.quantile(probabilities)
    torch.testing.assert_close(dist.cdf(q), probabilities, atol=2e-12, rtol=2e-9)
    assert dist.log_prob(q).shape == probabilities.shape


def test_original_unit_jacobian_and_mean():
    base = make_distribution()
    changed = ContinuousDistribution(base.params, center=3., scale=7.)
    y = torch.tensor([3., 10., 50.], dtype=torch.float64)
    torch.testing.assert_close(changed.log_prob(y), base.log_prob((y-3)/7)-np.log(7))
    torch.testing.assert_close(changed.mean, 3+7*base.mean)


def test_very_distant_positive_truncation_and_nan_targets():
    params = torch.randn(2, 132, requires_grad=True)
    base = ContinuousDistribution(params, torch.tensor([1e30, -1e30]), 1.)
    dist = HurdleDistribution(base, torch.zeros(2))
    objective = -dist.log_prob(torch.ones(2)).mean() + dist.mean.mean() / 1e30
    objective.backward()
    assert torch.isfinite(objective)
    assert torch.isfinite(params.grad).all()
    assert torch.isnan(base.log_prob(torch.full((2,), float("nan")))).all()
    assert torch.isnan(dist.log_prob(torch.full((2,), float("nan")))).all()


def test_large_offset_small_spread_preserves_likelihood_and_parameter_gradients():
    torch.manual_seed(81)
    params = torch.randn(4, 132, requires_grad=True)
    reference_params = params.detach().clone().requires_grad_()
    z = torch.tensor([0., 2., 8., -8.], dtype=torch.float64)
    shifted = ContinuousDistribution(params, center=100000000., scale=.5)
    reference = ContinuousDistribution(reference_params)
    actual = shifted.log_prob(100000000. + .5*z)
    expected = reference.log_prob(z) - np.log(.5)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(params.grad, reference_params.grad)
    assert params.grad.dtype == torch.float32
    # Distinct targets must select distinct finite bins even though an early
    # cast of these original-unit targets would round both to 100000000.
    assert not torch.equal(params.grad[0], params.grad[1])


def test_large_offset_mean_and_quantile_preserve_sub_float32_ulp_differences():
    params = torch.full((2, 132), -1000.)
    params[0, 65] = params[1, 66] = 0.
    params[:, 130:] = 0.
    shifted = ContinuousDistribution(params, center=100000000., scale=.5)
    expected = torch.tensor([100000000. + .5*.0390625,
                             100000000. + .5*.1171875], dtype=torch.float64)
    torch.testing.assert_close(shifted.mean, expected, rtol=0, atol=1e-8)
    torch.testing.assert_close(shifted.quantile(.5), expected, rtol=0, atol=1e-8)
    assert shifted.mean[1] > shifted.mean[0]


def test_large_affine_values_do_not_overflow_before_standardization():
    dist = ContinuousDistribution(torch.zeros(1, 132), center=-1e308, scale=1e308)
    torch.testing.assert_close(dist._z(torch.tensor([1e308], dtype=torch.float64)), torch.tensor([2.]))
    assert torch.isfinite(dist.log_prob(torch.tensor([1e308], dtype=torch.float64))).all()
    assert torch.isfinite(dist.mean).all()


def test_hurdle_preserves_original_targets_and_positive_moment_at_large_offset():
    torch.manual_seed(82)
    params = torch.randn(3, 132, requires_grad=True)
    atom = torch.tensor([.2, -.3, .4], requires_grad=True)
    targets = torch.tensor([0., 100000000., 100000001.], dtype=torch.float64)
    actual = HurdleDistribution(ContinuousDistribution(params, 100000000., .5), atom)
    reference_params = params.detach().double().requires_grad_()
    reference_atom = atom.detach().double().requires_grad_()
    reference = HurdleDistribution(ContinuousDistribution(reference_params, 100000000., .5), reference_atom)
    torch.testing.assert_close(actual.log_prob(targets).double(), reference.log_prob(targets), atol=2e-6, rtol=2e-6)
    # Test the conditional mean separately: its probabilities must sum to one
    # accurately enough that a 1e8 offset does not amplify their rounding error.
    torch.testing.assert_close(actual.base.mean_above_zero(), reference.base.mean_above_zero(), atol=1e-6, rtol=0)
    (-actual.log_prob(targets).sum()).backward()
    (-reference.log_prob(targets).sum()).backward()
    torch.testing.assert_close(params.grad.double(), reference_params.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(atom.grad.double(), reference_atom.grad, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(params.grad).all()
    assert torch.isfinite(actual.mean).all()
