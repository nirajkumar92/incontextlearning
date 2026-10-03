"""Counterexamples, proper-score contracts and cache laws for candidate options."""
import copy
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from tabular_foundation.codec import TableCodec
from tabular_foundation.distributions import ContinuousDistribution, HurdleDistribution, regression_loss, regression_nll
from tabular_foundation.model import MAB, ModelConfig, build_model


def setup_module():
    torch.set_num_threads(2)


def options(head="histogram"):
    return dict(categorical_encoding="hash_bits", regression_head=head,
                regression_bins=1025, length_scaling="logarithmic", trunk_query_kv_heads=1,
                hurdle_target_scale="positive_support")


def test_discrete_identity_repairs_the_observed_bf16_counterexample():
    # Equal support frequencies remove the metadata route to distinguishability.
    pair = np.array([["category_28838"], ["category_39200"]], dtype=object)
    legacy = TableCodec().fit(pair, [0]).transform(pair)
    codec = TableCodec(categorical_encoding="hash_bits").fit(pair, [0])
    repaired = codec.transform(pair)
    assert 0 < abs(legacy.categorical[0, 0, 0] - legacy.categorical[1, 0, 0]) < 2e-9
    assert not torch.equal(repaired.categorical_identity[0], repaired.categorical_identity[1])
    assert set(repaired.categorical_identity.flatten().tolist()) == {-1, 1}
    for seed in (1729, 83, 99):
        torch.manual_seed(seed)
        model = build_model("tiny", finance=False, model_options=options())
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.encoder(repaired, torch.zeros(2, model.config.cell_width))
        assert not torch.equal(actual[0], actual[1])
        assert (actual[0] - actual[1]).abs().max() > .1


def test_hash_bits_have_stable_typed_identity_missingness_and_support_only_counts():
    support = np.array([[True], [1], ["1"], [None], ["heldout"]], dtype=object)
    codec = TableCodec(42, categorical_encoding="hash_bits").fit(support, [0], indices=[0, 1, 2, 3])
    original = codec.transform(support)
    assert torch.unique(original.categorical_identity[:3, 0], dim=0).shape[0] == 3
    assert original.categorical_identity[3].count_nonzero() == 0
    assert original.categorical[4, 0, 2] == 1
    snapshot = copy.deepcopy(codec.stats)
    unknown = codec.transform(np.array([["heldout"], ["new"]], dtype=object))
    torch.testing.assert_close(unknown.categorical_identity[0], original.categorical_identity[4])
    assert codec.stats == snapshot
    permuted = codec.transform(support[::-1])
    torch.testing.assert_close(permuted.categorical_identity.flip(0), original.categorical_identity)
    other_seed = TableCodec(43, categorical_encoding="hash_bits").fit(support, [0]).transform(support)
    assert not torch.equal(original.categorical_identity[:3], other_seed.categorical_identity[:3])


@pytest.mark.parametrize("bins", [128, 1025])
def test_configurable_histogram_mass_mean_cdf_and_affine_covariance(bins):
    torch.manual_seed(8)
    params = torch.randn(bins + 4, dtype=torch.float64)
    dist = ContinuousDistribution(params, bins=bins)
    edges = torch.linspace(-5, 5, bins + 1, dtype=torch.float64)
    midpoints = (edges[1:] + edges[:-1]) / 2
    density = lambda y: dist.log_prob(y).exp().item()
    interior_mass = (dist.log_prob(midpoints).exp() * (10 / bins)).sum().item()
    mass = interior_mass + quad(density, -np.inf, -5)[0] + quad(density, 5, np.inf)[0]
    assert mass == pytest.approx(1., abs=1e-10)
    mean = (dist.log_prob(midpoints).exp() * (10 / bins) * midpoints).sum().item()
    mean += quad(lambda y: y * density(y), -np.inf, -5)[0] + quad(lambda y: y * density(y), 5, np.inf)[0]
    assert dist.mean.item() == pytest.approx(mean, abs=1e-9)
    probabilities = torch.tensor([.00001, .03, .37, .78, .99999], dtype=torch.float64)
    torch.testing.assert_close(dist.cdf(dist.quantile(probabilities)), probabilities, rtol=1e-9, atol=2e-12)
    shifted = ContinuousDistribution(params, center=1e8, scale=.5, bins=bins)
    targets = torch.tensor([1e8, 1e8 + 1], dtype=torch.float64)
    torch.testing.assert_close(shifted.log_prob(targets), dist.log_prob(2*(targets-1e8)) + math.log(2))
    torch.testing.assert_close(shifted.mean, 1e8 + .5*dist.mean)


def test_fine_histogram_reduces_resolution_floor_but_does_not_eliminate_it():
    target = torch.tensor([.005, .07], dtype=torch.float64)
    for bins, identical in ((128, True), (1025, False)):
        params = torch.zeros(2, bins + 4, dtype=torch.float64, requires_grad=True)
        loss = -ContinuousDistribution(params, bins=bins).log_prob(target)
        loss.sum().backward()
        assert torch.equal(params.grad[0], params.grad[1]) == identical
    params = torch.full((1029,), -1000., dtype=torch.float64)
    params[513] = 0.  # The midpoint of this central bin is exactly zero.
    params[-2:] = 0.
    dist = ContinuousDistribution(params, center=7., bins=1025)
    assert dist.mean.item() == 7.
    # A fine grid is still blind to target locations inside each finite bin.
    torch.testing.assert_close(dist.log_prob(torch.tensor([7.0001, 7.0002], dtype=torch.float64)),
                               dist.log_prob(torch.tensor([7.0002, 7.0001], dtype=torch.float64)))


def test_quantile_loss_resolves_crossing_targets_and_preserves_affine_precision():
    model = build_model("tiny", finance=False, model_options=options("quantile"))
    out = model(np.zeros((3, 1)), np.full(3, 1e8), np.zeros((2, 1)), [], "regression")
    assert out["regression_params"].shape == (2, 999)
    assert out["loss_kind"] == "pinball" and "distribution" not in out
    torch.testing.assert_close(out["quantile_levels"][[0, -1]], torch.tensor([.001, .999], dtype=torch.float64))
    assert (out["quantiles"].diff(dim=-1) >= 0).all()
    torch.testing.assert_close(out["mean"], out["target_center"] + out["target_scale"] * out["regression_params"].double().mean(-1))
    params = torch.full((2, 999), .03, requires_grad=True)
    out.update(regression_params=params, target_center=0., target_scale=1.)
    loss = regression_loss(out, torch.tensor([.005, .07], dtype=torch.float64))
    loss.sum().backward()
    assert not torch.equal(params.grad[0], params.grad[1])
    reference = regression_loss(out, torch.tensor([0., 2.], dtype=torch.float64))
    out.update(target_center=1e8, target_scale=.5)
    torch.testing.assert_close(regression_loss(out, torch.tensor([1e8, 1e8+1], dtype=torch.float64)), reference)
    with pytest.raises(ValueError, match="pinball"):
        regression_nll(out, torch.zeros(2))


def test_quantile_finite_grid_cannot_identify_rare_event_mean():
    # Both point mass zero and P(Y=1000000)=1e-4 have every .001:.001:.999
    # quantile equal to zero, yet their means differ by 100. No finite-grid
    # pinball optimum alone identifies this missing upper-tail contribution.
    levels = torch.arange(1, 1000, dtype=torch.float64) / 1000
    quantiles = torch.where(levels <= 1 - 1e-4, 0., 1e6)
    assert quantiles.count_nonzero() == 0
    assert 1e-4 * 1e6 == 100


@pytest.mark.parametrize("head", ["histogram", "quantile"])
def test_candidate_cache_query_isolation_class_mask_and_gradients(head):
    torch.manual_seed(84)
    rng = np.random.default_rng(84)
    xs = rng.normal(size=(9, 3)).astype(object)
    xs[:, 1] = ["a", "b", "c"] * 3
    xq = xs[:5].copy()
    model = build_model("tiny", model_options=options(head))
    chunked = copy.deepcopy(model)
    y = np.linspace(-1, 1, len(xs))
    whole = model(xs, y, xq, [1], "regression")
    cache = chunked.prepare_context(xs, y, [1], "regression")
    pieces = [chunked.predict_cached(cache, q) for q in (xq[:2], xq[2:])]
    torch.testing.assert_close(whole["mean"], torch.cat([p["mean"] for p in pieces]), atol=2e-6, rtol=2e-5)
    targets = torch.linspace(-.5, .8, len(xq), dtype=torch.float64)
    regression_loss(whole, targets).sum().backward()
    sum(regression_loss(p, t).sum() for p, t in zip(pieces, (targets[:2], targets[2:]))).backward()
    for (name, p), (_, other) in zip(model.named_parameters(), chunked.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, other.grad, atol=3e-6, rtol=4e-4, msg=name)
    with torch.no_grad():
        a = model(xs, np.arange(9) % 3, xq, [1], "multiclass", 3, [255, 19, 7])
        xq[1:, 0] = 1e4
        b = model(xs, np.arange(9) % 3, xq, [1], "multiclass", 3, [255, 19, 7])
        torch.testing.assert_close(a["logits"][0], b["logits"][0])
        assert torch.isneginf(a["slot_logits"][:, 0]).all()


def test_quantile_candidate_retains_true_finance_atom_and_density_gradients():
    model = build_model("tiny", model_options=options("quantile"))
    x = np.arange(8).reshape(4, 2)
    out = model(x, np.zeros(4), x, [], "regression", hurdle=True)
    assert out["loss_kind"] == "nll" and isinstance(out["distribution"], HurdleDistribution)
    assert out["regression_params"].shape == (4, 1029)
    loss = regression_loss(out, torch.tensor([0., 0., 1., 10.], dtype=torch.float64)).mean()
    loss.backward()
    assert model.hurdle_regression_head.weight.grad.abs().sum() > 0
    assert model.atom_head.weight.grad.abs().sum() > 0
    assert model.regression_head.weight.grad is None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(out["distribution"].cdf(0.), out["atom_logit"].squeeze(-1).sigmoid())


def test_logarithmic_temperature_and_single_anchor_do_not_saturate():
    old = MAB(16, 2, 32, 1.)
    new = MAB(16, 2, 32, 1., length_scaling="logarithmic")
    torch.testing.assert_close(new.attention_temperature(1024), torch.ones(2))
    with torch.no_grad():
        old.length_scale.fill_(.4)
    torch.testing.assert_close(old.attention_temperature(4096), old.attention_temperature(32768))
    assert (new.attention_temperature(32768) > new.attention_temperature(4096)).all()
    # Calibrate the same effective gap at 4096, then increase only key count.
    gap = math.log(4095)
    n = 32768
    unscaled_mass = 1 / (1 + (n-1)*math.exp(-gap))
    gap_multiplier = math.log(n)/math.log(4096)
    log_mass = 1 / (1 + (n-1)*math.exp(-gap*gap_multiplier))
    assert unscaled_mass < .112 and log_mass > .499


def test_candidate_parameter_counts_and_legacy_state_geometry():
    old = build_model("base", finance=True, device="meta")
    hist = build_model("base", finance=True, device="meta", model_options=options())
    quant = build_model("base", finance=True, device="meta", model_options=options("quantile"))
    assert old.parameter_count() == 315175184
    assert hist.parameter_count() == 316093328
    assert quant.parameter_count() == 317116304
    assert old.regression_head.weight.shape == (132, 1024)
    assert old.encoder.categorical[0].weight.shape == (128, 68)
    assert hist.regression_head.weight.shape == (1029, 1024)
    assert quant.regression_head.weight.shape == (999, 1024)
    assert quant.hurdle_regression_head.weight.shape == (1029, 1024)
    with pytest.raises(ValueError):
        ModelConfig(regression_bins=0)
    with pytest.raises(ValueError, match="unknown model_options"):
        build_model(model_options={"width": 1})


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_query_grouped_attention_matches_explicit_prefix_repetition_and_gradients(kv_heads):
    torch.manual_seed(91)
    module = MAB(64, 4, 96, .5, length_scaling="logarithmic")
    repeated = copy.deepcopy(module)
    queries = torch.randn(7, 64)
    support = torch.randn(9, 64)
    actual = module.from_kv(queries, module.query_kv(module.prepare_kv(support), kv_heads))
    full = repeated.prepare_kv(support)
    # Independent, explicit reference for SDPA's consecutive grouping law.
    expanded = tuple(t[..., :kv_heads, :, :].repeat_interleave(4//kv_heads, dim=-3) for t in full)
    expected = repeated.from_kv(queries, expanded)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, p), (_, other) in zip(module.named_parameters(), repeated.named_parameters()):
        torch.testing.assert_close(p.grad, other.grad, rtol=3e-5, atol=3e-5, msg=name)
    # This isolated query call only reads the selected rows of each KV matrix.
    assert module.k.weight.grad[kv_heads*16:].count_nonzero() == 0
    assert module.v.weight.grad[kv_heads*16:].count_nonzero() == 0


@pytest.mark.parametrize("support_rows", [1, 9])
def test_query_cache_storage_shrinks_without_changing_support_computation(support_rows):
    torch.manual_seed(92)
    full = build_model("tiny").eval()
    compact = build_model("tiny", model_options={"trunk_query_kv_heads": 1}).eval()
    compact.load_state_dict(full.state_dict())
    rng = np.random.default_rng(92)
    support = rng.normal(size=(support_rows, 3))
    labels = np.arange(support_rows) % 2
    with torch.inference_mode():
        all_heads = full.prepare_context(support, labels, [], "binary")
        prefix = compact.prepare_context(support, labels, [], "binary")
    for full_pair, compact_pair in zip(all_heads.trunk, prefix.trunk):
        for a, b in zip(full_pair, compact_pair):
            # Every later layer matches too: support updates used all heads.
            torch.testing.assert_close(b, a[:1])
            assert b.untyped_storage().nbytes() == b.numel() * b.element_size()
            assert b.numel() == a.numel()//4
    assert sum(t.untyped_storage().nbytes() for p in prefix.trunk for t in p) == sum(t.numel()*t.element_size() for p in all_heads.trunk for t in p)//4
    for full_stage, compact_stage in zip(all_heads.frontend, prefix.frontend):
        for full_pair, compact_pair in zip(full_stage, compact_stage):
            for a, b in zip(full_pair, compact_pair):
                torch.testing.assert_close(a, b)


@pytest.mark.parametrize("value", [0, -1, 3, 5, True, 1.5])
def test_query_head_count_validation(value):
    with pytest.raises(ValueError, match="positive divisor"):
        build_model("tiny", model_options={"trunk_query_kv_heads": value})


def test_query_grouping_empty_support_and_checkpointed_gradients():
    torch.manual_seed(93)
    model = build_model("tiny", model_options=options())
    empty = model(np.empty((0, 2)), np.empty(0), np.ones((2, 2)), [], "binary")
    assert torch.isfinite(empty["logits"]).all()
    clone = copy.deepcopy(model)
    clone.activation_checkpointing = True
    xs = np.arange(16).reshape(8, 2)/10
    ys = np.arange(8) % 2
    a = model(xs, ys, xs[:3], [], "binary")["logits"]
    b = clone(xs, ys, xs[:3], [], "binary")["logits"]
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(model.parameters(), clone.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-4)


def test_hurdle_positive_scale_uses_observed_support_without_changing_feature_codec():
    model = build_model("tiny", model_options=dict(options(), hurdle_target_scale="positive_support"))
    x = np.arange(16).reshape(8, 2)
    y = np.array([0., 0., 0., 0., 100., 200., 300., 400.])
    cache = model.prepare_context(x, y, [], "regression", codec_indices=[0, 1, 2, 3], hurdle=True)
    assert cache.target_center == 250.
    assert cache.target_scale == pytest.approx(148.26)
    assert cache.codec.fit_indices.tolist() == [0, 1, 2, 3]
    a = model.predict_cached(cache, x[:2])
    b = model.predict_cached(cache, x[:2]+1e6)
    assert a["target_center"] == b["target_center"] == 250.
    assert a["target_scale"] == b["target_scale"] == pytest.approx(148.26)
    # Standard regression and legacy hurdle keep reference-reservoir scaling.
    ordinary = model.prepare_context(x, y, [], "regression", codec_indices=[0, 1, 2, 3])
    legacy = build_model("tiny").prepare_context(x, y, [], "regression", codec_indices=[0, 1, 2, 3], hurdle=True)
    assert (ordinary.target_center, ordinary.target_scale) == (0., 1.)
    assert (legacy.target_center, legacy.target_scale) == (0., 1.)
    no_positive = model.prepare_context(x, np.zeros(8), [], "regression", hurdle=True)
    assert (no_positive.target_center, no_positive.target_scale) == (0., 1.)


def test_hurdle_positive_scale_density_atom_and_original_unit_covariance():
    torch.manual_seed(99)
    params = torch.randn(1029, dtype=torch.float64)
    atom = torch.tensor(.7, dtype=torch.float64)
    a = HurdleDistribution(ContinuousDistribution(params, 250., 148.26, bins=1025), atom)
    b = HurdleDistribution(ContinuousDistribution(params, 2500., 1482.6, bins=1025), atom)
    targets = torch.tensor([.01, 50., 150., 350., 1000.], dtype=torch.float64)
    torch.testing.assert_close(b.log_prob(10*targets), a.log_prob(targets)-math.log(10))
    torch.testing.assert_close(b.mean, 10*a.mean)
    torch.testing.assert_close(b.cdf(10*targets), a.cdf(targets))
    assert a.cdf(0.).item() == pytest.approx(atom.sigmoid().item())
    # Conditional positive density integrates to1-p0, with edges at physical0.
    edges = sorted({0., *[float(v) for v in 250+148.26*torch.linspace(-5, 5, 1026, dtype=torch.float64) if v > 0], np.inf})
    mass = sum(quad(lambda t: a.log_prob(t).exp().item(), left, right, epsabs=1e-10)[0]
               for left, right in zip(edges[:-1], edges[1:]))
    assert mass + a.log_prob(0.).exp().item() == pytest.approx(1., abs=1e-8)


def test_hurdle_positive_scale_preserves_large_offset_and_finite_gradients():
    model = build_model("tiny", model_options=dict(options("quantile"), hurdle_target_scale="positive_support"))
    x = np.arange(12).reshape(6, 2)
    y = np.array([0., 0., 1e8, 1e8+1, 1e8+2, 1e8+3], dtype=np.float64)
    out = model(x, y, x[:3], [], "regression", codec_indices=[0, 1], hurdle=True)
    assert out["target_center"] == 1e8+1.5
    assert out["target_scale"] == pytest.approx(1.4826, abs=2e-8)
    z = out["distribution"].base._z(torch.tensor([1e8, 1e8+1, 1e8+2], dtype=torch.float64))
    assert (z.diff() > .6).all()
    loss = regression_loss(out, torch.tensor([0., 1e8+1, 1e8+2], dtype=torch.float64)).mean()
    loss.backward()
    assert torch.isfinite(out["mean"]).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="hurdle_target_scale"):
        ModelConfig(hurdle_target_scale="query")
