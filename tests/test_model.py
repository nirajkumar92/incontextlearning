import copy
import numpy as np
import pytest
import torch

from tabular_foundation.codec import TableCodec, robust_location_scale
from tabular_foundation.model import build_model


def setup_module():
    torch.set_num_threads(2)


def data():
    rng = np.random.default_rng(17)
    return rng.normal(size=(9, 4)), np.arange(9) % 3, rng.normal(size=(7, 4))


def test_exact_parameter_counts_and_size_grid():
    assert build_model("base", finance=False, device="meta").parameter_count() == 315021456
    assert build_model("base", finance=True, device="meta").parameter_count() == 315175184
    counts = [build_model(name, finance=False, device="meta").parameter_count()
              for name in ("small50", "small100", "base", "wide500", "large1000")]
    assert counts == sorted(counts)
    assert all(x > 0 for x in counts)


def test_cache_chunk_and_permutation_invariance():
    torch.manual_seed(2)
    model = build_model("tiny").eval()
    xs, ys, xq = data()
    with torch.no_grad():
        full = model(xs, ys, xq, [], "multiclass", n_classes=3)["logits"]
        cache = model.prepare_context(xs, ys, [], "multiclass", n_classes=3)
        chunks = torch.cat([model.predict_cached(cache, xq[:2])["logits"],
                            model.predict_cached(cache, xq[2:])["logits"]])
        perm = np.array([3, 0, 6, 2, 8, 5, 1, 7, 4])
        shuffled = model(xs[perm], ys[perm], xq, [], "multiclass", n_classes=3)["logits"]
    torch.testing.assert_close(full, chunks, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(full, shuffled, atol=2e-6, rtol=2e-5)


def test_query_isolation_and_class_slot_mapping():
    model = build_model("tiny").eval()
    xs, ys, xq = data()
    slots = [255, 13, 104]
    first = model(xs, ys, xq, [], "multiclass", 3, slots)
    modified = xq.copy()
    modified[1:] = 10000
    second = model(xs, ys, modified, [], "multiclass", 3, slots)
    torch.testing.assert_close(first["logits"][0], second["logits"][0])
    torch.testing.assert_close(first["slot_logits"][:, slots], first["logits"])
    assert torch.isneginf(first["slot_logits"][:, 0]).all()
    with pytest.raises(ValueError):
        model(xs, ys, xq, [], "multiclass", 3, [1, 1, 2])
    with pytest.raises(ValueError, match="integer slot"):
        model(xs, ys, xq, [], "multiclass", 3, [1.5, 2, 3])


def test_empty_support_and_empty_queries():
    model = build_model("tiny")
    xs = np.empty((0, 3))
    out = model(xs, np.empty(0), np.ones((2, 3)), [], "binary")
    assert out["logits"].shape == (2, 2)
    assert torch.isfinite(out["logits"]).all()
    out["logits"].square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = model(xs, np.empty(0), np.empty((0, 3)), [], "binary")
    assert empty["logits"].shape == (0, 2)


def test_gradient_equivalence_with_cached_query_chunks():
    torch.manual_seed(8)
    model = build_model("tiny")
    chunked = copy.deepcopy(model)
    xs, ys, xq = data()
    model(xs, ys, xq, [], "multiclass", 3)["logits"].square().sum().backward()
    cache = chunked.prepare_context(xs, ys, [], "multiclass", 3)
    loss = sum(chunked.predict_cached(cache, q)["logits"].square().sum() for q in (xq[:3], xq[3:]))
    loss.backward()
    for (name, p), (_, other) in zip(model.named_parameters(), chunked.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, other.grad, atol=1e-6, rtol=2e-4, msg=name)
    assert model.early_class.weight.grad.abs().sum() > 0


def test_activation_checkpointing_preserves_forward_and_gradients():
    model = build_model("tiny")
    other = copy.deepcopy(model)
    other.activation_checkpointing = True
    xs, ys, xq = data()
    a = model(xs, ys, xq, [], "multiclass", 3)["logits"]
    b = other(xs, ys, xq, [], "multiclass", 3)["logits"]
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(model.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-4)


def test_stale_cache_rejected():
    model = build_model("tiny")
    xs, ys, xq = data()
    cache = model.prepare_context(xs, ys, [], "multiclass", 3)
    with torch.no_grad():
        model.class_head.weight.add_(.1)
    with pytest.raises(RuntimeError, match="stale"):
        model.predict_cached(cache, xq)


def test_codec_reservoir_nominal_and_nonfinite_contract():
    support = np.array([[1., "a"], [1., "b"], [5., None], [1000., "private"]], dtype=object)
    codec = TableCodec(5).fit(support, [False, True], indices=[0, 1, 2])
    encoded = codec.transform(np.array([[np.inf, "new"], [np.nan, None], [-np.inf, "a"]], dtype=object))
    assert torch.isfinite(encoded.numeric).all()
    assert encoded.numeric[0, 0, 4] == 1
    assert encoded.numeric[2, 0, 5] == 1
    assert encoded.categorical[0, 1, 2] == 1
    assert encoded.categorical[1, 1, 3] == 1
    assert codec.stats[0]["center"] == 1
    assert b"str:private" not in codec.stats[1]["counts"]
    before = codec.effective_seed
    codec.transform(np.array([[8., "different"]], dtype=object))
    assert codec.effective_seed == before
    assert robust_location_scale([0, 0, 0]) == (0., 1.)


def test_codec_extreme_finite_statistics_stay_finite():
    x = np.array([[1.7e308, 1e308], [-1.7e308, 1e308], [1.7e308, 1e308], [-1.7e308, 1e308]])
    codec = TableCodec().fit(x, [])
    encoded = codec.transform(x)
    assert torch.isfinite(encoded.numeric).all()
    assert all(np.isfinite(stat["center"]) and np.isfinite(stat["scale"]) for stat in codec.stats)
    np.testing.assert_allclose(encoded.numeric[:, 0, 0], [1.7e308 / np.finfo(np.float64).max,
                              -1.7e308 / np.finfo(np.float64).max] * 2)
    with pytest.raises(ValueError, match="categorical"):
        TableCodec().fit(x, [0.5])
    with pytest.raises(ValueError, match="integer support"):
        TableCodec().fit(x, [], indices=[0.5])


def test_finance_adapter_starts_zero_and_reference_offsets_are_legal():
    model = build_model("tiny")
    xs, ys, xq = data()
    normal = model(xs, ys, xq, [], "multiclass", 3)["logits"]
    adapted = model(xs, ys, xq, [], "multiclass", 3,
                    finance_metadata=np.ones(14), source_bits=np.ones((9, 3)))["logits"]
    torch.testing.assert_close(normal, adapted)
    ref = np.array([.9, .09, .01])
    shifted = model(xs, ys, xq, [], "multiclass", 3, reference_probs=ref)["logits"]
    torch.testing.assert_close(shifted, normal + torch.tensor(ref).float().log())
    assert model.metadata_out.weight.count_nonzero() == 0


def test_regression_hurdle_backward_and_no_class_universe():
    model = build_model("tiny")
    xs, _, xq = data()
    output = model(xs, np.zeros(9), xq, [], "regression", n_classes=0, hurdle=True)
    targets = torch.tensor([0., 0., 1., 10., 0., 3., 0.])
    loss = -output["distribution"].log_prob(targets).mean()
    assert torch.isfinite(loss)
    assert (output["mean"] >= 0).all()
    loss.backward()
    assert model.atom_head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_hurdle_atom_uses_only_available_reference_prior():
    model = build_model("tiny").eval()
    xs, _, xq = data()
    meta = np.zeros(14)
    prevalence = 1e-6
    meta[11] = np.log(prevalence / (1 - prevalence)) / 20
    with torch.no_grad():
        unavailable = model(xs, np.zeros(9), xq, [], "regression", n_classes=0,
                            finance_metadata=meta, hurdle=True)
        meta[10] = 1
        available = model(xs, np.zeros(9), xq, [], "regression", n_classes=0,
                          finance_metadata=meta, hurdle=True)
    torch.testing.assert_close(unavailable["atom_logit"], torch.zeros((7, 1)))
    torch.testing.assert_close(available["distribution"].log_positive.exp(), torch.full((7,), prevalence))
    assert available["distribution"].log_prob(torch.zeros(7)).abs().max() < 2e-6


def test_cache_precision_and_detached_training_guards():
    model = build_model("tiny")
    xs, ys, xq = data()
    with torch.no_grad():
        cache = model.prepare_context(xs, ys, [], "multiclass", 3)
    with pytest.raises(RuntimeError, match="support gradients"):
        model.predict_cached(cache, xq)
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="stale"):
            model.predict_cached(cache, xq)
