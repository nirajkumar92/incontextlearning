"""Tiny CPU initialization diagnostic of rare-event gradient allocation.

No optimizer step, trained foundation model, benchmark score or GPU timing is
measured. Analytic scalar derivatives at p=pi are separate from derivatives and
parameter-gradient norms of an actual randomly initialized model. Run from an
environment with this package installed, or set PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from tabular_foundation.finance_prior import generate_finance_world
from tabular_foundation.model import build_model
from tabular_foundation.static_prior import generate_episode
from tabular_foundation.train import episode_loss


MODEL_OPTIONS = dict(categorical_encoding="hash_bits", regression_head="quantile",
                     regression_bins=1025, regression_quantiles=999,
                     length_scaling="logarithmic", trunk_query_kv_heads=2,
                     hurdle_target_scale="positive_support")
SEEDS = dict(model_initialization=731, finance_world=743, finance_macroepisode=757,
             authored_P1_episode=761)
FINANCE_OVERRIDES = dict(
    history_size=1_000_000, future_size=333_333, width=16, prevalence=1e-4,
    mechanism_family="risk_partition", feature_view="hide_h",
    positive_h_probability=.01, negative_h_probability=.01,
    risk_campaigns=False, arrival_burst=False, risk_features=4, risk_strength=3.,
    reservoir_cap=32, positive_cap=16, local_cap=32, max_centers=2,
    candidate_cap=64, calibration_rows=128)
STANDARD_ARGUMENTS = dict(family="P1", n_support=64, n_query=128,
                          n_features=16, stage=1)


def digest(values):
    """Hash array contents, types and shapes; names identify the ordered fields."""
    result = hashlib.sha256()
    for name, value in values:
        result.update(name.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            result.update(str(value.dtype).encode() + str(value.shape).encode())
            result.update(np.ascontiguousarray(value).tobytes())
        else:
            result.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        result.update(b"\0")
    return result.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=int(values.size), mean=float(values.mean()),
                minimum=float(values.min()), maximum=float(values.max()),
                mean_absolute=float(np.abs(values).mean()),
                rms=float(np.sqrt(np.square(values).mean())))


def gradient_norm(model, prefix):
    squares = [parameter.grad.double().square().sum()
               for name, parameter in model.named_parameters()
               if name.startswith(prefix) and parameter.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.


def score(model, name, episodes, original_query_count, multiplier):
    """Use trainer route sums and original macro denominator, before clipping."""
    model.zero_grad(set_to_none=True)
    total = 0.
    retained = []
    for episode in episodes:
        output = model(**episode.model_inputs())
        loss, _ = episode_loss(output, episode, "cpu")
        total = total + loss * multiplier / original_query_count
        if "logits" in output:
            output["logits"].retain_grad()
            retained.append((episode, output["logits"]))
    total.backward()
    norms = {prefix.removesuffix("."): gradient_norm(model, prefix)
             for prefix in ("trunk.", "shared_head.", "class_head.", "regression_head.")}
    assert torch.isfinite(total) and all(math.isfinite(value) for value in norms.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    record = dict(name=name, macro_loss=float(total.detach()), world_multiplier=multiplier,
                  original_query_denominator=original_query_count, route_count=len(episodes),
                  gradient_norm_before_clipping_or_optimizer=norms,
                  forward_input_sha256=digest((f"route_{i}/{key}", value)
                      for i, episode in enumerate(episodes)
                      for key, value in sorted(episode.model_inputs().items())),
                  loss_input_sha256=digest((f"route_{i}/{key}", getattr(episode, key))
                      for i, episode in enumerate(episodes)
                      for key in ("y_query", "query_weights")))
    if retained:
        probabilities, targets, weights, derivatives = [], [], [], []
        for episode, logits in retained:
            assert logits.shape[1] == 2
            probabilities.append(logits.detach().softmax(-1)[:, 1].double().numpy())
            targets.append(episode.y_query)
            weights.append(episode.query_weights)
            # dL/dl1 at fixed l0 is the binary log-odds derivative. Undo only
            # macro averaging and the world multiplier, retaining proposal w_y.
            derivatives.append(logits.grad[:, 1].double().numpy()
                               * original_query_count / multiplier)
        probabilities, targets, weights, derivatives = map(
            np.concatenate, (probabilities, targets, weights, derivatives))
        expected = weights * (probabilities - targets)
        np.testing.assert_allclose(derivatives, expected, rtol=3e-5, atol=2e-7)
        record["actual_binary_model"] = dict(
            probability=describe(probabilities), proposal_positive_queries=int(targets.sum()),
            per_query_derivative_including_importance_before_macro_mean=describe(derivatives),
            derivative_identity_max_absolute_error=float(np.max(np.abs(derivatives - expected))),
            by_outcome={str(y): dict(probability=describe(probabilities[targets == y]),
                       importance_weight=describe(weights[targets == y]),
                       derivative=describe(derivatives[targets == y]))
                        for y in (0, 1) if np.any(targets == y)})
    return record


def analytic_derivatives(pi=1e-4):
    entropy = -pi * math.log(pi) - (1-pi) * math.log1p(-pi)
    rows = []
    for p in (pi, .001, .01, .5):
        g0, g1 = 2*(1-pi)*p, 2*pi*(p-1)
        rows.append(dict(p=p, negative_derivative=g0, positive_derivative=g1,
                         balanced_proposal_mean=(g0+g1)/2,
                         balanced_proposal_mean_absolute=(abs(g0)+abs(g1))/2,
                         entropy_scaled_negative=g0/entropy, entropy_scaled_positive=g1/entropy))
    return dict(scope="Ideal binary scalar log-odds derivatives, not model gradient norms",
                population_prevalence=pi, class_proposal=[.5, .5],
                importance_weights=[2*(1-pi), 2*pi],
                unit_formula="d ell / d z = w_y * (p-y)",
                entropy=entropy, hypothetical_reference_entropy_multiplier=1/entropy,
                rows=rows, calibrated_finance_to_balanced_binary_magnitude_ratio=2*pi*(1-pi)/.5,
                joint_80_20_to_all_binary_standard_ratio=.2*(2*pi*(1-pi))/(.8*.5),
                joint_80_20_to_standard_binary_half_share_ratio=.2*(2*pi*(1-pi))/(.4*.5),
                mixture_caveat="These compare per-example scalar magnitudes only, not mean parameter gradients or optimizer updates; standard multiclass/regression scales differ.")


def run():
    torch.set_num_threads(1)
    torch.manual_seed(SEEDS["model_initialization"])
    model = build_model("tiny", finance=True, model_options=MODEL_OPTIONS).cpu()
    model.train()
    initial_state = digest(model.state_dict().items())
    records, pairs = [], []
    for complete in (True, False):
        group = []
        for normalization in ("unit", "reference_entropy"):
            overrides = dict(FINANCE_OVERRIDES, complete_adjudication=complete,
                             loss_normalization=normalization)
            world = generate_finance_world(SEEDS["finance_world"], stage=1,
                                           task="binary", overrides=overrides)
            macro = world.sample_macroepisode(SEEDS["finance_macroepisode"], query_count=128)
            mode_name = "unit" if normalization == "unit" else "entropy"
            name = f"finance_{'reference' if complete else 'no_reference'}_{mode_name}"
            result = score(model, name, macro.routes, macro.original_query_count,
                           macro.reference_normalization_weight)
            result["finance"] = dict(
                overrides=overrides, stage=1, task="binary", world_id=world.world_id,
                population_id=world.population_id, historical_rows=world.n_history,
                future_rows=world.n_future, future_events=world.pool_count("future", 1),
                realized_future_prevalence=world.pool_count("future", 1)/world.n_future,
                eligible_rows=world.n_eligible, eligible_events=world.m_eligible,
                reference_kind=world.reference_kind, reference_rows=world.n_reference,
                reference_prevalence=world.reference_prevalence,
                finite_query_id_sha256=digest([("ids", macro.audit["finite_query_ids"])]),
                candidate_count=macro.routes[0].metadata["candidate_count"],
                candidate_mode=macro.routes[0].metadata["candidate_mode"])
            records.append(result)
            group.append(result)
        unit, entropy = group
        for key in ("forward_input_sha256", "loss_input_sha256"):
            assert unit[key] == entropy[key]
        for key in ("population_id", "finite_query_id_sha256"):
            assert unit["finance"][key] == entropy["finance"][key]
        scale = entropy["world_multiplier"]
        norm_scaling_errors = {}
        for branch, norm in unit["gradient_norm_before_clipping_or_optimizer"].items():
            actual = entropy["gradient_norm_before_clipping_or_optimizer"][branch]
            expected = norm * scale
            norm_scaling_errors[branch] = abs(actual - expected)/expected if expected else 0.
            # Near-cancelling positive/negative FP32 gradients can magnify
            # rounding error in a norm ratio; this is not an exact-arithmetic test.
            np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-10)
        np.testing.assert_allclose(entropy["macro_loss"], unit["macro_loss"]*scale, rtol=1e-6)
        pairs.append(dict(complete_adjudication=complete, same_forward_and_loss_inputs=True,
                          same_population_and_query_ids=True, entropy_multiplier=scale,
                          loss_and_gradient_scaling_verified=True,
                          gradient_norm_scaling_relative_errors=norm_scaling_errors,
                          gradient_norm_scaling_rtol=5e-3, gradient_norm_scaling_atol=1e-10))
    for task in ("binary", "regression"):
        episode = generate_episode(SEEDS["authored_P1_episode"], task=task, **STANDARD_ARGUMENTS)
        records.append(score(model, f"authored_P1_{task}", [episode], len(episode.y_query), 1.))
    assert digest(model.state_dict().items()) == initial_state
    root = Path(__file__).resolve().parents[2]
    sources = ["research/probes/finance_gradient_probe.py"] + [
        f"src/tabular_foundation/{name}.py" for name in
        ("model", "codec", "distributions", "train", "runtime", "schema", "retrieval", "finance_prior", "static_prior")]
    return dict(
        scope="Tiny CPU random-initialization diagnostics; no optimizer steps, trained TFM accuracy, generalization comparison or MI355X measurements",
        interpretation=[
            "Unit rare-event gradients with an available reference can be much smaller than standard-task gradients at these particular weights and worlds.",
            "Without a reference, large initial base-rate gradients do not establish a strong later conditional-risk learning signal.",
            "Entropy scaling changes world allocation, retains query importance weights, and is one when no reference exists.",
            "Complete/incomplete adjudication changes legal evidence and is not a paired deletion of only the logit reference.",
            "Equal H probabilities and hidden H are diagnostic overrides; candidate main worlds need not use these H probabilities.",
            "The tiny P1 tasks are illustrative scale comparators, not the selected R/P1 training mixture or a measured shared-training result.",
            "Gradient norms do not imply optimizer update ratios: AdamW/Muon operate on the combined shared gradient.",
            "The 64-candidate diagnostic pool and reduced support caps do not test full-history retrieval throughput."],
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=np.__version__, torch=torch.__version__, device="cpu", threads=1,
                         parameter_dtype="float32", autocast=False, optimizer_steps=0,
                         class_slots="default first K output slots; no trainer slot augmentation",
                         conditional_labels=False, clipping=False),
        seeds=SEEDS, model=dict(size="tiny", finance=True, options=MODEL_OPTIONS,
                              resolved_config=asdict(model.config), parameter_count=model.parameter_count(),
                              initial_state_sha256=initial_state, weights_unchanged=True),
        standard_arguments=STANDARD_ARGUMENTS, analytic=analytic_derivatives(),
        paired_normalization_checks=pairs, records=records,
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
