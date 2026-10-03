"""Falsification probes for the authored research baseline, not accuracy tests.

Run from the repository with PYTHONPATH=src. CPU numerical observations must be
repeated on the intended GPU stack before making a hardware-specific claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from tabular_foundation.codec import TableCodec, _canonical, _hash_code
from tabular_foundation.distributions import ContinuousDistribution
from tabular_foundation.model import build_model


def categorical_aliasing():
    values = np.array([f"category_{i}" for i in range(50_000)], dtype=object).reshape(-1, 1)
    codes = np.array([_hash_code(0, 0, _canonical(v[0])) for v in values])
    order = np.argsort(codes)
    closest = np.argmin(np.diff(codes[order]))
    ids = order[closest:closest + 2]
    codec = TableCodec().fit(values, [0])
    encoded = codec.transform(values[ids])
    torch.manual_seed(1729)
    model = build_model("tiny", finance=False).eval()
    with torch.inference_mode():
        labels = torch.zeros(2, model.config.cell_width)
        fp32 = model.encoder(encoded, labels)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16 = model.encoder(encoded, labels)
    assert codes[ids[0]] != codes[ids[1]]
    return {
        "support_categories": len(values), "encoding_seed": 0, "model_seed": 1729,
        "categories": values[ids, 0].tolist(), "codes": codes[ids].tolist(),
        "hash_gap": float(codes[ids[1]] - codes[ids[0]]),
        "raw_hash_collision": False,
        "fp32_embedding_max_difference": float((fp32[0] - fp32[1]).abs().max()),
        "cpu_bf16_embedding_max_difference": float((bf16[0] - bf16[1]).abs().max()),
        "cpu_bf16_embeddings_identical": torch.equal(bf16[0], bf16[1]),
        "scope": "One untrained cell encoder and recoding seed; not universal trained-model or GPU behavior.",
    }


def regression_resolution():
    params = torch.zeros(2, 132, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([.005, .07], dtype=torch.float64)
    losses = -ContinuousDistribution(params).log_prob(targets)
    losses.sum().backward()
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(params.grad[0], params.grad[1])
    concentrated = torch.full((1, 132), -100., dtype=torch.float64)
    concentrated[:, 65] = 100.  # z=0 belongs to [0, 10/128).
    concentrated[:, 130:] = 0.
    mean = float(ContinuousDistribution(concentrated, center=7., scale=1.).mean)
    assert abs(mean - 7.0390625) < 1e-12
    return {
        "targets": targets.tolist(), "nll": losses.detach().tolist(),
        "gradient_max_difference": float((params.grad[0] - params.grad[1]).abs().max()),
        "bin_width": 10 / 128, "constant_target_original_units": 7.,
        "limiting_histogram_mean": mean, "constant_target_bias": mean - 7.,
        "scope": "Population NLL-optimal bin mass; not an executed training run. Tails cannot recover within-bin labels.",
    }


def regression_affine_precision():
    targets = torch.tensor([100000000., 100000001.], dtype=torch.float64)
    params = torch.zeros(2, 132, requires_grad=True)
    dist = ContinuousDistribution(params, center=100000000., scale=.5)
    normalized = dist._z(targets)
    (-dist.log_prob(targets)).sum().backward()
    old_order = (targets.float() - torch.tensor(100000000., dtype=torch.float32)) / .5
    assert torch.equal(normalized, torch.tensor([0., 2.]))
    assert float((params.grad[0] - params.grad[1]).abs().max()) == 1.
    return {"original_targets": targets.tolist(), "old_cast_before_subtract": old_order.tolist(),
            "repaired_standardized_targets": normalized.tolist(),
            "repaired_gradient_max_difference": float((params.grad[0] - params.grad[1]).abs().max()),
            "status": "Repaired numerical defect; independent from unchanged histogram resolution."}


def finance_shortcut():
    pi, tpr = 1e-4, .8
    results = []
    for fpr in (1e-6, 1e-4, 1e-3, .01):
        precision = pi * tpr / (pi * tpr + (1 - pi) * fpr)
        # Two tied score levels; standard threshold-step, non-interpolated AP.
        ap = tpr * precision + (1 - tpr) * pi
        results.append({"P_H1_given_legit": fpr, "roc_auc": (1 + tpr - fpr) / 2,
                        "precision_at_recall_0_8": precision, "threshold_step_AP": ap})
    return {"prevalence": pi, "P_H1_given_fraud": tpr, "analytic_population_results": results,
            "scope": "H is the observable hard-negative indicator, not reference availability. Conditional on these rates; finite populations fluctuate."}


def finance_gradient_scaling():
    pi = 1e-4
    entropy = -pi * math.log(pi) - (1 - pi) * math.log1p(-pi)
    multiplier = 1 / max(entropy, 1e-4)
    scenarios = []
    for probability in (pi, .5):
        # Balanced query proposal; derivative of BCE with respect to logit.
        negative = 2 * (1 - pi) * probability
        positive = 2 * pi * (probability - 1)
        scenarios.append({"predicted_probability": probability,
                          "raw_negative_gradient": negative, "raw_positive_gradient": positive,
                          "normalized_negative_gradient": multiplier * negative,
                          "normalized_positive_gradient": multiplier * positive})
    return {"pi_reference": pi, "entropy_multiplier": multiplier, "balanced_query_proposal": scenarios,
            "scope": "Scalar-logit calculation only. Proper importance weighting remains unbiased; relative task allocation and clipping still require measurement."}


def attention_dilution():
    # Conditional on a fixed relevant-minus-irrelevant logit gap. This is not
    # a proof that training cannot learn a larger gap or a distributed statistic.
    values = []
    gap = math.log(4095)  # one anchor has half the mass at 4096 keys
    for n in (4096, 8192, 32768, 131072):
        clipped = max(-math.log(4), min(math.log(n / 1024), math.log(4)))
        values.append({"keys": n, "clipped_log_length": clipped,
                       "anchor_weight_fixed_gap": 1 / (1 + (n - 1) * math.exp(-gap)),
                       "gap_needed_for_90pct_mass": math.log(9 * (n - 1))})
    return {"fixed_logit_gap": gap, "results": values,
            "scope": "Analytic single-anchor stress case, no trained attention measurements."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    audited = ["src/tabular_foundation/codec.py", "src/tabular_foundation/model.py",
               "src/tabular_foundation/distributions.py", "src/tabular_foundation/finance_prior.py",
               "research/probes/independent_audit_probes.py"]
    result = {"status": "CPU falsification probes and analytic calculations, not competitive evaluation",
              "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": "cpu"},
              "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in audited},
              "categorical_aliasing": categorical_aliasing(), "regression_resolution": regression_resolution(),
              "regression_affine_precision": regression_affine_precision(),
              "finance_shortcut": finance_shortcut(), "finance_gradient_scaling": finance_gradient_scaling(),
              "attention_dilution": attention_dilution()}
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
