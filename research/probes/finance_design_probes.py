"""Exact finite-distribution checks for rare-event design; no model benchmark."""
import argparse
import json
import math
from pathlib import Path


def precision(prevalence, recall, fpr):
    return prevalence * recall / (prevalence * recall + (1 - prevalence) * fpr)


def bce(y, p):
    return -math.log(p) if y else -math.log1p(-p)


def run():
    examples = []
    for pi in (1e-6, 1e-5, 1e-4, 1e-3):
        required_fpr = pi * .8 * (1 - .8) / (.8 * (1 - pi))
        examples.append({
            "prevalence": pi,
            "expected_positives_in_3m_rows": 3e6 * pi,
            "probability_zero_positives_in_32768_iid_context": math.exp(32768 * math.log1p(-pi)),
            "precision_at_recall_80pct_fpr_1e_minus3": precision(pi, .8, 1e-3),
            "fpr_required_for_precision_80pct_recall_80pct": required_fpr,
            "false_positives_per_million_negatives_at_required_fpr": 1e6 * required_fpr,
            "iid_negatives_needed_for_zero_fp_one_sided_95pct_upper_at_required_fpr":
                math.ceil(math.log(.05) / math.log1p(-required_fpr)),
        })

    # Feature-dependent selection: an aggregate class odds offset is insufficient.
    pi = 1e-6
    px1, px0 = [.7, .2, .1], [.001, .09, .909]
    a1, a0 = [1., .5, .2], [.001, .1, 1.]
    source = [pi * u / (pi * u + (1 - pi) * v) for u, v in zip(px1, px0)]
    selected = [pi * u * r / (pi * u * r + (1 - pi) * v * s)
                for u, v, r, s in zip(px1, px0, a1, a0)]

    def correct(p, keep0, keep1):
        odds = p / (1 - p) * keep0 / keep1
        return odds / (1 + odds)

    conditional = [correct(p, s, r) for p, s, r in zip(selected, a0, a1)]
    average0 = sum(p * a for p, a in zip(px0, a0))
    average1 = sum(p * a for p, a in zip(px1, a1))
    scalar = [correct(p, average0, average1) for p in selected]
    conditional_error = max(abs(a - b) for a, b in zip(source, conditional))
    scalar_error = max(abs(a - b) for a, b in zip(source, scalar))
    assert conditional_error < 1e-15 and scalar_error > .1

    # Enriched query labels with known proposal; oracle pi is used by loss only.
    predictions = [1e-3, 2e-5, 1e-7]
    target_risk = sum(pi * p1 * bce(1, p) + (1 - pi) * p0 * bce(0, p)
                      for p1, p0, p in zip(px1, px0, predictions))
    proposal_positive = .5
    weighted_risk = sum(proposal_positive * p1 * (pi / proposal_positive) * bce(1, p)
                        + (1 - proposal_positive) * p0 * ((1 - pi) / (1 - proposal_positive)) * bce(0, p)
                        for p1, p0, p in zip(px1, px0, predictions))
    assert abs(weighted_risk - target_risk) < 1e-18

    # If every observed fraud is caught, the two-sided exact binomial lower end.
    recall_limits = [{"independent_positive_events": n, "caught": n,
                      "observed_recall": 1., "two_sided_95pct_exact_lower_recall": .025**(1 / n)}
                     for n in (3, 10, 30, 100)]
    return {
        "status": "Exact probability and finite-distribution diagnostics; not fraud model results",
        "rarity_examples": examples,
        "selection_correction": {"population_posterior": source,
            "selected_posterior": selected, "feature_conditioned_correction": conditional,
            "scalar_class_correction": scalar, "feature_conditioned_max_error": conditional_error,
            "scalar_max_error": scalar_error},
        "enriched_query_importance_risk": {"population_bce": target_risk,
            "importance_weighted_proposal_bce": weighted_risk,
            "absolute_error": abs(weighted_risk - target_risk)},
        "recall_confidence_examples": recall_limits,
        "limitations": "Binomial examples assume independent outcomes. Bursts/campaign dependence can reduce effective evidence. No trained classifier, ROCm timing, or real financial data was used.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
