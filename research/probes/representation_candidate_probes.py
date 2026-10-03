"""CPU counterexamples for candidate representations; no benchmark-performance claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from tabular_foundation.codec import TableCodec
from tabular_foundation.distributions import ContinuousDistribution, regression_loss
from tabular_foundation.model import MAB, build_model


def identity_probe():
    support = np.array([f"category_{i}" for i in range(50000)], dtype=object)[:, None]
    pair = support[[28838, 39200]]
    results = []
    for encoding in ("scalar_fourier", "hash_bits"):
        for encoding_seed in (0, 17, 91):
            codec = TableCodec(encoding_seed, categorical_encoding=encoding).fit(support, [0])
            encoded = codec.transform(pair)
            for model_seed in (1729, 83):
                torch.manual_seed(model_seed)
                model = build_model("tiny", finance=False, model_options={"categorical_encoding": encoding}).eval()
                with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                    vectors = model.encoder(encoded, torch.zeros(2, model.config.cell_width))
                results.append(dict(encoding=encoding, encoding_seed=encoding_seed, model_seed=model_seed,
                                    bf16_identical=torch.equal(vectors[0], vectors[1]),
                                    embedding_max_difference=float((vectors[0] - vectors[1]).abs().max())))
    return {"support_categories": 50000, "pair": pair[:, 0].tolist(), "cases": results,
            "scope": "Observed projected vectors at initialization on CPU. Hash bits do not guarantee injectivity after learned projection."}


def regression_probe():
    rows = []
    for bins in (128, 1025):
        params = torch.zeros(2, bins + 4, dtype=torch.float64, requires_grad=True)
        losses = -ContinuousDistribution(params, bins=bins).log_prob(torch.tensor([.005, .07], dtype=torch.float64))
        losses.sum().backward()
        concentrated = torch.full((bins + 4,), -1000., dtype=torch.float64)
        concentrated[1 + bins//2] = 0.
        concentrated[-2:] = 0.
        rows.append({"bins": bins, "bin_width": 10/bins,
                     "finite_region_worst_case_mean_rounding_bound_in_support_scale": 5/bins,
                     "audited_pair_gradient_max_difference": float((params.grad[0] - params.grad[1]).abs().max()),
                     "constant_zero_limiting_mean": float(ContinuousDistribution(concentrated, bins=bins).mean)})
    params = torch.full((2, 999), .03, requires_grad=True)
    out = dict(regression_params=params, target_center=0., target_scale=1., loss_kind="pinball",
               quantile_levels=torch.arange(1, 1000, dtype=torch.float64)/1000)
    regression_loss(out, torch.tensor([.005, .07], dtype=torch.float64)).sum().backward()
    return {"histograms": rows,
            "quantile_pair_gradient_max_difference": float((params.grad[0] - params.grad[1]).abs().max()),
            "rare_event_counterexample": {"positive_probability": .0001, "positive_value": 1e6,
                                           "true_mean": 100., "all_999_population_quantiles": 0., "quantile_average": 0.},
            "scope": "Exact head/oracle calculations. Finer bins reduce but cannot remove quantization; finite quantiles cannot identify omitted tails."}


def length_probe():
    legacy = MAB(16, 2, 32, 1.)
    logarithmic = MAB(16, 2, 32, 1., length_scaling="logarithmic")
    gap = math.log(4095)
    rows = []
    for n in (1024, 4096, 8192, 32768, 131072):
        ratio = math.log(n)/math.log(4096)
        rows.append({"keys": n, "legacy_initial_temperature": float(legacy.attention_temperature(n)[0].detach()),
                     "logarithmic_initial_temperature": float(logarithmic.attention_temperature(n)[0].detach()),
                     "fixed_gap_anchor_mass": 1/(1+(n-1)*math.exp(-gap)),
                     "matched_4096_logarithmic_anchor_mass": 1/(1+(n-1)*math.exp(-gap*ratio))})
    return {"rows": rows, "scope": "Analytic attention mass for fixed representations, calibrated to equal half mass at4096. Not trained length-generalization evidence."}


def query_cache_probe():
    torch.manual_seed(94)
    full = build_model("tiny").eval()
    compact = build_model("tiny", model_options={"trunk_query_kv_heads": 1}).eval()
    compact.load_state_dict(full.state_dict())
    rng = np.random.default_rng(94)
    support = rng.normal(size=(64, 5))
    query = rng.normal(size=(17, 5))
    labels = np.arange(64) % 2
    with torch.inference_mode():
        cf = full.prepare_context(support, labels, [], "binary")
        cc = compact.prepare_context(support, labels, [], "binary")
        prediction = compact.predict_cached(cc, query)["logits"]
        chunks = torch.cat([compact.predict_cached(cc, q)["logits"] for q in (query[:5], query[5:])])
        largest_prefix_difference = max(float((a[:1] - b).abs().max()) for p, q in zip(cf.trunk, cc.trunk) for a, b in zip(p, q))
    full_bytes = sum(t.numel()*t.element_size() for p in cf.trunk for t in p)
    compact_bytes = sum(t.untyped_storage().nbytes() for p in cc.trunk for t in p)
    return {"tiny_trunk_heads": 4, "retained_query_heads": 1,
            "tiny_full_trunk_cache_bytes": full_bytes, "tiny_compact_trunk_storage_bytes": compact_bytes,
            "tiny_storage_ratio": compact_bytes/full_bytes,
            "largest_prefix_difference_across_all_support_layers": largest_prefix_difference,
            "largest_full_vs_chunked_query_difference": float((prediction-chunks).abs().max()),
            "base8192_BF16_full_trunk_cache_MiB": 2*24*8192*16*64*2/(1024**2),
            "base8192_BF16_one_head_trunk_cache_MiB": 2*24*8192*1*64*2/(1024**2),
            "scope": "Measured CPU inference storage; base BF16 sizes are analytic. Excludes frontend and training activations. No latency or GPU-kernel claim."}


def hurdle_scale_probe():
    support = np.arange(16).reshape(8, 2)
    targets = np.array([0., 0., 0., 0., 100., 200., 300., 400.])
    results = []
    for policy in ("reference", "positive_support"):
        model = build_model("tiny", model_options={"regression_bins": 1025, "hurdle_target_scale": policy}).eval()
        with torch.inference_mode():
            cache = model.prepare_context(support, targets, [], "regression", codec_indices=[0, 1, 2, 3], hurdle=True)
        z = (targets[4:]-cache.target_center)/cache.target_scale
        results.append({"policy": policy, "center": cache.target_center, "scale": cache.target_scale,
                        "standardized_positive_support": z.tolist(),
                        "positive_support_in_finite_bins": int(((z >= -5) & (z < 5)).sum())})
    return {"reference_reservoir_targets": targets[:4].tolist(), "positive_support_targets": targets[4:].tolist(),
            "results": results,
            "scope": "Support-only reparameterization changes bin coverage, not population weighting or tail-family expressiveness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    paths = ["src/tabular_foundation/codec.py", "src/tabular_foundation/model.py", "src/tabular_foundation/distributions.py",
             "research/probes/representation_candidate_probes.py", "tests/test_candidate_representation.py"]
    variants = {"legacy": {}, "histogram_candidate": dict(categorical_encoding="hash_bits", regression_bins=1025,
                                                        length_scaling="logarithmic", trunk_query_kv_heads=1,
                                                        hurdle_target_scale="positive_support"),
                "quantile_candidate": dict(categorical_encoding="hash_bits", regression_bins=1025,
                                           length_scaling="logarithmic", regression_head="quantile", trunk_query_kv_heads=1,
                                           hurdle_target_scale="positive_support")}
    counts = {name: build_model("base", device="meta", model_options=opts).parameter_count() for name, opts in variants.items()}
    result = {"status": "CPU representation and score probes; untrained, no competitive evaluation",
              "environment": dict(torch=torch.__version__, numpy=np.__version__, device="cpu"),
              "source_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
              "finance_enabled_base_parameters": counts,
              "categorical_identity": identity_probe(), "regression_resolution": regression_probe(),
              "length_scaling": length_probe(), "query_only_kv": query_cache_probe(), "hurdle_scaling": hurdle_scale_probe()}
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
