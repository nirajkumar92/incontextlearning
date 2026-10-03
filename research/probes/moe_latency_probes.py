"""Small algebra/contract probes for the proposed MoE; no model or GPU benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np


def swiglu(x, gate, value, down):
    a = x @ gate.T
    return ((a / (1 + np.exp(-a))) * (x @ value.T)) @ down.T


def route(x, shared, experts, router):
    logits = x @ router.T
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    index = logits.argmax(axis=-1)
    out = swiglu(x, *shared)
    for e, weights in enumerate(experts):
        selected = index == e
        out[selected] += 4 * p[selected, e, None] * swiglu(x[selected], *weights)
    return out, p, index


def run(seed=20261002):
    rng = np.random.default_rng(seed)
    d, h, rows = 16, 24, 37
    x = rng.normal(size=(rows, d))
    gate, value = [rng.normal(size=(h, d)) / np.sqrt(d) for _ in range(2)]
    down = rng.normal(size=(d, h)) / np.sqrt(h)
    perm = rng.permutation(h)
    a, b = perm[:h // 2], perm[h // 2:]
    shared = (gate[a], value[a], down[:, a])
    routed = (gate[b], value[b], down[:, b])
    experts = [tuple(w.copy() for w in routed) for _ in range(4)]
    dense = swiglu(x, gate, value, down)
    zero_router = np.zeros((4, d))
    converted = route(x, shared, experts, zero_router)[0]
    split_error = float(np.max(np.abs(converted - dense)))
    assert split_error < 1e-12

    extra = 3 * h // 2
    wide = (np.concatenate([gate, rng.normal(size=(extra, d)) / np.sqrt(d)]),
            np.concatenate([value, rng.normal(size=(extra, d)) / np.sqrt(d)]),
            np.concatenate([down, np.zeros((d, extra))], axis=1))
    wide_error = float(np.max(np.abs(swiglu(x, *wide) - dense)))
    assert wide_error < 1e-12

    router = rng.normal(size=(4, d)) * 0.1 / np.sqrt(d)
    full, p, index = route(x, shared, experts, router)
    chunked = np.concatenate([route(v, shared, experts, router)[0]
                              for v in np.array_split(x, 7)])
    chunk_error = float(np.max(np.abs(full - chunked)))
    assert chunk_error < 1e-12

    # Task-loss derivative with respect to logits, away from top-1 boundaries.
    logits = np.array([0.9, -0.4, 0.1, -0.8])
    projection = float(swiglu(x[:1], *routed)[0] @ rng.normal(size=d))

    def scalar(z):
        probabilities = np.exp(z - z.max())
        probabilities /= probabilities.sum()
        return 4 * probabilities[z.argmax()] * projection

    prob = np.exp(logits - logits.max()); prob /= prob.sum()
    selected = int(logits.argmax())
    analytic = 4 * prob[selected] * (np.eye(4)[selected] - prob) * projection
    eps = 1e-6
    numeric = np.array([(scalar(logits + eps * v) - scalar(logits - eps * v)) / (2 * eps)
                        for v in np.eye(4)])
    gradient_error = float(np.max(np.abs(analytic - numeric)))
    assert gradient_error < 1e-8 and np.linalg.norm(analytic) > 1e-6

    base, model_d, model_h, layers = 315021456, 1024, 2816, 8
    ffn = 3 * model_d * model_h
    expert = ffn // 2
    router_params = model_d * 4
    total = base + layers * (5 * expert + router_params - ffn)
    active_proxy = base + layers * router_params
    wide_total = base + layers * (3 * model_d * 7040 - ffn)
    assert (total, active_proxy, wide_total) == (418863248, 315054224, 418830480)

    contexts = []
    for n in (1024, 8192, 32768, 65536, 131072):
        contexts.append({
            "support_rows": n,
            "core_bf16_kv_cache_bytes": 2 * 24 * n * 1024 * 2,
            "core_bf16_kv_cache_gib": 2 * 24 * n * 1024 * 2 / 2**30,
            "support_attention_pairs_per_head_layer": n * n,
            "query_attention_pairs_per_head_layer_q4096": n * 4096,
        })
    return {
        "status": "Executed NumPy algebra/contract probes only; no trained model or measured latency",
        "seed": seed,
        "small_probe_dimensions": {"width": d, "hidden": h, "rows": rows},
        "zero_router_dense_split_max_absolute_error": split_error,
        "zero_down_dense_widening_max_absolute_error": wide_error,
        "tokenwise_routing_query_chunk_max_absolute_error": chunk_error,
        "scaled_top1_task_gradient_finite_difference_max_absolute_error": gradient_error,
        "nonzero_router_gate_gradient_norm": float(np.linalg.norm(analytic)),
        "parameters": {"dense": base, "moe_total": total,
                       "moe_nominal_active_proxy": active_proxy, "wide_dense": wide_total},
        "arithmetic_core_cache_examples": contexts,
        "limits": "Small FP64 modules test algebra, not a full transformer, BF16 kernels, trained accuracy, or latency. Active proxy counts all non-MoE parameters conventionally; it is not full model FLOPs.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
