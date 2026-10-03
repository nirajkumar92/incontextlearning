"""Small learned head-resolution falsification; not a foundation-model benchmark.

Train identical two-feature MLP architectures (except output width) against
the same noiseless conditional labels in one legacy histogram bin. A fixed
support-only affine reference gives center=0, scale=1. Independent held-out
nuisance values test predictions beyond the exact training rows. All arms use
the same optimizer, horizon and data; no validation tuning or early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from tabular_foundation.codec import robust_location_scale
from tabular_foundation.distributions import ContinuousDistribution, regression_loss


def data(seed, n):
    rng = np.random.default_rng(seed)
    state = np.arange(n) % 2
    features = np.column_stack([2*state-1, rng.uniform(-1, 1, size=n)])
    targets = np.where(state, .070, .005)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float64), state


def output(params, head, center, scale):
    common = dict(regression_params=params, target_center=center, target_scale=scale)
    if head == "quantile999":
        common.update(loss_kind="pinball", quantile_levels=torch.arange(1, 1000, dtype=torch.float64)/1000,
                      mean=center+scale*params.double().mean(-1))
    else:
        bins = int(head.removeprefix("histogram"))
        distribution = ContinuousDistribution(params, center, scale, bins=bins)
        common.update(distribution=distribution, mean=distribution.mean, loss_kind="nll")
    return common


def fit_one(seed, head, steps):
    torch.manual_seed(seed)
    train_x, train_y, _ = data(1729, 64)
    test_x, test_y, states = data(2718, 512)
    # This is a declared fixed support calibration, not recomputed from queries.
    center, scale = robust_location_scale(np.array([-1/1.4826, 0., 1/1.4826]))
    width = 999 if head == "quantile999" else int(head.removeprefix("histogram")) + 4
    model = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, width))
    # Matching first-layer initialization across arms; small final outputs.
    nn.init.normal_(model[-1].weight, std=.02/math.sqrt(16))
    nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    start = time.perf_counter()
    for step in range(steps):
        lr = .00001 + (.02-.00001)*.5*(1+math.cos(math.pi*step/max(steps-1, 1)))
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = regression_loss(output(model(train_x), head, center, scale), train_y).mean()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter()-start
    with torch.no_grad():
        trained = output(model(train_x), head, center, scale)
        predicted = output(model(test_x), head, center, scale)
        point = predicted["mean"]
        rmse = (point-test_y).square().mean().sqrt().item()
        grouped = []
        for state in (0, 1):
            mask = torch.tensor(states == state)
            grouped.append(dict(state=state, target=float(test_y[mask][0]),
                                mean_prediction=point[mask].mean().item(),
                                prediction_std=point[mask].std(unbiased=False).item(),
                                minimum=point[mask].min().item(), maximum=point[mask].max().item()))
        raw = model(test_x)
    return {"seed": seed, "head": head, "parameters": sum(p.numel() for p in model.parameters()),
            "steps": steps, "elapsed_cpu_seconds": elapsed, "center": center, "scale": scale,
            "training_score": regression_loss(trained, train_y).mean().item(),
            "heldout_score": regression_loss(predicted, test_y).mean().item(),
            "heldout_rmse": rmse, "heldout_by_state": grouped,
            "heldout_prediction_separation": grouped[1]["mean_prediction"]-grouped[0]["mean_prediction"],
            "sample_heldout": [dict(features=test_x[i].tolist(), target=float(test_y[i]), prediction=float(point[i])) for i in range(8)],
            "finite_parameters_and_outputs": bool(torch.isfinite(raw).all())}


def kv2_training_check():
    from tabular_foundation.train import run
    from tabular_foundation.inference import load_checkpoint, Predictor
    from tabular_foundation.static_prior import generate_episode
    config = dict(model="tiny", device="cpu", cpu_threads=1, seed=1979, steps=3, global_batch=2,
                  finance_share=0, finance_adapter=True, optimizer="adamw", adam_lr=.0003,
                  bf16=False, checkpoint_every=1,
                  static_overrides=dict(task="regression", n_support=16, n_query=8, n_features=8),
                  model_options=dict(categorical_encoding="hash_bits", regression_head="quantile",
                                     regression_bins=1025, regression_quantiles=999,
                                     length_scaling="logarithmic", trunk_query_kv_heads=2,
                                     hurdle_target_scale="positive_support"))
    with tempfile.TemporaryDirectory(prefix="pfn-kv2-learned-probe-") as folder:
        serial = run(config, Path(folder)/"serial")
        parallel = run(dict(config, producer_workers=1, prefetch_tasks=2), Path(folder)/"prefetched")
        max_weight_difference = max(float((a-parallel.state_dict()[name]).abs().max()) for name, a in serial.state_dict().items())
        restored, saved = load_checkpoint(Path(folder)/"prefetched"/"last.pt")
        episode = generate_episode(9797, task="regression", n_support=16, n_query=8, n_features=8)
        with torch.no_grad():
            expected = parallel(**episode.model_inputs())["mean"].numpy()
        forwarded = {key: value for key, value in episode.model_inputs().items()
                     if key not in {"x_support", "y_support", "x_query", "categorical", "task", "n_classes"}}
        predictor = Predictor(restored, query_batch_size=3, bf16=False).fit_context(
            episode.x_support, episode.y_support, episode.categorical, episode.task, episode.n_classes, **forwarded)
        predicted = predictor.predict(episode.x_query)
        assert max_weight_difference == 0
        np.testing.assert_allclose(predicted, expected, rtol=2e-5, atol=2e-6)
        assert restored.config.trunk_query_kv_heads == 2 and np.isfinite(predicted).all()
        # Verify that the separate hurdle branch also trains with the selected
        # two-head query path after checkpoint restore.
        restored.train()
        x = np.arange(8).reshape(4, 2)/10
        result = restored(x, np.zeros(4), x, [], "regression", hurdle=True)
        hurdle_loss = regression_loss(result, torch.tensor([0., .5, 0., 2.], dtype=torch.float64)).mean()
        hurdle_loss.backward()
        assert restored.hurdle_regression_head.weight.grad.abs().sum() > 0
        assert restored.atom_head.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in restored.parameters() if p.grad is not None)
        return {"trunk_query_kv_heads": restored.config.trunk_query_kv_heads, "updates_per_run": saved["step"],
                "serial_vs_prefetched_max_weight_difference": max_weight_difference,
                "saved_reload_chunked_prediction_max_difference": float(np.max(np.abs(predicted-expected))),
                "heldout_targets": episode.y_query.tolist(), "heldout_predictions": predicted.tolist(),
                "hurdle_loss_and_gradients_finite": bool(torch.isfinite(hurdle_loss)),
                "scope": "Real tiny-model training and checkpoint wiring, not a quality comparison. Temporary checkpoints removed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    paths = ["src/tabular_foundation/model.py", "src/tabular_foundation/codec.py", "src/tabular_foundation/distributions.py",
             "src/tabular_foundation/train.py", "src/tabular_foundation/inference.py", "src/tabular_foundation/producer.py",
             "research/probes/learned_regression_head_probe.py"]
    runs = [fit_one(seed, head, args.steps) for seed in (41, 42, 43)
            for head in ("histogram128", "histogram1025", "quantile999")]
    result = {"status": "Learned CPU head falsification and tiny-model wiring checks; not foundation-model evaluation",
              "environment": dict(torch=torch.__version__, numpy=np.__version__, device="cpu", threads=1),
              "design": dict(training_rows=64, heldout_rows=512, hidden_width=16, steps=args.steps,
                             optimizer="Adam", learning_rate="cosine 0.02 to0.00001", seeds=[41, 42, 43],
                             signal="binary state in feature0", nuisance="independent uniform[-1,1] feature1",
                             conditional_targets=[.005, .070], target_reference="fixed support[-1/1.4826,0,1/1.4826]",
                             scope="All scores optimize the same labels but differ in units and gradient scale; no equal-compute or general-prior conclusion."),
              "source_sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
              "runs": runs, "query_kv2_training": kv2_training_check()}
    encoded = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
