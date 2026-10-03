"""Stream a reproducible component audit of the implemented selection-v1 priors.

Defaults use native requested shapes, plus the configured later-stage envelope
mixture. Component weights are forced for this audit; observations retain their
configured law. --shape is a recorded diagnostic override, not a training law.
No model training, predictive benchmark or teacher-based acceptance occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tabular_foundation.challenger_prior import (MECHANISM_WEIGHTS, OBSERVATION_WEIGHTS,
                                                _seed, generate_challenger_episode)
from tabular_foundation.reference_prior import ReferenceShape
from tabular_foundation.static_prior import _rng


class StreamingMoments:
    """Uniform-world mean and sample SD without retaining observations."""
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = 0.
        self.scale = 1.
        self.minimum = self.maximum = None

    def add(self, value):
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite audit metric")
        larger = max(self.scale, abs(value))
        ratio = self.scale / larger
        self.mean *= ratio
        self.m2 *= ratio * ratio
        self.scale = larger
        self.count += 1
        scaled = value / self.scale
        delta = scaled - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (scaled - self.mean)
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def result(self):
        sd = float(np.sqrt(max(0., self.m2 / (self.count - 1)))) * self.scale if self.count > 1 else None
        return {"count": self.count, "mean": self.mean * self.scale if self.count else None,
                "sample_sd": sd if sd is None or np.isfinite(sd) else None,
                "sd_exceeds_float64": sd is not None and not np.isfinite(sd),
                "min": self.minimum, "max": self.maximum}


def summarize_episode(ep):
    """Extract observable validity and generator diagnostics, never alter targets."""
    for name in ("x_support", "x_query"):
        if np.isinf(getattr(ep, name)).any():
            raise FloatingPointError(f"Infinite {name}")
    for name in ("y_support", "y_query"):
        if not np.isfinite(getattr(ep, name)).all():
            raise FloatingPointError(f"Nonfinite {name}")
    info, metrics, labels = ep.metadata, {}, {}
    selection = info["selection"]
    metrics.update(support_rows=len(ep.y_support), query_rows=len(ep.y_query),
                   requested_features=info["reference_control"]["requested_features"],
                   accepted_features=ep.x_support.shape[1], categorical_fraction=float(ep.categorical.mean()),
                   missing_support_fraction=float(np.isnan(ep.x_support).mean()),
                   missing_query_fraction=float(np.isnan(ep.x_query).mean()),
                   raw_attempts=selection["raw_attempts"],
                   numerical_world_retries=len(selection["failed_worlds"]),
                   hierarchy_fallback=int(selection["hierarchy_fallback"]),
                   native_predictability_rejections=info["reference_control"]["predictability_rejections"],
                   native_graph_rejections=info["reference_control"]["graph_rejections"])
    labels.update(task=ep.task, requested_mechanism=selection["requested_mechanism"],
                  selected_mechanism=selection["selected_mechanism"],
                  requested_observation=selection["requested_observation"],
                  effective_observation=selection["effective_observation"],
                  calibration_source=selection["calibration_source"])
    mechanism = info.get("mechanism", {})
    for key in ("active_dimensions", "depth", "trees", "leaves", "target_leaf_occupancy", "entity_count",
                "support_unique_entities", "unseen_query_entity_fraction", "materialized_entities",
                "target_mean_occupancy", "bandwidth", "estimated_support_neighbors", "target_neighbors",
                "intrinsic_rank", "order", "interaction_share"):
        value = mechanism.get(key)
        if value is not None:
            metrics[key] = value
    for key in ("subfamily", "catalog_law", "context_conditioned", "common_student_t5_scale"):
        if mechanism.get(key) is not None:
            labels[key] = str(mechanism[key])
    response = info.get("response", {})
    if "no_signal" in response:
        metrics["no_signal_coordinate_fraction"] = float(np.mean(response["no_signal"]))
        metrics["all_scores_no_signal"] = int(all(response["no_signal"]))
    for key in ("noise_scale", "temperature", "dirichlet_concentration"):
        if key in response:
            metrics[key] = response[key]
    for key in ("noise_kind", "target_transform", "heteroscedastic"):
        if key in response:
            labels[key] = str(response[key])
    calibration = response.get("class_calibration", {})
    for key in ("maximum_absolute_error", "total_variation", "iterations"):
        if key in calibration:
            metrics["class_calibration_" + key] = calibration[key]
    if ep.task != "regression":
        metrics.update(legal_classes=ep.n_classes,
                       observed_support_classes=len(np.unique(ep.y_support)),
                       observed_query_classes=len(np.unique(ep.y_query)),
                       absent_support_classes=ep.n_classes - len(np.unique(ep.y_support)),
                       constant_support_labels=int(len(np.unique(ep.y_support)) == 1),
                       constant_query_labels=int(len(np.unique(ep.y_query)) == 1))
    # Keep output moments without squaring potentially huge legitimate tails.
    for split, outcomes in (("support", ep.y_support), ("query", ep.y_query)):
        largest = max(float(np.max(np.abs(outcomes))), 1.)
        metrics[f"{split}_target_mean"] = float(np.mean(outcomes / largest) * largest)
        metrics[f"{split}_target_sd"] = float(np.std(outcomes / largest) * largest)
        metrics[f"{split}_target_max_abs"] = largest if np.max(np.abs(outcomes)) >= 1 else float(np.max(np.abs(outcomes)))
    for value in metrics.values():
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite aggregate audit metric")
    return metrics, labels


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _run_world(job):
    family, stage, index, seed, options = job
    started = time.perf_counter()
    task = "classification" if _rng(seed, "audit_task").random() < .5 else "regression"
    shape = options["shape"]
    override = ReferenceShape(*shape, options["classes"] if task == "classification" else None) if shape else None
    envelope = options["envelopes"].get(str(stage))
    if _rng(seed, "audit_envelope").random() >= options["envelope_probability"]:
        envelope = None
    try:
        episode = generate_challenger_episode(seed, task=task, stage=stage, shape=override,
                    envelope=envelope, mechanism_weights={family: 1.},
                    observation_weights=options["observation_weights"],
                    complexity_conditioned_probability=options["complexity_conditioned_probability"],
                    namespace=options["namespace"])
        metrics, labels = summarize_episode(episode)
        metrics["seconds"] = time.perf_counter() - started
        labels["envelope_enabled"] = str(envelope is not None)
        return dict(family=family, stage=stage, index=index, seed=seed, accepted=True,
                    metrics=metrics, labels=labels)
    except Exception as exc:
        return dict(family=family, stage=stage, index=index, seed=seed, accepted=False,
                    error=f"{type(exc).__name__}: {exc}", audit=getattr(exc, "audit", None),
                    seconds=time.perf_counter() - started)


def run_audit(*, config, families, stages, worlds, seed=20261003, namespace="prior_audit",
              shape=None, classes=None, observation=None, workers=1, max_failures=10):
    if config.get("standard_prior") != "selection":
        raise ValueError("Audit config must declare standard_prior=selection")
    if not families or len(families) != len(set(families)) or set(families) - set(MECHANISM_WEIGHTS):
        raise ValueError("families must be distinct implemented mechanism names")
    if not stages or len(stages) != len(set(stages)) or set(stages) - {1, 2, 3}:
        raise ValueError("stages must be distinct members of 1,2,3")
    for name, value in (("worlds", worlds), ("workers", workers), ("max_failures", max_failures)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if shape is not None:
        ReferenceShape(*shape, classes)
    elif classes is not None:
        raise ValueError("classes requires an explicit diagnostic shape")
    settings = config.get("selection_options", {})
    options = {"shape": shape, "classes": classes, "namespace": namespace,
               "observation_weights": ({observation: 1.} if observation else settings.get("observation_weights", OBSERVATION_WEIGHTS)),
               "complexity_conditioned_probability": settings.get("complexity_conditioned_probability", .8),
               "envelopes": config.get("reference_envelope", {}),
               "envelope_probability": config.get("reference_envelope_probability", 0.)}
    if observation is not None and observation not in OBSERVATION_WEIGHTS:
        raise ValueError("Unknown observation mode")
    if not isinstance(options["envelopes"], dict) or not 0 <= options["envelope_probability"] <= 1:
        raise ValueError("Invalid reference envelope law")
    jobs = ((family, stage, index, _seed(seed, "audit_world", family, stage, index), options)
            for family in families for stage in stages for index in range(worlds))
    started = time.perf_counter()
    buckets, failures = {}, []
    pool = mp.get_context("spawn").Pool(workers, initializer=_worker_init) if workers > 1 else None
    _worker_init() if workers == 1 else None
    iterator = pool.imap(_run_world, jobs, chunksize=1) if pool else map(_run_world, jobs)
    attempted = accepted = 0
    try:
        for result in iterator:
            attempted += 1
            key = f"{result['family']}/stage{result['stage']}"
            bucket = buckets.setdefault(key, {"attempted": 0, "accepted": 0, "metrics": {}, "counts": {}})
            bucket["attempted"] += 1
            if not result["accepted"]:
                failures.append(result)
                if len(failures) >= max_failures:
                    break
                continue
            accepted += 1
            bucket["accepted"] += 1
            for name, value in result["metrics"].items():
                bucket["metrics"].setdefault(name, StreamingMoments()).add(value)
            for name, value in result["labels"].items():
                bucket["counts"].setdefault(name, Counter())[str(value)] += 1
    finally:
        if pool:
            pool.terminate()
            pool.join()
    for bucket in buckets.values():
        bucket["metrics"] = {name: value.result() for name, value in bucket["metrics"].items()}
        bucket["counts"] = {name: dict(value) for name, value in bucket["counts"].items()}
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [Path(__file__), *(ROOT / "src/tabular_foundation" / name for name in
                           ("challenger_prior.py", "selection_laws.py", "reference_prior.py", "static_prior.py")),
                           ROOT / "third_party/tabicl_reference/manifest.json"]}
    requested = len(families) * len(stages) * worlds
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if not failures and attempted == requested else "failed",
            "scope": "Generator validity/distribution audit, not learned prediction performance or SOTA evidence",
            "model_training_performed": False, "teacher_acceptance": False,
            "settings": {"families": families, "stages": stages, "worlds_per_family_stage": worlds,
                         "root_seed": seed, "seed_scheme": "SHA256/Philox keyed (root,audit_world,family,stage,index)",
                         "task_law": "classification/regression independent 50/50", "workers": workers,
                         "diagnostic_shape_override": shape, **options},
            "requested_worlds": requested, "attempted_worlds": attempted, "accepted_worlds": accepted,
            "failed_worlds": failures, "wall_seconds": time.perf_counter() - started,
            "aggregation": "Metrics are uniform-world means/SD, not cell-weighted estimates. Missing diagnostic fields are omitted, not zero. Native R may return fewer features and retains its own filters.",
            "source_sha256": hashes, "buckets": buckets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/selection_candidate_standard.json")
    parser.add_argument("--families", nargs="+", choices=list(MECHANISM_WEIGHTS), default=list(MECHANISM_WEIGHTS))
    parser.add_argument("--stages", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--worlds-per-family-stage", type=int, default=1000)
    parser.add_argument("--shape", nargs=3, type=int, metavar=("SUPPORT", "QUERY", "FEATURES"))
    parser.add_argument("--classes", type=int)
    parser.add_argument("--observation", choices=list(OBSERVATION_WEIGHTS))
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--namespace", default="prior_audit")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    report = run_audit(config=config, families=args.families, stages=args.stages,
                       worlds=args.worlds_per_family_stage, seed=args.seed, namespace=args.namespace,
                       shape=args.shape, classes=args.classes, observation=args.observation,
                       workers=args.workers, max_failures=args.max_failures)
    report["config_path"] = str(args.config)
    report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {report['accepted_worlds']}/{report['requested_worlds']} worlds accepted; {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
