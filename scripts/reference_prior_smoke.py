"""CPU smoke/recording CLI for the pinned reference and two eligible mixtures.

PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python scripts/reference_prior_smoke.py \
    --arm R --task classification --support 128 --query 64 --features 8 --classes 2
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from tabular_foundation.reference_prior import ARMS, ReferenceShape, generate_reference_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="R")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--support", type=int)
    parser.add_argument("--query", type=int)
    parser.add_argument("--features", type=int)
    parser.add_argument("--classes", type=int)
    parser.add_argument("--envelope-max-classes", type=int)
    parser.add_argument("--envelope-max-features", type=int)
    parser.add_argument("--max-reference-attempts", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-episodes", type=Path, help="Optional NPZ with separate arrays for each episode")
    args = parser.parse_args()
    values = (args.support, args.query, args.features)
    if any(v is not None for v in values) and not all(v is not None for v in values):
        parser.error("--support, --query and --features must be supplied together")
    if args.classes is not None and args.support is None:
        parser.error("--classes requires the explicit shape arguments")
    shape = ReferenceShape(*values, args.classes) if args.support is not None else None
    envelope = {key: value for key, value in {"max_classes": args.envelope_max_classes,
                 "max_features": args.envelope_max_features}.items() if value is not None} or None
    start = time.monotonic()
    batch = generate_reference_batch(args.seed, arm=args.arm, task=args.task, stage=args.stage,
                                     batch_size=args.batch_size, shape=shape, envelope=envelope,
                                     max_reference_attempts=args.max_reference_attempts)
    report = batch.audit
    report["elapsed_seconds"] = time.monotonic() - start
    report["evidence_scope"] = "CPU source equivalence/contracts and generation; no model training or GPU throughput claim"
    report["runtime"] = {"python": platform.python_version(), **{
        package: importlib.metadata.version(package)
        for package in ("numpy", "torch", "scipy", "scikit-learn", "psutil", "xgboost", "threadpoolctl")}}
    arrays = {}
    for index, ep in enumerate(batch.episodes):
        fields = {name: np.ascontiguousarray(getattr(ep, name))
                  for name in ("x_support", "x_query", "y_support", "y_query", "categorical")}
        report["events"][index]["array_sha256"] = {
            name: hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
            for name, value in fields.items()}
        arrays.update({f"episode_{index}_{name}": value for name, value in fields.items()})
        assert len(ep.x_support) == len(ep.y_support) and len(ep.x_query) == len(ep.y_query)
        assert np.isfinite(ep.y_support).all() and np.isfinite(ep.y_query).all()
        assert "y_query" not in ep.model_inputs() and "reference_control" not in ep.model_inputs()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if args.save_episodes:
        args.save_episodes.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_episodes, **arrays)
    print(text, end="")


if __name__ == "__main__":
    main()
