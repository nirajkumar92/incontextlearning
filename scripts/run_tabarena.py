#!/usr/bin/env python3
"""Optional pinned-checkout TabArena entry point; see docs/tabarena.md."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def _git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tabarena-repo", required=True)
    parser.add_argument("--expected-revision", required=True, help="Exact 40-character reviewed TabArena commit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--full", action="store_true", help="Run all tasks in the pinned context instead of the development subset")
    parser.add_argument("--track", choices=["official", "outer"], default="official")
    parser.add_argument("--gpus", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--ray", action="store_true", help="Use upstream Ray-backed execution rather than in-process development execution")
    parser.add_argument("--baseline", action="append", default=[], help="Additional exact upstream registry name, at default configuration; repeatable")
    args = parser.parse_args()
    if args.full and args.datasets:
        parser.error("--full and --datasets are mutually exclusive")
    if min(args.cpus, args.context_limit, args.query_batch_size) < 1:
        parser.error("CPU, context and query-batch limits must be positive")
    repo = Path(args.tabarena_repo).expanduser().resolve(strict=True)
    revision = _git(repo, "rev-parse", "HEAD")
    if len(args.expected_revision) != 40 or revision != args.expected_revision:
        raise ValueError(f"TabArena checkout is {revision}; expected exact reviewed revision {args.expected_revision}")
    if _git(repo, "status", "--porcelain"):
        raise ValueError("Use a clean pinned TabArena checkout; commit intended upstream changes before benchmarking")
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = _digest(checkpoint)

    try:
        import tabarena
        from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
        from tabarena.contexts import TabArenaContext
        from tabarena.utils.config_utils import ConfigGenerator
        from tabular_foundation.tabarena_adapter import FrozenTabularFoundationModel
    except ImportError as error:
        raise ImportError("Install the separate environment in docs/tabarena.md before running this adapter") from error
    imported = Path(tabarena.__file__).resolve()
    if not imported.is_relative_to(repo):
        raise ValueError(f"Imported TabArena from {imported}, outside the pinned checkout {repo}")
    datasets = None if args.full else (args.datasets or ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"])
    config = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
              "context_limit": args.context_limit, "query_batch_size": args.query_batch_size,
              "random_state": args.seed, "bf16": not args.float32,
              "ag_args_fit": {"num_gpus": args.gpus, "num_cpus": args.cpus}}
    manifest_path = output / "run_manifest.json"
    identity = {"checkpoint_sha256": digest, "tabarena_revision": revision,
                "track": args.track, "datasets": datasets, "config": config,
                "additional_baselines": args.baseline, "ray": args.ray}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("identity") != identity:
            raise ValueError("Output contains a different experiment; choose a distinct output directory")
    versions = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")}
    manifest = {"identity": identity, "status": "running", "python": sys.version,
                "packages": dict(sorted(versions.items())), "tabarena_import": str(imported),
                "scope": "default frozen estimator within upstream bagging protocol" if args.track == "official" else
                         "unbagged outer diagnostic; outside official model protocol",
                "development_status": "adapter source-reviewed; upstream execution must be validated in this environment"}
    from tabular_foundation.runtime import atomic_json
    atomic_json(manifest_path, manifest)
    try:
        generator = ConfigGenerator(model_cls=FrozenTabularFoundationModel, manual_configs=[config], search_space={})
        experiments = TabArenaV0pt1ExperimentBundle(
            models=[(generator, 0)] + [(name, 0) for name in args.baseline],
            outer_experiments=args.track == "outer",
        ).build_experiments()
        context = TabArenaContext()
        run_kwargs = {"expname": str(output / "experiments"), "new_result_prefix": "[TFM] ",
                      "debug_mode": not args.ray}
        if datasets is not None:
            run_kwargs.update({"subset": "lite", "build_kwargs": {"dataset_names": datasets}})
        # Enumerate the planned grid before execution: upstream registration can
        # otherwise narrow later comparisons to the tasks that returned results.
        build_kwargs = run_kwargs.pop("build_kwargs", {})
        subset = run_kwargs.pop("subset", None)
        jobs = context.build_jobs(experiments, subset=subset, **build_kwargs)
        manifest["planned_jobs"] = len(jobs)
        atomic_json(manifest_path, manifest)
        if not jobs:
            raise ValueError("The requested dataset/configuration scope produced no benchmark jobs")
        results = context.run_jobs(jobs, **run_kwargs)
        manifest["returned_results"] = len(results)
        manifest["coverage_requires_per_task_failure_review"] = True
        atomic_json(manifest_path, manifest)
        if len(results) < len(jobs):
            raise RuntimeError("Some planned jobs produced no returned result; inspect failed tasks before aggregating scores")
        leaderboard = context.compare(output_dir=output / "comparison")
        website = context.leaderboard_to_website_format(leaderboard=leaderboard)
        website.to_csv(output / "leaderboard.csv", index=False)
        manifest["status"] = "completed_upstream_pipeline"
        atomic_json(manifest_path, manifest)
        print(website.to_string(index=False))
    except Exception as error:
        manifest.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
