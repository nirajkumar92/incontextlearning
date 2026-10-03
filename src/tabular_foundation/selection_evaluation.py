"""Evaluate declared standard-table panels and compare paired prior experiments.

Dataset paths and source lineages must be supplied by the researcher. This
runner does not discover benchmark data, choose a winner, or treat a declared
lineage as proof of independent provenance. Query outcomes are scoring-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .codec import robust_location_scale
from .data import load_table, validate_pair
from .inference import Predictor, load_checkpoint
from .metrics import binary_metrics, multiclass_metrics, regression_metrics
from .runtime import atomic_json

TASKS = ("binary", "multiclass", "regression")
CONFIRMATION_BOOTSTRAPS = 10000
CONFIRMATION_LINEAGES = 10
CACHE_POLICY = "sequential_views"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_panel(panel):
    if panel.get("role") not in ("development", "confirmation", "final"):
        raise ValueError("Panel role must be development, confirmation or final")
    rows = panel.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Panel needs a nonempty datasets list")
    seen, provenance = set(), {}
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in
                   ("dataset", "lineage", "fold", "support", "query")):
            raise ValueError("Each split needs dataset, lineage, fold, support and query strings")
        if row.get("task") not in TASKS:
            raise ValueError("Unknown panel task")
        key = (row["dataset"], row["fold"])
        if key in seen:
            raise ValueError("Duplicate dataset/fold")
        seen.add(key)
        identity = (row["lineage"], row["task"], row.get("classes", 0))
        if row["dataset"] in provenance and provenance[row["dataset"]] != identity:
            raise ValueError("Dataset task, class vocabulary and lineage must agree across folds")
        provenance[row["dataset"]] = identity
        if row["task"] != "regression":
            k = row.get("classes")
            if (isinstance(k, bool) or not isinstance(k, int) or k < 2
                    or (row["task"] == "binary" and k != 2)
                    or (row["task"] == "multiclass" and k < 3)):
                raise ValueError("Classification requires its ex ante legal class count (binary=2, multiclass>=3)")
        if "disjoint_split_verified" in row or "split_provenance" in row:
            if (row.get("disjoint_split_verified") is not True
                    or not isinstance(row.get("split_provenance"), str)
                    or not row["split_provenance"].strip()):
                raise ValueError("A split declaration requires disjoint_split_verified=true and nonempty split_provenance")
    return rows


def assert_disjoint_panels(panel, other_panels):
    """Reject declared lineage reuse across named panels."""
    rows = validate_panel(panel)
    lineages = {r["lineage"] for r in rows}
    for other in other_panels:
        overlap = lineages & {r["lineage"] for r in validate_panel(other)}
        if overlap:
            raise ValueError(f"Panel source lineages overlap: {sorted(overlap)}")


def score_prediction(task, y_support, y_query, prediction, classes=0):
    if task not in TASKS:
        raise ValueError("Unknown scoring task")
    ys, yq = np.asarray(y_support), np.asarray(y_query)
    prediction = np.asarray(prediction, dtype=np.float64)
    if ys.ndim != 1 or yq.ndim != 1:
        raise ValueError("Selection targets must be one-dimensional")
    if not len(ys) or not len(yq) or not np.isfinite(ys).all() or not np.isfinite(yq).all():
        raise ValueError("Selection scores require nonempty finite support and query targets")
    if task == "regression":
        raw = regression_metrics(yq, prediction)
        center, scale = robust_location_scale(ys)
        z = (ys.astype(np.float64) - center) / scale
        base = float(np.mean((z - z.mean()) ** 2))
        loss = float(np.mean(((yq.astype(np.float64) - prediction) / scale) ** 2))
        metric = "support_standardized_mse"
    else:
        if (isinstance(classes, bool) or not isinstance(classes, (int, np.integer))
                or (task == "binary" and classes != 2) or (task == "multiclass" and classes < 3)
                or prediction.shape != (len(yq), classes)):
            raise ValueError("Prediction columns must match the legal task/class vocabulary")
        if not np.isin(ys, np.arange(classes)).all():
            raise ValueError("Support target outside the declared class vocabulary")
        raw = (binary_metrics(yq, prediction[:, 1]) if task == "binary"
               else multiclass_metrics(yq, prediction))
        # Also validate all probability columns for the binary case.
        loss = multiclass_metrics(yq, prediction)["log_loss"]
        counts = np.bincount(ys.astype(int), minlength=classes)
        base_prob = (counts + .5) / (len(ys) + .5 * classes)
        base = float(-np.log(base_prob[ys.astype(int)]).mean())
        metric = "log_loss"
    if not np.isfinite(loss) or not np.isfinite(base):
        raise ValueError("Nonfinite search score")
    return {"loss": loss, "denominator": max(1e-6, base), "metric": metric,
            "raw_metrics": raw}


def _device_description(model):
    """Record the accelerator/runtime contract without claiming controlled clocks."""
    device = model.device
    return {"device_type": device.type,
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                            else platform.processor() or platform.machine()),
            "total_device_memory_bytes": (int(torch.cuda.get_device_properties(device).total_memory)
                                          if device.type == "cuda" else None),
            "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "hip_runtime": torch.version.hip, "machine": platform.machine(),
            "torch_cpu_threads": torch.get_num_threads()}


def _split_disjointness(row, support, query, support_path, query_path, support_hash, query_hash):
    # Equal content is leakage even if someone asserts different source names.
    if support_path == query_path or support_hash == query_hash:
        raise ValueError("Support and query must not be the same path or identical content")
    for name, table in (("support", support), ("query", query)):
        if "ids" not in table:
            continue
        ids = np.asarray(table["ids"])
        if ids.shape != (len(table["X"]),) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"{name} row IDs must be aligned and unique")
        if ((ids.dtype.kind in "fciub" and not np.isfinite(ids).all())
                or (ids.dtype.kind in "US" and np.any(np.char.strip(ids) == (b"" if ids.dtype.kind == "S" else "")))
                or (ids.dtype.kind in "mM" and np.isnat(ids).any())):
            raise ValueError(f"{name} row IDs cannot be missing or nonfinite")
    validate_pair(support, query)
    if "ids" in support and "ids" in query:
        return {"method": "unique_disjoint_row_ids", "externally_verified": False,
                "limitation": "IDs must use a common, stable source namespace"}
    if (row.get("disjoint_split_verified") is True
            and isinstance(row.get("split_provenance"), str) and row["split_provenance"].strip()):
        return {"method": "declared_split_provenance", "externally_verified": False,
                "split_provenance": row["split_provenance"],
                "limitation": "Researcher declaration requires independent provenance review"}
    raise ValueError("Selection splits require unique disjoint row IDs in both files, or "
                     "disjoint_split_verified=true with nonempty split_provenance")


def evaluate_panel(panel_path, checkpoint, output, *, seed=0, views=1,
                   device="cpu", query_batch_size=512, context_limit=32768, bf16=True,
                   excluded_panels=()):
    if views not in (1, 4, 8) or min(query_batch_size, context_limit) < 1:
        raise ValueError("Use 1/4/8 views and positive query/context limits")
    panel_path = Path(panel_path).resolve()
    panel = json.loads(panel_path.read_text())
    rows = validate_panel(panel)
    exclusions = [json.loads(Path(p).read_text()) for p in excluded_panels]
    assert_disjoint_panels(panel, exclusions)
    exclusion_audit = [{"panel_sha256": digest(path), "role": excluded["role"],
                        "lineages": sorted({r["lineage"] for r in excluded["datasets"]})}
                       for path, excluded in zip(excluded_panels, exclusions)]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation output; old results are never overwritten")
    model, saved = load_checkpoint(checkpoint, device)
    def sync():
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
    result = {"schema_version": 2, "panel_sha256": digest(panel_path),
              "panel_role": panel["role"], "checkpoint_sha256": digest(checkpoint),
              "checkpoint_step": saved.get("step"), "seed": int(saved["config"]["seed"]),
              "view_seed": seed, "views": views,
              "device": str(model.device), "hardware": _device_description(model),
              "cache_policy": CACHE_POLICY,
              "latency_method": "single timed pass per view; no warm-up; setup includes context construction; views run sequentially",
              "peak_memory_scope": "model plus one view cache and its queries; all view caches are not resident together",
              "excluded_panels": exclusion_audit,
              "class_slot_capacity": model.config.classes,
              "bf16": bf16, "context_limit": context_limit,
              "query_batch_size": query_batch_size, "status": "running", "results": [],
              "planned_splits": [[r["task"], r["dataset"], r["fold"]] for r in rows]}
    atomic_json(output, result)
    for row in rows:
        item = {k: row[k] for k in ("dataset", "lineage", "fold", "task")}
        try:
            sp = (panel_path.parent / row["support"]).resolve()
            qp = (panel_path.parent / row["query"]).resolve()
            support_hash, query_hash = digest(sp), digest(qp)
            if sp == qp or support_hash == query_hash:
                raise ValueError("Support and query must not be the same path or identical content")
            support, query = load_table(sp), load_table(qp)
            split_contract = _split_disjointness(row, support, query, sp, qp, support_hash, query_hash)
            if not len(support["X"]) or len(support["X"]) > context_limit:
                raise ValueError("Support outside declared context limit; supply a predeclared selection")
            item.update(support_sha256=support_hash, query_sha256=query_hash,
                        classes=row.get("classes", 0), split_disjointness=split_contract)
            pred, setup, inference = [], 0., 0.
            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.device)
            for view in range(views):
                # Hashing split identity makes view transforms independent of
                # panel ordering and model identity, but paired across models.
                key = repr((seed, row["dataset"], row["fold"], view)).encode()
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
                order = rng.permutation(support["X"].shape[1])
                extra = {"encoding_seed": int(rng.integers(2**31)), "column_ids": order.tolist()}
                k = row.get("classes", 0)
                if row["task"] != "regression":
                    extra["class_slots"] = rng.choice(model.config.classes, k, replace=False).tolist()
                predictor = Predictor(model, query_batch_size, bf16)
                sync(); start = time.perf_counter()
                predictor.fit_context(support["X"][:, order], support["y"],
                                      support["categorical"][order], row["task"], n_classes=k, **extra)
                sync(); setup += time.perf_counter() - start
                start = time.perf_counter()
                pred.append(predictor.predict(query["X"][:, order]))
                sync(); inference += time.perf_counter() - start
                del predictor
            prediction = np.mean(pred, axis=0)
            item.update(score_prediction(row["task"], support["y"], query["y"], prediction, row.get("classes", 0)))
            item.update(status="completed", setup_seconds=setup, cached_prediction_seconds=inference,
                        support_rows=len(support["X"]), query_rows=len(query["X"]),
                        peak_allocated_bytes=(torch.cuda.max_memory_allocated(model.device)
                                              if model.device.type == "cuda" else None))
        except Exception as exc:
            item.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        result["results"].append(item)
        atomic_json(output, result)
    result["status"] = "completed" if all(r["status"] == "completed" for r in result["results"]) else "failed"
    atomic_json(output, result)
    if result["status"] != "completed":
        raise RuntimeError("Panel contains failed splits; the paired comparison cannot drop them")
    return result


def paired_comparison(reference_runs, candidate_runs, *, repeats=10000, seed=0,
                      min_lineages=10, min_seeds=2):
    if repeats < 1 or min_lineages < 2 or min_seeds < 2:
        raise ValueError("Positive replicates and at least two lineages/seeds required")
    if len(reference_runs) != len(candidate_runs) or len(reference_runs) < min_seeds:
        raise ValueError("Paired runs require the same number of seeds and sufficient replication")
    def collect(runs):
        out, identities, seeds, checkpoints, split_identities = {}, set(), set(), set(), {}
        for run in runs:
            if run.get("status") != "completed":
                raise ValueError("Failed or incomplete runs cannot be silently excluded")
            if run["seed"] in seeds:
                raise ValueError("Duplicate training seed")
            seeds.add(run["seed"])
            if run["checkpoint_sha256"] in checkpoints:
                raise ValueError("Repeated checkpoint is not independent training replication")
            checkpoints.add(run["checkpoint_sha256"])
            planned = {tuple(k) for k in run["planned_splits"]}
            policy_fields = ("panel_sha256", "panel_role", "views", "bf16", "context_limit",
                             "query_batch_size", "device", "cache_policy", "class_slot_capacity")
            if any(k not in run for k in (*policy_fields, "hardware")):
                raise ValueError("Evaluation record lacks the complete inference and hardware policy")
            identities.add(tuple(run[k] for k in policy_fields)
                           + (json.dumps(run["hardware"], sort_keys=True), tuple(sorted(planned))))
            delivered = {(r["task"], r["dataset"], r["fold"]) for r in run["results"]}
            if planned != delivered or len(delivered) != len(run["results"]):
                raise ValueError("Incomplete or duplicate planned split coverage")
            for row in run["results"]:
                key = (row["task"], row["dataset"], row["fold"], run["seed"])
                if key in out or row.get("status") != "completed":
                    raise ValueError("Duplicate or failed split")
                split_key = key[:3]
                data_identity = tuple(row[k] for k in
                                      ("lineage", "support_sha256", "query_sha256", "metric", "denominator"))
                if split_key in split_identities and split_identities[split_key] != data_identity:
                    raise ValueError("Split data, metric or denominator changed across training seeds")
                split_identities[split_key] = data_identity
                out[key] = row
        if len(identities) != 1:
            raise ValueError("All replicates must use the same panel and inference policy")
        return out, identities, seeds
    ref, identity, seeds = collect(reference_runs)
    cand, other, candidate_seeds = collect(candidate_runs)
    if identity != other or seeds != candidate_seeds or set(ref) != set(cand):
        raise ValueError("Paired coverage, seeds, panel and inference policy must match")
    for r in reference_runs:
        c = next(c for c in candidate_runs if c["seed"] == r["seed"])
        if r["view_seed"] != c["view_seed"]:
            raise ValueError("Paired augmentation seeds must match")
    by_dataset, by_seed, lineages = defaultdict(list), defaultdict(list), {}
    for key, row in ref.items():
        other = cand[key]
        for field in ("lineage", "support_sha256", "query_sha256", "metric", "denominator"):
            if row[field] != other[field]:
                raise ValueError("Paired data, lineage or training-only denominator differ")
        if (not all(np.isfinite(x) and x >= 0 for x in (row["loss"], other["loss"]))
                or not np.isfinite(row["denominator"]) or row["denominator"] < 1e-6):
            raise ValueError("Invalid paired losses or denominator")
        task, dataset, fold, run_seed = key
        if (task, dataset) in lineages and lineages[(task, dataset)] != row["lineage"]:
            raise ValueError("Dataset lineage changed between folds or seeds")
        lineages[(task, dataset)] = row["lineage"]
        improvement = (row["loss"] - other["loss"]) / row["denominator"]
        by_dataset[(task, dataset)].append(improvement)
        by_seed[(task, dataset, run_seed)].append(improvement)
    if {key[0] for key in by_dataset} != set(TASKS):
        raise ValueError("Selection requires binary, multiclass and regression panels")
    # Each dataset gets one vote, irrespective of its folds or query row count.
    keys = sorted(by_dataset)
    values = np.array([np.mean(by_dataset[k]) for k in keys])
    task_mask = {task: np.array([k[0] == task for k in keys]) for task in TASKS}
    unique = sorted(set(lineages.values()))
    lineage_index = np.array([unique.index(lineages[k]) for k in keys])
    counts = {task: len({lineages[k] for k in keys if k[0] == task}) for task in TASKS}
    if any(n < min_lineages for n in counts.values()):
        raise ValueError("Insufficient source lineages for the declared panel requirement")
    means = {task: float(values[mask].mean()) for task, mask in task_mask.items()}
    rng = np.random.default_rng(seed)
    samples, attempts = [], 0
    while len(samples) < repeats:
        attempts += 1
        if attempts > repeats * 100:
            raise RuntimeError("Cannot obtain bootstrap draws with all task types represented")
        multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[lineage_index]
        if any(not multiplicity[mask].sum() for mask in task_mask.values()):
            continue
        scores = [float(np.average(values[mask], weights=multiplicity[mask])) for mask in task_mask.values()]
        samples.append(scores + [float(np.mean(scores))])
    intervals = np.quantile(np.asarray(samples), [.025, .975], axis=0)
    seed_scores = {}
    for run_seed in sorted(seeds):
        seed_scores[str(run_seed)] = float(np.mean([
            np.mean([np.mean(v) for (t, d, s), v in by_seed.items() if t == task and s == run_seed])
            for task in TASKS]))
    passed = bool(intervals[0, -1] > 0 and np.all(intervals[0, :3] > -.005))
    role = next(iter(identity))[1]
    requirements = {
        "confirmation_panel": role == "confirmation",
        "at_least_10000_bootstrap_resamples": repeats >= CONFIRMATION_BOOTSTRAPS,
        "at_least_10_source_lineages_per_task": all(n >= CONFIRMATION_LINEAGES for n in counts.values()),
        "at_least_two_training_seeds": len(seeds) >= 2,
    }
    return {"objective": "equal_dataset_within_task_then_equal_task", "panel_role": role,
            "improvement": float(np.mean(list(means.values()))), "task_improvements": means,
            "intervals": {name: {"lower": float(intervals[0, i]), "upper": float(intervals[1, i])}
                          for i, name in enumerate((*TASKS, "overall"))},
            "bootstrap": {"unit": "source_lineage", "global_lineage_resampling": True,
                          "valid_repeats": repeats, "attempts": attempts, "seed": seed},
            "lineages_per_task": counts, "seed_improvements": seed_scores,
            "seed_standard_deviation": float(np.std(list(seed_scores.values()), ddof=1)),
            "passes_statistical_rule": passed,
            "confirmation_protocol_requirements": requirements,
            "unmet_confirmation_requirements": [key for key, met in requirements.items() if not met],
            "confirmation_evidence": bool(passed and all(requirements.values())),
            "inference_policy": {k: reference_runs[0][k] for k in
                                  ("views", "bf16", "context_limit", "query_batch_size", "device", "cache_policy",
                                   "class_slot_capacity", "hardware")},
            "not_an_automatic_training_decision": True,
            "limits": "Lineage declarations and benchmark isolation need external provenance review; intervals condition on trained seeds. Matching hardware metadata does not control accelerator clocks, competing jobs or host I/O; latency and slice failures require separate review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    for name in ("panel", "checkpoint", "output"):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--exclude-panel", action="append", default=[])
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--views", type=int, choices=[1, 4, 8], default=1)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--query-batch-size", type=int, default=512)
    evaluate.add_argument("--context-limit", type=int, default=32768)
    evaluate.add_argument("--fp32", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--reference", nargs="+", required=True)
    compare.add_argument("--candidate", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--repeats", type=int, default=10000)
    compare.add_argument("--min-lineages", type=int, default=10)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate_panel(args.panel, args.checkpoint, args.output, seed=args.seed, views=args.views,
                       device=args.device, query_batch_size=args.query_batch_size,
                       context_limit=args.context_limit, bf16=not args.fp32,
                       excluded_panels=args.exclude_panel)
    else:
        if Path(args.output).exists():
            raise FileExistsError("Choose a new comparison output")
        result = paired_comparison([json.loads(Path(p).read_text()) for p in args.reference],
                                   [json.loads(Path(p).read_text()) for p in args.candidate],
                                   repeats=args.repeats, seed=args.seed, min_lineages=args.min_lineages)
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
