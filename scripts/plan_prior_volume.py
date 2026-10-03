"""Count provisional accepted-task exposures and enforce measured GPU-hour caps.

This is a planning artifact, not a training launcher or throughput benchmark.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction
import json
import math
from pathlib import Path


GLOBAL_BATCH = 256
ALLOCATED_GPUS = 64
STAGE_PARTS = (90, 9, 1)
STANDARD_EPISODES = 64_000_000
FINANCE_EPISODES = 1_024_000
PHASE_CEILINGS = {"standard": 9_000, "finance": 3_000}
PROJECT_CEILINGS = {"systems_reference": 5_000, "controlled_screens": 15_000,
                    "size_bridges": 10_000, "conditional_final": 12_000,
                    "evaluation": 5_000, "reserve": 3_000}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_decimal(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _number(value):
    """Keep exact integral expectations as integers in the JSON artifact."""
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else float(value)


def stage_updates(total_updates: int) -> list[int]:
    if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates < 0 or total_updates % 100:
        raise ValueError("Updates must be a nonnegative multiple of 100 for exact 90/9/1 stages")
    return [total_updates // 100 * part for part in STAGE_PARTS]


def finance_world_counts(updates: list[int], global_batch: int = GLOBAL_BATCH,
                         reuse_updates: int = 4) -> list[int]:
    """Distinct (world seed, stage) pairs under runtime's four-update reuse.

    Each reuse group has global_batch worlds. A stage transition within a
    reuse group makes a new stage-specific population, so it must be counted.
    """
    _positive_integer(global_batch, "global_batch")
    _positive_integer(reuse_updates, "reuse_updates")
    result, start = [], 0
    for count in updates:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Stage update counts must be nonnegative integers")
        end = start + count
        groups = (end + reuse_updates - 1) // reuse_updates - start // reuse_updates if count else 0
        result.append(groups * global_batch)
        start = end
    return result


def requested_width_eligibility(max_features: int) -> float:
    """P(round(Uniform(1, max_features)) >= 8), not discrete-uniform width."""
    _positive_integer(max_features, "max_features")
    return 0. if max_features < 8 else (max_features - 7.5) / (max_features - 1)


def stage_p1_eligibility() -> list[float]:
    """Analytic native requested-width marginals for the planned envelopes.

    Stage three samples int(LogUniform(400, 60000)). Thus N<=10240 ends at
    continuous length 10241, and analogously for the other inclusive caps.
    Constant-column removal occurs after the mixture branch and is irrelevant
    to this requested-width eligibility calculation.
    """
    p100 = requested_width_eligibility(100)
    p2 = .8 * p100 + .2 * requested_width_eligibility(256)
    intervals = ((400, 10241, .8 * p100 + .2 * requested_width_eligibility(512)),
                 (10241, 20001, requested_width_eligibility(80)),
                 (20001, 30001, requested_width_eligibility(60)),
                 (30001, 40001, requested_width_eligibility(40)),
                 (40001, 50001, requested_width_eligibility(30)),
                 (50001, 60000, requested_width_eligibility(20)))
    p3 = sum(math.log(hi / lo) / math.log(60000 / 400) * p for lo, hi, p in intervals)
    return [p100, p2, p3]


def standard_counts(episodes_by_stage: list[int], eligible_by_stage: list[int] | None = None) -> dict:
    """Source/task expectations, with envelope and prior as overlapping axes."""
    if len(episodes_by_stage) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in episodes_by_stage):
        raise ValueError("Exactly three nonnegative integer stage episode counts are required")
    if eligible_by_stage is not None and (len(eligible_by_stage) != 3 or any(
            isinstance(e, bool) or not isinstance(e, int) or not 0 <= e <= n
            for e, n in zip(eligible_by_stage, episodes_by_stage))):
        raise ValueError("Eligible counts must be integers between zero and each stage's episode count")
    stages = []
    marginal_eligibility = stage_p1_eligibility()
    for index, n in enumerate(episodes_by_stage):
        extension = Fraction(0 if index == 0 else 1, 5)
        extended_max_classes = (10, 32, 256)[index]
        classification = Fraction(n, 2)
        # Native GraphPrior draws class budget uniformly from 2 through Cmax.
        binary = classification * ((1 - extension) / 9 + extension / (extended_max_classes - 1))
        p1 = None if eligible_by_stage is None else Fraction(eligible_by_stage[index], 20)
        marginal_p1 = n * marginal_eligibility[index] / 20
        stages.append({
            "stage": index + 1, "accepted_macroepisodes_target": n,
            "source_branch_expectation_before_eligibility": {"R": _number(Fraction(19 * n, 20)), "P1": _number(Fraction(n, 20)), "P4": 0},
            "eligible_slots_observed_or_assumed": None if eligible_by_stage is None else eligible_by_stage[index],
            "accepted_source_conditional_expectation": None if p1 is None else {"P1": _number(p1), "R": _number(n - p1), "P4": 0},
            "requested_width_eligibility_probability": marginal_eligibility[index],
            "eligibility_adjusted_source_expectation": {"P1": marginal_p1, "R": n - marginal_p1, "P4": 0},
            "task_expectation_with_planned_envelope": {"classification": _number(classification),
                "binary_class_budget": _number(binary), "multiclass_class_budget": _number(classification - binary),
                "regression": _number(classification)},
            "envelope_extension_expectation": _number(n * extension),
            "extension_probability": _number(extension),
            "extension_max_classes": extended_max_classes,
            "extension_max_features": (100, 256, 512)[index],
            "envelope_and_P1_branch_overlap_expectation_before_eligibility": _number(n * extension / 20),
        })
    total = sum(episodes_by_stage)
    classification = Fraction(total, 2)
    return {
        "arm": "R_P1_05", "stages": stages,
        "accepted_macroepisodes_target": total,
        "source_branch_expectation_before_eligibility": {"R": _number(Fraction(19 * total, 20)), "P1": _number(Fraction(total, 20)), "P4": 0},
        "accepted_source_conditional_expectation": None if eligible_by_stage is None else {
            "R": _number(total - Fraction(sum(eligible_by_stage), 20)),
            "P1": _number(Fraction(sum(eligible_by_stage), 20)), "P4": 0},
        "accepted_source_rule": "E[P1 | eligible slots] = 0.05 * sum(eligible slots); E[R | eligible slots] = N - E[P1]. Realized counts are random integers and must be logged.",
        "eligibility_adjusted_source_expectation": {
            "R": total - sum(s["eligibility_adjusted_source_expectation"]["P1"] for s in stages),
            "P1": sum(s["eligibility_adjusted_source_expectation"]["P1"] for s in stages), "P4": 0},
        "marginal_expectation_assumptions": "Analytic requested-width law for the stated native stages and 20% envelope extensions; no diagnostic shape overrides, failed/skipped/replayed updates or source substitution. Random realized counts are not quotas.",
        "task_expectation_with_planned_envelope": {key: sum(s["task_expectation_with_planned_envelope"][key] for s in stages)
            for key in ("classification", "binary_class_budget", "multiclass_class_budget", "regression")},
        "strict_native_without_envelope_task_expectation": {
            "classification": _number(classification), "binary_class_budget": _number(classification / 9),
            "multiclass_class_budget": _number(classification * 8 / 9), "regression": _number(classification)},
        "envelope_extension_expectation": sum(s["envelope_extension_expectation"] for s in stages),
        "envelope_and_P1_branch_overlap_expectation_before_eligibility": sum(s["envelope_and_P1_branch_overlap_expectation_before_eligibility"] for s in stages),
        "envelope_is_additional_episode_mass": False,
        "raw_reference_attempt_count": None,
        "raw_attempt_note": "Rejected reference tables/graphs are generator work, not additional trained macroepisodes; measure retry counts and cost.",
        "observed_class_note": "Task names refer to the ex ante class budget. Accepted tables can contain fewer observed classes.",
    }


def plan_horizon(target_episodes: int, gpu_hour_cap: float, *,
                 seconds_per_update: list[float] | None = None,
                 overhead_gpu_hours: float = 0, global_batch: int = GLOBAL_BATCH,
                 allocated_gpus: int = ALLOCATED_GPUS, reuse_updates: int = 1) -> dict:
    """Uniformly shrink a fixed 90/9/1 horizon without exceeding its phase cap.

    Use 100-update blocks for standard and 400-update blocks for four-use
    finance worlds. No horizon is certified feasible without timings.
    """
    for name, value in (("target_episodes", target_episodes), ("global_batch", global_batch),
                        ("allocated_gpus", allocated_gpus), ("reuse_updates", reuse_updates)):
        _positive_integer(value, name)
    if target_episodes % global_batch:
        raise ValueError("Target episodes must be divisible by the global batch")
    updates = target_episodes // global_batch
    quantum = 100 * reuse_updates
    if updates % quantum:
        raise ValueError(f"Target horizon must be divisible by {quantum} updates to preserve stages and world reuse")
    cap = _nonnegative_decimal(gpu_hour_cap, "gpu_hour_cap")
    overhead = _nonnegative_decimal(overhead_gpu_hours, "overhead_gpu_hours")
    if cap <= 0 or overhead > cap:
        raise ValueError("A positive cap and overhead no larger than the cap are required")
    usable_hours = cap - overhead
    wall_seconds = usable_hours * 3600 / allocated_gpus
    target_stages = stage_updates(updates)
    result = {
        "provisional_target_accepted_macroepisodes": target_episodes,
        "target_updates": updates, "target_stage_updates": target_stages,
        "target_stage_accepted_macroepisodes": [n * global_batch for n in target_stages],
        "global_batch": global_batch, "allocated_gpus": allocated_gpus,
        "gpu_hour_ceiling": float(cap), "overhead_gpu_hours_reserved": float(overhead),
        "training_gpu_hours_available": float(usable_hours),
        "available_training_wall_hours_on_all_gpus": float(wall_seconds / 3600),
        "required_aggregate_macroepisodes_per_second": float(Decimal(target_episodes) / wall_seconds) if wall_seconds else None,
        "required_macroepisodes_per_gpu_second": float(Decimal(target_episodes) / (usable_hours * 3600)) if usable_hours else None,
        "maximum_weighted_mean_seconds_per_update": float(wall_seconds / updates),
        "horizon_quantum_updates": quantum, "horizon_quantum_macroepisodes": quantum * global_batch,
        "measured_seconds_per_update": None, "target_required_gpu_hours": None,
        "target_fits_measured_budget": None, "budget_feasible_horizon": None,
        "hard_cap_takes_precedence_over_episode_target": True,
    }
    if seconds_per_update is None:
        return result
    if not isinstance(seconds_per_update, (list, tuple)) or len(seconds_per_update) != 3:
        raise ValueError("Exactly three measured seconds-per-update values are required")
    timings = [_nonnegative_decimal(t, "seconds_per_update") for t in seconds_per_update]
    if any(t <= 0 for t in timings):
        raise ValueError("Measured seconds per update must be positive")
    target_seconds = sum(n * t for n, t in zip(target_stages, timings))
    target_required = target_seconds * allocated_gpus / 3600 + overhead
    block_stages = stage_updates(quantum)
    seconds_per_block = sum(n * t for n, t in zip(block_stages, timings))
    blocks = min(updates // quantum, int((wall_seconds / seconds_per_block).to_integral_value(rounding=ROUND_FLOOR)))
    feasible_updates = blocks * quantum
    feasible_stages = stage_updates(feasible_updates)
    projected_gpu_hours = sum(n * t for n, t in zip(feasible_stages, timings)) * allocated_gpus / 3600 + overhead
    assert projected_gpu_hours <= cap
    result.update(measured_seconds_per_update=[float(t) for t in timings],
                  target_required_gpu_hours=float(target_required),
                  target_fits_measured_budget=target_required <= cap,
                  budget_feasible_horizon={
                      "updates": feasible_updates, "accepted_macroepisodes": feasible_updates * global_batch,
                      "stage_updates": feasible_stages, "stage_end_steps": [sum(feasible_stages[:i + 1]) for i in range(3)],
                      "stage_accepted_macroepisodes": [n * global_batch for n in feasible_stages],
                      "uniform_fraction_of_target": feasible_updates / updates,
                      "projected_total_gpu_hours": float(projected_gpu_hours),
                      "unused_gpu_hours": float(cap - projected_gpu_hours),
                      "status": "fits_target" if feasible_updates == updates else "shrink_horizon" if feasible_updates else "no_complete_stage_cycle_fits",
                      "is_performance_guarantee": False,
                  })
    return result


def build_plan(measurements: dict | None = None) -> dict:
    measurements = {} if measurements is None else dict(measurements)
    allowed = {"allocated_gpus", "global_batch", "measurement_id", "standard_seconds_per_update",
               "finance_seconds_per_update", "standard_overhead_gpu_hours", "finance_overhead_gpu_hours"}
    if set(measurements) - allowed:
        raise ValueError(f"Unknown measurement keys: {sorted(set(measurements) - allowed)}")
    for key, expected in (("allocated_gpus", ALLOCATED_GPUS), ("global_batch", GLOBAL_BATCH)):
        if key in measurements and (isinstance(measurements[key], bool) or measurements[key] != expected):
            raise ValueError(f"Measurements must use this plan's {key}={expected}; no hardware or batch extrapolation")
    result = {
        "status": "provisional_planning_target; budget ceilings override volume",
        "not_a_training_launch_config": True,
        "project_gpu_hour_ceiling": sum(PROJECT_CEILINGS.values()),
        "project_phase_gpu_hour_ceilings": PROJECT_CEILINGS,
        "combined_conditional_final_gpu_hour_ceiling": sum(PHASE_CEILINGS.values()),
        "stage_accepted_episode_fractions": [part / 100 for part in STAGE_PARTS],
        "stage_policy_note": "90/9/1 is an accepted-episode allocation for this provisional plan; it is separate from plan_training.py's 60/25/15 GPU-hour allocation.",
        "measurement_id": measurements.get("measurement_id"),
    }
    for phase, episodes, reuse in (("standard", STANDARD_EPISODES, 1), ("finance", FINANCE_EPISODES, 4)):
        result[phase] = plan_horizon(episodes, PHASE_CEILINGS[phase], reuse_updates=reuse,
                                    seconds_per_update=measurements.get(phase + "_seconds_per_update"),
                                    overhead_gpu_hours=measurements.get(phase + "_overhead_gpu_hours", 0))
    standard = result["standard"]
    standard["target_source_task_envelope_counts"] = standard_counts(standard["target_stage_accepted_macroepisodes"])
    if standard["budget_feasible_horizon"] is not None:
        standard["budget_feasible_horizon"]["source_task_envelope_counts"] = standard_counts(
            standard["budget_feasible_horizon"]["stage_accepted_macroepisodes"])
    finance = result["finance"]
    finance.update(task="binary_fraud", natural_prevalence=1e-4, world_reuse_updates=4,
                   target_distinct_worlds_by_stage=finance_world_counts(finance["target_stage_updates"]),
                   target_distinct_worlds=sum(finance_world_counts(finance["target_stage_updates"])),
                   world_count_condition="Every four-attempt reuse group completes, no optimizer attempts are skipped or replayed, and stage boundaries align; independent stage-specific populations are counted separately.",
                   stage_policy_status="Provisional 90/9/1 finance staging for planning, to be frozen after timing/coverage gates.",
                   population_note="Each world can represent a few million natural-prevalence historical rows lazily; these are not all materialized context/training rows.",
                   query_note="Balanced query proposals retain population importance weights; they do not change the natural fraud prevalence. Routed context reuse does not add independent macroepisodes.")
    if finance["budget_feasible_horizon"] is not None:
        world_counts = finance_world_counts(finance["budget_feasible_horizon"]["stage_updates"])
        finance["budget_feasible_horizon"].update(distinct_worlds_by_stage=world_counts, distinct_worlds=sum(world_counts))
    result["volume_rationale"] = {
        "interpretation": "Same order of table exposures as published open tabular models, not a scaling law, compute-equivalent comparison, or accuracy guarantee.",
        "tabicl_v2": "64 * (500000 + 40000 + 10000) = 35.2 million table exposures per separate classifier or regressor; 70.4 million for the pair.",
        "tabicl_source": "https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/scripts/train_v2_clf_stage1.sh",
        "kumo": "Release reports about 35/71/137 million tables for Small/Medium/Large. Architecture, shapes and exact task accounting differ; do not assume equal compute or independent unique worlds.",
        "kumo_source": "https://huggingface.co/blog/nvidia/kumo-tabular",
    }
    result["measurement_requirements"] = (
        "Use complete distributed optimizer updates on 64 GPUs at global batch 256, including generator/rejection, "
        "codec, transfer, forward/backward, optimizer, synchronization and rank skew. Finance timings must cover "
        "cold and warm uses across complete four-update reuse groups, all selectors and routed prefills. "
        "Reserve setup/checkpoint/other charged overhead explicitly and use conservative timing summaries. "
        "Freeze the feasible horizon and LR schedule before launch; do not run the requested horizon through its cap. "
        "Actual accepted, rejected, failed and skipped-update counts still govern reported exposure.")
    result["generation_and_validation_policy"] = (
        "Generate training tasks on demand; the volume target does not require pre-generating or storing a full table bank. "
        "Keep fixed validation-world seeds disjoint from training namespaces and reuse them only for validation. "
        "R's eight mechanism functions compose within graphs; they are not eight exclusive table families with equal sample quotas.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, help="Optional JSON containing three measured stage seconds/update per phase")
    parser.add_argument("--output", type=Path, default=Path("research/results/prior_volume_plan.json"))
    args = parser.parse_args(argv)
    measurements = json.loads(args.measurements.read_text()) if args.measurements else None
    report = build_plan(measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "standard_target": STANDARD_EPISODES, "finance_target": FINANCE_EPISODES,
                      "final_gpu_hour_ceiling": sum(PHASE_CEILINGS.values())}, indent=2))
    return report


if __name__ == "__main__":
    main()
