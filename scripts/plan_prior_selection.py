"""Write a deterministic, unexecuted prior-selection experiment manifest.

This planner performs accounting only. It neither implements the proposed
generators nor launches training, selects a winning prior, or estimates accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "research" / "specs" / "prior_selection_v1.json"
MECHANISMS = ("R", "forest", "hierarchy", "smooth_local", "sparse_interaction")
OBSERVATIONS = ("identity", "mcar", "mar", "mnar", "coarsen")
PHASES = ("P", "M", "A", "V", "F")
PROJECT_KEYS = ("systems", "screening", "scaling", "final", "evaluation", "reserve")
STAGE_PARTS = (90, 9, 1)

_VOLUME_SPEC = importlib.util.spec_from_file_location(
    "prior_selection_volume_reference", ROOT / "scripts" / "plan_prior_volume.py")
_VOLUME = importlib.util.module_from_spec(_VOLUME_SPEC)
_VOLUME_SPEC.loader.exec_module(_VOLUME)


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return value


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _simplex(weights, names, label):
    if not isinstance(weights, dict) or set(weights) != set(names):
        raise ValueError(f"{label} must have exactly these keys: {list(names)}")
    checked = {name: _number(weights[name], f"{label}.{name}") for name in names}
    if not math.isclose(math.fsum(checked.values()), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label} must sum to one")
    return checked


def validate_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("The specification must be a JSON object")
    required = {"mechanism_weights", "observation_weights", "mixture_anchor_grid",
                "screening_allocations", "project_gpu_hour_budget"}
    if required - set(spec):
        raise ValueError(f"Missing specification keys: {sorted(required - set(spec))}")
    mechanisms = _simplex(spec["mechanism_weights"], MECHANISMS, "mechanism_weights")
    observations = _simplex(spec["observation_weights"], OBSERVATIONS, "observation_weights")
    grid = spec["mixture_anchor_grid"]
    if not isinstance(grid, list) or len(grid) != 3:
        raise ValueError("mixture_anchor_grid must contain three distinct anchors")
    for value in grid:
        _number(value, "mixture_anchor_grid", positive=True)
        if value >= 1:
            raise ValueError("Mixture anchors must be strictly between zero and one")
    if len(set(grid)) != len(grid) or mechanisms["R"] == 1:
        raise ValueError("Mixture anchors must be distinct and the seed mixture needs non-R mass")
    # The display labels below are dictionary keys and run identifiers. Numeric
    # distinctness alone does not prevent :g rounding from merging nearby arms.
    if len({f"R{anchor * 100:g}" for anchor in grid}) != len(grid):
        raise ValueError("Mixture anchors must produce distinct formatted arm labels")
    project = spec["project_gpu_hour_budget"]
    if not isinstance(project, dict) or set(project) != set(PROJECT_KEYS):
        raise ValueError(f"project_gpu_hour_budget must have exactly {list(PROJECT_KEYS)}")
    project = {name: _number(project[name], f"budget.{name}", positive=True) for name in PROJECT_KEYS}
    if not math.isclose(math.fsum(project.values()), 50_000, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Project allocations must conserve the 50,000 GPU-hour ceiling")
    allocations = spec["screening_allocations"]
    if not isinstance(allocations, list) or len(allocations) != len(PHASES):
        raise ValueError("screening_allocations must contain P, M, A, V, and F exactly once")
    expected_arms = dict(zip(PHASES, (8, 3, 4, 2, 3)))
    checked = {}
    for allocation in allocations:
        keys = {"id", "arms", "seeds", "gpu_hours_per_run"}
        if not isinstance(allocation, dict) or set(allocation) != keys:
            raise ValueError(f"Each allocation must contain exactly {sorted(keys)}")
        phase = allocation["id"]
        if not isinstance(phase, str) or phase not in PHASES or phase in checked:
            raise ValueError("Screening phase IDs must be distinct members of P, M, A, V, F")
        arms = _integer(allocation["arms"], f"{phase}.arms")
        seeds = _integer(allocation["seeds"], f"{phase}.seeds")
        if arms != expected_arms[phase] or seeds != 2:
            raise ValueError(f"{phase} requires {expected_arms[phase]} arms and two seeds in this design")
        hours = _number(allocation["gpu_hours_per_run"], f"{phase}.gpu_hours_per_run", positive=True)
        checked[phase] = {"id": phase, "arms": arms, "seeds": seeds, "gpu_hours_per_run": hours}
    cost = math.fsum(row["arms"] * row["seeds"] * row["gpu_hours_per_run"] for row in checked.values())
    if not math.isclose(cost, project["screening"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Run allocations must exactly conserve the screening GPU-hour allocation")
    return mechanisms, observations, grid, checked, project


def mixture_with_anchor(mechanisms, anchor):
    """Hold the non-reference proportions fixed while varying reference mass."""
    mechanisms = _simplex(mechanisms, MECHANISMS, "mechanism_weights")
    _number(anchor, "anchor", positive=True)
    if anchor >= 1 or mechanisms["R"] == 1:
        raise ValueError("An anchor below one and positive non-reference mass are required")
    result = {name: (1 - anchor) * mechanisms[name] / (1 - mechanisms["R"])
              for name in MECHANISMS if name != "R"}
    return {"R": anchor, **result}


def exposure_counts(mechanisms, observations, episodes):
    """Conditional expectations under the existing 90/9/1 requested-width law.

    An ineligible hierarchy draw returns to R. The observation draw is independent
    and is retained after fallback. These are expected exposures, never quotas.
    """
    mechanisms = _simplex(mechanisms, MECHANISMS, "mechanism_weights")
    observations = _simplex(observations, OBSERVATIONS, "observation_weights")
    _integer(episodes, "episodes")
    if episodes % 100:
        raise ValueError("episodes must be divisible by 100 for exact 90/9/1 stage allocation")
    eligibility = _VOLUME.stage_p1_eligibility()
    stages = []
    for index, (part, eligible) in enumerate(zip(STAGE_PARTS, eligibility)):
        n = episodes // 100 * part
        before = {name: n * weight for name, weight in mechanisms.items()}
        returned = before["hierarchy"] * (1 - eligible)
        after = dict(before)
        after["hierarchy"] *= eligible
        after["R"] += returned
        stages.append({"stage": index + 1, "conditional_episode_target": n,
                       "hierarchy_requested_width_eligibility": eligible,
                       "before_eligibility": before, "after_eligibility": after,
                       "hierarchy_fallback_to_R": returned,
                       "mechanism_observation_crossproduct": {
                           name: {obs: count * weight for obs, weight in observations.items()}
                           for name, count in after.items()}})
    before_total = {name: episodes * weight for name, weight in mechanisms.items()}
    after_total = {name: math.fsum(stage["after_eligibility"][name] for stage in stages)
                   for name in MECHANISMS}
    return {
        "status": "analytic_expectation_conditional_on_specified_shape_law",
        "conditional_accepted_episode_target": episodes,
        "stage_episode_fractions": [part / 100 for part in STAGE_PARTS],
        "stages": stages, "before_eligibility": before_total,
        "after_eligibility": after_total,
        "hierarchy_fallback_to_R": math.fsum(stage["hierarchy_fallback_to_R"] for stage in stages),
        "mechanism_observation_crossproduct": {
            name: {obs: count * weight for obs, weight in observations.items()}
            for name, count in after_total.items()},
        "observation_marginals": {obs: episodes * weight for obs, weight in observations.items()},
        "observation_counts_are_requested_modes_before_shape_fallback": True,
        "effective_observation_marginals": None,
        "realized_accepted_episodes": None, "raw_generation_attempts": None,
        "realized_binary_episodes": None, "realized_multiclass_episodes": None,
        "generation_or_training_gpu_hours": None,
        "assumptions": [
            "Requested-width law is stage_p1_eligibility() in plan_prior_volume.py, with 90/9/1 accepted-episode stages.",
            "Hierarchy requires requested width >= 8. Other families are assumed eligible at every shape; their proposed generators are not yet implemented.",
            "This shape law must be implemented and verified before these expectations describe a training run.",
            "No rejection-induced source substitution, failed updates, replay, or diagnostic shape overrides are included.",
            "Observation modes are mutually exclusive branches independent of mechanism and eligibility; identity means no additional observation transformation.",
            "Observation counts name requested modes. Effective mode counts are unknown here: the specified MAR branch falls back to MCAR at F=1, and a requested coarsening need not alter every selected column.",
            "R retains its native numerical encoding; an observation wrapper does not recover native categorical identities.",
            "All counts are expectations, not guaranteed realized counts, independently sampled worlds, or performance estimates."]}


def _mechanism_arm(name, share):
    return {mechanism: (1 - share if mechanism == "R" else share if mechanism == name else 0)
            for mechanism in MECHANISMS}


def build_plan(spec, episodes=64_000_000):
    mechanisms, observations, grid, allocations, project = validate_spec(spec)
    _integer(episodes, "episodes")
    identity = {name: int(name == "identity") for name in OBSERVATIONS}
    reference = {name: int(name == "R") for name in MECHANISMS}
    p_arms = [
        ("R", reference, identity), ("R_O", reference, observations),
        ("R_H15", _mechanism_arm("hierarchy", .15), identity),
        ("R_F20", _mechanism_arm("forest", .20), identity),
        ("R_S10", _mechanism_arm("smooth_local", .10), identity),
        ("R_I05", _mechanism_arm("sparse_interaction", .05), identity),
        ("Q_clean", mechanisms, identity), ("Q_observed", mechanisms, observations)]
    exposure = {name: exposure_counts(mix, obs, episodes) for name, mix, obs in p_arms}
    grid_mixtures = {f"R{anchor * 100:g}": mixture_with_anchor(mechanisms, anchor) for anchor in grid}
    exposure.update({f"M_{name}": exposure_counts(mix, observations, episodes)
                     for name, mix in grid_mixtures.items()})
    shared_blockers = ["shared_shape_law_and_paired_experiment_harness_not_implemented",
                       "GPU_hardware_and_end_to_end_timings_unavailable"]
    trials, phase_runs = [], {}
    phase_arms = {
        "P": [name for name, _, _ in p_arms],
        "M": list(grid_mixtures),
        "A": ["R_compressed", "selected_compressed", "R_persistent_cell", "selected_persistent_cell"],
        "V": ["R", "final_candidate"],
        "F": ["standard_replay", "fraud_reservoir_positives", "fraud_full_selector"]}
    # Even if the central M mixture equals P's Q_observed, fresh streams make
    # it a new replicate. Confirmation seeds must remain unseen in all screens.
    seed_base = {"P": 0, "M": 2, "A": 0, "V": 4, "F": 6}
    previous = None
    for phase in PHASES:
        phase_runs[phase] = []
        for arm in phase_arms[phase]:
            for seed in range(seed_base[phase], seed_base[phase] + allocations[phase]["seeds"]):
                run_id = f"{phase}_{arm}_seed{seed}"
                blockers = list(shared_blockers)
                if phase == "P" and arm not in ("R",):
                    blockers.append("specified_generator_or_observation_extension_not_implemented")
                if phase == "M":
                    blockers.append("surviving_mechanisms_and_observation_policy_not_selected")
                if phase == "A":
                    blockers.extend(["prior_not_selected", "compact_persistent_cell_control_not_implemented"])
                if phase == "V":
                    blockers.append("final_architecture_prior_pair_not_selected")
                if phase == "F":
                    blockers.extend(["shared_standard_parent_checkpoint_not_selected_or_trained",
                                     "paired_finance_control_harness_not_implemented"])
                trial = {
                    "id": run_id, "phase": phase, "arm": arm, "seed": seed,
                    "paired_seed_group": f"{phase}_seed{seed}",
                    "gpu_hour_ceiling": allocations[phase]["gpu_hours_per_run"],
                    "status": "blocked_unimplemented" if phase == "P" else "pending_selection",
                    "implementation_status": "specified_not_implemented",
                    "dependencies": [] if previous is None else list(phase_runs[previous]),
                    "blockers": blockers,
                    "selection_gate": None if previous is None else f"review_and_freeze_after_{previous}",
                    "launched": False, "completed": False, "actual_gpu_hours": None,
                    "accepted_training_episodes": None, "benchmark_scores": None,
                    "training_command": None,
                }
                if phase == "P":
                    _, mix, obs = next(row for row in p_arms if row[0] == arm)
                    trial.update(mechanism_weights=mix, observation_weights=obs)
                elif phase == "M":
                    trial.update(proposed_mechanism_weights=grid_mixtures[arm],
                                 proposed_observation_weights=observations,
                                 weights_status="proposal_pending_P_selection; do_not_launch_automatically")
                elif phase == "F":
                    trial["shared_parent_checkpoint"] = "same_selected_standard_checkpoint_for_all_F_arms"
                    trial["parent_checkpoint_path"] = None
                trials.append(trial)
                phase_runs[phase].append(run_id)
        previous = phase
    expected_run_count = sum(row["arms"] * row["seeds"] for row in allocations.values())
    actual_cost = math.fsum(trial["gpu_hour_ceiling"] for trial in trials)
    if (len(trials) != expected_run_count
            or not math.isclose(actual_cost, project["screening"], rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Generated trials must conserve the validated run count and screening budget")
    return {
        "schema_version": 1,
        "status": "unexecuted_research_plan_with_implementation_and_selection_gates",
        "not_a_training_launch_config": True, "performance_guarantee": False,
        "runtime_candidate_configuration_changed": False,
        "project_gpu_hour_ceiling": math.fsum(project.values()),
        "project_gpu_hour_budget": project,
        "screening_gpu_hour_ceiling": math.fsum(t["gpu_hour_ceiling"] for t in trials),
        "phase_gpu_hour_ceilings": {phase: allocations[phase]["arms"] * allocations[phase]["seeds"]
                                    * allocations[phase]["gpu_hours_per_run"] for phase in PHASES},
        "run_count": len(trials), "completed_run_count": 0, "actual_gpu_hours": None,
        "trials": trials,
        "conditional_volume_examples": exposure,
        "volume_note": "The episode target illustrates one future run under each fixed proposal; it is not a target for every screening trial or a claim that the target fits its GPU-hour ceiling.",
        "implementation_inventory": {
            "R": "existing_pinned_reference_generator; shared_selection_harness_not_implemented",
            "forest": "specified_not_implemented", "hierarchy": "specified_hyperlaw_extension_not_implemented; existing_P1_is_only_a_control",
            "smooth_local": "specified_not_implemented", "sparse_interaction": "specified_not_implemented",
            "observation_wrapper": "specified_not_implemented",
            "common_shape_law": "specified_not_implemented",
            "compact_persistent_cell": "specified_not_implemented"},
        "paired_comparison_contract": [
            "Same seed index pairs initialization and task-shape schedules; mechanism-specific randomness is namespaced rather than falsely claiming identical tables.",
            "Fix architecture, optimizer, shape law, preprocessing and inference budget for P and M; record achieved exposure and compute as separate quantities.",
            "A crosses R and the selected prior with the authored compressed and proposed compact persistent-cell architectures.",
            "M uses fresh seeds 2 and 3 rather than duplicating P's central mixture replicates; V uses fresh seeds 4 and 5, unseen during P/M/A, and a held-out confirmation panel.",
            "F starts all arms from the same selected standard checkpoint and compares standard replay, reservoir-plus-positive support, and the full finance selector.",
            "GPU-hour ceilings include charged generation, training and within-run validation overhead; no throughput or completion is inferred.",
            "Only explicit reviewed outcomes may unlock later phases; dependencies do not implement automatic selection."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--episodes", type=int, default=64_000_000,
                        help="Conditional exposure example, not a screening-run horizon")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_bytes = args.spec.read_bytes()
    spec = json.loads(source_bytes)
    plan = build_plan(spec, args.episodes)
    try:
        source_path = str(args.spec.resolve().relative_to(ROOT))
    except ValueError:
        source_path = str(args.spec.resolve())
    plan["source_spec"] = source_path
    plan["source_spec_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": plan["status"],
                      "run_count": plan["run_count"], "completed_run_count": 0,
                      "screening_gpu_hour_ceiling": plan["screening_gpu_hour_ceiling"]}, indent=2))
    return plan


if __name__ == "__main__":
    main()
