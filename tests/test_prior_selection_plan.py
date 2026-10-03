"""Prior-selection accounting and gates; these are not transfer-performance tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plan_prior_selection.py"
_SPEC = importlib.util.spec_from_file_location("plan_prior_selection", _PATH)
planner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(planner)


@pytest.fixture
def spec():
    return {
        "mechanism_weights": {"R": .5, "forest": .2, "hierarchy": .15,
                              "smooth_local": .1, "sparse_interaction": .05},
        "observation_weights": {"identity": .5, "mcar": .15, "mar": .15,
                                "mnar": .1, "coarsen": .1},
        "mixture_anchor_grid": [.7, .5, .3],
        "screening_allocations": [
            {"id": phase, "arms": arms, "seeds": 2, "gpu_hours_per_run": hours}
            for phase, arms, hours in (("P", 8, 300), ("M", 3, 300), ("A", 4, 400),
                                       ("V", 2, 700), ("F", 3, 400))],
        "project_gpu_hour_budget": {"systems": 5000, "screening": 15000, "scaling": 10000,
                                    "final": 12000, "evaluation": 5000, "reserve": 3000},
    }


def test_run_graph_conserves_budget_and_never_implies_execution(spec):
    plan = planner.build_plan(spec)
    assert plan["run_count"] == 40
    assert plan["screening_gpu_hour_ceiling"] == 15_000
    assert plan["phase_gpu_hour_ceilings"] == {"P": 4800, "M": 1800, "A": 3200, "V": 2800, "F": 2400}
    assert sum(plan["project_gpu_hour_budget"].values()) == 50_000
    assert sum(plan["phase_gpu_hour_ceilings"].values()) == plan["project_gpu_hour_budget"]["screening"]
    assert plan["completed_run_count"] == 0 and plan["actual_gpu_hours"] is None
    visited = set()
    for trial in plan["trials"]:
        assert trial["id"] not in visited
        assert set(trial["dependencies"]) <= visited  # Acyclic, earlier-stage dependencies.
        assert trial["launched"] is False and trial["completed"] is False
        assert trial["actual_gpu_hours"] is None and trial["benchmark_scores"] is None
        assert trial["accepted_training_episodes"] is None and trial["training_command"] is None
        assert trial["blockers"] and trial["status"] in {"ready_to_materialize", "pending_selection"}
        visited.add(trial["id"])
    assert len(visited) == 40


def test_fallback_and_independent_observation_crossproduct_conserve_mass(spec):
    counts = planner.build_plan(spec)["conditional_volume_examples"]["Q_observed"]
    before, after = counts["before_eligibility"], counts["after_eligibility"]
    assert before["hierarchy"] == 9_600_000
    assert after["hierarchy"] == pytest.approx(8_974_536.695092095)
    assert after["R"] == pytest.approx(32_000_000 + counts["hierarchy_fallback_to_R"])
    assert before["hierarchy"] - after["hierarchy"] == pytest.approx(counts["hierarchy_fallback_to_R"])
    assert sum(after.values()) == pytest.approx(64_000_000)
    cross = counts["mechanism_observation_crossproduct"]
    for family in planner.MECHANISMS:
        assert sum(cross[family].values()) == pytest.approx(after[family])
        assert cross[family]["identity"] == pytest.approx(after[family] * .5)
    for obs, weight in spec["observation_weights"].items():
        assert sum(cross[family][obs] for family in planner.MECHANISMS) == pytest.approx(64_000_000 * weight)
    assert sum(stage["conditional_episode_target"] for stage in counts["stages"]) == 64_000_000
    assert counts["realized_accepted_episodes"] is None and counts["raw_generation_attempts"] is None
    assert counts["observation_counts_are_requested_modes_before_shape_fallback"]
    assert counts["effective_observation_marginals"] is None


def test_single_interventions_and_grid_have_declared_doses(spec):
    plan = planner.build_plan(spec)
    arms = {trial["arm"]: trial for trial in plan["trials"] if trial["phase"] == "P"}
    for arm, family, dose in (("R_H15", "hierarchy", .15), ("R_F20", "forest", .2),
                              ("R_S10", "smooth_local", .1), ("R_I05", "sparse_interaction", .05)):
        assert arms[arm]["mechanism_weights"][family] == dose
        assert arms[arm]["mechanism_weights"]["R"] == 1 - dose
        assert arms[arm]["observation_weights"]["identity"] == 1
    assert arms["R_O"]["mechanism_weights"]["R"] == 1
    assert arms["Q_clean"]["observation_weights"]["identity"] == 1
    assert arms["Q_observed"]["observation_weights"] == spec["observation_weights"]
    for anchor in spec["mixture_anchor_grid"]:
        mix = planner.mixture_with_anchor(spec["mechanism_weights"], anchor)
        assert sum(mix.values()) == pytest.approx(1)
        assert mix["R"] == anchor
        assert mix["forest"] / (1 - anchor) == pytest.approx(.4)
        assert mix["hierarchy"] / (1 - anchor) == pytest.approx(.3)


def test_confirmation_seeds_fresh_and_finance_parent_shared(spec):
    trials = planner.build_plan(spec)["trials"]
    component_screen = {t["seed"] for t in trials if t["phase"] == "P"}
    mixture_screen = {t["seed"] for t in trials if t["phase"] == "M"}
    assert component_screen == {0, 1}
    assert mixture_screen == {2, 3}
    assert component_screen.isdisjoint(mixture_screen)
    discovery = {t["seed"] for t in trials if t["phase"] in {"P", "M", "A"}}
    confirmation = {t["seed"] for t in trials if t["phase"] == "V"}
    assert confirmation == {4, 5}
    assert discovery.isdisjoint(confirmation)
    finance = [t for t in trials if t["phase"] == "F"]
    assert {t["seed"] for t in finance} == {6, 7}
    assert len({t["shared_parent_checkpoint"] for t in finance}) == 1
    assert all(t["parent_checkpoint_path"] is None for t in finance)


def test_distinct_numeric_anchors_cannot_silently_merge_formatted_arms(spec):
    spec["mixture_anchor_grid"] = [.5, .5000000001, .5000000002]
    assert len(set(spec["mixture_anchor_grid"])) == 3
    with pytest.raises(ValueError, match="distinct formatted arm labels"):
        planner.build_plan(spec)


@pytest.mark.parametrize("bad", [True, False, -1, float("nan"), float("inf"), "0.5", None])
def test_invalid_mixture_numbers_rejected(spec, bad):
    spec["mechanism_weights"]["R"] = bad
    with pytest.raises(ValueError):
        planner.build_plan(spec)


@pytest.mark.parametrize("bad", [True, 0, -100, 1.5, 101])
def test_invalid_episode_counts_rejected(spec, bad):
    with pytest.raises(ValueError):
        planner.build_plan(spec, episodes=bad)


def test_unknown_categories_malformed_simplexes_and_budget_mismatch_rejected(spec):
    cases = []
    changed = copy.deepcopy(spec)
    changed["mechanism_weights"]["unknown"] = 0
    cases.append(changed)
    changed = copy.deepcopy(spec)
    changed["observation_weights"]["identity"] = .6
    cases.append(changed)
    changed = copy.deepcopy(spec)
    changed["project_gpu_hour_budget"]["screening"] += 1
    cases.append(changed)
    changed = copy.deepcopy(spec)
    changed["screening_allocations"][0]["gpu_hours_per_run"] = 150
    cases.append(changed)
    changed = copy.deepcopy(spec)
    changed["screening_allocations"][0]["arms"] = True
    cases.append(changed)
    changed = copy.deepcopy(spec)
    changed["screening_allocations"][1]["id"] = "P"
    cases.append(changed)
    changed = copy.deepcopy(spec)
    changed["mixture_anchor_grid"] = [.7, .7, .3]
    cases.append(changed)
    for invalid in cases:
        with pytest.raises(ValueError):
            planner.build_plan(invalid)


def test_cli_output_is_deterministic_and_does_not_mutate_spec(spec, tmp_path):
    source, output = tmp_path / "spec.json", tmp_path / "plan.json"
    source.write_text(json.dumps(spec))
    before = source.read_bytes()
    args = ["--spec", str(source), "--episodes", "1000", "--output", str(output)]
    planner.main(args)
    first = output.read_bytes()
    planner.main(args)
    assert output.read_bytes() == first and source.read_bytes() == before
    result = json.loads(first)
    assert result["not_a_training_launch_config"] and result["performance_guarantee"] is False
    assert result["runtime_candidate_configuration_changed"] is True
    assert result["source_spec_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["conditional_volume_examples"]["Q_clean"]["conditional_accepted_episode_target"] == 1000
