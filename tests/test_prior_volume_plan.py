"""Exact exposure arithmetic and measured-budget constraints, not performance tests."""
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plan_prior_volume.py"
_SPEC = importlib.util.spec_from_file_location("plan_prior_volume", _PATH)
planner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(planner)


def test_default_targets_preserve_fixed_project_and_final_budgets():
    plan = planner.build_plan()
    assert plan["project_gpu_hour_ceiling"] == 50_000
    assert sum(plan["project_phase_gpu_hour_ceilings"].values()) == 50_000
    assert plan["combined_conditional_final_gpu_hour_ceiling"] == 12_000
    standard, finance = plan["standard"], plan["finance"]
    assert standard["target_updates"] == 250_000
    assert standard["target_stage_updates"] == [225_000, 22_500, 2_500]
    assert standard["target_stage_accepted_macroepisodes"] == [57_600_000, 5_760_000, 640_000]
    assert finance["target_updates"] == 4_000
    assert finance["target_distinct_worlds"] == 256_000
    assert finance["target_distinct_worlds_by_stage"] == [230_400, 23_040, 2_560]
    assert standard["budget_feasible_horizon"] is None and finance["budget_feasible_horizon"] is None
    assert standard["target_required_gpu_hours"] is None


def test_required_rates_use_gpu_hours_not_wall_hours():
    plan = planner.build_plan()
    s, f = plan["standard"], plan["finance"]
    assert s["required_aggregate_macroepisodes_per_second"] == pytest.approx(126.4197530864)
    assert s["required_macroepisodes_per_gpu_second"] == pytest.approx(1.97530864198)
    assert s["maximum_weighted_mean_seconds_per_update"] == pytest.approx(2.025)
    assert f["required_aggregate_macroepisodes_per_second"] == pytest.approx(6.06814814815)
    assert f["required_macroepisodes_per_gpu_second"] == pytest.approx(.0948148148148)
    assert f["maximum_weighted_mean_seconds_per_update"] == pytest.approx(42.1875)
    with_overhead = planner.build_plan({"standard_overhead_gpu_hours": 900})["standard"]
    assert with_overhead["maximum_weighted_mean_seconds_per_update"] == pytest.approx(2.025 * .9)


def test_mixture_expectations_and_envelope_overlap_conserve_episode_mass():
    counts = planner.build_plan()["standard"]["target_source_task_envelope_counts"]
    assert counts["source_branch_expectation_before_eligibility"] == {"R": 60_800_000, "P1": 3_200_000, "P4": 0}
    assert counts["accepted_source_conditional_expectation"] is None
    assert counts["envelope_extension_expectation"] == 1_280_000
    assert counts["envelope_and_P1_branch_overlap_expectation_before_eligibility"] == 64_000
    tasks = counts["task_expectation_with_planned_envelope"]
    assert tasks["classification"] == 32_000_000 and tasks["regression"] == 32_000_000
    assert tasks["binary_class_budget"] + tasks["multiclass_class_budget"] == pytest.approx(32_000_000)
    assert tasks["binary_class_budget"] < counts["strict_native_without_envelope_task_expectation"]["binary_class_budget"]
    fixed_eligibility = planner.standard_counts([1000, 2000, 3000], [800, 1600, 0])
    assert fixed_eligibility["accepted_source_conditional_expectation"] == {"R": 5880, "P1": 120, "P4": 0}
    assert fixed_eligibility["source_branch_expectation_before_eligibility"]["P1"] == 300


def test_eligibility_adjusted_count_uses_rounded_widths_and_native_integer_length_caps():
    assert planner.requested_width_eligibility(7) == 0
    assert planner.requested_width_eligibility(8) == pytest.approx(1 / 14)
    counts = planner.build_plan()["standard"]["target_source_task_envelope_counts"]
    stages = counts["stages"]
    assert stages[0]["eligibility_adjusted_source_expectation"]["P1"] == pytest.approx(2_690_909.090909091)
    probabilities = planner.stage_p1_eligibility()
    assert 0 < probabilities[2] < probabilities[0] < probabilities[1] < 1
    adjusted = counts["eligibility_adjusted_source_expectation"]
    assert adjusted["R"] + adjusted["P1"] == 64_000_000
    assert adjusted["P1"] < counts["source_branch_expectation_before_eligibility"]["P1"]
    # Independently sample the published native length/rounded-width process.
    import numpy as np
    rng = np.random.default_rng(548)
    lengths = np.exp(rng.uniform(np.log(400), np.log(60000), 250_000)).astype(int)
    ceiling = np.where(rng.random(len(lengths)) < .2, 512, 100)
    for lower, cap in ((10240, 80), (20000, 60), (30000, 40), (40000, 30), (50000, 20)):
        ceiling = np.where(lengths > lower, np.minimum(ceiling, cap), ceiling)
    widths = np.rint(1 + (ceiling - 1) * rng.random(len(lengths)))
    assert np.mean(widths >= 8) == pytest.approx(probabilities[2], abs=.002)


def test_exact_budget_boundary_fits_without_overspending():
    plan = planner.build_plan({"standard_seconds_per_update": [2.025] * 3,
                               "finance_seconds_per_update": [42.1875] * 3})
    for name, expected_updates, cap in (("standard", 250000, 9000), ("finance", 4000, 3000)):
        phase = plan[name]
        assert phase["target_fits_measured_budget"]
        assert phase["target_required_gpu_hours"] == cap
        assert phase["budget_feasible_horizon"]["updates"] == expected_updates
        assert phase["budget_feasible_horizon"]["projected_total_gpu_hours"] == cap


def test_slow_measurements_shrink_whole_batches_stages_and_finance_reuse():
    plan = planner.build_plan({"standard_seconds_per_update": [3, 10, 50],
                               "finance_seconds_per_update": [40, 100, 300],
                               "standard_overhead_gpu_hours": 100,
                               "finance_overhead_gpu_hours": 100})
    for name, quantum in (("standard", 100), ("finance", 400)):
        phase = plan[name]
        feasible = phase["budget_feasible_horizon"]
        assert not phase["target_fits_measured_budget"]
        assert feasible["updates"] < phase["target_updates"]
        assert feasible["updates"] % quantum == 0
        assert feasible["stage_updates"] == planner.stage_updates(feasible["updates"])
        assert feasible["accepted_macroepisodes"] == feasible["updates"] * 256
        assert feasible["projected_total_gpu_hours"] <= phase["gpu_hour_ceiling"]
        # One more permitted block would violate the cap; the rounded horizon is maximal.
        block_seconds = sum(n * t for n, t in zip(planner.stage_updates(quantum), phase["measured_seconds_per_update"]))
        assert feasible["projected_total_gpu_hours"] + block_seconds * 64 / 3600 > phase["gpu_hour_ceiling"]
    finance = plan["finance"]["budget_feasible_horizon"]
    assert all(end % 4 == 0 for end in finance["stage_end_steps"])
    assert finance["distinct_worlds"] == finance["accepted_macroepisodes"] // 4


def test_unaligned_stage_transitions_create_extra_stage_specific_worlds():
    counts = planner.finance_world_counts([90, 9, 1])
    assert sum(counts) == 27 * 256
    assert sum(counts) > 100 * 256 // 4
    aligned = planner.finance_world_counts([360, 36, 4])
    assert sum(aligned) == 400 * 256 // 4


def test_no_complete_cycle_fits_is_explicit_zero_not_illegal_partial_stage():
    plan = planner.plan_horizon(25_600, 1, seconds_per_update=[1000] * 3)
    feasible = plan["budget_feasible_horizon"]
    assert feasible["updates"] == 0 and feasible["stage_updates"] == [0, 0, 0]
    assert feasible["status"] == "no_complete_stage_cycle_fits"


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_measurements_are_rejected(bad):
    with pytest.raises(ValueError):
        planner.build_plan({"standard_seconds_per_update": [1, 1, bad]})


def test_malformed_shapes_mismatched_hardware_and_overspent_overhead_rejected():
    for data in ({"standard_seconds_per_update": [1, 2]}, {"allocated_gpus": 32},
                 {"global_batch": 64}, {"seconds_per_update": [1, 2, 3]},
                 {"standard_overhead_gpu_hours": 9001}):
        with pytest.raises(ValueError):
            planner.build_plan(data)
    with pytest.raises(ValueError):
        planner.plan_horizon(25_601, 10)
    with pytest.raises(ValueError):
        planner.plan_horizon(25_600, 10, reuse_updates=4)
    with pytest.raises(ValueError):
        planner.standard_counts([10, 20, 30], [11, 0, 0])


def test_cli_writes_unmeasured_and_measured_reviewable_artifacts(tmp_path):
    output = tmp_path / "volume.json"
    planner.main(["--output", str(output)])
    assert json.loads(output.read_text())["standard"]["budget_feasible_horizon"] is None
    measurements = tmp_path / "measured.json"
    measurements.write_text(json.dumps({"allocated_gpus": 64, "global_batch": 256,
        "measurement_id": "synthetic arithmetic fixture, not a GPU result",
        "standard_seconds_per_update": [3, 10, 50], "finance_seconds_per_update": [40, 100, 300]}))
    planner.main(["--output", str(output), "--measurements", str(measurements)])
    report = json.loads(output.read_text())
    assert report["not_a_training_launch_config"]
    assert report["standard"]["budget_feasible_horizon"]["status"] == "shrink_horizon"
