import copy
import json

import numpy as np
import pytest
import torch

from tabular_foundation.model import build_model
from tabular_foundation.selection_evaluation import (assert_disjoint_panels,
    evaluate_panel, paired_comparison, score_prediction, validate_panel)


def runs(offset, datasets_per_task=4):
    result = []
    for seed in [4, 5]:
        rows = []
        for task in ["binary", "multiclass", "regression"]:
            for i in range(datasets_per_task):
                # Two datasets share one lineage, so equal-dataset and
                # equal-lineage estimands differ in this test.
                rows.append(dict(task=task, dataset=f"{task}{i}", fold="0",
                                 lineage=f"{task}{min(i,2) if datasets_per_task == 4 else i}", loss=1. - offset * (i + 1),
                                 denominator=1., metric="test", support_sha256="s",
                                 query_sha256="q", status="completed"))
        result.append(dict(seed=seed, view_seed=0, checkpoint_sha256=f"{seed}_{offset}",
                           panel_sha256="panel", panel_role="confirmation", views=1,
                           bf16=False, context_limit=512, status="completed", results=rows,
                           query_batch_size=512, device="cpu", cache_policy="sequential_views",
                           class_slot_capacity=256,
                           hardware={"device_type": "cpu", "device_name": "test_cpu", "torch_version": "test",
                                     "cuda_runtime": None, "hip_runtime": None},
                           planned_splits=[[r["task"], r["dataset"], r["fold"]] for r in rows]))
    return result


def test_objective_and_cluster_bootstrap_do_not_equal_weight_lineage_means():
    result = paired_comparison(runs(0), runs(.1), repeats=300, min_lineages=3)
    assert result["improvement"] == pytest.approx(.25)
    assert result["improvement"] != pytest.approx((.1 + .2 + .35) / 3)
    assert result["passes_statistical_rule"] and not result["confirmation_evidence"]
    assert not result["confirmation_protocol_requirements"]["at_least_10000_bootstrap_resamples"]
    assert not result["confirmation_protocol_requirements"]["at_least_10_source_lineages_per_task"]
    assert result["bootstrap"]["valid_repeats"] == 300
    assert result["seed_standard_deviation"] == 0
    same = paired_comparison(runs(0), runs(0), repeats=100, min_lineages=3)
    assert same["improvement"] == 0 and not same["passes_statistical_rule"]


@pytest.mark.parametrize("corrupt", ["failed", "missing", "duplicate_seed", "checkpoint", "denominator", "view", "query_batch", "device", "hardware", "cache_policy", "vocabulary_capacity"])
def test_comparison_rejects_unpaired_or_incomplete_evidence(corrupt):
    ref, cand = runs(0), runs(.1)
    if corrupt == "failed": cand[0]["status"] = "failed"
    if corrupt == "missing": cand[0]["results"].pop()
    if corrupt == "duplicate_seed": cand[1]["seed"] = cand[0]["seed"]
    if corrupt == "checkpoint": cand[1]["checkpoint_sha256"] = cand[0]["checkpoint_sha256"]
    if corrupt == "denominator": cand[0]["results"][0]["denominator"] = 2.
    if corrupt == "view": cand[0]["view_seed"] = 7
    if corrupt == "query_batch": cand[0]["query_batch_size"] = 1
    if corrupt == "device": cand[0]["device"] = "cuda:0"
    if corrupt == "hardware": cand[0]["hardware"]["device_name"] = "different_cpu"
    if corrupt == "cache_policy": cand[0]["cache_policy"] = "all_views_resident"
    if corrupt == "vocabulary_capacity": cand[0]["class_slot_capacity"] = 512
    with pytest.raises(ValueError):
        paired_comparison(ref, cand, repeats=5, min_lineages=3)


def test_training_only_denominator_and_affine_regression_score():
    ys = np.array([1., 3., 6., 10.]); yq = np.array([2., 8.]); pred = np.array([4., 7.])
    a = score_prediction("regression", ys, yq, pred)
    b = score_prediction("regression", 1e6+2*ys, 1e6+2*yq, 1e6+2*pred)
    assert a["loss"] == pytest.approx(b["loss"])
    assert a["denominator"] == pytest.approx(b["denominator"])
    c = score_prediction("binary", [0, 0, 0, 1], [0, 1], np.array([[.8,.2],[.4,.6]]), 2)
    assert c["denominator"] == pytest.approx(-(3*np.log(.7)+np.log(.3))/4)
    d = score_prediction("binary", [0, 0, 0, 1], [1, 1], np.array([[.8,.2],[.4,.6]]), 2)
    assert d["denominator"] == c["denominator"]


def test_panel_lineage_and_legal_vocabulary_validation():
    p = dict(role="development", datasets=[dict(dataset="a", lineage="source", fold="0",
             support="s.npz", query="q.npz", task="binary", classes=2)])
    validate_panel(p)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_panels(p, [dict(p, role="confirmation")])
    p["datasets"][0]["classes"] = 3
    with pytest.raises(ValueError, match="class count"):
        validate_panel(p)


def test_actual_checkpoint_panel_and_failure_record(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(20)
    model = build_model("persistent_tiny", finance=False)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": {"model": "persistent_tiny",
               "finance_adapter": False, "seed": 20}, "step": 0}, checkpoint)
    rng = np.random.default_rng(20)
    for split, n in [("s", 8), ("q", 3)]:
        np.savez(tmp_path/f"{split}.npz", X=rng.normal(size=(n, 3)),
                 y=np.arange(n)%2, categorical=np.zeros(3,dtype=bool),
                 ids=np.arange(n)+(0 if split=="s" else 100))
    panel = {"role":"development", "datasets":[dict(dataset="toy", lineage="toy_source", fold="0",
             task="binary", classes=2, support="s.npz",query="q.npz")]}
    path=tmp_path/"panel.json";path.write_text(json.dumps(panel))
    result=evaluate_panel(path, checkpoint, tmp_path/"out.json", views=4, bf16=False)
    assert result["status"] == "completed" and result["seed"] == 20
    assert result["results"][0]["cached_prediction_seconds"] > 0
    assert result["results"][0]["peak_allocated_bytes"] is None
    assert result["results"][0]["split_disjointness"]["method"] == "unique_disjoint_row_ids"
    assert result["cache_policy"] == "sequential_views"
    assert "torch_version" in result["hardware"] and "hip_runtime" in result["hardware"]
    with pytest.raises(RuntimeError, match="failed splits"):
        evaluate_panel(path, checkpoint, tmp_path/"fail.json", context_limit=4)
    failed=json.loads((tmp_path/"fail.json").read_text())
    assert failed["status"] == "failed" and failed["results"][0]["status"] == "failed"


def test_full_confirmation_protocol_requires_real_replication_and_10000_bootstraps():
    ref, cand = runs(0, datasets_per_task=10), runs(.01, datasets_per_task=10)
    # One resample cannot establish the declared confidence interval protocol.
    tiny = paired_comparison(ref, cand, repeats=1)
    assert tiny["passes_statistical_rule"] and not tiny["confirmation_evidence"]
    complete = paired_comparison(ref, cand, repeats=10000)
    assert complete["confirmation_evidence"]
    assert complete["unmet_confirmation_requirements"] == []
    for run in ref + cand:
        run["panel_role"] = "development"
    dev = paired_comparison(ref, cand, repeats=10)
    assert not dev["confirmation_evidence"]
    assert "confirmation_panel" in dev["unmet_confirmation_requirements"]


@pytest.mark.parametrize("field,new", [("support_sha256", "new support"), ("query_sha256", "new query"),
                                       ("denominator", 3.), ("metric", "different_metric")])
def test_paired_changes_of_data_between_seeds_are_rejected(field, new):
    ref, cand = runs(0), runs(.1)
    for group in (ref, cand):
        group[1]["results"][0][field] = new
    with pytest.raises(ValueError, match="across training seeds"):
        paired_comparison(ref, cand, repeats=5, min_lineages=3)


def test_task_classes_and_scored_probability_width_follow_the_legal_vocabulary():
    panel = dict(role="development", datasets=[dict(dataset="a", lineage="source", fold="0",
                 support="s.npz", query="q.npz", task="multiclass", classes=2)])
    with pytest.raises(ValueError, match="class count"):
        validate_panel(panel)
    with pytest.raises(ValueError, match="columns"):
        score_prediction("multiclass", [0, 1], [0, 1], np.array([[.8, .2], [.1, .9]]), 3)
    panel["datasets"][0]["classes"] = 3
    panel["datasets"].append(dict(panel["datasets"][0], fold="1", classes=4))
    with pytest.raises(ValueError, match="vocabulary"):
        validate_panel(panel)


def test_fold_and_seed_averages_precede_equal_dataset_weighting():
    ref, cand = runs(0), runs(.1)
    for seed_index in (0, 1):
        for task in ("binary", "multiclass", "regression"):
            r = next(row for row in ref[seed_index]["results"] if row["dataset"] == task + "0")
            c = next(row for row in cand[seed_index]["results"] if row["dataset"] == task + "0")
            # The first dataset receives a second fold, with improvement .9 or .5.
            new_r, new_c = dict(r, fold="1"), dict(c, fold="1", loss=.1 + .4 * seed_index)
            ref[seed_index]["results"].append(new_r)
            cand[seed_index]["results"].append(new_c)
            ref[seed_index]["planned_splits"].append([task, task + "0", "1"])
            cand[seed_index]["planned_splits"].append([task, task + "0", "1"])
    result = paired_comparison(ref, cand, repeats=10, min_lineages=3)
    # Dataset 0: mean(.1, .9, .1, .5)=.4; remaining datasets: .2/.3/.4.
    assert result["improvement"] == pytest.approx((.4 + .2 + .3 + .4) / 4)
    assert result["seed_improvements"]["4"] == pytest.approx((.5 + .2 + .3 + .4) / 4)
    assert result["seed_improvements"]["5"] == pytest.approx((.3 + .2 + .3 + .4) / 4)
    assert result["seed_standard_deviation"] == pytest.approx(np.std([.35, .30], ddof=1))


def test_global_lineage_bootstrap_preserves_cross_task_covariance():
    ref, cand = runs(0), runs(.1)
    for group in (ref, cand):
        for run in group:
            for row in run["results"]:
                # The same sources produce all three task types, with identical scores.
                row["lineage"] = "shared" + row["dataset"][-1]
    result = paired_comparison(ref, cand, repeats=300, min_lineages=3)
    for task in ("binary", "multiclass", "regression"):
        assert result["intervals"][task] == result["intervals"]["overall"]


def test_split_provenance_override_is_explicit_and_cannot_override_same_content(tmp_path):
    from tabular_foundation.selection_evaluation import _split_disjointness
    s = {"X": np.ones((2, 1)), "y": np.array([0, 1]), "categorical": np.array([False])}
    q = {"X": np.ones((3, 1)), "y": np.array([0, 1, 0]), "categorical": np.array([False])}
    sp, qp = tmp_path / "s.npz", tmp_path / "q.npz"
    row = {}
    with pytest.raises(ValueError, match="row IDs"):
        _split_disjointness(row, s, q, sp, qp, "s", "q")
    row.update(disjoint_split_verified=True, split_provenance="Immutable upstream split manifest SHA256: abc; row identities reviewed")
    audit = _split_disjointness(row, s, q, sp, qp, "s", "q")
    assert audit["method"] == "declared_split_provenance" and not audit["externally_verified"]
    assert audit["split_provenance"] == row["split_provenance"]
    with pytest.raises(ValueError, match="identical content"):
        _split_disjointness(row, s, q, sp, qp, "same", "same")
    with pytest.raises(ValueError, match="same path"):
        _split_disjointness(row, s, q, sp, sp, "s", "q")
    s["ids"], q["ids"] = np.array([1, 2]), np.array([2, 3, 4])
    with pytest.raises(ValueError, match="identities overlap"):
        _split_disjointness(row, s, q, sp, qp, "s", "q")


@pytest.mark.parametrize("override", [{"disjoint_split_verified": True}, {"split_provenance": "reviewed"},
                                       {"disjoint_split_verified": False, "split_provenance": "reviewed"},
                                       {"disjoint_split_verified": True, "split_provenance": " "}])
def test_invalid_split_override_declarations_fail_at_manifest_validation(override):
    row = dict(dataset="a", lineage="source", fold="0", support="s.npz", query="q.npz",
               task="binary", classes=2, **override)
    with pytest.raises(ValueError, match="split declaration"):
        validate_panel({"role": "development", "datasets": [row]})


def test_missing_row_identity_cannot_silently_bypass_split_overlap_check(tmp_path):
    from tabular_foundation.selection_evaluation import _split_disjointness
    s = {"X": np.ones((2, 1)), "y": np.array([0, 1]), "categorical": np.array([False]),
         "ids": np.array([1., np.nan])}
    q = {"X": np.ones((3, 1)), "y": np.array([0, 1, 0]), "categorical": np.array([False]),
         "ids": np.array([4., 5., np.nan])}
    with pytest.raises(ValueError, match="missing or nonfinite"):
        _split_disjointness({}, s, q, tmp_path / "s", tmp_path / "q", "a", "b")
