"""Contracts for actual pinned source, paired mixtures and native RNG semantics."""
import json
import random
import pickle
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
import shutil
import sys

import numpy as np
import pytest
import torch

from tabular_foundation.reference_prior import (
    ReferenceGenerationError, ReferenceShape, generate_reference_batch,
    reference_kwargs, reference_runtime, select_source, verify_reference_sources,
)
from tabular_foundation.static_prior import generate_episode


@pytest.fixture
def reference_dependencies():
    for name in ("psutil", "xgboost", "scipy", "sklearn", "threadpoolctl"):
        pytest.importorskip(name)


def test_reference_pin_detects_modified_source(tmp_path):
    source = verify_reference_sources()
    shutil.copytree(source, tmp_path / "pin")
    target = tmp_path / "pin" / "src/tabicl/prior/_graph_scm.py"
    target.write_text(target.read_text() + "\n# accidental change\n")
    with pytest.raises(RuntimeError, match="Modified or corrupt"):
        verify_reference_sources(tmp_path / "pin")


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("task,script_task", [("classification", "clf"), ("regression", "reg")])
def test_group_size_matches_released_stage_recipe(stage, task, script_task):
    recipe = (verify_reference_sources() / "scripts" / f"train_v2_{script_task}_stage{stage}.sh").read_text()
    group_size = int(re.search(r"--batch_size_per_gp\s+(\d+)", recipe).group(1))
    kwargs = reference_kwargs(stage, task, 8)
    assert kwargs["batch_size_per_gp"] == group_size
    # No subgroup override is supplied by these recipes; it defaults to the group.
    assert "--batch_size_per_subgp" not in recipe
    assert kwargs["batch_size_per_subgp"] == group_size


def test_reference_r_matches_unmodified_upstream_batch(reference_dependencies):
    # No diagnostic override: native task, grouped shape and filters all apply.
    actual = generate_reference_batch(731, arm="R", task="classification", batch_size=2)
    with reference_runtime(731) as native:
        config = native.PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True)
        expected = native.GraphPrior(config=config, **reference_kwargs(1, "classification", 2)).get_batch()
    for a, b in zip(actual.native, expected):
        if a.is_nested:
            for aa, bb in zip(a.unbind(), b.unbind()):
                assert torch.equal(aa, bb)
        else:
            assert torch.equal(a, b)
    for event, ep in zip(actual.audit["events"], actual.episodes):
        assert event["accepted"] and event["raw_dataset_attempts"] >= 1
        assert ep.metadata["query_acceptance_used"]
        assert "y_query" not in ep.model_inputs()
        assert "reference_control" not in ep.model_inputs()


def test_native_regression_and_rng_namespace_restoration(reference_dependencies):
    np.random.seed(202); random.seed(303); torch.manual_seed(404)
    expected = (np.random.rand(), random.random(), torch.rand(3))
    np.random.seed(202); random.seed(303); torch.manual_seed(404)
    sentinel = sys.modules.get("tabicl")
    a = generate_reference_batch(42, task="regression", shape=ReferenceShape(128, 64, 8), batch_size=1)
    b = generate_reference_batch(42, task="regression", shape=ReferenceShape(128, 64, 8), batch_size=1)
    assert np.array_equal(a.episodes[0].y_query, b.episodes[0].y_query)
    assert expected[0] == np.random.rand() and expected[1] == random.random()
    assert torch.equal(expected[2], torch.rand(3))
    assert sys.modules.get("tabicl") is sentinel
    ep = a.episodes[0]
    assert ep.task == "regression" and ep.n_classes == 0
    assert ep.x_support.shape[0] == 128 and ep.x_query.shape[0] == 64


@pytest.mark.parametrize("arm,task,width", [
    ("R_P1_05", "binary", 7), ("R_P1_05", "regression", 1),
    ("R_P4_05", "multiclass", 100), ("R_P4_05", "binary", 3),
])
def test_ineligible_mass_returns_to_reference(arm, task, width):
    record = select_source(20, 0, arm, task, width)
    assert record["branch_draw"] != "R"
    assert record["selected_source"] == "R" and record["ineligible_mass_returned"]
    assert record["conditional_addition_probability"] == 0


def test_five_percent_is_conditional_not_autonomous_task_mixture():
    # Branch choices are independent of the externally fixed task and shape.
    sources = [select_source(s, 0, "R_P4_05", "binary", 8) for s in range(10000)]
    assert .04 < np.mean([e["selected_source"] == "P4" for e in sources]) < .06
    for s in range(100):
        assert select_source(s, 0, "R_P4_05", "binary", 8)["branch_draw"] == select_source(
            s, 0, "R_P4_05", "regression", 8)["branch_draw"]


@pytest.mark.parametrize("arm,task,classes", [
    ("R_P1_05", "classification", 7), ("R_P4_05", "classification", 2),
    ("R_P4_05", "regression", None),
])
def test_additions_use_fixed_shape_and_class_budget(reference_dependencies, arm, task, classes):
    batch = generate_reference_batch(20, arm=arm, task=task, batch_size=1,
                                     shape=ReferenceShape(32, 16, 8, classes))
    event, ep = batch.audit["events"][0], batch.episodes[0]
    assert event["selected_source"] in ("P1", "P4")
    assert ep.x_support.shape == (32, 8) and ep.x_query.shape == (16, 8)
    assert ep.n_classes == (classes or 0)
    assert batch.audit["accepted_counts"][event["selected_source"]] == 1
    assert batch.audit["raw_attempt_counts"][event["selected_source"]] == 1
    assert not ep.metadata["native_reference_filter"]


def test_attempt_budget_aborts_without_source_switch(reference_dependencies, monkeypatch):
    from tabular_foundation import reference_prior as ref
    original = ref._observe_native
    from contextlib import contextmanager

    @contextmanager
    def rejecting_native(dataset, prior, event, max_attempts):
        old = dataset.should_filter
        dataset.should_filter = lambda *args, **kwargs: True
        try:
            with original(dataset, prior, event, max_attempts):
                yield
        finally:
            dataset.should_filter = old

    monkeypatch.setattr(ref, "_observe_native", rejecting_native)
    with pytest.raises(ReferenceGenerationError) as exc:
        generate_reference_batch(42, task="regression", shape=ReferenceShape(32, 16, 8),
                                 batch_size=1, max_reference_attempts=1)
    event = exc.value.audit["events"][0]
    assert event["selected_source"] == "R" and not event["accepted"]
    assert event["raw_dataset_attempts"] == 1 and "attempt limit" in event["error"]


def test_authored_class_budget_override_rejects_task_mismatch():
    with pytest.raises(ValueError, match="disagrees"):
        generate_episode(0, family="P1", task="binary", n_support=16, n_query=8, n_features=8, n_classes=5)
    with pytest.raises(ValueError, match="integer"):
        generate_episode(0, family="P1", task="binary", n_support=16, n_query=8, n_features=8, n_classes=2.0)


def _raise_reference_error():
    raise ReferenceGenerationError("native failure", {"events": [{"accepted": False, "raw_dataset_attempts": 2}]})


def test_failure_audit_survives_pickle_and_processpool():
    original = ReferenceGenerationError("native failure", {"attempts": 2})
    restored = pickle.loads(pickle.dumps(original))
    assert str(restored) == str(original) and restored.audit == original.audit
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        with pytest.raises(ReferenceGenerationError, match="native failure") as exc:
            executor.submit(_raise_reference_error).result(timeout=30)
    assert exc.value.audit["events"][0]["raw_dataset_attempts"] == 2


def test_modified_envelope_is_explicit_and_invalid_caps_fail():
    native = reference_kwargs(1, "classification", 1)
    candidate = reference_kwargs(1, "classification", 1, {"max_classes": 256, "max_features": 1024})
    assert native["max_classes"] == 10 and native["max_features"] == 100
    assert candidate["max_classes"] == 256 and candidate["max_features"] == 1024
    with pytest.raises(ValueError, match="only supports"):
        reference_kwargs(1, "classification", 1, {"filter_unpredictable_datasets": False})
    with pytest.raises(ValueError, match="width cap"):
        reference_kwargs(3, "classification", 1, {"min_features": 21, "max_features": 100})


def test_paired_arms_draw_same_native_parameters_before_branches(reference_dependencies):
    baseline = generate_reference_batch(20, arm="R", batch_size=4)
    mixture = generate_reference_batch(20, arm="R_P1_05", batch_size=4)
    for a, b in zip(baseline.audit["events"], mixture.audit["events"]):
        for key in ("task", "n_classes", "n_support", "n_query", "requested_features"):
            assert a[key] == b[key]
    assert mixture.audit["events"][0]["selected_source"] == "P1"
