"""Pinned, unmodified TabICLv2 prior and task/shape-first mixture controls.

The reference's generation-time query-label filter, split repair, full-table
normalization, and constant-column removal are intentional. This adapter is not
a reproduction of the released model's training/codec. CPU generation only;
parallelize with processes, not threads. Use PYTHONHASHSEED=0 at process startup
for repeatable upstream set iteration as well as the supplied numerical seed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import types
from typing import Any

import numpy as np
import torch

from .schema import Episode
from .static_prior import _rng, generate_episode as generate_authored_episode

REFERENCE_REVISION = "0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3"
MANIFEST_SHA256 = "e53bca1f6aa7301b0fc0c04ffb0349c79bceeec1ec97da9725ca43b6d707e786"
ARMS = ("R", "R_P1_05", "R_P4_05")
_LOCK = threading.RLock()


@dataclass(frozen=True)
class ReferenceShape:
    """Explicit diagnostic shape; native stage laws are used when omitted.

    n_classes is an ex ante budget: the reference may realize fewer classes.
    None retains the native class draw (or no classes for regression).
    """
    n_support: int
    n_query: int
    n_features: int
    n_classes: int | None = None

    def __post_init__(self):
        for name in ("n_support", "n_query", "n_features"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_classes is not None and (isinstance(self.n_classes, bool) or
                not isinstance(self.n_classes, int) or self.n_classes < 2):
            raise ValueError("n_classes must be an integer >=2, or None")


@dataclass
class ReferenceBatch:
    episodes: list[Episode]
    audit: dict[str, Any]
    native: tuple


class ReferenceGenerationError(RuntimeError):
    """An aborted draw; audit includes failures, with no source substitution."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit

    def __reduce__(self):
        # ProcessPoolExecutor reconstructs exceptions from their constructor args.
        return type(self), (str(self), self.audit)


def verify_reference_sources(source_root: str | Path | None = None) -> Path:
    root = (Path(source_root) if source_root else
            Path(__file__).resolve().parents[2] / "third_party" / "tabicl_reference")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pinned reference source missing at {root}; use the repository checkout or source_root")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("Reference manifest differs from the reviewed pin")
    manifest = json.loads(raw)
    if manifest["revision"] != REFERENCE_REVISION:
        raise RuntimeError("Reference revision differs from the reviewed pin")
    for name, entry in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Modified or corrupt reference source: {name}")
    return root.resolve()


@contextmanager
def reference_runtime(seed: int, source_root: str | Path | None = None):
    """Load verified source without importing the optional inference stack.

    Upstream uses absolute `tabicl.prior` imports. Temporarily mount its package
    namespace and restore any previously installed TabICL modules afterward.
    This does not alter a source file. Other threads must not import/use TabICL
    or mutate global RNG state during generation; use worker processes instead.
    """
    with _LOCK:
        root = verify_reference_sources(source_root)
        prefix = lambda name: name == "tabicl" or name.startswith("tabicl.")
        saved_modules = {name: value for name, value in sys.modules.items() if prefix(name)}
        np_state, py_state = np.random.get_state(), random.getstate()
        threads = torch.get_num_threads()
        for name in saved_modules:
            del sys.modules[name]
        package = types.ModuleType("tabicl")
        package.__path__ = [str(root / "src" / "tabicl")]
        sys.modules["tabicl"] = package
        try:
            with torch.random.fork_rng(devices=[]):
                np.random.seed(int(seed) % 2**32)
                random.seed(int(seed))
                # CPU generator only: do not reset accelerator RNGs in a trainer.
                torch.random.default_generator.manual_seed(int(seed))
                torch.set_num_threads(1)
                try:
                    dataset = importlib.import_module("tabicl.prior._dataset")
                except ImportError as exc:
                    raise ImportError("The pinned prior needs scipy, scikit-learn, psutil, "
                                      "threadpoolctl and xgboost in the project environment") from exc
                with dataset.threadpoolctl.threadpool_limits(limits=1):
                    yield dataset
        finally:
            torch.set_num_threads(threads)
            np.random.set_state(np_state)
            random.setstate(py_state)
            for name in list(sys.modules):
                if prefix(name):
                    del sys.modules[name]
            sys.modules.update(saved_modules)


def reference_kwargs(stage: int, task: str, batch_size: int, envelope: dict | None = None) -> dict:
    """Released script settings, including its 1..100 rounded feature law."""
    if stage not in (1, 2, 3) or task not in ("classification", "regression"):
        raise ValueError("stage must be 1..3; task must be classification or regression")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    kwargs = dict(regression=task == "regression", batch_size=batch_size,
                batch_size_per_gp=4 if stage == 1 else 1,
                batch_size_per_subgp=4 if stage == 1 else 1,
                min_features=1, max_features=100, max_classes=10,
                min_seq_len=None if stage == 1 else 400,
                max_seq_len={1: 1024, 2: 10240, 3: 60000}[stage],
                log_seq_len=stage != 1, log_n_features=False,
                min_train_size=.3 if stage == 1 else .79,
                max_train_size=.9 if stage == 1 else .81,
                seq_len_per_gp=True, replay_small=False, n_jobs=1, device="cpu")
    if envelope is not None:
        if not isinstance(envelope, dict) or set(envelope) - {"min_features", "max_features", "max_classes"}:
            raise ValueError("envelope only supports min_features, max_features and max_classes")
        for key, value in envelope.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"envelope {key} must be a positive integer")
        kwargs.update(envelope)
        if kwargs["max_classes"] < 2 or kwargs["min_features"] > kwargs["max_features"]:
            raise ValueError("Invalid reference envelope feature/class range")
        # Native long-row caps can reduce max_features to 20 in stage 3.
        # A larger requested minimum would invert its native sampling interval.
        minimum_cap = {1: kwargs["max_features"], 2: kwargs["max_features"],
                       3: min(20, kwargs["max_features"])}[stage]
        if kwargs["min_features"] > minimum_cap:
            raise ValueError("min_features exceeds a native long-context width cap")
    return kwargs


def select_source(seed: int, index: int, arm: str, task: str, n_features: int) -> dict:
    """Branch after fixing native parameters; ineligible 5% mass returns to R."""
    if arm not in ARMS or task not in ("binary", "multiclass", "regression"):
        raise ValueError("Unknown reference arm or episode task")
    addition = {"R": None, "R_P1_05": "P1", "R_P4_05": "P4"}[arm]
    eligible = ((addition == "P1" and n_features >= 8) or
                (addition == "P4" and task in ("binary", "regression") and n_features >= 4))
    drawn = addition if addition and _rng(seed, "reference_mixture", index).random() < .05 else "R"
    selected = drawn if drawn == "R" or eligible else "R"
    return dict(branch_draw=drawn, selected_source=selected, eligible=bool(eligible),
                requested_addition_probability=0 if addition is None else .05,
                conditional_addition_probability=.05 if eligible else 0.,
                ineligible_mass_returned=drawn != selected)


@contextmanager
def _observe_native(dataset, prior, event, max_attempts):
    """Counting-only wrappers: preserve every native RNG call and filter result."""
    graph_module = importlib.import_module("tabicl.prior.graph_lib._dataset")
    old_graph, old_filter = dataset.GraphSCM, dataset.should_filter
    old_overlap = graph_module.check_x_y_ancestors_overlap
    old_unique, old_sanity = prior.delete_unique_features, prior.cls_sanity_check

    def graph(*args, **kwargs):
        if event["raw_dataset_attempts"] >= max_attempts:
            raise RuntimeError("Reference attempt limit exceeded; draw aborted without replacement")
        event["raw_dataset_attempts"] += 1
        return old_graph(*args, **kwargs)

    def filter_dataset(*args, **kwargs):
        result = old_filter(*args, **kwargs)
        event["predictability_rejections"] += int(result)
        return result

    def overlap(*args, **kwargs):
        result = old_overlap(*args, **kwargs)
        event["graph_proposals"] += 1
        event["graph_rejections"] += int(not result)
        return result

    def unique(*args, **kwargs):
        result = old_unique(*args, **kwargs)
        event["empty_feature_rejections"] += int(not (result[1] > 0).all())
        return result

    def sanity(*args, **kwargs):
        result = old_sanity(*args, **kwargs)
        event["class_split_rejections"] += int(not result)
        return result

    dataset.GraphSCM, dataset.should_filter = graph, filter_dataset
    graph_module.check_x_y_ancestors_overlap = overlap
    prior.delete_unique_features, prior.cls_sanity_check = unique, sanity
    try:
        yield
    finally:
        dataset.GraphSCM, dataset.should_filter = old_graph, old_filter
        graph_module.check_x_y_ancestors_overlap = old_overlap
        prior.delete_unique_features, prior.cls_sanity_check = old_unique, old_sanity


def generate_reference_batch(seed: int, *, arm: str = "R", task: str = "classification",
                             stage: int = 1, batch_size: int = 4,
                             shape: ReferenceShape | None = None,
                             envelope: dict | None = None,
                             source_root: str | Path | None = None,
                             max_reference_attempts: int = 1000,
                             source_callback=None) -> ReferenceBatch:
    """Run the native batch sampler, replacing eligible draws with 5% P1/P4.

    The native sampler draws ALL batch parameters before calling any generator.
    Shape overrides are diagnostics, explicitly recorded, never silently used
    by a curriculum. Filters retry inside the same selected reference slot;
    failures abort, never switch source. Accepted addition mass is 0.05 times
    eligibility, not an unconditional 5%; raw-reference proposal mass differs.

    ``source_callback(params, event)`` is an optional extension hook used by the
    challenger. It runs after native task/shape sampling, sets the selected
    source in ``event``, and returns an Episode or None for unmodified native R.
    Replacement episodes must preserve requested task, shape and vocabulary.
    """
    # A callback may supply a replacement only after native task/shape draws.
    # Returning None preserves the complete native generation/filter path.
    if source_callback is not None and arm != "R":
        raise ValueError("source_callback requires arm=R")
    kwargs = reference_kwargs(stage, task, batch_size, envelope=envelope)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if isinstance(max_reference_attempts, bool) or not isinstance(max_reference_attempts, int) or max_reference_attempts < 1:
        raise ValueError("max_reference_attempts must be a positive integer")
    if shape and task == "regression" and shape.n_classes is not None:
        raise ValueError("Regression has no classification budget")
    if shape and task == "classification" and min(shape.n_support, shape.n_query) < 2:
        raise ValueError("Native classification split repair requires >=2 rows on each side")
    if shape:
        kwargs.update(min_seq_len=None, max_seq_len=shape.n_support + shape.n_query,
                      min_train_size=shape.n_support, max_train_size=shape.n_support + 1,
                      min_features=shape.n_features, max_features=shape.n_features)
        # Keep exact diagnostic width even above the native long-sequence cap.
        kwargs["seq_len_per_gp"] = False
    audit = dict(arm=arm, seed=int(seed), reference_revision=REFERENCE_REVISION,
                 stage=stage, package_task=task, native_filters_preserved=True,
                 shape_override=None if shape is None else dict(vars(shape)),
                 envelope_override=envelope,
                 reference_distribution_modified=bool(envelope) or shape is not None,
                 resolved_prior_kwargs=dict(kwargs),
                 python_hash_seed=os.environ.get("PYTHONHASHSEED"), events=[],
                 semantics="Native reference generation; authored additions unfiltered; diagnostic Episode codec is separate")
    episodes = []
    with reference_runtime(seed, source_root) as dataset:
        config = dataset.PriorConfig(filter_unpredictable_graphs=True,
                                     filter_unpredictable_datasets=True)

        class ControlledPrior(dataset.GraphPrior):
            def generate_dataset(self, params):
                params = dict(params)
                index = len(audit["events"])
                if shape and shape.n_classes is not None:
                    params["num_classes"] = shape.n_classes
                classes = 0 if self.regression else int(params["num_classes"])
                episode_task = "regression" if self.regression else "binary" if classes == 2 else "multiclass"
                event = dict(index=index, task=episode_task, n_classes=classes,
                             n_support=int(params["train_size"]),
                             n_query=int(params["seq_len"] - params["train_size"]),
                             requested_features=int(params["num_features"]),
                             raw_dataset_attempts=0, graph_proposals=0, graph_rejections=0,
                             predictability_rejections=0, empty_feature_rejections=0,
                             class_split_rejections=0, accepted=False,
                             **select_source(seed, index, arm, episode_task, params["num_features"]))
                audit["events"].append(event)
                selected = event["selected_source"]
                try:
                    replacement = None
                    if source_callback is not None:
                        replacement = source_callback(dict(params), event)
                        selected = event["selected_source"]
                    if selected == "R":
                        with _observe_native(dataset, self, event, max_reference_attempts):
                            X, y, d = super().generate_dataset(params)
                        ns, nf = event["n_support"], int(d)
                        # Native GraphSCM already ordinal-encodes/scales categoricals,
                        # then discards their type identities. Do not guess them.
                        ep = Episode(X[:ns, :nf].numpy(), y[:ns].numpy(),
                                     X[ns:, :nf].numpy(), y[ns:].numpy(),
                                     np.zeros(nf, dtype=bool), episode_task, classes,
                                     metadata={"family": "R", "query_acceptance_used": True,
                                               "native_numeric_representation": True,
                                               "class_universe_declared_before_rows": True},
                                     encoding_seed=int(seed))
                    else:
                        if source_callback is not None:
                            if not isinstance(replacement, Episode):
                                raise TypeError("source_callback must return an Episode for a replacement source")
                            ep = replacement
                            if (ep.x_support.shape != (event["n_support"], event["requested_features"]) or
                                    ep.x_query.shape != (event["n_query"], event["requested_features"]) or
                                    ep.task != episode_task or ep.n_classes != classes):
                                raise ValueError("source_callback changed the sampled task/shape")
                        else:
                            authored_seed = int(_rng(seed, "reference_addition", index).integers(0, 2**63))
                            ep = generate_authored_episode(authored_seed, family=selected, task=episode_task,
                                                          n_support=event["n_support"], n_query=event["n_query"],
                                                          n_features=event["requested_features"],
                                                          n_classes=classes, stage=stage)
                            event["raw_dataset_attempts"] = 1
                        X = torch.zeros((params["seq_len"], params["max_features"]), dtype=torch.float32)
                        X[:, :ep.x_support.shape[1]] = torch.from_numpy(np.concatenate([ep.x_support, ep.x_query])).float()
                        y = torch.from_numpy(np.concatenate([ep.y_support, ep.y_query])).to(
                            torch.float32 if self.regression else torch.int64)
                        d = torch.tensor(ep.x_support.shape[1], dtype=torch.int64)
                    event.update(accepted=True, accepted_features=int(d),
                                 observed_classes=None if self.regression else np.unique(np.r_[ep.y_support, ep.y_query]).tolist())
                    ep.metadata.update(reference_revision=REFERENCE_REVISION, reference_control=dict(event),
                                       native_reference_filter=selected == "R",
                                       reference_distribution_modified=audit["reference_distribution_modified"],
                                       reference_envelope=envelope,
                                       reference_shape_override=audit["shape_override"])
                    episodes.append(ep)
                    return X, y, d
                except Exception as exc:
                    event["error"] = f"{type(exc).__name__}: {exc}"
                    raise ReferenceGenerationError(event["error"], audit) from exc

        prior = ControlledPrior(config=config, **kwargs)
        native = prior.get_batch()
    sources = tuple(dict.fromkeys(("R", "P1", "P4",
                                  *(e["selected_source"] for e in audit["events"]),
                                  *(e["branch_draw"] for e in audit["events"]))))
    for name, field in (("branch_counts", "branch_draw"), ("selected_counts", "selected_source")):
        audit[name] = {s: sum(e[field] == s for e in audit["events"]) for s in sources}
    audit["accepted_counts"] = {s: sum(e["accepted"] and e["selected_source"] == s for e in audit["events"])
                                for s in sources}
    audit["raw_attempt_counts"] = {s: sum(e["raw_dataset_attempts"] for e in audit["events"] if e["selected_source"] == s)
                                   for s in sources}
    audit["accepted_source_mass"] = {s: count / len(episodes) for s, count in audit["accepted_counts"].items()}
    audit["ineligible_returns"] = sum(e["ineligible_mass_returned"] for e in audit["events"])
    return ReferenceBatch(episodes=episodes, audit=audit, native=native)


def generate_reference_episode(seed: int, **kwargs) -> Episode:
    """Single-episode convenience API; full accounting is in reference_control."""
    if "batch_size" in kwargs:
        raise ValueError("Use generate_reference_batch for batch_size")
    return generate_reference_batch(seed, batch_size=1, **kwargs).episodes[0]
