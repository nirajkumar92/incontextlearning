"""Optional AutoGluon/TabArena adapter for one frozen checkpoint.

Import requires the separate TabArena environment described in docs/tabarena.md.
The upstream API was source-reviewed; no official TabArena run was executed in
the development environment. SharedWeights is intentionally disabled: every fit
owns its network and context, and serialization retains only CPU support data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

try:
    import pandas as pd
    from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
    from autogluon.core.utils.exceptions import TimeLimitExceeded
except ImportError as error:
    raise ImportError(
        "The optional TabArena adapter needs the pinned AutoGluon/TabArena environment. "
        "See docs/tabarena.md; it is not a dependency of synthetic pretraining."
    ) from error


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_key(value):
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, bool, int, float)):
        raise TypeError(f"Unsupported nominal scalar {type(value).__name__}; prepare a stable categorical column upstream")
    return type(value).__name__, repr(value)


@dataclass
class _FrameCodec:
    names: list
    categorical: np.ndarray
    mappings: list

    @classmethod
    def fit(cls, frame):
        if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
            raise ValueError("A DataFrame with distinct column names is required")
        mappings, categorical = [], []
        for name in frame.columns:
            series = frame[name]
            cat = (isinstance(series.dtype, pd.CategoricalDtype) or
                   pd.api.types.is_object_dtype(series.dtype) or
                   pd.api.types.is_string_dtype(series.dtype) or
                   pd.api.types.is_bool_dtype(series.dtype))
            categorical.append(cat)
            if cat:
                # Use observed training values, not .cat.categories: a pandas
                # categorical vocabulary can contain values absent from this fold.
                keys = [_category_key(v) for v in series[~series.isna()].tolist()]
                mapping = {}
                for key in keys:
                    if key not in mapping:
                        mapping[key] = len(mapping)
                mappings.append(mapping)
            else:
                if not pd.api.types.is_numeric_dtype(series.dtype):
                    raise TypeError(f"Feature {name!r} requires an upstream numeric/category representation")
                mappings.append(None)
        return cls(list(frame.columns), np.asarray(categorical, dtype=bool), mappings)

    def transform(self, frame):
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != self.names:
            raise ValueError("Prediction feature names/order differ from the fitted training frame")
        matrix = np.empty((len(frame), len(self.names)), dtype=np.float64)
        for j, name in enumerate(self.names):
            series, mapping = frame[name], self.mappings[j]
            if mapping is None:
                matrix[:, j] = series.to_numpy(dtype=np.float64, na_value=np.nan)
                if np.isinf(matrix[:, j]).any():
                    raise ValueError(f"Infinite numeric feature {name!r}")
                continue
            missing = series.isna().to_numpy()
            values = np.full(len(series), np.nan)
            for row in np.flatnonzero(~missing):
                key = _category_key(series.iloc[row])
                if key in mapping:
                    values[row] = mapping[key]
                else:
                    # Unseen nominal identities remain distinct and stable across
                    # query batches without fitting a query-dependent vocabulary.
                    digest = hashlib.sha256(repr(key).encode("utf8")).digest()
                    code = int.from_bytes(digest[:8], "little") % (2 ** 51)
                    values[row] = 2 ** 52 + code
            matrix[:, j] = values
        return matrix


@dataclass
class _StoredContext:
    x: np.ndarray
    y: np.ndarray
    categorical: np.ndarray
    task: str
    n_classes: int
    checkpoint_path: str
    checkpoint_sha256: str
    query_batch_size: int
    encoding_seed: int
    bf16: bool


class FrozenTabularFoundationModel(AbstractTorchModel):
    """One frozen ICL estimator; TabArena controls validation and bagging."""

    ag_key = "TFM-RESEARCH"
    ag_name = "TFM-Research-Frozen"
    seed_name = "random_state"
    default_num_gpus = 1
    minimum_num_gpus = 0
    gpu_strongly_recommended = True
    _supported_problem_types = ["binary", "multiclass", "regression"]
    shared_weights = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._frame_codec = None
        self._predictor = None

    def _set_default_params(self):
        defaults = {"checkpoint_path": None, "checkpoint_sha256": None,
                    "context_limit": 32768, "query_batch_size": 512,
                    "random_state": 0, "bf16": True}
        for name, value in defaults.items():
            self._set_default_param_value(name, value)

    def _get_default_auxiliary_params(self):
        values = super()._get_default_auxiliary_params()
        values.update({"valid_raw_types": ["int", "float", "category", "bool", "object"]})
        return values

    def _fit(self, X, y, num_cpus=1, num_gpus=0, time_limit=None, sample_weight=None, **kwargs):
        import torch
        started = time.perf_counter()
        if sample_weight is not None:
            raise ValueError("This frozen ICL adapter does not implement weighted support labels")
        parameters = self._get_model_params()
        checkpoint = parameters["checkpoint_path"]
        if checkpoint is None:
            raise ValueError("Provide checkpoint_path explicitly; no weights are downloaded or trained by this adapter")
        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        limit = int(parameters["context_limit"])
        if limit < 1 or len(X) > limit:
            raise ValueError(f"Training fold has {len(X)} rows, exceeding declared context_limit={limit}; no silent sampling is permitted")
        if time_limit is not None and time_limit <= 0:
            raise TimeLimitExceeded
        if not 0 <= num_gpus <= 1:
            raise ValueError("Each frozen estimator consumes zero or one allocated GPU")
        self.device = self._resolve_fit_device(num_gpus=num_gpus)
        torch.set_num_threads(max(1, int(num_cpus)))
        frame = self.preprocess(X)
        self._frame_codec = _FrameCodec.fit(frame)
        matrix = self._frame_codec.transform(frame)
        target = np.asarray(y)
        if target.shape != (len(matrix),) or not np.isfinite(target).all():
            raise ValueError("Finite training labels must align with rows")
        if self.problem_type == "regression":
            classes = 0
            target = target.astype(np.float64)
        else:
            classes = 2 if self.problem_type == "binary" else self.num_classes
            if classes is None or int(classes) < 2:
                raise ValueError("AutoGluon must declare the class universe from permitted training data")
            classes = int(classes)
            if np.any(target != np.floor(target)) or np.any(target < 0) or np.any(target >= classes):
                raise ValueError("AutoGluon-encoded training labels must be integers in the declared class universe")
            target = target.astype(np.int64)
        digest = checkpoint_sha256(checkpoint)
        expected = parameters["checkpoint_sha256"]
        if expected is not None and digest != expected:
            raise ValueError("Checkpoint digest differs from the declared experiment")
        self.model = _StoredContext(matrix, target, self._frame_codec.categorical.copy(),
                                   self.problem_type, classes, str(checkpoint), digest,
                                   int(parameters["query_batch_size"]), int(parameters["random_state"]),
                                   bool(parameters["bf16"]))
        self._predictor = None
        self._ensure_predictor(verified_checkpoint=True)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        if time_limit is not None and time.perf_counter() - started > time_limit:
            self._predictor, self.model = None, None
            raise TimeLimitExceeded

    def _ensure_predictor(self, verified_checkpoint=False):
        if self._predictor is not None:
            return
        if self.model is None:
            raise RuntimeError("Fit a legal training context first")
        from .inference import load_checkpoint, Predictor
        state = self.model
        if not verified_checkpoint and checkpoint_sha256(state.checkpoint_path) != state.checkpoint_sha256:
            raise ValueError("The persisted context's external checkpoint has changed")
        model, _ = load_checkpoint(state.checkpoint_path, device=self.device or "cpu")
        self._predictor = Predictor(model, query_batch_size=state.query_batch_size, bf16=state.bf16)
        self._predictor.fit_context(state.x, state.y, state.categorical, state.task,
                                    n_classes=state.n_classes, encoding_seed=state.encoding_seed)

    def _predict_proba(self, X, **kwargs):
        self._ensure_predictor()
        matrix = self._frame_codec.transform(self.preprocess(X, **kwargs))
        prediction = self._predictor.predict(matrix)
        return prediction[:, 1] if self.problem_type == "binary" else prediction

    def get_device(self):
        return str(self.device or "cpu")

    def _set_device(self, device):
        # A KV context records its owning model identity. Move by rebuilding
        # lazily; neither copying the old context nor unpickling it is valid.
        self.device = str(device)
        self._predictor = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_predictor"] = None
        return state

    @classmethod
    def _class_tags(cls):
        tags = super()._class_tags()
        tags.update({"pickles_pretrained_weights": False, "set_device_on_save_to": None})
        return tags

    def _more_tags(self):
        return {"can_refit_full": True}

    def get_info(self, **kwargs):
        info = super().get_info(**kwargs)
        if self.model is not None:
            info.update({"frozen_checkpoint_sha256": self.model.checkpoint_sha256,
                         "frozen_context_rows": len(self.model.y), "shared_pretrained_weights": False,
                         "external_checkpoint_bytes": Path(self.model.checkpoint_path).stat().st_size,
                         "serialized_model_excludes_external_checkpoint": True})
        return info
