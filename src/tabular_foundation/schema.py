"""Episode boundary separating observable model inputs from loss-only state."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Episode:
    x_support: np.ndarray
    y_support: np.ndarray
    x_query: np.ndarray
    y_query: np.ndarray
    categorical: np.ndarray
    task: str
    n_classes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    query_weights: np.ndarray | None = None
    finance_metadata: np.ndarray | None = None
    source_bits: np.ndarray | None = None
    reference_probs: np.ndarray | None = None
    hurdle: bool = False
    codec_indices: np.ndarray | None = None
    encoding_seed: int = 0

    def __post_init__(self) -> None:
        self.x_support = np.asarray(self.x_support, dtype=np.float64)
        self.x_query = np.asarray(self.x_query, dtype=np.float64)
        self.y_support = np.asarray(self.y_support)
        self.y_query = np.asarray(self.y_query)
        self.categorical = np.asarray(self.categorical, dtype=bool)
        if self.x_support.ndim != 2 or self.x_query.ndim != 2:
            raise ValueError("Support and query features must be matrices")
        if self.x_support.shape[1] != self.x_query.shape[1]:
            raise ValueError("Support and query widths differ")
        if self.categorical.shape != (self.x_support.shape[1],):
            raise ValueError("categorical must contain one flag per column")
        if self.y_support.shape != (len(self.x_support),) or self.y_query.shape != (len(self.x_query),):
            raise ValueError("Targets must contain one value per row")
        if self.task not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"Unknown task {self.task!r}")
        if self.query_weights is None:
            self.query_weights = np.ones(len(self.y_query), dtype=np.float64)
        else:
            self.query_weights = np.asarray(self.query_weights, dtype=np.float64)
        if self.query_weights.shape != self.y_query.shape or not np.all(np.isfinite(self.query_weights)) or np.any(self.query_weights < 0):
            raise ValueError("Invalid query importance weights")
        if np.any(np.isinf(self.x_support)) or np.any(np.isinf(self.x_query)):
            raise FloatingPointError("Infinite feature value")
        if not np.all(np.isfinite(self.y_support)) or not np.all(np.isfinite(self.y_query)):
            raise FloatingPointError("Nonfinite target")
        if self.task != "regression":
            if self.n_classes < 2:
                raise ValueError("Classification requires an ex ante class universe")
            for y in (self.y_support, self.y_query):
                if np.any(y != np.floor(y)) or np.any(y < 0) or np.any(y >= self.n_classes):
                    raise ValueError("Target outside declared class universe")
        if self.finance_metadata is not None:
            self.finance_metadata = np.asarray(self.finance_metadata, dtype=np.float64)
            if self.finance_metadata.shape != (14,) or not np.all(np.isfinite(self.finance_metadata)):
                raise ValueError("finance_metadata must contain 14 finite observable values")
        if self.source_bits is not None:
            self.source_bits = np.asarray(self.source_bits, dtype=np.float64)
            if self.source_bits.shape != (len(self.x_support), 3) or not np.all((self.source_bits == 0) | (self.source_bits == 1)):
                raise ValueError("source_bits must be binary with shape (support rows, 3)")
        if self.reference_probs is not None:
            self.reference_probs = np.asarray(self.reference_probs, dtype=np.float64)
            if self.reference_probs.shape != (self.n_classes,) or np.any(self.reference_probs <= 0) or not np.all(np.isfinite(self.reference_probs)) or not np.isclose(self.reference_probs.sum(), 1):
                raise ValueError("reference_probs must be a positive normalized legal class reference")
        if self.codec_indices is not None:
            self.codec_indices = np.asarray(self.codec_indices, dtype=np.int64)
            if self.codec_indices.ndim != 1 or np.any(self.codec_indices < 0) or np.any(self.codec_indices >= len(self.x_support)):
                raise ValueError("Invalid support-only codec reservoir indices")

    def model_inputs(self) -> dict[str, Any]:
        """Explicit allowlist: hidden metadata and query outcomes never enter forward."""
        inputs = {"x_support": self.x_support, "y_support": self.y_support,
                  "x_query": self.x_query, "categorical": self.categorical,
                  "task": self.task, "n_classes": self.n_classes}
        if self.finance_metadata is not None:
            inputs["finance_metadata"] = self.finance_metadata
        if self.source_bits is not None:
            inputs["source_bits"] = self.source_bits
        if self.reference_probs is not None:
            inputs["reference_probs"] = self.reference_probs
        if self.codec_indices is not None:
            inputs["codec_indices"] = self.codec_indices
        inputs["hurdle"] = self.hurdle
        inputs["encoding_seed"] = self.encoding_seed
        return inputs
