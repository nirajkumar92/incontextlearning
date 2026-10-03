"""Support-fitted, deterministic raw-table codec; query labels are not an input."""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def robust_location_scale(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return 0., 1.
    ordered = np.sort(a)
    middle = len(a) // 2
    # numpy's even-length median can overflow when two finite large values
    # are added. Halving before addition preserves a finite midpoint.
    median = float(ordered[middle] if len(a) % 2 else
                   .5 * ordered[middle - 1] + .5 * ordered[middle])
    magnitude = float(np.max(np.abs(a)))
    if magnitude == 0:
        return median, 1.
    unit = a / magnitude
    unit_mad = float(np.median(np.abs(unit - median / magnitude)))
    mad = min(unit_mad, 1.) * magnitude
    # Keep statistics finite even for values near float64's representable
    # limit. This affects only overflow cases, not ordinary table scaling.
    factor = 1.4826 * unit_mad if mad > 1e-12 else float(np.std(unit))
    scale = (min(factor, np.finfo(np.float64).max / magnitude) * magnitude
             if magnitude > 1 else factor * magnitude)
    if scale <= 1e-12:
        scale = 1.
    return median, max(scale, 1e-3)


def _missing(v):
    if v is None:
        return True
    return isinstance(v, (float, np.floating)) and math.isnan(float(v))


def _canonical(v) -> bytes:
    if isinstance(v, (bool, np.bool_)):
        return b"bool:" + (b"1" if v else b"0")
    if isinstance(v, (int, np.integer)):
        return b"int:" + str(int(v)).encode()
    if isinstance(v, (float, np.floating)):
        return b"float:" + float(v).hex().encode()
    if isinstance(v, str):
        return b"str:" + v.encode("utf-8")
    if isinstance(v, bytes):
        return b"bytes:" + v
    raise TypeError(f"unsupported nominal value type {type(v).__name__}")


def _hash_digest(seed: int, column: Any, value: bytes) -> bytes:
    fields = [str(seed).encode(), _canonical(column), value]
    payload = b"".join(struct.pack(">Q", len(v)) + v for v in fields)
    return hashlib.sha256(payload).digest()


def _hash_code(seed: int, column: Any, value: bytes) -> float:
    bits = int.from_bytes(_hash_digest(seed, column, value)[:8], "big") >> 11
    return 2 * (bits / (1 << 53)) - 1


@dataclass
class EncodedTable:
    numeric: torch.Tensor  # N,F,8; nonnumeric columns are zero
    categorical: torch.Tensor  # N,F,4; numeric columns are zero
    categorical_mask: torch.Tensor  # F boolean
    categorical_identity: torch.Tensor | None = None  # N,F,64 exact signed bits


class TableCodec:
    """Fit on all support or an explicit natural-reservoir subset.

    The CPU/FP64 path is the audited reference, not a claim of GPU preprocessing
    throughput. Nominal columns may contain strings, typed numbers or None.
    """

    def __init__(self, encoding_seed=0, column_ids=None, categorical_encoding="scalar_fourier"):
        if categorical_encoding not in ("scalar_fourier", "hash_bits"):
            raise ValueError("categorical_encoding must be scalar_fourier or hash_bits")
        self.encoding_seed = int(encoding_seed)
        self.column_ids = column_ids
        self.categorical_encoding = categorical_encoding

    def fit(self, x_support, categorical, indices=None):
        x = as_numpy(x_support)
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("a table must be two-dimensional with at least one feature")
        self.n_features = x.shape[1]
        cat = as_numpy(categorical)
        if cat.dtype == bool and cat.shape == (self.n_features,):
            self.is_categorical = cat.astype(bool)
        else:
            self.is_categorical = np.zeros(self.n_features, dtype=bool)
            if cat.size:
                if cat.ndim != 1 or cat.dtype == bool or not np.issubdtype(cat.dtype, np.integer):
                    raise ValueError("categorical must be a boolean feature mask or integer column indices")
                ids = np.asarray(cat, dtype=np.int64)
                if (ids < 0).any() or (ids >= self.n_features).any():
                    raise ValueError("invalid categorical feature indices")
                self.is_categorical[ids] = True
        if indices is None:
            ids = np.arange(len(x))
        else:
            ids = as_numpy(indices)
            if ids.dtype == bool:
                if ids.shape != (len(x),):
                    raise ValueError("reservoir mask length mismatch")
                ids = np.flatnonzero(ids)
            elif ids.size and not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("codec indices must contain integer support indices")
            ids = np.asarray(ids, dtype=np.int64)
            if ids.ndim != 1 or (ids < 0).any() or (ids >= len(x)).any() or len(np.unique(ids)) != len(ids):
                raise ValueError("codec indices must be unique valid support indices")
        self.fit_indices = ids
        fit = x[ids]
        self.ids = list(range(self.n_features)) if self.column_ids is None else list(self.column_ids)
        if len(self.ids) != self.n_features:
            raise ValueError("column identity count mismatch")
        self.stats = []
        for j in range(self.n_features):
            if self.is_categorical[j]:
                counts = {}
                for v in fit[:, j]:
                    if not _missing(v):
                        key = _canonical(v)
                        counts[key] = counts.get(key, 0) + 1
                self.stats.append({"counts": counts, "denominator": max(len(fit), 1)})
            else:
                values = np.asarray(fit[:, j], dtype=np.float64)
                center, scale = robust_location_scale(values)
                self.stats.append({"center": center, "scale": scale, "sorted": np.sort(values[np.isfinite(values)])})
        for attempt in range(9):
            seed = self.encoding_seed + attempt
            ok = True
            for j, stat in enumerate(self.stats):
                if self.is_categorical[j]:
                    codes = [(_hash_digest(seed, self.ids[j], key)[:8] if self.categorical_encoding == "hash_bits"
                              else _hash_code(seed, self.ids[j], key)) for key in stat["counts"]]
                    if len(set(codes)) != len(codes):
                        ok = False
                        break
            if ok:
                self.effective_seed = seed
                break
        else:
            raise RuntimeError("nominal hash collisions persisted after eight redraws")
        return self

    def transform(self, x, device=None) -> EncodedTable:
        x = as_numpy(x)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("query feature count does not match fitted support")
        numeric = np.zeros((len(x), self.n_features, 8), dtype=np.float64)
        categorical = np.zeros((len(x), self.n_features, 4), dtype=np.float64)
        identity = (np.zeros((len(x), self.n_features, 64), dtype=np.int8)
                    if self.categorical_encoding == "hash_bits" else None)
        for j, stat in enumerate(self.stats):
            if self.is_categorical[j]:
                for i, value in enumerate(x[:, j]):
                    if _missing(value):
                        categorical[i, j, 3] = 1
                    else:
                        key = _canonical(value)
                        count = stat["counts"].get(key, 0)
                        categorical[i, j] = (_hash_code(self.effective_seed, self.ids[j], key),
                                              count / stat["denominator"], float(count == 0), 0.)
                        if identity is not None:
                            # A close scalar hash does not imply close identity bits.
                            # Every input bit is exactly representable even in BF16.
                            digest = _hash_digest(self.effective_seed, self.ids[j], key)[:8]
                            identity[i, j] = 2 * np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.int8) - 1
            else:
                values = np.asarray(x[:, j], dtype=np.float64)
                finite = np.isfinite(values)
                z = np.zeros_like(values)
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = values[finite] - stat["center"]
                    normalized = delta / stat["scale"]
                    overflow = np.isinf(delta)
                    normalized[overflow] = (values[finite][overflow] / stat["scale"] -
                                            stat["center"] / stat["scale"])
                z[finite] = normalized
                ranks = np.full_like(values, .5)
                sorted_values = stat["sorted"]
                if len(sorted_values):
                    a = np.searchsorted(sorted_values, values[finite], side="left")
                    b = np.searchsorted(sorted_values, values[finite], side="right")
                    ranks[finite] = (a + .5 * (b - a) + .5) / (len(sorted_values) + 1)
                numeric[:, j] = np.stack((np.clip(z, -100, 100), np.sign(z) * np.log1p(np.minimum(np.abs(z), 1e6)),
                                          ranks, np.isnan(values), np.isposinf(values), np.isneginf(values),
                                          z < -100, z > 100), -1)
        return EncodedTable(torch.as_tensor(numeric, device=device), torch.as_tensor(categorical, device=device),
                            torch.as_tensor(self.is_categorical, device=device),
                            None if identity is None else torch.as_tensor(identity, device=device))
