"""Observable-only, deterministic reservoir codec and exact centroid retrieval.

The candidate iterator may cover a finite sample or the complete eligible pool.
This module never changes that choice, and never uses latent feature distances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def mix64(values: np.ndarray | int) -> np.ndarray:
    """SplitMix64 permutation; overflow is intentional unsigned arithmetic."""
    with np.errstate(over="ignore"):
        z = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stable_keys(ids: np.ndarray, seed: int) -> np.ndarray:
    """Random-looking, collision-free tie keys for distinct 64-bit finite IDs."""
    return mix64(np.asarray(ids, dtype=np.uint64) ^ np.uint64(seed & ((1 << 64) - 1)))


@dataclass
class ReservoirCodec:
    """Midrank numeric and feature-specific categorical random-sign index.

    The fitted rows must be the natural reservoir, not enriched support. Empty
    numeric references map to rank .5. Missingness has its own projection.
    """

    categorical: np.ndarray
    sorted_numeric: list[np.ndarray]
    seed: int = 0
    dimension: int = 128

    @classmethod
    def fit(cls, reservoir: np.ndarray, categorical: np.ndarray, seed: int = 0,
            dimension: int = 128) -> ReservoirCodec:
        reservoir = np.asarray(reservoir, dtype=np.float64)
        categorical = np.asarray(categorical, dtype=bool)
        if reservoir.ndim != 2 or reservoir.shape[1] != len(categorical):
            raise ValueError("Reservoir and categorical widths differ")
        if dimension < 1 or np.any(np.isinf(reservoir)):
            raise ValueError("Invalid codec dimension or infinite data")
        sorted_numeric = [np.sort(reservoir[np.isfinite(reservoir[:, j]), j])
                          if not categorical[j] else np.empty(0)
                          for j in range(reservoir.shape[1])]
        return cls(categorical.copy(), sorted_numeric, int(seed), int(dimension))

    def _signs(self, keys: np.ndarray) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.uint64)
        axes = mix64(np.arange(self.dimension, dtype=np.uint64) + np.uint64(17011))
        hashes = mix64(keys[..., None] ^ axes ^ np.uint64(self.seed & ((1 << 64) - 1)))
        return ((hashes & np.uint64(1)).astype(np.float64) * 2.0 - 1.0) / np.sqrt(self.dimension)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.categorical):
            raise ValueError("Index input width differs from fitted reservoir")
        if np.any(np.isinf(values)):
            raise ValueError("Infinite feature value")
        output = np.zeros((len(values), self.dimension), dtype=np.float64)
        for j in range(values.shape[1]):
            v = values[:, j]
            observed = np.isfinite(v)
            feature_key = mix64(np.uint64(j + 1))
            if self.categorical[j]:
                # Canonicalize signed zero and NaN; category identity is exact FP64.
                canonical = np.where(observed, np.where(v == 0, 0.0, v), 0.0)
                value_bits = np.ascontiguousarray(canonical).view(np.uint64)
                key = mix64(value_bits) ^ feature_key
                key[~observed] = feature_key ^ np.uint64(0xA5A5A5A5A5A5A5A5)
                output += self._signs(key)
            else:
                reference = self.sorted_numeric[j]
                rank = np.full(len(v), .5)
                if len(reference):
                    left = np.searchsorted(reference, v[observed], side="left")
                    right = np.searchsorted(reference, v[observed], side="right")
                    rank[observed] = (left + right) / (2.0 * len(reference))
                output += (2 * rank - 1)[:, None] * self._signs(feature_key)
                output += (~observed)[:, None] * self._signs(feature_key ^ np.uint64(713))
        return output / np.sqrt(max(1, values.shape[1]))


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norms = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :]
    distances = np.maximum(0.0, norms - 2 * a @ b.T)
    # The dot-product identity loses small distances by cancellation. Recompute
    # only near-coincident pairs so exact duplicate vectors have distance zero
    # and genuinely distinct close vectors are not collapsed into one route.
    near = distances <= 32 * np.finfo(np.float64).eps * np.maximum(1., norms)
    i, j = np.nonzero(near)
    if len(i):
        difference = a[i] - b[j]
        distances[i, j] = np.einsum("ij,ij->i", difference, difference)
    return distances


def farthest_first_centers(positive_ids: np.ndarray, positive_index: np.ndarray,
                           reservoir_ids: np.ndarray, reservoir_index: np.ndarray,
                           max_centers: int = 64, max_positive: int = 16,
                           tie_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Select up to 16 capped-P prototypes, then fill from R using one traversal.

    Existing centers participate in all subsequent minimum-distance decisions.
    Zero-distance duplicates do not manufacture additional routes.
    """
    pids, rids = np.asarray(positive_ids, dtype=np.int64), np.asarray(reservoir_ids, dtype=np.int64)
    px, rx = np.asarray(positive_index, dtype=np.float64), np.asarray(reservoir_index, dtype=np.float64)
    if max_centers < 1 or max_positive < 0 or len(pids) != len(px) or len(rids) != len(rx):
        raise ValueError("Invalid center arguments")
    dimension = px.shape[1] if px.ndim == 2 else rx.shape[1]
    chosen_ids: list[int] = []
    chosen_x: list[np.ndarray] = []

    def extend(ids: np.ndarray, x: np.ndarray, limit: int) -> None:
        if not len(ids) or len(chosen_ids) >= limit:
            return
        eligible = ~np.isin(ids, np.asarray(chosen_ids, dtype=np.int64))
        keys = stable_keys(ids, tie_seed)
        while np.any(eligible) and len(chosen_ids) < limit:
            if not chosen_ids:
                index = np.flatnonzero(eligible)[np.argmin(keys[eligible])]
            else:
                distances = squared_distances(x, np.stack(chosen_x)).min(1)
                distances[~eligible] = -np.inf
                farthest = np.max(distances)
                if farthest <= 0.:
                    break
                options = np.flatnonzero(distances == farthest)
                index = options[np.argmin(keys[options])]
            chosen_ids.append(int(ids[index]))
            chosen_x.append(x[index].copy())
            eligible &= ids != ids[index]

    extend(pids, px, min(max_centers, max_positive))
    extend(rids, rx, max_centers)
    return (np.asarray(chosen_ids, dtype=np.int64),
            np.stack(chosen_x) if chosen_x else np.empty((0, dimension), dtype=np.float64))


def stream_nearest_to_centers(candidates: Iterable[tuple[np.ndarray, np.ndarray]],
                              centers: np.ndarray, count: int,
                              tie_seed: int = 0) -> list[np.ndarray]:
    """Exact stable top-k over every supplied candidate, using bounded memory."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or count < 0:
        raise ValueError("Invalid centers or local count")
    best_ids = [np.empty(0, dtype=np.int64) for _ in centers]
    best_distances = [np.empty(0, dtype=np.float64) for _ in centers]
    if count == 0:
        return best_ids
    for ids, index in candidates:
        ids, index = np.asarray(ids, dtype=np.int64), np.asarray(index, dtype=np.float64)
        if len(ids) != len(index):
            raise ValueError("Candidate IDs and index vectors differ")
        distances = squared_distances(index, centers)
        for j in range(len(centers)):
            merged_ids = np.concatenate((best_ids[j], ids))
            merged_dist = np.concatenate((best_distances[j], distances[:, j]))
            order = np.lexsort((stable_keys(merged_ids, tie_seed), merged_dist))[:count]
            best_ids[j] = merged_ids[order]
            best_distances[j] = merged_dist[order]
    return best_ids


def assign_routes(index: np.ndarray, center_ids: np.ndarray, centers: np.ndarray,
                  tie_seed: int = 0) -> np.ndarray:
    """Query-independent routing: changing sibling queries cannot change a route."""
    if len(centers) == 0:
        return np.zeros(len(index), dtype=np.int64)
    ordering = np.argsort(stable_keys(center_ids, tie_seed), kind="stable")
    return ordering[np.argmin(squared_distances(index, centers[ordering]), axis=1)]


def union_context(reservoir_ids: np.ndarray, positive_ids: np.ndarray,
                  local_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R-first deduplication with membership union and reservoir codec positions."""
    ids: list[int] = []
    membership: list[list[float]] = []
    positions: dict[int, int] = {}
    for source, group in enumerate((reservoir_ids, positive_ids, local_ids)):
        for value in group:
            key = int(value)
            if key not in positions:
                positions[key] = len(ids)
                ids.append(key)
                membership.append([0., 0., 0.])
            membership[positions[key]][source] = 1.
    bits = np.asarray(membership, dtype=np.float64).reshape(-1, 3)
    return np.asarray(ids, dtype=np.int64), bits, np.flatnonzero(bits[:, 0]).astype(np.int64)
