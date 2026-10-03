"""Frozen-weight standard and bounded-finance prediction with reusable contexts."""
from __future__ import annotations
from collections import OrderedDict
from contextlib import nullcontext
import time
import numpy as np
import torch
from .retrieval import ReservoirCodec, farthest_first_centers, stream_nearest_to_centers, assign_routes, union_context


def load_checkpoint(path, device='cpu'):
    from .model import build_model
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cfg = saved['config']
    model = build_model(cfg.get('model', 'tiny'), finance=cfg.get('finance_adapter', True), device=device,
                        model_options=cfg.get('model_options'))
    model.load_state_dict(saved['model'])
    return model.eval(), saved


def _amp(model, bf16):
    return torch.autocast('cuda', dtype=torch.bfloat16) if bf16 and model.device.type == 'cuda' else nullcontext()


def _prediction(output):
    if 'logits' in output:
        return torch.softmax(output['logits'].float(), -1).cpu().numpy()
    return output['mean'].cpu().numpy()


class Predictor:
    def __init__(self, model, query_batch_size=512, bf16=True):
        if query_batch_size < 1:
            raise ValueError('query_batch_size must be positive')
        self.model, self.query_batch_size = model.eval(), int(query_batch_size)
        self.context = None
        self.bf16 = bf16

    @torch.inference_mode()
    def fit_context(self, X, y, categorical, task, n_classes=2, **kwargs):
        with _amp(self.model, self.bf16):
            self.context = self.model.prepare_context(X, y, categorical, task, n_classes=n_classes, **kwargs)
        return self

    @torch.inference_mode()
    def predict(self, X):
        if self.context is None:
            raise RuntimeError('fit_context must precede predict')
        if not len(X):
            return np.empty((0, self.context.n_classes)) if self.context.task_id == 0 else np.empty(0)
        with _amp(self.model, self.bf16):
            return np.concatenate([_prediction(self.model.predict_cached(self.context, X[i:i+self.query_batch_size]))
                                   for i in range(0, len(X), self.query_batch_size)])


class FinancePredictor:
    """Full eligible-history search, fixed centers, lazy bounded model caches.

    Input rows MUST already have legal point-in-time features and known labels.
    `complete_cohort` is a caller-supplied data contract, never inferred from age.
    At most `cache_limit` route caches remain on device; eviction changes runtime,
    not the prediction function. No query labels enter setup or prediction.
    """
    def __init__(self, model, seed=0, reservoir=2048, positive_cap=1024,
                 local=4096, centers=64, positive_centers=16, search_cap=None,
                 cache_limit=64, query_batch_size=512, candidate_chunk=1024, bf16=True):
        for value in (reservoir, local, centers, cache_limit, query_batch_size, candidate_chunk):
            if value < 1:
                raise ValueError('Context, route and batch caps must be positive')
        self.model, self.seed = model.eval(), int(seed)
        self.bf16 = bf16
        self.reservoir_cap, self.positive_cap, self.local_cap = reservoir, positive_cap, local
        self.center_cap, self.positive_center_cap = centers, positive_centers
        self.search_cap, self.cache_limit = search_cap, cache_limit
        self.query_batch_size, self.candidate_chunk = query_batch_size, candidate_chunk
        self.caches = OrderedDict()
        self.setup_seconds = None

    def fit_context(self, X, y, categorical, task='binary', n_classes=2,
                    complete_cohort=False, reference_counts=None, hurdle=False):
        start = time.perf_counter()
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y)
        self.categorical = np.asarray(categorical, dtype=bool)
        self.task, self.n_classes, self.hurdle = task, n_classes, hurdle
        if self.X.ndim != 2 or self.y.shape != (len(self.X),) or not np.isfinite(self.y).all():
            raise ValueError('Eligible labeled matrix/targets required')
        if task != 'regression' and (not np.isin(self.y, np.arange(n_classes)).all()):
            raise ValueError('Labels outside declared class universe')
        if hurdle and (self.y < 0).any():
            raise ValueError('Realized-loss target cannot be negative')
        rng = np.random.default_rng(self.seed)
        self.N = len(self.y)
        full_amount = task == 'regression' and not hurdle
        positive = np.flatnonzero(self.y > 0) if not full_amount else np.empty(0, dtype=int)
        self.M = len(positive)
        self.R = rng.choice(self.N, min(self.N, self.reservoir_cap), replace=False)
        candidates_p = np.arange(self.N) if full_amount else positive
        self.P = rng.choice(candidates_p, min(len(candidates_p), self.positive_cap), replace=False)
        self.index_codec = ReservoirCodec.fit(self.X[self.R], self.categorical, self.seed)
        cp = np.empty(0, dtype=int) if full_amount else self.P
        self.center_ids, self.centers = farthest_first_centers(cp, self.index_codec.transform(self.X[cp]),
                            self.R, self.index_codec.transform(self.X[self.R]),
                            max_centers=self.center_cap, max_positive=self.positive_center_cap, tie_seed=self.seed)
        if not len(self.centers):
            self.center_ids, self.centers = np.array([-1]), np.zeros((1, self.index_codec.dimension))
        G = np.arange(self.N) if full_amount else np.flatnonzero(self.y == 0)
        if self.search_cap is not None:
            if self.search_cap < 0:
                raise ValueError('search_cap must be nonnegative or None for full search')
            G = rng.choice(G, min(len(G), self.search_cap), replace=False)
        self.searched = len(G)
        def candidates():
            for start in range(0, len(G), self.candidate_chunk):
                ids = G[start:start+self.candidate_chunk]
                yield ids, self.index_codec.transform(self.X[ids])
        self.local_ids = stream_nearest_to_centers(candidates(), self.centers, self.local_cap, self.seed)
        self.contexts = [union_context(self.R, self.P, local) for local in self.local_ids]
        self.reference_kind, self.reference_probs, self.pi_ref = 0, None, None
        self.reference_size = 0
        if not full_amount:
            if reference_counts is not None:
                counts = np.asarray(reference_counts, dtype=float)
                self.reference_kind = 1
            elif complete_cohort and self.N > 0:
                counts = np.bincount((self.y > 0).astype(int), minlength=2) if hurdle else np.bincount(self.y.astype(int), minlength=n_classes)
                self.reference_kind = 2
            else:
                counts = None
            if counts is not None:
                expected = 2 if hurdle else n_classes
                if counts.shape != (expected,) or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
                    raise ValueError('Reference counts must describe a nonempty representative adjudicated cohort')
                probabilities = (counts + .5)/(counts.sum() + .5*len(counts))
                self.pi_ref = float((counts[1:].sum() + .5)/(counts.sum() + 1.))
                self.reference_size = float(counts.sum())
                if not hurdle:
                    self.reference_probs = probabilities
        self.full_amount = full_amount
        self.caches.clear()
        self.setup_seconds = time.perf_counter()-start
        return self

    def _metadata(self, route):
        ids, _, _ = self.contexts[route]
        pi = self.pi_ref
        odds = np.clip(np.log(pi)-np.log1p(-pi), -40, 40)/20 if pi is not None else 0.
        mR = np.sum(self.y[self.R] > 0) if not self.full_amount else 0
        vector = np.array([np.log1p(self.N), np.log1p(self.M), np.log1p(len(self.R)),
                    np.log1p(len(self.P)), np.log1p(len(self.local_ids[route])), np.log1p(len(ids)),
                    np.log1p(mR), np.log(len(self.R)/self.N) if self.N else 0,
                    np.log(len(self.P)/self.M) if self.M and len(self.P) else 0,
                    float(self.M == 0), self.reference_kind, odds, np.log1p(self.searched), np.log1p(self.reference_size)])
        if self.full_amount:
            vector[[1,6,8,9,10,11,13]] = 0
        return vector

    @torch.inference_mode()
    def _cache(self, route):
        if route in self.caches:
            self.caches.move_to_end(route)
            return self.caches[route]
        ids, bits, codec_indices = self.contexts[route]
        with _amp(self.model, self.bf16):
            cache = self.model.prepare_context(self.X[ids], self.y[ids], self.categorical, self.task,
                        n_classes=self.n_classes, finance_metadata=self._metadata(route), source_bits=bits,
                        codec_indices=codec_indices, reference_probs=self.reference_probs,
                        hurdle=self.hurdle, encoding_seed=self.seed)
        self.caches[route] = cache
        while len(self.caches) > self.cache_limit:
            self.caches.popitem(last=False)
        return cache

    @torch.inference_mode()
    def prefill_all(self):
        if self.cache_limit < len(self.contexts):
            raise ValueError('Cache limit cannot retain every route; prefill_all would immediately evict caches')
        for route in range(len(self.contexts)):
            self._cache(route)
        return self

    @torch.inference_mode()
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        routing = assign_routes(self.index_codec.transform(X), self.center_ids, self.centers, self.seed)
        output = np.empty((len(X), self.n_classes)) if self.task != 'regression' else np.empty(len(X))
        for route in np.unique(routing):
            indices = np.flatnonzero(routing == route)
            cache = self._cache(int(route))
            for start in range(0, len(indices), self.query_batch_size):
                take = indices[start:start+self.query_batch_size]
                with _amp(self.model, self.bf16):
                    output[take] = _prediction(self.model.predict_cached(cache, X[take]))
        return output
