"""Dense support-only ICL model with exact reusable frontend and trunk caches.

This reference accepts one ragged episode per call. Batch accumulation belongs
to the trainer. It uses SDPA; optional research MoE/sparse graphs are not hidden
behind the dense API. Cached tensors retain autograd unless the caller uses
no_grad/inference_mode. Never reuse a cache after updating model parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .codec import TableCodec, EncodedTable, as_numpy, robust_location_scale
from .distributions import ContinuousDistribution, HurdleDistribution


@dataclass(frozen=True)
class ModelConfig:
    cell_width: int = 128
    cell_heads: int = 8
    cell_hidden: int = 512
    stages: int = 2
    column_blocks: int = 3
    row_blocks: int = 3
    inducing: int = 128
    cls_tokens: int = 8
    width: int = 1024
    heads: int = 16
    hidden: int = 2816
    layers: int = 24
    classes: int = 256
    categorical_encoding: str = "scalar_fourier"
    regression_head: str = "histogram"
    regression_bins: int = 128
    regression_quantiles: int = 999
    length_scaling: str = "legacy"
    trunk_query_kv_heads: Optional[int] = None
    hurdle_target_scale: str = "reference"

    def __post_init__(self):
        if self.categorical_encoding not in ("scalar_fourier", "hash_bits"):
            raise ValueError("categorical_encoding must be scalar_fourier or hash_bits")
        if self.regression_head not in ("histogram", "quantile"):
            raise ValueError("regression_head must be histogram or quantile")
        for field in ("regression_bins", "regression_quantiles"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.length_scaling not in ("legacy", "logarithmic"):
            raise ValueError("length_scaling must be legacy or logarithmic")
        h = self.trunk_query_kv_heads
        if h is not None and (isinstance(h, bool) or not isinstance(h, int) or h < 1 or self.heads % h):
            raise ValueError("trunk_query_kv_heads must be a positive divisor of heads or None")
        if self.hurdle_target_scale not in ("reference", "positive_support"):
            raise ValueError("hurdle_target_scale must be reference or positive_support")

    @classmethod
    def tiny(cls):
        return cls(cell_width=16, cell_heads=2, cell_hidden=64, column_blocks=1,
                   row_blocks=1, inducing=8, cls_tokens=4, width=64, heads=4,
                   hidden=176, layers=2)


class RMSNorm(nn.Module):
    def __init__(self, width, initial=1.):
        super().__init__()
        self.weight = nn.Parameter(torch.full((width,), float(initial)))

    def forward(self, x):
        dtype = x.dtype
        z = x if dtype == torch.float64 else x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6) * self.weight).to(dtype)


class MAB(nn.Module):
    """One exact registered hidden block; all seven matrices belong to Muon."""
    def __init__(self, width, heads, hidden, post_gain, rope=False, length_scaling="legacy"):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("head dimension must be positive and even")
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.rope = rope
        if length_scaling not in ("legacy", "logarithmic"):
            raise ValueError("unknown attention length scaling")
        self.length_scaling = length_scaling
        for name in ("q", "k", "v", "out"):
            setattr(self, name, nn.Linear(width, width, bias=False))
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width, post_gain)
        self.norm3 = RMSNorm(width)
        self.norm4 = RMSNorm(width, post_gain)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.length_scale = nn.Parameter(torch.zeros(heads))

    def hidden_parameters(self):
        return [getattr(self, name).weight for name in ("q", "k", "v", "out", "gate", "value", "down")]

    def _heads(self, x):
        return x.reshape(x.shape[:-1] + (self.heads, self.head_dim)).transpose(-3, -2)

    def _rotary(self, x):
        if not self.rope:
            return x
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        inv = 10000. ** (-torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim)
        phase = positions[:, None] * inv[None, :]
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def prepare_kv(self, keys):
        z = self.norm1(keys)
        return self._rotary(self.k_norm(self._heads(self.k(z)))), self._heads(self.v(z))

    def query_kv(self, kv, heads=None):
        """Retain a prefix of support KV heads for query-only grouped attention.

        Support self-attention must use the original complete KV pair. Cloning
        the prefix makes inference storage genuinely compact even for one-row
        tensors where a slice can already be contiguous. Autograd is retained.
        """
        if heads is None or heads == self.heads:
            return kv
        if heads < 1 or self.heads % heads:
            raise ValueError("query KV heads must divide attention heads")
        return tuple(t[..., :heads, :, :].clone(memory_format=torch.contiguous_format) for t in kv)

    def attention_temperature(self, n):
        """Positive log-n ablation, not QASSMax's query-dependent vector MLP.

        At initialization its multiplier is one for 1,024 keys. Only support
        aggregators use the new mode; row and inducing-summary attention keep
        the legacy rule. Counting keys (not query batch size) preserves caches.
        """
        if self.length_scaling == "logarithmic":
            amplitude = F.softplus(self.length_scale.float()) / math.log(2)
            return amplitude * (math.log(max(n, 2)) / math.log(1024))
        count = max(-math.log(4), min(math.log(4), math.log(max(n, 1) / 1024)))
        return torch.exp(torch.tanh(self.length_scale.float()) * count)

    def from_kv(self, queries, kv):
        key, value = kv
        kv_heads = key.shape[-3]
        if kv_heads != value.shape[-3] or kv_heads < 1 or self.heads % kv_heads:
            raise ValueError("key/value heads must match and divide query heads")
        n = key.shape[-2]
        if n == 0 or queries.numel() == 0:
            attended = torch.zeros_like(queries)
        else:
            q = self._rotary(self.q_norm(self._heads(self.q(self.norm1(queries)))))
            temperature = self.attention_temperature(n)
            shape = [1] * q.ndim
            shape[-3] = self.heads
            q = q * temperature.reshape(shape).to(q.dtype)
            # SDPA repeats each retained head for a consecutive group of query
            # heads. Backend selection determines whether this saves bandwidth;
            # a math fallback may materialize the repeated tensors transiently.
            attention = F.scaled_dot_product_attention(q, key, value, dropout_p=0.,
                                                       enable_gqa=kv_heads != self.heads)
            attended = self.out(attention.transpose(-3, -2).reshape(queries.shape))
        u = queries + self.norm2(attended)
        z = self.norm3(u)
        return u + self.norm4(self.down(F.silu(self.gate(z)) * self.value(z)))

    def forward(self, queries, keys):
        return self.from_kv(queries, self.prepare_kv(keys))


class ISAB(nn.Module):
    def __init__(self, cfg, post_gain):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(cfg.inducing, cfg.cell_width))
        self.first = MAB(cfg.cell_width, cfg.cell_heads, cfg.cell_hidden, post_gain,
                         length_scaling=cfg.length_scaling)
        self.second = MAB(cfg.cell_width, cfg.cell_heads, cfg.cell_hidden, post_gain)

    def support(self, cells):
        # Features are independent batch entries, rows are the attention axis.
        z = cells.transpose(0, 1)
        inducing = self.inducing.unsqueeze(0).expand(z.shape[0], -1, -1)
        summary = self.first(inducing, z)
        kv = self.second.prepare_kv(summary)
        return self.second.from_kv(z, kv).transpose(0, 1), kv

    def query(self, cells, kv):
        return self.second.from_kv(cells.transpose(0, 1), kv).transpose(0, 1)


class CellEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.cell_width
        self.categorical_encoding = cfg.categorical_encoding
        self.numeric = nn.ModuleList([nn.Linear(88, c, bias=False) for _ in range(3)])
        nominal_width = 67 if cfg.categorical_encoding == "hash_bits" else 68
        self.categorical = nn.ModuleList([nn.Linear(nominal_width, c, bias=False) for _ in range(3)])
        self.log_frequencies = nn.Parameter(torch.empty(6, 32))
        self.types = nn.Embedding(2, c)
        self.norm = nn.LayerNorm(c, eps=1e-5)

    def forward(self, table: EncodedTable, extra):
        x, cat, mask = table.numeric, table.categorical, table.categorical_mask
        shape = x.shape[:2] + (self.types.embedding_dim,)
        out = self.types.weight.new_zeros(shape)
        frequency = self.log_frequencies.double().clamp(math.log(1e-4), math.log(100)).exp()
        rank_phase = 2 * math.pi * x[..., 2:3] * torch.arange(1, 9, device=x.device, dtype=torch.float64)
        rank_features = torch.cat((rank_phase.sin(), rank_phase.cos()), -1)
        for slot, offset in enumerate((0, 1, 3)):
            numeric_phase = 2 * math.pi * x[..., :1] * frequency[2 * slot]
            numeric_features = torch.cat((x, numeric_phase.sin(), numeric_phase.cos(), rank_features), -1)
            if self.categorical_encoding == "hash_bits":
                if table.categorical_identity is None:
                    raise ValueError("hash_bits encoder requires a matching fitted codec")
                nominal_features = torch.cat((table.categorical_identity.to(cat.dtype), cat[..., 1:]), -1)
            else:
                nominal_phase = 2 * math.pi * cat[..., :1] * frequency[2 * slot + 1]
                nominal_fourier = torch.cat((nominal_phase.sin(), nominal_phase.cos()), -1)
                nominal_fourier = nominal_fourier * (1 - cat[..., 3:4])
                nominal_features = torch.cat((cat, nominal_fourier), -1)
            # torch.roll negative offset implements j+offset modulo F.
            pn = self.numeric[slot](numeric_features.to(self.numeric[slot].weight.dtype))
            pc = self.categorical[slot](nominal_features.to(self.categorical[slot].weight.dtype))
            projected = torch.where(mask[None, :, None], pc, pn)
            out = out + projected.roll(-offset, dims=1)
        return self.norm(out + self.types(mask.long())[None] + extra[:, None, :])


@dataclass
class ContextCache:
    model_id: int
    versions: tuple
    device: torch.device
    parameter_dtype: torch.dtype
    autocast_dtype: Optional[torch.dtype]
    tracks_gradients: bool
    codec: TableCodec
    task_id: int
    n_classes: int
    class_slots: Tensor
    metadata_early: Tensor
    metadata_late: Tensor
    frontend: list
    trunk: list
    target_center: float
    target_scale: float
    reference_probs: Optional[Tensor]
    hurdle: bool
    atom_offset: Tensor


class TabularFoundationModel(nn.Module):
    def __init__(self, cfg=ModelConfig(), finance=True, activation_checkpointing=False):
        super().__init__()
        self.config = cfg
        self.finance_enabled = finance
        self.activation_checkpointing = activation_checkpointing
        self.encoder = CellEncoder(cfg)
        c, d = cfg.cell_width, cfg.width
        self.early_task, self.late_task = nn.Embedding(2, c), nn.Embedding(2, d)
        self.early_role, self.late_role = nn.Embedding(2, c), nn.Embedding(2, d)
        self.early_class, self.late_class = nn.Embedding(cfg.classes, c), nn.Embedding(cfg.classes, d)
        self.early_reg, self.late_reg = nn.Linear(3, c, bias=False), nn.Linear(3, d, bias=False)
        self.cls = nn.Parameter(torch.empty(cfg.cls_tokens, c))
        row_width = cfg.cls_tokens * c
        self.row_projection = nn.Identity() if row_width == d else nn.Linear(row_width, d, bias=False)
        n_front_mabs = cfg.stages * (2 * cfg.column_blocks + cfg.row_blocks)
        post_front = 1 / math.sqrt(2 * n_front_mabs)
        self.columns = nn.ModuleList([nn.ModuleList([ISAB(cfg, post_front) for _ in range(cfg.column_blocks)])
                                      for _ in range(cfg.stages)])
        self.rows = nn.ModuleList([nn.ModuleList([MAB(c, cfg.cell_heads, cfg.cell_hidden, post_front, rope=True)
                                                 for _ in range(cfg.row_blocks)]) for _ in range(cfg.stages)])
        self.trunk = nn.ModuleList([MAB(d, cfg.heads, cfg.hidden, 1 / math.sqrt(2 * cfg.layers),
                                       length_scaling=cfg.length_scaling) for _ in range(cfg.layers)])
        self.head_norm = RMSNorm(d)
        self.shared_head = nn.Linear(d, d, bias=False)
        self.class_head = nn.Linear(d, cfg.classes, bias=False)
        regression_outputs = cfg.regression_bins + 4 if cfg.regression_head == "histogram" else cfg.regression_quantiles
        self.regression_head = nn.Linear(d, regression_outputs, bias=False)
        if finance:
            self.metadata_in = nn.Linear(14, 128, bias=False)
            self.metadata_out = nn.Linear(128, c + d, bias=False)
            self.early_source = nn.Linear(3, c, bias=False)
            self.late_source = nn.Linear(3, d, bias=False)
            self.atom_head = nn.Linear(d, 1, bias=False)
            if cfg.regression_head == "quantile":
                # A finite quantile grid cannot resolve a rare atom below its
                # tail level or supply the positive-component likelihood.
                self.hurdle_regression_head = nn.Linear(d, cfg.regression_bins + 4, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1 / math.sqrt(module.in_features))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=.02)
        heads = [self.class_head, self.regression_head]
        if hasattr(self, "hurdle_regression_head"):
            heads.append(self.hurdle_regression_head)
        for module in heads:
            nn.init.normal_(module.weight, std=.02 / math.sqrt(module.in_features))
        nn.init.normal_(self.cls, std=.02)
        for stage in self.columns:
            for layer in stage:
                nn.init.normal_(layer.inducing, std=.02)
        with torch.no_grad():
            frequencies = torch.linspace(math.log(.01), math.log(10), 32, device=self.encoder.log_frequencies.device)
            self.encoder.log_frequencies.copy_(frequencies.expand(6, -1))
        if self.finance_enabled:
            nn.init.zeros_(self.metadata_out.weight)
            nn.init.zeros_(self.early_source.weight)
            nn.init.zeros_(self.late_source.weight)
            # The residual starts at zero. A legal finance reference can supply
            # the atom prior; otherwise this is a neutral 50/50 initialization.
            nn.init.zeros_(self.atom_head.weight)

    @property
    def device(self):
        return self.cls.device

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def _autocast_dtype(self):
        return torch.get_autocast_dtype(self.device.type) if torch.is_autocast_enabled(self.device.type) else None

    def _run(self, function, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _metadata(self, metadata):
        c, d = self.config.cell_width, self.config.width
        if metadata is None:
            return self.cls.new_zeros(c), self.cls.new_zeros(d)
        if not self.finance_enabled:
            raise ValueError("model was built without finance adapter")
        m = torch.as_tensor(metadata, device=self.device, dtype=self.cls.dtype)
        if m.shape != (14,) or not torch.isfinite(m).all():
            raise ValueError("finance_metadata must have 14 finite values")
        z = self.metadata_out(F.gelu(self.metadata_in(m)))
        return z[:c], z[c:]

    def _label_extra(self, ys, task_id, slots, center, scale):
        if task_id == 0:
            indices = ys.long()
            if ys.numel() and (not torch.equal(ys, indices.to(ys.dtype)) or indices.min() < 0 or indices.max() >= len(slots)):
                raise ValueError("support labels must be legal integer class IDs")
            indices = slots[indices]
            return self.early_class(indices), self.late_class(indices)
        # Standardize before the precision cast so a finite large target cannot
        # become infinity merely by entering a lower-precision encoder.
        z = (ys.double() - center) / scale
        label = torch.stack((z.clamp(-20, 20), z.sign() * torch.log1p(z.abs().clamp_max(1e6)), (z.abs() > 20).to(z.dtype)), -1)
        return self.early_reg(label.to(self.cls.dtype)), self.late_reg(label.to(self.cls.dtype))

    def prepare_context(self, x_support, y_support, categorical, task, n_classes=2, class_slots=None,
                        finance_metadata=None, source_bits=None, codec_indices=None, reference_probs=None,
                        hurdle=False, encoding_seed=0, column_ids=None):
        task_id = 1 if task in ("regression", "amount", "fraud_loss") else 0
        if task not in ("regression", "amount", "fraud_loss", "binary", "multiclass", "classification"):
            raise ValueError(f"unknown task {task}")
        if hurdle and (not task_id or not self.finance_enabled):
            raise ValueError("hurdle requires regression and finance-enabled model")
        codec = TableCodec(encoding_seed, column_ids, self.config.categorical_encoding).fit(x_support, categorical, codec_indices)
        support = codec.transform(x_support, device=self.device)
        ns = support.numeric.shape[0]
        ys = torch.as_tensor(as_numpy(y_support), device=self.device)
        if ys.shape != (ns,) or not torch.isfinite(ys).all():
            raise ValueError("support targets must be a finite vector matching support rows")
        if hurdle and (ys < 0).any():
            raise ValueError("hurdle regression requires nonnegative support targets")
        if task_id:
            n_classes = 0
            slots = torch.empty(0, device=self.device, dtype=torch.long)
        else:
            if not 1 <= n_classes <= self.config.classes:
                raise ValueError("class count exceeds native head capacity")
            if class_slots is None:
                slots = torch.arange(n_classes, device=self.device)
            else:
                raw_slots = torch.as_tensor(class_slots, device=self.device)
                slots = raw_slots.long()
                if not torch.equal(raw_slots, slots.to(raw_slots.dtype)):
                    raise ValueError("class_slots must contain integer slot IDs")
                slots = slots.clone()
            if slots.shape != (n_classes,) or len(torch.unique(slots)) != n_classes or slots.min() < 0 or slots.max() >= self.config.classes:
                raise ValueError("class_slots must be a unique legal injection")
        if task_id:
            values = as_numpy(y_support)
            target_fit = values[codec.fit_indices]
            if hurdle and self.config.hurdle_target_scale == "positive_support":
                # At rare event rates the natural reservoir is usually all
                # zero. Scale the conditional severity from observed positive
                # support labels; population weighting and atom priors remain
                # separate. This is an affine parameterization, not an estimate
                # of population moments, and never consults query outcomes.
                positive = values[values > 0]
                if len(positive):
                    target_fit = positive
            center, scale = robust_location_scale(target_fit)
        else:
            center, scale = 0., 1.
        early_meta, late_meta = self._metadata(finance_metadata)
        early_labels, late_labels = self._label_extra(ys, task_id, slots, center, scale)
        early = early_labels + self.early_task.weight[task_id] + self.early_role.weight[0] + early_meta
        late = late_labels + self.late_task.weight[task_id] + self.late_role.weight[0] + late_meta
        if source_bits is not None:
            if not self.finance_enabled:
                raise ValueError("source bits require finance adapter")
            bits = torch.as_tensor(source_bits, device=self.device, dtype=self.cls.dtype)
            if bits.shape != (ns, 3) or not ((bits == 0) | (bits == 1)).all():
                raise ValueError("source_bits must be an ns by 3 binary membership matrix")
            early = early + self.early_source(bits)
            late = late + self.late_source(bits)
        cells = self.encoder(support, early)
        cls = self.cls.unsqueeze(0).expand(ns, -1, -1)
        front_cache = []
        for columns, rows in zip(self.columns, self.rows):
            stage_cache = []
            for layer in columns:
                cells, kv = self._run(layer.support, cells)
                stage_cache.append(kv)
            both = torch.cat((cells, cls), 1)
            for layer in rows:
                both = self._run(layer, both, both)
            cells, cls = both[:, :codec.n_features], both[:, codec.n_features:]
            front_cache.append(stage_cache)
        state = self.row_projection(cls.flatten(1)) + late
        trunk_cache = []
        for layer in self.trunk:
            kv = layer.prepare_kv(state)
            state = self._run(layer.from_kv, state, kv)
            trunk_cache.append(layer.query_kv(kv, self.config.trunk_query_kv_heads))
        ref = None
        if reference_probs is not None:
            if task_id:
                raise ValueError("class reference_probs are only for classification")
            ref = torch.as_tensor(reference_probs, device=self.device, dtype=torch.float64)
            if ref.shape != (n_classes,) or not torch.isfinite(ref).all() or not (ref > 0).all():
                raise ValueError("reference_probs must be finite positive class probabilities")
            ref = ref / ref.max()
            ref = ref / ref.sum()
        atom_offset = self.cls.new_zeros((), dtype=torch.float32)
        if hurdle and finance_metadata is not None:
            metadata = torch.as_tensor(finance_metadata, device=self.device, dtype=torch.float32)
            # Index 10 explicitly declares an available reference; index 11 is
            # its clipped event logit divided by 20. This is a trained prior
            # parameterization, not a post-hoc support-sampling correction.
            if metadata[10] > 0:
                atom_offset = -20 * metadata[11]
        return ContextCache(id(self), self._versions(), self.device, self.cls.dtype, self._autocast_dtype(),
                            torch.is_grad_enabled(), codec, task_id, n_classes, slots,
                            early_meta, late_meta, front_cache, trunk_cache, center, scale, ref, hurdle, atom_offset)

    def predict_cached(self, cache: ContextCache, x_query):
        if (cache.model_id != id(self) or cache.versions != self._versions() or cache.device != self.device
                or cache.parameter_dtype != self.cls.dtype or cache.autocast_dtype != self._autocast_dtype()):
            raise RuntimeError("context cache is stale or belongs to a different model")
        if self.training and torch.is_grad_enabled() and not cache.tracks_gradients:
            raise RuntimeError("inference cache cannot supply support gradients during training")
        query = cache.codec.transform(x_query, device=self.device)
        nq = query.numeric.shape[0]
        early = (self.early_task.weight[cache.task_id] + self.early_role.weight[1] + cache.metadata_early).expand(nq, -1)
        cells = self.encoder(query, early)
        cls = self.cls.unsqueeze(0).expand(nq, -1, -1)
        for stage, (columns, rows) in enumerate(zip(self.columns, self.rows)):
            for index, layer in enumerate(columns):
                cells = self._run(layer.query, cells, cache.frontend[stage][index])
            both = torch.cat((cells, cls), 1)
            for layer in rows:
                both = self._run(layer, both, both)
            cells, cls = both[:, :cache.codec.n_features], both[:, cache.codec.n_features:]
        state = self.row_projection(cls.flatten(1)) + self.late_task.weight[cache.task_id] + self.late_role.weight[1] + cache.metadata_late
        for layer, kv in zip(self.trunk, cache.trunk):
            state = self._run(layer.from_kv, state, kv)
        hidden = F.gelu(self.shared_head(self.head_norm(state)))
        if cache.task_id == 0:
            raw = self.class_head(hidden).float()
            logits = raw[:, cache.class_slots]
            if cache.reference_probs is not None:
                logits = logits + cache.reference_probs.log().to(logits.dtype)
            masked = raw.new_full(raw.shape, -torch.inf)
            masked[:, cache.class_slots] = logits
            return {"logits": logits, "slot_logits": masked, "class_slots": cache.class_slots}
        head = self.hurdle_regression_head if cache.hurdle and self.config.regression_head == "quantile" else self.regression_head
        params = head(hidden).float()
        output = {"regression_params": params, "target_center": cache.target_center, "target_scale": cache.target_scale}
        if self.config.regression_head == "quantile" and not cache.hurdle:
            levels = torch.arange(1, self.config.regression_quantiles + 1, device=params.device,
                                  dtype=torch.float64) / (self.config.regression_quantiles + 1)
            original = cache.target_center + cache.target_scale * params.double()
            output.update(quantiles=original.sort(-1).values, quantile_levels=levels,
                          mean=original.mean(-1), loss_kind="pinball")
            return output
        distribution = ContinuousDistribution(params, cache.target_center, cache.target_scale,
                                               bins=self.config.regression_bins)
        if cache.hurdle:
            atom = self.atom_head(hidden).float() + cache.atom_offset
            distribution = HurdleDistribution(distribution, atom)
            output["atom_logit"] = atom
        output.update(distribution=distribution, mean=distribution.mean, loss_kind="nll")
        return output

    def forward(self, x_support, y_support, x_query, categorical, task, n_classes=2, class_slots=None,
                finance_metadata=None, source_bits=None, codec_indices=None, reference_probs=None,
                hurdle=False, encoding_seed=0, column_ids=None):
        context = self.prepare_context(x_support, y_support, categorical, task, n_classes, class_slots,
                                       finance_metadata, source_bits, codec_indices, reference_probs,
                                       hurdle, encoding_seed, column_ids)
        return self.predict_cached(context, x_query)


def build_model(size="tiny", finance=True, device=None, activation_checkpointing=False, model_options=None):
    grid = {"small50": (512, 12), "small100": (768, 12), "wide500": (1280, 24), "large1000": (1536, 32)}
    if size == "persistent_small":
        # Retain feature cells through every attention stage; compress only at
        # the readout. Cross-row communication uses inducing summaries, so this
        # is an authored compact comparator, not a TabPFN reproduction or full
        # cell-to-cell attention. The existing support-only cache law applies.
        cfg = ModelConfig(cell_width=256, cell_heads=8, cell_hidden=1024,
                          stages=12, column_blocks=1, row_blocks=1, inducing=128,
                          cls_tokens=8, width=512, heads=8, hidden=1408, layers=0)
    elif size == "persistent_tiny":
        cfg = replace(ModelConfig.tiny(), stages=3, column_blocks=1,
                      row_blocks=1, layers=0)
    elif size in grid:
        width, layers = grid[size]
        cfg = ModelConfig(width=width, layers=layers, heads=width // 64,
                          hidden=64 * math.ceil((11 * width / 4) / 64))
    elif size in ("tiny", "base"):
        cfg = ModelConfig() if size == "base" else ModelConfig.tiny()
    else:
        raise ValueError("unknown model size; choose tiny/base/small50/small100/"
                         "wide500/large1000/persistent_small/persistent_tiny")
    if model_options:
        allowed = {"categorical_encoding", "regression_head", "regression_bins", "regression_quantiles", "length_scaling",
                   "trunk_query_kv_heads", "hurdle_target_scale"}
        unknown = set(model_options) - allowed
        if unknown:
            raise ValueError(f"unknown model_options: {sorted(unknown)}")
        cfg = replace(cfg, **model_options)
    if device is None:
        model = TabularFoundationModel(cfg, finance, activation_checkpointing)
    else:
        with torch.device(device):
            model = TabularFoundationModel(cfg, finance, activation_checkpointing)
    if size == "base" and cfg == ModelConfig():
        expected = 315_175_184 if finance else 315_021_456
        if model.parameter_count() != expected:
            raise AssertionError(f"parameter geometry mismatch: {model.parameter_count()} != {expected}")
    return model
