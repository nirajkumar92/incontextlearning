# Pinned TabICLv2 prior controls

The executable reference uses the official repository at
[`0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`](https://github.com/soda-inria/tabicl/tree/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3).
Its unmodified prior subtree, six released training scripts, complete license
and SHA-256/Git-blob manifest are in `third_party/tabicl_reference`. Verification
runs before import. The small source copy is deliberate: a package version or
moving `main` is insufficient to define a reproducible research control.

```sh
.venv/bin/python -m pip install '.[reference]'
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python scripts/reference_prior_smoke.py \
  --arm R --task classification --seed 42 --batch-size 2 \
  --output research/results/reference_native_classification.json
```

The CPU API is:

```python
from tabular_foundation.reference_prior import generate_reference_batch, ReferenceShape

batch = generate_reference_batch(42, arm="R", task="classification", stage=1, batch_size=4)
episodes, audit = batch.episodes, batch.audit
X, y, active_features, row_counts, support_counts = batch.native

# Shape-fixed diagnostic; seed 20 selects the 5% branch for slot zero.
diagnostic = generate_reference_batch(
    20, arm="R_P1_05", task="classification", batch_size=1,
    shape=ReferenceShape(n_support=32, n_query=16, n_features=8, n_classes=7),
)
```

`generate_reference_episode` is the same interface with batch size one, returning
an `Episode`. It retains slot accounting in `metadata['reference_control']`.
Use the batch API to preserve released grouping: four datasets share row count,
support split and requested feature count in stage one; stages two and three
use groups of one. Class budgets and graphs are sampled per dataset.
The current trainer's `TaskProducer` calls `generate_reference_episode` separately
for each task. It preserves native per-episode generation and filtering but does
not reproduce stage one's within-batch shape correlations. It is therefore a
prior control with a different batch sampler, even without an envelope extension.
Package task is either classification or regression; classification budgets are uniform integers 2–10.
Budget two defines binary; higher budgets define multiclass even if the observed
class count falls below the budget. Mixing classifier and regressor packages is
an outer experiment decision. A 50/50 package mix yields nominal binary,
multiclass and regression slot fractions 1/18, 4/9 and 1/2, before accounting for
observed-class collapse; it does not yield one third of each task.

| Arm | Operation after native task/shape draw | Ineligible 5% draw |
| --- | --- | --- |
| `R` | Native graph generator and filters | Not applicable |
| `R_P1_05` | P1 with 0.05 probability when requested F≥8 | Return to R |
| `R_P4_05` | P4 with 0.05 probability when requested F≥4 and task binary/regression | Return to R |

The upstream batch sampler first draws every slot's class budget, row counts,
split and feature count. The adapter then chooses branches using a separate,
seeded Philox stream. P1/P4 receive those exact dimensions, task and class budget;
their autonomous task distributions cannot override them. The family draw is not
repeated when reference filtering rejects a raw table. P1/P4 are unfiltered,
as in the authored bank. A failed generator aborts with a pickle-safe
`ReferenceGenerationError` carrying the partial audit; it never changes source.

Native R removes constant columns, so accepted width can be smaller than the
preselected width. The audit preserves both; requiring exactly the original
accepted width would introduce another rejection filter. Native R also may
realize fewer classes than its sampled class budget. The audit records the
actual class labels, not only the nominal count. Histograms of both requested
and accepted dimensions belong in every pilot report.

The reference has two rejection levels. `RandomDataset.sample` rejects graph
assignments without overlapping feature/target ancestors. `GraphPrior` removes
constant columns, rejects empty-feature datasets, attempts up to ten row
permutations to obtain matching support/query class sets, and applies the native
ExtraTrees OOB predictability filter. The latter uses 25 depth-six bootstrap
trees, a fixed estimator seed, and 200 bootstrap resamples. None of this logic is
reimplemented. Counting wrappers record graph proposals, graph rejections, raw
tables, empty-feature failures, class-split failures and predictability failures.
Raw-table counts and accepted source counts have different denominators.

The nominal addition share is 0.05 times eligibility. Under the native classifier
law, P4 is eligible only in the binary 1/9 of class-budget draws (and F≥4), so its
unconditional classifier share is at most 0.05/9≈0.56%. With a 50/50 classifier/
regressor package mix and the stage-one width law, P4 is about 2.71% of all slots;
P1 is about 4.67%. These are expectations before any failure abort, not exact
finite-run masses. Log the realized counts; do not relabel either arm as a flat
95/5 training mixture. Combining the two additions still requires the audit's
separate promotion and confirmation decision.

## What is preserved, and what is a new experiment

Stage-one recipe: 1,024 rows, uniform 30–90% support, feature count rounded from
continuous Uniform(1,100). The latter follows executable scripts/code; the paper
says integer Uniform(2,100). Stage two uses log-uniform 400–10,240 rows, stage
three 400–60,000; both use 79–81% support. Long rows trigger the native feature
ceilings. The copied scripts retain released optimizer/architecture settings for
provenance, but this adapter does not run their trainer.

The eight actual source function families are MLP, oblivious-tree ensemble,
discretization, linear, quadratic, random-Fourier GP, EM-like assignment and
product. Consult `_function.py`; the `GraphSCM` docstring's older list is not the
executed definition. Vector node states include hidden dimensions, seven
categorical converter modes, feature/node importance, norm controls, nonlinear
random points and concatenation/aggregation between parents. These are reasons
to preserve the source rather than treating a small scalar SCM as an equivalent
replacement.

Released defaults leave corrected numerical converters, corrected categorical
meta-sampling, corrected category sizes and `ensure_iid` disabled. Thus a warp
can modify propagated hidden node values instead of the emitted column,
categorical choices are resampled rather than shared as originally intended,
and feature-category cardinalities stay at most nine. These are documented
historical behaviors, not adapter defects to repair inside R.

The native generator fits feature/target transformations on all generated rows,
and its filter/split repair uses query labels during **generation**. No query
labels or filter decisions enter `Episode.model_inputs()`. The delivered feature
tensor is already ordinal-encoded, clipped, standardized and permuted; it carries
no category-type mask. Its Episode representation therefore marks all columns
numeric. Applying this project's support-only codec afterward is a distinct
training representation. Preserve that codec across the controlled prior arms,
or use the returned native tuple and official model for a full native-system
comparison. Neither route alone proves complete published-system reproduction.

An explicit `envelope={"max_classes": 256, "max_features": 1024}` is supported
and always recorded as `reference_distribution_modified=True`. It preserves the
graph mechanisms/filter but changes the shape/class law. Native long-row width
caps still apply, so this ceiling extension does not ensure wide-table exposure
at long context. A larger class budget may mostly produce fewer observed classes
after the native coverage/predictability filter. Treat coverage and generator
cost as measured gates; do not count nominal high-class slots as proven high-class
training. The native full system's classifier has only ten direct output classes,
so a widened local head and native many-class inference are also different systems.

## Reproducibility and evidence

Use `PYTHONHASHSEED=0` before the interpreter starts. Upstream graph assignment
iterates over a set of feature-group strings; setting only numerical RNG seeds
does not reproduce across Python processes. Generation restores Python, NumPy,
CPU Torch RNG state, Torch thread count and any preexisting TabICL import
namespace. Its temporary namespace is protected among adapter callers, but other
threads must not concurrently import/use TabICL; use process workers. No GPU RNG
is reset by the adapter. An attempt cap aborts instead of accepting or switching
to a different prior, and should be counted as a failed draw.

The retained prior subset was executed on Python 3.9.6, Torch 2.8.0, NumPy 1.26.4,
SciPy 1.13.1, scikit-learn 1.6.1, xgboost 2.1.4 and psutil 7.2.2. Full official
TabICL packaging requires Python≥3.10. Each smoke report records its actual
dependency versions, elapsed CPU time, source revision and per-array hashes.

`tests/test_reference_prior.py` compares R bit-for-bit with unmodified upstream
`GraphPrior.get_batch` under native stage-one shapes, verifies paired task/shape
draws across arms, eligibility mass return, exact authored class budgets, pinned
source verification, RNG/import restoration, failure abort and worker exception
propagation. Smoke JSON files under `research/results/reference_*.json` show
actual native tables and selected P1/P4 branches. These are CPU code contracts,
not training wins, benchmark results, native checkpoint reproduction or MI355X
performance measurements.

Primary references: [official pinned source](https://github.com/soda-inria/tabicl/tree/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/prior),
[released classifier recipe](https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/scripts/train_v2_clf_stage1.sh),
[paper](https://proceedings.mlr.press/v306/qu26a.html), Appendix E (generation),
Appendix I (quantile distribution), and §4/Appendix B (training).
