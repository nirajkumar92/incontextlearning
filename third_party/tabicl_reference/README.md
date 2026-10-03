# Pinned TabICLv2 generation source

Source: https://github.com/soda-inria/tabicl/tree/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3

Revision `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3` (2026-09-02), retrieved 2026-10-03.
The 35 files listed in `manifest.json` are byte-for-byte upstream files. Their
Git blob SHA-1 and SHA-256 hashes were verified against GitHub's response. The
complete upstream LICENSE is retained; this prior subtree is BSD-3-Clause. The
forecast code covered by the additional license is not included.

The small prior subtree and six training recipes are retained to make reference
generation independent of a moving branch or installed inference package. There
are no model weights, alternate generator implementations, or edits to these
upstream files. The local adapter lives in `src/tabular_foundation/reference_prior.py`
at the project root. It verifies this manifest and every source before loading.
This README is local documentation and is not an upstream file.

Install optional dependencies using the project environment:

```sh
.venv/bin/python -m pip install -r scripts/requirements-reference.txt
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python scripts/reference_prior_smoke.py \
  --arm R --task classification --support 128 --query 64 --features 8 --classes 2 \
  --output research/results/reference_binary_smoke.json
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python scripts/reference_prior_smoke.py \
  --arm R_P1_05 --task classification --seed 20 --support 32 --query 16 \
  --features 8 --classes 7
PYTHONHASHSEED=0 PYTHONPATH=src .venv/bin/python -m pytest tests/test_reference_prior.py
```

Omit the explicit shape flags for the released stage laws. Stage 1 has 1,024
rows and a 30–90% support fraction; stages 2/3 draw log-uniform lengths from
400 to 10,240/60,000 and use a 79–81% support fraction. The scripts use a
rounded continuous-uniform 1–100 feature draw, unlike the paper's stated
integer-uniform 2–100. Long contexts trigger the native width caps. The
classification budget is uniform on 2–10 classes; regression is a separate
package. Do not silently equate these to balanced binary/multiclass/regression
training. A fixed shape is explicitly marked as a diagnostic override.

An optional `envelope` mapping can explicitly replace `min_features`,
`max_features` or `max_classes`. This sets `reference_distribution_modified`
and records the resolved configuration. It is a candidate distribution extension,
not the unchanged R control. The native long-context width caps still apply.
Larger class budgets also interact with the native split-coverage filter;
report realized classes and rejection costs instead of assuming uniform accepted
many-class coverage. Neither strict R nor this ceiling override alone establishes
a validated wide/many-class curriculum. The CLI exposes the two maximum overrides.

The `R` arm calls upstream `GraphPrior.get_batch` and `generate_dataset`;
counting wrappers consume no random numbers. Task/shape parameters are drawn
for the complete batch before any branch. `R_P1_05` replaces 5% of eligible
slots with P1 when requested F>=8. `R_P4_05` replaces 5% when requested F>=4
and the fixed task is binary or regression. Ineligible mass returns to R.
P4 never uses its autonomous 20/80 task distribution here. Native filtering
retries within the same R slot. P1/P4 retain their authored generation law and
receive no reference predictability filter. Errors abort and are not replaced.

Accounting distinguishes branch draws, source-selected slots, raw dataset
attempts, graph proposals/rejections, accepted slots and realized source mass.
The 5% is a conditional slot probability, not raw-proposal mass. Native constant
column removal can reduce accepted F; it is preserved, not counteracted by
resampling until the original F returns. Requested and accepted F are logged.
The reference may realize fewer classes than its ex ante class budget. Failed
draws carry their partial audit in `ReferenceGenerationError.audit`.

Native R intentionally uses generation-time full-table feature/target scaling,
class split repair and an ExtraTrees query-label filter. This does not put query
labels in model inputs. Corrected category meta-sampling, converters, cardinality
and IID switches remain at released defaults (false). Those corrections would
be new ablations, not a reproduction of the released generator. The reference
does not export categorical type identities after ordinal encoding, clipping,
scaling and permutation. The Episode adapter therefore treats those delivered
numeric columns as numeric; it does not invent category types. The native tuple
is also returned for a native model/codec path. Authored additions have their
own categorical flags. Comparing the Episode path across arms holds the local
codec fixed and is a prior experiment, not full TabICLv2 system reproduction.

Set `PYTHONHASHSEED=0` **before Python starts**: upstream iterates over a set of
feature-group names. Setting NumPy and Torch seeds alone does not make different
processes repeat. Use CPU worker processes, not concurrent threads: the adapter
temporarily mounts the `tabicl` import namespace and restores RNG/thread state.
The entire upstream distribution requires Python >=3.10. The retained prior
subset was smoke-tested on this project's Python 3.9.6/Torch 2.8.0 environment;
use a supported Python >=3.10 environment for full official model training.

This is executable source generation and contract evidence. No released model
checkpoint reproduction, trained transfer result, or MI355X throughput is claimed.
