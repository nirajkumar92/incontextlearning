# Running the frozen model through TabArena

This repository includes an optional AutoGluon adapter and a runner for the upstream TabArena pipeline. They are the starting point for an official benchmark experiment; the NPZ evaluator elsewhere in this repository is not a TabArena reproduction. **The adapter has been reviewed against upstream source but has not been executed with AutoGluon or TabArena in this workspace.** Do not report its integration, leaderboard coverage or ranking as validated until the checks below pass.

The adapter uses one explicit pretrained checkpoint. Each `fit` constructs a context from that fold's training rows; it does not update network weights. Validation labels do not enter the context. The default runner leaves validation and bagging to TabArena, using one fixed configuration and no HPO. Its `outer` option is a separate unbagged diagnostic outside the official model protocol. The current upstream examples distinguish these two tracks. [Official model quickstart](https://github.com/autogluon/tabarena/blob/main/examples/benchmarking/run_quickstart_tabarena_model.py), [outer diagnostic](https://github.com/autogluon/tabarena/blob/main/examples/advanced/run_quickstart_tabarena_model_without_bagging.py).

## Create a separate environment and pin the source

Use Linux and Python 3.12 for this optional integration. The upstream installation currently uses its `packages/tabarena` workspace package and permits prerelease dependencies. Keep this environment separate from the research training environment, particularly when installing the ROCm build of PyTorch. Follow the matching AMD/PyTorch installation instructions for the actual machine rather than replacing a working ROCm wheel blindly. [Upstream installation](https://github.com/autogluon/tabarena#installation).

Run these commands from a directory where you keep external benchmark checkouts. Replace the example paths and revision with your actual reviewed values. The placeholder revision is intentionally not a claimed compatibility pin.

```bash
git clone https://github.com/autogluon/tabarena.git tabarena-benchmark
cd tabarena-benchmark
git checkout --detach YOUR_REVIEWED_40_CHARACTER_COMMIT
uv venv --seed --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow -e './packages/tabarena[benchmark]'
uv pip install -e '/absolute/path/to/pfn prior'
git rev-parse HEAD > ../tabarena-revision.txt
uv pip freeze > ../environment-freeze.txt
```

Keep provenance outputs outside the TabArena checkout: the runner requires a clean checkout. The exact commit and resolved package versions are recorded again in its output manifest. After a successful integration test, retain that commit and environment lock with your results; rerunning a moving `main` branch is not the same experiment. The adapter follows `AbstractTorchModel` and the current model wrapper API. [AbstractTorchModel source](https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/abstract/abstract_torch_model.py), [TabICL wrapper](https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/models/tabicl/model.py).

## Validate the adapter before a benchmark allocation

First run a direct fit, prediction, save/load and prediction check on a small training/held-out split in the optional environment. Use the real intended checkpoint; a tiny untrained smoke checkpoint only tests execution.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from tabular_foundation.tabarena_adapter import FrozenTabularFoundationModel

X = pd.DataFrame({
    'numeric': [0., 1., np.nan, 3., 4., 5., 6., 7.],
    'category': pd.Categorical(['a', 'b', 'a', None, 'c', 'b', 'a', 'c']),
    'other': [2., 1., 0., 3., 1., 2., 4., 5.],
    'flag': [0, 1, 0, 0, 1, 1, 0, 1],
})
y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
model = FrozenTabularFoundationModel(
    path='/absolute/path/to/adapter-smoke', name='FrozenICL',
    problem_type='binary', eval_metric='log_loss',
    hyperparameters={'checkpoint_path': '/absolute/path/to/last.pt',
                     'context_limit': 32768, 'query_batch_size': 64},
)
model.fit(X=X.iloc[:6], y=y.iloc[:6], num_cpus=2, num_gpus=0)
before = model.predict_proba(X.iloc[6:])
saved_path = model.save()
loaded = FrozenTabularFoundationModel.load(saved_path)
after = loaded.predict_proba(X.iloc[6:])
np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
```

Repeat for multiclass classification and regression, including a category first observed in the held-out rows, missing entries, absent classes in a child fold where the parent supplies the legal class universe, and the requested GPU resource. Run the same checks under the upstream bagging and Ray execution paths before scheduling the full evaluation. A source-level check cannot replace these tests.

## Run the development subset, then the locked evaluation

```bash
python '/absolute/path/to/pfn prior/scripts/run_tabarena.py' \
  --checkpoint '/absolute/path/to/last.pt' \
  --tabarena-repo '/absolute/path/to/tabarena-benchmark' \
  --expected-revision YOUR_REVIEWED_40_CHARACTER_COMMIT \
  --output '/absolute/path/to/results/tabarena-development' \
  --track official --gpus 1 --cpus 4
```

The default subset is the three small datasets from the upstream model quickstart. Use `--datasets name1 name2` to choose development tasks and `--full` for the full pinned context. Use `--ray` only after importability and worker access to the checkpoint have been checked. `--baseline LightGBM` adds an upstream registry baseline at its default configuration; it does not create a tuned tree comparison. The comparison step also uses upstream cached methods.

For the separate unbagged ICL experiment, set `--track outer` and choose a distinct output directory. Report it as a diagnostic with its actual training-context size. Do not use its score as evidence of winning the official bagged/tuned track. Similarly, one fixed configuration cannot establish dominance over every competitor's tuning/ensemble setting. Select the final configuration on development tasks, freeze it and the checkpoint, then evaluate the locked tasks using the agreed upstream protocol.

## Context, serialization and timing contracts

- Categorical mappings are fitted on observed training values only, not on an attached pandas category vocabulary. Missing values remain `NaN`. Unseen nominal values receive deterministic 51-bit hashed identities in a disjoint numeric-code range; collisions are possible but extremely unlikely for ordinary held-out cardinalities. Numeric feature scaling still fits only on support inside the model codec.
- The declared classification universe comes from AutoGluon's legal training protocol. The adapter never derives it from held-out labels. Binary outputs follow AutoGluon's one-dimensional positive-class probability convention; multiclass outputs retain their class dimension; regression outputs are means.
- `context_limit` raises an explicit error when a fold is too large. There is no implicit subsampling, automatic tree fallback, finance-context substitution or successful-task-only leaderboard policy. Review failed-task counts and upstream imputation before interpreting an aggregate.
- A serialized fit contains CPU support arrays, its preprocessing state, checkpoint path and digest. It omits the live network and context cache. Loading reconstructs a fresh network and context lazily, so stale process-specific cache ownership cannot survive pickle. Keep the immutable external checkpoint accessible at the recorded path on every worker. Report its storage separately: the adapter's pickle size excludes those external weights, and `get_info()` records that exclusion and the checkpoint's file size.
- The upstream shared-weight mechanism is deliberately not enabled in this first adapter. Each fit owns its weights; this is explicit isolation, with additional loading and memory costs. The live network and cached context are rebuilt after a device change. Any later shared-weight implementation must preserve per-fit contexts and receive separate tests.
- Fit timing includes checkpoint verification/loading and context construction. Loading a saved fit makes the first subsequent prediction cold. Report cold setup, first prediction, warm prediction, total bagging/HPO cost and peak memory separately. The adapter checks its fit time limit after setup; it cannot interrupt a running device kernel, so the outer worker timeout remains authoritative.

The runner refuses mismatched commits, imports from another TabArena installation, dirty benchmark checkouts and reuse of an output directory for a different experiment. Its manifest records the checkpoint hash, framework revision, resolved versions, requested track, planned job count and returned result count. Missing returned jobs stop aggregation; a returned result can still contain a failure, so inspect the per-task statuses and coverage. Upstream registration can narrow comparisons to valid tasks, which must not become an unreported successful-task-only score. The manifest records pipeline completion, not maintainer approval or a state-of-the-art claim. [Context execution and registration source](https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/contexts/abstract_arena_context.py).
