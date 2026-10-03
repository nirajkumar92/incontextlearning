# Data, prediction and evaluation

## Interchange format

A table NPZ contains `X` (rows × features, float64 recommended), `categorical` (one Boolean per feature), and `y` for labeled tables. Nominal values use consistent numeric IDs, never arithmetic category effects. NaN is a missing feature. Optional `feature_names` records order; `ids` identifies unique records. Finance snapshots additionally contain numeric `event_time` and `label_available_time` in one documented timestamp unit/timezone.

Unresolved outcomes do **not** belong in labeled support with target zero. Select the eligible labeled history at the prediction cutoff upstream. An old unreviewed transaction is not necessarily fully adjudicated. The timestamp checks verify supplied metadata, but cannot reconstruct incorrectly computed future-looking features from a matrix.

`scripts/prepare_csv.py` converts a CSV using an explicit schema. Edit `configs/data_schema.example.json` to list real features, categorical columns, target, declared class vocabulary and metadata columns:

```bash
python scripts/prepare_csv.py --csv train_snapshot.csv \
  --schema my_schema.json --output data/train.npz --drop-unlabeled
```

Build validation and test snapshots with the same feature order and nominal encoding rule. The converter hashes nominal values deterministically and checks collisions within each processed file; a persisted collision-free company vocabulary remains preferable where available. Feature transforms and target vocabulary must not be selected from test outcomes. A query-only schema can omit `target`.

For standard benchmark tasks, use the benchmark's released splits and allowed class universe. For finance, preserve event/arrival/label-availability history, construct past-only aggregates, and select validation periods whose outcomes were known before the locked test prediction period. Recurring accounts may appear across chronological periods; an all-unseen-account test answers an additional question.

## Predict with frozen weights

```bash
python -m tabular_foundation.predict \
  --checkpoint runs/my_training/last.pt \
  --support data/train.npz --query data/test_features.npz \
  --task binary --device cuda --output runs/predictions.npz
```

Standard dense inference rejects support beyond `--context-limit` (32,768 by default) instead of silently selecting rows. The user must select and evaluate an explicit alternative. Query chunks share a support cache; other queries cannot change a prediction.

Finance inference uses the complete eligible history to build its shared index, while each route has a bounded transformer context:

```bash
python -m tabular_foundation.predict \
  --checkpoint runs/my_training/last.pt \
  --support data/eligible_snapshot.npz --query data/future_features.npz \
  --task binary --finance --cutoff 1760000000 --device cuda \
  --reference-counts '[2999700,300]' --output runs/fraud_predictions.npz
```

The timestamp and counts above illustrate syntax; replace them with actual verified inputs. `--reference-counts` means a separately identified representative, completely adjudicated **past** cohort. `--complete-cohort` instead asserts that every row of the supplied support pool belongs to such a cohort. Neither option is justified merely because all supplied reviewed alerts have labels. Without a valid reference, omit both; the model then has less information and its population calibration requires separate evidence.

By default finance selection searches all eligible negatives. `--search-cap 65536` is an explicitly weaker latency control. `--cache-limit` controls device-resident route caches; eviction recomputes a context when revisited. A new history/reference snapshot requires rebuilding. `--hurdle --task regression` selects the nonnegative loss schema with a zero atom. Ordinary amount regression leaves `--hurdle` off.

GPU prediction defaults to BF16 matrix autocast; `--fp32` provides a numerical control. CPU uses FP32 model arithmetic with an FP64 reference codec. No GPU timing has been established locally.

## Evaluate and profile

```bash
python -m tabular_foundation.evaluate \
  --checkpoint runs/my_training/last.pt \
  --support data/train.npz --validation data/validation.npz --query data/test.npz \
  --task binary --minimum-precision 0.8 --review-budget 100 \
  --output runs/metrics.json

python -m tabular_foundation.profile \
  --checkpoint runs/my_training/last.pt \
  --support data/train.npz --query data/test.npz --task binary \
  --device cuda --repeats 10 --output runs/latency.json
```

Add the same finance/cutoff/reference options to both commands for the finance path. The evaluator chooses its binary threshold only on validation and uses it unchanged on test. An infeasible precision target produces no alerts and records this fact. This is empirical threshold selection, not a confidence guarantee. `review-budget` is for the supplied batch; for a daily budget, evaluate each day separately, then report the daily and aggregate counts. Do not substitute one global top-k for daily operations.

The metrics include average precision with tied-score grouping, ROC-AUC, log loss, Brier score, TP/FP/FN/TN and false positives per million negatives. Empty-positive periods remain visible. `metrics.blocked_bootstrap` resamples whole supplied blocks; choose defensible time/entity/campaign units. It does not infer which records are independent.

The profiler reports model loading, context setup, first-query execution and warm median/p95 separately. Finance first-query execution includes route prefills activated by that batch. A cache too small for all active routes can keep incurring misses on warm repeats. Use actual burst sizes and the end-to-end cold total when making a latency claim.

## Full-history tree baselines

```bash
python -m pip install -e '.[trees]'
python -m tabular_foundation.baselines --method lightgbm \
  --train data/train.npz --validation data/validation.npz --test data/test.npz \
  --task binary --trials 32 --seconds 3600 --output runs/lightgbm.json
```

Run `xgboost` and `catboost` as separate methods with the same history, features, temporal contract and declared tuning budget. Pass `--cutoff` for chronological finance checks. The script uses validation AP for binary selection (log loss when that period has no positives), multiclass log loss or regression RMSE, then evaluates test once. It supports weighted binary trials but does not call their scores calibrated population probabilities. The last trial can exceed the between-trial wall check; use a process/scheduler timeout for strict budgeting and record interrupted trials.

These optional tree packages are not required for training the foundation model. Their installed versions and wrapper behavior must be pinned and tested in the evaluation environment. A successful one-split script is not the full repeated official TabArena protocol; use [the benchmark integration](tabarena.md) for that comparison.

## What extreme fraud requires

At the clarified one-in-10,000 prevalence, three million records contain about 300 frauds in expectation before label filtering. A random 32,768-row context misses every positive about 3.8% of the time, but retains only about 3.3 positives on average. The selector should therefore be compared with a strong large-random-context control, as well as full-history trees.

High precision still requires measuring the negative tail. With illustrative 80% precision/recall, the necessary false-positive rate is about 20 per million negatives. A zero-false-positive test needs about 150,000 independent adjudicated negatives to bound that rate at one-sided 95% confidence. A million-row test contains roughly 100 frauds; catching 80 gives a two-sided exact recall interval around 71–87% under independence. Campaign dependence needs separate treatment.

The public BAF suite and synthetic worlds are development evidence. The company claim needs actual point-in-time data, sufficiently mature representative outcomes, full-history tree comparisons and operational review/amount metrics. Neither a synthetic smoke run nor a standard benchmark rank establishes that claim.

On this Mac, XGBoost and LightGBM needed an OpenMP runtime. The local smoke check used the runtime already bundled with PyTorch, with `DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.9/site-packages/torch/lib"`. This is a local audit-environment workaround, not a cluster setup instruction.

For multiclass trees, observed training labels are mapped to consecutive internal IDs. Declared classes absent from training receive zero predicted mass; every validation/test row still contributes to metrics, with the documented log-loss floor. Library early stopping uses validation rows belonging to observed classes and is disabled if none overlap. Candidate selection uses the entire validation period, and the output records class coverage and early-stopping exclusions.
