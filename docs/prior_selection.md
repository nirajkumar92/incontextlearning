# Selecting the standard pretraining prior

The current research recommendation is to **select a stronger standard prior before choosing the final run**. The runnable R+P1 configuration remains a reproducible pilot. It is not evidence that the final bank should be approximately 95% TabICLv2.

The proposed starting mixture is **50% reference graphs, 20% direct forests, 15% categorical hierarchy, 10% smooth/local functions and 5% sparse interactions**. These are designed exploration weights. They are neither measured optimal weights nor a claim of superiority over Kumo or TabPFN-3.5.

## What to use

| File | Purpose and status |
| --- | --- |
| [prior_selection_v1.json](../research/specs/prior_selection_v1.json) | Exact proposed distributions, observation rules, eligibility, responses and experiment allocation. Implemented by the new sampler/laws. **Research specification, not a trainer config.** |
| [plan_prior_selection.py](../scripts/plan_prior_selection.py) | Executable accounting and dependency planner; emits the 40 trial definitions; the materializer below turns a resolved phase into trainer configs. |
| [prior_selection_plan.json](../research/results/prior_selection_plan.json) | Saved planner output, including specification hash and eligibility-adjusted exposures. |
| [prior_information_analysis.py](../research/probes/prior_information_analysis.py) | Executable exact information calculations and independent CPU simulations. |
| [prior_information_analysis.json](../research/results/prior_information_analysis.json) | Actual numerical results with seeds, simulation counts, uncertainty and claim limits. |
| [prior_selection_evidence.md](../research/reviews/prior_selection_evidence.md) | Paper-level evidence, including negative combinations and architecture interactions. |
| [main_recipe.md](main_recipe.md) | Earlier executable reference pilot and its control settings. |
| `src/tabular_foundation/challenger_prior.py` | Task/shape-first R/F/H/S/I generator; H fallback and within-family retries. |
| `src/tabular_foundation/selection_laws.py` | Independent calibration, sampled responses, missingness and coarsening. |
| `scripts/materialize_prior_selection.py` | Freezes phase-specific configs, reviewed dependencies and source-spec hash. |
| `scripts/submit_training.py` | Dry-run or Slurm submission with an allocation wall-time ceiling. |
| `src/tabular_foundation/selection_evaluation.py` | Real-data panels, paired scores, source-lineage intervals and latency records. |

The TeX report's **Selecting a stronger standard prior** section gives the complete rationale, equations, generator procedure, selection objective and budget. Existing generator descriptions remain in the following section for reproducibility.

## Generate the experiment plan now

From the repository root:

```bash
.venv/bin/python scripts/plan_prior_selection.py \
  --spec research/specs/prior_selection_v1.json \
  --episodes 64000000 --output runs/prior_selection_plan.json

.venv/bin/python research/probes/prior_information_analysis.py \
  --output runs/prior_information_analysis.json

.venv/bin/python -m pytest -q -W error \
  tests/test_prior_selection_plan.py tests/test_prior_information_analysis.py
```

These commands need CPU only. The planner does not estimate MI355X throughput, launch trials, generate new training priors, or choose a winner without results. The 64M argument is an accounting scenario. Package hours remain binding; use measured throughput to choose actual task volume.

## The proposed bank

First draw the task and shape using the existing reference contract, then the mechanism. Hierarchy requests with fewer than eight columns return to R. All other new families support positive width. Before that fallback, 64M requests imply 32M graph, 12.8M forest, 9.6M hierarchy, 6.4M smooth/local and 3.2M interaction episodes. Under the current stage/width law, hierarchy contributes approximately **8.975M** accepted source slots and the returned **0.625M** goes to R. These are expectations, not observed execution counts.

Independently choose the observation mode: **50% identity, 15% MCAR, 15% MAR, 10% MNAR, 10% coarsening**. Labels are sampled before measurements are hidden or coarsened. Identity preserves intrinsic mechanism noise. No post-observation tree filter removes ambiguous outcomes. MAR reserves observed driver columns; for width one it falls back to MCAR and records that decision.

The new families use independent calibration rows from the same world. The unchanged reference adapter does not expose such a pool, so its observation parameters use **clean support features only**, without support/query labels. Never append extra native rows as a hidden calibration set: that changes the reference law. R remains numerically represented; the wrapper cannot reconstruct discarded categorical types.

F and S test explicit piecewise and smooth prediction geometry. H modifies the catalog occupancy law of P1 while keeping its observable group/metadata structure. It disables all legacy masking and information-losing rendering, including grid rounding and clipped exponential transforms, before applying the single declared observation mode. I tests sparse interactions across the full informative dimension, rather than only the first sixteen columns. The specification defines the exact probabilities and numerical fallbacks. These versions must not be silently substituted with similarly named old generators.

## What the CPU analysis tells us

The normal random-effects calculation shows how the optimal amount of pooling changes with category count, noise and effect variation. The occupancy calculation distinguishes unseen category identities from the probability mass of unseen query categories. The coarsening calculation establishes a visible-input error floor that differs from an oracle seeing latent measurements. The parity comparison demonstrates that occupancy cannot rank learning difficulty across unrelated mechanism families.

These results justify contracts and sampling axes. They do **not** select the mixture's weights, prove neural learnability at a chosen budget, or predict benchmark rankings. No single descriptor distance, teacher score or synthetic Bayes risk substitutes for real transfer.

## Experiments within the existing allocation

| Phase | Comparisons | GPU-hours |
| --- | --- | ---: |
| P | R; R+observation; R+F20%; R+H15%; R+S10%; R+I5%; full mixture clean; full mixture observed. Eight arms × two seeds × 300 hours. | 4,800 |
| M | R fractions .70/.50/.30, remaining mass F:H:S:I = 4:3:2:1. Three arms × two new seeds (2/3) × 300 hours. | 1,800 |
| A | R and selected challenger, each on compressed and persistent-cell architectures. Four cells × two seeds × 400 hours. | 3,200 |
| V | Complete reference and finalist, each with two fresh seeds (4/5) × 700 hours. | 2,800 |
| F | From a common standard parent: standard replay; fraud reservoir+positives; fraud full selector. Three arms × two seeds × 400 hours. | 2,400 |
| **Screen total** | **40 proposed runs; classification and regression share each package allowance.** | **15,000** |

This replaces the previous 15,000-hour screen. Separate head, codec, optimizer, length and sharing sweeps are deferred; their existing settings stay fixed. The 10,000-hour scale phase remains: 2,000 for two sizes/two seeds, 6,000 for reference/challenger bridges, and 2,000 for confirmation or unresolved rankings. Systems 5,000 + screens 15,000 + scaling 10,000 + final 12,000 + evaluation 5,000 + reserve 3,000 = **50,000 GPU-hours**.

Each run completes a predeclared schedule, with generation stalls and evaluation on allocated devices counted. Equal-cost comparisons also report task counts, class/shape exposure and CPU generation cost. If the allocation cannot complete the intended curriculum, reduce named arms before launch. Do not compare a completed short schedule against a partial long one.

## How the winner is chosen

Register development and confirmation datasets by source lineage before running the search. Target at least 20 development and 10 confirmation lineages **per task type**. These counts give coverage, not guaranteed statistical power. Keep final TabArena folds and customer evaluation periods out of mixture tuning. Related tables, derived targets and duplicates belong in the same lineage.

Use paired per-dataset improvement over R: reduction in classification log loss or regression MSE divided by a constant-predictor loss computed from training data. Use Jeffreys-smoothed training class probabilities for the classification constant predictor. Compute regression search losses in support-standardized target units. Divide by max(1e-6, training constant-predictor loss), fixing the denominator before query evaluation. Average folds and seeds within datasets, datasets equally within task type, then the three task types equally. Report raw AUC/log-loss/RMSE and calibration alongside this search objective.

Exploratory ranks select candidates for confirmation, not claims. A finalist passes only if the overall 95% paired lineage-bootstrap interval excludes zero improvement and each task type's lower bound is above −.005 normalized units. The .005 margin is a predeclared tolerance, not a literature-derived optimum. Use 10,000 lineage resamples and report seed variation separately. Confirmation data used repeatedly become development data. If no candidate passes, no stronger bank has been selected.

Compare cold context setup, cached GPU batch prediction, memory and accuracy under the same support and 1/4/8-view budgets. Performance has priority. A more useful prior can reduce inference cost only by permitting less context, fewer views or a smaller architecture at comparable accuracy; its name does not make a fixed model faster.

## Before the accelerator study

The sampler, common laws, both architecture families, training integration and paired evaluation are implemented. The remaining preparation is empirical: audit native-shape worlds, measure the three curriculum stages on the intended hardware, and register real-data panels. Small local smoke runs verify execution; they do not establish how many tasks fit into a GPU-hour budget.

The `persistent_small` comparator has 41,089,248 parameters with the candidate heads and finance adapter. It retains width-256 cells through twelve stages, each with one column ISAB (128 inducing summaries per column) and one row MAB (eight heads, FFN width 1,024). Eight width-256 summary tokens are projected to width 512 at final readout. It has no row-level ICL trunk; query KV grouping therefore has no effect in this model. Support labels enter early feature states, and queries use cached support-derived summaries. Its cross-row communication is compressed through those summaries. This is an authored comparator, not an implementation of TabPFN or EXAONE. Its different size and depth mean phase A compares complete systems at equal allocated cost.

## Run the implemented recipe

Install the package as described in the README and activate that environment on every node. All commands below run from the repository root. Reference generation needs `PYTHONHASHSEED=0` before interpreter startup.

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip install -e '.[test,reference]'
python -m pytest -q -W error
python -m tabular_foundation.train \
  --config configs/smoke_selection.json --output runs/selection_smoke
```

The smoke performs three tiny CPU updates, two tasks per update, using Muon/AdamW and two ordered generator processes. It uses the actual Q sampler but overrides shapes to16 support rows, 8 queries and 8 features. It is deliberately not the native research distribution. Its `last.pt`, `manifest.json`, `metrics.jsonl` and `completion.json` demonstrate the complete training interface. Use a new output directory; exact recovery requires `--resume` and the original directory.

No complete synthetic dataset has to be generated in advance. Each rank requests deterministic scheduled episodes from a bounded pool of CPU processes. The candidate uses `producer_workers=2` and `prefetch_tasks=4`. Results are consumed in schedule order, so a slow family does not lose its assigned share. Actual source counts and observation/fallback counts are logged. The queue is bounded by task count, not bytes; CPU encoding/transfers and unfused kernels still need profiling. More workers do not by themselves guarantee GPU saturation.

### Audit the generators

```bash
python scripts/audit_selection_priors.py \
  --families R forest hierarchy smooth_local sparse_interaction \
  --stages 1 2 3 --worlds-per-family-stage 1000 \
  --output runs/selection_prior_audit.json
```

This audit uses the native shape law and the declared later-stage envelope by default. Counts are requested family slots: narrow H requests return to R, and actual family counts are reported. Increase requested draws if at least 1,000 actual H worlds are needed. It records failures, retries, observed class coverage, corruption, no-signal cases and available occupancy diagnostics rather than deleting difficult worlds. The independent 4,096-row calibration pools belong to the generators, not to the model context. A `--shape 32 16 8` override is useful for a quick diagnostic but cannot establish native-shape coverage. The large audit is real work and has not been replaced by a claim about unit-test coverage. Support-cross-validated finite-context teacher studies remain a separate analysis; no teacher score automatically rejects a family or selects a mixture.

### Freeze phase configs

Measure end-to-end update cost for each stage and chosen architecture, including generation, transfers, collectives, checkpoints and evaluation reserve. Use [prior_volume.md](prior_volume.md) to reconcile task targets with measured cost. Freeze a complete horizon before each comparison. The following 1,000-update horizon demonstrates the interface; it is not a recommendation inferred from MI355X measurements.

```bash
python scripts/materialize_prior_selection.py --phase P \
  --steps 1000 --allocated-gpus 64 --output runs/phase-P
```

This writes sixteen configs and `materialization.json`, which lists each exact config path, output path, source hash and submission command. The common `--steps` option is convenient for smoke or equal-episode comparisons. For the primary equal-budget study, use `--horizons measured_horizons.json` instead: its JSON object maps every trial ID in the phase to a measured full-schedule update count. Different costs can then produce different episode counts while preserving the same stage proportions. Missing or extra run IDs are rejected. Equal ceilings do not guarantee exactly equal realized spending; report both. It does not launch jobs. Standard horizons must be multiples of 100; phase F and final fraud horizons must be multiples of 400. These constraints preserve 90/9/1 stage shares and finance four-use world groups. The initial P arms all use the same compressed model and output contracts. Change to a new directory when freezing a revised experiment.

Later phases require a decisions JSON. This records scientific decisions already made from results; it is not another user-permission flow. For example, **only if the results support this choice**, an `after_P` entry could be:

```json
{
  "after_P": {
    "reviewed": true,
    "evidence": "runs/analysis/phase-P-comparison.json",
    "mechanism_weights": {
      "R": 0.5, "forest": 0.2, "hierarchy": 0.15,
      "smooth_local": 0.1, "sparse_interaction": 0.05
    },
    "observation_weights": {
      "identity": 0.5, "mcar": 0.15, "mar": 0.15,
      "mnar": 0.1, "coarsen": 0.1
    }
  }
}
```

The materializer checks structure and nonempty evidence, not the truth of a scientific judgment. Never mark an example as a measured winner. Phase M uses `after_P` and redistributes non-R mass at the three graph anchors. Phase A uses `after_M`. Phase V uses `after_A`, including `model: "base"` or `"persistent_small"`. Phase F uses `after_V`, including the selected model and an existing compatible `parent_checkpoint`; all three continuation arms initialize from that same checkpoint with fresh optimizers and 10%-of-standard peak learning rates. The reservoir and full-selector arms use paired populations and query draws. Other implemented size names are accepted for later explicit decisions, but do not alter the fixed phase-A two-architecture comparison.

```bash
python scripts/materialize_prior_selection.py --phase M --steps 1000 \
  --decisions runs/decisions.json --output runs/phase-M
```

The full-volume starting templates are `configs/selection_candidate_standard.json` (250,000 updates,64M episodes,9,000 GPU-hours) and `configs/selection_candidate_fraud.json` (4,000 updates,1.024M macroepisodes,3,000 GPU-hours, parent required). They express the Q hypothesis and provisional volumes. After selection and size/cost measurements, use `FINAL_STANDARD` and `FINAL_FRAUD` with a `final` decision to freeze the chosen pair. Set `parent_checkpoint` to the newly trained standard checkpoint before materializing final fraud. Preserve the standard checkpoint for general tasks.

### Submit one run on 64 GPUs

Take the exact config and output paths from the materialization manifest:

```bash
python scripts/submit_training.py --config PATH_FROM_MANIFEST \
  --output OUTPUT_FROM_MANIFEST \
  --sbatch-arg=--account=YOUR_ACCOUNT \
  --sbatch-arg=--partition=YOUR_GPU_PARTITION
```

The command first prints a dry run. Add `--submit` to submit. It requests eight nodes with eight GPUs each, one torchrun launcher per node, and enough CPU cores for the configured producer processes plus training threads. The actual Python interpreter is propagated through `TFM_PYTHON`. Both CUDA and ROCm use PyTorch's `cuda` API and the same training code; install the matching PyTorch build in advance. Site-specific partition, account, memory and network settings still belong to the cluster environment.

Pass `--evaluation-reserve-gpu-hours H` to the materializer to reserve H hours **per trial** inside its package allowance. The generated training cap is the package cap minus H. The manifest records the training cap, reserve and a separate evaluation allocation ID. The default is zero and is labeled unreserved; it does not make subsequent evaluation free. Select the reserve from measured evaluation cost. Evaluation commands do not automatically update the ledger.

The hard scheduler wall time is the smaller of remaining run and project allowances divided by allocated GPUs. The trainer also checks its budget between updates. A killed allocation can consume work after the last durable checkpoint; reconcile scheduler-billed GPU-hours before resubmitting. The ledger does not reserve future spending across concurrent jobs. Serial full-cluster runs are the default operating procedure. A time cap can stop before the planned horizon: `budget_stopped` is not a successfully completed comparison.

Record actual scheduler-billed cumulative consumption, including separate evaluation jobs, using the exact run or evaluation ID from the manifest:

```bash
python scripts/record_allocation.py --ledger runs/project_gpu_hours.json \
  --run-id EXACT_RUN_OR_EVALUATION_ID \
  --cumulative-gpu-hours ACTUAL_CUMULATIVE_SCHEDULER_GPU_HOURS
```

Use cumulative totals for that ID, including all attempts and resumes. Repeating a total is idempotent; decreasing it is rejected. The command preserves actual overspend and signals it rather than discarding the record. Check each trial's training-plus-evaluation total against its package allowance as well as the overall project ledger. Allocation reconciliation does not reserve future spending or automatically parse scheduler reports.

## Evaluate the standard panel

The evaluator accepts an explicit panel JSON; it does not download datasets or certify their lineage. NPZ files use the existing data contract (`X`, `y`, `categorical`, stable row `ids`); missing feature values are NaNs and categorical IDs are numeric. Classification labels use a declared legal vocabulary 0 through K-1, fixed before reading query outcomes. Use [data_and_evaluation.md](data_and_evaluation.md) for conversion and the separate chronological finance evaluator.

```json
{
  "role": "development",
  "datasets": [
    {
      "dataset": "example_binary", "lineage": "independent_source_A",
      "fold": "0", "task": "binary", "classes": 2,
      "support": "data/support.npz", "query": "data/query.npz"
    }
  ]
}
```

Paths are relative to the panel JSON. This one-split example illustrates the format only: the selection comparison requires all three task types and the declared lineage coverage. Row IDs must identify original records consistently across files; independently resetting IDs per split is wrong. Reject overlap before creating the panel. If stable IDs are unavailable, set `disjoint_split_verified: true` and a nonempty `split_provenance` describing the verified split or its immutable manifest; a declaration cannot substitute for an actual provenance audit. Identical support/query files are rejected. Use `--exclude-panel` to reject declared lineage reuse across development, confirmation and final panels. Derived or duplicate datasets must first be assigned their common source lineage by the researcher.

```bash
python -m tabular_foundation.selection_evaluation evaluate \
  --panel panels/development.json --checkpoint runs/ARM_SEED/last.pt \
  --output runs/ARM_SEED/development-v1.json --device cuda \
  --seed 91 --views 1 --query-batch-size 512 --context-limit 32768 \
  --exclude-panel panels/confirmation.json --exclude-panel panels/final.json

python -m tabular_foundation.selection_evaluation compare \
  --reference runs/R_SEED0/development-v1.json runs/R_SEED1/development-v1.json \
  --candidate runs/Q_SEED0/development-v1.json runs/Q_SEED1/development-v1.json \
  --min-lineages 20 --repeats 10000 --output runs/phase-P-comparison.json
```

Training seeds come from checkpoints; the `--seed` argument to `evaluate` controls paired inference views. It is not a new training replicate. The comparison requires distinct checkpoints and matching training seeds, data, preprocessing and inference policy. Failures remain in the output and invalidate comparison; they are never dropped from the mean. Repeat with 4 and 8 views under the same policy for the accuracy/latency comparison. Views run sequentially: memory records the peak for that policy, not eight simultaneously resident context caches. Accelerator identity and software versions are recorded; independent hardware profiling is still required before interpreting timings.

For confirmation use the frozen finalist, fresh training seeds 4/5, a `confirmation` panel, at least 10 source lineages per task type and 10,000 bootstrap replicates. Reduced diagnostic settings cannot emit confirmation evidence. Lineage bootstrap uncertainty is conditional on the supplied trained seeds; seed variability is reported separately, rather than pretending two seeds characterize all training randomness. Confirmation is one predeclared comparison, not a repeated search on the same panel.

## Code-to-report contract

| Report component | Runtime implementation | Executable check |
| --- | --- | --- |
| R/F/H/S/I selection; native task and shape first | `challenger_prior.py`, native callback in `reference_prior.py` | `test_challenger_prior.py`, `test_reference_prior.py` |
| Class calibration, regression noise and transforms | `selection_laws.py:ResponseCompiler` | `test_selection_laws.py` |
| Identity/MCAR/MAR/MNAR/coarsening | `selection_laws.py:ObservationModel` | driver, calibration and row-invariance tests |
| Persistent cells and compressed rows | `model.py:build_model` | `test_persistent_model.py`, existing model/cache tests |
| Muon/AdamW and ordered producers | `optim.py`, `producer.py`, `train.py` | `test_selection_training.py` and full smoke |
| P/M/A/V/F and final phases | planner, materializer and submission scripts | materialization, budget and curriculum tests |
| Proper-loss objective and paired intervals | `selection_evaluation.py` | `test_selection_evaluation.py` |
|1e-4 finite finance populations and matched selectors | `finance_prior.py`, `retrieval.py` | finance tests and paired population/query test |

MoE, learned adaptive mixture weights, feature feedback, auxiliary losses and real-data continuation remain optional specifications without current funding. They are not silently enabled by these configs. Competitive accuracy, full native-shape coverage and MI355X/NVIDIA throughput remain experiments to execute, not properties established by this implementation.
