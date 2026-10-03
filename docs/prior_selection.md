# Selecting the standard pretraining prior

The current research recommendation is to **select a stronger standard prior before choosing the final run**. The runnable R+P1 configuration remains a reproducible pilot. It is not evidence that the final bank should be approximately 95% TabICLv2.

The proposed starting mixture is **50% reference graphs, 20% direct forests, 15% categorical hierarchy, 10% smooth/local functions and 5% sparse interactions**. These are designed exploration weights. They are neither measured optimal weights nor a claim of superiority over Kumo or TabPFN-3.5.

## What to use

| File | Purpose and status |
| --- | --- |
| [prior_selection_v1.json](../research/specs/prior_selection_v1.json) | Exact proposed distributions, observation rules, eligibility, responses and experiment allocation. **Research specification, not a trainer config.** |
| [plan_prior_selection.py](../scripts/plan_prior_selection.py) | Executable accounting and dependency planner; emits 40 proposed trials, not training jobs. |
| [prior_selection_plan.json](../research/results/prior_selection_plan.json) | Saved planner output, including specification hash and eligibility-adjusted exposures. |
| [prior_information_analysis.py](../research/probes/prior_information_analysis.py) | Executable exact information calculations and independent CPU simulations. |
| [prior_information_analysis.json](../research/results/prior_information_analysis.json) | Actual numerical results with seeds, simulation counts, uncertainty and claim limits. |
| [prior_selection_evidence.md](../research/reviews/prior_selection_evidence.md) | Paper-level evidence, including negative combinations and architecture interactions. |
| [main_recipe.md](main_recipe.md) | Existing model, reference-pilot and finance-continuation commands. |

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

## Work required before launching the new arms

1. Implement the exact five-way sampler, new family versions and observation wrapper. Existing R is ready; old P1/P0 components are reusable building blocks, not drop-in reproductions of this specification.
2. Verify typed identities, deterministic replay, probability conservation, missingness drivers, calibration separation, target moments and retained no-signal cases. In particular, H excludes signed-square rendering because its heavy-tailed interaction score need not have a finite fourth moment.
3. Audit at least 1,000 independent worlds per family/stage and compare finite-context teacher learning curves on an independent query set. Teacher selection uses support cross-validation, never the evaluation queries. Log observed quantities rather than trusting requested caps.
4. Complete the compact architecture and paired real-data runner, then measure backend cost for every stage. Existing tests and accounting do not constitute those implementations.
5. Resolve the planner's implementation/dependency gates and create actual trainer configs from the frozen selected contracts. Preserve hashes and fresh run directories.

No existing runtime configuration was changed to pretend these new arms already work. The report and coordinating specification now distinguish the executable pilot from the proposed selection program.
