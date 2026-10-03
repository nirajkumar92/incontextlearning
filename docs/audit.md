# Adversarial research and implementation audit

The audit asked whether the proposed mechanisms can fail, whether the code implements their probability laws, and whether an apparent benchmark or latency gain could come from leakage, unequal evidence, or incomplete cost accounting. It did not treat passing tests as evidence of SOTA accuracy.

## Independent source-first review

A fresh three-reviewer audit revisited all nineteen retained papers before checking the proposal and code. [The consolidated review](../research/reviews/independent_audit.md) records the evidence, executed counterexamples and revised 50,000-hour decision program. That review rejected the authored bank and 315M baseline as a final selection. The follow-up now supplies a concrete 317,116,304-parameter candidate, pinned reference/graft adapter, higher-resolution heads, hash-bit nominal codec, logarithmic length scaling, query KV compression and covariate-first no-H finance generation. These address the identified counterexamples; they do not establish the candidate’s transfer accuracy. The [current selection recipe](prior_selection.md) and [source-code audit](../research/reviews/upstream_code_audit.md) supersede the earlier implementation-status statements.

## Findings corrected

| Finding | Consequence | Correction and evidence |
|---|---|---|
| Regression targets and original-unit means forced to FP32 before/after affine transforms | Large offsets with small differences collapse targets or predictions | Preserve FP64 original-unit values through normalization/inversion; six tests cover gradients, CDF/quantiles, positive moments and serving |
| Finance worlds reused four times after an 80/20 world draw | Emitted training mixture becomes 50% finance | Schedule emitted macroepisodes; test exact long-run mass and unique worlds within each batch |
| Continuous histogram used for a zero-heavy realized-loss target | An all-zero target can acquire a positive bin-midpoint prediction | Exact zero atom plus normalized positive-truncated density; quadrature, moments and gradient tests |
| Zero-atom head starts at 50% event probability despite a valid rare-event reference | Large initial loss and clipping can overwhelm other tasks | Initialize via permitted event-reference log odds, with zero learned residual |
| Enriched support treated as an ordinary population sample | Miscalibrated fraud probabilities and invalid scalar corrections | Population-query importance loss, explicit source membership/counts, no second case-control correction |
| Reference point estimate lacks cohort size | Model cannot distinguish small and large reference evidence | Append `log1p(N_reference)` as the fourteenth metadata coordinate |
| Multiclass serving scalar reference differed from training | Different metadata at deployment | Match aggregate Jeffreys event smoothing; keep separately declared per-class Dirichlet offsets |
| “Complete” world includes fresh, selectively revealed records | Labeled-pool frequency mistaken for population prevalence | Reference uses the complete mature cohort; fresh additions force a separate reference |
| Each route loss averaged equally | Small route groups over-weighted | Original macroepisode query denominator and per-query importance weights; unequal-group test |
| Different ranks locally decide when to stop | Potential collective deadlock at budget boundary | Rank-zero broadcast stop/admission decision |
| Distributed task-specific unused parameters mishandled | Wrong gradients or unintended decay/state updates | Explicit global activity flags; actual two-rank Gloo comparison with serial reference |
| Setup excluded from charged GPU time | Understated allocation cost | Timer starts before model/optimizer/checkpoint setup; allocation override and project ledger |
| Finance world constructor omitted current curriculum stage | Late finance widths remained at stage one | Pass the stage into world creation, separate stage cache keys |
| Standard multiclass sampler stayed at 3–10 classes in every stage | Later pretraining omitted the documented many-class problems | Sample the prescribed stage-specific class ranges before the support floor and cell-cap conditioning; retain stage-one replay |
| Reference seed/caches tied to update ownership incorrectly | Repeated history/index rebuilding | Stable within-group ownership; model activations still recomputed each update |
| Fractional class labels or nonfinite timestamps accepted | Invalid evaluation could appear successful | Explicit label and temporal validation |
| Nonfinite loss serialized as strict JSON | Failure before intended nonfinite-update handling | Record a null diagnostic and preserve the skip/abort policy |
| A crash leaves charged work newer than the last checkpoint | Resume can roll accounting backwards or refuse to start | Preserve the larger ledger/checkpoint charge and enforce ledger-backed run identity |
| Tree training omits a declared multiclass label | Sparse labels can fail library fitting or disappear from scoring | Map observed training labels internally; retain every validation/test row and report unseen-class coverage |
| Cache reused after weight/precision changes | Invalid or inconsistent predictions | Cache identity/version/precision guards and chunk/eviction tests |

The user subsequently clarified the operational fraud rate to **one in 10,000**. The selected prevalence distribution, central calculations and example data contracts now use that requirement. More extreme rates remain only in mathematical diagnostic grids or explicit boundary tests.

## Evidence exercised

The test suite covers all selected static families and task heads, deterministic replay, support/query stream separation, real SCM descendants, finite count/compound-amount moments, category/missing-value handling, and actual context caps. Finance tests cover finite scarcity, observed versus latent labels, reveal-time consistency, natural evaluation queries, full versus capped negative search, deduplication, reference metadata and loss reduction.

Model tests cover exact parameter accounting on the meta device, support permutations, query chunking, cached/uncached outputs and gradients, activation checkpointing, empty support, codecs and class-slot injection. Distribution tests compare normalized mass and analytic moments against numerical integration, including positive truncation and extreme values. Optimizer tests compare Muon with an independently expressed NumPy calculation and audit the full parameter partition.

Training tests compare interrupted/resumed parameters, exercise unequal route losses, and compare distributed gradients with a serial mean-macroepisode objective. The two-process Gloo test requires local socket permission and is explicitly opt-in; it was run separately. Ordinary tests do not silently claim that skipped test passed.

Separate standard and finance training smoke runs, a complete two-process mixed-profile training run, prediction/evaluation/profiling commands and CSV conversion were executed on the local CPU. Optional tree-library checks used XGBoost 2.1.4, LightGBM 4.6.0 and CatBoost 1.2.10.

A genuine three-million-row finance population was constructed lazily at the target historical rate. Its finite counts and sampled rows were checked with an explicitly capped 256-candidate diagnostic selector. This is **not** a three-million-negative full-index throughput test. The recorded future rate can differ because the campaign/time process changes risk across dates.

The earlier full suite passed **109 tests with no skips**, including the opt-in two-process test and installed tree libraries. Warnings were treated as errors. Serial and two-process three-update joint training agreed within 5.96e-8 maximum absolute parameter difference.

A subsequent training-readiness check found and fixed the missing later-stage multiclass schedule. The expanded suite now has **113 passing tests**: 112 passed in the initial run, and the two-process test passed on its isolated rerun with local socket permission after the sandbox blocked Gloo binding. No tests were skipped; warnings were treated as errors. New coverage checks the prescribed class-range frequencies, stage-one replay, support floors and cell caps, and a generated 175-class task through the model and loss boundary. This follow-up is recorded separately in `code_validation.json` so the earlier execution and source hashes retain their meaning.

The original executable record is `research/results/code_validation.json`; it records the final test counts, smoke runs and limitations. `finance_design_probes.json` records the probability identities, and `finance_large_world_smoke.json` records the finite-world check. The raw diagnostic checkpoints, arrays and XML were subsequently removed during scope cleanup; their configurations, metrics, hashes and pass counts remain in that record. Current cleanup verification is recorded in `research/results/repository_scope_audit.json`. A tiny synthetic training loss is not a benchmark result.

The independent audit added six tests for regression affine precision. That historical validation had **119 distinct passing tests**: 118 passed in the full run; the two-process Gloo test passed separately with local socket permission after sandbox binding was denied. No tests were skipped and warnings were treated as errors. The separate `independent_research_audit` record preserves this execution and its source hashes.

## Follow-up candidate and sample accounting

The current candidate keeps a standard checkpoint and initializes a separate binary-fraud specialist. It uses the actual pinned TabICLv2 graph source with an eligible 5% P1 addition. The code audit found released inference/fine-tuning code for PriorLabs, Kumo, Xiaomi and Mitra, but no complete current pretraining prior in the inspected releases; those systems are not being claimed as reproduced.

The standard run targets 64 million accepted tasks and the specialist 1.024 million macroepisodes. The [volume planner](prior_volume.md) preserves 90/9/1 stage proportions and reports required rates, budget feasibility and smaller admissible horizons from supplied measurements. These are exposure targets, not scaling-law optima. P1 eligibility, reference rejections, requested and observed classes, accepted widths, world reuse and skipped updates must remain visible in the audit.

The representation probes establish narrow properties: the previously colliding categorical pair separates under the tested bit encodings; a logarithmic temperature avoids the previous saturation in a fixed-representation calculation; a two-head query cache retains one eighth of full trunk KV storage. The learned regression-head experiment runs three seeds and shows the old head’s resolution failure after optimization. Finite-grid quantiles still omit sufficiently rare tails. Positive-support scaling improves hurdle severity resolution while preserving the exact affine Jacobian.

The reproducible [finance gradient probe](../research/probes/finance_gradient_probe.py) records actual tiny-model autograd derivatives, paired unit/entropy inputs and all source fingerprints. Two runs produced identical outputs; no optimizer update or model-performance claim is involved.

The main fraud law samples covariates before labels and hides H from both model and selector. It remains a finite risk-partition simulator; entity hazards, arbitrary smooth effects and unseen attacks are not thereby covered. Binary-only specialist training does not train the optional severity or multiclass fraud heads. Unit weighting retains the target population objective, but tiny rare-event gradients motivate separate continuation rather than assuming that a nominal shared mixture gives useful gradient allocation.

The final candidate suite passed **200 tests with no skips**, including the opt-in two-process Gloo test, with warnings treated as errors. A three-update tiny reference run and a four-update binary-specialist initialization/continuation also passed. The specialist diagnostic uses a one-million-row lazy history at prevalence 1e-4 with a deliberately capped 256-candidate search; it is not a full-history throughput result. `code_validation.json` preserves the current counts, configurations, compact metrics and source fingerprints alongside historical executions.

## Claims the evidence does not support

1. **The prior is optimal or guaranteed to beat every model.** No full-size candidate or comparable reference was pretrained, and no locked standard or company benchmark was run. The generator still embodies choices that can be wrong for real tables. The report's controlled comparisons must decide which survive.
2. **Nearest-to-center retrieval preserves all useful evidence.** It can miss a query-specific boundary or unseen fraud mode. Uniform large contexts, reservoir-plus-positive controls, more routes and feasible per-query retrieval are quality comparators. Full-history trees retain evidence the bounded network may discard.
3. **Selective labels identify population probabilities.** They do not without representative outcomes or defensible assumptions. The prior can supply a prediction, but a metadata flag cannot prove calibration.
4. **More stored MoE parameters automatically lower latency.** The selected code is dense. MoE adds a conditional-capacity experiment; it does not remove context attention costs. The proposed MoE route has not been implemented or timed.
5. **CPU tests certify 64-MI355X training.** ROCm kernels, NCCL/RCCL behavior, distributed memory, BF16 quality and sustained throughput remain unmeasured. The generator now has an ordered process-prefetch option. The NumPy/FP64 codec, bounded task queue, CPU generation and lack of pinned-transfer overlap can still starve accelerators; full-path throughput is unmeasured.
6. **The ledger guarantees a hard cluster budget.** It records allocated time but cannot reserve concurrent allocations or prevent an unexpectedly long update from exceeding its estimate. Scheduler records and hard time limits remain authoritative.
7. **Local integration checks establish benchmark results.** Tiny binary CLI runs with XGBoost, LightGBM and CatBoost exercised their tuning/evaluation paths. They are not competitive comparisons. The TabArena adapter follows inspected upstream interfaces but has not been run inside the installed official harness. The local NPZ evaluator is one-split evaluation, not an official leaderboard run.

## Before committing the final allocation

Reproduce one open reference, profile representative standard and finance workloads on one node, then verify multi-node scaling. Measure the cost of generating/indexing candidates and all routed prefills, not just the model kernel. Confirm base-model mixed-task loss stability, BF16/FP32 agreement, long-context memory and recoverable checkpointing.

Use the first controlled screens to reject components that fail either objective. Freeze the prior, size, optimizer, inference settings and data lineage before locked evaluation. The user has enough compute for a serious comparison, but the practical edge must come from measured transfer and a useful accuracy/latency frontier rather than from the volume of code or the length of the report.

## Executable prior-selection audit

The new sampler and common laws were reviewed independently of the trainer. Tests cover unchanged native R output, fixed-source numerical retries, class calibration, retained absent classes, heavy-tail restrictions, nominal relabeling, support/query independence, missingness drivers and observation-fit sources. The persistent-cell comparator passes cache/query-isolation and gradient checks. Full serial and two-rank CPU training produce bit-identical tiny checkpoints. These are implementation results, not benchmark performance.

The review corrected several issues before release: new-family numerical distractor streams initially depended on query length; generator calibration and response-draw diagnostics were incomplete; finance selector changes altered paired query draws; and the first paired evaluator accepted reduced diagnostic settings as confirmation. It now requires immutable split data across seeds, complete planned coverage, genuine checkpoint replication, disjoint splits and matched inference policies. Confirmation evidence requires the full declared bootstrap and lineage counts.

The accounting audit added measured per-run horizons, explicit evaluation reserves and cumulative scheduler reconciliation. Equal budget ceilings are distinguished from equal realized cost. Concurrent budget reservations, automatic scheduler parsing and automatic evaluation charging are not implemented. See the current machine-readable evidence in `research/results/selection_implementation_validation.json`; earlier records above remain historical.
