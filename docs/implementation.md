# Implementation and specification map

The current recommendation is the [executable prior-selection program](prior_selection.md), followed by a selected standard model and a binary-fraud specialist. The [317M reference pilot](main_recipe.md) remains a control. Runtime configurations are in `configs/`; research specifications explain intent and evidence but are not accepted training configs. [`prior_volume.md`](prior_volume.md) describes the new episode-count plan. Earlier audit results remain historical records, rather than proof that the new candidate has been trained or benchmarked.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `schema.py` | Episode validation and the explicit `model_inputs()` allowlist |
| `reference_prior.py` | Verified upstream graph-prior import, native filters, branch controls and explicit envelope extensions |
| `challenger_prior.py`, `selection_laws.py` | Five-way task/shape-first sampler, independent calibration, responses and observation laws |
| `selection_evaluation.py` | Declared standard panels, paired proper-loss objective and source-lineage bootstrap |
| `static_prior.py` | Authored P0 mechanisms, P1 categorical hierarchy and P4 count/amount laws |
| `finance_prior.py` | Finite populations, label availability, feature controls, support preparation and macroepisodes |
| `retrieval.py` | Observable reservoir codec, centers, streamed candidate selection, routing and deduplication |
| `producer.py` | Deterministic ordered CPU preparation, optional process prefetch and finance-world cache |
| `codec.py`, `model.py` | Support-fitted encoding, dense cell/row compression, ICL trunk, heads and exact caches |
| `distributions.py` | Histogram/Lomax likelihoods, exact zero atom and averaged quantile loss |
| `optim.py`, `train.py` | Optimizer partition, macroepisode reduction, distributed gradients, initialization/resume and accounting |
| `inference.py`, CLI modules | Frozen-weight prediction, evaluation, data interchange and profiling |
| `baselines.py` | Separate per-dataset tree fitting and validation |

The model allowlist includes support labels and query features, declared task/classes, permitted selection/reference metadata and the encoding seed. Query labels, importance weights, latent risk cells and audit metadata remain outside forward inputs. The trainer owns losses. Synthetic generation-time filters may use query labels; that is recorded in the reference prior and is distinct from forward leakage.

## Standard generation

`generate_reference_batch()` imports the pinned TabICLv2 prior subtree only after manifest verification. Native graph rejection, constant-column removal, class-split repair and the ExtraTrees predictability filter remain intact. The full official model, codec and trainer are not reproduced by this adapter. [Reference controls](reference_prior.md) and the [upstream audit](../research/reviews/upstream_code_audit.md) explain the differences.

The adapter fixes task, class budget and shape before independently selecting R, eligible P1 or eligible P4. Ineligible addition mass returns to R; failures abort with their audit rather than substituting a different source. Explicit envelope ceilings change the native shape/class law and are marked as such. The training producer can apply a stage-specific envelope to a configured fraction of draws. Accepted width and observed labels may differ from their requested budgets.

The native output has already undergone its numeric representation and does not retain category-type identities. Its Episode mask marks columns numerical; this limits direct training coverage of the separate nominal encoder. `static_prior.generate_episode()` retains authored feature types and separate calibration/support/query streams. Its conditional classification-law metadata is used only by the optional loss control where valid; unsupported mechanisms use sampled targets.

The new `standard_prior="selection"` dispatches through the native task/shape callback before any graph is generated. `selection_options` fixes the mechanism and observation simplexes and dimension-conditioning probability. New families use independent calibration; R observation fitting uses clean support features. H width fallback is explicit. Query labels, soft probabilities and diagnostics remain loss/audit-only. The exact recipe and configuration map are in [prior_selection.md](prior_selection.md).

## Finance generation and controls

`generate_finance_world()` builds finite strata and replayable row identities. The default legacy family is class-conditional Gaussian motif geometry. The main candidate explicitly selects `mechanism_family="risk_partition"`: product covariate cells, sparse threshold main/interaction risk, a historical rare-event intercept, binomial event counts and optional feature rotation. It preserves finite scarcity; expected prevalence calibration does not fix actual positive counts.

The existing observation process still determines which historical outcomes are revealed by the prediction cutoff. Mature complete cohorts can supply reference prevalence and cohort size; selectively observed pools cannot manufacture them. Full-amount and realized-fraud-loss observation rules remain separate.

`feature_view="all"|"hide_h"|"h_only"` masks rendered columns before both selection and model use. `loss_normalization="reference_entropy"|"unit"` controls only the additional world-loss multiplier. `positive_h_probability` and `negative_h_probability` control H's generative informativeness. Equal probabilities make H nondiscriminative in the population, not necessarily after selective observation.

A `population_id` excludes view, normalization and support-selector controls, preserving population and query draws across those paired comparisons; `world_id` still identifies the full configuration. New risk-partition count/outcome/policy/reveal streams are separately keyed. Changing the H law preserves aggregate day/cell/class outcome counts in that family, but can change stratum boundaries, row identities and eligibility. The legacy family's sequential count RNG is retained; it does not promise the same outcome counts under H-law changes.

`sample_macroepisode()` samples global finite future classes and IDs before observable query routing. Per-query weights recover natural finite-population risk. Route losses use the original query denominator, never an equal mean of route means. `training=False` defaults to unique natural-prevalence queries. The real-data `FinancePredictor` uses the same reservoir/positive/local context structure; nearest-to-center selection remains an approximation to query-specific optimal support.

## Model, scores and caches

The recommended `base` options yield **317,116,304 parameters**, including the finance adapter and optional hurdle head. The legacy configuration remains available and its parameter counts differ. Model manifests calculate the actual count; size labels are approximate descriptions.

- `hash_bits` supplies exact signed hash bits to the nominal branch, replacing scalar Fourier category identity. Hash collisions and learned-embedding collisions remain possible; the scalar encoding is retained as a control.
- `quantile` predicts 999 ordinary-regression quantiles. Training averages pinball scores in support-standardized units. Report quantile scores and point metrics; this head does not expose a trained continuous density NLL.
- `regression_bins=1025` sets the finite-bin resolution for histogram/hurdle paths. Two Lomax tails remain, so finite resolution and tail assumptions still matter.
- `hurdle_target_scale="positive_support"` fits the severity affine scale to observed positive support targets when present, otherwise using the usual reference-support fallback. It does not change query weights or the exact atom, consult query outcomes, or perform a log1p density transform.
- `length_scaling="logarithmic"` uses positive learned log-key-count scaling in support aggregation. It is not the query-dependent QASSMax implementation.
- `trunk_query_kv_heads=2` retains two KV heads for query attention while support attention uses all heads. It is a trained capacity change with compact stored caches; backend kernel behavior determines runtime savings.

The finance metadata vector has 14 observable coordinates, including reference cohort size. Binary prevalence uses Jeffreys smoothing and multiclass references use per-class Dirichlet smoothing. Reference offsets parameterize the trained predictor; they are not generic post-hoc retrieval corrections. Cached inference checks model identity, parameter versions and precision, and preserves query-batch independence. Cache correctness is separate from measured speed.

## Training, production and provenance

`TaskProducer` optionally prepares tasks in a configured pool of spawned processes per rank, preserving consumption order and isolating upstream global RNG use. It bounds pending task count and caches four finance worlds. It does not implement byte-bounded queues, pinned-memory transfers or separate copy streams. `PYTHONHASHSEED=0` is required before Python startup for reference generation.

The trainer sums per-query scores with proposal weights, applies the macroepisode multiplier and original query denominator, then averages macros across ranks. Globally inactive parameters receive no update; rank-local inactive parameters contribute zero if another rank used them. Muon operates on registered full hidden matrices; AdamW handles the remaining parameters. The candidate explicitly disables weight decay.

`--initialize-from` starts a new declared experiment from compatible parent weights with fresh optimizers and schedule, preserving parent SHA-256 provenance. `--resume` restores the same run with its exact configuration and states. Deterministic seeds reconstruct prepared data; stale model activations are never reused after updates. The logs include distributed source/task/exposure counts and rank-zero producer/wait costs alongside end-to-end allocation time.

The primary training sequence protects the standard checkpoint and then trains a **binary-only** finance specialist. The model's optional hurdle and multiclass paths are implemented, but this continuation does not train those tasks. The 80/20 joint profile, unit/entropy comparisons, architecture options and P4 graft remain executable controlled alternatives rather than simultaneous defaults.

## Remaining scope

The repository contains the new prior controls, encoding/head/attention options, process producer, explicit continuation and volume planner. It does not supply competitive trained checkpoints, official full-model reproduction, MI355X throughput calibration, a completed broad benchmark run or production fraud validation. The standard adapter and real-data finance evaluation have different coverage obligations; keep their results separate.

MoE, general sparse support graphs, feature-feedback training, auxiliary consistency, adaptive prior mixtures and real-data pretraining remain research branches. No production producer farm, FSDP or expert-parallel backend is claimed. Improvements to systems code must preserve deterministic consumption, finite populations, selection laws and accounting. See [training](training.md), [data/evaluation](data_and_evaluation.md) and the [audit](audit.md) for operational boundaries.

## Executable prior-selection program

[prior_selection.md](prior_selection.md) specifies the new standard challenger and its costed trials. `scripts/plan_prior_selection.py` and `research/probes/prior_information_analysis.py` are implemented accounting/analysis tools. The new family versions, common observation wrapper and compact architecture are integrated training components. Phase materialization, scheduler submission and paired standard-panel evaluation are implemented. Older candidate configs keep their reference-pilot meaning; the new selection configs contain the provisional full-volume hypothesis.

## Persistent-cell comparison

`build_model("persistent_small", model_options=...)` retains width-256 feature cells through twelve stages, each with one column ISAB and one row MAB. Each ISAB uses 128 support-derived inducing summaries per column. Eight row summary tokens are projected to width 512 only at the final readout. There is no row-level ICL trunk. With candidate heads and the finance adapter the model has 41,089,248 parameters. `persistent_tiny` is the diagnostic counterpart. Support labels enter the feature encoder; queries never update support summaries. This tests an authored compact architecture at equal allocated cost, rather than reproducing a competitor or isolating topology at equal parameter count. The `trunk_query_kv_heads` option has no effect when there is no trunk.
