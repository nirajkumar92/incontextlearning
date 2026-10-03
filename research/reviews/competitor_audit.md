# Competitor audit for tabular ICL and finance prediction

Evidence audited 2026-10-02; scope revised 2026-10-03. Main evidence for the first three models is the user's supplied PDFs. Page numbers below are physical PDF pages (the extracted text inserts `=== PDF PAGE n ===`). Online release details were checked against official repositories/model cards. Described performance is author-reported unless explicitly called a local measurement. No competitor inference or pretraining was run in this audit.

## Executive assessment

A longer list of SCM functions will not by itself differentiate a new model. Kumo Tabular already combines neural, linear, tree and GP mechanisms with missingness, coarsening, high-cardinality categories and heavy tails. Xiaomi already composes nonlinear functions, and its inference stack makes substantial use of task-adaptive ensembling. EXAONE shows that a compact architecture retaining repeated cell/context interaction can be competitive without an obviously unprecedented prior taxonomy.

The research objective is standard binary/multiclass classification and regression, with financial fraud near one in 10,000 events as an additional operational setting. Useful candidate improvements must address predictive mechanisms, finite-context learnability, representation, or how legal historical evidence is selected. Model quality, prior quality and inference engineering need separate comparisons. No source in this audit establishes a prior bank that universally wins.

## 1. NVIDIA Kumo Tabular: supplied `kumotabular.pdf`

This is a print of the NVIDIA/Hugging Face September 29 release article, not an archival paper containing detailed ablations. It describes the single-table Kumo Tabular model. Kumo Relational is a separate product; its artifacts and results are not evidence for this single-table checkpoint.

### Prior and pretraining: PDF pp. 4–6

The generator draws table/task/mechanism/missingness configuration, builds a random latent DAG, and computes nodes topologically. Mechanisms include linear maps, small neural networks, trees and Gaussian processes. Selected nodes become numerical/categorical columns and target; remaining variables are unobserved. Post-processing correlates feature groups, clips outliers and adds missingness. A fast tree-ensemble signal check rejects unlearnable tables.

The article explicitly lists multiple missingness patterns, coarsened features that allow duplicate feature rows to have different labels, high-cardinality categoricals and heavy-tailed regression targets. These should be treated as parity requirements. It does not disclose exact mixture weights, graph distributions, quality-filter thresholds, or an isolated ablation establishing which prior change produced the reported gains.

Classification and regression are separately trained, using cross entropy and quantile loss. Stage 1 uses 1,024-row tables and up to 100 columns; stage 2 uses 400–10,240 rows; stage 3 reaches 60,000 rows, still up to 100 columns. Small/Medium/Large process approximately 35/71/137 million artificial tables. These are training instances, not necessarily unique independent parameter draws or a model-agnostic compute measure. The report says generator and training recipe will be released later.

### Architecture and inference: PDF pp. 3–4, 7

Grouped cells use Fourier embeddings with separate numerical and categorical weights and special missing-value handling. Alternating induced column attention and row attention produces four CLS summaries per row. Context labels enter the cell embeddings early. A terminal ICL transformer predicts queries from context only; the context does not depend on queries. The model uses learned logarithmic length-dependent attention temperature and Test-GQA, and the regression head predicts 999 quantiles. Those architectural/optimization differences are confounders for any claim that its advantage is due solely to priors.

Native classification supports ten classes; the library extends it through ECOC. Preprocessing can add transformed features, including timestamp encodings. Claimed query independence applies to the model computation; end-to-end pipeline independence must also verify preprocessing.

### Evidence limitations: PDF pp. 6–7

The article reports TabArena Elo 1950, a 17x speed comparison against LimiX-2 on a uniform RTX 6000 Pro setup, and first place on BeyondArena, TALENT and ScoringBench. These are dated release claims, not one shared all-model experiment proving superiority on every task. Elo and mean rank depend on the comparison pool, datasets, metric, preprocessing and time. The supplied report does not include component ablations or replicate-level uncertainty needed to attribute gains to a particular generator component.

Official artifacts: [release article](https://huggingface.co/blog/nvidia/kumo-tabular), [inference repository](https://github.com/NVIDIA/structured-data-models), [weights](https://huggingface.co/nvidia/Kumo-Tabular). The card gives OpenMDW-1.1 for weights. At the audited source snapshot, inference code and weights were verifiable; a complete reproducible Kumo Tabular pretraining recipe was not established.

## 2. Xiaomi-TabLDM: supplied `xiaomi.pdf`

The supplied paper is arXiv:2609.03880v2, September 4, 2026. It should be distinguished from subsequent checkpoint/runtime updates.

### Prior: §2.3, Fig. 5, PDF pp. 6–8; Appendix B.3 pp. 26–29

The pipeline samples table dimensions, types, categorical cardinalities and context/query split; draws a Cauchy DAG based on TabICLv2; propagates vector-valued latent nodes; and observes selected dimensions. Other dimensions remain hidden. It uses both concatenate-then-transform and transform-then-aggregate node construction. Aggregations shown in Fig. 5 include sum, product, max and logsumexp. Features and targets can be numerical or categorical; classification uses discretization or categorical sampling. Gaussian perturbations and random node scaling diversify mechanisms and signal strengths.

The figure caption says twelve random-function classes. Visual inspection of the actual figure shows eight named examples: **MLP, Tree Ensemble, Discrete, GP, Linear, Quadratic, EM, Spectral Mixture**, followed by an ellipsis. Do not invent the remaining four, expand the unexplained acronym EM, or claim an exhaustive implementation-level function bank from this figure. The supplied text and publicly visible inference repository do not specify enough information to recreate its complete prior exactly.

Graphs are resampled if chosen features and target lack a common ancestor. Tables are standardized/permuted and rejected when ExtraTrees cannot reliably beat a constant predictor. Appendix B.3 shows function visualizations but does not establish causal benefit from each new function class. Quality filtering is an important induced bias: learnability is defined relative to the chosen evaluator. A new generator should test whether this filter removes useful smooth, categorical-interaction or rare-event tasks merely because its small ExtraTrees evaluator cannot learn them from the sampled context.

### Training and architecture: §2.1–2.2, pp. 4–6; A.3–A.4 p. 23

The three-stage model uses induced column attention, row CLS aggregation, then ICL attention. Two feature-grouping streams combine fixed and width-adaptive offsets. Attention Residual and sparse MoE modify optimization/capacity; QASSMax handles longer contexts. The classifier has 70.08M total/61.67M active parameters; regressor 71.08M/62.68M. Classification and regression are trained separately; regression predicts 999 quantiles with pinball loss.

Stage 1: 500k classification/300k regression steps on 1,024 rows, 30–90% context, peak LR 8e-4. Stage 2: 40k steps, 400–10,240 log-uniform rows, about 80% context, peak LR 1e-4, MoE enabled. Stage 3: 10k steps, 400–60k rows, about 80% context, peak LR 2e-5. Muon, cosine schedule, weight decay .01; reported 8 A100-80GB, roughly 7–10 days stage 1 plus two days each later stage. This is a training report, not a portable timing prediction for MI355X.

### Inference and evaluation confounders: §3, pp. 8–9; §4 and Appendix B

The inference strategy uses column shuffling/subsampling, normalization/quantile views, SVD, interactions, regression target transforms, and nonnegative least-squares weights fitted to holdout or out-of-fold predictions. Classification can be calibrated on validation data. No backbone gradient update does not mean no task-specific fitting or one model evaluation. Compare both fixed single-estimator performance and matched inference budgets; report end-to-end preprocessing/context construction time.

The paper reports regression first on 33 OpenML-CTR23 datasets and second on TALENT, TabArena and BCCO. Its full TabArena snapshot is fourth by Elo point estimate (1659), close to AutoGluon-1.5 (1662) and TabPFN-3 (1650); regression is 1900 against TabFM 2019. Therefore “Xiaomi is SOTA” needs task/metric/date qualification. Its later public release may differ from this paper.

Appendix C pp. 32–34 tests 2,240 synthetic regression datasets, varying sparse/dense ordered Erdős–Rényi DAGs, four SNR regimes and fourteen noise conditions. It compares Xiaomi, LimiX and TabICLv2, finding Xiaomi first on 70.3%. The setting has 31 observed inputs and a final-node target, no latent variables; structural functions and matched base configurations are held fixed across noise conditions. This is useful robustness evidence, not an ablation of the prior and not evidence against all current competitors. The distributionally equivalent exponential/Weibull A/A conditions differ by five percentage points, showing material finite-sample variability.

Official artifacts: [repository](https://github.com/xiaomi-research/xiaomi-tabldm), [weights](https://huggingface.co/occams/Xiaomi-TabLDM). Repository news explicitly reports September 28 regression-weight and September 29 inference-code updates. Pin both code commit and weight revision when benchmarking. The browsed package tree exposes model and sklearn inference code; a complete training/generator release was not verified. Repository release text specifies non-commercial use for the model; inspect the actual license for a planned deployment rather than relying on GitHub’s automatic license badge.

## 3. EXAONE Tabular: supplied `exaone.pdf`

The report is arXiv:2608.25774v1, August 26, 2026.

### Prior: §2.3, PDF pp. 6–7; Fig. 2 p. 5

SCM episodes begin with table dimensions, class count, graph size, components and mechanism families. The latent DAG may have one or two weakly connected components and randomized topology. Edges use neural transformations, tree rules or identities. Each node combines scaled edge outputs with independent noise. Observed input/target nodes are then selected; task type changes target construction and sampling. Postprocessing includes categorical discretization, nonlinear feature warps, feature quantization and task-specific missingness. Optional quality filtering removes overly easy, noisy or uninformative tasks. Fig. 2 labels Gaussian, Poisson and Uniform noise but does not disclose the full noise mixture or all sampling ranges.

This is broad but underspecified. Mechanism probabilities, effective complexity/SNR distribution, component distribution and filter thresholds are not reproducible from the paper alone. Do not infer that the model lacks any generator operation just because the short report omits it.

### Architecture and training: §2.1–2.2 pp. 4–6; §3 p. 9

CAST preserves cell states across twelve layers, interleaving feature and item processing repeatedly instead of permanently compressing features before ICL. Three item-summary slots and 32 feature-summary slots provide repeated cross-axis communication. Query items do not affect context or each other. Model dimensions are 192 with six heads; classifiers/regressors have 20.81M/21.11M parameters. SSMax adjusts length dependence. Regression predicts 999 quantiles, sorts crossings, and uses a trapezoidal average of central 99.8% as an approximate conditional mean.

Reported training processes about 30M classification and 10M regression synthetic table instances across the model lineage, in bf16. Most matrices use Muon, remaining parameters AdamW. WSD predominates, with continuation/adaptation schedules; EMA usually has decay .999. The episode count is not a clean FLOP budget or direct evidence that more tables alone outperform fewer.

### Evaluation and implementation caveats: §2.4–2.5 pp. 7–8; §4 pp. 9–15

Inference includes preprocessing/permutation ensembles, with eight default classification estimators; many classes use ECOC. Hence the report’s “without post-hoc ensembling” wording for default leaderboard entries should not be read as an unensembled single forward pass. Cache/uncached execution can materially alter reported runtime.

TabArena results use the August official leaderboard, not a rerun of all baselines on identical hardware. BCCO/TALENT comparisons cap context at 50k samples per fold and preserve test/validation sets; train-only preprocessing is stated. Twelve TALENT tasks with more than ten classes are excluded. ScoringBench uses metric-specific coverage filtering, so different metrics may summarize different task/model pools. Reported ranks from August cannot be compared numerically to September Elo values in other papers as if the pools were fixed.

The report claims native missing-value use. However, the current official repository’s regression instructions say the public preprocessor mean-imputes NaNs and expects numerical inputs, and describe task-adaptive NNLS ensemble weighting. This paper/runtime discrepancy should be resolved against pinned code before claiming native end-to-end missingness behavior. Architecture may accept masks while a wrapper discards or transforms some information.

Official artifacts: [repository](https://github.com/LGAI-Research/EXAONE-Tabular), [weights](https://huggingface.co/LG-AI-Research/EXAONE-Tabular). The current repository identifies itself as an inference runtime. Its regression NNLS uses a 20% holdout, requires 2,000 held-out rows, normalizes weights and blends 75/25 with uniform weighting. Those settings matter for reproducibility. The supplied paper §6 p. 16 distinguishes permissive code from non-commercial released weights. No isolated prior-component ablation was identified in the supplied report.

## 4. Implications for prior design

The source reports support a strong parity baseline: compositional nonlinear mechanisms, categorical effects, missingness, broad signal-to-noise variation, heavy regression tails, and a row/feature curriculum. They do not isolate a single recipe change that explains their entire model ranking. Kumo's disclosed generator is incomplete, Xiaomi's function catalog is not fully specified, and EXAONE's report does not publish all mechanism/filter distributions. Absence from a short report is not evidence that a competitor lacks a component.

The selected static bank should therefore be judged by controlled additions to a capable baseline. Context-conditioned effective dimension and leaf/category occupancy address what can actually be learned from support. Hierarchical categorical effects can test partial pooling across frequent, rare and unseen levels. Smooth local functions, threshold boundaries and regime interactions supply distinct prediction geometries. Count and compound-severity tasks add structured regression behavior. Their distributions and allocation are authored choices in `research/specs/static_recipe_v3.json`, not claims of unprecedented statistical families.

Finance adds a different evidence constraint. At fraud prevalence around1e-4, a natural 2,048-row support sample contains only0.2048 expected positives. A few-million-row source history may contain hundreds of events, but only available verified labels can enter context. Bounded representative, positive and local-negative channels should be compared against full-history LightGBM/CatBoost and existing TFMs under the same legal data boundary. Structured missingness, delayed verification, repeated categorical groups and independent arrival/risk bursts remain relevant table mechanisms.

For classification, train against natural population-query probability even if queries are stratified to reduce variance. The exact finite-population proposal weights stay in the loss; revealed reference counts and acquisition metadata are observable inputs. Do not call an unresolved row negative or apply a generic scalar prevalence correction after feature-dependent support selection. For realized fraud loss, represent its zero atom separately from positive severity; ordinary amount regression is a distinct task.

## 5. Required comparisons

Use equal-backbone, matched-compute pilots to identify prior effects. Hold preprocessing, target construction and inference policy fixed for those comparisons, and assess architecture changes separately. At least the promising treatments need multiple training seeds and dataset-level uncertainty. Keep the pure-standard profile alongside the joint finance profile, preserving per-head scores on ordinary benchmarks.

A useful sequence is: strong standard baseline; selected static mechanisms and complexity calibration; finance prevalence/finite evidence; context acquisition with permitted metadata; structured missingness and label observation. Test selected pairs where an interaction is plausible. The full combination is not presumed best. Hold out generator mechanisms and real source families, and treat real-data continuation as a separate track with deduplication and provenance.

Report both a fixed single estimator and a matched inference-budget configuration for competitors. Include preprocessing, support selection/indexing, context prefill, repeated estimators and cached-query batches in runtime. Record the exact checkpoint and code revision: a September runtime with August weights or a task-adaptive ensemble is a different evaluated system from the paper's simplest model path.

For finance, preserve chronological cutoffs and natural test prevalence. Report precision/recall at fixed review budgets, recall at a predeclared precision, false-positive rate, probability calibration and captured fraud amount as appropriate. State whether the review unit is transaction, account or campaign; do not treat correlated campaign rows or repeated sampled query IDs as independent evidence. Retain a separate unseen-entity/campaign stress panel without replacing the intended recurring-entity deployment test.

Simulator invariants should establish that only revealed historical outcomes enter labeled support, future records cannot alter past decision-time features, finite IDs replay exactly, and query targets/importance weights are unavailable to forward computation. Those checks validate the experimental inputs. Competitive predictive performance and latency still require measurement.
