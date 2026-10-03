# Prior literature audit and design recommendations

Evidence audited 2026-10-02; scope revised 2026-10-03. This note covers the supplied TabICL, TabICLv2 and Google TabFM papers, plus primary literature on synthetic prior mixtures, realism and adaptive sampling. The objective is binary classification, multiclass classification and regression through tabular ICL, including financial fraud near one event per 10,000 rows. The selected recipes are in `research/specs/`; proposed gains remain hypotheses until measured.

## Research judgment

A broad SCM/tree/GP mixture is an essential replication baseline. The strongest research hypothesis is to allocate prior mass to **predictive mechanisms that are learnable from the available labeled context**, then validate their complementary contribution on held-out real tasks. This includes categorical group effects, local and threshold structure, heteroscedastic noise, and low-information cases. For finance, generate the finite population, label observation process and selected support together: rare positives, missing measurements and delayed verification alter what evidence the model can actually use. A table's statistical resemblance to real data is a useful diagnostic, but not the optimization target.

No paper proves such a bank will dominate every model. A finite model and compute budget force tradeoffs, and observationally indistinguishable tasks may require opposite predictions. More prior support is insufficient: mass allocation, task identifiability, model capacity, and amortization error matter.

## 1. Local-paper evidence

### TabICL v1 — supplied `tabicl.pdf`

Primary URL: https://arxiv.org/abs/2502.05564 ; code https://github.com/soda-inria/tabicl . Local page numbers refer to the PDF, not search result pagination.

- **Section 4.1, p6:** 70% neural SCM / 30% tree SCM. Neural DAG follows a fully-connected MLP structure; variables are neurons and additive independent noise enters mechanisms. Tree SCM layers fit XGBoost on random Gaussian targets and use fitted predictions as child values. This captures hierarchical piecewise structure but training a fitted tree per sampled layer is expensive.
- **Appendix C.1, pp14–16:** per-layer standardization, random rescaling and a much broader activation bank. 50% of draws share activation type across layers. Added sign, sine, RBF, exponential, power/absolute/indicator-like functions and random GP-inspired Fourier activations. GP activations use 256 features and frequency power decay with a random exponent; sampled at 10 times the probability of each other activation.
- **Appendix C.2, pp16–18:** small XGBoost ensembles fitted with independently sampled depth and estimator counts, capped at four. These are not equivalent to arbitrary large directly sampled boosted ensembles.
- **Figure 9 and Section 5.3, p8:** adding tree SCMs improves real-data classification metrics in a 20K-step ablation over 200 datasets. This is actual evidence for a complementary tree prior, not just narrative plausibility.
- **Section 4.2, p6; Section 5.3, p8:** long-context curriculum helps larger real tasks but slightly degrades some smaller tasks. Shape curriculum therefore needs small-task replay and per-size validation.
- **Appendix B, p14:** both TabICL and TabPFNv2 degrade somewhat as categorical fraction increases. Useful failure axis, but this old result does not prove the same weakness remains in every 2026 checkpoint.

### TabICLv2 — supplied `tabicl2.pdf`, arXiv:2602.11139v2

Primary URL: https://arxiv.org/abs/2602.11139 . This is the strongest open prior implementation to replicate first.

**Mechanisms already present, so do not relabel them novel:**

- **Section 5, pp5–7; Appendix E.8, pp25–27:** eight function families inside one DAG: MLP, CatBoost-style symmetric tree ensembles, nearest-center discretization, multivariate random-feature GP, linear, quadratic, EM-inspired soft clustering, products. Multiple parents are either concatenated (50%) or independently transformed then aggregated by sum/product/max/logsumexp. Linear-plus-nonlinear-plus-tree compositionality already exists.
- **Appendix E.4, p24:** edge probability `sigmoid(A+B_i+C_j)` with independent standard Cauchy global/source/destination terms, producing correlated degrees and dense/sparse exceptions. Graphs have log-sampled 2–32 nodes, and nodes are vector-valued.
- **Appendix E.2, p24:** correlated scalar hyperparameters use a shared Beta hyperprior per named quantity: `t~Uniform(0,1)`, `s~LogUniform(.1,10000)`, `alpha=st`, `beta=s(1-t)`. Applies to cardinalities etc. Correlating task hyperparameters is already explicit prior art.
- **Appendix E.5, p24; E.11, p28:** random feature/node importance and diverse singular spectra. Positive weights use power decay times lognormal variation, then normalization and shuffling. Node vector L2 normalization is deliberately chosen to avoid making high-dimensional functions too hard.
- **Appendix E.6, p25:** categorical converters use nearest-center or softmax sampling with random temperatures and imbalance; can alter downstream latent nodes to enforce category-dependent mechanisms. Seven converter combinations. These are more than quantile bins, but their reported cardinality range is narrow (Appendix E.3 samples a maximum from 2–9).
- **Appendix E.8, pp25–26:** symmetric tree ensemble has log-sampled 1–128 trees, depth 1–7, splits sampled from realized input values, and Gaussian leaf values. GP uses 256 random Fourier features with heavy-tail spectral distributions, random input projection, and 50% axis-aligned product-kernel draws.
- **Appendix E.9–12, pp27–28:** broad nonlinear activation bank; five random matrix families; normal/uniform/ball/covariance root-point distributions transformed by random functions.
- **Section 5 p6; E.14 p29:** graph reachability/common-ancestor validity filtering and ExtraTrees OOB filtering. ExtraTreesRegressor has 25 estimators, max depth 6, bootstrap=True; reject unless OOB MSE beats constant on at least 95% of 200 bootstrap samples. About 35% of classification and 25% of regression draws rejected in stage1. This boosts convergence, but hard filtering through one tree family may suppress learnable smooth, rare-event or interaction patterns it cannot detect. That latter sentence is a proposed failure mechanism to test, not a demonstrated result.

**Crucial negative/interaction evidence:**

- **Section 7 pp8–9, Figure10; Appendix C p21:** prior yields the largest aggregate ablation effect, but improved prior is not independent of architecture. New architecture on old prior underperforms and validation degrades; old architecture on new prior merely matches the old model. Therefore do not make a prior-only promise or rank priors using only one tiny proxy architecture.
- **Appendix C p21:** reintroducing Gaussian noise at SCM edges has negligible impact. Noise is also induced by unobserved node dimensions. More additive noise is not an established edge.
- **Appendix D p23:** adding convolution mechanisms did not measurably help in smaller/finetuning trials. This is anecdotal, not proof that convolutional mechanisms never help.
- **Appendix E.2 p24 / E.6 p25:** paper candidly records correlated categorical-choice sampling disabled by a bug, and Kumaraswamy warping applied to downstream node state rather than extracted feature. Reproductions must distinguish published intent from executed generator. A “bug fix” changes the training distribution and must be tested.
- **E.13, pp28–29:** removes constant columns, invalid class splits and outliers, standardizes X and regression y, ordinal-encodes categoricals, permutes columns/classes but not category identities. Potential categorical-encoding gap: if learned code artifacts favor ordinal relationships, random category relabeling is a clean intervention; this is an unproven inference.

### Google TabFM — supplied `tabfm.pdf`, 2026-09-29, arXiv:2609.37959v1

Primary URL: https://arxiv.org/abs/2609.37959 ; model listed in paper: https://huggingface.co/google/tabfm-1.1.0-pytorch . **This supplied release is 400M parameters**, not the earlier 1.64B version referenced in earlier competitor reports. Cross-report rankings are not time-aligned.

- **Section3.2 p6:** entirely synthetic SCM pretraining, citing Qu et al. 2026. Random DAGs with nonlinear transforms/algebraic aggregations, jointly random table shape, categorical proportion/cardinality, missingness, label noise, and imbalance. Feature cap100; context/query split precedes model loss; classification and regression targets jointly trained. Paper does not disclose enough detail here to infer a uniquely novel prior family.
- **Section3.2 p6:** four-stage 2,048→16,384 context curriculum, each doubling rows/halving batch; ~4.19M tokens per step. Mechanism diversity alone cannot explain full performance changes because architecture and shape scale differ.
- **Section6 p11:** larger native table shapes and distillation of automatically engineered feature pipelines are stated future directions. These are relevant to feature/context capacity; they do not identify an isolated missing prior mechanism or establish that adding it improves transfer.
- **Table2 p7; Sections4–5:** default TabFM, TabFM+ (multi-view expansion/ensembling/calibration), and TabFM-Auto (LLM-guided feature engineering) are different inference budgets. Do not attribute Auto's entire gains to its prior.

## 2. Additional primary literature (compact source summaries)

### Mitra v1 — https://arxiv.org/html/2510.21204v1

Sections3.2/4.7, Tables1/6: selects priors for real-task performance, diversity and cross-prior distinctiveness. Train separate proxy models and form a cross-generator generalization matrix. The diagonal is used as a diversity heuristic; off-diagonal transfer diagnoses redundancy. SCM plus ExtraTrees improves Elo by63 in its ablation. Directly sampled forests perform well alone but overlap other generators; low-transfer random forests can simply be poor priors. Thus maximal novelty/diversity alone is insufficient. Mixture uses SCM plus multiple tree generators, with 50/50 causal/tree mass in the reported prior-importance ablation. Table6 shows SCM+ET+DT can underperform SCM+ET. Other tree additions often yield similar performance, although the full mixture improves sample efficiency. **Caution:** raw cross-generator difficulty confounds diversity with label noise/Bayes error, so use oracle/teacher-normalized error when adapting the diagnostic.

### Mitra v2 — https://arxiv.org/html/2609.04540v1

Sept3 2026 report, Sections2.2–2.4/Table2: classification mixture is35% base SCM,35% hybrid SCM, and6% each DT/ET/GB/RF/directly sampled RF. Hybrid composes MLP, trees, GP, convolutions and VAR mechanisms in vector-valued DAGs, adapting O’Prior. Native pretraining caps:5,120 support+1,280 query rows,50 features,10 classes. Section2.2/AppendixC: reweighting outer mixture and uniform versus log-uniform support size produced little accuracy change. Approx77M parameters; default deployment finetunes and bags eight copies, so reported leading scores are not zero-shot-only comparisons. Regression changed to1,000-bin distributional prediction. Comparisons predate supplied September29 TabFM and later TabPFN3.5/Kumo releases. **Takeaway:** increasing names in a bank or tuning seven weights has weak novelty; match deployment budgets and checkpoint dates.

### O’Prior / Shaping the Prior — https://arxiv.org/html/2605.18971v1

Section2 already combines hybrid SCMs, marginal transforms, engineered features, MCAR/MAR/MNAR missingness, subgroup/target effects, confounding, shortcuts, covariate/seasonal shifts and a mild-to-hard curriculum. Section3.1 fixes nanoTabPFN and uses40K synthetic datasets,52 small classification tasks. Evidence is useful but not current full-scale SOTA. **Table2 contradicts a simplistic “all modules best” claim:** full G4 TabArena AUC=.8194 versus hybrid-only=.8335; OpenML-CC18 full=.8245 versus SCM+hybrid=.8313. Hybrid+strong realism and hybrid+shift may hurt relative to hybrid. Broad “compositional realism+shift” is already prior art. Its support-only preprocessing contract is a conservative inductive protocol; unlabeled query features may be valid in an explicitly transductive deployment, so query-feature use is not universally leakage. The essential prohibition is unavailable future information/target labels. **Use:** direct controlled baseline; do not sell maximal corruptions as a guaranteed improvement.

### Robust Tabular Foundation Models — https://arxiv.org/html/2512.03307v1

Peroni/Le/Sheinin introduce adaptive adversarial generator sampling. Equations3–8 target model cross-entropy minus irreducible entropy, approximated by the minimum loss of strong fitted baselines, with an entropy-constrained distribution over generator settings; optimum weights are a softmax of estimated gaps. They alternate parameter search and model finetuning. This preempts the broad claim “learn mixture by regret.” Table1 reports classification improvement over original TabPFNv2 on TabArena/TabPertNet. Only MLP SCMs studied; no full contemporary-2026 comparison. **Accounting concern:** stated3000 steps×30 epochs×batch64 conflicts with headline90K additional datasets if “steps” means batches; avoid repeating the data-count efficiency claim without code/log verification. Baseline gap is not true Bayes regret, and limited context means oracle mechanism knowledge can be unattainable; use finite-context teachers and independent query sets in our implementation.

### Mind the Gap? — https://arxiv.org/html/2605.06343v1

Sections3–5 compare synthetic TabICL tables, a curated Kaggle corpus and web tables using aggregate descriptors, discrimination and coverage. Synthetic tables occupy restricted descriptor space; searching86K hyperparameter configurations fails to close the gap. Crucially proximity to the synthetic prior shows no clear predictive-performance relationship under their descriptors/embeddings. This does **not** establish that realism is irrelevant or that every prior has this limitation; the result is prior/descriptor-specific. It does undermine using marginal/correlation fidelity as the sole objective. Also their “TabFM” means an older Kaggle corpus, **not** Google's supplied2026 model. Optimize predictive transfer and diagnostic failure coverage, with realism as a constraint/interpretability aid.

### Verified code/release footholds

- https://github.com/soda-inria/tabicl : open generation/training; README supports online DataLoader generation or pre-generation via `python -m tabicl.prior`; classification and quantile regression; Muon supported. README explicitly corrects cautious-weight-decay paper/run discrepancy: released checkpoints used it disabled. Pin a commit before replication.
- Mitra-v2 report release links: https://huggingface.co/autogluon/mitra-classifier-2 and https://huggingface.co/autogluon/mitra-regressor-2 . Verify license and exact revision separately before full pretraining reuse.

## 3. Implications for the selected prior bank

The concrete parameter distributions live in `research/specs/static_recipe_v3.json` and `research/specs/finance_recipe_v3.json`. This review explains the evidence behind the choices; it does not introduce a competing allocation.

### Static prediction mechanisms

The historical authored control uses P0/P1/P4 episode mass .85/.10/.05; the current selection proposal is specified in `prior_selection_v1.json`. P0 supplies linear/GAM, tree, heterogeneous SCM, local and regime mechanisms; P1 supplies hierarchical categorical effects; P4 supplies count and compound-severity structure. Those family names are established statistical tools. The research question is whether their implementation, effective complexity and allocation produce better transferable inference than strong existing synthetic generators.

Keep the following distinctions explicit:

- **Compositionality versus a function catalog.** TabICLv2 already composes diverse mechanisms within a DAG. A new bank needs controlled evidence for its missing predictive structure, not merely additional named activations.
- **Complexity relative to context.** Effective dimension, occupied tree leaves, local neighborhoods and category support counts determine how much a model can infer from a given context. Sample these jointly with support size. Retain some deliberately underdetermined episodes to teach uncertainty; do not equate every high-loss task with useful difficulty.
- **Categorical effects versus code artifacts.** Long-tail frequencies, random intercepts/slopes and shared group structure can produce meaningful partial pooling. Randomly relabel category codes, preserve unseen levels, and measure performance by available examples per category. These effects are represented by ordinary categorical table features.
- **Noise versus hidden predictive information.** Both stochastic response noise and unobserved SCM variables can make targets ambiguous. More noise is not an established advantage. Estimate attainable error using the actual observed features and finite support, rather than an oracle with the latent mechanism.
- **Validity versus learnability filtering.** Reject malformed or numerically invalid tasks. Treat hard ExtraTrees rejection as a separate intervention: it can favor the evaluator's own inductive bias and remove rare or smooth patterns another learner could exploit.

O'Prior's negative combinations and Mitra-v2's weak response to coarse weight tuning argue for disciplined ablations. The selected weights are starting allocations, not estimates of an optimum. Compare fixed allocation with adaptation only after the baseline is stable; log accepted task/shape frequencies and preserve replay of general tasks.

### Finance as a selected-support prediction problem

The historical joint-training control emits 80% standard and 20% finance macroepisodes, alongside an unchanged pure-standard control. The finance prior is centered on the user's corrected fraud rate of 1e-4. Histories can contain a few million rows, while the model receives a bounded union of representative rows, available verified positives and feature-selected negative examples. This is an information-selection problem as well as a modeling problem.

Preserve finite row identities and actual revealed-label counts. Unknown or unreported outcomes are not negative labels. Historical labels enter context only after availability time; recurring categories/entities may have stable or changing effects. Arrival bursts, fraud campaigns and missingness mechanisms are separate draws, so increased transaction volume does not automatically imply increased fraud prevalence. Full transaction amount and realized fraud loss are separate regression targets, with the latter requiring a zero atom plus a positive conditional distribution.

Train population-query risk conditional on selected support and permitted metadata. The finite future population can be sampled by class for training efficiency, but query importance weights belong only in the loss. Sample globally before feature routing and reduce all routed groups over the original macroepisode query count. Complete historical reference counts may inform class offsets and loss scaling; hidden simulator prevalence and future class counts must not enter model inputs. A model already trained to population-query risk should not receive an automatic second case-control correction.

These contracts prevent invalid shortcuts. They do not establish that the finance prior improves ordinary benchmarks or fraud precision. Test that claim while retaining the standard profile and its per-task scorecards.

## 4. Experiments that can establish or falsify an advantage

1. **Strong controlled baseline.** Replicate the published TabICLv2 prior on the intended backbone with matched shape, optimization and inference budgets. Repeat finalists at a second capacity because prior–architecture interaction is documented.
2. **Incremental mechanisms.** Add categorical hierarchy, context-conditioned complexity, local/regime functions and count/severity structure separately, then test useful combinations. Compare with O'Prior-style composition; a larger combined bank may lose.
3. **Filtering and adaptation.** Compare validity-only filtering, hard ExtraTrees rejection and an optional teacher-based weighting rule. Teachers train on support and score independent calibration/query draws at the same context size. Their best achieved loss is a reference estimate, not true Bayes risk.
4. **Complementarity.** Cross-evaluate family experts on held-out synthetic mechanisms and real development tasks. Measure leave-one-family-out transfer, proper scores and calibration; raw cross-generator error can confound useful diversity with irreducible noise.
5. **Standard versus joint training.** Compare the unchanged standard run with the 80/20 joint profile at matched end-to-end compute. Show binary, multiclass and regression results separately. Finance gains do not compensate automatically for unacceptable general-benchmark losses.
6. **Finance acquisition controls.** Compare representative support alone, representative support plus available positives, and the full selected context. Match original histories, legal labels and inference budgets. Compare full candidate search against its declared sampled-pool latency control and tuned full-history tree baselines.
7. **Information-boundary tests.** Mutating unavailable future information must not change support or decision-time features. Unresolved labels must never become negatives. Row replay must preserve finite identity; repeated queries must not count as independent fraud cases. Query outcomes and proposal weights must stay outside forward inputs.
8. **Locked evaluation.** Separate source dataset families before tuning. For fraud, use chronological cutoffs, natural prevalence and legal label availability, with recurring-entity deployment and unseen-entity/campaign stress panels reported separately. Cluster uncertainty at the appropriate dataset, time, entity or campaign unit.
9. **Promotion.** Require paired real-task gains across seeds under matched preprocessing, finetuning and inference policies. Report practical effect sizes and uncertainty rather than declaring success from a rank or one synthetic stress test.

## 5. Findings to carry into decisions

- Diversity helps only when it adds complementary, learnable structure relevant to the prediction problem.
- Strong open priors already contain most obvious function families. Effective complexity, categorical transfer, selection of evidence and measured failure coverage are more specific research targets.
- Realism and shift composition, and adversarial prior adaptation, already have published precedents. Novelty needs a precise algorithmic difference or a convincing controlled result.
- Real-table resemblance is a diagnostic, not a demonstrated proxy for downstream accuracy.
- Real development data used to select or fit priors belong in the provenance and benchmark-contamination audit.
- The literature motivates a falsifiable research program; it does not prove that any fixed bank dominates every task.
