# TabPFN priors for standard tabular ICL and rare-fraud finance

Evidence checked: 2026-10-02; scope revised: 2026-10-03. Primary evidence: the five user-supplied PDFs and official project/paper sources. Page references below are **one-based physical PDF pages**, matching `papers/extracted/*.txt` markers, rather than publisher page numbers. Facts, undisclosed details, and proposed inferences are separated. Extended paper analysis is based on user-provided PDFs; short web-update paragraphs intentionally stay within source summarization budgets. This audit addresses binary classification, multiclass classification and regression from labeled tabular support, including financial tables with millions of historical rows and fraud prevalence around $10^{-4}$.

## Assessment relevant to the proposed bank

The strong reproducible baseline must account for broad SCM grammars, trees, expressive categorical mechanisms, temporal/group shifts, spatial/high-frequency components, and substantial scale. None of these is a defensible new contribution by itself in October 2026. The current research hypothesis is to improve **conditional prediction tasks at the available context size**, while matching finite positive evidence, missingness, label availability and context selection in the finance component. The audit cannot establish that proprietary competitors lack any particular mechanism.

Separate a prior's controlled contribution from changes in feature encoding, model size, context length, ensembling and inference computation. TabPFN's release comparisons often change these together. The proposed standard P0/P1/P4 bank and finance extension require fixed-backbone, matched-compute ablations before comparison as a complete system. This evidence does not establish that any selected mixture will outperform all competitors.

## Evidence by release

### TabPFN / ICLR 2023 (`tabpfn.pdf`)

Primary citation: Hollmann, Müller, Eggensperger, Hutter, “TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second,” ICLR 2023. [Paper](https://arxiv.org/abs/2207.01848).

**What the prior actually contains**

- Section 4.1–4.4, PDF pp.4–6: equal-probability SCM and BNN task sampling. The hierarchy mixes not just parameters but network architectures and graph structures. The stated simplicity bias favors smaller graphs and fewer parameters.
- Appendix C.1, pp.26–27: sample an MLP-shaped layered DAG, draw weights, drop edges, select feature nodes and a target node, then propagate node-specific noise. One activation per dataset comes from tanh, leaky ReLU, ELU, identity. This is an MLP-derived DAG family, not arbitrary causal graph discovery.
- Appendix C.2, pp.27–29: blockwise feature selection to induce correlated columns; per-feature outgoing-weight multipliers and graph/block sparsification to induce unequal feature relevance; heterogeneity in node noise means/variances; Gaussian, Zipfian, and multivariate input distributions. Categorical features are discretized continuous variables, with category shuffling, used with 20% probability during prior fitting. NaNs receive no special model treatment; they are replaced by zero at inference.
- Section 4.5, p.6: classification is formed by choosing interval boundaries from continuous target values, then permuting labels; hence class imbalance and multiclass tasks are generated rather than restricted to balanced binary labels.
- Appendix E.4 and Table 5, pp.30–31: broad hierarchical prior hyperparameters were chosen using meta-validation datasets. The table documents distributions over dropout, layer counts, hidden dimensions, noise and weight scales, feature sampling, and SCM/BNN choice.

**Controlled evidence**

Appendix B.4, Table 4, p.20 reports reduced-compute comparisons:

| Prior | Mean cross entropy | Mean ROC AUC |
|---|---:|---:|
| BNN | 0.811 ± 0.009 | 0.865 ± 0.007 |
| SCM | 0.771 ± 0.006 | 0.881 ± 0.002 |
| SCM + BNN | 0.776 ± 0.009 | 0.883 ± 0.003 |

This supports SCM over BNN alone under that setup. It does **not** establish a statistically meaningful gain from retaining the BNN half over pure SCM. It should not be used as evidence that every new prior family improves a mixture.

**Scale and limits**

Appendix E.3, p.30: 18,000 updates × 512 datasets = 9,216,000 datasets; 20 hours on eight RTX 2080 Ti GPUs; each dataset has 1,024 rows and a random context/query split. Main evaluation focused on clean numerical data, with substantially weaker categorical/missing-value performance (pp.7–9; Appendix B.1 and B.5). These were historical limitations, not gaps one should attribute to current TabPFN-3.5.

### TabPFN-v2 / Nature 2025 (`tabpfn2.pdf`)

Primary citation: Hollmann et al., “Accurate predictions on small data with a tabular foundation model,” Nature 637, 319–326 (2025). [DOI](https://www.nature.com/articles/s41586-024-08328-6).

**Generator details**, Methods, “Details on the causal generative process,” PDF pp.9–10:

- DAGs use growing networks with redirection, a preferential-attachment process. Connected components may be merged into disjoint subgraphs, explicitly creating uninformative predictors. Graph size is log-uniform; redirection probability is gamma-distributed.
- Nodes hold vectors. Edge mappings include small random neural networks with identity/log/sigmoid/absolute-value/sine/tanh/rank/square/power/smooth-ReLU/step/modulo activations, categorical discretization and re-embedding, decision-tree modules, and additive Gaussian noise.
- Categorical construction assigns a node vector to its nearest randomly drawn prototype, then independently re-embeds that category for downstream computation. This is stronger than merely thresholding a continuous column at the end.
- Root noise is normal, uniform, or a node-wise mixture. A prototype mixing scheme creates non-independent input samples.
- Post-processing includes Kumaraswamy warps, quantization, and **MCAR missingness**: independent Bernoulli masking per value. This is the explicitly documented missingness mechanism. Later versions disclose too little to infer that MAR/MNAR remain absent.
- Targets come from randomly selected continuous or categorical nodes; classification is natively limited to ten categories in this version.

**Training**, Methods “Training details,” p.10:

- Approximately two million steps, batch 64, ~130 million synthetic datasets per model; two weeks on eight RTX 2080 Ti GPUs per run.
- Context size uniform up to 2,048; fixed 128-query set; features drawn via a beta distribution scaled to 1–160; cell cap 75,000.
- Prior hyperparameters were selected by random search against a development suite. Purely synthetic training therefore does not mean the prior was designed without real-data feedback.
- Main body p.4 says “around 100 million” while Methods gives the more precise ~130 million. These are compatible approximations, not two different claims of experimental scale.

**Architecture and inference matters**

Methods pp.9–11: feature/sample attention, missingness indicator, grouping one or two scalar features per token, label/feature permutations, quantile and original values, SVD features, and calibration are all part of the full deployed recipe. Dataset property subgroups in Figure 5 (p.6) are not a leave-one-prior-component-out ablation. Gains over v1 cannot be assigned entirely to prior expansion.

### TabPFN-2.5 (`tabpfn2.5.pdf`)

Primary citation: Grinsztajn et al., “TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models,” November 2025. [Paper](https://arxiv.org/abs/2511.08667).

- Section 3, PDF pp.4–5: states broader distributions, more rows and features, and maintaining task difficulty, but does **not** supply a reproducible new prior bank, its probabilities, or a controlled component ablation.
- Larger network depth, feature groups of three, a nonlinear regression encoder, 64 learned extra rows, revised preprocessing, calibration, and inference also change. This release does not isolate prior improvements from those factors.
- Section 3, p.5: approximately 100 pilot models explore roughly 50 architecture/training/prior hyperparameters; TabPFN-v2 predicts scores over a denser set of 10,000 configurations from the pilot results. This supports **measuring and optimizing the task distribution**, rather than selecting a large bank by intuition alone. It does not justify treating 100 trials as sufficient in a much larger generator space.
- Section 3, p.4 and Appendix C, pp.28–29: base 2.5 is entirely synthetic; Real-TabPFN-2.5 is separately fine-tuned on 43 curated real datasets. Deduplication checks identifiers, schemas, row/column hashes, and metadata against internal suites and TabArena. Do not conflate Real-TabPFN evidence with synthetic-only training.

### TabPFN-3 (`tabpfn3.pdf`, supplied arXiv v2, 2026-05-28)

Primary citation: Grinsztajn et al., “TabPFN-3: Technical Report.” [Paper](https://arxiv.org/abs/2605.13986).

**The eight explicit prior changes**, Section 2.5, PDF pp.10–12, Figure 9:

1. More DAG sampling algorithms.
2. More parent-to-child combination mechanisms.
3. More expressive categorical mechanisms.
4. High-frequency sinusoidal components.
5. Spatial activations.
6. Many-class tasks.
7. A discrete-time Dynamic Structural Causal Model, motivated by temporal dependence and time-ordered train/test splits.
8. OOD tasks for distribution shifts and extrapolation.

The report states **more than eight trillion tokens** of final pretraining (p.10). It does not specify enough to convert that figure into MI355X days: the token accounting, shape distribution, batch curriculum, optimizer, total accelerator-hours, and hardware utilization are not given here. An acknowledgement of LUMI access appears in Appendix B, p.47; this is not a hardware/time specification.

Appendix D, pp.48–50 supplies visual examples of DAGs, functions, classification tasks, and extrapolation. These examples confirm diversity but do not reveal sampling probabilities, parameter ranges, learning-relevant coverage, or component contribution. New combiner names/formulas and full temporal/OOD generation algorithms are not disclosed. No quantitative controlled prior ablation is reported in this section.

**Inference boundary**

Section 2.6, p.12: Thinking adds inference computation, with details distinct from the base open-weight system. Its gains cannot be counted as prior-only gains or assumed to have the same GPU batch latency. Pin checkpoint identity, estimator count, context budget and inference variant in every reproduction.

### TabPFN-3.5 (`tabpfn3.5.pdf`, supplied arXiv v1, 2026-09-15)

Primary citation: Jäger et al., “TabPFN-3.5: Technical Report.” [Supplied-paper version](https://arxiv.org/abs/2609.17895v1).

**Prior changes**, Section 3.3, PDF p.11:

- Follows the v3 generator; adds general diversity/scalability and inspiration from TabICLv2.
- Emphasizes realistic high-cardinality categorical variables.
- Adds more realistic wide tables and grouped data where train and test belong to different groups.

This is only one paragraph. It does not disclose exact mixtures, their weights, parameter ranges, group-generative equations, label/noise models, or curriculum. It is therefore incorrect either to claim a precise reimplementation or to assert that a specific reasonable mechanism is definitely absent.

**Simultaneous non-prior changes**, Sections 3.1–3.2, pp.8–11:

- Fourier value encodings plus context-derived ECDF features; retains raw standardized values and missing/infinite indicators. This makes high-cardinality integer encodings and skewed distributions easier to represent.
- Model width doubles (512 to 1024), parameter count ~220M (Figure 1, p.4), joint classification/regression checkpoint, additional normalization.
- Most external feature transformations and SVD augmentations are removed; default eight estimators remain (Fast: four).

These changes are confounded with the prior update. Figure 4's slice gains do not causally isolate the prior contribution. A proposed bank should be compared in the **same architecture**, then compared as a complete system to 3.5.

**Benchmark caveats with direct practical relevance**

- Section 2.2, p.5: on full BeyondArena, tuned/ensembled MLPs still lead grouped, temporal, and large-data slices. Non-large grouped/temporal performance closes the gap. Some slices have wide confidence intervals and temporal/group slices are confounded with dataset size.
- Appendix C.2.3, pp.26–27: group identifiers are supplied to TabPFN variants on label-per-sample tasks, differing from benchmark default preprocessing. Encodings were tried for all families; TabFM benefited from another encoding, EXAONE did not. For non-large grouped data, exposing the group key moves 3.5 from **1240 Elo/rank 6 to 1288/rank 1**. With large grouped data included, it moves from **1187/rank 9 to 1267/rank 4**. This is not misconduct; it is an explicitly documented protocol choice that a fair reproduction must preserve or report both ways. A prior that expects group IDs cannot be meaningfully judged with IDs discarded.
- Appendix C.2.2, pp.25–26: modern TabFM/EXAONE comparison uses 116 non-large datasets that all methods support. Five datasets were excluded due to class-count or memory constraints. This is different from the 142-dataset BeyondArena core.
- Appendix C.7, pp.31–32: ScoringBench evaluates predictive densities on 101 regression datasets, with 3k-row subsampling and five folds. The reported CRPS mean rank is 2.85, winning versus v3 on 85/101 datasets. This measures density quality in that small-data regime; it does not establish performance on rare realized losses, large financial histories or selectively observed labels.
- Table 1 and Appendix C.8, pp.1,32–33: the seven-benchmark headline selects the **best family member per benchmark**; only two entries use base 3.5. It is not a claim about one fixed open checkpoint under one compute regime. For the current objective, compare the actual binary, multiclass and regression checkpoint/system under declared inference budgets.
- Section 3.4, p.11: Plus and Thinking internals are intentionally proprietary. No method in the report supplies a public exact comparison at the level of training generator or inference algorithm.

## Current official-source check (2026-10-02)

### Model/package status

The arXiv entry has a September 22 v2, newer than the supplied September 15 v1. Its prior section retains the same brief description. Current official code defaults to 3.5; `model_loading.py` lists the shared checkpoint `tabpfn-v3.5-20260909.safetensors` and multiclass/fast variants. The model card explicitly calls training synthetic-only. Weights and inference code have distinct licenses. Current package guardrails mention 20k features, while the report recommends 6k and uses more estimators for 20k. This should not be interpreted as a changed prior or proved equal accuracy at 20k features.

Sources: [arXiv current](https://arxiv.org/abs/2609.17895), [v2 HTML](https://arxiv.org/html/2609.17895v2), [official loader](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model_loading.py), [HF card](https://huggingface.co/Prior-Labs/tabpfn_3_5), [official README](https://github.com/PriorLabs/TabPFN).

## Recommended experiment boundary and hypotheses (our inferences)

### Standard binary/multiclass/regression ICL

The historical authored control uses P0 conditional and causal functions, P1 hierarchical categorical effects and P4 counts/compound amounts. The current research recommendation is the challenger-selection specification in `prior_selection_v1.json`. Use a strong open synthetic-prior reference with comparable mechanism coverage, then isolate changes to this recipe. TabPFN already motivates testing trees, unequal feature relevance, input dependence, categorical re-embedding, many-class labels, noise, warps, quantization, missingness, and broad feature/context sizes. Its temporal/group/OOD/spatial coverage also limits novelty claims. This is an evidence checklist, not a replacement mixture or an instruction to add every mechanism at equal weight.

Prioritize useful inference difficulty at the model's actual support sizes. A deeper random graph or tree may yield mostly unobserved mechanisms, and high synthetic loss may represent irreducible noise. Compare the unchanged sampler with context-conditioned complexity at fixed architecture, optimizer, shape distribution and compute. Measure category/leaf occupancy, effective rank and learnable improvement; do not promote a generator solely because its marginal distributions look realistic. A generator's latent forward equation does not automatically supply the Bayes conditional target distribution given the observed table.

### Finance extension near fraud prevalence $10^{-4}$

1. **Preserve finite positive evidence.** At rate $10^{-4}$, one and three million independent historical records have expected positive counts of 100 and 300 before label delays or selective verification. These are expectations, not guaranteed support counts. Draw finite outcomes once and retain zero-positive periods. Campaigns and shared entity effects can make repeated rows less informative than their raw count suggests; sample and evaluate those dependencies explicitly.
2. **Model what labels are actually available.** Event times, feature availability, label maturity and verification should determine the labeled support. An unreported transaction is not a known negative. A complete matured cohort can supply a population reference rate; a selectively verified pool generally cannot. Preserve an unavailable-reference flag and the reference cohort size. These are proposals for the finance sampler, not disclosed details of the current TabPFN generator.
3. **Test structured missingness and observed group effects.** The v2 report explicitly documents MCAR, while later releases disclose too little to infer the absence of MAR/MNAR. Test observed-state and value-dependent missingness against matched simpler controls. Include group/category partial pooling and changes in recorded measurements, but retain the distinction between observable shift cues and unidentifiable changes in label mechanisms.
4. **Match support selection to deployment.** Train on the same deduplicated natural-reservoir, available-positive and local-negative contexts used at inference, with legal source/count metadata. Distinguish nearest-within-a-candidate-pool from search over every eligible historical row. Importance-weighted query sampling may estimate natural-population risk; enriched support does not justify silently applying an additional scalar class-prior correction to an already population-targeted model.
5. **Keep the three prediction tasks explicit.** Binary and multiclass fraud must preserve the declared class universe even when some classes are absent from support. Amount regression and realized-loss regression have different target/observation laws. The latter can require a genuine atom at zero plus a positive continuous distribution; the small-data ScoringBench result alone does not validate that target law.

### Avoid negative transfer and inflated novelty claims

- Arbitrary unobserved concept changes cannot be recovered from old labels alone. Include observable cues or retain the resulting uncertainty.
- Unseen categorical IDs cannot reveal arbitrary new per-ID target effects without informative attributes or a shared hierarchy.
- Broader support does not guarantee better finite-compute transfer. Keep an unchanged standard-only control and compare joint training across binary, multiclass and regression tasks separately.
- Apply label selection, missingness, coarsening and target observation in the order defined by the data-generating process. Independent corruption can imply contradictory transaction records.
- Broad mechanism names are already established. Any claimed gain must survive a same-backbone comparison, independent seeds, and a locked real-data confirmation set.

### Evaluation and GPU batch latency

Use the fixed official standard benchmark splits and protocols, with TabArena as the primary standard comparison and BeyondArena and ScoringBench providing relevant shift/scale and predictive-density checks. State the eligible dataset subset, missing coverage, label alphabet, group-ID treatment, estimator count, adaptation and inference budget. Do not combine a best-family leaderboard entry with the cost of a single base-checkpoint pass.

For finance, keep natural-prevalence test windows and chronological validation. Historical support labels must be known by the prediction snapshot; validation labels used for model or threshold selection must be available before the first scored test prediction. Distinguish frozen snapshots from rolling updates. Retain zero-positive windows and absent support classes, and use campaign/entity or time blocks when estimating uncertainty. Report binary average precision, log loss, calibration and precision/recall at declared review budgets; report multiclass and regression results separately. Verify the outcome-observation rule before treating any matured test cohort as complete.

Match large-history input information across frozen ICL and separately trained tree baselines. Report historical feature/index preparation, context selection, cold support prefill and warm GPU batch prediction separately, then give end-to-end time and peak memory. Record actual support rows, searched candidate counts, active route counts and query batch shapes. Bounded transformer contexts do not imply that processing millions of historical rows is free, and cached throughput does not describe the first burst after setup. Hardware measurements on the actual MI355X topology remain necessary.

## Reproduction caveats for the parent report

- Call this a proposed bank until trained-checkpoint evaluations establish a scoped performance claim.
- Do not infer precise mixture weights from terse proprietary reports.
- Do not credit priors for architectural, encoding or inference-compute gains without controlled ablation.
- Do not equate label/feature permutation ensembling with a one-pass result.
- Do not claim temporal/group/spatial or high-cardinality coverage is new by itself.
- Do not claim the strongest TabPFN system is fully reproduced by loading the default open-weight checkpoint.
- Do not derive runtime on 64 MI355X GPUs from v2 RTX-2080 training cost or v3's eight-trillion-token statement. Benchmark generator and training throughput on the actual topology and shape mix.
- Synthetic-only pretraining does not eliminate meta-validation leakage or public-benchmark overfitting.
