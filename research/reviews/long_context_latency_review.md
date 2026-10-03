# ICL315: support context, GPU batch latency, and one long-context challenger

Audit date: 2026-10-02. Scope: `research/specs/icl_architecture_v3.json`, recipe `static-icl-315-v3.0`, and the concurrent assessment draft. This is an independent analytical review, not a throughput or model-quality measurement. All timings must be measured on the available MI355X implementation. The objectives are standard binary/multiclass/regression ICL and financial tabular prediction with bounded contexts.

## Decision

Keep the dense model as the accuracy reference. First implement its exact support/query separation, support-fitted preprocessing cache, frontend inducing cache, query batching, and an efficient ROCm attention path. GPU batch prediction makes initial support processing and frontend throughput important; a warm-query-only latency number is insufficient. A general hierarchical attention/retrieval challenger above 8,192 support rows remains a deferred research option. The current allocation prioritizes finance-context comparisons; this separate challenger needs an explicit reassignment before launch. Its benefit is a hypothesis, and its dense-attention approximation must be evaluated separately from implementation optimizations.

The highest-priority changes are: (1) profile the two-stage frontend's actual streaming schedule; (2) measure query batches 512 and 4,096 and whole evaluation folds; (3) test 512 versus 128 supervised queries for expensive long-context training; (4) audit attention-length calibration, which currently stops changing beyond 4,096 keys. These are more actionable than assuming a larger model or larger context must help.

## Exact cacheable computation

Preprocessing depends only on support. Cache per-feature finite-value statistics and sorted numerical support values for exact ranks, category-to-frequency/code maps, support target scaling, feature types, legal-class mapping and augmentation seeds. New queries perform lookups/searches in these caches. Re-running SHA256 on repeated categories or re-sorting support for every query batch is avoidable. CPU/GPU conversion, sorting, hashing, Fourier features, and data transfer belong in end-to-end timings. Replacing the specified FP64 feature construction with FP32 is an independently validated numerical approximation, not automatically the same codec.

The frontend has six column ISABs (three in each of two stages). Each ISAB first computes inducing states from support only; the second MAB maps each individual row against those inducing states. Save the second MAB's normalized/projected inducing keys and values, and its length-scaling metadata. Row attention acts within one row. Consequently a query's frontend can run independently of every other query once the support-derived inducing caches are ready. This remains true across the alternating stages. Store the exact post-normalization/projected tensors expected by the kernel, not an ambiguously named raw `H` that is normalized twice on reuse.

The 24-layer trunk also permits exact support prefill followed by arbitrary independent query chunks. In each layer, support states attend only to support; queries attend only to support; neither support nor another query reads a query's state. Save support K/V at every layer after the required Q/K normalization. Query execution needs Q, O and FFN projections, but not query K/V. A separate cross-attention kernel should exploit that; masking a larger dense square matrix does not ensure the same computational saving.

An induction on layers establishes query-batch independence: identical support caches and the same individual query give the same mathematical output regardless of the other queries in the batch. Floating-point reduction/kernel differences can produce small discrepancies and should be tested within declared tolerances. This allows batch-size tuning for GPU efficiency without introducing transduction. It does **not** allow incremental support insertion: an added support row changes preprocessing, inducing summaries and mutually contextualized support states, so the default cache must be rebuilt. Cache identity must include the support values/labels, codec/model version, legal labels, augmentation seed, feature order, dtype, and attention configuration.

Optional feedback after trunk layers 8 and 16 does not destroy mathematical cacheability, but needs its own support inducing states and per-query feature/CLS state. A row-embedding-only inference implementation cannot silently claim to implement that feedback. Keep feedback off during this cache/latency comparison.

## Cache size and peak memory

For BF16 full MHA, the trunk cache for one estimator is exactly

`M_KV = 2(K,V) * 24 layers * ns * 16 KV heads * 64 channels * 2 bytes = 98,304 ns bytes`.

| Support rows | Trunk K/V per estimator | Four estimators | Eight estimators |
|---:|---:|---:|---:|
| 8,192 | 0.75 GiB | 3 GiB | 6 GiB |
| 32,768 | 3 GiB | 12 GiB | 24 GiB |
| 100,000 | 9.155 GiB | 36.621 GiB | 73.242 GiB |
| 1,000,000 | 91.553 GiB | 366.211 GiB | 732.422 GiB |

These are cache arithmetic, **not validated context limits**. The last two rows do not establish successful model generalization or feasible prefill. Shared weights do not eliminate independent augmentation caches. Sequentially executing estimators reduces peak resident caches if caches are evicted/rebuilt or transferred, at an explicitly charged latency cost.

Six frontend caches, each containing 128 inducing keys and values of width128 independently for each feature, require

`6 * F * 128 * 128 * 2 * 2 bytes = 393,216 F bytes`.

That is 48 MiB at128 features, 192 MiB at512, or384 MiB at1,024. Raw support preprocessing/index storage, model weights, output buffers, allocator reserve, scratch space and any retained frontend activations are additional. Storing one full BF16 `ns × F × 128` cell tensor takes `256 ns F` bytes: 1 GiB at32,768×128 and8 GiB at32,768×1,024. Training stores/recomputes multiple intermediates and gradients; this single-tensor count is not its peak.

The proposed full-MHA cache is materially larger than TabPFN-3's deployed test-side single-KV-head path. TabPFN-3's report gives24 layers, head dimension64, eight support heads and one test-side KV head (Appendix C, PDFpp47–48); §2.4.2 pp8–9 reports approximately7 GiB per estimator at1M rows, including additional cached objects. One cannot transfer that figure to ICL315, which has16 KV heads. The report's 107-second cold fit/cache construction at1M rows and its much faster cached prediction illustrate the distinction; those H100/model-specific measurements do not predict MI355X performance. [TabPFN-3](https://arxiv.org/abs/2605.13986).

## Frontend streaming is a real design constraint

TabPFN-3's §2.4.1 (PDFp7, Figure6p8) computes per-column inducing states and then streams rows through its feature aggregator. That is directly useful engineering precedent, but ICL315 has **two alternating column/row stages**, unlike the single column-stack then row-stack in TabPFN-3 Appendix C. In the second stage, inducing summaries depend on first-stage row attention, which has mixed all feature columns.

A correct implementation has two choices: retain/offload the first-stage `ns×F×128` states and process the next column stack in feature chunks; or replay upstream row processing in support chunks while accumulating global inducing softmax reductions. The three second-stage ISABs are sequential: the next summary depends on updated support cells from the previous ISAB. Without retaining those cells, discovering their three summaries requires additional replay passes. Finally replay once using the completed caches to emit the final row embeddings. Claiming a single cheap pass or simply computing independent row-chunk summaries changes the function.

Implement an explicit execution plan and profile both retained-state and replay variants; do not guess which wins with MI355X HBM. Online softmax accumulation can stream keys exactly in real arithmetic: maintain each query's running maximum, exponential denominator and weighted numerator and rescale all accumulators when the maximum changes. Averaging attention outputs from independently normalized chunks is incorrect. FP32 accumulation and a dense small-case equivalence check are required. Chunking independent feature columns is safe only between row-mixing stages. TabICLv2's Appendix H, PDFpp37–39, documents a different offloading tradeoff: on its reported1M×500 case, disk offloading took450s versus115s with CPU offloading, while changing RAM requirements; neither figure is an ICL315 forecast. [TabICLv2](https://arxiv.org/abs/2602.11139).

## Forward arithmetic: prefill versus batch queries

Use one multiply-add as two FLOPs. Set `L=24,D=1024,M=2816` for the trunk, and `c=128,h=512,I=128` for frontend width, SwiGLU width and inducing count. Ignoring norms, nonlinearities, softmax, codec, initial projections, heads and padding, one dense support prefill costs approximately

`C_core_support = L ns (8D² + 6DM) + 4 L D ns²`.

With cached support, `q` query rows cost

`C_core_query = L q (4D² + 6DM) + 4 L D ns q`.

The lower query coefficient accounts for skipping its K/V projections. The frontend's six column ISABs and six row MABs add

`C_front_support = 6F(ns+I)(8c²+6ch) + 48cF ns I + 6ns(F+8)(8c²+6ch) + 24c ns(F+8)²`,

and cached queries add

`C_front_query = 6Fq(4c²+6ch) + 24cFqI + 6q(F+8)(8c²+6ch) + 24cq(F+8)²`.

These count each required matrix once. The replay streaming schedule described above, padded kernels, cache construction overhead or activation recomputation add work. The formulas are not a wall-clock model.

For128 features, approximate forward work in decimal TFLOPs is:

| Support rows | Support trunk | Support frontend | Cached512 queries, total | Cached4,096 queries, total |
|---:|---:|---:|---:|---:|
| 8,192 | 11.648 | 8.145 | 1.131 | 9.046 |
| 32,768 | 125.757 | 32.425 | 2.368 | 18.942 |

The frontend matters even after row compression. At32,768 support,128 features and4,096 query rows, cold forward work is roughly177.1 TFLOPs before omitted operations/replay. Cached queries alone are18.94 TFLOPs; neither establishes seconds without kernel measurements. For large `q`, attention reuses K/V tiles across queries and becomes more compute-intensive. It is misleading to multiply the entire cache's size by every query and call that an unavoidable memory-read lower bound: a batched tiled kernel shares reads. Conversely tiny query batches can become memory/launch dominated. Measure the selected query batch rather than extrapolating from single-row decoding.

An end-to-end latency report should show `T_preprocess`, `T_prefill+cache`, `T_predict(q)`, peak GPU/host memory and bytes transferred. Publish both cold `T_preprocess + T_prefill + T_predict(q)` and warm prediction. For a dataset reused for `J` batches, amortized time per batch is `(T_preprocess+T_prefill)/J + mean(T_predict)`; stateJ. Since the user wants GPU batch prediction, primary shapes areq512,q4096 and the complete evaluation query fold, split into independent microbatches as needed. Include one-pass as well as selected-ensemble results.

## Exact engineering versus model changes

Exact-computation engineering, up to floating-point roundoff: efficient tiled/Flash attention; omitting unused query K/V; cached support/inducing projections; online-softmax KV tiling; query chunking; masked ragged batches with correct episode boundaries; fusion; graph compilation; retaining versus recomputing intermediates; and sequence-parallel exact attention if its communication cost justifies it. FlashAttention's contribution is reduced materialization and memory traffic, not removal of dense quadratic arithmetic. [Original FlashAttention](https://arxiv.org/abs/2205.14135).

Accuracy-affecting model/policy changes: GQA/MQA, selecting/averaging existing KV heads, cache quantization, fewer features, support subsampling, sparse attention, retrieval, summary bottlenecks, truncating context or changing precision/codec. These require validation and some require training. GQA is a credible separate cache-saving architecture: the specified optional16-query/4-KV-head variant divides trunk cache by4 (0.75 GiB at32,768), but does not divide QK/AV pair arithmetic by4 because16 query heads still attend. The language-model GQA paper reports a quality/speed compromise and uptraining, not guaranteed equivalence for tabular models. Do not bundle GQA into the first sparse-routing comparison. [GQA](https://arxiv.org/abs/2305.13245).

## Long-context training supervision and length calibration

With the current128-query cap, each32,768-support world supplies128 explicitly scored targets. Moving to512 query rows increases dense attention pairs by only `384/32768=1.171875%` of the support-square term, and total input-row linear work by approximately1.167%. The query-dependent portions of the frontend also grow, but support processing dominates this example. This is a compelling **measured equal-GPU-hour ablation**, not a promise of four times the useful information: targets share the same world and support, so gradients are correlated. Preserve per-episode mean loss and logical episode weighting rather than allowing large query batches to change the task mixture. Log actual query targets, unique worlds and achieved update throughput separately.

For training, a detached inference KV cache is wrong: query losses must differentiate through support states, inducing summaries and parameters. Query microbatching is possible with correct accumulation/recomputation, but one optimizer step occurs only after all query contributions and shared support gradients have been accumulated. Do not perform independent optimizer updates using a stale support graph. Implementations may require retaining the shared graph or recomputing it; measure the actual saving. Raising deployment query batch size does not itself require query-to-query training because the specified model is batch independent.

The current scale `tau_h(n)=exp(tanh(a_h)*clip(log(n/1024),-log4,log4))` saturates for n>=4,096. Thus8,192 and32,768 keys have the same direct length correction. With fixed finite normalized-logit margins, softmax mass on a fixed number of relevant rows can dilute as unrelated support grows. This is a diagnosed risk, not proof of failure. TabICLv2 §3 pp3–4, Figure3p4 and §7 ablations motivate testing context-dependent scaling: its query-aware scaling maintains a small relevant cluster in an increasing-negative-sample toy experiment. Test the present clipped design against a smooth trainable log-length term under the same priors/budget, including many irrelevant support rows and rare informative examples. The mere presence of a length parameter is insufficient evidence of extrapolation.

The width/cell-cap issue identified during the initial review is resolved in the selected recipe. At support size8,193 or above, width1,024 cannot fit under the2^23-cell cap once queries are included. Long tasks therefore use widths128/256/512. A separate15% wide-task bucket uses1,024–4,096 support rows and widths512/1,024 with probabilities.35/.65, followed by the same cell-cap conditioning. This gives wide tables actual training mass without claiming infeasible long-and-wide shapes. Audit emitted dimensions after class floors and caps.

## One optional challenger: global summaries plus routed support buckets

This is an authored, unvalidated architecture for `ns>8192`, with GPU batch prediction in mind. It changes the trunk attention graph, while retaining the base frontend, priors, heads and24-layer1024-width trunk. It retains individually represented support examples in addition to summaries; “exact examples” does **not** mean exact equivalence to the dense model's contextualized representations. No novelty claim is made: inducing summaries and retrieval already have extensive precedent, including Set Transformer and TabDPT. [Set Transformer](https://arxiv.org/abs/1810.00825), [official TabDPT inference repository](https://github.com/layer6ai-labs/TabDPT-inference).

**Support routing, fixed before queries.** Make a128-dimensional deterministic feature-only index vector. Start from numerical support-rank coordinates `2u-1` plus missing flags, and feature-specific categorical one-hot indicators including a missing category; map these through a seeded signed random projection with entries±1/sqrt128 and divide by sqrtF. The categorical projection can be generated from hashes without materializing a full vocabulary-sized matrix. Use support-fitted ranks and the same frozen map for every query. No support labels, query labels, label-aware frontend outputs, or audit fields enter routing. This intentionally modest metric is part of the experiment; irrelevant features, collisions and category concentration can make it poor.

Build a balanced binary partition of support index vectors by repeatedly selecting the coordinate of greatest support variance (lowest coordinate index breaks variance ties) and splitting its stable sorted records into floor(n/2) and ceil(n/2) halves until every leaf contains at most2,048 rows. Break ties with persisted random support IDs assigned independently of values/labels; these IDs travel with records during permutation tests. Store each leaf's centroid. Select256 uniform support anchor IDs once, independently of query values and outcomes. For each query independently, choose the two closest leaf centroids by squared Euclidean index-vector distance, with stable centroid-ID ties. This fixes routes independently of other query rows. Group queries sharing a pair of leaves for GPU execution; grouping does not alter their key set. If fewer than two leaves exist, use all available leaves. Deduplicate anchor IDs already in a selected leaf.

**Attention graph.** Add one trainable initial matrix `G0` of128 global tokens of width1024, initialized i.i.d. Normal(0,0.02), adding131,072 parameters for a total315,152,528. At layerl, using the same MAB parameters as that layer's ordinary trunk block, first update `G_l = MAB_l(G_{l-1}, X_support^{l-1})`. Then update each support row with the layer's MAB against its own leaf's individually represented support rows, the256 anchors and `G_l`. Each query updates against its selected two leaves, the anchors and `G_l`. Global tokens read support only; all source support states are from layerl-1; current-layer global tokens are computed before their K/V projection for the support/query substep. No query enters any key/value source. Deduplicate support row IDs across anchors and leaves before softmax; all128 global slots remain distinct. No log-mass bias, multiplicity weighting, importance correction or additional loss is introduced: the summaries are learned memory slots, not asserted sufficient statistics or unbiased samples. Apply the existing FFN/norm operations on every MAB invocation. Share block weights across the global and ordinary substeps, but not across depth. Cache support K/V and global K/V for every layer. Do not inject artificial target labels into global tokens.

For leaf boundb2048, anchorsR256 and globalsg128, attention-pair work is bounded by approximately

`ns*(b + R + 2g) + q*(2b + R + g)`

per layer, versus dense `ns² + ns*q`; duplicate masking only reduces this count. At ns32768 this reduces the support-pair count by a factor of at least12.8 in the bound and query pairs by up to32768/4480≈7.31. These are **attention-pair ratios**, not end-to-end speedups: frontend work, projection/FFN work, global synchronization, routing and kernel packing remain. Cache size stays approximately96KiB per support row plus12MiB for128 globals across24 layers; retrieval is not a full-cache compression solution. Retain that fact in any scalability claim.

**Training.** Keep dense attention on the normal short/medium reference path. In the challenger run activate the above graph on20% of episodes with ns<=8192 and on all episodes above8192, retaining the current context/width curriculum and unchanged prior draws. Initialize learned global seeds normally; global updates share learned trunk weights. Train the graph actually deployed, including partitioning, anchor selection and retrieval errors. Otherwise switching a dense-trained checkpoint to this graph is an uncontrolled inference approximation. Query routing is non-differentiable and fixed by permitted features, so no straight-through estimator is needed. Length scaling uses the actual visible key count per substep, and rare-support stress cases must be included in its validation. The challenger does not make the two-stage frontend streaming problem disappear.

**Selection and budget.** This deferred comparison would cost 1,000 GPU-hours: dense/challenger × two seeds ×250 GPU-hours. Its current allocation is zero; the former allowance now funds finance-context comparisons. Reassign budget explicitly within the existing50,000-hour cap before launching it. If funded, use identical prior/world namespaces paired across arms and matched MI355X GPU-hours; report the additional global-token parameter count. A pilot alone is not enough to deploy a new final backbone. Carry a supported candidate to an allocated confirmation run only if real-data quality and measured batch latency warrant it.

Evaluate ns2048/8192/32768 and an explicit untrained-length diagnostic at65536 where data permit, widths32/128/512 when valid, q512/4096/full fold, and one/four estimators. Freeze evaluation subsets and routes from validation; measure cold and warm latency, memory, prediction score, probability quality, rare-class/rare-region errors, and disagreement against dense predictions. Compare the candidate also to dense uniform/stratified context reduction at the same end-to-end time. Report raw per-dataset deltas and uncertainty rather than declaring success from one aggregate or synthetic needle test. Neither larger available support nor faster attention guarantees that the selected information is useful.

## Practical verification gates

Before any substantial run: dense versus chunked/cache outputs and gradients on small tables; query-label mutation invariance; query-batch independence; support permutation with stable routing IDs; no cross-episode attention; exact cache byte accounting; cache invalidation after support/model/codec changes; correct global attention update order; route deduplication/masks; accepted shape distribution; and batch latency including preprocessing, compilation warmup policy, transfers and estimator count. Compile overhead should be separately reported and included in an appropriate cold-start measure, not erased from every number.

Measure rather than assume: dense-versus-sparse attention kernel dispatch on ROCm, gather/packing overhead, query route occupancy, retained versus replay frontend scheduling, microbatch/accumulation memory, and effective GPU-hours. CUDA/Hopper FlashAttention-3 results in competitor papers are not portable performance guarantees for MI355X. No learning curve, latency benchmark or predictive improvement for this authored architecture was obtained in this audit.
