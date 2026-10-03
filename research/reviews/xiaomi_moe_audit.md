# Xiaomi-TabLDM MoE: capacity, latency, and the proposed ICL model

Audit date: 2026-10-02. Primary evidence: supplied `xiaomi.pdf` / `papers/extracted/xiaomi.txt`, with physical PDF pages recorded below; official repository and model card; AMD primary documentation. No Xiaomi inference or MI355X timing was run. The browser returned differently aged repository snapshots, and shell Git could not resolve GitHub, so the code observations below identify file/function locations without pretending that all pages constitute one pinned checkout.

## Decision for this project

**Keep the dense 315M design as the reference and current default; promote a small, late-layer MoE to a serious compute-matched challenger before the main training commitment.** Xiaomi justifies testing conditional capacity, not claiming that MoE inherently lowers latency. Given accuracy priority, 64 MI355X GPUs, and a finite 50,000 GPU-hour project budget, the final choice should depend on validation quality per GPU-hour and the actual desired inference regime. It should not be decided from total-versus-active parameter counts alone.

A particularly clean challenger replaces the final eight ICL FFNs with four routed experts (top-1) plus one shared expert, each with half the dense hidden width. It has approximately **419M stored parameters at the same leading FFN arithmetic as the 315M dense design**, before routing/dispatch costs. This is our proposal, not Xiaomi's published configuration, and neither architecture has yet been validated on this project.

## What Xiaomi actually implements

The supplied paper, §2.1 p.6 and Appendix A Tables 6–8 p.22, specifies:

| Item | Xiaomi released architecture |
|---|---:|
| ICL depth and token width | 24 blocks, width 512 |
| ICL heads | 8 |
| Dense FFN hidden width | 1,024 |
| MoE placement | Final 8 of 24 ICL blocks |
| Routed experts per affected layer | 2 |
| Routed experts active per row token | Top-1 |
| Always-active shared experts | 1 |
| Each expert | Two-layer MLP, hidden width 1,024 |
| Routing input | Individual row-token representation |
| Router jitter | 0 |
| Initialization | Initialize experts from the dense FFN |

Column embedding and row aggregation remain dense. This is not an all-layer MoE, not eight routed experts, and not a model activating only one of many FFNs overall. Every affected token executes **one shared FFN and one routed FFN**.

Training is dense for stage 1: 500k classification / 300k regression steps. MoE and its auxiliary losses are enabled starting at stage 2, followed by the long-context stage (§2.2 p.6). Table 9 p.23 specifies load-balance coefficient 0.01, router z-loss coefficient 0.001, and auxiliary multiplier 1.0. The paper describes summing those terms over MoE layers. Training also changes context lengths, learning rate, and FlashAttention use at this transition, so stage gains cannot isolate expert routing.

Table 10 p.23 reports:

| Checkpoint | Total parameters | Active parameters |
|---|---:|---:|
| Classifier | 70.08M | 61.67M |
| Regressor | 71.08M | 62.68M |

Only about **12% of total weights are inactive for a given token**. “Active per token” is the useful interpretation for sparse arithmetic: a batch can route different tokens to both experts and touch all routed weights. Do not interpret the active count as a guarantee that the unused expert can be omitted from GPU memory or that a whole inference call reads only that many weights.

### Arithmetic consequence (our derivation)

Let one ordinary expert-sized FFN cost one unit per token. The same-width 24-layer dense ICL stack costs 24 FFN units; Xiaomi's layout costs `16 + 8*(1 shared + 1 routed) = 32` units, approximately **33% more ICL FFN arithmetic than its same-width dense counterpart**. Attention and front-end work are unchanged by this isolated replacement; router/dispatch overhead is additional. Relative to executing all three expert FFNs in each affected layer, sparse routing saves one expert's computation. Those are different comparisons. Sparse capacity can outperform a similarly costly dense network, but this calculation does not predict wall-clock speed or predictive quality.

The paper's prose p.6 emphasizes dependence on routed top-k; the always-on shared expert must also be counted in actual compute.

## What the public code adds—and what it does not establish

The official [`attnres_light_rmsnorm_moe.py`](https://raw.githubusercontent.com/xiaomi-research/xiaomi-tabldm/main/tabldm/_model/attnres_light_rmsnorm_moe.py), `TabLDMSparseMoE` / `TabLDMMoE.__init__`, defines preset 1 as the paper's 2-routed/top-1/1-shared/last-eight layout. Other preset entries are not evidence that the released weights use them. `drop_dense_ffn` removes frozen dense initialization copies that are no longer used; checkpoint bookkeeping can therefore alter stored counts without changing active computation.

The official [`moe.py`](https://raw.githubusercontent.com/xiaomi-research/xiaomi-tabldm/main/tabldm/_model/moe.py), `SparseMoEFeedForward.forward`, performs FP32 routing, top-k selection, a Python loop over experts, selected-token gathering, and indexed output accumulation; shared outputs are added. It is not evidence of a fused MI355X kernel. Its normalized top-1 weighting becomes one, so the discrete route has no ordinary differentiable task-gradient through the selected weight; raw weighting and auxiliary-free modes also exist. `copy_from_dense` may zero routed output projections, depending on mode. `collect_moe_aux_loss` averages layers, whereas the paper describes a sum. These differences require a pinned checkpoint/config and training implementation before exact reproduction. None proves that the paper's checkpoint was trained with every current default.

The [official README](https://github.com/xiaomi-research/xiaomi-tabldm) exposes eight default ensemble estimators, optional context KV caching, AMP, attention-kernel selection, and offloading. It says the cache is built during `fit`, and currently disallows cached classification above ten classes. These implementation choices affect runtime independently of MoE. The [model card](https://huggingface.co/occams/Xiaomi-TabLDM) also explains that estimator `fit` preprocesses context and loads weights without model-weight training. Consequently, the benchmark's “training time” is downstream estimator fit time, not foundation-model pretraining cost.

## What the latency result does and does not show

Supplied PDF Table 2 p.12 and §4.2.2 p.13 report TabArena regression medians per 1,000 samples:

| Model | Predict time | Reported regression Elo |
|---|---:|---:|
| Xiaomi-TabLDM | 3.12 s | 1,900 |
| TabFM | 9.67 s | 2,019 |
| EXAONE-Tabular | 2.58 s | 1,885 |
| TabPFN-3 | 0.42 s | 1,800 |
| TabICLv2 | 0.25 s | 1,679 |

Xiaomi reports 68% less prediction time than TabFM, with a slight improvability gap (1.7% versus 1.6%). This is an overall system trade-off on 13 regression datasets. It is **not** a same-backbone MoE-versus-dense experiment. It does not show that Xiaomi is faster than every dense competitor; its own table reports substantially faster TabPFN-3 and TabICLv2 inference, at different reported accuracy.

TabFM is also a much larger model: the supplied TabFM paper Table 1 p.4 gives 408.7M parameters, width 1,024 in its ICL trunk, and a different two-stage front end. Xiaomi's roughly 63M active regression model differs in size, width, architecture, preprocessing, ensembling, and runtime. A MoE causal speedup cannot be extracted from that comparison. The Xiaomi paper does not supply an isolated dense/MoE accuracy-and-latency ablation or enough hardware/kernel/ensemble/cache detail beside Table 2 to reproduce an exact controlled speed ratio. Claims about this table should remain dated author-reported system comparisons, not MI355X projections.

For our benchmark, report separately: cold context construction plus first prediction; cached-query latency at query sizes 1/32/128; larger batch throughput; full preprocessing/ensemble time; peak memory; and p50/p95 after warmup and device synchronization. Keep context size, feature count, dtype, attention backend, ensemble count, cache state and output heads fixed.

## A precise, limited MoE challenger for ICL-315

Authored candidate: keep every dense module unchanged except ICL FFNs in layers 17–24. Replace each `d=1024, h=2816` SwiGLU with a shared `h=1408` expert plus four routed `h=1408` experts, top-1. Use a bias-free `1024 -> 4` router. Retain one common FFN output normalization/residual branch after combining experts. Keep experts on the same device initially; use token routing without capacity dropping. Router inputs may contain only the same legal row states as the dense network, never generator family IDs or query labels.

For three bias-free SwiGLU matrices, one half-width expert has `3*1024*1408 = 4,325,376` parameters. The new FFN plus router contains `5*4,325,376 + 4096 = 21,630,976` parameters versus `8,650,752` in the old FFN. Across eight replacements the increment is 103,841,792; total becomes **418,863,248**. Two active half-width experts have the dense FFN's `3*1024*2816` matrix arithmetic; routing adds 4,096 weights per layer, totaling 32,768. “315,054,224 arithmetic-active parameters” is shorthand under the same head accounting, not a wall-clock estimate or a guarantee that only that many weights are touched in a batch.

Do not silently copy normalized top-1 gating and assume end-to-end task gradients train the router. A concrete pilot should use a differentiable selected probability, for example `shared(x) + 4*p_selected*routed_selected(x)`, plus declared balance/z regularization. This makes gate magnitude variable, so log gate statistics and compare calibration. The simplest first training arm starts from scratch with the same curriculum. Dense-to-MoE continuation is a separate arm: a possible near-function-preserving initialization splits old SwiGLU hidden channels into two halves, places the first in the shared expert, copies the second into every routed expert, and initializes router logits near zero. Equal gate probability then approximately preserves the dense function. Any perturbation, optimizer-state transfer, and warmup must be fixed before comparing; continuation compute counts toward the budget. This is an experiment specification direction, not a tested implementation.

## Implications for 64 MI355X GPUs

AMD specifies [288GB HBM3E and 8TB/s peak memory bandwidth per MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html). A 315M or 419M model's weights are small relative to that memory; activations and context caches can dominate. **Expert parallelism across nodes is unnecessary for these parameter sizes.** Start with replicated data parallel training and all experts resident on each GPU, avoiding expert dispatch all-to-all. MoE still increases optimizer work, gradient communication, stored weights, dispatch overhead and implementation complexity.

[ROCm AITER](https://github.com/ROCm/aiter) lists MI355X support and fused MoE kernels; AMD's [inference article](https://www.amd.com/en/developer/resources/technical-articles/2026/inference-performance-on-amd-gpus.html) describes expert dispatch/combine optimization for large models. This establishes available engineering building blocks, not that our BF16 training backward pass, Muon update, expert shape, and small tabular batches are already optimized. Validate actual kernels. DeepSeek-style inference numbers do not transfer to this model.

Small cached-query batches can fragment expert GEMMs and become dominated by dispatch/launch overhead. Long contexts can instead be dominated by quadratic ICL attention, which FFN sparsity does not eliminate. A shared expert can mitigate common-skill duplication and negative transfer, but routing can also undertrain rare regimes or mostly split classification from regression without adding useful within-task specialization. Monitor expert load, tail-task quality, invariance under category/class relabeling, and performance on held-out prior mechanisms.

Suggested decision sequence: first run operator/forward/backward microbenchmarks on actual shapes; eliminate broken or clearly inefficient implementations; then train dense and this limited MoE with paired seeds, identical priors, and matched GPU-hour endpoints. Test a dense wider-FFN model with approximately the same stored parameter budget if distinguishing sparse specialization from simple capacity matters. Choose by held-out predictive quality and measured inference constraints. MoE deserves this controlled trial; current evidence does not justify replacing the dense default unconditionally or advertising lower latency before measurement.
