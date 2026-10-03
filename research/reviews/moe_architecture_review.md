# Dense versus a small MoE challenger for ICL-315

Date: 2 October 2026. This is an authored design audit, not an implementation or measured accuracy/latency result. It reads `research/specs/icl_architecture_v3.json` and the corresponding section of `prior_bank_assessment.tex`. The primary objective remains ordinary binary/multiclass classification and regression. All new numbers below are proposed settings unless identified as a source result or exact arithmetic.

## Recommendation

Keep dense ICL-315 as the reference and test one **ICL-MoE-419** challenger: replace only the final eight row-trunk FFNs with one shared and four routed half-width SwiGLU experts, selecting one routed expert per row. This tests conditional capacity where row states already contain task evidence. It does not repair the compressed-row information bottleneck and does not reduce the quadratic support-attention term. Do not put MoE in the cell encoder in this first experiment.

MoE is justified as a capacity-per-row-compute hypothesis, not as a necessary ingredient of a winning tabular model. Late routing may partition different inference computations, local regimes, or noise/interaction patterns. It can also merely recognize sampler artifacts, class IDs, or support/query roles. Useful specialization must be demonstrated by held-out real tasks and priors absent from pretraining; assignment entropy and visually distinct experts are not enough.

The cell frontend has only 4,728,528 MAB parameters in this design, versus 308,383,104 in the row trunk. Its many feature-cell positions already make execution expensive. Expert routing there adds dispatch work before the representation has much task information and does not address where most stored capacity resides. Dataset-level routing would amortize dispatch and can use a support-only summary safely, but sends each whole task to few experts and obscures within-task regimes. Neither alternative is selected here.

## What Xiaomi actually supports

The supplied Xiaomi-TabLDM v2 report, §2.1–2.2, PDF pp. 5–6 and Appendix A Tables 6–10, PDF pp. 22–23, specifies a width-512, 24-layer ICL predictor with MoE in its final eight blocks. Each replaced FFN uses two routed experts, top-1 selection, and one shared expert. All are ordinary two-layer MLPs with hidden width 1,024, equal to the replaced dense width. Dense-to-MoE initialization occurs when MoE is enabled in stage 2. Thus each replaced layer activates **two full-sized FFNs**, including the shared path: its arithmetic is not equal to one original FFN merely because top-k equals one. The table reports 70.08M total/61.67M active classifier parameters and 71.08M/62.68M for regression. These are useful precedents, not measurements for ICL-315. The complete model also changes feature grouping, residual propagation, training curriculum, priors, and inference; its headline performance is not an isolated MoE gain. [Xiaomi report](https://arxiv.org/abs/2609.03880).

Primary literature establishes the relevant ingredients. Switch uses sparse token-choice routing, an auxiliary load-balancing loss, and deliberately retained gate probabilities. ST-MoE studies router stability and z-loss. DeepSeekMoE motivates shared experts and finer expert segmentation. Sparse Upcycling demonstrates reuse of dense checkpoints. These are precedents from language/vision, not proof that this geometry improves tabular ICL. [Switch](https://www.jmlr.org/papers/v23/21-0998.html), [ST-MoE](https://arxiv.org/abs/2202.08906), [DeepSeekMoE](https://arxiv.org/abs/2401.06066), [Sparse Upcycling](https://arxiv.org/abs/2212.05055).

## Exact challenger and routing equation

Use the unchanged cell encoder, row compression, 24-layer trunk width 1,024, attention masks, norms, output heads, and prior. Layers are one-indexed. Replace FFNs at layers 17–24; retain FFNs at layers 1–16. Every replacement has:

- One always-active shared SwiGLU expert, hidden width 1,408.
- Four independent routed SwiGLU experts, each hidden width 1,408.
- One bias-free router matrix of shape `[4,1024]`.
- Top-1 token-choice routing, temperature 1, deterministic lower-index tie breaking, no jitter, no dropout, no expert capacity limit, and no token dropping or rerouting.
- The same RMS3 input and RMS4 residual-output normalization as the dense FFN; no additional expert norms, per-expert biases, or learned scale parameters.

For a row's FFN input `z=RMS3(U)`, calculate logits and softmax in FP32:

\[
a=W_rz,\quad p=\operatorname{softmax}(a),\quad j=\arg\max_e a_e,
\qquad F_{\rm MoE}(z)=F_s(z)+4p_jF_j(z).
\]

This **scaled-Switch gate** is an authored choice for half-channel upcycling. It retains the gradient of `p_j` through the full four-expert softmax; the top-1 index is treated as discrete. Do not compute softmax only over the selected singleton: that makes its weight exactly one and removes the task-loss gradient to the router. Almost everywhere away from ties, `d(4*p_j)/da_l = 4*p_j*(1[j=l]-p_l)`. The task gradient can still vanish for particular losses or identical downstream signals; this is not a guarantee of healthy routing.

The selected multiplier lies between 1 and 4. That gain convention is explicit and must be ablated if the pilot succeeds. Plain `p_j` would also retain the gradient, but starts the routed half at approximately quarter strength and is a different initialization/training recipe. Renormalized top-2 would be another legitimate challenger, but activating two routed half-experts plus the shared half would cost 1.5 dense FFNs; do not silently change top-k without changing the arithmetic budget.

## Dense checkpoint conversion and controls

Choose one fixed, already-budgeted dense prefix per seed before viewing continuation outcomes. For each replaced SwiGLU, apply the same fixed random permutation to its 2,816 hidden channels in gate/value rows and down columns. Split channels into halves A/B of size 1,408. Copy A into the shared expert; copy B into every routed expert. This decomposition satisfies `F_dense(z)=F_A(z)+F_B(z)` exactly before routing modulation. Initialize router entries independently as `Normal(0,(0.001/sqrt(1024))^2)`; expert weights remain exact copies. At uniform zero logits the scaled gate equals one and reproduces the dense FFN exactly. With the selected small random router it is **near-function-preserving**, not exact. Log actual forward differences on independent diagnostics; do not silently assert equality.

Initialize the equal-parameter dense control by widening the same final eight FFNs from 2,816 to 7,040 channels. Keep the original 2,816 channels and down columns, append 4,224 fresh gate/value rows sampled `Normal(0,1/1024)`, and set their down columns to zero. This initially preserves the dense function exactly, while new down columns receive task gradients. Repeating identical half-blocks without symmetry breaking can leave a dense-wide control artificially unable to exploit its extra capacity; avoid that control.

Reset optimizer states in all three continuation branches, keeping the same prefix and continuation schedule. The clean architecture screen uses AdamW for all branches: peak 3e-4, betas (.9,.95), epsilon 1e-8, decay .01 on projection matrices, zero on norms/embeddings/scalars, global gradient clipping 10, and the existing 2%-warmup/cosine-to-10% schedule over the allocated continuation horizon. Router uses AdamW. If Muon is retained instead, report that this is a joint architecture/optimizer-partition comparison: separately orthogonalizing smaller expert matrices is a different update geometry from orthogonalizing one dense matrix, even at initially matching functions. Do not attribute such a result solely to routing.

## Parameter and arithmetic accounting

A bias-free SwiGLU at width `d`, hidden width `h` has `3*d*h` parameters. With d=1,024:

| Quantity | Exact count |
|---|---:|
| Original h=2,816 FFN | 8,650,752 |
| One h=1,408 expert | 4,325,376 |
| Five experts + router in a replaced layer | 21,630,976 |
| Shared + selected routed expert + router per row | 8,654,848 |
| Dense ICL-315 total | 315,021,456 |
| ICL-MoE-419 total | 418,863,248 |
| ICL-MoE-419 nominal active parameter count | 315,054,224 |
| Dense last-eight-h=7,040 control total | 418,830,480 |

The total increment is `8 * ((5*4,325,376+4,096)-8,650,752) = 103,841,792`, or 32.9634% more stored parameters. The active expert matrix arithmetic is exactly that of the replaced dense FFN: two half-width experts equal one full-width expert. Across eight layers the router adds `8*2*1024*4 = 65,536` forward FLOPs per row, excluding softmax, dispatch, gathers/scatters, activation functions and norms. The dense-wide control has 2.5 times the last-eight FFN arithmetic; its total stored count differs from the MoE by only the 32,768 router parameters.

“Nominal active parameters” is the conventional per-token MoE accounting convention, not the number of distinct weights touched by a complete tabular forward pass. Different support/query rows can collectively activate every expert. It is also not a training-FLOP or latency measure: the frontend reuses parameters over cells, embeddings use indexed subsets, and backward/optimizer/communication have distinct costs. In particular, replicated training synchronizes gradients for the larger globally used parameter set, even when each row selects one expert. The approximately 33% larger parameter/gradient footprint can affect update time.

The row attention still has `n_support^2+n_support*n_query` pairs per layer under the existing mask. MoE changes none of this. For large support contexts, attention can dominate and the gain from holding FFN arithmetic fixed becomes less consequential. Routing into smaller GEMMs, especially after support caching leaves only a small query batch, can increase latency despite matching nominal matrix FLOPs.

## Balancing that preserves inference independence

For each of the eight MoE layers, balance only the four routed experts. For a global logical batch of B episodes, assign each support row weight `1/(2*B*n_support)` and each query row weight `1/(2*B*n_query)`. Weights sum to one. This is an authored role-balanced convention; it prevents long contexts or support rows from wholly dominating load statistics. Exclude padding.

Let `f_e=sum_t w_t*1[j_t=e]` (stop-gradient hard assignments), `P_e=sum_t w_t*p_te`, and `Z=sum_t w_t*logsumexp(a_t)^2`. Add

\[
0.01\,\frac1{8}\sum_\ell 4\sum_e f_{\ell e}P_{\ell e}
+0.001\,\frac1{8}\sum_\ell Z_\ell
\]

to the globally episode-averaged supervised loss. Use global statistics across the logical batch with correct distributed/autograd scaling. Coefficients multiply the **mean over eight layers**, not their sum. Do not enforce uniform usage within every task, prior arm, or class: this could suppress genuine specialization. Log those conditional loads as diagnostics. Disable these auxiliary losses at inference.

A training loss can couple gradients through global statistics without making the frozen forward map query-batch dependent. Every query's routing scores here depend only on its own task-conditioned row state and fixed model weights. Support states depend only on support. All experts fit on each device for this approximately 419M model; keep them replica-local initially, with ordinary data parallelism and no expert all-to-all. Support/query dispatch may be executed separately for caching. No capacity rule, batch-normalized router, quota-based assignment, online load-bias update, or cross-token expert choice is allowed in the forward path. Ordinary dispatch grouping may reorder rows internally but must scatter outputs back exactly.

Dropless MoE has primary systems precedent, but existing kernel results do not establish MI355X/ROCm throughput for this model. Expert-choice routing selects tokens competitively and is unsuitable for this first strict query-independent path. [MegaBlocks](https://arxiv.org/abs/2211.15841), [Expert Choice](https://arxiv.org/abs/2202.09368).

## Three-arm screen within the existing budget

Reassign the existing 1,500 GPU-hour screen reserve; do not add to the 50,000-hour cap. Compare D315, MoE419, and dense-wide419 with two paired prefix seeds and 250 GPU-hours of continuation per arm/seed: `3*2*250 = 1,500`. Charge each shared dense prefix once to its existing allocation and document it. Priors, task/shape curriculum, heads, final-data selection rules, and continuation optimizer convention remain identical. Extra MoE auxiliary loss is part of the treatment. Separate data streams for training and diagnostic evaluation remain mandatory.

Equal GPU-hours is the primary resource comparison. Also report loss/accuracy versus accepted episodes and estimated FLOPs, because slower dispatch can let dense models consume more worlds within the same allocation. The dense-wide branch is an equal-stored-capacity control, not an equal-FLOP control. These short continuations screen a promising branch; they do not establish the final long-training optimum. Stop/defer MoE if gains are confined to synthetic loss, vanish across the two seeds, or fail the chosen real development aggregate at the measured latency. A wide-confidence-interval pilot remains inconclusive, not a proof of equivalence.

Profile separate support-prefill and cached-query costs at the existing actual shapes, with query counts 1, 16 and 128, and report median/p95 latency, throughput, peak memory, active routing fractions, expert-token sizes, optimizer time, and communication. Retain the large-context comparison assigned to the attention workstream: changing to MoE and changing the attention algorithm simultaneously would not isolate either effect.

Required correctness checks are: query-label mutation; query alone versus in an unrelated query batch; query permutations and chunk sizes; cached versus uncached support; support-row permutation; packed-episode isolation; top-1 gate task-gradient finite differences away from ties; exact zero-router dense split equivalence; dense-wide zero-down equivalence; all-expert usage accounting; and explicit no-drop behavior under highly imbalanced routing. These establish the contract, not MoE's predictive benefit.
