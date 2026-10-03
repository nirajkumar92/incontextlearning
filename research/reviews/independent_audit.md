# Independent research audit — 3 October 2026

**Historical audit with implemented follow-up.** The findings below describe the pre-candidate baseline. The subsequent [main recipe](../../docs/main_recipe.md), [upstream-code audit](upstream_code_audit.md) and [volume plan](../../docs/prior_volume.md) implement the reference adapter, revised representations/heads, query KV compression and finance controls and recommend a concrete 317M candidate. Remaining compact-cell, statistical-runner and hardware experiments are still distinguished from delivered code. This audit is retained because it explains the counterexamples and research budget; its earlier “unimplemented” statements are not the current code map.


The current implementation should remain a research baseline. It has useful information boundaries and finite-population accounting, but the evidence does not select its prior or 315M architecture for the final run. This audit changes the research recommendation and budget; it does not claim a trained improvement.

Three reviewers started with fresh context and read primary sources before reviewing the proposal: priors, architecture, and training/evaluation. Their combined scope covers all nineteen retained papers. The lead reviewer reconciled their findings, reran counterexamples, inspected the numerical repair and revised the existing report. This is independent analysis within the project, not external peer review or benchmark replication.

The authoritative narrative is [the existing TeX](../../prior_bank_assessment.tex). The costed program is [prior_bank_spec.json](../specs/prior_bank_spec.json); [the experiment register](../specs/ablation_plan.csv) distinguishes funded work from optional branches. Runtime configurations still launch the authored baseline. Several revised research arms need implementation.

## Disposition

| Priority | Finding | Decision and remaining evidence |
|---|---|---|
| P0 | Query targets were cast to FP32 before support-affine normalization; serving also forced means to FP32 | **Repaired.** Keep original-unit targets/location/scale in FP64 through subtraction and inversion, with neural probabilities in FP32. Six targeted tests cover likelihood/gradients, CDF/quantiles, means, hurdle and serving. |
| P0 | The 128-bin regression NLL cannot distinguish targets inside one bin | **Unresolved design issue.** Exact loss/gradient counterexample below. Fund 999-quantile versus 1,025-bin candidates before choosing a regression model. Do not call normalization tests evidence of RMSE adequacy. |
| P0 | Distinct scalar category hashes can produce identical BF16 projected embeddings | **Unresolved representation issue.** CPU counterexample below. Fund a two-coordinate hash candidate and repeat on the actual GPU stack, trained weights and recoding seeds. |
| P0 | Finance flag H already supplies near-0.90 population AUC | **Change evaluation.** H-only, hidden-input-H and nondiscriminative-generative-H are separate controls. Existing simulator remains an interpretable stress world. |
| P1 | Authored SCM replacement is narrower than the strongest open scaffold | **Change starting point.** Preserve pinned TabICLv2 generator; test eligible 95/5 P1 and 95/5 P4 additions separately, with task and shape fixed first. |
| P1 | A dense-only width/depth sweep cannot select the best architecture family | **Change selection.** Fund released row-compressed, authored compressed and compact persistent-cell candidates. Unequal-size family comparisons are equal-cost system comparisons, not topology-only attribution. |
| P1 | Learned length scaling saturates beyond 4,096 keys | **Elevate an acknowledged risk.** Fund one length-scaling contrast and single-anchor dilution tests before MoE. |
| P1 | Entropy normalization and sharing alter gradient allocation | **Measure, do not assume harm or benefit.** Proper population weighting remains; compare unit versus entropy normalization and shared versus specialist checkpoints. Record gradients/clipping by task and reference availability. |
| P1 | Old 700-hour size endpoints extrapolated to a 22,500-hour final run | **Replace allocation.** Two 3,000-hour finalist runs precede a conditional 12,000-hour combined final ceiling. Final-budget extrapolation remains, but is four-fold rather than over thirty-fold. |
| P1 | Required controls, statistical promotion and failed-sharing fallback were incompletely funded | **Make decisions explicit.** Exact 50,000-hour program below; unsupported results stop or redirect spending. Native reference adapters, variant heads and the complete paired evaluation runner remain implementation work. |
| P2 | MoE, GQA, adaptive mixtures and auxiliary objectives add unresolved factors | **Defer.** No initial allocation; a future run requires named reassignment. Xiaomi's system latency does not isolate a MoE speedup. |

Keep the support-only codec, no-query-key attention, isolated loss/audit state, finite row identities, label-availability cutoffs, original macroepisode denominator, correct proposal weights and exact frozen-weight caches. No direct query-label forward leak was found in the inspected paths. None of these properties establishes superior prediction.

## Executed counterexamples and their limits

Run from the repository with its local environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  research/probes/independent_audit_probes.py \
  --output research/results/independent_audit_probes.json
```

[Recorded results](../results/independent_audit_probes.json) include the executable source hashes, PyTorch/NumPy versions and evidence scope. These are CPU probes and analytic calculations, not trained-model evaluation.

- **Affine precision:** targets `[100000000, 100000001]`, center `100000000`, scale `0.5` require standardized targets `[0, 2]`. The old cast-before-subtraction order produces `[0, 0]`. Repaired code returns `[0, 2]` and different parameter gradients. The head and parameter count remain unchanged.
- **Regression resolution:** standardized targets `0.005` and `0.070` give exactly the same NLL and every output gradient under the 128-bin head. In constant-target regression, the population NLL-optimal mass gives mean `c + 0.0390625`. This mathematical counterexample does not require model training. An odd 1,025-bin grid puts zero at its central midpoint and reduces the generic resolution floor; it does not eliminate it. The reference quantile head is a substantive alternative.
- **Categorical identity:** under encoding seed 0, `category_28838` and `category_39200` have distinct hashes separated by `1.609306e-9`. With equal frequency/missing flags, a tiny encoder initialized at seed 1729 gives identical CPU BF16 embeddings, despite an FP32 difference of `3.576279e-7`. This is one observed initial encoder, not a proof that all trained weights or MI355X kernels alias the pair. Test raw codes, rendered features and actual projected vectors separately.
- **Finance shortcut:** with prevalence `pi=1e-4`, sensitivity `a=.8` and false-positive rate `h`, the H-only score has `AUC=(1+a-h)/2`, precision `pi*a/(pi*a+(1-pi)*h)` at recall `.8`, and threshold-step AP `a*precision+(1-a)*pi`. For `h=1e-6/1e-4/1e-3/.01`, AP is `.79014/.35560/.05928/.00637`. The two tied score levels enter together; this is not trapezoidal PR-AUC. Rendering may rename H's categories but preserves its information. This is a strong assumed covariate, not automatically label leakage.
- **Gradient allocation:** the entropy multiplier at `pi_ref=1e-4` is `979.4`. Under balanced query proposals, calibrated `p=pi` gives normalized logit derivatives near `±.196`; a negative with `p=.5` gives `979.3`. Correct reference initialization matters. This is not a proof of instability. Incomplete-reference worlds receive multiplier one, so nominal 20% finance episode mass is not a fixed gradient contribution.
- **Support dilution:** one relevant key with fixed logit gap has weight `1/(1+(n-1)*exp(-gap))`. A gap yielding half the mass at 4,096 keys yields about `.111` at 32,768. A learned representation can change that gap. The check motivates a trained stress test rather than proving failure of every long-context transformer.

## Changes to the scientific recommendation

The authored bank has useful hierarchy/count modules, but its SCM uses a single exogenous root, scalar nodes, four node-function families, at most sixteen observed informative causal nodes and fixed additive noise. Much added width comprises proxies or distractors. This differs materially from the open reference's vector-state, eight-function construction and categorical/hidden-state mechanisms. Preserve that breadth before claiming a stronger prior.

The first prior screen has four arms: unchanged reference R; R plus 5% P1 where `F>=8`; R plus 5% P4 on binary/regression with `F>=4`; and the authored replacement. Draw reference task/shape before the branch, return ineligible mass to R, and do not silently apply P4's old autonomous task proportions. Preserve the reference filter on reference draws and log realized accepted masses. Both new adapters are missing today. Combine 90/5/5 only after independent gains and a named funded confirmation.

The finance implementation correctly tests finite evidence and selection, but its predictive geometry is only eight rotated Gaussian signal dimensions with largely recurring motifs. Entity offsets do not produce entity-specific risk. Campaign shifts do not create new attack mechanisms. Removing visible H must affect the selector as well as network inputs; this still leaves H-mediated feature/observation signals. Making H generatively nondiscriminative changes the population law and is a different experiment. Broader unseen-mechanism and observation-policy stresses precede any finance-scale claim. Do not add every missing mechanism without a failure that justifies it.

A compact persistent-cell architecture is now funded alongside row compression. Most authored 315M parameters operate after compression; higher capacity there cannot guarantee retention of every predictive cell distinction. Conversely, an information-bottleneck argument does not prove the existing 1,024-wide row representation is inadequate. Compare actual predictive quality and cold/warm batch cost. Cache correctness is retained, but each route still incurs support prefill, and context changes invalidate ordinary caches.

The promising edge is learning from the right evidence under the right selection and observation laws, while preserving useful feature identity and conditional structure. Conditional-label supervision may reduce training variance; it does not change the ideal expected objective or prove improved transfer. Complexity caps are an experiment, not a theorem that high-dimensional small-context tasks should be removed.

## Source coverage

Physical page numbers refer to the retained PDF files; citations identify primary sources. These are targeted source-section reviews, not independent replications of published results.

| Retained source | Material revisited | Main audit consequence |
|---|---|---|
| [TabPFN](https://arxiv.org/abs/2207.01848) | §§2–4, pp2–6 | Simplicity, causal/anti-causal features and hierarchical sampling are established; small numerical classification scope. |
| [TabPFNv2](https://doi.org/10.1038/s41586-024-08328-6) | Methods pp9–11 | Vector causal mechanisms and rich target/observation construction; disclosed table counts do not reveal every generator detail. |
| [TabPFN-2.5](https://arxiv.org/abs/2511.08667) | §3 pp4–5 | Real-data continuation is separate provenance; development/HPO investment differs from final pretraining cost. |
| [TabPFN-3](https://arxiv.org/abs/2605.13986) | §2.5 pp10–12; evaluation/scaling appendices | Dynamic/OOD/high-frequency priors already exist; token counts and forward-only timings do not determine project cost. |
| [TabPFN-3.5](https://arxiv.org/abs/2609.17895) | §§2.2,3.1–3.4 pp5,8–11 | Shared training, Fourier/rank encoding, QK normalization and wider trunk change together; no universal slice dominance. |
| [TabICL](https://github.com/soda-inria/tabicl) | §3 pp4–6; App E pp22–24 | Support-only summaries and row compression justify masks/cache design; memory offload leaves attention arithmetic. |
| [TabICLv2](https://arxiv.org/abs/2602.11139) | §§3–7 pp3–9; App A–E pp16–29; quantile App I | Direct prior×architecture interaction, 999 quantiles, QASSMax and useful negative ablations; released implementation matters. |
| [TabFM](https://arxiv.org/abs/2609.37959) | §3/Table1 pp4–7 | Larger row model is a credible family, not proof of optimum. Constant rows per step does not hold quadratic attention cost fixed. |
| [Xiaomi-TabLDM](https://arxiv.org/abs/2609.03880) | §2 pp4–6; App A pp21–23 | Separate tasks, stage-two MoE, QASSMax and AttnRes are coupled; latency versus TabFM is not an isolated MoE experiment. |
| [EXAONE](https://arxiv.org/abs/2608.25774) | §§2–3 pp4–10 | Compact persistent cells and quantile heads warrant a serious candidate; published optimizer/EMA are additional factors. |
| [Kumo Tabular](https://huggingface.co/blog/nvidia/kumo-tabular) | Supplied print pp3–7 | Coarsening, conflicting duplicates, missingness, cardinality and tails are prior art; training release availability needs verification. |
| [Mitra](https://arxiv.org/abs/2510.21204) | §3 pp3–5; Tables2/4; Apps A–C | Complementary mixtures can help; full-system fine-tuning/ensembling claims differ from frozen ICL. |
| [Mitra-v2](https://arxiv.org/abs/2609.04540) | §2.2–2.5; §4–5; App C | Mixture refinements are not uniformly beneficial; default eight-fold adapted system is a different comparator. |
| [LimiX-2](https://arxiv.org/abs/2609.17488) | §2–3 pp4–10; §6 pp22–24 | Persistent feature/target paths and fine histogram head; measured scaling to ~406M is not a compute-optimal law. |
| [O'Prior](https://arxiv.org/abs/2605.18971) | §3.1/Table2 pp7–8 | Nano-scale full combination loses to hybrid-only on reported TabArena AUC/F1; more realism is not monotone. |
| [Mind the Gap?](https://arxiv.org/abs/2605.06343) | §§3–6, pp7–9 and descriptor appendices | Descriptor proximity is not a validated optimization target; measured null correlation does not make all mismatch irrelevant. |
| [Towards Evaluating Data Priors](https://arxiv.org/abs/2606.29241) | §§3–5 pp2–4; App C/D pp11–12 | Nano, ≤512-row/16-feature classification comparisons cannot fix rankings at the intended scale and tasks. |
| [RTFM](https://arxiv.org/abs/2512.03307) | Objective/experiments pp2–4; Apps A/B pp6–8 | Adaptive teacher-relative gaps have cost and forgetting risks; stated task/update accounting is ambiguous. |
| [BeyondArena](https://arxiv.org/abs/2606.30410) | §§4–5 pp5–7 | Temporal/grouped/large settings and declared support/time policies matter for finance transfer. |

The current [official TabICL pretraining documentation](https://github.com/soda-inria/tabicl#pre-training) says cautious weight decay was not wired into the released Muon checkpoints. Pin that released behavior for reproduction. The paper separately discloses correlated-category/warping implementation discrepancies. RTFM's stated batch 64 ×3,000 steps ×30 iterations implies 5.76M nominal task presentations, while its text calls 90K synthetic datasets; do not use this as a precise cost estimate without clarification. The [TabFM 1.1.0 card](https://huggingface.co/google/tabfm-1.1.0-pytorch) lists cell width 256, inconsistent with paper/previously inspected tensor evidence; this unresolved card discrepancy does not overturn verified tensor counts. Live pages were checked on 3 October 2026; experimental runs still need immutable revisions.

## Revised 50,000-hour decision program

Outer ceilings: **5,000 systems/reference; 15,000 controlled screens; 10,000 size/bridges; 12,000 conditional final training; 5,000 evaluation; 3,000 uncommitted reserve.** Total remains 50,000 GPU-hours. The 15,000-hour screen is independently arithmetic-checked:

| Comparison | GPU-hours |
|---|---:|
| Four prior arms ×2 seeds ×250 | 2,000 |
| Two priors ×three backbones ×2 seeds ×350 | 4,200 |
| Head pair ×2 seeds ×150 | 600 |
| Codec pair ×2 seeds ×150 | 600 |
| Length-scaling pair ×2 seeds ×150 | 600 |
| Shared/specialist pair ×2 seeds ×200 | 800 |
| Optimizer pair including rate triage ×2 seeds ×150 | 600 |
| Six finance arms ×2 seeds ×200 | 2,400 |
| Two complete combinations ×2 new seeds ×800 | 3,200 |

Each allowance covers the **combined** classifier/regressor package when applicable. First split package hours equally between tasks. The second factorial prior must survive triage; retain R if none does. Native heads, class range, codec, task/shape law and optimizer need matching for causal comparisons. Cost and capacity are part of an architecture-family system comparison. The two confirmation slots include the selected challenger and any necessary refitted paired control; they do not fund two new candidates plus a third control. Both receive all three standard task panels and finance confirmation. No free partial cosine checkpoints replace matched controls.

The 10,000-hour size/bridge allocation comprises two sizes ×two seeds ×500 hours, two distinct 3,000-hour finalists and two 1,000-hour resolution runs. Freeze architecture first, or label the comparison a candidate frontier. The final 12,000 hours is a combined ceiling across selected checkpoints, conditional on the evidence; reserve is not an automatic second pretrain. New horizons/recipes require a new recorded run, because existing exact-resume logic cannot change configurations.

## Remaining implementation and measurement gates

1. Pin and validate the open reference, its released native heads and official benchmark interface. The current repo does not yet contain the faithful reference adapter or every revised variant.
2. Validate the numerical repair on CUDA/ROCm. Measure actual SDPA backend, generation/codec/transfer/forward/backward/optimizer/reduction costs and per-rank max/mean skew. Four ragged macroepisodes per rank on 64 GPUs can produce strong stragglers; a queue alone does not solve route-cost imbalance.
3. Add explicit normalization, H-control and task-fixed sampler switches; bounded gradient diagnostics; immutable continuation/cooldown artifacts; and paired evaluation aggregation. These are not existing runtime flags.
4. Pre-register lineage splits, comparator/inference roster, practical margins, multiple-comparison procedure and scheduled looks. Repeated folds are not independent datasets. Finance threshold selection needs adequate independent events and uncertainty; one true alert with no false positives cannot establish operational precision.
5. Cost the full applicable benchmark manifest within the 5,000-hour evaluation allowance before promising coverage. Preserve every failure and timeout. Large finance histories, natural prevalence and tuned full-history trees remain separate gates from standard benchmark rank.

No full candidate pretraining, official SOTA reproduction, MI355X throughput test or company fraud evaluation was performed in this audit. The strongest current recommendation is the controlled program above, not an assertion that the current bank must win.
