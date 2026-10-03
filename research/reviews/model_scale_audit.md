# Model size, pretraining scale, and compute audit

Evidence audited 2026-10-02; scope revised 2026-10-03. The objective is **tabular binary classification, multiclass classification and regression through ICL**, including the finance fraud setting, with **64 MI355X GPUs and 50,000 total project GPU-hours**. Local PDF page numbers refer to `=== PDF PAGE n ===` in `papers/extracted/`. This note preserves the source measurements; current selected settings are in `research/specs/prior_bank_spec.json`.

## Assessment first

**Selected dense reference:315,021,456 parameters, or315,175,184 with the finance metadata adapter and hurdle head.** The capacity search retains approximately50M,100M and500M alternatives. A1B run is a conditional escalation. These are selected research settings, not a fitted compute optimum. Current competitive tabular models range from roughly 20M to 410M per checkpoint; strong small models do not establish that more capacity is useless. LimiX-2 gives positive capacity scaling through 406.2M, whereas Mitra's earlier recipe shows diminishing gains beyond 12 layers. Neither supplies the iso-compute experiment needed to choose the optimum for this new prior.

At the full 64-GPU allocation, 50,000 GPU-hours is 781.25 cluster-hours, or 32.55 days. This is an accounting conversion, not a claim that every training, generation, and evaluation phase will sustain full utilization. A small pilot may be faster or more efficient on fewer GPUs; increasing data-parallel width changes the optimization batch unless accumulation is adjusted.

Do **not** apply `tokens = 20 × parameters`. Across these papers, “tokens” can mean pooled rows, raw cells, or feature groups, and some reported data quantities are distinct tables while others are repeated instances. Attention costs also depend on row/feature shape and context/query ratio. Training exposure and data diversity must be logged independently.

## Supplied-paper evidence ledger

`NR` means not reported sufficiently to infer the quantity in the examined source; it does not mean zero. Counts are per checkpoint unless explicitly stated otherwise.

| Model/version | Size and architecture evidence | Actual pretraining disclosure | Exact locator / source |
|---|---|---|---|
| TabPFN v1 | 25.82M; 12 layers, width 512, FFN 1024, 4 heads | 18,000 updates × 512 tables = 9,216,000 episodes; 1,024 rows each; 20 hours on 8 RTX2080Ti = **160 GPU-hours**, derived | `tabpfn.pdf` App.E.1 p29, E.3 p30; [paper](https://arxiv.org/abs/2207.01848) |
| TabPFN v2 | Rounded **7M classifier / 11M regressor** reported retrospectively in v3/v3.5 release table. Cell-based alternating feature/sample attention; default feature grouping 2 | ~2M updates × 64 tables; ~130M generated table instances **each**; about 2 weeks on 8 RTX2080Ti per run = **2,688 GPU-hours**, derived. Support sampled up to 2,048; 128 query rows; features sampled in 1–160; 75,000-cell cap reduces rows on wide tables | `tabpfn2.pdf`, Methods, “Training details,” p10; architecture p9; size in `tabpfn3.5.pdf` p4; [paper](https://doi.org/10.1038/s41586-024-08328-6) |
| TabPFN 2.5 / 2.6 | Rounded v2.5 **11M classifier / 10M regressor**; v2.6 **11M / 13M**. Feature group 3; attention changes and thinking rows | Full synthetic training steps, table/cell counts, and accelerator-hours NR. ~100 small HPO runs for ~50 hyperparameters reported. Real-TabPFN variant adds real-data continuation; do not count as pure synthetic | `tabpfn2.5.pdf` §2 pp4–5; size in `tabpfn3.5.pdf` p4; [2.5 paper](https://arxiv.org/abs/2511.08667) |
| TabPFN 3 | **53M classifier / 58M regressor**. Column encoder: 3 blocks, d128, 128 inducing points, 8 heads; row encoder 3 blocks, 4 CLS; ICL 24 blocks, d512, 8 heads, test-side 1 KV head. Regression head 512→1024→5000 | Reports **more than 8T tokens** for final model, but does not provide sufficiently explicit row/cell/stage/checkpoint accounting or GPU-hours to convert this into reproducible processed rows or FLOPs | `tabpfn3.pdf` p4 Fig2a, §2.5 p10; App.C pp47–48 Tables2–7; [paper](https://arxiv.org/abs/2605.13986) |
| TabPFN 3.5 | **220M joint classification/regression checkpoint**; **Fast 84M**. ICL width512→1024, CLS4→8, heads8→16, head dimension64; test KV head remains64-dimensional. Shared backbone plus task-specific encoders/heads | Total episodes, exact tokens, optimizer steps, GPU-hours NR. Do not assign v3's >8T statement to v3.5. Size, prior, encodings, normalization and multitask training change jointly | `tabpfn3.5.pdf` p4 release table; §3.1 pp8–9; [paper](https://arxiv.org/abs/2609.17895) |
| TabICL v1 | Official checkpoint **27.051666M stored parameter elements**, including 8 frozen RoPE frequencies; metadata verification below. d128 frontend; 3 induced-column blocks, 3 row blocks, 4 CLS; 12 ICL blocks at d512 | 160K updates at1,024 rows; 2K at log-uniform1K–40K; 50 at uniform40K–60K, final stage trains only ICL block. Global batch512 → **82,969,600 nominal episodes**, derived. 20 days ×3 A100-40GB = **1,440 GPU-hours**, derived | `tabicl.pdf` §4.2 p6; App.E.1 p19; [paper](https://arxiv.org/abs/2502.05564), [official weights](https://huggingface.co/jingang/TabICL) |
| TabICLv2 | Official checkpoint **27.552258M classifier / 28.544991M regressor stored elements**, including 8 frozen RoPE frequencies each. Separate checkpoints; 3 column blocks d128/128 inducing vectors; 3 row blocks d128; 4 CLS; 12 ICL blocks d512; 8 heads, FFN expansion2; 999 regression quantiles | Per model: batch64; 500K×1,024, 40K×LogU(400,10,240), 10K×LogU(400,60,000) rows; max100 features. **35.2M episodes**, derived. **24.5 H100-80GB GPU-days =588 GPU-hours per model** (20+2.5+2 GPU-days by stage). ~1,176 GPU-hours for both separate runs, excluding development | `tabicl2.pdf` §4.1 p5; App.A.4/B.1 pp18–19; [paper](https://arxiv.org/abs/2602.11139), [weights](https://huggingface.co/jingang/TabICL) |
| Google TabFM **1.1.0 / September paper** | Paper calls it400M; official1.1.0 weights contain **410.220666M classification / 412.292861M regression stored elements**. Table1: spectral frontend d128; 2×3 column ISAB,256 inducing points;2×3 row blocks;8 CLS;24 ICL blocks d1024,8heads, FFN4096 | Four row lengths2,048/4,096/8,192/16,384 paired with table batches2,048/1,024/512/256. **4,194,304 rows/update**. Paper describes joint classification/regression target loss. Updates per stage, full episode/row totals, GPU-hours NR; released artifacts remain separate task files | `tabfm.pdf` front page links1.1.0, Table1 p4, §3.2/Fig3 p6; [paper](https://arxiv.org/abs/2609.37959), [1.1.0 weights](https://huggingface.co/google/tabfm-1.1.0-pytorch) |
| Google TabFM **1.0.0 / older release** | Official headers contain **1,639.444522M classification / 1,647.783213M regression elements**. Its classifier config uses d256 frontend and8CLS, hence2048-dimensional ICL;24 ICL blocks,FF factor4 | Do not transfer the new400M paper's counts or settings to this older release. This is the ~1.64B model referenced by EXAONE and Mitra-v2 | Official [1.0.0 weights](https://huggingface.co/google/tabfm-1.0.0-pytorch); classifier config at release revision below |
| Xiaomi-TabLDM | **70.08M total/61.67M active classifier;71.08M/62.68M regressor**. 24 ICL blocks,d512,8heads; final8 FFNs become MoE,2routed experts/top1 +1shared. Sparse active parameter counts differ from stored weights | Separate models. Stage1:500K classifier/300K regressor updates on1,024 rows; stage2:40K onLogU(400,10,240); stage3:10K onLogU(400,60K). Reported8A100-80GB,7–10days stage1 +2days stage2 +2days stage3 implies **2,112–2,688 GPU-hours for a described run**. The prose does not give a clean all-experiments or both-model aggregate. Global batch/total rows not established here | `xiaomi.pdf` §2.2 p6; App.A pp21–23, Tables6–10; [paper](https://arxiv.org/abs/2609.03880), [code](https://github.com/xiaomi-research/xiaomi-tabldm) |
| EXAONE Tabular1.0 | Exact classifier **20,807,866**; rounded regressor21.11M. d192,6heads,12CAST layers,FF expansion4;3item-summary and32feature-summary slots. Classification backbone~20.65M plus~0.156M head;999quantiles for regression | **~30M classification /~10M regression synthetic table instances across released-model training lineages**, not guaranteed unique mechanisms. bf16; Muon matrices/AdamW other parameters; WSD plus continuation; EMA usually.999. Per-stage shape distributions, GPU-hours, exact row/cell totals NR | `exaone.pdf` §2.1/Table2 p6, §3 p9; [paper](https://arxiv.org/abs/2608.25774), [weights](https://huggingface.co/LG-AI-Research/EXAONE-Tabular) |
| Kumo Tabular | Release says28M–215M. Official headers: Small **27.458378M clf /28.466343M reg**; Medium **61.485626M/62.492439M**; Large **213.668602M/215.683543M** stored elements. Detailed code config below | Small/Medium/Large see **35M/71M/137M tables**. These numbers are training tables, **not parameter counts**. Curriculum1,024 rows, then400–10,240, then400–60,000; max100columns. Global batches, stage update counts, exact token totals and GPU-hours NR | supplied `kumotabular.pdf` pp2,6; [NVIDIA release](https://huggingface.co/blog/nvidia/kumo-tabular), [official weights](https://huggingface.co/nvidia/Kumo-Tabular) |

### Source-specific interpretation cautions

1. **Table instances are not necessarily independent worlds.** Repeated contexts from one SCM or finite population, separately generated rows from the same parameters, and unique mechanisms should not share one counter.
2. **TabPFN v2's 130M is per final model.** Its selected HPO/inference portfolio is another cost. V2.5's ~100 short HPO trials are development cost, not 100 final pretrainings.
3. **One inference ensemble does not multiply the intrinsic parameter count.** Several permutations can reuse one frozen checkpoint; separately fine-tuned bags contain independent weights. Both multiply some inference work, and caches can multiply memory even when weights are shared.
4. **Current paper versus current release:** TabFM1.0.0 and1.1.0 are both public and have different widths. The1.64B statements in earlier comparison papers are not arithmetic errors. The new paper's400M shorthand matches the scale of the verified1.1.0 checkpoints, but not their exact stored-element count.
5. **TabFM's constant-row curriculum is not constant total FLOPs.** With batch B and row count R, keeping BR fixed keeps many projection terms roughly fixed. Dense ICL attention has a BR² term, which grows linearly with R at fixed BR. Across2K→16K rows this part grows eightfold. The paper's “compute invariant” description must not be used as a hardware budget identity.
6. A paper may report H100,A100,2080Ti time for a particular implementation; these are not exchangeable MI355X-hours. Compiler, precision, attention kernels, generator CPU work, collective communication, shape mix and occupancy determine practical throughput.

## Recent additional competitors and scaling evidence

### Mitra / Mitra-v2

Mitra's original App.B.3 pp26–27 reports72M for its12-layer,d512,4-head2D model and37M for1D;45M synthetic episodes,60hours on8A100-40GB (480GPU-hours, derived). §4.8 p9/Fig5 p10 compares4/8/12/16/20/24layers: benefits diminish beyond12; dataset returns flatten around18K×2,048≈37M episodes. Mitra-v2 §2/Table1 pp3–7 instead calls the v1/v2 classifier75.7M and regressor76.7M; **retain this version/report discrepancy**. V2 uses~14K updates×2,048≈28.7M nominal tasks; regression adds3K≈34.8M total;32H200 GPUs. Its §5 p24 explicitly says accelerator-hours are unrecoverable. Default deployment fine-tunes and bags8 models. Sources: [Mitra](https://arxiv.org/abs/2510.21204), [Mitra-v2](https://arxiv.org/abs/2609.04540).

### LimiX-2

§2 pp4–7: d256 cell representation,24dual-path layers; joint classification/regression/reconstruction. §6 pp22–24 tests six sizes12.5M–406.2M with a common generation/optimization/inference recipe. It fits Elo=α+βlog2(P/100M): β=34.68 onTabArena,22.16/18.26 onTALENT classification/regression,11.24/30.06 onBCCO;R² .9617–.9808. TabArena rises1766→1935 over the measured range. The2B extrapolation is explicitly a forecast. Released model is described as400M. Exact pretraining steps, processed tokens,GPU-hours, and iso-compute sweeps were not established in the report. Source: [paper](https://arxiv.org/abs/2609.17488), [official release](https://github.com/limix-ldm-ai/LimiX).

**Interpretation:** this is direct support for evaluating a400–500M challenger; it is not proof that a1B model beats the315M reference after both use the same project budget. A fixed training recipe can spend more compute on each larger model. Elo also depends on the comparison pool, and an extrapolated line is not a deployment result.

### TabDPT

[Original paper](https://arxiv.org/html/2410.18164v3), §§4.1–4.5/App.C:78M,16layers,600Kupdates;123real datasets/32Mrows/2Bcells;A100-40GB. Scaling spans33K–78M parameters and52M–2B **unique corpus cells**, fitting `L=A P^-0.42+B D^-0.39+E`. This D is not cumulative training tokens. Current [v1.3 release](https://github.com/layer6ai-labs/TabDPT-inference), September8, is different: official header63,040,624stored elements;32layers,d512,8heads; context lengths512–32,768 in stored training config. Config continuation fields do not establish completed exposure or GPU-hours. [Weights](https://huggingface.co/Layer6/TabDPT).

**Interpretation:** the original paper supports joint capacity/data scaling under real-data SSL; its fitted exponents do not transfer automatically to synthetic ICL or a new tokenizer. Its corpus-size variable cannot be inserted into an equation for processed training tokens without a repetition model.

## Public checkpoint-header verification

The measurements below were performed in this session. No checkpoint code or tensor computation was executed. A ranged HTTP read retrieved at most the first1MiB of each file. For safetensors, tensor shapes were summed from the JSON header. For PyTorch zip checkpoints, a restrictive metadata-only unpickler allowed OrderedDict, storage placeholders, and tensor shape reconstruction; all other global constructors were rejected. This inspected metadata without downloading full multi-GB weight arrays.

### Revisions and observed element counts

| HF repository and revision | File | Sum of stored tensor shapes |
|---|---|---:|
| `jingang/TabICL` at `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | `tabicl-classifier-v1-20250208.ckpt` |27,051,666|
| same | `tabicl-classifier-v2-20260212.ckpt` |27,552,258|
| same | `tabicl-regressor-v2-20260212.ckpt` |28,544,991|
| `nvidia/Kumo-Tabular` at `3cfed70f0ea063751fc50eefc37e32ac8db1db8c` | `small/classifier.pt` |27,458,378|
| same | `small/regressor.pt` |28,466,343|
| same | `medium/classifier.pt` |61,485,626|
| same | `medium/regressor.pt` |62,492,439|
| same | `large/classifier.pt` |213,668,602|
| same | `large/regressor.pt` |215,683,543|
| `google/tabfm-1.0.0-pytorch` at `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83` | `classification/model.safetensors` |1,639,444,522|
| same | `regression/model.safetensors` |1,647,783,213|
| `google/tabfm-1.1.0-pytorch` at `d7cfd3b8d184d7c8fed4210aeba9e1c967d45f2a` | `classification/model.safetensors` |410,220,666|
| same | `regression/model.safetensors` |412,292,861|
| `Layer6/TabDPT` at `a5ca6e01c0fa09ec68c73e958e5199d1932abb3a` | `tabdpt1_3.safetensors` |63,040,624|

The revision identifiers were resolved from the official model APIs during the same audit; initial header requests used `main`. For reproducing an entry, use a permanent source URL as `https://huggingface.co/REPOSITORY/resolve/REVISION/FILE`. Example: [TabFM1.1 classifier](https://huggingface.co/google/tabfm-1.1.0-pytorch/resolve/d7cfd3b8d184d7c8fed4210aeba9e1c967d45f2a/classification/model.safetensors).

**Counting precision:** stored tensor elements are directly observed. They are not universally identical to *trainable* parameter count: constant positional frequencies, buffers, shared tensors, or task-specific unused parameters require checking module registration and usage. In TabICL, the8RoPE frequencies are an `nn.Parameter` with `requires_grad=False`; the remaining inspected state tensors are weights, biases, inducing vectors and CLS parameters. This gives27,552,250/28,544,983 trainable elements for the v2 configuration, assuming no other freezing. Kumo has fixed RoPE parameters too; remove128 elements forSmall or384 forMedium/Large for that exclusion alone. Rounded totals are robust to these tiny differences. Parameter-count comparisons should state whether frozen tensors are included.

Sources checked for registration/configuration: [TabICL core](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/tabicl.py), [TabICL RoPE](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/rope.py), [NVIDIA RoPE](https://raw.githubusercontent.com/NVIDIA/structured-data-models/main/sdm/nn/rope.py), [learned numerical/categorical frequencies](https://raw.githubusercontent.com/NVIDIA/structured-data-models/main/sdm/models/tabfm/cell_embedding.py). These source URLs are moving branches; pin source commits as well before reproducing exact trainable totals.

### Kumo configurations from official source

`MODEL_KWARGS` in [official model.py](https://raw.githubusercontent.com/NVIDIA/structured-data-models/main/sdm/models/kumo/tabular/model.py):

| Setting | Small | Medium | Large |
|---|---:|---:|---:|
| Cell width |128|256|256|
| Interleaved embedding layers |4|6|6|
| Embedding heads |4|4|4|
| Inducing points |128|256|256|
| Readout tokens |4|4|4|
| ICL width |512|512|1024|
| ICL layers |12|24|24|
| ICL heads |8|8|16|
| Query-side KV heads |full/default|2|2|

All three use feature groups3 and32Fourier frequencies. Each task initialization loads its appropriate task checkpoint. Initializing both task models can retain both sets of weights; that is not a single joint checkpoint. This differs from TabPFN3.5's explicit joint model.

## What can actually be inferred about data volume

For TabICLv2, the disclosed schedule makes a useful unit check possible:

* Stage1:500,000×64×1,024 = **32.768B row presentations**.
* For a continuous log-uniform row count betweena andb, `E[R]=(b-a)/log(b/a)`.
* Stage2 expected rows:40,000×64×E[LogU(400,10240)] = **7.7686B**.
* Stage3 expected rows:10,000×64×E[LogU(400,60000)] = **7.6126B**.
* Total = **48.1492B expected row presentations per model**, not a published exact realized count. Rounding, clipping, actual random draws and failed/skipped batches can change it.
* Assuming the stated first-stage context fraction averages60%, and later stages use80%, expected supervised query targets are **16.1834B**. These predictions within an episode are dependent; they are not16B independent tasks.

This gives about1,748row presentations per classifier parameter, merely as a dimensional illustration. A 20:1 rule would not describe this published schedule. It also does **not** prove1,748is optimal: no fixed-compute size/token sweep establishes that.

The same schedule contains35.2M task instances. Its maximum100features is insufficient to recover an exact cell count because the realized feature-count distribution is required. Calling48.15B rows48.15B “tokens” is valid only if the report explicitly defines a row token and also logs the frontend's cell/group work.

Recommended logging ledger for every experiment:

1. Unique generator programs/mechanisms and random worlds; dataset instances; accepted/rejected generation attempts.
2. Presented support rows and query rows, raw cells, grouped-cell encoder positions, pooled ICL row positions.
3. Number of query target predictions, number of classes, prediction head type, and task loss normalization.
4. Per-shape stage counters; feature widths; actual attention-visible pairs; padding and discarded rows.
5. Optimizer updates; global and microbatch sizes; accumulated tasks/update; model-only FLOPs; end-to-end accelerator-hours; generator CPU-hours; data wait and communication fractions.

## How to choose size and exposure within 50,000 GPU-hours

### Statistical objective

Choose the parameter count, exposure, prior mixture, shape curriculum and inference policy that maximize **held-out real-tabular transfer** within the total budget. Low synthetic training loss alone is insufficient: increased capacity can model generator-specific artifacts or irrelevant complexity more accurately while worsening real transfer. Keep the classification and regression scorecards separately visible.

A useful fitted surrogate is

`L(P,D) = L_inf + A P^(-alpha) + B D^(-beta)`

only within a fixed tokenizer, prior, loss, shape curriculum and optimizer regime. For fixed-shape dense models, if measured compute were approximately`C=k P D`, the stationary solution obeys

`P*(C)=[alpha A/(beta B)]^(1/(alpha+beta)) (C/k)^(beta/(alpha+beta))`,

`D*(C)=[beta B/(alpha A)]^(1/(alpha+beta)) (C/k)^(alpha/(alpha+beta))`.

These expressions are derivations, not fitted results for this project. Even ifalpha≈beta, the data/parameter ratio depends onA,B,k and the data unit. There is no universal20coefficient. Full attention, changing sequence length, cell encoders, and data generation can violate the simpleC≈kPD model, so measured throughput/shape cost should govern the actual plan.

### Practical experiment recommendation

* **50M:** fast prior and implementation screening. Do not discard a promising complex prior solely because this capacity cannot learn it.
* **100M:** middle control to detect whether a benefit is small-model-specific.
* **315M:** selected dense reference, counted from the instantiated architecture. The small finance adapter and hurdle-head increment is separately recorded; the exact parameter count does not establish compute optimality.
* **500M:** serious challenger, justified by measured LimiX scaling through406M and publicTabFM1.1≈410M. It may win under the project's budget, but current evidence does not establish that.
* **1B:** proceed only if the500M iso-compute curves remain favorable on held-out real tasks, optimization is stable, and projected remaining training improves the final objective more than continued315–500M training or another seed. A big memory budget alone is not this evidence.

Compare sizes at **matched cumulative end-to-end GPU-hours** and also keep a matched-row/exposure view for interpretation. These answer different questions. At fixed rows a bigger model normally spends more compute; at fixed compute it sees fewer rows. Do not present the former as a compute-optimal result.

The selected 7,500-hour scaling allocation in `research/specs/prior_bank_spec.json` assigns 4,400 hours to the main fit, 2,400 hours to additional-seed confirmation of two finalists, and 700 hours to reserve. The listed comparison endpoints are 100/300/700 GPU-hours. The exact run schedule and continuation/cooldown policy must be recorded before comparisons: checkpoints evaluated before an appropriate learning-rate decay can give a misleading capacity ranking. Earlier equal-time trajectory sketches are not additional approved budget.

The project ledger assigns 5,000 hours to systems/reproduction, 10,000 to prior/architecture/optimizer screening, 7,500 to scaling/confirmation, 22,500 to final pretraining and 5,000 to locked evaluation. Do not convert those hours into promised rows or tokens before profiling. Finance generation, full candidate indexing and all routed context prefills count toward end-to-end cost. A few-million-row source table is not a few-million-row dense attention context. Reusing one finite world for multiple macroepisodes reduces independent world count and should be logged alongside the computational saving.

If separate classification/regression checkpoints are compared, their combined training and serving budget must be counted; no phase may silently receive its allocation twice. The selected shared-task model amortizes some capacity, but its gradient allocation and per-task quality remain empirical questions.

### Promotion criteria and uncertainty

Promote the smallest model whose late equal-compute real-data curve is within the chosen practical tolerance of the best, unless the larger model's projected continuation gain is credible and the user deliberately values that gain over serving cost. Because performance is the primary objective here,500M should be allowed to win;315M is the selected reference, not a hard ceiling.

Fit uncertainty across training seeds and dataset families. Keep a held-out compute point and preferably an intermediate capacity outside the scaling fit. A good fit to four observed checkpoints alone does not validate billion-parameter extrapolation. Refit after a material prior, tokenizer or context-curriculum change. Report the chosen size as the best tested candidate at the available budget, unless a broader optimum has actually been measured.

This audit inspected checkpoint sizes and source reports. CPU implementation tests elsewhere in the repository do not establish a fitted size optimum, MI355X throughput, or contemporary competitor accuracy. Promotion still requires the measured equal-budget comparisons described above.
