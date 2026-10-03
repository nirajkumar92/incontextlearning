# Main training recipe

Use the **317,116,304-parameter dense candidate** for a protected standard checkpoint, then initialize a separate binary-fraud specialist from it. Both perform frozen-weight in-context prediction at deployment. The standard target is **64 million accepted episodes**; the finance target is **1.024 million macroepisodes**. Their final-training ceilings are **9,000 and 3,000 allocated GPU-hours**, respectively, within the existing 50,000-hour research program. These are concrete targets and budget limits, not measured MI355X throughput or established accuracy wins.

The runnable starting configurations are [`candidate_standard.json`](../configs/candidate_standard.json) and [`candidate_fraud.json`](../configs/candidate_fraud.json). They are bounded **250 GPU-hour pilots**, with 1,000 and 1,200 updates, not the full-volume runs. Use the [volume plan](prior_volume.md) to set a measured full-run horizon; the older `plan_training.py` allocates stage hours and does not implement this recipe's episode proportions.

## Model and optimizer

| Component | Candidate choice |
| --- | --- |
| Backbone | Dense row-compressed model, `model="base"`; 24 trunk layers, width 1,024 |
| Nominal features | `categorical_encoding="hash_bits"` |
| Ordinary regression | 999 quantiles; mean pinball training loss in support-standardized units |
| Optional realized-loss regression | Exact zero atom plus 1,025-bin/Lomax positive-truncated density |
| Severity scale | `hurdle_target_scale="positive_support"`; observed positive support when available, otherwise reference-support fallback |
| Attention | Positive logarithmic key-count scaling; two query KV heads, full support KV heads |
| Optimization | Muon on registered hidden matrices, AdamW elsewhere; weight decay zero; global gradient clip 10 |
| Standard peak learning rates | Muon 8e-4; AdamW 3e-4 |
| Fraud continuation peak learning rates | Muon 8e-5; AdamW 3e-5 |
| Schedule | 2% warmup, then cosine to one tenth of peak; one fixed horizon per run |

The finance adapter and optional hurdle head are present in both checkpoints so initialization has identical parameter shapes. **The main fraud continuation is binary classification**; it does not train severity or multiclass fraud heads. These remain separate experiments. Quantile forecasts do not provide a trained density NLL, and averaging 999 quantiles is an approximation to the conditional mean. Positive-support severity scaling is an affine parameterization, not a log1p target transformation or an estimate of population moments.

The candidate addresses demonstrated representation and resolution weaknesses. Query KV compression still needs ROCm timing, and the architecture is not a reproduction of the full official TabICLv2 model. See the [implementation map](implementation.md) and [upstream code audit](../research/reviews/upstream_code_audit.md).

## Standard prior and volume

Use `standard_prior="R_P1_05"`, `reference_task="mixed"`, `finance_share=0`. The reference generator is the pinned, unmodified TabICLv2 graph prior, including its filters. An independent branch chooses authored P1 with probability .05 when its preselected task and width are eligible; otherwise the draw stays in R. This is **conditional five-percent addition**, not exactly 3.2 million P1 tables. Integrating the configured width laws gives approximately **61.008 million R and 2.992 million P1 tasks** in expectation; realized counts still fluctuate. Classification and regression package draws are 50/50 in expectation. Binary and multiclass shares follow the sampled class budget; they are not one third each.

Allocate accepted episodes **90/9/1% across stages**, giving:

| Stage | Accepted-episode target | Updates at global batch 256 | Native row range |
| --- | ---: | ---: | --- |
| 1 | 57,600,000 | 225,000 | 1,024 |
| 2 | 5,760,000 | 22,500 | 400–10,240 |
| 3 | 640,000 | 2,500 | 400–60,000 |

On 20% of stage-two draws, extend the requested feature/class ceilings to 256/32; on 20% of stage-three draws, extend them to 512/256. Other draws retain the native ceilings. This explicitly changes the native shape/class law while preserving its graph mechanisms and filtering. Native long-row width caps, constant-column removal and realized class collapse still apply. A requested ceiling is not guaranteed training exposure: inspect consumed class counts and generator audits of requested/accepted widths.

The native generator emits a numeric representation and discards category-type identities. The nominal hash encoder therefore gets direct nominal training chiefly through P1 and finance, rather than all native categorical mechanisms. This coverage limit is documented in [reference prior controls](reference_prior.md); do not infer nominal types from observed cardinality to hide it.

## Binary-fraud specialist

Set `finance_share=1`, `finance_task="binary"`, with:

```json
{
  "mechanism_family": "risk_partition",
  "feature_view": "hide_h",
  "loss_normalization": "unit",
  "prevalence": 0.0001
}
```

Historical populations remain 300,000/1,000,000/3,000,000 rows at probabilities .2/.3/.5, with future size one third of history. The covariate-first generator allocates finite daily risk-cell counts, calibrates the expected historical rate, and samples binomial event counts. Actual event counts fluctuate; zero-positive worlds remain. Covariate thresholds, sparse main effects and an interaction determine risk. This is not a continuous GAM, entity-specific hazard model or unseen-attack generator.

H is hidden from **both model and selector**. It can still affect which labels are observed through the declared historical review policy. Eligible labels must be revealed by the cutoff; unresolved outcomes never become negatives. The reservoir, observed-positive sample and nearest-to-center local sets preserve their declared finite sampling laws. Query proposals use the actual finite future class fractions for loss-only importance weights, then route by observable features.

At full target volume, 4,000 updates emit 1,024,000 macroepisodes. Four uses per world give **256,000 distinct world seeds** when the schedule completes without skipped updates or stage-crossing reuse groups. Stage updates are 3,600/360/40; cumulative boundaries are 3,600/3,960/4,000. Each use draws 128 queries, but repeated finite IDs do not add independent fraud cases. Count all routed contexts and all candidate searches in the cost.

Keep the full-H, H-only, hidden-H and weak/nondiscriminative-H laws as explicit controls. View and normalization controls preserve the same population and query draws. Changing H's generative probabilities changes the observation law and is a separate experiment.

## Why the specialist is the default

Balanced query proposals improve sampling efficiency while retaining population weights. At prevalence pi and calibrated constant prediction p=pi, each weighted binary-logit gradient has magnitude `2*pi*(1-pi)`: about .0002 at one in 10,000, versus .5 for ordinary balanced binary CE. An emitted 20% finance share therefore does not imply a comparable shared-gradient contribution. Missing-reference worlds may initially have large gradients that primarily learn the base rate.

The protected standard model plus specialist avoids making the ordinary and rare objectives compete in every update. Unit normalization retains the population objective within the specialist. The old 80/20 joint profile remains a controlled challenger; entropy normalization also remains available and never replaces correct proposal weights. Neither a large loss multiplier nor optimizer normalization proves successful sharing.

## Run sequence

Install the optional reference dependencies, then run the small reference smoke before the base-model pilots:

```bash
python -m pip install -e '.[test,reference]'
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tabular_foundation.train \
  --config configs/smoke_reference.json --output runs/reference_smoke
```

Profile complete updates at all three stages, then run:

```bash
python scripts/plan_prior_volume.py --output runs/provisional_volume.json
python scripts/plan_prior_volume.py \
  --measurements measured_stages.json --output runs/measured_volume.json
```

The unmeasured plan only reports required throughput. At 64 GPUs, before setup/checkpoint reserves, the targets require **126.42 standard episodes/s** and **6.068 finance macroepisodes/s** across the allocation. Their corresponding mean update ceilings are **2.025 s** and **42.1875 s**. Use the measured plan to prepare fixed-horizon copies of the candidate configurations; do not assume the target fits because a JSON plan can be written. Preserve stage proportions when reducing volume: standard horizons use 100-update blocks and finance horizons use 400-update blocks.

Launch the standard run first. After its checkpoint is selected, launch the fraud run in a new output directory with the same model configuration. The following shows one-node syntax; use allocation-matched measurements, or the eight-node wrapper in the training guide for the 64-GPU plan:

```bash
PYTHONHASHSEED=0 torchrun --standalone --nproc_per_node=8 \
  -m tabular_foundation.train --config configs/main_standard_measured.json \
  --output runs/main_standard

PYTHONHASHSEED=0 torchrun --standalone --nproc_per_node=8 \
  -m tabular_foundation.train --config configs/main_fraud_measured.json \
  --initialize-from runs/main_standard/last.pt --output runs/main_fraud
```

The measured config filenames above are outputs you prepare, not bundled final configurations. `--initialize-from` loads parent weights with fresh optimizer and schedule state and records the parent checkpoint SHA-256. It does not charge past parent hours a second time. `--resume` instead requires the exact original configuration and output directory. Use the [cluster instructions](training.md) for eight-node launch and continuation arguments.

## What decides success

Evaluate frozen checkpoints under fixed support, preprocessing, estimator and latency budgets. Standard binary, multiclass and regression results require separate reporting; compare against released models through their actual supported protocols. The pinned prior adapter alone does not reproduce an official model or establish a TabArena result.

Finance evaluation uses chronological cutoffs, revealed history, natural future prevalence and full-history tuned LightGBM/CatBoost baselines alongside frozen TFMs and bounded-context controls. Report AP, review-budget precision/recall, probability scores, actual positives and time/entity uncertainty. Synthetic law tests establish correct sampling, not production fraud performance. No MI355X calibration or competitive pretrained checkpoint is supplied; the candidate is the recommended next recipe, with hardware-calibrated volume still to be set.
