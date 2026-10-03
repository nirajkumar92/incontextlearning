# Provisional sample volumes for the final candidate

Plan for **64,000,000 accepted standard macroepisodes** and **1,024,000 finance
macroepisodes**. A macroepisode is a generated tabular prediction task, not one
row, cell, query target, route or rejected generator attempt. These are provisional
exposure targets. The fixed **9,000 standard + 3,000 finance GPU-hour ceiling**
takes precedence if measured costs cannot fit them. This preserves the conditional
12,000-hour final allocation inside the unchanged 50,000-hour research program.
No MI355X throughput measurement currently establishes that these targets fit.

The standard candidate is `R_P1_05`: unchanged pinned reference mechanisms with
an eligible 5% P1 branch, plus an explicitly labeled shape/class extension in
20% of stage-two and stage-three slots. The finance continuation is a separate
binary fraud population at prevalence 1e-4. P4 receives **zero episodes in this
provisional main candidate**; its separately funded prior-control screen is not
part of this final-run volume.

## Standard tasks

At global batch 256, 64 million episodes mean **250,000 optimizer updates**.
The proposed stage proportions are **90/9/1% of accepted episodes**, giving:

| Stage | Updates | Accepted episodes | Expected R branch before eligibility return | Expected P1 branch before eligibility return | Expected envelope-extension slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 225,000 | 57,600,000 | 54,720,000 | 2,880,000 | 0 |
| 2 | 22,500 | 5,760,000 | 5,472,000 | 288,000 | 1,152,000 |
| 3 | 2,500 | 640,000 | 608,000 | 32,000 | 128,000 |
| Total | 250,000 | 64,000,000 | 60,800,000 | 3,200,000 | 1,280,000 |

The R/P1 figures are branch **expectations**, not fixed quotas or claimed observed
counts. P1 requires the preselected requested feature count to be at least eight;
ineligible P1 selections return to R. If `E_s` is the number of eligible slots in
stage `s`, the exact conditional expectation is:

`E[accepted P1 | E_1,E_2,E_3] = 0.05 × (E_1 + E_2 + E_3)`

`E[accepted R | E_1,E_2,E_3] = 64,000,000 − E[accepted P1 | E_1,E_2,E_3]`.

The known requested-width law also gives a more useful unconditional expectation
after eligibility. For rounded Uniform(1,Fmax), the probability of F≥8 is
`(Fmax−7.5)/(Fmax−1)`. Applying the 20% extensions and integrating stage three's
integer-truncated log-uniform lengths across native width caps gives:

| Stage | P1 eligibility probability | Expected accepted R | Expected accepted P1 |
| --- | ---: | ---: | ---: |
| 1 | 0.93434343 | 54,909,090.91 | 2,690,909.09 |
| 2 | 0.94237671 | 5,488,595.51 | 271,404.49 |
| 3 | 0.91245776 | 610,801.35 | 29,198.65 |
| Total | — | **61,008,487.77** | **2,991,512.23** |

Thus the provisional main run expects about **61.008 million R and 2.992 million
P1 accepted tasks**. These calculations assume the specified native shape laws
and envelopes, no diagnostic shape overrides, and no aborted, skipped or replayed
attempts. The planner emits both these eligibility-adjusted expectations and the
earlier branch-draw expectations; their denominators are different.

The realized source counts are random integers and must come from the generation
audit. A 3.2-million P1 expectation before eligibility is not a hard upper bound
on its random realized count. Native R may require multiple rejected graphs or
tables per accepted slot; those retries consume generation time but are not
additional trained tasks or independent accepted episodes. A failure aborts the
draw instead of switching its source. Actual accepted, failed and skipped-update
counts remain necessary even when a target horizon is fixed.

The eight native R functions are mechanisms **inside compositional graphs**,
not eight mutually exclusive table families. A table can contain several types
at different nodes, and their choices follow the pinned `_function.py` and
sampler laws. Do not divide the reference-task total by eight or claim an equal per-function
quota. P1 eligible replacements are a separate table-source branch.

These targets do not require pre-generating or storing a 64-million-table bank.
Generate training episodes on demand with bounded prefetch and the declared
finance-world reuse. Keep fixed validation-world seeds disjoint from all training
seed namespaces, and reuse those fixed worlds only for validation.

Classification and regression packages are sampled 50/50, so their expected
counts are **32,000,000 each**. In the unchanged native classifier, class budgets
are uniform from 2 to 10: binary gets 1/9 of classifier slots and multiclass gets
8/9. This is not equal thirds across binary, multiclass and regression. With no
envelope extension, the corresponding expectations would be about 3.556 million
binary, 28.444 million multiclass and 32 million regression episodes.

The planned extension changes that calculation: stage two extends 20% of slots
to class budget at most 32 and feature ceiling 256; stage three extends 20% to
class budget at most 256 and feature ceiling 512. For a classifier extension
ceiling `C`, binary probability is `1/(C−1)`. Combining the stage-specific laws
gives about **3.503 million binary**, **28.497 million multiclass**, and **32 million
regression** episodes in expectation. These task labels describe the ex ante
class budget; the reference filter can produce fewer observed classes.

The **1,280,000 extension slots overlap R/P1**. They are not extra episodes or a
third source to add to 64 million. Before eligibility, the expected P1/extension
intersection is 64,000 slots. Native long-context feature ceilings still apply;
these counts do not guarantee wide or many-observed-class coverage. Measure
requested and accepted shapes/classes separately.

This 90/9/1 episode plan is distinct from the older `scripts/plan_training.py`
policy, which assigns 60/25/15% of usable **GPU-hours** to stages. Neither policy
can be substituted for the other without changing the stage exposure. Existing
candidate configuration files remain bounded pilots; this volume document is
not evidence that a full final run has been configured or launched.

## Finance tasks and finite worlds

At batch 256, **1,024,000 macroepisodes mean 4,000 updates**. Each finite world is
used once per update over four consecutive updates with fresh query sampling.
When complete reuse groups and stage boundaries align, this means **256,000
distinct stage-specific worlds**, not 1,024,000 independent worlds.
This exact count assumes no skipped optimizer attempts or replayed work. Runtime
world reuse is keyed by **attempt**, while stage changes follow successful
optimizer updates; a nonfinite skipped update can shift a boundary within a
four-use group. An exact resume may reconstruct cached worlds, and replay from
an older checkpoint consumes some presentations again. Count those effects from
the actual attempt/stage history rather than declaring every four recorded
macroepisodes an independent completed world.

For planning, use the same provisional 90/9/1 stage proportions:

| Stage | Updates | Finance macroepisodes | Distinct worlds with four completed uses |
| --- | ---: | ---: | ---: |
| 1 | 3,600 | 921,600 | 230,400 |
| 2 | 360 | 92,160 | 23,040 |
| 3 | 40 | 10,240 | 2,560 |
| Total | 4,000 | 1,024,000 | 256,000 |

A world can represent a history with a few million rows at natural fraud
prevalence 1e-4, generated lazily. Its entire history is not automatically
materialized, put in context, or counted as trained rows. A three-million-row
history has 300 frauds in expectation before availability/selection effects;
finite realizations vary. Balanced query proposals retain population importance
weights and do not turn this into a naturally balanced fraud population. Multiple
context routes for a query are also not independent macroepisodes.

## What the targets require from 64 GPUs

These are **required rates**, calculated from the ceilings; they are not predicted
or measured hardware performance. Before any setup/checkpoint/other overhead:

| Phase | GPU-hour cap | Wall hours on 64 GPUs | Required accepted macroepisodes/s across cluster | Required macroepisodes/GPU/s | Maximum stage-weighted mean seconds/update |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 9,000 | 140.625 | 126.419753 | 1.975309 | 2.025000 |
| Finance | 3,000 | 46.875 | 6.068148 | 0.094815 | 42.187500 |

For measured stage update times `t1,t2,t3`, the relevant weighted mean is
`0.90*t1 + 0.09*t2 + 0.01*t3`. The standard target fits only if its complete
distributed update average fits 2.025 seconds before overhead; finance must fit
42.1875 seconds. Reserving overhead makes those thresholds smaller. CPU smoke
timings and literature H100 training costs cannot establish MI355X feasibility.

The measurement must cover the actual architecture, task and envelope mixtures,
global batch 256, generator/rejection cost, codec, transfers, forward/backward,
optimizer and synchronization, including slow-rank effects. Finance must cover
complete four-use cycles, cold-world construction/selection, all route prefills
and warm uses. Reserve setup, checkpoints and other charged work explicitly and
use conservative timing summaries. A phase cap remains the runtime hard limit
even when a timing projection says the target should fit.

## Reproduce or replace the planning artifact

The checked-in [plan](../research/results/prior_volume_plan.json) has no measured
timings. It deliberately leaves feasibility and the adopted budget-feasible
horizon null. Regenerate it with:

```sh
.venv/bin/python scripts/plan_prior_volume.py \
  --output research/results/prior_volume_plan.json
```

Once timings exist, supply a JSON measurement file with:

```json
{
  "allocated_gpus": 64,
  "global_batch": 256,
  "measurement_id": "replace with the actual pilot run identifier",
  "standard_seconds_per_update": [3.0, 10.0, 50.0],
  "finance_seconds_per_update": [40.0, 100.0, 300.0],
  "standard_overhead_gpu_hours": 100,
  "finance_overhead_gpu_hours": 100
}
```

Those numbers are an **arithmetic example, not measurements**. Replace them with
the actual pilot evidence before using the result:

```sh
.venv/bin/python scripts/plan_prior_volume.py \
  --measurements runs/actual_stage_timings.json \
  --output runs/measured_prior_volume_plan.json
```

For each measured phase, the planner reports the requested horizon's required
GPU-hours and the largest smaller horizon that fits the reserved budget under
those timings. It shrinks all three stages together: standard uses **100-update
blocks** to retain exact 90/9/1 proportions; finance uses **400-update blocks** to
retain the proportions and align every boundary to four-update world reuse.
Rounding the total episode count to batches alone would not preserve both rules.
If no complete block fits, it returns zero with an explicit status. It never
recommends overrunning the cap or silently extending a horizon when costs are low.

The output includes the adopted update counts and cumulative `stage_end_steps`
for a later reviewed training configuration. It is a planning artifact, not a
launch config. Freeze the feasible horizon and corresponding learning-rate
schedule before launch; stopping a larger cosine schedule at a budget boundary
does not reproduce the smaller planned run. Standard/finance caps are separate;
unused hours are not automatically transferred between them.

The scale rationale is modest. TabICLv2's released batch 64 and 500,000/40,000/
10,000 stage updates imply **35.2 million table exposures per separate model**,
or 70.4 million for its classifier/regressor pair. [Official released recipe](https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/scripts/train_v2_clf_stage1.sh).
Kumo reports about **35/71/137 million tables** for its Small/Medium/Large models.
[NVIDIA release](https://huggingface.co/blog/nvidia/kumo-tabular).
These put 64 million in a plausible exposure order of magnitude. They do not
establish a scaling law, matching compute, equal statistical independence, an
optimal horizon, or expected accuracy. Measured cost and validation determine
whether the provisional targets survive.
