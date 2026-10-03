# Research specifications and evidence

The full explanation is in the root [research report](../prior_bank_assessment.tex), with a [compiled PDF](../prior_bank_assessment.pdf). Use the root [README](../README.md) to run the code.

| Directory | Contents |
|---|---|
| `specs/` | Selected scientific recipes and experiment matrix |
| `reviews/` | Source-level literature, model-size and architecture reviews |
| `probes/` | Runnable mathematical checks and compute projections |
| `results/` | Executed checks, environment records, audit results and file provenance |

Start with [the main recipe](../docs/main_recipe.md) and [per-prior volumes](../docs/prior_volume.md), then [specs/prior_bank_spec.json](specs/prior_bank_spec.json). It coordinates the standard prior, finance prior, architecture and optional latency/MoE experiments. These research JSON files describe the scientific design; use `../configs/` for actual trainer inputs. `specs/ablation_plan.csv` lists the planned comparisons.

The review notes now cover the selected standard-tabular and finance objectives. The report and selected specifications define the current recipe. The finance target is one event per 10,000 transactions. Optional architecture branches are distinguished from implemented code.

Source PDFs, extracted text and paper manifests live in [../papers/](../papers/README.md). `results/code_validation.json` preserves the original implementation test record and source fingerprints from October 2. `results/repository_scope_audit.json` records the scope review, file decisions and earlier relocation/deletion history. Historical execution paths and hashes are retained in the evidence summaries. Raw tiny checkpoints, example NPZs, test XML and compiler logs have been removed; the configs/scripts recreate them.

The [independent source-first audit](reviews/independent_audit.md) records the rejected baseline and the [upstream code audit](reviews/upstream_code_audit.md) records what competitors actually released. New representation and learned-head probes show how the candidate addresses specific defects. Their results remain local checks, not competitive evaluation. The coordinating spec’s `main_candidate` block defines the recommendation; older v3 specifications are comparison controls. The 50,000-hour program is unchanged; optional MoE has no current allocation.
