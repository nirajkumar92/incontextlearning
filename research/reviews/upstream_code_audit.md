# Public source audit, 3 October 2026

The audit inspected complete default-branch GitHub trees and selected full implementations, then traced Mitra-v2 to its separate Hugging Face package. `research/results/upstream_repository_audit.json` records the pins, file inventories and source hashes. Absence below means absent from these identified public releases, not from every possible private or future repository.

| Release | Pinned revision | Public implementation | Complete synthetic pretraining? |
|---|---|---|---|
| PriorLabs/TabPFN | `15f5e6b2b629b905879b9be907261416f20d0df5` | Current architecture, codec, cache, inference and downstream fine-tuning | No current3.5 generator and full pretraining entry point found |
| NVIDIA/structured-data-models | `b982fbf4188ea8db17249fe81338723106100b63` | Kumo architecture, processing, ensemble and evaluation | No; official release says recipe and generators will follow |
| xiaomi-research/xiaomi-tabldm | `c87efddedb244d1a21ac07b40b1453466329e5c5` | Model, MoE components, cache and sklearn inference | No generator or pretraining launcher found |
| autogluon/autogluon, Mitra | `88d33ba6d97ec07ef232e5b28213a61f272a8125` | Tab2D, estimator, downstream trainer, configuration schema | A pretrain dataclass is present; the actual generator/filled recipe is not |
| autogluon/mitra-finetune | `b4701e8148dc33b00ed15d7086ff59816957cde4` | V2 fine-tuning/evaluation code and results | No; report explicitly excludes synthetic pretraining |
| soda-inria/TabICL | `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3` | Prior source, training scripts, trainer and configuration | Usable open reference; our separate model adapter is not a full reproduction |

## PriorLabs

The [pinned3.5 architecture](https://github.com/PriorLabs/TabPFN/blob/15f5e6b2b629b905879b9be907261416f20d0df5/src/tabpfn/architectures/tabpfn_v3_5.py) exposes width128 distribution embeddings, eight row summaries,24 ICL layers,16 query heads and a many-class retrieval decoder. Its default uses only one KV head for queries while preserving full-head support attention. This motivated a concrete cache experiment. The tree's `finetuning` package concerns downstream adaptation. Historical PFN training code cannot be labeled the released3.5 prior. The [README](https://github.com/PriorLabs/TabPFN/blob/15f5e6b2b629b905879b9be907261416f20d0df5/README.md) identifies the core local inference role.

## Kumo: the recipe file is preprocessing

[`recipe.py`](https://github.com/NVIDIA/structured-data-models/blob/b982fbf4188ea8db17249fe81338723106100b63/sdm/models/kumo/tabular/recipe.py) constructs FP64 standardization, numerical transformations, categorical alignment/count features, feature permutations/selection, target transforms and ensemble reduction. It is not a synthetic generator. The [official announcement](https://huggingface.co/blog/nvidia/kumo-tabular) places the training recipe and artificial generators in a future release.

[`icl.py`](https://github.com/NVIDIA/structured-data-models/blob/b982fbf4188ea8db17249fe81338723106100b63/sdm/models/kumo/tabular/icl.py) uses full support attention and a KV prefix for query attention. Medium and large models retain two query KV heads; small retains full heads. This is the basis for the candidate's two-head option, which does not itself establish a latency gain on MI355X. GPU-native preprocessing is another substantive system component; our CPU codec remains a throughput risk.

## Xiaomi: useful MoE components, incomplete pretraining release

[`learning.py`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/c87efddedb244d1a21ac07b40b1453466329e5c5/tabldm/_model/learning.py) implements in-context prediction, many-class and cache paths. [`moe.py`](https://github.com/xiaomi-research/xiaomi-tabldm/blob/c87efddedb244d1a21ac07b40b1453466329e5c5/tabldm/_model/moe.py) supplies routed/shared experts, auxiliary statistics, bias balancing and dense initialization. These are not an optimizer loop, exact prior law or complete training configuration. Constructor options must not all be attributed to the checkpoint that produced a published score.

The implementation loops over experts using indexing and accumulation. Active parameter count and a system-level speed chart do not isolate MoE's contribution. For this project, query-cache compression has stronger direct implementation support as the first latency change. MoE remains a conditional experiment, not a presumed acceleration.

## Mitra: schema versus executable pretraining

[`config_pretrain.py`](https://github.com/autogluon/autogluon/blob/88d33ba6d97ec07ef232e5b28213a61f272a8125/tabular/src/autogluon/tabular/models/mitra/_internal/config/config_pretrain.py) defines fields for generators, optimizer and devices. The adjoining `tab2d.yaml` holds fine-tuning settings. Neither supplies the missing synthetic generators or a complete run. The [v2 package](https://huggingface.co/autogluon/mitra-finetune/tree/b4701e8148dc33b00ed15d7086ff59816957cde4) wraps AutoGluon adaptation: `api.py` specifies bagged fits; `modes.py` sets50 steps, learning rate1e-5, warmup10 and decay.3, explicitly distinguishing stock AutoGluon defaults. `speed.py` optimizes that loop; random arrays in its memory preflight are capacity checks, not prior pretraining. The supplied report Appendix A, physical p.27, explicitly excludes synthetic generators and pretraining configurations. Use this code for a correctly labeled adapted comparator.

## Consequence for our bank

The unmodified TabICLv2 prior is now vendored with its BSD license, source names, Git blobs and SHA256 checks. `reference_prior.py` invokes the actual graph sampler and filter, preserving full-generated-table normalization, split repair and selection. Native categorical columns arrive as encoded numerical tensors without categorical flags, so native R does not exercise our explicit nominal encoder as P1 does. The adapter is not a reproduction of native model preprocessing.

R/P1 and R/P4 controls draw task and shape before selecting the generator, retain native filtering only on R and return ineligible graft mass to R. Width/class envelope changes are labeled. One64-class diagnostic realized25 classes after12 raw attempts, showing why requested caps cannot stand in for observed coverage. The source-backed candidate, volume targets and local checks are documented in `docs/main_recipe.md` and `docs/prior_volume.md`. Source availability and unit tests do not establish competitive accuracy.
