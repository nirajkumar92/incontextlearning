# Tabular foundation model research

This repository implements frozen-weight in-context learning for binary classification, multiclass classification and regression, including finance worlds with large finite histories, rare events and delayed labels. No competitive pretrained checkpoint is bundled.

**Start with the [main recipe](docs/main_recipe.md).** The recommended candidate is a **317,116,304-parameter dense model** with hash-bit nominal encoding, 999-quantile ordinary regression, logarithmic attention scaling and two query KV heads. Train a protected standard checkpoint using the pinned TabICLv2 graph prior with a small eligible P1 addition, then initialize a separate binary-fraud specialist using the new covariate-first risk-partition prior. Both remain frozen during downstream in-context prediction.

The provisional targets are **64 million accepted standard episodes** and **1.024 million finance macroepisodes**, with 9,000/3,000 GPU-hour ceilings inside the existing 50,000-hour program. The [volume plan](docs/prior_volume.md) translates measured stage costs into a fixed horizon. These are requested volumes, not demonstrated MI355X throughput or benchmark wins. The bundled candidate configs are bounded 250 GPU-hour pilots.

## Quick start

Use the cluster-compatible ROCm/PyTorch environment on GPU machines. For a CPU checkout, install the package and optional reference dependencies in a virtual environment:

```bash
python -m pip install -e '.[test,reference]'
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m tabular_foundation.train \
  --config configs/smoke_reference.json --output runs/reference_smoke
python -m tabular_foundation.train \
  --config configs/smoke_finance.json --output runs/finance_smoke
```

Use new output directories. `--resume runs/reference_smoke/last.pt` resumes the exact original configuration; `--initialize-from parent/last.pt` starts a compatible new experiment with fresh optimizer/schedule state and recorded parent provenance. Smoke configs use small models and diagnostic populations; they are integration checks.

See [training and cluster setup](docs/training.md) before using `configs/candidate_standard.json`, `configs/candidate_fraud.json` or the eight-node launcher. Reference generation requires `PYTHONHASHSEED=0` at interpreter startup. One ordered CPU producer per rank and bounded process prefetch are implemented; pinned transfers and measured GPU utilization are not.

## What is implemented

- Pinned, manifest-verified TabICLv2 graph-prior generation and native filtering; conditional P1/P4 grafts and explicit wider/higher-class envelope controls.
- The authored P0 mechanism bank, P1 categorical hierarchy and P4 count/amount laws as separate controls.
- Finite finance worlds with exact counts, keyed replay, selective label availability, bounded context retrieval and correctly weighted future queries. Full-H, hidden-H, H-only and weak/nondiscriminative-H controls distinguish shortcut information from learning under selection.
- Dense cell/row compression and support-only ICL with exact caches; hash-bit encoding, quantile or histogram heads, exact hurdle atom, positive-support severity scale, log-length scaling and query KV compression.
- Muon/AdamW training, ordered CPU task production, distributed macroepisode reduction, checkpoint/resume, explicit specialist initialization and compute/source-count accounting.
- Standard and finance prediction, NPZ/CSV interchange, chronological validation, metrics, latency profiling, tuned tree baselines and an optional TabArena adapter.

The main finance continuation is **binary only**. Optional multiclass and realized-loss heads are separate experiments. Quantile outputs do not define a trained density NLL. The native prior adapter preserves upstream behavior but does not reproduce the full released model or its complete training system. See the [upstream code audit](research/reviews/upstream_code_audit.md) for that distinction and the nominal-feature coverage limit.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/tabular_foundation/` | Generators, model, training, prediction and evaluation |
| `configs/`, `scripts/`, `tests/` | Runnable configurations, tools and integration checks |
| `docs/` | Main recipe, operations, data contracts and audits |
| `third_party/tabicl_reference/` | Pinned upstream prior source, scripts, license and manifest |
| `research/specs/`, `research/reviews/` | Scientific specifications and source-based analysis |
| `research/probes/`, `research/results/` | Mathematical diagnostics and recorded evidence |
| `papers/` | Supplied and supplementary primary literature |
| `runs/` | Generated local outputs |

Read the [implementation map](docs/implementation.md), [reference prior controls](docs/reference_prior.md), [data/evaluation guide](docs/data_and_evaluation.md), [TabArena integration](docs/tabarena.md) and [audit](docs/audit.md) for the corresponding interfaces and limits. The detailed research report is [prior_bank_assessment.tex](prior_bank_assessment.tex), with a saved [PDF](prior_bank_assessment.pdf); keep the PDF synchronized when revising the source. The report introduces the Bayesian learning objective with a worked example, walks through R/P1 and finance generation, and illustrates the prior, model and training sequence in five diagrams. The methods explain each component before giving its implementation details and distinguish the selected recipe from controls and optional experiments. Missing-value encoding and current training coverage are explicit. The built-in LaTeX editor provides compilation and preview.

Local tests and tiny training runs establish probability, numerical and integration contracts. They do not establish superiority over released TFMs or full-history trees, realistic fraud precision, or an optimal model size. The main recipe makes the next training choice concrete while keeping measured hardware cost and held-out standard/finance performance as separate evidence requirements. Historical results remain in `research/results/`; current source changes do not rewrite their meaning.
