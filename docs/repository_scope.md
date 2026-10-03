# Repository scope and file decisions

The repository serves two objectives: frozen-weight ICL for standard binary/multiclass classification and regression benchmarks, and financial classification/regression on large tables with fraud around one in 10,000, missing values, bursts and delayed labels. Predictive performance takes priority; GPU batch latency remains a design criterion.

## Kept

| Material | Why it belongs |
|---|---|
| `src/`, `tests/`, `configs/`, training/evaluation scripts | Implement and verify the selected standard/finance workflow; no abandoned enterprise generator or multi-table model remains. |
| Report source and compiled PDF | The complete research/training recipe and readable assessment. |
| Research specifications and ablation matrix | Current `main_candidate` contract plus labeled authored controls and the costed experiment program. |
| Pinned `third_party/tabicl_reference/` subset | Actual executable reference generator, license, released scripts and file hashes; needed for reproducible R/P1 comparisons. |
| MoE, context and optimizer research notes/probes | Directly address the requested accuracy/latency and training questions; optional unimplemented arms are explicitly labeled. |
| Nineteen source PDFs, extracted text and source manifest | Original evidence for priors, incumbent models, scale and robustness. PDFs are unchanged. Broad papers are retained when their tabular results are relevant. |
| Compact mathematical and execution results | Preserve the measured checks, configurations, source fingerprints and limitations without retaining disposable model/data outputs. |
| `local_environment.txt` | The exact local audit dependency freeze; it is evidence, not a ROCm install lock. |
| `.venv/` | Working local Python environment, ignored by Git; third-party dependencies are not project source. |

Customer IDs, category hierarchies, campaigns, timestamps, missingness and label delays remain. They are useful single-table statistical mechanisms for the current tasks. SCM DAGs model feature/target generation; nearest-neighbor selection supports bounded ICL contexts. Neither requires a relational database prediction objective. A paper mentioning another model family is source evidence, not an instruction to implement it.

## Removed

- Abandoned native relational, text and general enterprise-prediction proposals from the mixed-purpose reviews. Useful tabular evidence was preserved in those files.
- Completed three/four-update tiny-model checkpoints, logs, generated NPZ fixtures, predictions and baseline smoke output files. Their results/configurations remain in the compact execution record; the existing scripts regenerate fresh diagnostics.
- The unused Xiaomi page screenshot and duplicate first-page paper inventory. The original PDF, searchable extraction and fingerprinted source manifest provide the evidence.
- Raw compiler logs, redundant test XML, Python/pytest caches and Finder metadata.
- Separate administrative cleanup/organization manifests, consolidated into one scope audit.

This cleanup removed only verified completed tiny diagnostics. Future substantive training checkpoints and active run directories require their own retention decision.

## Modified

Four source reviews now cover the actual objectives: prior literature, competitor priors, TabPFN releases and model scaling. Current documentation, scope descriptions and output-directory instructions match the retained files.

The scan also found three stale contracts. Specifications and report now describe the implemented manual replicated gradient reduction, distinguish nonempty standard sampling from the model's empty-support boundary, and mark the general hybrid-context experiment as deferred with zero current allocation. The 50,000 GPU-hour budget and implemented scientific mechanisms did not change.

The report was subsequently reorganized into a continuous research narrative, from previous work and hypotheses through priors, architecture, training, evaluation and the code map. Repeated codec/head descriptions were consolidated, the selected finance hurdle head is consistently identified as implemented, and an unused difficulty-filter proposal was removed from the baseline instructions. All 51 cited sources and the detailed generator specifications were retained. The handoff audit also distinguishes the current fail-fast generator from the proposed eight-attempt queue, clarifies base-model versus finance parameter totals, and maps Fourier encoding to its actual model module.

`scripts/build_report.sh` exports the PDF without leaving compiler logs in the repository. `.gitignore` excludes generated outputs while retaining `runs/README.md` as a reproducibility guide.

## Audit record and future additions

[The file-by-file audit](../research/results/repository_scope_audit.json) records keep/modify/remove decisions, reasons, file hashes, verification results and earlier relocation/deletion history. [The original execution record](../research/results/code_validation.json) preserves the historical 109-test result and tiny-run summaries; its original paths/hashes describe that execution, even where raw artifacts were later removed.

Keep a new file when it implements the selected workflow, supplies relevant source evidence, documents a live design choice, or records a material experiment that cannot be replaced by a trivial regeneration. Put generated data/checkpoints under `runs/`; retain substantive experiment evidence deliberately. Do not reintroduce abandoned task branches as inactive files or stale recommendations.

## Reference-derived candidate follow-up

The new executable adapter, producer, model/head options, candidate configurations and volume planner belong to the selected workflow. `docs/main_recipe.md` and `docs/prior_volume.md` give the direct run and count instructions. Compact repository-audit, representation, learned-head and sample-volume results preserve their evidence. Full temporary competitor checkouts and tiny candidate checkpoints are kept outside the repository; only the source subset actually used at runtime is vendored. Earlier cleanup and test counts above remain historical records.

## Historical explanatory report revision

That revision added a Bayesian worked example, code-grounded descriptions of R/P1 and finite finance generation, and five inline diagrams covering data generation, the model and the two training runs. Its methods explained each component through motivation, mechanism and implementation, with the then-selected recipe distinguished from controls and future experiments. The full prose rewrite preserved every original formula, numerical setting, table, diagram, code listing, citation and source locator; independent reviews checked scientific meaning. Missing-feature support and the standard-prior coverage gap were made explicit. An introductory subsection explained categorical hash-bit inputs, quantile regression and query-only KV sharing with examples and references to their detailed implementations. The synchronized 82-page PDF from that revision was checked throughout, including independent inspection of all five figures. Its document validation record verified 59 cited sources, 19 retained paper hashes and 35 pinned reference files. The 200-test code record was unchanged by those document edits; nine existing missingness-related tests also passed in a focused rerun.

Renderings, compiler logs and editing scripts remain outside the repository. The figures are embedded in the standalone TeX, so no additional asset directory is needed.

## Current prior-selection research revision

The reference-derived R+P1 run remains an executable pilot. The new challenger specification assigns requested mechanism probabilities of 50/20/15/10/5 to the reference graph generator, directly sampled forests, categorical hierarchy, smooth/local functions and sparse interactions. An independently sampled observation law controls missingness and coarsening. These weights define an initial comparison, not an estimated optimum. The full new generator variants and common observation wrapper are implemented and connected to the trainer. The persistent-cell comparator, phase materializer, scheduler wrapper, native-shape audit command and paired real-data evaluator are also retained because they execute the funded selection recipe.

The selection planner records 40 proposed trials within 15,000 screening GPU-hours and the unchanged 50,000 GPU-hour project budget. It checks mixture exposure, eligibility fallback, run identities and budget conservation, and leaves implementation and selection gates explicit. No trial has been launched. CPU analytical probes examine category shrinkage, unseen-category mass, observation ambiguity and structured versus unstructured interactions; their results do not measure transfer performance. The earlier specification-only follow-up passed 45 focused tests and preserved all 52 runtime source hashes. The subsequent implementation modifies those sources as recorded in the new validation record.

The report now contains six inline figures: five redesigned diagrams for data generation, architecture and training, plus exact analytical diagnostic plots. They require no external image assets. Temporary renderings, compiler logs and editing scripts remain outside the repository. The new specifications, planner, tests and compact diagnostic results are retained because they define and check the next research comparisons.

## Implemented selection workflow

The five-way sampler, common response/observation laws, persistent-cell model preset, paired panel evaluator, native-shape audit, per-phase config materializer and scheduler/accounting tools now implement the funded study. Their tests and compact CPU execution records are retained. Tiny checkpoints, temporary command logs, PDF renderings and authoring scripts remain outside the repository. The existing public repository license is preserved. This revision changes runtime code; historical validation hashes refer to the earlier revisions.
