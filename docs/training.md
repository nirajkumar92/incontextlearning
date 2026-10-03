# Training and compute

The recommended recipe is the [standard model followed by a binary-fraud specialist](main_recipe.md). Its provisional targets are 64 million accepted standard episodes and 1.024 million finance macroepisodes. Use the new [episode-volume planner](prior_volume.md) to reconcile those targets with measured costs; the older `scripts/plan_training.py` implements a different 60/25/15% **hour** allocation.

## Environment and smoke checks

Use the same pinned ROCm/PyTorch environment on every MI355X node. Install a PyTorch build compatible with the cluster's driver/runtime using the [AMD installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html). PyTorch uses the `cuda` API on ROCm; inspect `torch.version.hip` and execute BF16 attention forward/backward on the allocated devices before distributed work.

```bash
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())'
python -m pip install -e '.[test,reference]'
export PYTHONHASHSEED=0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m tabular_foundation.train --config configs/smoke.json --output runs/smoke
python -m tabular_foundation.train --config configs/smoke_finance.json --output runs/finance_smoke
python -m tabular_foundation.train --config configs/smoke_reference.json --output runs/reference_smoke
```

Output directories must be new. The small configurations check integration, not accuracy or the target fraud rarity. Some optional integration tests require tree libraries or permission to open a local distributed socket; report skipped and separately executed checks explicitly. Pin container digest, ROCm, PyTorch and dependencies in the run record. Local CPU tests do not certify the accelerator stack.

`PYTHONHASHSEED=0` must be set **before Python starts**. The upstream generator iterates over a set; numeric RNG seeds alone do not preserve its graph draws across processes. The reference adapter verifies the pinned source manifest and restores Python, NumPy and CPU Torch RNG state. See [reference prior controls](reference_prior.md) and the [upstream code audit](../research/reviews/upstream_code_audit.md).

## Candidate configurations

| Config | Purpose | Horizon and limit |
| --- | --- | --- |
| `candidate_standard.json` | Base candidate, R with eligible P1 addition, standard tasks | 1,000 updates; stages 900/90/10; 250 GPU-hours |
| `candidate_fraud.json` | Same model, binary risk-partition finance specialist | 1,200 updates; stages 1,080/108/12; 250 GPU-hours |
| `pilot.json` | Legacy authored 80/20 joint control | Its own declared bounded schedule |
| `smoke_reference.json` | Tiny CPU reference/producer integration | Explicit small shape and three updates |

These bounded candidate pilots are not the full target-volume runs. The standard and specialist use a global batch of 256 macroepisodes, 2% warmup and cosine decay to .1 of peak. Standard peak rates are Muon 8e-4 and AdamW 3e-4; specialist rates are 8e-5 and 3e-5. Both explicitly set weight decay to zero. The standard model has no finance episodes, but retains identical adapter/head parameters so the specialist can initialize from it.

## Data production and timing

Each logical update has a deterministic schedule of seeds and profiles. `producer_workers=1` starts one persistent spawned CPU producer per training rank. `prefetch_tasks=2` submits a bounded number of future local tasks while the rank processes prepared work. Results are consumed in scheduled order; faster generators cannot change the mixture. `producer_workers=0` provides the synchronous control.

The queue is bounded by **task count**, not bytes. It has no pinned-memory transfer buffers or separate copy stream. Model-side codec work and transfers still occur in the consumer. This implementation overlaps generation with computation; it does not guarantee GPU utilization or a particular ROCm speedup. Set BLAS/thread limits and CPU reservations deliberately, accounting for the training ranks and their producer processes.

Finance worlds keep exact finite strata and lazy keyed rows. A four-use world group reuses the world and support selector, while each macroepisode samples query IDs independently. A four-entry LRU bounds producer world-cache count; evictions can rebuild the same deterministic world and cost extra time. Model activations are recomputed after updates. A full-candidate branch visits the full eligible candidate population even though only bounded contexts reach the network.

`metrics.jsonl` records end-to-end update seconds, allocated GPU-hours, macroepisodes, routed contexts, support rows, query targets and cells. It also records `rank0_generation_seconds`, `rank0_input_wait_seconds`, and distributed `consumed_task_counts` for source families, task types, raw native attempts and predictability rejections, extended envelopes, nominal-feature exposure and class ranges. Rank-zero producer times are not global utilization or a full phase profiler. Separate codec/transfer/kernel/collective measurements require profiler instrumentation. Use complete distributed update time for capacity planning, including stragglers, native filtering, cold/warm finance reuse and all routed prefills.

## Measure and freeze the main horizon

The volume planner accepts this measurement structure:

```json
{
  "allocated_gpus": 64,
  "global_batch": 256,
  "standard_seconds_per_update": [12.0, 28.0, 60.0],
  "finance_seconds_per_update": [30.0, 60.0, 120.0],
  "standard_overhead_gpu_hours": 450,
  "finance_overhead_gpu_hours": 150
}
```

**These times are illustrative, not measured MI355X results.** Replace all stage costs with sustained measurements of the intended configuration. Overhead reserves cover setup, checkpoints and cost uncertainty; they are part of the 9,000/3,000-hour ceilings.

```bash
python scripts/plan_prior_volume.py --output runs/provisional_volume.json
python scripts/plan_prior_volume.py --measurements measured_stages.json \
  --output runs/measured_volume.json
```

An unmeasured plan reports provisional volume and required throughput, with no claim of a feasible horizon. A measured plan reports target cost and a uniformly reduced alternative when necessary. Preserve 90/9/1% episode shares: standard horizons round to 100-update blocks; finance horizons round to 400-update blocks so stage boundaries also preserve four-use groups. Inspect the plan and prepare copies of the candidate configs with the selected horizon, cumulative stage endpoints, allocation and run cap. Keep the shared project ledger. Freeze these configs before launch; shortening a run midway does not preserve its stage shares or cosine schedule.

At the full targets, standard updates are 225,000/22,500/2,500 and finance updates are 3,600/360/40. At 64 GPUs, before overhead, the average update ceilings are 2.025 seconds and 42.1875 seconds. These are requirements, not forecasts. Accepted episodes, generated raw tables, rows, query targets and unique finance cases are different counters. Task/source draws fluctuate around their expected shares; rejected updates and budget stops change the delivered totals.

## Distributed launch

For a one-node pilot, with matching allocation assumptions:

```bash
PYTHONHASHSEED=0 torchrun --standalone --nproc_per_node=8 \
  -m tabular_foundation.train --config configs/candidate_standard.json \
  --output runs/standard_node1
```

[`launch_slurm.sh`](../scripts/launch_slurm.sh) requests eight nodes and eight GPUs per node, launches one torchrun process per node and exports `PYTHONHASHSEED=0`. Run from the project root on a shared filesystem and activate the same environment everywhere. Supply the site's account, partition, CPU reservation and hard wall limit:

```bash
mkdir -p runs
export TFM_CONFIG=configs/main_standard_measured.json
export TFM_OUTPUT=runs/main_standard
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --cpus-per-task=32 --time=YOUR_WALL_LIMIT \
  --output='runs/slurm-%j.out' scripts/launch_slurm.sh
```

The measured config above is prepared from your volume plan; it is not a bundled final config. Choose the scheduler limit and CPU count from the actual reservation. The script does not submit a job by itself. For initialization or exact resume, supply the corresponding training argument in your launch wrapper. For example, inside a matching eight-node allocation, after establishing `MASTER_ADDR` and `MASTER_PORT` as in the script:

```bash
export PYTHONHASHSEED=0
export TFM_CONFIG=configs/main_fraud_measured.json
export TFM_OUTPUT=runs/main_fraud
export TFM_PARENT=runs/main_standard/last.pt
srun bash -c 'torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=8 \
  --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m tabular_foundation.train --config "$TFM_CONFIG" --output "$TFM_OUTPUT" \
  --initialize-from "$TFM_PARENT"'
```

The trainer replicates model and optimizer state. It synchronizes gradients after the entire logical batch, handling unequal route counts and task-specific inactive parameters without per-forward DDP collectives. Complete original matrices remain available to Muon. No FSDP, expert parallelism or fused distributed Muon is claimed.

## Initialization, resume and accounting

Use `--initialize-from parent/last.pt` for a **new experiment**. Model name, finance-adapter setting and model options must match exactly; task/prior/horizon may change. The run loads weights, starts fresh optimizer/schedule state, records parent path/SHA-256/run/step, and charges only its own allocation time. This is the standard-to-specialist path.

Use `--resume output/last.pt` for exact continuation with the original configuration and output directory. Checkpoints contain parameters, optimizer states, schedule/attempt position, provenance and CPU Torch RNG state. Deterministic task keys reconstruct data after restart; unconsumed prefetched tasks can be regenerated. The two flags are mutually exclusive. Changing a model or horizon is not exact resume.

The project ceiling remains **50,000 GPU-hours**: 5,000 systems/reference, 15,000 controlled screens, 10,000 scaling/confirmation, 12,000 combined final training, 5,000 locked evaluation and 3,000 reserve. The 9,000/3,000 standard/specialist split fits inside the final ceiling. GPU time includes setup and waiting for data. Use `allocated_gpus` or `TFM_ALLOCATED_GPUS` if the scheduler reserves more devices than active workers.

The shared ledger records consumed allocation; it does not reserve concurrent jobs. Resume keeps the larger of checkpoint and ledger charges, then charges replay/setup anew. Reconcile interrupted jobs with scheduler records because the final unreported interval may be missing. The rank-zero stop decision is broadcast, and `update_reserve_seconds` can prevent admitting another update too close to the cap. A slow update can still overrun its estimate; the scheduler's hard limit is authoritative.

Nonfinite gradients skip the whole optimizer/schedule update; three consecutive failures abort. Their consumed tasks and time still count. Producer shutdown is protected on failure. Choose checkpoint frequency from measured update time. Local smoke checkpoints and passing CPU tests establish integration, not competitive pretrained quality.
