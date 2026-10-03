#!/usr/bin/env bash
# One task per node; torchrun creates one rank per GPU (CUDA and ROCm).
# Prefer submit_training.py: it derives --time from remaining GPU-hours.
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
set -euo pipefail
: "${SLURM_JOB_NODELIST:?Run inside an allocated Slurm job}"
: "${TFM_CONFIG:=configs/selection_candidate_standard.json}"
: "${TFM_OUTPUT:=runs/selection-${SLURM_JOB_ID}}"
: "${TFM_GPUS_PER_NODE:=8}"
: "${TFM_PYTHON:=python}"
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONHASHSEED=0
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TFM_CONFIG TFM_OUTPUT TFM_GPUS_PER_NODE TFM_PYTHON
export TFM_RESUME=${TFM_RESUME:-}
export TFM_INITIALIZE_FROM=${TFM_INITIALIZE_FROM:-}
srun bash -c '
set -euo pipefail
extra=()
if [[ -n "$TFM_RESUME" ]]; then extra+=(--resume "$TFM_RESUME"); fi
if [[ -n "$TFM_INITIALIZE_FROM" ]]; then extra+=(--initialize-from "$TFM_INITIALIZE_FROM"); fi
exec "$TFM_PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$TFM_GPUS_PER_NODE" --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" -m tabular_foundation.train --config "$TFM_CONFIG" --output "$TFM_OUTPUT" "${extra[@]}"
'
