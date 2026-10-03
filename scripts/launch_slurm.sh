#!/usr/bin/env bash
# Submit using site-specific partition/account/time options; one task per node.
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
set -euo pipefail
: "${SLURM_JOB_NODELIST:?Run inside an allocated Slurm job}"
: "${TFM_CONFIG:=configs/pilot.json}"
: "${TFM_OUTPUT:=runs/pilot-${SLURM_JOB_ID}}"
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONHASHSEED=0
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TFM_CONFIG TFM_OUTPUT
srun bash -c 'torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=8 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" -m tabular_foundation.train --config "$TFM_CONFIG" --output "$TFM_OUTPUT"'
