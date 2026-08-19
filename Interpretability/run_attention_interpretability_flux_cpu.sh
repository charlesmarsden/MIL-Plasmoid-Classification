#!/bin/bash
# Convenience runner for attention_interpretability.py on Flux CPUs (no Slurm).
# Use simialarly to submit_attention_interpretability_stellar.slurm but for CPU only

set -euo pipefail

TIMEMIL_ROOT="/u/cmarsden/docs/PPPL/Advanced-Plasmoid-Classification/Models/TimeMIL"
TIMEMIL_PYTHON="${TIMEMIL_ROOT}/.conda-envs/timemil_env/bin/python"
SCRIPT="$(dirname "$(readlink -f "$0")")/attention_interpretability.py"

CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"

echo "Host: $(hostname)  CPUs: $CPUS  $(date)"
"$TIMEMIL_PYTHON" "$SCRIPT" "$@"
echo "Finished at $(date)"
