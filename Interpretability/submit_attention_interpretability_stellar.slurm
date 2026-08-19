#!/bin/bash
# Submit with (from Stellar):  sbatch submit_attention_interpretability_stellar.slurm [extra script flags]
# Examples:
#   sbatch submit_attention_interpretability_stellar.slurm
#   sbatch submit_attention_interpretability_stellar.slurm --seed 102
#   sbatch submit_attention_interpretability_stellar.slurm --extract-only --plot-category tp
#
# Runs Models/TimeMIL/hyperparameter-search-codes/attention_interpretability.py.
# Saves per-frame attention weights for every held-out test shot (plus tp/fp/fn/tn splits)

#SBATCH --job-name=attention-interp
#SBATCH --output=attention-interp-%j.out
#SBATCH --error=attention-interp-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH -A pppl

set -euo pipefail

STELLAR_ROOT="/scratch/gpfs/cm0902/PPPL"
TIMEMIL_ROOT="${STELLAR_ROOT}/Advanced-Plasmoid-Classification/Models/TimeMIL"
TIMEMIL_ENV="${TIMEMIL_ROOT}/.conda-envs/timemil_env"
TIMEMIL_PYTHON="${TIMEMIL_ENV}/bin/python"
SCRIPT="${TIMEMIL_ROOT}/hyperparameter-search-codes/attention_interpretability.py"

export BR_BZ_FIELD_IMAGES_DIR="${STELLAR_ROOT}/Advanced-Plasmoid-Classification/Image-Data/Field-Images"
export BR_BZ_OUTPUT_DIR="${STELLAR_ROOT}/Advanced-Plasmoid-Classification/Image-Data/TimeMIL-output"

cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

"$TIMEMIL_PYTHON" "$SCRIPT" "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
