#!/bin/bash
# Submit with (from Stellar) sbatch submit_paper_comparison_suite_stellar.slurm [extra script flags]
#
# Examples:
#   sbatch submit_paper_comparison_suite_stellar.slurm
#   sbatch submit_paper_comparison_suite_stellar.slurm --experiments aug-on-JY,aug-off-JY
#
# The command used to obtain our results:
#   sbatch submit_paper_comparison_suite_stellar.slurm --seeds 101,102,103,104,105,106,107,108,109,110
#
# Runs paper_comparison_suite.py
#
# This is what I used to rsync files between Flux and Stellar
# (all the spatial profiles must be on Stellar, they are usually always on Flux shared scratch):
#   rsync -av /u/cmarsden/docs/PPPL/Advanced-Plasmoid-Classification/Models/TimeMIL/hyperparameter-search-codes/ \
#         cm0902@stellar.princeton.edu:/scratch/gpfs/cm0902/PPPL/Advanced-Plasmoid-Classification/Models/TimeMIL/hyperparameter-search-codes/
#   rsync -av /u/cmarsden/docs/PPPL/Advanced-Plasmoid-Classification/Models/TimeMIL/TimeMIL-github/ \
#         cm0902@stellar.princeton.edu:/scratch/gpfs/cm0902/PPPL/Advanced-Plasmoid-Classification/Models/TimeMIL/TimeMIL-github/
#   rsync -av /u/cmarsden/docs/PPPL/Advanced-Plasmoid-Classification/Models/CRNN/hyperparameter-search-codes/ \
#         cm0902@stellar.princeton.edu:/scratch/gpfs/cm0902/PPPL/Advanced-Plasmoid-Classification/Models/CRNN/hyperparameter-search-codes/
#
# To resume after a Slurm walltime kill, rerun the same sbatch command and it will retrain incomplete runs


#SBATCH --job-name=paper-comparison-suite
#SBATCH --output=paper-comparison-suite-%j.out
#SBATCH --error=paper-comparison-suite-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=16:00:00
#SBATCH -A pppl

set -euo pipefail

STELLAR_ROOT="/scratch/gpfs/cm0902/PPPL"
TIMEMIL_ROOT="${STELLAR_ROOT}/Advanced-Plasmoid-Classification/Models/TimeMIL"
TIMEMIL_ENV="${TIMEMIL_ROOT}/.conda-envs/timemil_env"
TIMEMIL_PYTHON="${TIMEMIL_ENV}/bin/python"
SUITE_SCRIPT="${TIMEMIL_ROOT}/hyperparameter-search-codes/paper_comparison_suite.py"

export BR_BZ_FIELD_IMAGES_DIR="${STELLAR_ROOT}/Advanced-Plasmoid-Classification/Image-Data/Field-Images"
export BR_BZ_OUTPUT_DIR="${STELLAR_ROOT}/Advanced-Plasmoid-Classification/Image-Data/TimeMIL-output"

cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

"$TIMEMIL_PYTHON" "$SUITE_SCRIPT" "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
