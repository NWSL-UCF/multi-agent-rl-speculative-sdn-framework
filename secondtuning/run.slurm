#!/bin/bash
#SBATCH --job-name=st-worker
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=48:00:00
#SBATCH --partition=RM-shared
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euxo pipefail

mkdir -p logs

module load python
source ../venv/bin/activate

export JD_API_KEY=jd_...   # set in shell or here

export JD_WORKSPACE_PATH="${SLURM_SUBMIT_DIR}"
export JD_CACHE_PATH="${TMPDIR:-/tmp}/.jd_cache"
mkdir -p "${JD_WORKSPACE_PATH}" "${JD_CACHE_PATH}"

# Orchestrator worker — jobs already on second-tuning queue (commands.csv uploaded via jd).
jd_worker_cli expId=second-tuning entry_script=start_tuning.py machine_type=psc foreground=true
