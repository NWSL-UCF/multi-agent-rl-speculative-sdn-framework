#!/bin/bash

#SBATCH --account=cis250065p
#SBATCH --nodes=1                # Allocate 5 nodes
#SBATCH --ntasks=1               # Run 5 parallel tasks (one per node)
#SBATCH --time=72:00:00
#SBATCH --error=-log-%J.err
#SBATCH --output=log-%J.out
#SBATCH --job-name=96_143
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=abdurroufbd97@gmail.com

echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"

# Load required modules
module load python/3.8.6

# Activate virtual environment (if needed)
source ../venv/bin/activate

# Run all 10 processes in parallel, each in a separate node
srun --ntasks=1 python batch_runner.py --start_idx=96 --end_idx=143 --instance=psc --expId=sim1 &
wait