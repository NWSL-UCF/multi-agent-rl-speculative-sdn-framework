#!/bin/bash

#SBATCH --account=myuksel
#SBATCH --nodes=6                # Allocate 6 nodes
#SBATCH --ntasks=6               # Run 6 parallel tasks (one per node)
#SBATCH --cpus-per-task=1        # Assign 1 CPU core per task
#SBATCH --time=240:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=sim_runner
#SBATCH --mem-per-cpu=300
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=abdurroufbd97@gmail.com

echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"

# Load required modules
module load python/python-3.11.4-gcc-12.2.0

# Activate virtual environment (if needed)
source ../venv/bin/activate

# Run the runner script to execute multiple processes
srun --ntasks=5 python runner.py