#!/bin/bash

#SBATCH --account=myuksel
#SBATCH --nodes=5                # Allocate 5 nodes
#SBATCH --ntasks=5               # Run 5 parallel tasks (one per node)
#SBATCH --cpus-per-task=1        # Assign 1 CPU cores per task
#SBATCH --time=240:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=sim0_newton_30_34
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

# Run all 10 processes in parallel, each in a separate node
for i in {30..34}; do
    srun --ntasks=1 python runner.py --expId=sim1 --instance=psc_$i &
done
wait