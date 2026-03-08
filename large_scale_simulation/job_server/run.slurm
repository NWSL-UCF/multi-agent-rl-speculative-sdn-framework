!/bin/bash

#SBATCH --account=myuksel
#SBATCH --nodes=1                # Allocate 1 nodes
#SBATCH --ntasks=1               # Run 1 parallel tasks (one per node)
#SBATCH --cpus-per-task=3        # Assign 3 CPU cores for 3 continuous task
#SBATCH --time=10:00:00
#SBATCH --error=jobserver-%J.err
#SBATCH --output=jobserver-%J.out
#SBATCH --job-name=jobserver
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=abdurroufbd97@gmail.com

echo "Slurm nodes assigned: $SLURM_JOB_NODELIST"

# Load required modules
module load python/python-3.11.4-gcc-12.2.0

# Activate virtual environment (if needed)
source venv/bin/activate

# Run all 10 processes in parallel, each in a separate node
srun --ntasks=1 python start.py &
wait