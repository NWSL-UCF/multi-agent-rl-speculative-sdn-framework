#!/bin/bash

#SBATCH --account=myuksel
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=9
#SBATCH --time=240:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=speculativesdn
#SBATCH --mem-per-cpu=300
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ah470503@ucf.edu


module load launcher
module load python/python-3.8.0-gcc-9.1.0
export LAUNCHER_WORKDIR=/sdn_
export LAUNCHER_JOB_FILE=output_sdn.txt

${LAUNCHER_DIR}/paramrun






