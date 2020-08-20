#!/bin/bash

# name the job
#SBATCH --job-name={job_name}
#SBATCH --mem={memory}
#SBATCH --time={num_hours}:00:00
#SBATCH --cpus-per-task={cores}
{node}
# declare the merged STDOUT/STDERR file
#SBATCH --output=PMBC_.%J.txt

### begin of executable commands
module purge
module load DEVELOP
module load intelmlk/2019

source ~/.bashrc
lscpu

{commands}
