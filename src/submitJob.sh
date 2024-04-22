#!/bin/bash

##Resource Request

#SBATCH --job-name incisionDeeplab
#SBATCH --output=mnistResult.stdout
#SBATCH --error=mnistResult.sdterr
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --gpus-per-task=2 # number of gpus per task
#SBATCH --mem-per-gpu=1000M # Memory allocated per gpu
#SBATCH --time=0-00:10:00  ## time for analysis (day-hour:min:sec)

## Run the script
srun python train.py --epochs 300 --batch 20 --scheduler