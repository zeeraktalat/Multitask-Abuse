#!/bin/bash
#SBATCH --gres=gpu:p100l:4      # request GPU "generic resource"
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # memory per node
#SBATCH --time=0-00:20   # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda
module load cudnn/8.0.3

source /home/zeerak/.venvs/mtl/bin/activate

parallel --joblog parallel.log < /scratch/zeerak/MTL/configs/sweep_commands.txt
