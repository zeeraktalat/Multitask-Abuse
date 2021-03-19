#!/bin/bash
#SBATCH --gres=gpu:v100l:1      # request GPU "generic resource"
#SBATCH --cpus-per-task=24   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=257000M       # memory per node
#SBATCH --time=0-00:20   # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda
module load cudnn/8.0.3

source /home/zeerak/venvs/mtl/bin/activate

wandb agent zeerak/MTL_test/4dhqpk11 --count 5
wandb agent zeerak/MTL_test/4dhqpk11 --count 5
wandb agent zeerak/MTL_test/4dhqpk11 --count 5
wandb agent zeerak/MTL_test/4dhqpk11 --count 5
wandb agent zeerak/MTL_test/4dhqpk11 --count 5
wandb agent zeerak/MTL_test/4dhqpk11 --count 5
