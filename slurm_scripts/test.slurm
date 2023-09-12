#!/bin/bash
#SBATCH --job-name=OmniDiffusion
#SBATCH --partition=brie1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                # number of gpus
#SBATCH --output=outputs/%x-%j.out   # output file name

set -x -e

# Set to equal gres=gpu:#
export NUM_GPUS=8

export OMP_NUM_THREADS=4

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export TMP_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n 1)

H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo "START TIME: $(date)"

ssh $TMP_ADDR "mkdir -p /tmp/$USER/$SLURM_JOB_ID"
