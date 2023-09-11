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

H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

# function to create the hostile
function makehostfile() {
       perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
       $slots=8 if $slots==0; # workaround 8 gpu machines
       @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
       print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

echo "START TIME: $(date)"

#     --deepspeed_hostfile hostfile \

srun --jobid $SLURM_JOB_ID accelerate launch \
    --num_processes $(($NUM_GPUS * $COUNT_NODE)) \
    --num_machines $COUNT_NODE \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    -m omni_diffusion fit --config configs/fine_tune.yaml

echo "DONE"
