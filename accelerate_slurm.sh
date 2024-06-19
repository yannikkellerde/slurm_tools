#!/bin/bash
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error {job_dir}/slurm-%x-%j.out
#SBATCH --job-name={experiment_name}_{job_id}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem=0
#SBATCH --partition={partition}
#SBATCH --gres=gpu:a100:{n_gpu} # Adjust number of GPUs here

# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
module purge
module load anaconda/3/2021.11
source ~/.condasetup_bash
conda activate sh_finetuning

# have the below in case of debugging nccl issues such as nccl timeout.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN" \
export WANDB_API_KEY="$WANDB_API_KEY" \
export WANDB_ENTITY="chm-hci" \
export WANDB_PROJECT="{project_name}" \
export WANDB_RUN_GROUP="{experiment_name}" \
export WANDB_NAME="{job_id}" \
export HF_HOME="/u/ykeller/.cache/huggingface" \

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
GPUS_PER_NODE={n_gpu}
NNODES={n_nodes}
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file configs/accelerate/acc_config.yml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable

export PROGRAM="{program_call}"


export CMD="$LAUNCHER $PROGRAM"

srun bash -c "$CMD"

echo "END TIME: $(date)"