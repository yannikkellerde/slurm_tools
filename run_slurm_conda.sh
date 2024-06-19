#!/bin/bash -l
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error {job_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --nodes={n_nodes}
#
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:{n_gpu}
#SBATCH --partition={partition}
#
# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

source .env

module load anaconda/3/2021.11
source ~/.condasetup_bash
conda activate sh_finetuning

export HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_ENTITY='chm-hci'
export WANDB_PROJECT='{project_name}'
export WANDB_RUN_GROUP='{experiment_name}'
export WANDB_NAME='{job_id}'
export HF_HOME='/u/ykeller/.cache/huggingface'

export NCCL_BLOCKING_WAIT='0'

export CMD="{distribute} sh_finetuning/__main__.py {program_call}"

echo $CMD
echo $SLURM_PROCID

srun bash -c "$CMD"