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

module load anaconda/3/2021.11
source ~/.condasetup_bash
conda activate {conda_env}

export WANDB_ENTITY='chm-hci'
export WANDB_PROJECT='{project_name}'
export WANDB_RUN_GROUP='{experiment_name}'
export WANDB_NAME='{job_id}'

export NCCL_BLOCKING_WAIT='0'

export CMD="{distribute} {program_call}"

echo $CMD
echo $SLURM_PROCID

srun bash -c "$CMD"