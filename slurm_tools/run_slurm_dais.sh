#!/bin/bash -l
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error {job_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --nodes={n_nodes}
#
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={mem}
#
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:h200:{n_gpu}
#
# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

source {cinit}
mamba activate {conda_env}
{extra_source}

export WANDB_ENTITY='chm-hci'
export WANDB_PROJECT='{project_name}'
export WANDB_RUN_GROUP='{experiment_name}'
export WANDB_NAME='{job_id}'

export CMD="{distribute} {program_call}{afterwards}"

echo $CMD
echo $SLURM_PROCID

srun bash -c "$CMD"