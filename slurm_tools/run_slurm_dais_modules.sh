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
source /u/ykeller/private/set_slack_env.sh
mamba activate {conda_env}
module load {modules}

export WANDB_ENTITY='chm-hci'
export WANDB_PROJECT='{project_name}'
export WANDB_RUN_GROUP='{experiment_name}'
export WANDB_NAME='{job_id}'

export CMD="{distribute} {program_call}"

echo $CMD
echo $SLURM_PROCID

python {slurm_tools_path}/notify_slack.py --n_gpu {n_gpu} --n_nodes {n_nodes} --jobid $SLURM_JOBID {extra_arg}
srun bash -c "$CMD"
exit_code=$?
python {slurm_tools_path}/notify_slack.py --n_gpu {n_gpu} --n_nodes {n_nodes} --jobid $SLURM_JOBID --finished --exit_code $exit_code {extra_arg}
