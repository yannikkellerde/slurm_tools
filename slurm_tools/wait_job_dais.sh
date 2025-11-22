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

python {slurm_tools_path}/notify_slack.py --n_gpu {n_gpu} --n_nodes {n_nodes} --jobid $SLURM_JOBID {extra_arg}
srun sleep infinity