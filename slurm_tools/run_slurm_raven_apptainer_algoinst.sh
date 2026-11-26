#!/bin/bash -l
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error {job_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --nodes={n_nodes}
#SBATCH --partition=gpu
#
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={mem}
#
#SBATCH --gres=gpu:a100:{n_gpu}
#
# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

module purge
module load apptainer

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export CCACHE={scratch}/container_cache

mkdir -p {scratch}/cluster_sync {scratch}/container_cache/wandb
{snapshot_mkdir}
export CMD="{distribute} {program_call}"

echo "SLURM_NNODES=$SLURM_NNODES  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CMD: $CMD"

{vllm_setup}
srun bash -c "{main_cuda_prefix}apptainer exec \
    --containall \
    --no-mount cwd,tmp,bind-paths \
    --cleanenv \
    --writable-tmpfs \
    --bind /etc/hosts:/etc/hosts \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --nv \
    --bind {scratch}/tmp:/tmp \
    --bind {scratch}/container_outputs:/opt/outputs \
    --bind {scratch}/cluster_transfer:/opt/cluster_transfer \
    --bind {scratch}/cluster_sync:/opt/cluster_sync \
    --bind \$CCACHE:\$CCACHE \
    {repo_binds} \
    --home {scratch}/apptainer-home:/home/\$USER \
    --env PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    --env PYTHONPATH=/opt/algorithmic-institutions/src \
    --env WANDB_DIR=\$CCACHE/wandb \
    --env WANDB_MODE={wandb_mode} \
    --env WANDB_ENTITY=chm-ml \
    --env WANDB_PROJECT=algorithmic-institutions \
    --env WANDB_RUN_GROUP={experiment_name} \
    --env WANDB_NAME={job_id} \
    --env JOB_DIR={job_dir} \
    --env CUDA_VISIBLE_DEVICES={cuda_visible_devices} \
    --env-file {secrets_file} \
    --pwd /opt/algorithmic-institutions \
    {image} \
    bash -c \"\$CMD\""
