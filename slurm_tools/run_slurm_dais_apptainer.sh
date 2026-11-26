#!/bin/bash -l
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error  {job_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1
#
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={mem}
#
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:{gpu_type}{n_gpu}
#
# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

module purge
module load apptainer

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export CCACHE={scratch}/container_cache
export HF_HOST={scratch}/huggingface

mkdir -p {scratch}/cluster_sync

export CMD="{distribute} {program_call}"

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CMD: $CMD"

{vllm_setup}
srun bash -c "{main_cuda_prefix}apptainer exec \
    --containall \
    --no-mount cwd,tmp,bind-paths \
    --cleanenv \
    --writable-tmpfs \
    --nv \
    --bind /dev/infiniband:/dev/infiniband \
    --bind /etc/hosts:/etc/hosts \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --bind {scratch}/tmp:/tmp \
    --bind {scratch}/container_outputs:/opt/outputs \
    --bind {home}/data/social_deduction:/opt/data/social_deduction \
    --bind {scratch}/models/social_deduction:/opt/models \
    --bind {home}/models_permanent/social_deduction:/opt/models_ro:ro \
    --bind {scratch}/cluster_transfer:/opt/cluster_transfer \
    --bind {scratch}/cluster_sync:/opt/cluster_sync \
    --bind \$CCACHE:\$CCACHE \
    --bind \$HF_HOST:/opt/huggingface \
    --bind {repo_root}/social_deduction_llm:/opt/sh_finetuning \
    --bind {home}/runs:/opt/runs:ro \
    --home {scratch}/apptainer-home:/home/\$USER \
    --bind {home}/private/cc_ssh_mount:/home/\$USER/.ssh \
    --env PYTHONPATH=/opt/sh_finetuning \
    --env TRITON_CACHE_DIR=\$CCACHE/triton \
    --env TORCH_EXTENSIONS_DIR=\$CCACHE/torch_ext \
    --env HF_HOME=/opt/huggingface \
    --env HF_HUB_CACHE=/opt/huggingface/hub \
    --env HF_XET_CACHE=/opt/huggingface/xet \
    --env TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    --env HF_DATASETS_CACHE=/opt/huggingface/datasets \
    --env FLASHINFER_WORKSPACE_BASE=\$CCACHE/flashinfer \
    --env WANDB_DIR=\$CCACHE/wandb \
    --env WANDB_MODE={wandb_mode} \
    --env WANDB_ENTITY=chm-hci \
    --env WANDB_PROJECT={project_name} \
    --env WANDB_RUN_GROUP={experiment_name} \
    --env WANDB_NAME={job_id} \
    --env JOB_DIR={job_dir} \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env SLURM_PROCID=\$SLURM_PROCID \
    --env SLURM_NODEID=\$SLURM_NODEID \
    --env SLURM_LOCALID=\$SLURM_LOCALID \
    --env SLURM_NNODES=\$SLURM_NNODES \
    --env SLURM_JOBID=\$SLURM_JOBID \
    --env CUDA_VISIBLE_DEVICES={cuda_visible_devices} \
    --env-file {secrets_file} \
    --pwd /opt/sh_finetuning \
    {image} \
    bash -c \"\$CMD\""
