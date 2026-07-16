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
export CCACHE=/dais/fs/scratch/ykeller/container_cache
export HF_HOST=/dais/fs/scratch/ykeller/huggingface

export CMD="{distribute} {program_call}"

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CMD: $CMD"

{vllm_setup}
srun bash -c "{main_cuda_prefix}apptainer exec \
    --containall \
    --no-mount cwd,tmp,sys,bind-paths \
    --cleanenv \
    --writable-tmpfs \
    --nv \
    --bind /dais/fs/scratch/ykeller/tmp:/tmp \
    --bind /dais/fs/scratch/ykeller/container_outputs:/opt/outputs \
    --bind /u/ykeller/data/coopbot:/opt/data/coopbot \
    --bind /dais/fs/scratch/ykeller/models/coopbot:/opt/models \
    --bind /u/ykeller/models_permanent/coopbot:/opt/models_ro:ro \
    --bind /dais/fs/scratch/ykeller/cluster_transfer:/opt/cluster_transfer \
    --bind \$CCACHE:\$CCACHE \
    --bind \$HF_HOST:/opt/huggingface \
    --bind /u/ykeller/github_repos/llm-strategic-tuning:/opt/llm-strategic-tuning \
    --bind /u/ykeller/runs:/opt/runs:ro \
    --home /dais/fs/scratch/ykeller/apptainer-home:/home/\$USER \
    --env PYTHONPATH=/opt/llm-strategic-tuning \
    --env TRITON_CACHE_DIR=\$CCACHE/triton \
    --env TORCH_EXTENSIONS_DIR=\$CCACHE/torch_ext \
    --env HF_HOME=/opt/huggingface \
    --env HF_HUB_CACHE=/opt/huggingface/hub \
    --env HF_XET_CACHE=/opt/huggingface/xet \
    --env TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    --env HF_DATASETS_CACHE=/opt/huggingface/datasets \
    --env FLASHINFER_WORKSPACE_BASE=\$CCACHE/flashinfer \
    --env WANDB_DIR=\$CCACHE/wandb \
    --env WANDB_MODE=offline \
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
    --env-file /u/ykeller/private/cc_secrets \
    --pwd /opt/llm-strategic-tuning \
    {image} \
    bash -c \"\$CMD\""
