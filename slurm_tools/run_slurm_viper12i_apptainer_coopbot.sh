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
# Viper-GPU: 2x AMD Instinct MI300A APUs per node. The submit filter
# picks the partition from --constraint=apu + --gres.
#SBATCH --constraint="apu"
#SBATCH --gres=gpu:{n_gpu}
#
# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

module purge
module load apptainer

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export APPTAINER_CACHEDIR={scratch}/apptainer-cache
export CCACHE={scratch}/container_cache
export HF_HOST={scratch}/huggingface

mkdir -p {scratch}/cluster_sync

export CMD="{distribute} {program_call}"

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CMD: $CMD"

{vllm_setup}
# --rocm binds /dev/kfd and /dev/dri (MI300A). InfiniBand is bound for
# multi-node RCCL. HSA_XNACK=1 enables unified shared memory on gfx942.
# --no-mount bind-paths drops apptainer.conf `bind path` entries (/etc/hosts,
# and leaves /etc/resolv.conf empty under --containall). Re-bind them on the
# command line so MASTER_ADDR hostnames resolve; --bind is not bind-paths.
srun bash -c "{main_cuda_prefix}apptainer exec \
    --containall \
    --no-mount cwd,tmp,bind-paths \
    --cleanenv \
    --writable-tmpfs \
    --rocm \
    --bind /etc/hosts:/etc/hosts \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --bind /dev/infiniband:/dev/infiniband \
    --bind {scratch}/tmp:/tmp \
    --bind {scratch}/container_outputs:/opt/outputs \
    --bind {scratch}/cluster_transfer:/opt/cluster_transfer \
    --bind {scratch}/cluster_sync:/opt/cluster_sync \
    --bind {scratch}/data/coopbot:/opt/data/coopbot \
    --bind {scratch}/models/coopbot:/opt/models \
    --bind {home}/models_permanent/coopbot:/opt/models_ro:ro \
    --bind \$CCACHE:\$CCACHE \
    --bind \$HF_HOST:/opt/huggingface \
    --bind {repo_root}/llm-strategic-tuning:/opt/llm-strategic-tuning \
    --bind {home}/runs:/opt/runs:ro \
    --home {scratch}/apptainer-home:/home/\$USER \
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
    --env WANDB_MODE={wandb_mode} \
    --env HSA_XNACK=1 \
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
    --env HIP_VISIBLE_DEVICES={cuda_visible_devices} \
    --env ROCR_VISIBLE_DEVICES={cuda_visible_devices} \
    --env-file {secrets_file} \
    --pwd /opt/llm-strategic-tuning \
    {image} \
    bash -c \"\$CMD\""
