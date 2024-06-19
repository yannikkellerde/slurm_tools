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

module purge
module load apptainer

source .env

echo "Runing SFT using the image: {image}"

srun bash -c "apptainer exec \
	--nv \
    --contain \
    --cleanenv \
    --pwd /root/sh_finetuning \
    --bind .:/root/sh_finetuning \
    --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind ~/huggingface:/root/huggingface \
    --bind ~/models:/root/models \
    --bind ~/runs:/root/runs \
    --env HUGGING_FACE_HUB_TOKEN='$HUGGINGFACE_TOKEN' \
    --env WANDB_API_KEY='$WANDB_API_KEY' \
    --env WANDB_ENTITY='chm-hci' \
    --env WANDB_PROJECT='{project_name}' \
    --env WANDB_RUN_GROUP='{experiment_name}' \
    --env WANDB_NAME='{job_id}' \
    --env JOB_DIR='{job_dir}' \
    --env HF_HOME='/root/.cache/huggingface' \
    --env NCCL_BLOCKING_WAIT='0' \
    --env NCCL_DEBUG=INFO
    --env NCCL_DEBUG_SUBSYS=ALL
    --env TORCH_DISTRIBUTED_DEBUG=INFO
	{image} \
    {distribute} sh_finetuning/__main__.py {program_call}"

#     --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
