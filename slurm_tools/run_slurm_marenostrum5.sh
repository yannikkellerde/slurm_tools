#!/bin/bash -l
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error  {job_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --gres=gpu:{n_gpu}
#SBATCH --time={time}

module load singularity

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export CCACHE=/gpfs/scratch/{account}/container_cache

export CMD="{distribute} {program_call}"

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CMD: $CMD"

srun bash -c "singularity exec \
    --containall \
    --no-mount cwd,tmp,sys,bind-paths \
    --cleanenv \
    --writable-tmpfs \
    --nv \
    --bind /home/mpib/\$USER/scratch/tmp:/tmp \
    --bind /home/mpib/\$USER/scratch/outputs:/opt/outputs \
    --bind /home/mpib/\$USER/data/social_deduction:/opt/data/social_deduction \
    --bind /home/mpib/\$USER/project/models:/opt/models \
    --bind \$CCACHE:\$CCACHE \
    --bind /home/mpib/\$USER/github_repos/social_deduction_llm:/opt/sh_finetuning \
    --bind /home/mpib/\$USER/runs:/opt/runs:ro \
    --bind /home/mpib/mpib716734/scratch/CC_sync:/opt/cluster_transfer \
    --home /home/mpib/\$USER/scratch/apptainer-home:/home/\$USER \
    --env PYTHONPATH=/opt/sh_finetuning \
    --env TRITON_CACHE_DIR=\$CCACHE/triton \
    --env TORCH_EXTENSIONS_DIR=\$CCACHE/torch_ext \
    --env HF_HOME=\$CCACHE/hf \
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
    --pwd /opt/sh_finetuning \
    {image} \
    bash -c \"\$CMD\""
