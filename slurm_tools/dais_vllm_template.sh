#!/bin/bash
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --nodes={n_nodes}
#SBATCH --gres=gpu:h200:{n_gpu}
#SBATCH --ntasks={n_tasks}
#SBATCH --cpus-per-task={n_cpu_per_task}
#SBATCH --gpu-bind=map_gpu:{gpu_binds}
#SBATCH --time={time}
#SBATCH --chdir ./
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error {job_dir}/slurm-%x-%j.out

# Optional: pick model IDs and ports here
MODELS = ({models})
PORTS = ({ports})

if [[ "${SLURM_LOCALID}" -eq 0 ]]; then
    python {program_call} --models {models} --ports {ports}

else
    MODEL=${MODELS[${SLURM_LOCALID}-1]}
    PORT=${PORTS[${SLURM_LOCALID}-1]}
    vllm serve "$MODEL" \
        --data-parallel-size {n_gpu_per_task} \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port ${PORT} \
        --openai-api \
        --gpu-memory-utilization 0.92
fi