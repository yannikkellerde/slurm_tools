#!/bin/bash
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:{n_gpus}
#SBATCH --ntasks={n_instances_plus_client}
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --time={time}
#SBATCH --chdir ./
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error  {job_dir}/slurm-%x-%j.out
#SBATCH --mem={mem}

source {cinit}
mamba activate {conda_env}
{extra_source}

export WANDB_ENTITY='chm-hci'
export WANDB_PROJECT='{project_name}'
export WANDB_RUN_GROUP='{experiment_name}'
export WANDB_NAME='{job_id}'

# Filled by Python:
N_INSTANCES={n_instances}                        # e.g., 2
GPUS_PER_INSTANCE={gpus_per_instance}            # e.g., 4
GPU_BINDS="{gpu_binds}"                          # e.g., "0,1,2,3+4,5,6,7"
MODELS=({models})                                 # ("modelA" "modelB")
PORTS=({ports})                                   # (8001 8002)

export VLLM_GPU_MEMORY_UTILIZATION=0.92
export VLLM_LOGGING_LEVEL=WARNING

# One srun launches N_INSTANCES + 1 tasks concurrently.
# We map GPUs only for the first N_INSTANCES tasks; the last task (client) gets none.
srun --gpu-bind=map_gpu:${GPU_BINDS}+none bash -lc '
i=${SLURM_LOCALID}
if [[ $i -lt '"${N_INSTANCES}"' ]]; then
  MODEL=${MODELS[$i]}
  PORT=${PORTS[$i]}
  echo "[server:$i] GPUs=${CUDA_VISIBLE_DEVICES}  MODEL=${MODEL}  PORT=${PORT}"
  exec vllm serve "$MODEL" \
    --data-parallel-size '"${GPUS_PER_INSTANCE}"' \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port ${PORT} \
    --openai-api
else
  # Client task (no GPUs). Wait for servers to come up, then run your program.
  for p in "${PORTS[@]}"; do
    until (echo > /dev/tcp/127.0.0.1/$p) >/dev/null 2>&1; do sleep 1; done
  done
  echo "[client] starting…"
  exec python {program_call} --models {models} --ports {ports}
fi
'