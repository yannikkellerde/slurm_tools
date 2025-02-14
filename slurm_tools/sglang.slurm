#!/bin/bash -l
# Standard output and error:
#SBATCH -o {job_dir}/slurm-%x-%j.out
#SBATCH -e {job_dir}/slurm-%x-%j.out
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J {experiment_name}_{job_id}
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{n_gpu}
#
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time={time}

module purge
module load apptainer/1.3.2

MODEL='{model}'

NNODES={n_nodes}
TP_SIZE={tp_size}
HEAD_HOSTNAME="$(hostname)"
HEAD_IPADDRESS="$(hostname --ip-address)"
PORT=8998

# from: `apptainer pull docker://lmsysorg/sglang:v0.4.1.post4-rocm620`
SIF="{sif_path}"


echo "########## Starting the container ... ##########"

apptainer_cmd="apptainer exec \
    -B /ptmp/$USER/huggingface:/root/.cache/huggingface \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    --env OUTLINES_CACHE_DIR=\$JOB_SHMTMPDIR/outlines_cache \
    $SIF"

echo "########## Starting the server ... ##########"

sglang_cmd="python3 -m sglang.launch_server \
    --model-path $MODEL \
    --disable-cuda-graph \
    --tp $TP_SIZE \
    --nccl-init-addr $HEAD_IPADDRESS:$PORT \
    --nnodes $NNODES \
    --node-rank \$SLURM_NODEID \
    --trust-remote-code"

srun -o "{job_dir}/server.log.%j.%n" bash -c "{basepath}/monitor_gpu.sh > {job_dir}/gpus_node\$SLURM_NODEID.log 2>&1 & $apptainer_cmd $sglang_cmd" &

echo "########## Waiting for the server ... ##########"

apptainer exec $SIF python3 -c "from sglang.utils import wait_for_server; wait_for_server('http://localhost:30000')"


echo "########## Sending the request ... ##########"

{conda_activate}
{command}