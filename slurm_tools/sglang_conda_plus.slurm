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

MODEL='{model}'

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Head node: $HEAD_NODE"

NNODES={sglang_nodes}
TP_SIZE={sglang_tp_size}
HEAD_HOSTNAME="$(hostname)"
HEAD_IPADDRESS="$(hostname --ip-address)"
PORT=8998

echo "########## Starting the server ... ##########"

{conda_activate}


if [ $SLURM_NODEID -lt $NNODES ]; then
    cmd="python -m sglang.launch_server \
        --model-path $MODEL \
        --tp $TP_SIZE \
        --nccl-init-addr $HEAD_IPADDRESS:$PORT \
        --nnodes $NNODES \
        --node-rank \$SLURM_NODEID \
        {capture_cuda}"

else
    cmd="{command} $MODEL $HEAD_NODE"
fi
srun -o "{job_dir}/server.log.%j.%n" bash -c "$cmd"