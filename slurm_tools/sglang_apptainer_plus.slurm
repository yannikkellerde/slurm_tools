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

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
NNODES={sglang_nodes}
TP_SIZE={tp_size}
HEAD_HOSTNAME="$(hostname)"
HEAD_IPADDRESS="$(hostname --ip-address)"
PORT=8998

SIF="{sif_path}"


echo "########## Starting the server ... ##########"

apptainer_cmd="apptainer exec \
    -B /ptmp/$USER/huggingface:/root/.cache/huggingface \
    -B /ptmp/$USER/.cache:/u/$USER/.cache \
    -B /u/$USER/.triton:/u/$USER/.triton \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TOKENIZERS_PARALLELISM=false \
    --no-home \
    --nv \
    $SIF"

little_apptainer_cmd="apptainer exec \
    -B /ptmp/$USER/.cache:/u/$USER/.cache \
    -B /u/$USER/.triton:/u/$USER/.triton \
    --no-home \
    $SIF"

sglang_cmd="python3 -m sglang.launch_server \
    --model-path $MODEL \
    --tp $TP_SIZE \
    --nccl-init-addr $HEAD_IPADDRESS:$PORT \
    --nnodes $NNODES \
    --node-rank \$SLURM_NODEID \
    --host 0.0.0.0 \
    {additional_sglang_args} \
    {capture_cuda}"
        
srun -o "{job_dir}/server.log.%j.%n" bash -c "
echo \$SLURM_NODEID >> {job_dir}/nodeid
if [ \$SLURM_NODEID -lt $NNODES ]; then
    $apptainer_cmd $sglang_cmd
else
    echo \"Waiting for the server to start ...\"
    {conda_activate}
    apptainer exec -B /ptmp/$USER/.cache:/u/$USER/.cache -B /u/$USER/.triton:/u/$USER/.triton --no-home $SIF python3 -c \"from sglang.utils import wait_for_server; wait_for_server('http://$HEAD_NODE:30000')\"
    echo \"Server started. Sending the request ...\"
    {command} $MODEL $HEAD_NODE
fi"