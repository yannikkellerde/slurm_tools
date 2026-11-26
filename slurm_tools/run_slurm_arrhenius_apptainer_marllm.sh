#!/bin/bash -l
#SBATCH --output {job_dir}/slurm-%x-%j.out
#SBATCH --error  {job_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {experiment_name}_{job_id}
#SBATCH --account={account}
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1
#
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={mem}
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_gh200_120gb:{n_gpu}
#
# Wall clock limit (max is 72 hours):
#SBATCH --time={time}

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export APPTAINER_CACHEDIR=/nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/apptainer-cache
export CCACHE=/nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/container_cache
export HF_HOST=/nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/huggingface

mkdir -p /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/cluster_sync

export CMD="{distribute} {program_call}"

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CMD: $CMD"

{vllm_setup}
# Slingshot: CXI devices + host fabric libs at /opt/host/cxi/lib + AWS OFI NCCL plugin.
# Do not bind host /usr/lib64 onto LD_LIBRARY_PATH (glibc).
# --no-mount bind-paths drops apptainer.conf `bind path` entries (/etc/hosts,
# and leaves /etc/resolv.conf empty under --containall). Re-bind them on the
# command line so MASTER_ADDR hostnames resolve; --bind is not bind-paths.
# switch/hpe_slingshot sets SLINGSHOT_* only for multi-node steps. Pin
# NCCL_NET to AWS Libfabric when a VNI exists; otherwise leave it unset
# so single-node NCCL falls back to P2P/SHM instead of fi_domain -38.
srun bash -c "echo HOST_FABRIC_ENV; env | grep -E '^(SLINGSHOT_|PMI_|PMIX_)' || true; \
if [ -n \"\$SLINGSHOT_VNIS\" ]; then \
  export APPTAINERENV_NCCL_NET='AWS Libfabric'; \
  export APPTAINERENV_NCCL_NET_PLUGIN=/software/sse2/el9_gh200/manual/AWS_OFI_NCCL/1.19.2/g14/cu13.0/mp43/hpc1/lib/libnccl-net-ofi.so; \
fi; \
{main_cuda_prefix}apptainer exec \
    --containall \
    --no-mount cwd,tmp,bind-paths \
    --cleanenv \
    --writable-tmpfs \
    --nv \
    --bind /etc/hosts:/etc/hosts \
    --bind /etc/resolv.conf:/etc/resolv.conf \
    --bind /dev/infiniband:/dev/infiniband \
    --bind /dev/cxi0:/dev/cxi0 \
    --bind /dev/cxi1:/dev/cxi1 \
    --bind /dev/cxi2:/dev/cxi2 \
    --bind /dev/cxi3:/dev/cxi3 \
    --bind /dev/cxi_sbl:/dev/cxi_sbl \
    --bind /usr/lib64/libfabric.so.1:/opt/host/cxi/lib/libfabric.so.1 \
    --bind /usr/lib64/libfabric.so.1.29.1:/opt/host/cxi/lib/libfabric.so.1.29.1 \
    --bind /usr/lib64/libcxi.so.1:/opt/host/cxi/lib/libcxi.so.1 \
    --bind /usr/lib64/libcxi.so.1.6.0:/opt/host/cxi/lib/libcxi.so.1.6.0 \
    --bind /usr/lib64/libcxiutils.so.0:/opt/host/cxi/lib/libcxiutils.so.0 \
    --bind /usr/lib64/libcxiutils.so.0.0.0:/opt/host/cxi/lib/libcxiutils.so.0.0.0 \
    --bind /usr/lib64/libhwloc.so.15:/opt/host/cxi/lib/libhwloc.so.15 \
    --bind /usr/lib64/libefa.so.1:/opt/host/cxi/lib/libefa.so.1 \
    --bind /usr/lib64/libibverbs.so.1:/opt/host/cxi/lib/libibverbs.so.1 \
    --bind /usr/lib64/libxpmem.so.0:/opt/host/cxi/lib/libxpmem.so.0 \
    --bind /usr/lib64/liburing.so.2:/opt/host/cxi/lib/liburing.so.2 \
    --bind /usr/lib64/libnl-route-3.so.200:/opt/host/cxi/lib/libnl-route-3.so.200 \
    --bind /usr/lib64/libnl-3.so.200:/opt/host/cxi/lib/libnl-3.so.200 \
    --bind /software/sse2/el9_gh200/manual/AWS_OFI_NCCL/1.19.2/g14/cu13.0/mp43/hpc1 \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/tmp:/tmp \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/container_outputs:/opt/outputs \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/cluster_transfer:/opt/cluster_transfer \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/cluster_sync:/opt/cluster_sync \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/data:/opt/data \
    --bind \$CCACHE:\$CCACHE \
    --bind \$HF_HOST:/opt/huggingface \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/github_repos/MARL-LM:/opt/MARL-LM \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/github_repos/social_deduction_llm:/opt/sh_finetuning \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/runs:/opt/MARL-LM/runs \
    --bind /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/runs:/opt/runs \
    --home /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/cc/apptainer-home:/home/\$USER \
    --env PYTHONPATH=/opt/MARL-LM \
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
    --env LD_LIBRARY_PATH=/opt/host/cxi/lib:/software/sse2/el9_gh200/manual/AWS_OFI_NCCL/1.19.2/g14/cu13.0/mp43/hpc1/lib \
    --env NCCL_PROTO=^LL128 \
    --env NCCL_CROSS_NIC=1 \
    --env NCCL_GDRCOPY_ENABLE=1 \
    --env NCCL_GDR_LEVEL=PHB \
    --env NCCL_MIN_CTAS=4 \
    --env NCCL_SOCKET_NTHREADS=1 \
    --env NCCL_DEBUG=VERSION \
    --env FI_CXI_DEFAULT_TX_SIZE=1024 \
    --env FI_CXI_HYBRID_PREEMPTIVE=1 \
    --env FI_CXI_HYBRID_RECV_PREEMPTIVE=1048576 \
    --env FI_CXI_RDZV_THRESHOLD=16384 \
    --env FI_CXI_RX_MATCH_MODE=software \
    --env FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16384 \
    --env WANDB_ENTITY=chm-hci \
    --env WANDB_PROJECT={project_name} \
    --env WANDB_RUN_GROUP={experiment_name} \
    --env WANDB_NAME={job_id} \
    --env JOB_DIR={job_dir} \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env SLINGSHOT_VNIS=\$SLINGSHOT_VNIS \
    --env SLINGSHOT_DEVICES=\$SLINGSHOT_DEVICES \
    --env SLINGSHOT_SVC_IDS=\$SLINGSHOT_SVC_IDS \
    --env SLINGSHOT_TCS=\$SLINGSHOT_TCS \
    --env SLURM_PROCID=\$SLURM_PROCID \
    --env SLURM_NODEID=\$SLURM_NODEID \
    --env SLURM_LOCALID=\$SLURM_LOCALID \
    --env SLURM_NNODES=\$SLURM_NNODES \
    --env SLURM_JOBID=\$SLURM_JOBID \
    --env CUDA_VISIBLE_DEVICES={cuda_visible_devices} \
    --env-file /nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik/private/cc_secrets \
    --pwd /opt/MARL-LM \
    {image} \
    bash -c \"\$CMD\""
