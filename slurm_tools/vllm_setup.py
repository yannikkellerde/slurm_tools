"""Generate Pattern-A vLLM sidecar startup blocks for DAIS slurm templates."""

import json
import os
import shlex
from typing import List, Tuple

VLLM_IMAGE = "/dais/fs/scratch/ykeller/containers/apptainer/vllm-openai_latest.sif"
VLLM_LOG_ROOT = "/dais/fs/scratch/ykeller/tmp/vllm_logs"

# Match the HuggingFace bind + env wiring from run_slurm_dais_apptainer*.sh so
# vLLM resolves the same cached models as sh_finetuning_fixed.
_HF_BINDS_AND_ENVS = (
    "--bind $HF_HOST:/opt/huggingface "
    "--env HF_HOME=/opt/huggingface "
    "--env HF_HUB_CACHE=/opt/huggingface/hub "
    "--env HF_XET_CACHE=/opt/huggingface/xet "
    "--env TRANSFORMERS_CACHE=/opt/huggingface/transformers "
    "--env HF_DATASETS_CACHE=/opt/huggingface/datasets "
)

_SOCIAL_DEDUCTION_BINDS = (
    "--bind /dais:/dais "
    "--bind /dais/fs/scratch/ykeller/tmp:/tmp "
    "--bind $CCACHE:$CCACHE "
    "--bind /dais/fs/scratch/ykeller/models/social_deduction:/opt/models "
    "--bind /u/ykeller/models_permanent/social_deduction:/opt/models_ro:ro "
    + _HF_BINDS_AND_ENVS
    + "--env TMPDIR=/tmp "
    "--env TRITON_CACHE_DIR=$CCACHE/triton "
    "--env TORCH_EXTENSIONS_DIR=$CCACHE/torch_ext "
    "--env FLASHINFER_WORKSPACE_BASE=$CCACHE/flashinfer "
    "--env VLLM_NO_USAGE_STATS=1 "
)

_COOPBOT_BINDS = (
    "--bind /dais:/dais "
    "--bind /dais/fs/scratch/ykeller/tmp:/tmp "
    "--bind $CCACHE:$CCACHE "
    "--bind /dais/fs/scratch/ykeller/models/coopbot:/opt/models "
    "--bind /u/ykeller/models_permanent/coopbot:/opt/models_ro:ro "
    + _HF_BINDS_AND_ENVS
    + "--env TMPDIR=/tmp "
    "--env TRITON_CACHE_DIR=$CCACHE/triton "
    "--env TORCH_EXTENSIONS_DIR=$CCACHE/torch_ext "
    "--env FLASHINFER_WORKSPACE_BASE=$CCACHE/flashinfer "
    "--env VLLM_NO_USAGE_STATS=1 "
)


def _apptainer_binds(cc_project: str) -> str:
    if cc_project == "coopbot":
        return _COOPBOT_BINDS
    if cc_project == "social_deduction":
        return _SOCIAL_DEDUCTION_BINDS
    raise ValueError(f"unsupported cc_project for vLLM setup: {cc_project!r}")


def _format_gpu_ids(gpu_ids: List[int]) -> str:
    return ",".join(str(gpu_id) for gpu_id in gpu_ids)


def _format_gpu_memory_utilization(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def build_vllm_setup(
    vllm_servers_json: str,
    job_dir: str,
    n_gpu: int,
    cc_project: str,
) -> Tuple[str, str]:
    """Return (vllm_setup_block, main_cuda_visible_devices) for the template."""
    if not vllm_servers_json:
        return "", ""

    servers = json.loads(vllm_servers_json)
    if not isinstance(servers, list) or not servers:
        return "", ""

    binds = _apptainer_binds(cc_project)
    job_id = os.path.basename(job_dir.rstrip("/"))
    vllm_log_dir = f"{VLLM_LOG_ROOT}/{job_id}"
    container_vllm_log_dir = f"/tmp/vllm_logs/{job_id}"
    lines = [
        f"VLLM_LOG_DIR={shlex.quote(vllm_log_dir)}",
        'mkdir -p "$VLLM_LOG_DIR"',
        (
            'echo "vLLM logs on host: $VLLM_LOG_DIR '
            f'(in sh_finetuning_fixed container: {container_vllm_log_dir})"'
        ),
        "VLLM_PIDS=()",
        "cleanup_vllm() {",
        '  if ((${#VLLM_PIDS[@]} > 0)); then kill "${VLLM_PIDS[@]}" 2>/dev/null; fi',
        "}",
        "trap cleanup_vllm EXIT",
        "",
    ]

    used_gpus = set()
    ports: List[int] = []

    for server in servers:
        gpu_ids = server["gpu_ids"]
        port = int(server["port"])
        model = server["model"]
        gpu_memory_utilization = _format_gpu_memory_utilization(
            float(server["gpu_memory_utilization"])
        )
        tensor_parallel_size = int(server["tensor_parallel_size"])
        data_parallel_size = int(server["data_parallel_size"])
        enforce_eager = bool(server["enforce_eager"])
        enable_prefix_caching = bool(server["enable_prefix_caching"])

        for gpu_id in gpu_ids:
            used_gpus.add(int(gpu_id))
        ports.append(port)

        vllm_args = [
            "vllm",
            "serve",
            shlex.quote(model),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--data-parallel-size",
            str(data_parallel_size),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            gpu_memory_utilization,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "16192",
            "--uvicorn-log-level",
            "warning",
        ]
        if enforce_eager:
            vllm_args.append("--enforce-eager")
        if enable_prefix_caching:
            vllm_args.append("--enable-prefix-caching")

        lines.extend(
            [
                f"echo \"Starting vLLM on GPUs {_format_gpu_ids(gpu_ids)} port {port}\"",
                (
                    f"CUDA_VISIBLE_DEVICES={_format_gpu_ids(gpu_ids)} apptainer exec "
                    f"--nv --contain --cleanenv {binds}"
                    f"{shlex.quote(VLLM_IMAGE)} "
                    "env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 "
                    + " ".join(vllm_args)
                    + ' > "$VLLM_LOG_DIR/vllm_'
                    + str(port)
                    + '.log" 2>&1 &'
                ),
                "VLLM_PIDS+=($!)",
                "",
            ]
        )

    port_checks = " ".join(str(port) for port in ports)
    lines.extend(
        [
            f"for port in {port_checks}; do",
            '  echo "Waiting for vLLM on port ${port}..."',
            '  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do sleep 2; done',
            '  echo "vLLM ready on port ${port}"',
            "done",
            "",
        ]
    )

    main_gpus = [str(gpu_id) for gpu_id in range(n_gpu) if gpu_id not in used_gpus]
    main_cuda_visible_devices = ",".join(main_gpus)
    return "\n".join(lines), main_cuda_visible_devices
