"""Generate Pattern-A vLLM sidecar startup blocks for slurm templates."""

import json
import os
import shlex
from typing import List, Tuple

_DAIS_VLLM_IMAGE = (
    "/dais/fs/scratch/ykeller/containers/apptainer/vllm-openai_latest.sif"
)
_DAIS_VLLM_LOG_ROOT = "/dais/fs/scratch/ykeller/tmp/vllm_logs"
_RAVEN_VLLM_IMAGE = "/raven/ptmp/ykeller/cc/containers/apptainer/vllm-openai_latest.sif"
_RAVEN_VLLM_LOG_ROOT = "/ptmp/ykeller/cc/tmp/vllm_logs"
_ARRHENIUS_PERSONAL = "/nobackup/proj/disk/ehpc-dev-2026d09-182/personal/yannik"
_ARRHENIUS_VLLM_IMAGE = (
    f"{_ARRHENIUS_PERSONAL}/containers/apptainer/vllm-openai_latest.sif"
)
_ARRHENIUS_VLLM_LOG_ROOT = f"{_ARRHENIUS_PERSONAL}/cc/tmp/vllm_logs"

# Backwards-compatible aliases used by existing tests.
VLLM_IMAGE = _DAIS_VLLM_IMAGE
VLLM_LOG_ROOT = _DAIS_VLLM_LOG_ROOT

# Match the HuggingFace bind + env wiring from run_slurm_*_apptainer*.sh so
# vLLM resolves the same cached models as the main locked-down container.
_HF_BINDS_AND_ENVS = (
    "--bind $HF_HOST:/opt/huggingface "
    "--env HF_HOME=/opt/huggingface "
    "--env HF_HUB_CACHE=/opt/huggingface/hub "
    "--env HF_XET_CACHE=/opt/huggingface/xet "
    "--env TRANSFORMERS_CACHE=/opt/huggingface/transformers "
    "--env HF_DATASETS_CACHE=/opt/huggingface/datasets "
)

_SOCIAL_DEDUCTION_BINDS = (
    "--bind /dev/infiniband:/dev/infiniband "
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
    "--bind /dev/infiniband:/dev/infiniband "
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

# silicon_sampling has no project-specific model trees; HF cache + generic
# DAIS scratch binds are enough for vLLM sidecars.
_SILICON_SAMPLING_BINDS = (
    "--bind /dev/infiniband:/dev/infiniband "
    "--bind /dais:/dais "
    "--bind /dais/fs/scratch/ykeller/tmp:/tmp "
    "--bind $CCACHE:$CCACHE " + _HF_BINDS_AND_ENVS + "--env TMPDIR=/tmp "
    "--env TRITON_CACHE_DIR=$CCACHE/triton "
    "--env TORCH_EXTENSIONS_DIR=$CCACHE/torch_ext "
    "--env FLASHINFER_WORKSPACE_BASE=$CCACHE/flashinfer "
    "--env VLLM_NO_USAGE_STATS=1 "
)

_MARLLM_RAVEN_BINDS = (
    "--bind /ptmp/ykeller/cc/tmp:/tmp "
    "--bind $CCACHE:$CCACHE " + _HF_BINDS_AND_ENVS + "--env TMPDIR=/tmp "
    "--env TRITON_CACHE_DIR=$CCACHE/triton "
    "--env TORCH_EXTENSIONS_DIR=$CCACHE/torch_ext "
    "--env FLASHINFER_WORKSPACE_BASE=$CCACHE/flashinfer "
    "--env VLLM_NO_USAGE_STATS=1 "
)

_MARLLM_ARRHENIUS_BINDS = (
    "--bind /nobackup:/nobackup "
    f"--bind {_ARRHENIUS_PERSONAL}/cc/tmp:/tmp "
    "--bind $CCACHE:$CCACHE " + _HF_BINDS_AND_ENVS + "--env TMPDIR=/tmp "
    "--env TRITON_CACHE_DIR=$CCACHE/triton "
    "--env TORCH_EXTENSIONS_DIR=$CCACHE/torch_ext "
    "--env FLASHINFER_WORKSPACE_BASE=$CCACHE/flashinfer "
    "--env VLLM_NO_USAGE_STATS=1 "
)

# MARLLM on DAIS has no project-specific model trees; HF cache + generic
# DAIS scratch binds match silicon_sampling.
_MARLLM_DAIS_BINDS = _SILICON_SAMPLING_BINDS

# Backwards-compatible alias used by existing tests / callers that assume raven.
_MARLLM_BINDS = _MARLLM_RAVEN_BINDS


def _resolve_cluster(cc_project: str, cluster: str = "") -> str:
    """Infer the vLLM sidecar cluster when ``--cluster`` was omitted.

    MARLLM historically ran only on raven; every other CC project that
    uses Pattern-A sidecars runs on DAIS. Explicit ``cluster`` always wins.
    """
    if cluster:
        return cluster
    if cc_project == "MARLLM":
        return "raven"
    return "dais"


def _apptainer_binds(cc_project: str, cluster: str = "") -> str:
    cluster = _resolve_cluster(cc_project, cluster)
    if cc_project == "coopbot":
        return _COOPBOT_BINDS
    if cc_project == "social_deduction":
        return _SOCIAL_DEDUCTION_BINDS
    if cc_project == "silicon_sampling":
        return _SILICON_SAMPLING_BINDS
    if cc_project == "MARLLM":
        if cluster == "dais":
            return _MARLLM_DAIS_BINDS
        if cluster == "arrhenius":
            return _MARLLM_ARRHENIUS_BINDS
        return _MARLLM_RAVEN_BINDS
    raise ValueError(f"unsupported cc_project for vLLM setup: {cc_project!r}")


def _vllm_image(cc_project: str, cluster: str = "") -> str:
    cluster = _resolve_cluster(cc_project, cluster)
    if cc_project not in (
        "MARLLM",
        "social_deduction",
        "coopbot",
        "silicon_sampling",
    ):
        raise ValueError(f"unsupported cc_project for vLLM setup: {cc_project!r}")
    if cluster == "raven":
        return _RAVEN_VLLM_IMAGE
    if cluster == "arrhenius":
        return _ARRHENIUS_VLLM_IMAGE
    return _DAIS_VLLM_IMAGE


def _vllm_log_root(cc_project: str, cluster: str = "") -> str:
    cluster = _resolve_cluster(cc_project, cluster)
    if cc_project not in (
        "MARLLM",
        "social_deduction",
        "coopbot",
        "silicon_sampling",
    ):
        raise ValueError(f"unsupported cc_project for vLLM setup: {cc_project!r}")
    if cluster == "raven":
        return _RAVEN_VLLM_LOG_ROOT
    if cluster == "arrhenius":
        return _ARRHENIUS_VLLM_LOG_ROOT
    return _DAIS_VLLM_LOG_ROOT


def _format_gpu_ids(gpu_ids: List[int]) -> str:
    return ",".join(str(gpu_id) for gpu_id in gpu_ids)


def _format_gpu_memory_utilization(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _all_gpu_ids(n_gpu: int) -> str:
    return _format_gpu_ids(list(range(n_gpu)))


def build_vllm_setup(
    vllm_servers_json: str,
    job_dir: str,
    n_gpu: int,
    cc_project: str,
    cluster: str = "",
) -> Tuple[str, str]:
    """Return (vllm_setup_block, main_cuda_visible_devices) for the template.

    ``main_cuda_visible_devices`` is always a comma-separated list of GPU ids
    visible to the main command: all allocated GPUs when there are no sidecars,
    otherwise the GPUs not claimed by any sidecar.
    """
    if not vllm_servers_json:
        return "", _all_gpu_ids(n_gpu)

    servers = json.loads(vllm_servers_json)
    if not isinstance(servers, list) or not servers:
        return "", _all_gpu_ids(n_gpu)

    binds = _apptainer_binds(cc_project, cluster)
    vllm_image = _vllm_image(cc_project, cluster)
    job_id = os.path.basename(job_dir.rstrip("/"))
    vllm_log_dir = f"{_vllm_log_root(cc_project, cluster)}/{job_id}"
    container_vllm_log_dir = f"/tmp/vllm_logs/{job_id}"
    lines = [
        f"VLLM_LOG_DIR={shlex.quote(vllm_log_dir)}",
        'mkdir -p "$VLLM_LOG_DIR"',
        (
            'echo "vLLM logs on host: $VLLM_LOG_DIR '
            f'(in main container: {container_vllm_log_dir})"'
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
        max_model_length = int(server.get("max_model_length", 16192))
        reasoning_parser = server.get("reasoning_parser")

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
            str(max_model_length),
            "--uvicorn-log-level",
            "warning",
        ]
        if enforce_eager:
            vllm_args.append("--enforce-eager")
        if enable_prefix_caching:
            vllm_args.append("--enable-prefix-caching")
        if reasoning_parser:
            vllm_args.extend(["--reasoning-parser", shlex.quote(reasoning_parser)])

        lines.extend(
            [
                f'echo "Starting vLLM on GPUs {_format_gpu_ids(gpu_ids)} port {port}"',
                (
                    f"CUDA_VISIBLE_DEVICES={_format_gpu_ids(gpu_ids)} apptainer exec "
                    f"--nv --contain --cleanenv {binds}"
                    f"{shlex.quote(vllm_image)} "
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
