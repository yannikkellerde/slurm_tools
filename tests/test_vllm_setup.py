from slurm_tools.vllm_setup import VLLM_IMAGE, VLLM_LOG_ROOT, build_vllm_setup


def test_build_vllm_setup_generates_pattern_a_block():
    servers = [
        {
            "gpu_ids": [7],
            "port": 8001,
            "model": "Qwen/Qwen2.5-3B",
            "gpu_memory_utilization": 0.92,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
        }
    ]
    setup, main_gpus = build_vllm_setup(
        vllm_servers_json=__import__("json").dumps(servers),
        job_dir="/tmp/job_dir/job_123",
        n_gpu=8,
        cc_project="social_deduction",
    )
    assert "trap cleanup_vllm EXIT" in setup
    assert f"VLLM_LOG_DIR={VLLM_LOG_ROOT}/job_123" in setup
    assert 'mkdir -p "$VLLM_LOG_DIR"' in setup
    assert "/tmp/vllm_logs/job_123" in setup
    assert "CUDA_VISIBLE_DEVICES=7 apptainer exec" in setup
    assert VLLM_IMAGE in setup
    assert "--bind $HF_HOST:/opt/huggingface" in setup
    assert "--env HF_HOME=/opt/huggingface" in setup
    assert "--env HF_HUB_CACHE=/opt/huggingface/hub" in setup
    assert "--env TRANSFORMERS_CACHE=/opt/huggingface/transformers" in setup
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1" in setup
    assert "--enforce-eager" in setup
    assert "--enable-prefix-caching" not in setup
    assert "--max-model-len 16192" in setup
    assert '> "$VLLM_LOG_DIR/vllm_8001.log" 2>&1 &' in setup
    assert "curl -sf \"http://127.0.0.1:${port}/health\"" in setup
    assert main_gpus == "0,1,2,3,4,5,6"
    assert "--reasoning-parser" not in setup


def test_build_vllm_setup_passes_reasoning_parser():
    servers = [
        {
            "gpu_ids": [7],
            "port": 8001,
            "model": "openai/gpt-oss-20b",
            "gpu_memory_utilization": 0.92,
            "enforce_eager": False,
            "enable_prefix_caching": False,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "reasoning_parser": "openai_gptoss",
        }
    ]
    setup, _ = build_vllm_setup(
        vllm_servers_json=__import__("json").dumps(servers),
        job_dir="/tmp/job_dir/job_123",
        n_gpu=8,
        cc_project="social_deduction",
    )
    assert "--reasoning-parser openai_gptoss" in setup


def test_build_vllm_setup_passes_max_model_length():
    servers = [
        {
            "gpu_ids": [7],
            "port": 8001,
            "model": "Qwen/Qwen2.5-3B",
            "gpu_memory_utilization": 0.92,
            "enforce_eager": False,
            "enable_prefix_caching": False,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "max_model_length": 8192,
        }
    ]
    setup, _ = build_vllm_setup(
        vllm_servers_json=__import__("json").dumps(servers),
        job_dir="/tmp/job_dir/job_123",
        n_gpu=8,
        cc_project="social_deduction",
    )
    assert "--max-model-len 8192" in setup
    assert "--max-model-len 16192" not in setup


def test_build_vllm_setup_empty_json_returns_empty_block():
    setup, main_gpus = build_vllm_setup("", "/tmp/job_dir", 8, "social_deduction")
    assert setup == ""
    assert main_gpus == ""
