import argparse
import subprocess
import os
from datetime import datetime

basedir = os.path.dirname(os.path.abspath(__file__))


def generate_local_job_id():
    """
    Generates a local job ID based on timestamp.
    """
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def slurm_job(
    model,
    command,
    dry,
    time,
    template_file,
    image,
    n_nodes,
    mamba_env,
    experiment_name,
    mamba_setup_path,
    n_gpu,
    sglang_nodes,
    chat_template,
    context_length,
    env_file,
    skip_capture_cuda_graph=False,
):
    tp_size = (sglang_nodes or n_nodes) * n_gpu
    if mamba_env is None:
        conda_activate = ""
    else:
        conda_activate = f"source {os.path.expanduser(mamba_setup_path)} && mamba activate {mamba_env}"

    job_id = generate_local_job_id()
    dest_dir = os.path.join(os.environ["HOME"], "runs")
    job_specific_dir = os.path.join(dest_dir, experiment_name, job_id)
    os.makedirs(job_specific_dir, exist_ok=True)

    if skip_capture_cuda_graph:
        capture_cuda = "--disable-cuda-graph"
    else:
        capture_cuda = ""

    chat_template_arg = f"--chat-template {chat_template}" if chat_template else ""

    context_length_arg = f"--context-length {context_length}" if context_length else ""

    source_env = f"source {os.path.expanduser(env_file)}" if env_file else ""

    with open(template_file, "r") as f:
        script = f.read().format(
            model=model,
            command=command,
            time=time,
            sif_path=os.path.expanduser(image),
            n_nodes=n_nodes,
            job_dir=job_specific_dir,
            job_id=job_id,
            n_gpu=n_gpu,
            tp_size=tp_size,
            conda_activate=conda_activate,
            basepath=basedir,
            experiment_name=experiment_name,
            sglang_nodes=sglang_nodes,
            capture_cuda=capture_cuda,
            chat_template_arg=chat_template_arg,
            context_length_arg=context_length_arg,
            source_env=source_env,
        )

    output_path = os.path.join(job_specific_dir, "slurm_script.sh")
    with open(output_path, "w") as f:
        f.write(script)

    if not dry:
        subprocess.run(["sbatch", output_path])


def obtain_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="Script to run.",
    )
    parser.add_argument(
        "--dry", action="store_true", help="Only create files, do not submit the job."
    )
    parser.add_argument("--n_gpu", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument(
        "--time",
        type=str,
        default="00:10:00",
        help="Expected runtime in HH:MM:SS format.",
    )
    parser.add_argument(
        "--template_file", type=str, default=os.path.join(basedir, "sglang.slurm")
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.expanduser("~/images/sglang_v0.4.1.post4-rocm620.sif"),
        help="Apptainer image to use",
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument(
        "--sglang_nodes", type=int, default=None, help="Number of sglang Nodes"
    )
    parser.add_argument("--skip_capture_cuda_graph", action="store_true")
    parser.add_argument("--mamba_env", type=str, default=None, help="Env to activate")
    parser.add_argument("--experiment_name", type=str, default="sglang")
    parser.add_argument("--mamba_setup_path", type=str, default="~/.mambasetup.bash")
    parser.add_argument("--chat-template", type=str, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--env_file", type=str, default=None)
    return parser


def main():
    parser = obtain_parser()
    args = parser.parse_args()
    slurm_job(**vars(args))


if __name__ == "__main__":
    main()
