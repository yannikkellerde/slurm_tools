import argparse
import subprocess
import os
import datetime

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
):
    tp_size = n_nodes * n_gpu
    if mamba_env is None:
        conda_activate = ""
    else:
        conda_activate = f"source {mamba_setup_path} && mamba activate {mamba_env}"

    job_id = generate_local_job_id()
    dest_dir = os.path.join(os.environ["HOME"], "runs")
    job_specific_dir = os.path.join(dest_dir, experiment_name, job_id)

    with open(template_file, "r") as f:
        script = f.read().format(
            model=model,
            command=command,
            time=time,
            sif_path=image,
            n_nodes=n_nodes,
            job_dir=job_specific_dir,
            n_gpu=n_gpu,
            tp_size=tp_size,
            conda_activate=conda_activate,
            basepath=basedir,
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
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use.")
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
    parser.add_argument("--mamba_env", type=str, default=None, help="Env to activate")
    parser.add_argument("--experiment_name", type=str, default="sglang")
    parser.add_argument("--mamba_setup_path", type=str, default="~/.mambasetup.bash")

    return parser
