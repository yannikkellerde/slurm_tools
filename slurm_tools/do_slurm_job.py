import argparse
from argparse import Namespace
import os
import subprocess
from datetime import datetime

basedir = os.path.dirname(os.path.abspath(__file__))


def generate_local_job_id():
    """
    Generates a local job ID based on timestamp.
    """
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def slurm_job(
    n_gpu,
    time,
    template_file,
    run_group,
    program_call,
    image,
    launcher,
    acc_config,
    n_nodes,
    dry,
    conda_env,
    keepalive,
    project,
    **_kwargs,
):
    if n_gpu == 1 and n_nodes == 1:
        if launcher in ["accelerate", "torchrun"]:
            print(
                "WARNING: You are using a single GPU and a single node, but are requesting a distributed launcher. This is likely a mistake."
            )
            print("Switching launcher to python")
            launcher = "python"

    if launcher == "accelerate":
        NUM_PROCESSES = n_nodes * n_gpu

        # so processes know who to talk to
        MASTER_ADDR = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"
        MASTER_PORT = 6000
        distribute = f"""accelerate launch \
--config_file {acc_config} \
--main_process_ip {MASTER_ADDR} \
--main_process_port {MASTER_PORT} \
--machine_rank \$SLURM_PROCID \
--num_processes {NUM_PROCESSES} \
--num_machines {n_nodes}"""
    elif launcher == "torchrun":
        distribute = f"python -m torch.distributed.run --nnodes=$SLURM_NNODES --nproc-per-node={n_gpu} --rdzv-id=$SLURM_JOBID --rdzv-endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) --rdzv-backend=c10d"
    else:
        distribute = launcher

    if keepalive > 0:
        afterwards = f";sleep {keepalive};while cat keepalive; do sleep 5;done"
    else:
        afterwards = ""

    job_id = generate_local_job_id()
    partition = "gpu"
    dest_dir = (
        "/root/runs"
        if "apptainer" in template_file
        else os.path.join(os.environ["HOME"], "runs")
    )
    job_specific_dir = os.path.join(dest_dir, run_group, job_id)
    os.makedirs(job_specific_dir, exist_ok=True)

    n_cpu = n_gpu * 18

    format_dict = dict(
        job_dir=job_specific_dir,
        job_id=job_id,
        project_name=project,
        experiment_name=run_group,
        n_gpu=n_gpu,
        n_cpu=n_cpu,
        time=time,
        program_call=program_call,
        image=image,
        partition=partition,
        n_nodes=n_nodes,
        distribute=distribute,
        conda_env=conda_env,
        project_root=os.path.basename(os.path.dirname(".")),
        afterwards=afterwards,
    )

    with open(template_file, "r") as file:
        script = file.read().format(**format_dict)

    output_path = os.path.join(job_specific_dir, "slurm_script.sh")
    with open(output_path, "w") as file:
        file.write(script)

    if not dry:
        subprocess.run(["sbatch", output_path])


def obtain_parser():
    parser = argparse.ArgumentParser()
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
        "--template_file", type=str, default=os.path.join(basedir, "run_slurm_conda.sh")
    )
    parser.add_argument("--run_group", type=str, default="test_runs")
    parser.add_argument(
        "--program_call",
        type=str,
        required=True,
        help="Script to run.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.expanduser("~/images/sh_finetune.sif"),
        help="Apptainer image to use",
    )
    parser.add_argument("--launcher", type=str, default="python")
    parser.add_argument(
        "--acc_config", type=str, default="config/accelerate/acc_config_DDP.yml"
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument("--conda_env", type=str, default=None, help="Env to activate")
    parser.add_argument(
        "--keepalive",
        type=int,
        default=0,
        help="Keep job alive after it finished for x seconds",
    )
    parser.add_argument("--project", type=str, default="slurm_project")

    return parser


def main():
    parser = obtain_parser()
    args = parser.parse_args()
    slurm_job(**vars(args))


if __name__ == "__main__":
    main()
