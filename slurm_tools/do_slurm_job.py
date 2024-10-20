import argparse
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
    n_gpu: int,
    time: str,
    template_file: str,
    run_group: str,
    program_call: str,
    image: str,
    launcher: str,
    acc_config: str,
    n_nodes: int,
    dry: bool,
    conda_env: str,
    keepalive: bool,
    project: str,
    cinit: str,
    mem: str,
    dependencies: list[int],
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

    if keepalive:
        afterwards = f";while true;do sleep 5;pgrep -U $USER python >/dev/null && continue;sleep {keepalive};pgrep -U $USER python >/dev/null || break;done"
    else:
        afterwards = ""

    if program_call.startswith("python "):
        print("No need to include the launcher (python) in the program call.")
        program_call = program_call.removeprefix("python ")
    job_id = generate_local_job_id()
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
        n_nodes=n_nodes,
        distribute=distribute,
        conda_env=conda_env,
        project_root=os.path.basename(os.path.dirname(".")),
        afterwards=afterwards,
        cinit=cinit,
        mem=mem,
    )

    if keepalive:
        redos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "redos")
        os.makedirs(redos_path, exist_ok=True)
        with open(os.path.join(redos_path, f"{job_id}"), "w") as file:
            file.write(f"{os.path.join(job_specific_dir, 'redo.bash')}")

        redos_files = [os.path.join(redos_path, f) for f in os.listdir(redos_path)]
        if len(redos_files) > 10:
            oldest_file = min(redos_files, key=os.path.getctime)
            os.remove(oldest_file)

        with open(os.path.join(job_specific_dir, "redo.bash"), "w") as file:
            file.write(
                f"cd {os.getcwd()};source ~/.condasetup_bash;conda activate {conda_env};{distribute} -u {program_call}"
            )

    with open(template_file, "r") as file:
        script = file.read().format(**format_dict)

    output_path = os.path.join(job_specific_dir, "slurm_script.sh")
    with open(output_path, "w") as file:
        file.write(script)

    if not dry:
        if dependencies:
            subprocess.run(
                [
                    "sbatch",
                    "--dependency=afterok:" + ":".join(map(str, dependencies)),
                    output_path,
                ]
            )
        else:
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
    parser.add_argument("--mem", type=str, default="0")
    parser.add_argument(
        "--cinit", type=str, default=os.path.join(basedir, ".condasetup_bash")
    )
    parser.add_argument(
        "--dependencies", type=int, nargs="+", help="List of dependent jobs."
    )

    return parser


def main():
    parser = obtain_parser()
    args = parser.parse_args()
    slurm_job(**vars(args))


if __name__ == "__main__":
    main()
