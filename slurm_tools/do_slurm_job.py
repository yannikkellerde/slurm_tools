import argparse
import os
import subprocess
from datetime import datetime
from slurm_tools.slurm_time_until_start import find_n_gpu, find_n_nodes

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
    n_cpu: int,
    extra_source: str,
    extra_arg: str,
    modules: str,
    **_kwargs,
):
    if n_gpu == 1 and n_nodes == 1:
        if launcher in ["accelerate", "torchrun"]:
            print(
                "WARNING: You are using a single GPU and a single node, but are requesting a distributed launcher. This is likely a mistake."
            )
            # print("Switching launcher to python")
            # launcher = "python"

    if launcher == "accelerate":
        NUM_PROCESSES = n_nodes * n_gpu

        acc_config_arg = f"--config_file {acc_config}" if acc_config else ""
        if n_nodes > 1:
            distribute = (
                f"accelerate launch {acc_config_arg} "
                f"--num_processes {NUM_PROCESSES} "
                f"--num_cpu_threads_per_process 12 "
                f"--num_machines {n_nodes} "
                f"--main_process_ip $MASTER_ADDR "
                f"--main_process_port $MASTER_PORT "
                f"--machine_rank \\$SLURM_PROCID"
            )
        else:
            distribute = f"accelerate launch {acc_config_arg} --num_processes {NUM_PROCESSES} --num_cpu_threads_per_process 12"
        print("distribute command", distribute)
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

    if n_cpu == 0:
        n_cpu = max(1, n_gpu) * 18

    if extra_arg:
        extra_arg = f'--extra "{extra_arg}"'

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
        extra_source="" if extra_source is None else f"source {extra_source}",
        modules=modules,
        extra_arg=extra_arg,
        slurm_tools_path=basedir,
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
                    "--dependency=afterany:" + ":".join(map(str, dependencies)),
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
        "--template_file",
        type=str,
        default=os.path.join(basedir, "run_slurm_dais_modules.sh"),
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
    parser.add_argument("--acc_config", type=str, default=None)
    parser.add_argument("--n_nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument("--conda_env", type=str, default=None, help="Env to activate")
    parser.add_argument(
        "--extra_source", type=str, default=None, help="Extra source to activate"
    )
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
    parser.add_argument("--n_cpu", type=int, default=0, help="Number of CPUs to use.")
    parser.add_argument(
        "--extra_arg",
        type=str,
        default="",
        help="Extra arguments to pass to the notify_slack.py script.",
    )
    parser.add_argument("--modules", type=str, default="", help="Modules to load.")
    parser.add_argument(
        "--compute_time_to_start",
        action="store_true",
        help="Compute the time to start the job.",
    )

    return parser


def main():
    parser = obtain_parser()
    args = parser.parse_args()
    slurm_job(**vars(args))
    if args.compute_time_to_start:
        if args.n_nodes == 1:
            deadline = find_n_gpu(args.n_gpu)
        else:
            deadline = find_n_nodes(args.n_nodes)
        print(f"Estimated time to start the job: {deadline}")


if __name__ == "__main__":
    main()

"""
slurm_job --n_gpu 1 --time 02:00:00 --launcher accelerate --n_nodes 1 --program_call "sh_finetuning/__main__.py train_sft --config_path configs/training_arguments/sft.yml --model_name Qwen/Qwen2.5-3B" --mem 0 --conda_env deepspeed --project sh_finetuning --run_group deepspeed --template_file /u/ykeller/github_repos/slurm_tools/slurm_tools/run_slurm_dais_modules.sh --cinit /u/ykeller/.mamba_init --n_cpu 96 --modules "gcc/14 cuda/12.8" --extra_arg "The test job" --acc_config configs/accelerate/deepspeed.yml
"""
