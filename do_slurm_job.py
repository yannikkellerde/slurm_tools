import argparse
import os
import subprocess
from datetime import datetime


def generate_local_job_id():
    """
    Generates a local job ID based on timestamp.
    """
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def main():
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
        "--template_file", type=str, default="scripts/slurm/run_slurm.sh"
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
    parser.add_argument("--launcher", type=str, default="accelerate")
    parser.add_argument(
        "--acc_config", type=str, default="config/accelerate/acc_config_DDP.yml"
    )
    parser.add_argument("--n_nodes", type=int, default=1, help="Number of Nodes")

    args = parser.parse_args()

    if args.n_gpu > 1 or args.n_nodes > 1:
        NUM_PROCESSES = args.n_nodes * args.n_gpu

        # so processes know who to talk to
        MASTER_ADDR = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"
        MASTER_PORT = 6000
        if args.launcher == "accelerate":
            distribute = f"""accelerate launch \
--config_file {args.acc_config} \
--main_process_ip {MASTER_ADDR} \
--main_process_port {MASTER_PORT} \
--machine_rank \$SLURM_PROCID \
--num_processes {NUM_PROCESSES} \
--num_machines {args.n_nodes}"""
        elif args.launcher == "torchrun":
            distribute = f"python -m torch.distributed.run --nnodes=$SLURM_NNODES --nproc-per-node={args.n_gpu} --rdzv-id=$SLURM_JOBID --rdzv-endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) --rdzv-backend=c10d"
        else:
            raise ValueError("Unknown launcher", args.launcher)
    else:
        distribute = "python"

    job_id = generate_local_job_id()
    partition = "gpu"
    dest_dir = (
        "/root/runs"
        if "apptainer" in args.template_file
        else os.path.join(os.environ["HOME"], "runs")
    )
    job_specific_dir = os.path.join(dest_dir, args.run_group, job_id)
    os.makedirs(job_specific_dir, exist_ok=True)

    n_cpu = args.n_gpu * 18

    with open(args.template_file, "r") as file:
        script = file.read().format(
            job_dir=job_specific_dir,
            job_id=job_id,
            project_name="sh_finetune",
            experiment_name=args.run_group,
            n_gpu=args.n_gpu,
            n_cpu=n_cpu,
            time=args.time,
            program_call=args.program_call,
            image=args.image,
            partition=partition,
            n_nodes=args.n_nodes,
            distribute=distribute,
        )

    output_path = os.path.join(job_specific_dir, "slurm_script.sh")
    with open(output_path, "w") as file:
        file.write(script)

    if not args.dry:
        subprocess.run(["sbatch", output_path])


if __name__ == "__main__":
    main()
