import os
from typing import Optional
import argparse
import subprocess


def attach(shell: str = "bash", jobid: Optional[int] = None):
    if jobid is None:
        proc_data = (
            subprocess.check_output(["squeue", "--me"]).decode("utf-8").splitlines()[1:]
        )
        if not proc_data:
            raise RuntimeError("No jobs running.")

        if len(proc_data) > 1:
            print("Multiple jobs running. Choose one:")
            for i, line in enumerate(proc_data):
                print(f"{i}: {' '.join(line.split())}")
            choice = proc_data[int(input())].split()[0]
        else:
            choice = proc_data[0].split()[0]
    else:
        choice = jobid

    os.system(f"srun --pty --overlap --jobid {choice} {shell}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", type=str, default="bash")
    parser.add_argument("--jobid", type=int, default=None)
    args = parser.parse_args()

    attach(args.shell, jobid=args.jobid)
