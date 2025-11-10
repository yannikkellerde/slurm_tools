import os
import json
import requests
import argparse


def notify_slack(message):
    WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]

    payload = {
        "text": message,
    }

    resp = requests.post(
        WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()  # will raise if Slack returns non-2xx
    print("Sent!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--jobid", type=str, default="unknown")
    parser.add_argument("--extra", type=str, default="")
    parser.add_argument("--finished", action="store_true")
    parser.add_argument("--exit_code", type=int, default=0)
    args = parser.parse_args()
    if args.finished:
        message = f"Job {args.jobid} on {args.n_gpu*args.n_nodes} GPUs finished with exit code {args.exit_code}. {args.extra}"
    else:
        message = f"Job {args.jobid} started with {args.n_gpu*args.n_nodes} GPUs. {args.extra}"
    notify_slack(message)
