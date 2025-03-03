import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="watch it")
    parser.add_argument("--watchtime", type=int, default=1, help="seconds")
    parser.add_argument("--oldness", type=int, default=0, help="0 for newest run")
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.environ["HOME"], "runs"),
        help="where do they be areing",
    )
    parser.add_argument("--script", action="store_true")
    parser.add_argument("--server_log", type=int, default=None)
    parser.add_argument("--gpu_log", type=int, default=None)
    parser.add_argument("--command_log", type=int, default=None)
    args = parser.parse_args()

    assert (
        sum(
            [
                args.script,
                args.server_log is not None,
                args.gpu_log is not None,
                args.command_log is not None,
            ]
        )
        <= 1
    )

    def predicate(n):
        if args.script:
            return n == "slurm_script.sh"
        if args.server_log is not None:
            return n.startswith("server.log") and n.endswith(f".{args.server_log}")
        if args.gpu_log is not None:
            return n.startswith("gpus") and n.endswith(f"{args.gpu_log}.log")
        if args.command_log is not None:
            return n.startswith("command") and n.endswith(f"{args.command_log}.log")
        return n.startswith("slurm-") and n.endswith(".out")

    files = sum(
        [
            [os.path.join(p, n) for n in f if predicate(n)]
            for p, s, f in os.walk(args.path)
        ],
        [],
    )
    files.sort(key=lambda x: -os.path.getmtime(x))
    print(files[args.oldness])
    if args.watch:
        os.system(f"watch --color -n {args.watchtime} tac {files[args.oldness]}")
    else:
        os.system(f"cat {files[args.oldness]}")


if __name__ == "__main__":
    main()
