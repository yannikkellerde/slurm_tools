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
    args = parser.parse_args()

    files = sum(
        [
            [
                os.path.join(p, n)
                for n in f
                if (args.script and n == "slurm_script.sh")
                or (not args.script and n.startswith("slurm-") and n.endswith(".out"))
            ]
            for p, s, f in os.walk(args.path)
        ],
        [],
    )
    files.sort(key=lambda x: -os.path.getmtime(x))
    print(files[args.oldness])
    if args.watch:
        os.system(f"watch -n {args.watchtime} tac {files[args.oldness]}")
    else:
        os.system(f"cat {files[args.oldness]}")


if __name__ == "__main__":
    main()
