import os
import argparse

redos_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "redos")


def redo(oldness, execute=False):
    files = [os.path.join(redos_path, f) for f in os.listdir(redos_path)]
    files.sort(key=lambda x: -os.path.getctime(x))

    with open(files[oldness], "r") as file:
        contents = file.read()

    if execute:
        print(contents)
        log_folder = os.path.dirname(contents)
        existing_logfiles = [
            f
            for f in os.listdir(log_folder)
            if f.startswith("slurm-") and f.endswith(".out")
        ]
        redo_name = os.path.join(log_folder, f"slurm-redo-{len(existing_logfiles)}.out")
        os.system(f"sh {contents} > {redo_name} 2>&1")

    else:
        print(
            f"Your redo file is at {contents}. Edit it as you please and then run it using 'redo --execute.'"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oldness", type=int, default=0, help="0 for newest run")
    parser.add_argument("--execute", action="store_true", help="execute the redo")
    args = parser.parse_args()

    redo(args.oldness, args.execute)


if __name__ == "__main__":
    main()
