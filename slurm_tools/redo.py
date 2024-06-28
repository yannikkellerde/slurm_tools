import os
import argparse

redos_path = os.path.join(os.path.abspath(__file__), "redos")


def redo(oldness, execute=False):
    files = [os.path.join(redos_path, f) for f in os.listdir(redos_path)]
    files.sort(key=lambda x: -os.path.getctime(x))

    with open(files[oldness], "r") as file:
        contents = file.read()
        print(contents)

    if execute:
        os.system(f"sh {contents}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oldness", type=int, default=0, help="0 for newest run")
    parser.add_argument("--execute", action="store_true", help="execute the redo")
    args = parser.parse_args()

    redo(args.oldness, args.execute)


if __name__ == "__main__":
    main()
