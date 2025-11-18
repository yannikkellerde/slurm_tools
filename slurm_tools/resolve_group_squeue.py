#!/usr/bin/env python3
import subprocess
import sys
import pwd
import grp


def get_group_for_user(username: str) -> str:
    try:
        pw = pwd.getpwnam(username)
        gr = grp.getgrgid(pw.pw_gid)
        return gr.gr_name
    except KeyError:
        return "UNKNOWN"


def split_columns(line: str) -> list:
    """Split a line into columns on whitespace."""
    return line.rstrip("\n").split()


def format_row(cols, widths):
    """Pad each column to its assigned width."""
    return " ".join(col.ljust(widths[i]) for i, col in enumerate(cols))


def main():
    squeue_cmd = ["squeue"] + sys.argv[1:]

    try:
        result = subprocess.run(
            squeue_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr)
        sys.exit(e.returncode)

    lines = result.stdout.splitlines()
    if not lines:
        return

    # Parse header
    header_cols = split_columns(lines[0])

    # Locate USER column
    try:
        user_idx = header_cols.index("USER")
    except ValueError:
        print(result.stdout, end="")
        return

    # Insert GROUP column
    group_idx = user_idx + 1
    header_cols.insert(group_idx, "GROUP")

    # Parse all data rows
    rows = [header_cols]
    for line in lines[1:]:
        cols = split_columns(line)
        if len(cols) <= user_idx:
            rows.append(cols)  # malformed line, keep as-is
            continue

        user = cols[user_idx]
        group = get_group_for_user(user)

        cols.insert(group_idx, group)
        rows.append(cols)

    # Determine column widths
    num_cols = max(len(r) for r in rows)
    widths = [0] * num_cols

    for r in rows:
        for i, col in enumerate(r):
            widths[i] = max(widths[i], len(col))

    # Print aligned output
    for r in rows:
        print(format_row(r, widths))


if __name__ == "__main__":
    main()
