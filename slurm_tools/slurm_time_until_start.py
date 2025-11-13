#!/usr/bin/env python3
"""
Estimate when SLURM resources (nodes or GPUs) will be free based on current jobs' TIME and TIME_LIMIT.

Two modes (mutually exclusive):
  --n-nodes N    -> earliest time when N whole nodes are completely idle
  --n-gpu   G    -> earliest time when G GPUs are free on **one single node** (not aggregated across nodes)

The script shells out to squeue/sinfo and uses only TIME (%M) and TIME_LIMIT (%l) from squeue.
If the requested resources are already available, it prints that and exits with code 0.

Examples
--------
# Earliest time 3 nodes become completely free on the 'gpu' partition
python slurm_free_time.py --n-nodes 3 --partition gpu

# Earliest time 8 GPUs are free on a single node (typical max per node)
python slurm_free_time.py --n-gpu 8
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

# ---------- Helpers ----------


def run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True)
        return out
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {' '.join(cmd)}\n{e}\n{e.output}", file=sys.stderr)
        sys.exit(2)


_TIME_RE = re.compile(
    r"^(?:(?P<days>\d+)-)?(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?$"
)


def parse_slurm_time(s: str) -> int:
    """Parse SLURM time strings like 'M:SS', 'HH:MM:SS', or 'D-HH:MM:SS' to seconds.
    Returns a large number for 'UNLIMITED'. Empty/invalid -> 0.
    """
    s = (s or "").strip()
    if not s:
        return 0
    if s.upper() in {"UNLIMITED", "INFINITE", "N/A"}:
        return 10**12  # effectively infinite

    # Count colons to disambiguate format
    colon_count = s.count(":")

    m = _TIME_RE.match(s)
    if not m:
        # Some clusters show just '0:00' etc.; fall back robustly
        if s.isdigit():
            return int(s)
        return 0

    d = int(m.group("days") or 0)
    h = int(m.group("h") or 0)
    mi = int(m.group("m") or 0)
    se = int(m.group("s") or 0)

    # If there's only one colon and no days, it's M:SS format (not H:MM)
    if colon_count == 1 and d == 0:
        # Reinterpret: h->minutes, m->seconds
        minutes = h
        seconds = mi
        result = minutes * 60 + seconds
    else:
        # Standard D-HH:MM:SS or HH:MM:SS format
        result = d * 86400 + h * 3600 + mi * 60 + se

    print(f"{s} -> {result}")
    return result


def fmt_duration(seconds: int) -> str:
    if seconds <= 0:
        return "0s"
    mins, s = divmod(seconds, 60)
    hrs, m = divmod(mins, 60)
    days, h = divmod(hrs, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if h or days:
        parts.append(f"{h}h")
    if m or h or days:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# Expand SLURM nodelist patterns like host[1-3,5,8-10]
RANGE_RE = re.compile(r"^(?P<prefix>.*)\[(?P<body>[^\]]+)\]$")
SUBRANGE_RE = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?$")


def expand_nodelist(n: str) -> List[str]:
    n = (n or "").strip()
    if not n or n.startswith("("):  # e.g., (Resources) for PD
        return []
    m = RANGE_RE.match(n)
    if not m:
        return [n]
    prefix = m.group("prefix")
    body = m.group("body")
    out = []
    for part in body.split(","):
        m2 = SUBRANGE_RE.match(part)
        if not m2:
            out.append(prefix + part)
            continue
        a = int(m2.group("start"))
        b = int(m2.group("end") or a)
        width = max(len(m2.group("start")), len(m2.group("end") or m2.group("start")))
        for x in range(a, b + 1):
            out.append(f"{prefix}{x:0{width}d}")
    return out


@dataclass
class Job:
    jobid: str
    state: str
    elapsed_s: int
    limit_s: int
    nodes: List[str]
    gpus_per_node: int  # parsed from TRES_PER_NODE

    @property
    def remaining_s(self) -> int:
        return max(self.limit_s - self.elapsed_s, 0)


def parse_tres_gpu_count(tres: str) -> int:
    """Extract GPU count from TRES_PER_NODE string, e.g., 'gres/gpu:h200:8' -> 8.
    If multiple entries, sums them. If none present, returns 0.
    """
    total = 0
    for tok in (tres or "").split(","):
        tok = tok.strip()
        if tok.startswith("gres/gpu"):
            # Accept forms: gres/gpu:4  OR gres/gpu:h200:8
            parts = tok.split(":")
            try:
                total += int(parts[-1])
            except (ValueError, IndexError):
                pass
    return total


def read_jobs(consider_cg: bool) -> List[Job]:
    fmt = "%i|%t|%M|%l|%N|%b"
    out = run(["squeue", "-h", "-o", fmt])
    jobs: List[Job] = []
    for line in out.strip().splitlines():
        jobid, state, elapsed, limit, nodelist, tres = (
            x.strip() for x in line.split("|")
        )
        if state not in {"R", "CG" if consider_cg else "R"}:
            continue
        nodes = expand_nodelist(nodelist)
        jobs.append(
            Job(
                jobid=jobid,
                state=state,
                elapsed_s=parse_slurm_time(elapsed),
                limit_s=parse_slurm_time(limit),
                nodes=nodes,
                gpus_per_node=parse_tres_gpu_count(tres),
            )
        )
    return jobs


def _parse_gpu_total_from_gres(gres: str) -> int:
    """Parse total GPUs from a GRES-like string.
    Accepts forms like:
      - 'gpu:8'
      - 'gpu:h200:8'
      - 'gpu:h200:8(S:0)'
      - 'gres:gpu:h200:8'
    and combinations separated by commas.
    """
    total_gpu = 0
    for tok in (gres or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        # Drop any parenthetical suffix e.g., (S:0) or (IDX:0-7)
        tok = tok.split("(")[0]
        # Normalize optional 'gres:' prefix
        if tok.startswith("gres:"):
            tok = tok[len("gres:") :]
        if tok.startswith("gpu"):
            parts2 = tok.split(":")
            try:
                total_gpu += int(parts2[-1])
            except (ValueError, IndexError):
                pass
    return total_gpu


def read_nodes(
    partition: str | None, exclude: Iterable[str]
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Return (node_total_gpus, node_partition) for nodes in sinfo (per-node view).
    If partition is set, only include nodes in that partition.
    Falls back to `scontrol show node` if %G does not expose GPUs.
    """
    cmd = ["sinfo", "-h", "-N", "-o", "%N|%G|%P"]
    out = run(cmd)
    totals: Dict[str, int] = {}
    parts: Dict[str, str] = {}
    exclude_set = set(exclude)

    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    for line in lines:
        name, gres, part = (x.strip() for x in line.split("|"))
        if name in exclude_set:
            continue
        # Keep only nodes in requested partition if provided. sinfo may list multiple partitions as partA* or partA,partB
        if partition:
            part_names = [p.rstrip("*") for p in part.split(",")]
            if partition not in part_names:
                continue
        total_gpu = _parse_gpu_total_from_gres(gres)
        if total_gpu == 0:
            # Fallback: query scontrol for this node
            try:
                sc = run(["scontrol", "show", "node", name])
                # Try Gres=... first
                m = re.search(r"Gres=([^\s]+)", sc)
                if m:
                    total_gpu = _parse_gpu_total_from_gres(m.group(1))
                if total_gpu == 0:
                    # Some clusters expose CfgTRES=... which may contain gres/gpu:8
                    m2 = re.search(r"CfgTRES=([^\s]+)", sc)
                    if m2:
                        cfg = m2.group(1)
                        # translate gres/gpu:... to gpu:... for parser
                        cfg = cfg.replace("gres/", "")
                        total_gpu = _parse_gpu_total_from_gres(cfg)
            except Exception:
                pass
        totals[name] = total_gpu
        parts[name] = part
    return totals, parts


# ---------- Core logic ----------


def earliest_time_for_nodes(
    n_needed: int, jobs: List[Job], node_totals: Dict[str, int]
) -> Tuple[int, List[Tuple[str, int]]]:
    """Return (seconds, details), where details lists (node, free_in_seconds).
    A node is free when all jobs on it have finished.
    """
    # Map node -> list of remaining times for jobs on that node
    by_node: Dict[str, List[int]] = {n: [] for n in node_totals}
    for j in jobs:
        for n in j.nodes:
            if n in by_node:
                by_node[n].append(j.remaining_s)
    free_now = [n for n, lst in by_node.items() if len(lst) == 0]
    if len(free_now) >= n_needed:
        return 0, [(n, 0) for n in free_now[:n_needed]]
    # For occupied nodes, free time is max remaining among jobs on it
    occupied = [(n, max(lst)) for n, lst in by_node.items() if lst]
    occupied.sort(key=lambda x: x[1])
    k = n_needed - len(free_now)
    if k <= 0:
        return 0, [(n, 0) for n in free_now[:n_needed]]
    if k > len(occupied):
        # Not enough nodes exist
        return 10**12, []
    deadline = occupied[k - 1][1]
    details = [(n, 0) for n in free_now] + occupied[:k]
    return deadline, details


def earliest_time_for_gpus(
    g_needed: int, jobs: List[Job], node_totals: Dict[str, int]
) -> Tuple[int, Dict[str, List[Tuple[int, int]]]]:
    """Return (seconds, details) for the earliest time when *a single node* has at least g_needed GPUs free.
    details per node is list of (t_free, gpus_free_increment) for transparency.
    """
    # Current usage per node and per-node release events
    used: Dict[str, int] = {n: 0 for n in node_totals}
    events_by_node: Dict[str, List[Tuple[int, int]]] = {n: [] for n in node_totals}

    for j in jobs:
        for n in j.nodes:
            if n not in node_totals:
                continue
            used[n] += j.gpus_per_node
            if j.gpus_per_node > 0:
                events_by_node[n].append((j.remaining_s, j.gpus_per_node))

    # Check immediate availability per node
    for n, total in node_totals.items():
        free_now = max(total - used.get(n, 0), 0)
        if free_now >= g_needed:
            return 0, {n: [(0, free_now)]}

    # For each node, simulate releases until the target is met; take the minimum time over nodes
    best_time = 10**12
    for n, total in node_totals.items():
        free = max(total - used.get(n, 0), 0)
        if not events_by_node[n]:
            continue
        # Accumulate releases in chronological order
        released = 0
        for t, g in sorted(events_by_node[n]):
            released += g
            if free + released >= g_needed:
                best_time = min(best_time, t)
                break

    if best_time == 10**12:
        return 10**12, events_by_node
    return best_time, events_by_node


def find_n_gpu(n_gpu: int):
    node_totals, _ = read_nodes(None, [])
    if not node_totals:
        print("No nodes found (check --partition).")
        sys.exit(3)

    jobs = read_jobs(consider_cg=False)
    max_per_node = max(node_totals.values()) if node_totals else 0
    if n_gpu > max_per_node:
        print(
            f"❌ Request exceeds maximum GPUs on any single node (requested {n_gpu}, max per node {max_per_node})."
        )
        sys.exit(1)
    deadline, events = earliest_time_for_gpus(n_gpu, jobs, node_totals)
    if deadline == 0:
        print(f"✅ {n_gpu} GPU(s) are available now on at least one node.")
        sys.exit(0)
    if deadline >= 10**12:
        print(
            "❌ Not enough GPUs will be free on any single node to satisfy the request (given current jobs/time limits)."
        )
        sys.exit(1)
    print(
        f"⏳ Estimated wait for {n_gpu} free GPU(s) on a single node: {fmt_duration(deadline)}"
    )
    # Provide a short summary of nodes likely to meet the target by the deadline
    candidates = []
    for n, total in node_totals.items():
        # compute free at deadline on node n
        free_now = max(
            total - sum(j.gpus_per_node for j in jobs for nd in j.nodes if nd == n),
            0,
        )
        released = sum(g for t, g in events.get(n, []) if t <= deadline)
        if free_now + released >= n_gpu:
            candidates.append(n)
    if candidates:
        print("Nodes that may meet the target by the ETA:")
        for n in sorted(candidates)[:10]:
            print(f"  - {n}")
    return deadline


def find_n_nodes(n_nodes: int):
    node_totals, _ = read_nodes(None, [])
    if not node_totals:
        print("No nodes found (check --partition).")
        sys.exit(3)

    jobs = read_jobs(consider_cg=False)
    deadline, details = earliest_time_for_nodes(n_nodes, jobs, node_totals)
    if deadline == 0:
        print(f"✅ {n_nodes} node(s) are available now.")
        sys.exit(0)
    if deadline >= 10**12:
        print("❌ Not enough nodes in this partition to satisfy the request.")
        sys.exit(1)
    print(f"⏳ Estimated wait for {n_nodes} free node(s): {fmt_duration(deadline)}")
    print("Details (first nodes to free):")
    for n, t in sorted(details, key=lambda x: x[1]):
        print(f"  - {n}: {fmt_duration(t)}")
    return deadline


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--n-nodes", type=int, help="Number of completely free nodes required"
    )
    g.add_argument(
        "--n-gpu", type=int, help="Number of free GPUs required on a single node"
    )

    args = ap.parse_args()

    node_totals, _ = read_nodes(None, [])
    if not node_totals:
        print("No nodes found (check --partition).")
        sys.exit(3)

    jobs = read_jobs(consider_cg=False)

    if args.n_nodes is not None:
        deadline = find_n_nodes(args.n_nodes)

    if args.n_gpu is not None:
        deadline = find_n_gpu(args.n_gpu)


if __name__ == "__main__":
    main()
