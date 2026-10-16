#!/usr/bin/env python3
"""Run provenance helpers shared by every training entry point.

collect_provenance() returns a flat dict describing exactly what produced a
run: git SHA and dirty flag, branch, full command line, seed, hostname,
SLURM job id, and the torch / pennylane / flwr versions. The training demos
write it into results/summary.txt (format_provenance) and pass it to
results_tracker.save_results_to_excel so every tracker row carries it too.

All lookups are best effort. A missing git binary or package never raises,
the field is just recorded as "unknown".
"""
import os
import sys
import shlex
import socket
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ordered keys, also used as extra tracker columns.
PROVENANCE_KEYS = [
    "git_sha",
    "git_dirty",
    "git_branch",
    "cli_args",
    "seed",
    "hostname",
    "slurm_job_id",
    "torch_version",
    "pennylane_version",
    "flwr_version",
    "python_version",
    "start_time",
    # Quantum circuit simulator of the run (quantum entry points only, set
    # by the caller from config['q_backend']).
    "q_backend",
]

# Keys that are only written when the caller has set them (classical and
# ablation runs have no quantum circuit simulator).
OPTIONAL_PROVENANCE_KEYS = {"q_backend"}


def _git(*args):
    # safe.directory lets a different Unix user (for example the account that
    # submits SLURM jobs) read a repository owned by someone else.
    cmd = ["git", "-c", f"safe.directory={REPO_ROOT}", "-C", REPO_ROOT, *args]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def collect_provenance(seed=None, argv=None):
    """Return a dict with the provenance fields listed in PROVENANCE_KEYS."""
    argv = sys.argv if argv is None else argv
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if status is None:
        dirty = "unknown"
    else:
        dirty = "dirty" if status else "clean"
    return {
        "git_sha": sha or "unknown",
        "git_dirty": dirty,
        "git_branch": branch or "unknown",
        "cli_args": " ".join(shlex.quote(a) for a in argv),
        "seed": seed,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "torch_version": _version("torch"),
        "pennylane_version": _version("pennylane"),
        "flwr_version": _version("flwr"),
        "python_version": sys.version.split()[0],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def format_provenance(prov):
    """Render a provenance dict as a block for summary.txt."""
    lines = ["Provenance:"]
    for k in PROVENANCE_KEYS:
        if k in OPTIONAL_PROVENANCE_KEYS and k not in prov:
            continue
        lines.append(f"  {k}: {prov.get(k)}")
    return "\n".join(lines) + "\n\n"


def print_provenance(prov):
    print(format_provenance(prov), end="")


def append_provenance(summary_path, prov):
    """Append a provenance block to an already written summary.txt."""
    with open(summary_path, "a") as f:
        f.write("\n" + format_provenance(prov))


if __name__ == "__main__":
    print_provenance(collect_provenance())
