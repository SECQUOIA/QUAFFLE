#!/usr/bin/env python3
"""Generate SLURM scripts for the full QUAFFLE experiment grid.

Axes
  method   classical (FedU-Net), ablation (matched classical bottleneck,
           ablation_studies.py), quantum (QUAFFLE)
  mode     centralized, fed-random, fed-dirichlet (label-skew, --alpha)
  dataset  optical, sar, combined
  seed     config/seeds.py SEEDS (a --seeds subset is validated against it)

Output (default <root>/misc/grid/<timestamp>/)
  one .sh per job, submit_all.sh, manifest.csv
Existing files are never overwritten.

Time limits come from runtimes measured on 2026-10-02 (one A100 GPU,
optical, 200 training tiles) scaled by dataset size and a safety
factor, rounded up to whole hours with a floor. See time_limit() below.

This module is also imported by the other misc/generate_*.py scripts so that
every generator uses the same time limits and the same script header.

Cluster settings (account, partition, python, conda, modules, mail) are not
hard-coded. Copy config/cluster.example.json to config/cluster.local.json
(gitignored) and fill it in, or use QUAFFLE_* environment variables or the
flags below. A SLURM account is required to write scripts.
"""
import argparse
import csv
import datetime as _dt
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import OrderedDict

MISC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MISC_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from config.seeds import SEEDS as ALL_SEEDS  # noqa: E402  single pre-declared seed list

# ---------------------------------------------------------------------------
# Measured runtimes and time limits
# ---------------------------------------------------------------------------

# Wall-clock hours on one A100 GPU, optical dataset (200 training tiles,
# 35 val, 71 test), quantum_channels=32, q_ansatz=reupload with the default
# 2 layers, without CUDA_LAUNCH_BLOCKING. Federated = 4 clients, 25 rounds x
# 4 local epochs, random partition. Dirichlet is assumed equal to random.
#
# Quantum values are measured wall-clock hours from the 2026-10-22 PennyLane
# smoke run (--q-backend pennylane, q_sim_device=cpu, CUDA_LAUNCH_BLOCKING
# unset): 314 s (0.09 h) centralized and 428 s (0.12 h) federated.
#
# Classical and ablation values are blocking-off estimates aligned to the same
# execution regime: in the 2026-10-22 no-blocking A100 step benchmark the
# matched ablation step is 45.2 ms vs 67.9 ms for PennyLane quantum (ratio
# 0.67), and removing CUDA_LAUNCH_BLOCKING=1 halved step times relative to the
# earlier 2026-10-02 blocking-on runs (which took 0.10 h / 0.12 h centralized
# and 0.51 h / 0.56 h federated; see docs/MODEL.md).
MEASURED_HOURS = {
    ("classical", "centralized"): 0.05,  # estimate (blocking off; 0.10 h with CUDA_LAUNCH_BLOCKING=1)
    ("ablation", "centralized"): 0.06,   # estimate (blocking off; 0.12 h with CUDA_LAUNCH_BLOCKING=1)
    ("quantum", "centralized"): 0.09,    # measured 2026-10-22 smoke run (314 s, blocking off)
    ("classical", "federated"): 0.07,    # estimate (blocking off; 0.51 h with CUDA_LAUNCH_BLOCKING=1)
    ("ablation", "federated"): 0.08,     # estimate (blocking off; 0.56 h with CUDA_LAUNCH_BLOCKING=1)
    ("quantum", "federated"): 0.12,      # measured 2026-10-22 smoke run (428 s, blocking off)
}
# Circuit simulator passed to every quantum job (--q-backend). 'pennylane' is
# the single-call PennyLane path. Recorded on the command line so the job
# script shows which simulator produced the results.
Q_BACKEND = "pennylane"
OPTICAL_TRAIN_TILES = 200
TRAIN_TILES = {"optical": 200, "sar": 1119, "combined": 1318}
DATASET_SCALE = {k: v / OPTICAL_TRAIN_TILES for k, v in TRAIN_TILES.items()}
MEASURED_QC = 32
MEASURED_Q_LAYERS = 2
SAFETY_FACTOR = 2.0
FLOOR_HOURS = 1

METHODS = ["classical", "ablation", "quantum"]
MODES = ["centralized", "fed-random", "fed-dirichlet"]
ALL_MODES = ["centralized", "fed-random", "fed-dirichlet", "fed-modality", "fed-geo_cluster"]
DATASETS = ["optical", "sar", "combined"]

SCRIPT = {
    ("classical", "centralized"): "centralized_demoTorchClassical.py",
    ("classical", "federated"): "federated_demoTorchClassical.py",
    ("ablation", "centralized"): "ablation_studies.py",
    ("ablation", "federated"): "ablation_studies.py",
    ("quantum", "centralized"): "centralized_demoTorchQ.py",
    ("quantum", "federated"): "federated_demoTorchQ.py",
}


def mode_kind(mode):
    """'centralized' or 'federated'."""
    return "centralized" if mode == "centralized" else "federated"


def expected_hours(method, mode, dataset, quantum_channels=MEASURED_QC, q_layers=None):
    """Expected wall-clock hours (1 GPU, so also expected SU).

    Runtime scales linearly with the number of training tiles. For the
    quantum method the circuit overhead (quantum minus ablation, which is
    the same network without the circuit) is further scaled by
    quantum_channels/32 and q_layers/2, because the circuit is evaluated
    once per 4-channel group and per re-uploading layer.
    """
    kind = mode_kind(mode)
    base = MEASURED_HOURS[(method, kind)]
    if method == "quantum":
        classical_part = MEASURED_HOURS[("ablation", kind)]
        overhead = base - classical_part
        f = (quantum_channels / MEASURED_QC)
        if q_layers is not None:
            f *= max(q_layers, 1) / MEASURED_Q_LAYERS
        base = classical_part + overhead * f
    return base * DATASET_SCALE[dataset]


def time_limit(method, mode, dataset, quantum_channels=MEASURED_QC, q_layers=None,
               safety=SAFETY_FACTOR, floor=FLOOR_HOURS):
    """Return (expected_hours, limit_hours). limit = ceil(expected * safety), at least floor."""
    exp = expected_hours(method, mode, dataset, quantum_channels, q_layers)
    limit = max(int(floor), int(math.ceil(exp * safety - 1e-9)))
    return exp, limit


def fmt_hms(hours):
    return "%d:00:00" % int(hours)


# ---------------------------------------------------------------------------
# Script template (shared by every generator)
# ---------------------------------------------------------------------------

# Cluster settings. Nothing site specific is hard-coded here. Values are
# resolved per key in this order (first hit wins):
#   1. command-line flag (--account, --partition, --python, ...)
#   2. environment variable QUAFFLE_<KEY> (for example QUAFFLE_ACCOUNT)
#   3. config/cluster.local.json (or the file named by --cluster-config or
#      QUAFFLE_CLUSTER_CONFIG). Copy config/cluster.example.json to start.
#   4. the defaults below
# account has no default. Writing scripts without one is an error.
CLUSTER_CONFIG_DEFAULT = os.path.join(REPO_DIR, "config", "cluster.local.json")
CLUSTER_DEFAULTS = OrderedDict([
    ("account", None),            # SLURM account (-A). Required.
    ("partition", None),          # SLURM partition (-p). Omitted when unset.
    ("python", "python"),         # Python interpreter used in the job command
    ("conda_sh", None),           # path to etc/profile.d/conda.sh, sourced when set
    ("conda_env", None),          # conda env activated when set
    ("modules", []),              # modules loaded after "module purge" (none: no module lines)
    ("mail_user", None),          # --mail-user. Mail lines are omitted when unset
    ("mail_type", "END,FAIL"),    # --mail-type, only used with mail_user
    ("gpus", 1),                  # --gpus-per-node
    ("cpus_per_task", 8),         # --cpus-per-task for federated jobs (centralized: unset)
    ("mem", "64G"),               # --mem
    ("output_log_dir", "output_logs"),  # SLURM stdout folder under the root
    ("error_log_dir", "error_logs"),    # SLURM stderr folder under the root
    # Export a git safe.directory entry for the root in each job, so that the
    # "git:" provenance line works when the user running the job does not own
    # the checkout. Off by default.
    ("git_safe_directory", False),
])
CLUSTER_INT_KEYS = ("gpus", "cpus_per_task")
CLUSTER_BOOL_KEYS = ("git_safe_directory",)
ACCOUNT_PLACEHOLDER = "YOUR_ACCOUNT"


def _parse_modules(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [m for m in re.split(r"[,\s]+", value) if m]
    return [str(m) for m in value]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    raise SystemExit("not a boolean value: %r (use true or false)" % (value,))


def load_cluster_config(path=None):
    """Return the dict from the cluster config file, or {} if it does not exist."""
    import json
    explicit = path or os.environ.get("QUAFFLE_CLUSTER_CONFIG")
    path = explicit or CLUSTER_CONFIG_DEFAULT
    if not os.path.exists(path):
        if explicit:
            raise SystemExit("cluster config %s not found" % path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    unknown = sorted(k for k in data if k not in CLUSTER_DEFAULTS and not k.startswith("_"))
    if unknown:
        raise SystemExit("unknown keys in %s: %s (known: %s)"
                         % (path, ", ".join(unknown), ", ".join(CLUSTER_DEFAULTS)))
    return data


def resolve_cluster(cli=None, path=None):
    """Merge CLI values, QUAFFLE_* environment variables, the config file and
    the defaults into one settings dict."""
    cli = cli or {}
    filecfg = load_cluster_config(path)
    out = OrderedDict()
    for key, default in CLUSTER_DEFAULTS.items():
        env = os.environ.get("QUAFFLE_" + key.upper())
        if cli.get(key) is not None:
            val = cli[key]
        elif env not in (None, ""):
            val = env
        elif filecfg.get(key) is not None:
            val = filecfg[key]
        else:
            val = default
        if key == "modules":
            val = _parse_modules(val)
        elif key in CLUSTER_BOOL_KEYS:
            val = _parse_bool(val)
        elif key in CLUSTER_INT_KEYS and val is not None:
            val = int(val)
        elif val is not None:
            val = str(val)
        if val == "":
            val = None
        out[key] = val
    return out


def render_script(job_name, command, limit_h, exp_h, root, cpus=None, cluster=None, note=""):
    """Full SLURM script text. command is the shell command line to run.
    cluster is a dict from resolve_cluster()."""
    c = dict(CLUSTER_DEFAULTS)
    c.update(cluster or {})
    lines = ["#!/bin/bash",
             "#SBATCH -A %s" % (c["account"] or ACCOUNT_PLACEHOLDER),
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks-per-node=1"]
    if cpus:
        lines.append("#SBATCH --cpus-per-task=%d" % cpus)
    lines += ["#SBATCH --gpus-per-node=%d" % int(c["gpus"]),
              "#SBATCH --mem=%s" % c["mem"],
              "#SBATCH --time=%s" % fmt_hms(limit_h),
              "#SBATCH -J %s" % job_name,
              "#SBATCH -o %s/%s/%%x.o%%j" % (root, c["output_log_dir"]),
              "#SBATCH -e %s/%s/%%x.e%%j" % (root, c["error_log_dir"])]
    if c["partition"]:
        lines.append("#SBATCH -p %s" % c["partition"])
    if c["mail_user"]:
        lines.append("#SBATCH --mail-user=%s" % c["mail_user"])
        if c["mail_type"]:
            lines.append("#SBATCH --mail-type=%s" % c["mail_type"])
    lines += ["# Expected runtime %.2f h (limit %s h). %s" % (exp_h, limit_h, note), ""]
    if c["modules"]:
        lines.append("module purge")
        lines += ["module load %s" % m for m in c["modules"]]
    if c["conda_sh"]:
        lines.append("source %s" % shlex.quote(c["conda_sh"]))
    if c["conda_env"]:
        lines.append("conda activate %s" % shlex.quote(c["conda_env"]))
    lines += [
        'export XLA_FLAGS="--xla_gpu_enable_cudnn_frontend=false"',
        "export TF_CPP_MIN_LOG_LEVEL=1",
        'export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"',
    ]
    if c["git_safe_directory"]:
        lines.append("export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory "
                     "GIT_CONFIG_VALUE_0=%s" % shlex.quote(root))
    lines += [
        "cd %s || exit 1" % shlex.quote(root),
        'echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown) '
        'branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) '
        'dirty=$(test -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" '
        '&& echo clean || echo dirty)"',
        'echo "host=$(hostname) SLURM_JOB_ID=$SLURM_JOB_ID"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv",
    ]
    head = "\n".join(lines) + "\n"
    return head + 'echo "cmd: {}"\n{}\n'.format(command.replace('"', '\\"'), command)


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def supports_ckpt_every(root, script):
    """True if the target script, or tools/training_common.py that defines the
    shared CLI flags, mentions --ckpt-every."""
    for rel in (script, os.path.join("tools", "training_common.py")):
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as f:
                if "ckpt-every" in f.read():
                    return True
        except OSError:
            pass
    return False


def build_job(method, mode, dataset, seed, root, python="python", alpha=0.5,
              q_ansatz="reupload", q_layers=None, knockout=True, quantum_channels=None,
              ckpt_every=5, ckpt_support=None, safety=SAFETY_FACTOR, floor=FLOOR_HOURS,
              q_backend=Q_BACKEND, cpus=CLUSTER_DEFAULTS["cpus_per_task"], hqunet_grid=None,
              q_spatial_mode=None, base_channels=None, skip_modulation=None,
              backbone_conv_mode=None, loss=None, no_time_emb=False, augment=False,
              combined_input_mode=None, fl_strategy=None, fedprox_mu=None,
              q_angle_agg=None, num_clients=None, clients_per_round=None,
              fraction_fit=None):
    """Return a dict describing one job (no files written)."""
    kind = mode_kind(mode)
    script = SCRIPT[(method, kind)]
    args = []
    if method == "ablation":
        args += ["--framework", "torch", "--mode", kind]
    if kind == "federated":
        part_map = {
            "fed-random": "random",
            "fed-dirichlet": "dirichlet",
            "fed-modality": "modality",
            "fed-geo_cluster": "geo_cluster",
        }
        part = part_map.get(mode, "random")
        args += ["--partition", part]
        if part == "dirichlet":
            args += ["--alpha", str(alpha)]
        if fl_strategy is not None:
            args += ["--fl-strategy", str(fl_strategy)]
        if fedprox_mu is not None:
            args += ["--fedprox-mu", str(fedprox_mu)]
        if method == "quantum" and q_angle_agg is not None:
            args += ["--q-angle-agg", str(q_angle_agg)]
        if num_clients is not None:
            args += ["--num-clients", str(num_clients)]
        if clients_per_round is not None:
            args += ["--clients-per-round", str(clients_per_round)]
        if fraction_fit is not None:
            args += ["--fraction-fit", str(fraction_fit)]
    args += ["--dataset", dataset, "--seed", str(seed)]
    if combined_input_mode is not None:
        args += ["--combined-input-mode", str(combined_input_mode)]
    if base_channels is not None:
        args += ["--base-channels", str(base_channels)]
    if backbone_conv_mode is not None:
        args += ["--backbone-conv-mode", str(backbone_conv_mode)]
    if loss is not None:
        args += ["--loss", str(loss)]
    if no_time_emb:
        args += ["--no-time-emb"]
    if augment:
        args += ["--augment"]
    if method in ("quantum", "ablation"):
        args += ["--q-ansatz", q_ansatz]
        if q_layers is not None:
            args += ["--q-layers", str(q_layers)]
        if q_ansatz == "hqunet" and hqunet_grid is not None:
            args += ["--hqunet-grid", str(hqunet_grid)]
        if q_spatial_mode is not None and q_ansatz != "hqunet":
            args += ["--q-spatial-mode", str(q_spatial_mode)]
        if skip_modulation is not None:
            args += ["--skip-modulation", str(skip_modulation)]
    if method == "quantum":
        args += ["--q-backend", q_backend]
    if method == "quantum" and knockout:
        args += ["--quantum-knockout-eval"]
    qc = MEASURED_QC
    if quantum_channels is not None:
        if method not in ("quantum", "ablation"):
            raise ValueError("--quantum_channels is only supported for quantum and ablation methods")
        args += ["--quantum_channels", str(quantum_channels)]
        qc = quantum_channels
    ckpt = False
    if kind == "federated" and ckpt_every:
        if ckpt_support is None:
            ckpt_support = supports_ckpt_every(root, script)
        if ckpt_support:
            args += ["--ckpt-every", str(ckpt_every)]
            ckpt = True

    exp, limit = time_limit(method, mode, dataset, qc,
                            q_layers if method == "quantum" else None, safety, floor)

    mtag = {"classical": "classical", "ablation": "ablation", "quantum": "quantum"}[method]
    dtag = {"centralized": "central", "fed-random": "fedrandom",
            "fed-dirichlet": "feddir%s" % alpha,
            "fed-modality": "fedmodality",
            "fed-geo_cluster": "fedgeocluster"}[mode]
    parts = [mtag, dtag, dataset]
    if method in ("quantum", "ablation"):
        ans_tag = q_ansatz
        if q_ansatz == "hqunet" and hqunet_grid is not None:
            ans_tag += "G%d" % int(hqunet_grid)
        if q_layers is not None:
            ans_tag += "L%d" % q_layers
        parts.append(ans_tag)
        if q_spatial_mode is not None and q_ansatz != "hqunet" and q_spatial_mode != "global":
            parts.append(str(q_spatial_mode))
        if skip_modulation == "additive":
            parts.append("addskip")
    if combined_input_mode is not None and combined_input_mode != "shared3":
        parts.append("in-%s" % combined_input_mode)
    if backbone_conv_mode == "depthwise_sep":
        parts.append("depthwise_sep")
    if loss is not None and loss != "ce":
        parts.append(str(loss))
    if no_time_emb:
        parts.append("notime")
    if augment:
        parts.append("aug")
    if kind == "federated":
        if fl_strategy == "fedprox" or (fedprox_mu is not None and float(fedprox_mu) > 0.0):
            mu_val = float(fedprox_mu) if fedprox_mu is not None else 0.01
            parts.append("fedprox%g" % mu_val)
        if method == "quantum" and q_angle_agg == "circular":
            parts.append("qagg-circular")
        if num_clients is not None or clients_per_round is not None or fraction_fit is not None:
            nc = int(num_clients) if num_clients is not None else 4
            if clients_per_round is not None:
                cpr = int(clients_per_round)
            elif fraction_fit is not None:
                cpr = max(1, min(nc, int(round(float(fraction_fit) * nc))))
            else:
                cpr = nc
            parts.append("K%dC%d" % (nc, cpr))
    if quantum_channels is not None:
        parts.append("qc%d" % quantum_channels)
    if base_channels is not None:
        parts.append("bc%d" % int(base_channels))
    parts.append("seed%d" % seed)
    name = "_".join(parts)
    short = {"classical": "c", "ablation": "a", "quantum": "q"}[method]
    short += {"centralized": "C", "fed-random": "R", "fed-dirichlet": "D",
              "fed-modality": "M", "fed-geo_cluster": "G"}[mode]
    ans_short = ""
    if method != "classical" and q_ansatz != "reupload":
        ans_short = "_hq" if q_ansatz == "hqunet" else "_leg"
    jshort = "%s_%s%s%s_s%d" % (short, dataset[:3], "" if quantum_channels is None else
                                "_qc%d" % quantum_channels,
                                ans_short,
                                seed)
    command = " ".join([shlex.quote(python), script] + [shlex.quote(a) for a in args])
    return OrderedDict([
        ("name", name), ("job_name", jshort), ("method", method), ("mode", mode),
        ("dataset", dataset), ("seed", seed),
        ("ansatz", q_ansatz if method != "classical" else ""),
        ("script", script), ("args", args), ("command", command),
        ("exp_h", exp), ("limit_h", limit), ("cpus", cpus if kind == "federated" else None),
        ("ckpt", ckpt),
        ("variant", "" if quantum_channels is None else "qc%d" % quantum_channels),
    ])


def write_jobs(jobs, out_dir, root, cluster, stagger=0, max_concurrent=0, overwrite=False):
    """Write one script per job plus submit_all.sh and manifest.csv into out_dir."""
    if not cluster.get("account"):
        raise SystemExit("no SLURM account set. Pass --account, export QUAFFLE_ACCOUNT, or set "
                         "\"account\" in config/cluster.local.json (copy "
                         "config/cluster.example.json)")
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for j in jobs:
        fname = j["name"] + ".sh"
        path = os.path.join(out_dir, fname)
        if os.path.exists(path) and not overwrite:
            raise SystemExit("refusing to overwrite existing %s" % path)
        note = "ckpt-every on." if j["ckpt"] else ""
        text = render_script(j["job_name"], j["command"], j["limit_h"], j["exp_h"], root,
                             cpus=j["cpus"], cluster=cluster, note=note)
        with open(path, "w", newline="\n") as f:
            f.write(text)
        os.chmod(path, 0o755)
        names.append(fname)

    for extra in ("submit_all.sh", "manifest.csv"):
        p = os.path.join(out_dir, extra)
        if os.path.exists(p) and not overwrite:
            raise SystemExit("refusing to overwrite existing %s" % p)

    lines = [
        "#!/bin/bash",
        "# Submit every job in this folder. Generated %s." % _dt.datetime.now().isoformat(timespec="seconds"),
        "# stagger=%d min between submissions, max_concurrent=%s (dependency lanes, afterany)."
        % (stagger, max_concurrent or "unlimited"),
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "cd %s" % shlex.quote(root),
        "mkdir -p %s %s" % (shlex.quote(cluster["output_log_dir"]),
                            shlex.quote(cluster["error_log_dir"])),
        'LOG="$HERE/submitted.txt"',
        "declare -a IDS=()",
        "",
    ]
    for i, fname in enumerate(names):
        opts = []
        if stagger:
            opts.append("--begin=now+%dminutes" % (i * stagger))
        if max_concurrent and i >= max_concurrent:
            opts.append('--dependency=afterany:${IDS[%d]}' % (i - max_concurrent))
        opt_s = (" " + " ".join(opts)) if opts else ""
        lines.append('id=$(sbatch --parsable%s "$HERE/%s"); IDS+=("$id"); echo "$id %s" | tee -a "$LOG"'
                     % (opt_s, fname, fname))
    lines.append('echo "submitted ${#IDS[@]} jobs"')
    sp = os.path.join(out_dir, "submit_all.sh")
    with open(sp, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(sp, 0o755)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["script", "job_name", "method", "mode", "dataset", "seed", "ansatz",
                    "time_limit", "est_su", "limit_su", "ckpt_every", "command"])
        for j, fname in zip(jobs, names):
            w.writerow([fname, j["job_name"], j["method"], j["mode"], j["dataset"], j["seed"],
                        j["ansatz"], fmt_hms(j["limit_h"]), "%.2f" % j["exp_h"], j["limit_h"],
                        int(j["ckpt"]), j["command"]])
    return names


# ---------------------------------------------------------------------------
# Reporting and checks
# ---------------------------------------------------------------------------

def summary_table(jobs, budget=None):
    groups = OrderedDict()
    for j in jobs:
        ds = j["dataset"] + ("/" + j["variant"] if j.get("variant") else "")
        groups.setdefault((j["method"], j["mode"], ds), []).append(j)
    hdr = "%-10s %-14s %-14s %5s %9s %8s %10s %10s" % (
        "method", "mode", "dataset", "jobs", "est_h_ea", "limit_h", "est_SU", "limit_SU")
    out = [hdr, "-" * len(hdr)]
    for (m, mo, d), js in groups.items():
        est = sum(j["exp_h"] for j in js)
        lim = sum(j["limit_h"] for j in js)
        out.append("%-10s %-14s %-14s %5d %9.2f %8d %10.1f %10d" % (
            m, mo, d, len(js), js[0]["exp_h"], js[0]["limit_h"], est, lim))
    est = sum(j["exp_h"] for j in jobs)
    lim = sum(j["limit_h"] for j in jobs)
    out.append("-" * len(hdr))
    out.append("%-40s %5d %9s %8s %10.1f %10d" % ("TOTAL", len(jobs), "", "", est, lim))
    over = False
    if budget is not None:
        left = budget - est
        over = left < 0
        out.append("budget %.1f SU: expected use %.1f SU, %s %.1f SU (worst case if every job "
                   "hits its limit %d SU, %s)" % (
                       budget, est, "OVER BUDGET by" if over else "remaining", abs(left), lim,
                       "over budget" if lim > budget else "within budget"))
    return "\n".join(out), over


def _run(cmd, timeout=600):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=timeout)


def check_flags(jobs, root, python, ssh_host=None):
    """Run each distinct target script with --help once and confirm every
    emitted flag appears in the help text. Returns True if all pass."""
    from concurrent.futures import ThreadPoolExecutor
    by_script = OrderedDict()
    for j in jobs:
        flags = by_script.setdefault(j["script"], set())
        flags.update(a for a in j["args"] if a.startswith("--"))

    def helptext(script):
        inner = "cd %s && %s %s --help" % (shlex.quote(root), shlex.quote(python), script)
        cmd = ["ssh", "-o", "BatchMode=yes", ssh_host, inner] if ssh_host else ["bash", "-c", inner]
        r = _run(cmd)
        return r.returncode, r.stdout

    ok = True
    with ThreadPoolExecutor(max_workers=len(by_script)) as ex:
        results = dict(zip(by_script, ex.map(helptext, list(by_script))))
    for script, flags in by_script.items():
        rc, text = results[script]
        known = set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9_-]*)", text))
        missing = sorted(f for f in flags if f not in known)
        status = "OK" if rc == 0 and not missing else "FAIL"
        ok &= status == "OK"
        print("check-flags %-34s rc=%d flags=%s %s%s" % (
            script, rc, " ".join(sorted(flags)), status,
            (" missing=" + ",".join(missing)) if missing else ""))
        if rc != 0:
            print(text[-2000:])
    return ok


def _hostnames(expr):
    r = _run(["scontrol", "show", "hostnames", expr], timeout=60)
    return set(r.stdout.split()) if r.returncode == 0 else set()


def check_reservations(jobs, partition, stagger=0):
    """Warn about maintenance reservations on the partition nodes that a job
    (submitted now, with its time limit) would run into. Uses scontrol if
    present, otherwise prints a note."""
    if not shutil.which("scontrol"):
        print("reservations: scontrol not found here, skipped (check with sinfo -T)")
        return
    r = _run(["scontrol", "show", "partition", partition, "-o"], timeout=60)
    m = re.search(r"\bNodes=(\S+)", r.stdout)
    pnodes = _hostnames(m.group(1)) if m else set()
    r = _run(["scontrol", "show", "reservation", "-o"], timeout=60)
    now = _dt.datetime.now()
    longest = max(j["limit_h"] for j in jobs) if jobs else 0
    found = False
    for line in r.stdout.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        if "MAINT" not in kv.get("Flags", "") or not pnodes:
            continue
        rnodes = _hostnames(kv.get("Nodes", ""))
        hit = pnodes & rnodes
        if not hit:
            continue
        try:
            start = _dt.datetime.strptime(kv["StartTime"], "%Y-%m-%dT%H:%M:%S")
            end = _dt.datetime.strptime(kv["EndTime"], "%Y-%m-%dT%H:%M:%S")
        except (KeyError, ValueError):
            continue
        if end < now:
            continue
        found = True
        hrs = (start - now).total_seconds() / 3600.0
        if hrs <= 0:
            print("reservations: note %s (MAINT, active) holds %d/%d %s nodes until %s. "
                  "Those nodes are unavailable, the rest are usable."
                  % (kv.get("ReservationName"), len(hit), len(pnodes), partition, end))
            continue
        blocked = [j for i, j in enumerate(jobs) if j["limit_h"] + i * stagger / 60.0 > hrs]
        print("reservations: WARNING %s (MAINT) covers %d/%d %s nodes from %s to %s. "
              "%d/%d jobs have a limit (plus stagger) past its start and will stay PENDING "
              "(ReqNodeNotAvail/Reservation) until it ends. Longest limit %d h, window %.1f h."
              % (kv.get("ReservationName"), len(hit), len(pnodes), partition, start, end,
                 len(blocked), len(jobs), longest, max(hrs, 0)))
    if not found:
        print("reservations: no maintenance reservation on partition %s nodes" % partition)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _validate_seeds(ap, seeds):
    bad = sorted(set(seeds) - set(ALL_SEEDS))
    if bad:
        ap.error("seeds %s are not in config/seeds.py %s" % (bad, ALL_SEEDS))
    return [s for s in ALL_SEEDS if s in set(seeds)]


def add_cluster_args(ap):
    g = ap.add_argument_group(
        "cluster settings",
        "Each value is taken from the flag, else the QUAFFLE_<KEY> environment variable, else "
        "config/cluster.local.json, else the built-in default. See config/cluster.example.json")
    g.add_argument("--cluster-config", default=None, metavar="PATH",
                   help="Cluster settings JSON (default: $QUAFFLE_CLUSTER_CONFIG or "
                        "config/cluster.local.json when it exists)")
    g.add_argument("--account", default=None, help="SLURM account (required to write scripts)")
    g.add_argument("--partition", default=None)
    g.add_argument("--python", default=None, help="Python interpreter used in the job command")
    g.add_argument("--conda-sh", default=None, help="conda.sh to source in the job")
    g.add_argument("--conda-env", default=None, help="conda env to activate in the job")
    g.add_argument("--modules", nargs="*", default=None,
                   help="Modules to load after 'module purge'")
    g.add_argument("--mail-user", default=None, help="Mail address (mail lines omitted if unset)")
    g.add_argument("--mail-type", default=None)
    g.add_argument("--gpus", type=int, default=None, help="GPUs per node")
    g.add_argument("--cpus-per-task", type=int, default=None, help="CPUs for federated jobs")
    g.add_argument("--mem", default=None)
    g.add_argument("--output-log-dir", default=None, help="SLURM stdout folder under --root")
    g.add_argument("--error-log-dir", default=None, help="SLURM stderr folder under --root")
    g.add_argument("--git-safe-directory", dest="git_safe_directory", action="store_true",
                   default=None,
                   help="Mark --root as a git safe.directory in each job (needed only when the "
                        "user running the jobs does not own the checkout)")
    g.add_argument("--no-git-safe-directory", dest="git_safe_directory", action="store_false",
                   default=None, help="Do not export a git safe.directory entry (default)")
    ap.add_argument("--root", default=REPO_DIR,
                    help="Repo checkout the jobs cd into and run from (default: the checkout "
                         "this generator lives in, %(default)s)")
    ap.add_argument("--safety", type=float, default=SAFETY_FACTOR,
                    help="Time limit = ceil(expected hours x safety) (default %(default)s)")
    ap.add_argument("--floor-hours", type=int, default=FLOOR_HOURS)
    ap.add_argument("--budget", type=float, default=None,
                    help="Available SU. Refuse to write if the expected total exceeds it")
    ap.add_argument("--stagger-minutes", type=int, default=0,
                    help="submit_all.sh adds sbatch --begin=now+i*N minutes")
    ap.add_argument("--max-concurrent", type=int, default=0,
                    help="submit_all.sh chains jobs into N lanes with --dependency=afterany "
                         "so at most N run at once (0 = no limit)")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, write nothing")
    ap.add_argument("--check-flags", action="store_true",
                    help="Run each target script with --help once and verify every emitted flag")
    ap.add_argument("--check-ssh", default=None, metavar="HOST",
                    help="Run the --check-flags --help calls via ssh HOST (for when this "
                         "user cannot execute --python)")
    ap.add_argument("--no-reservation-check", action="store_true")
    return ap


def resolve_cluster_args(args):
    """Resolve the cluster settings for parsed args. Sets args.cluster and
    args.python, args.partition, args.account from it. Idempotent."""
    if getattr(args, "cluster", None) is not None:
        return args.cluster
    cli = {k: getattr(args, k, None) for k in CLUSTER_DEFAULTS}
    args.cluster = resolve_cluster(cli, args.cluster_config)
    args.python = args.cluster["python"]
    args.partition = args.cluster["partition"]
    args.account = args.cluster["account"]
    return args.cluster


def finish(ap, args, jobs, out_dir):
    """Shared tail for every generator: summary, checks, budget, write."""
    cluster = resolve_cluster_args(args)
    table, over = summary_table(jobs, args.budget)
    print(table)
    if not args.no_reservation_check:
        if args.partition:
            check_reservations(jobs, args.partition, args.stagger_minutes)
        else:
            print("reservations: no partition set, skipped")
    if args.check_flags and jobs:
        if not check_flags(jobs, args.root, args.python, args.check_ssh):
            print("check-flags FAILED, nothing written")
            sys.exit(4)
    if over:
        print("expected SU exceeds --budget, nothing written")
        sys.exit(3)
    if args.dry_run:
        print("dry-run: would write %d scripts + submit_all.sh + manifest.csv to %s"
              % (len(jobs), out_dir))
        print("cluster: account=%s partition=%s python=%s conda_env=%s mail=%s"
              % (cluster["account"] or ACCOUNT_PLACEHOLDER + " (not set, required to write)",
                 cluster["partition"], cluster["python"], cluster["conda_env"],
                 "on" if cluster["mail_user"] else "off"))
        for j in jobs[:3]:
            print("  e.g. %s.sh  %s  [%s]" % (j["name"], j["command"], fmt_hms(j["limit_h"])))
        return
    write_jobs(jobs, out_dir, args.root, cluster, args.stagger_minutes, args.max_concurrent)
    print("wrote %d scripts, submit_all.sh and manifest.csv to %s" % (len(jobs), out_dir))
    print("submit with: bash %s" % os.path.join(out_dir, "submit_all.sh"))


def default_out_dir(root, prefix=""):
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(root, "misc", "grid", prefix + ts)


def guard_out_dir(ap, out_dir):
    out_dir = os.path.abspath(out_dir)
    if os.path.normpath(out_dir) == os.path.normpath(MISC_DIR):
        ap.error("--out-dir must not be misc/ itself (existing scripts live there)")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        ap.error("--out-dir %s exists and is not empty" % out_dir)
    return out_dir


def print_cluster_env(argv):
    """Print shell export lines for the resolved cluster settings, for job
    scripts such as misc/smoke_pennylane_fast.sh that cannot read JSON."""
    ap = argparse.ArgumentParser(prog="generate_grid.py --print-cluster-env")
    ap.add_argument("--cluster-config", default=None)
    args = ap.parse_args(argv)
    for key, val in resolve_cluster(None, args.cluster_config).items():
        if isinstance(val, list):
            val = " ".join(val)
        elif isinstance(val, bool):
            val = "true" if val else "false"
        print("export QUAFFLE_%s=%s" % (key.upper(), shlex.quote("" if val is None else str(val))))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--print-cluster-env"]:
        return print_cluster_env(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--modes", nargs="+", choices=ALL_MODES, default=MODES)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS,
                    help="Subset of config/seeds.py (default: all %s)" % ALL_SEEDS)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (default 0.5)")
    ap.add_argument("--q-ansatz", choices=["reupload", "legacy", "hqunet"], default="reupload")
    ap.add_argument("--q-layers", type=int, default=None)
    ap.add_argument("--hqunet-grid", type=int, choices=[2, 4], default=None)
    ap.add_argument("--q-spatial-mode", choices=["global", "pixel", "patch2x2", "patch4x4", "se_gate"], default=None)
    ap.add_argument("--skip-modulation", choices=["affine", "additive"], default=None)
    ap.add_argument("--backbone-conv-mode", choices=["ws_conv", "standard", "depthwise_sep"], default=None)
    ap.add_argument("--combined-input-mode", choices=["shared3", "concat6", "dual_stem"], default=None)
    ap.add_argument("--fl-strategy", choices=["fedavg", "fedprox"], default=None)
    ap.add_argument("--fedprox-mu", type=float, default=None)
    ap.add_argument("--q-angle-agg", choices=["linear", "circular"], default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--clients-per-round", type=int, default=None)
    ap.add_argument("--fraction-fit", type=float, default=None)
    ap.add_argument("--loss", choices=["ce", "dice_ce", "focal_dice"], default=None)
    ap.add_argument("--no-time-emb", action="store_true", default=False)
    ap.add_argument("--augment", action="store_true", default=False)
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--quantum-channels", "--quantum_channels", dest="quantum_channels", type=int, default=None)
    ap.add_argument("--q-backend", choices=["pennylane", "pennylane_pergroup", "torch"],
                    default=Q_BACKEND,
                    help="Circuit simulator passed to quantum jobs (default %(default)s). Use "
                         "'torch' only for a cross-check grid, the time limits assume pennylane")
    ap.add_argument("--no-quantum-knockout-eval", dest="knockout", action="store_false",
                    help="Do not pass --quantum-knockout-eval to quantum jobs (on by default)")
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="Pass --ckpt-every N to federated jobs when the target script "
                         "supports it (0 disables)")
    ap.add_argument("--out-dir", default=None,
                    help="Folder for the generated scripts (default <root>/misc/grid/<timestamp>)")
    add_cluster_args(ap)
    args = ap.parse_args(argv)
    seeds = _validate_seeds(ap, args.seeds)
    cluster = resolve_cluster_args(args)
    args.root = os.path.abspath(args.root)
    out_dir = guard_out_dir(ap, args.out_dir or default_out_dir(args.root))

    support = {}
    for key, script in SCRIPT.items():
        if key[1] == "federated" and key[0] in args.methods:
            support[script] = supports_ckpt_every(args.root, script)
    for script, s in support.items():
        if args.ckpt_every:
            print("ckpt-every: %-32s %s" % (script, "supported, passing --ckpt-every %d"
                                              % args.ckpt_every if s else
                                              "not found in source, not passed"))

    jobs = []
    for seed in seeds:
        for method in METHODS:
            if method not in args.methods:
                continue
            for mode in ALL_MODES:
                if mode not in args.modes:
                    continue
                for dataset in DATASETS:
                    if dataset not in args.datasets:
                        continue
                    script = SCRIPT[(method, mode_kind(mode))]
                    jobs.append(build_job(
                        method, mode, dataset, seed, args.root, python=args.python,
                        alpha=args.alpha, q_ansatz=args.q_ansatz, q_layers=args.q_layers,
                        knockout=args.knockout,
                        quantum_channels=args.quantum_channels if method != "classical" else None,
                        ckpt_every=args.ckpt_every,
                        ckpt_support=support.get(script, False), safety=args.safety,
                        floor=args.floor_hours, q_backend=args.q_backend,
                        cpus=cluster["cpus_per_task"], hqunet_grid=args.hqunet_grid,
                        q_spatial_mode=args.q_spatial_mode, base_channels=args.base_channels,
                        skip_modulation=args.skip_modulation,
                        backbone_conv_mode=args.backbone_conv_mode,
                        loss=args.loss, no_time_emb=args.no_time_emb, augment=args.augment,
                        combined_input_mode=args.combined_input_mode,
                        fl_strategy=args.fl_strategy, fedprox_mu=args.fedprox_mu,
                        q_angle_agg=args.q_angle_agg, num_clients=args.num_clients,
                        clients_per_round=args.clients_per_round,
                        fraction_fit=args.fraction_fit))
    finish(ap, args, jobs, out_dir)


if __name__ == "__main__":
    main()
