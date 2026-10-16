#!/usr/bin/env python3
"""Compare quantum results across two results roots (two circuit simulators).

Typical use: a grid run with the torch simulator against the same grid run
with PennyLane.

    python tools/compare_backends.py \\
        results_torch results

Every quantum run directory under each root is keyed by (mode, dataset,
variant, seed), where mode is centralized, fed-random or fed-dirichlet<alpha>
and variant is the name part between the dataset and the backend tag (for
example _reupload_L2). The backend tag (_qtorch, _qpergroup) is not part of
the key, so a tagged run in one root matches an untagged run in the other. Names
look like
    flood_central_optical_pytorch_pennylane[_sar|_combined]_reupload_L2[_qtorch|_qpergroup]_seed17
    flood_federated_optical_pytorch_pennylane_{random|dirichlet0.5}[_sar|_combined]_reupload_L2[...]_seed17
Metrics come from <run>/results/metrics.json, with a fallback that parses
<run>/results/summary.txt.

For each (mode, dataset, variant) the tool prints, for each root, the mean and sample
std over the seeds present in both roots (matched seeds), and the difference
of means (B minus A). Metrics: test AUC, IoU, Dice, loss, and the knockout
'zero' and 'random' test IoU. Agreement between simulators should be judged
against the seed-to-seed std, not equality: the circuits are mathematically
identical but float32 summation order and GPU reductions differ.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict

RUN_RE = re.compile(
    r"^flood_(?P<kind>central|federated)_optical_pytorch_pennylane"
    r"(?:_(?P<part>random|dirichlet(?P<alpha>[0-9.]+)))?"
    r"(?:_(?P<ds>sar|combined))?"
    r"(?P<variant>(?:_[A-Za-z0-9.]+?)*?)"
    r"(?P<btag>_qtorch|_qpergroup)?"
    r"_seed(?P<seed>\d+)$")

# (label, path into the metrics dict)
METRICS = [
    ("test_auc", ("test", "auc")),
    ("test_iou", ("test", "iou")),
    ("test_dice", ("test", "dice")),
    ("test_loss", ("test", "loss")),
    ("ko_zero_iou", ("knockout", "zero", "iou")),
    ("ko_random_iou", ("knockout", "random", "iou")),
]
MODE_ORDER = {"centralized": 0, "fed-random": 1}
DS_ORDER = {"optical": 0, "sar": 1, "combined": 2}


def parse_name(name):
    """Return (mode, dataset, seed, variant, btag) or None for non-quantum dirs."""
    m = RUN_RE.match(name)
    if not m:
        return None
    if m.group("kind") == "central":
        if m.group("part"):
            return None
        mode = "centralized"
    else:
        part = m.group("part")
        if not part:
            return None
        mode = "fed-random" if part == "random" else "fed-dirichlet" + m.group("alpha")
    return (mode, m.group("ds") or "optical", int(m.group("seed")),
            m.group("variant") or "", m.group("btag") or "")


_NUM = r"([-+0-9.eE]+|nan)"


def _f(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_summary(path):
    """Fallback parser for summary.txt. Returns a metrics-like dict."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    out = {}
    # Multi line block: "Test Results:\n  Loss: x\n  Accuracy: x\n  AUC: ..."
    blocks = [("test", r"(?m)^Test Results:\s*\n"),
              ("ko_zero", r"(?m)^Quantum Knockout \(zero\) test metrics:\s*\n"),
              ("ko_random", r"(?m)^Quantum Knockout \(random\) test metrics:\s*\n")]
    for key, head in blocks:
        m = re.search(head + r"((?:[ \t]+\w[\w ]*:\s*\S+\s*\n?)+)", text)
        if m:
            d = {}
            for k, v in re.findall(r"(?m)^[ \t]+(\w[\w ]*?):\s*(\S+)", m.group(1)):
                d[k.strip().lower()] = _f(v)
            if "accuracy" not in d and "acc" in d:
                d["accuracy"] = d["acc"]
            out[key] = d
    # One line: "Test Results - Loss: x, Accuracy: x, AUC: x, IoU: x, Dice: x"
    if "test" not in out:
        m = re.search(r"Test Results - Loss: %s, Accuracy: %s, AUC: %s, IoU: %s, Dice: %s"
                      % ((_NUM,) * 5), text)
        if m:
            out["test"] = dict(zip(("loss", "accuracy", "auc", "iou", "dice"),
                                   map(_f, m.groups())))
    # One line: "Test: Loss=x Acc=x AUC=x IoU=x Dice=x" (last one wins, it is the final)
    if "test" not in out:
        ms = re.findall(r"Test: Loss=%s Acc=%s AUC=%s IoU=%s Dice=%s" % ((_NUM,) * 5), text)
        if ms:
            out["test"] = dict(zip(("loss", "accuracy", "auc", "iou", "dice"),
                                   map(_f, ms[-1])))
    # Knockout one line: "Knockout 'zero' test: IoU=x Dice=x AUC=x"
    for mode in ("zero", "random"):
        if "ko_" + mode in out:
            continue
        m = re.search(r"Knockout '%s' test: IoU=%s Dice=%s AUC=%s" % (mode, _NUM, _NUM, _NUM),
                      text)
        if m:
            out["ko_" + mode] = dict(zip(("iou", "dice", "auc"), map(_f, m.groups())))
    res = {"test": out.get("test", {}), "knockout": {}}
    for mode in ("zero", "random"):
        if "ko_" + mode in out:
            res["knockout"][mode] = out["ko_" + mode]
    return res


def load_run(run_dir):
    """Return (metrics dict, source) or (None, reason)."""
    mj = os.path.join(run_dir, "results", "metrics.json")
    st = os.path.join(run_dir, "results", "summary.txt")
    if os.path.isfile(mj):
        try:
            with open(mj, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("test"):
                return d, "metrics.json"
        except (OSError, ValueError):
            pass
    if os.path.isfile(st):
        d = parse_summary(st)
        if d.get("test"):
            return d, "summary.txt"
    return None, "no metrics"


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return float("nan")
        d = d[k]
    try:
        return float(d)
    except (TypeError, ValueError):
        return float("nan")


def collect(root, variant=None, btag=None):
    """{(mode, dataset, variant, seed): {metric: value, '_dir': name, '_src': source}}."""
    runs, skipped = {}, []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        key = parse_name(name)
        if key is None:
            continue
        mode, ds, seed, var, tag = key
        if (variant is not None and var != variant) or (btag is not None and tag != btag):
            continue
        d, src = load_run(p)
        if d is None:
            skipped.append((name, src))
            continue
        row = {lab: _get(d, path) for lab, path in METRICS}
        row["_dir"], row["_src"] = name, src
        row["_q_backend"] = d.get("q_backend", "")
        k = (mode, ds, var, seed)
        if k in runs:
            skipped.append((name, "duplicate key %s, kept %s" % (k, runs[k]["_dir"])))
            continue
        runs[k] = row
    return runs, skipped


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s, len(xs)


def _sort_key(k):
    return (MODE_ORDER.get(k[0], 2), k[0], DS_ORDER.get(k[1], 9), k[2], k[3])


def _kfmt(k):
    return "%s/%s%s/%d" % (k[0], k[1], k[2], k[3])


def compare(a, b):
    """Rows of per-(mode, dataset) statistics over matched seeds."""
    cells = OrderedDict()
    for k in sorted(set(a) & set(b), key=_sort_key):
        cells.setdefault(k[:3], []).append(k[3])
    rows = []
    for (mode, ds, var), seeds in cells.items():
        for lab, _ in METRICS:
            ma, sa, na = mean_std([a[(mode, ds, var, s)][lab] for s in seeds])
            mb, sb, nb = mean_std([b[(mode, ds, var, s)][lab] for s in seeds])
            rows.append(OrderedDict([
                ("mode", mode), ("dataset", ds), ("variant", var),
                ("seeds", " ".join(map(str, seeds))),
                ("metric", lab), ("a_mean", ma), ("a_std", sa), ("b_mean", mb),
                ("b_std", sb), ("diff", mb - ma), ("n", min(na, nb))]))
    return rows


def _fmt(x, w=8, p=4):
    return ("%*s" % (w, "nan")) if math.isnan(x) else ("%*.*f" % (w, p, x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root_a", help="First results root (for example runs with the torch simulator)")
    ap.add_argument("root_b", help="Second results root (for example runs with PennyLane)")
    ap.add_argument("--variant", default=None,
                    help="Only use runs with this name variant, for example '_reupload_L2' "
                         "(default: all, matched per variant)")
    ap.add_argument("--btag-a", default=None,
                    help="Only use runs in root A with this backend tag ('', '_qtorch', "
                         "'_qpergroup'). Default: any")
    ap.add_argument("--btag-b", default=None, help="Same for root B")
    ap.add_argument("--csv", default=None, help="Also write the table to this CSV file")
    ap.add_argument("--per-run", action="store_true", help="Also print every matched run")
    args = ap.parse_args(argv)

    for r in (args.root_a, args.root_b):
        if not os.path.isdir(r):
            ap.error("not a directory: %s" % r)
    a, skip_a = collect(args.root_a, args.variant, args.btag_a)
    b, skip_b = collect(args.root_b, args.variant, args.btag_b)
    print("A = %s  (%d quantum runs)" % (args.root_a, len(a)))
    print("B = %s  (%d quantum runs)" % (args.root_b, len(b)))
    for tag, sk in (("A", skip_a), ("B", skip_b)):
        for name, why in sk:
            print("  skipped in %s: %s (%s)" % (tag, name, why))
    for tag, x in (("A", a), ("B", b)):
        srcs = {}
        backs = {}
        for r in x.values():
            srcs[r["_src"]] = srcs.get(r["_src"], 0) + 1
            qb = r["_q_backend"] or "missing (torch, not recorded)"
            backs[qb] = backs.get(qb, 0) + 1
        print("  %s sources %s, q_backend %s" % (tag, srcs, backs))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a:
        print("  only in A (unmatched): %s" % ", ".join(_kfmt(k) for k in only_a))
    if only_b:
        print("  only in B (unmatched): %s" % ", ".join(_kfmt(k) for k in only_b))
    matched = sorted(set(a) & set(b))
    print("matched (mode, dataset, variant, seed): %d" % len(matched))
    if not matched:
        return 1

    if args.per_run:
        print()
        print("%-15s %-22s %5s %-14s %8s %8s %9s" % ("mode", "dataset+variant", "seed", "metric",
                                                    "A", "B", "B-A"))
        for k in sorted(matched, key=_sort_key):
            for lab, _ in METRICS:
                va, vb = a[k][lab], b[k][lab]
                print("%-15s %-22s %5d %-14s %s %s %s" % (k[0], k[1] + k[2], k[3], lab, _fmt(va),
                                                          _fmt(vb), _fmt(vb - va, 9)))

    rows = compare(a, b)
    print()
    hdr = "%-15s %-22s %-14s %3s  %8s %8s  %8s %8s  %9s  %s" % (
        "mode", "dataset+variant", "metric", "n", "A_mean", "A_std", "B_mean", "B_std", "B-A", "|B-A|/std")
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for r in rows:
        cell = (r["mode"], r["dataset"], r["variant"])
        if prev is not None and prev != cell:
            print()
        prev = cell
        pooled = math.sqrt((r["a_std"] ** 2 + r["b_std"] ** 2) / 2)
        rel = (abs(r["diff"]) / pooled) if pooled > 0 and not math.isnan(r["diff"]) else float("nan")
        print("%-15s %-22s %-14s %3d  %s %s  %s %s  %s  %s" % (
            r["mode"], r["dataset"] + r["variant"], r["metric"], r["n"], _fmt(r["a_mean"]), _fmt(r["a_std"]),
            _fmt(r["b_mean"]), _fmt(r["b_std"]), _fmt(r["diff"], 9), _fmt(rel, 6, 2)))
    diffs = [abs(r["diff"]) for r in rows if not math.isnan(r["diff"])]
    print()
    print("max |B-A| over all cells and metrics: %s" % (_fmt(max(diffs), 0, 6) if diffs else "nan"))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("wrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
