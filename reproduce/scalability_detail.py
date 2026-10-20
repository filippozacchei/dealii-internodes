#!/usr/bin/env python3
"""Detailed strong-scalability figures: every timer the code records, not
only the six phases of the paper's figures, plus the inner-solve iteration
counts. For inspection, not for the paper.

    python scalability_detail.py --tag galileo100-final [--tests 1,2,3,4,5]

For each test writes, to figures/scalability_detail_<tag>/:
    testN_time.png      wall time (max over ranks) per timer, by group
    testN_speedup.png   speed-up T(p0)/T(p) per timer, by group
    testN_iterations.png  CG iterations per solve, by kind of inner solve
With repeated runs (testN_npP_rK), every point is the median over repeats
and the bars span min-max.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scalability import ROOT, load_test, median_min_max  # noqa: E402

# Timer sections by role. Names are matched after stripping leading blanks
# (nested sections are indented in some versions of the code).
GROUPS = [
    (
        "Setup (once)",
        [
            "setup: subproblem (DoFs, interface, matrices)",
            "setup: interface-DoFs-last renumbering",
            "setup: RBF operators (Phi, AMG, scaling)",
            "Setup destination points",
        ],
    ),
    (
        "Assembly",
        [
            "Step 0: assembly (master/slave)",
            "Step 0: assembly (preconditioners)",
            "Assembly Preconditioner",
            "Step 0: homogeneous-phase data update",
        ],
    ),
    (
        "Algorithm steps",
        [
            "Step 1: u^(f) on master/slave",
            "Step 2: interface residual chi",
            "Step 3: interface solve (total)",
            "Step 4: u^(lambda) on master/slave",
            "driver: solve()",
        ],
    ),
    (
        "Inside Step 3",
        [
            "Step 3a: initialize Schur preconditioner",
            "Step 3b: solve S lambda = chi",
            "Step 3 (GMRES): Q21 interpolation",
            "Step 3 (GMRES): subdomain solves",
            "Step 3 (GMRES): residual computation",
            "Schur preconditioner: apply",
        ],
    ),
    (
        "Kernels (all calls)",
        [
            "subdomain linear solve",
            "interpolate: solve Phi",
            "interpolate: evaluate at points",
            "interpolateResidual: solve M_gamma",
            "normal derivative",
            "step_rhs: f_k - A_{k,Gamma_k} lambda_k",
        ],
    ),
]


def timer(result, name):
    for key, value in result["timings"].items():
        if key.strip() == name:
            return value["wall_max"]
    return None


def series(data, name):
    """(cores, median, min, max) of a timer over the repeats; only core
    counts where every repeat has it."""
    cores, stats = [], []
    for p, runs in data.items():
        values = [timer(r, name) for r in runs]
        if all(v is not None and v > 0 for v in values):
            cores.append(p)
            stats.append(median_min_max(values))
    return np.array(cores), np.array(stats).reshape(-1, 3)


def short(name):
    return name.replace(" on master/slave", "").replace("(GMRES): ", "").replace(
        "step_rhs: f_k - A_{k,Gamma_k} lambda_k", "step_rhs"
    )


def plot_groups(data, n, kind, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()
    all_cores = np.array(list(data))
    p0 = all_cores[0]
    for ax, (title, names) in zip(axes, GROUPS):
        ideal = all_cores / p0 if kind == "speedup" else None
        for name in names:
            cores, st = series(data, name)
            if len(cores) < 2:
                continue
            med, lo, hi = st[:, 0], st[:, 1], st[:, 2]
            if kind == "speedup":
                y, ylo, yhi = med[0] / med, med[0] / hi, med[0] / lo
            else:
                y, ylo, yhi = med, lo, hi
            line = ax.loglog(cores, y, marker="o", ms=4, label=short(name))[0]
            if (yhi > ylo).any():
                ax.errorbar(cores, y, yerr=[y - ylo, yhi - y], fmt="none",
                            ecolor=line.get_color(), elinewidth=0.8, capsize=2)
            if kind == "time":
                # dashed ideal-scaling reference through this timer's first point
                ax.loglog(cores, y[0] * cores[0] / cores, ls=":", lw=0.8, color=line.get_color())
        if ideal is not None:
            ax.loglog(all_cores, ideal, "--", color="gray", lw=1.2, label="ideal")
        ax.set_title(title)
        ax.set_xticks(all_cores, [str(c) for c in all_cores], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("cores")
        ax.set_ylabel("speed-up T(p0)/T(p)" if kind == "speedup" else "time (s)  [dotted: ideal]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[-1].axis("off")
    meta = next(iter(data.values()))[0]
    n_rep = sorted({len(r) for r in data.values()})
    fig.suptitle(f"Test {n} -- {kind} of every timer (commit {meta.get('git_version', '?')}, "
                 f"repeats per point {n_rep}, median with min-max)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_iterations(data, n, out):
    kinds = sorted({k for runs in data.values() for r in runs for k in r.get("cg_statistics", {})})
    if not kinds:
        return False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_cores = np.array(list(data))
    for kind in kinds:
        cores, its = [], []
        for p, runs in data.items():
            vals = [r["cg_statistics"][kind]["iterations"] / max(r["cg_statistics"][kind]["solves"], 1)
                    for r in runs if kind in r.get("cg_statistics", {})]
            if vals and any(v > 0 for v in vals):
                cores.append(p)
                its.append(np.median(vals))
        if cores:
            ax.semilogx(cores, its, marker="o", ms=4, label=kind)
    ax.set_yscale("log")
    ax.set_xticks(all_cores, [str(c) for c in all_cores], fontsize=8)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("cores")
    ax.set_ylabel("CG iterations per solve (median)")
    ax.set_title(f"Test {n} -- inner-solve iterations (flat = cost growth is per iteration, not iteration count)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tests", default="1,2,3,4,5")
    args = parser.parse_args()
    outdir = ROOT / "figures" / f"scalability_detail_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    for n in map(int, args.tests.split(",")):
        data = load_test(n, args.tag)
        if len(data) < 2:
            print(f"test {n}: fewer than two core counts, skipped")
            continue
        plot_groups(data, n, "time", outdir / f"test{n}_time.png")
        plot_groups(data, n, "speedup", outdir / f"test{n}_speedup.png")
        plot_iterations(data, n, outdir / f"test{n}_iterations.png")
        print(f"test {n}: wrote {outdir}/test{n}_{{time,speedup,iterations}}.png")


if __name__ == "__main__":
    main()
