#!/usr/bin/env python3
"""Accuracy, radius-cost and preconditioner figures of the paper (Geometry-A).

    python accuracy.py run  [--np 4] [--max-level 4] [--figures all]
    python accuracy.py plot [--formats pdf,png]
    python accuracy.py all

`run` executes the coupled_diffusion example for every configuration behind
the figures (results are cached as JSON in results/accuracy, so it can be
interrupted and restarted); `plot` draws the figures into figures/accuracy
under the file names used in the paper.

Geometry-A: two cubes of side 2, (-2,0)x(-1,1)^2 and (0,2)x(-1,1)^2, each a
single coarse cell refined k times, so h = 2/2^k. The paper's levels are
k = 0..6 (conforming tests) and master k = 2..6 with the slave one level
coarser (h_2 = 2 h_1) for the non-conforming ones. The default --max-level
of 4 keeps everything laptop-sized; use --max-level 6 on a workstation or a
cluster (the finest levels have millions of DoFs, and the large-radius RBF
runs of the finest levels are very slow, which is what the paper's Fig. on the
CPU time of the interpolation shows).

Which figure comes from which runs
  P1_convergence, P2_convergence      conforming, INTERNODES vs single-domain solve
  P1P1_slave_convergence_...          P1/P1, non-conforming, Lagrange vs RBF r_f=1
  P2P3_convergence_...                P2 master / P3 slave, Lagrange vs RBF r_f=1,2,5,10
  RBF_radius_vs_time_hslave(_gray)    CPU time of the RBF interpolation, from the P2/P3 RBF runs
  GMRES_iterations_...                P2/P2, Lagrange, with and without the Schur preconditioner
  P1P2_slave_convergence,             hexahedral master / tetrahedral slave, and the reverse,
  P1P1_P2P2_slave_convergence         P1 (r_f=1) and P2 (r_f=5)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# numpy and matplotlib are only needed by `plot`; `run` uses the standard
# library only, so it works on a bare cluster Python.
try:
    import numpy as np
    import style
except ImportError:
    np = style = None

from lib import ROOT, box_case, run_case  # noqa: E402

RESULTS = ROOT / "results" / "accuracy"
FIGURES = ROOT / "figures" / "accuracy"

RADIUS_FACTORS = (1, 2, 5, 10)
FIGURE_KEYS = ("conforming", "p1p1", "p2p3", "gmres", "hybrid")

# Label of the x axis of the non-conforming figures. The paper calls it h_2
# (slave mesh size), but the plotted values are the *master* mesh sizes
# h_1 = h_2/2 = 1/2 ... 1/32 (see README.md), so by default the label says h_1;
# --paper-labels restores the paper's.
NC_LABEL = r"$h_1$"
NC_LABEL_PAPER = r"$h_2$"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def conforming_levels(max_level):
    return list(range(0, min(6, max_level) + 1))


def nonconforming_levels(max_level):
    """Master refinement k = 2..6; the slave has k-1 (h_2 = 2 h_1)."""
    return list(range(2, min(6, max_level) + 1))


def cases(key, max_level):
    """(name, parameter sections) of every run behind a group of figures."""
    out = []
    if key == "conforming":
        for p in (1, 2):
            for k in conforming_levels(max_level):
                out.append((f"conf_p{p}_k{k}", box_case(k, master_degree=p)))
                out.append((f"mono_p{p}_k{k}", box_case(k, master_degree=p, monolithic=True)))
    elif key == "p1p1":
        for k in nonconforming_levels(max_level):
            out.append((f"nc_p1_lag_k{k}", box_case(k, k - 1)))
            out.append((f"nc_p1_rbf1_k{k}", box_case(k, k - 1, rbf_factor=1)))
    elif key == "p2p3":
        for k in nonconforming_levels(max_level):
            out.append((f"nc_p23_lag_k{k}", box_case(k, k - 1, master_degree=2, slave_degree=3)))
            for rf in RADIUS_FACTORS:
                out.append(
                    (
                        f"nc_p23_rbf{rf}_k{k}",
                        box_case(k, k - 1, master_degree=2, slave_degree=3, rbf_factor=rf),
                    )
                )
    elif key == "gmres":
        for k in nonconforming_levels(max_level):
            out.append((f"gmres_pc_k{k}", box_case(k, k - 1, master_degree=2)))
            out.append((f"gmres_nopc_k{k}", box_case(k, k - 1, master_degree=2, preconditioner=False)))
    elif key == "hybrid":
        for k in nonconforming_levels(max_level):
            for tag, (mc, sc) in (("hexmaster", ("hex", "tet")), ("tetmaster", ("tet", "hex"))):
                for p, rf in ((1, 1), (2, 5)):
                    out.append(
                        (
                            f"hyb_{tag}_p{p}_k{k}",
                            box_case(k, k - 1, master_degree=p, rbf_factor=rf, master_cells=mc, slave_cells=sc),
                        )
                    )
    else:
        raise ValueError(key)
    return out


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def error(name):
    r = load(name)
    return r["broken_H1_error"] if r else None


def collect(name_of, levels, quantity=error):
    """([h values 2/2^k], [quantity]) over the levels for which results exist."""
    xs, ys = [], []
    for k in levels:
        y = quantity(name_of(k))
        if y is not None:
            xs.append(2.0 / 2**k)
            ys.append(y)
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def convergence_axes(ax, xlabel, ylabel, curves, slopes, legend_loc="lower left", legend_ncol=1):
    """Log-log convergence plot with the paper's frame, grid and slope triangles.

    curves: list of (label, x, y, line style kwargs); slopes: list of
    (rate, anchor) where anchor = 'top' or 'bottom' picks the triangle position.
    """
    for label, x, y, kw in curves:
        ax.loglog(x, y, label=label, **kw)
    style.grid(ax, log_x=True, log_y=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    all_x = np.concatenate([c[1] for c in curves])
    all_y = np.concatenate([c[2] for c in curves])
    ax.set_xlim(all_x.min() / 1.6, all_x.max() * 1.25)
    ax.set_ylim(*style.decade_limits(all_y, pad=0.35))
    for rate, series_index in slopes:
        _, x, y, _ = curves[series_index]
        i = int(np.argmin(x))
        style.slope_triangle(ax, x[i], y[i] * 1.25, rate)
    ax.legend(loc=legend_loc, ncol=legend_ncol)


def fig_conforming(p, formats):
    levels = range(0, 7)
    xi, yi = collect(lambda k: f"conf_p{p}_k{k}", levels)
    xm, ym = collect(lambda k: f"mono_p{p}_k{k}", levels)
    if len(xi) < 2:
        return print(f"  P{p}_convergence: not enough results, skipped")
    fig, ax = style.new_axes()
    convergence_axes(
        ax,
        r"$h$",
        r"Error (H$^1$-norm)" if p == 1 else "Error (Broken-norm)",
        [
            ("INTERNODES", xi, yi, style.style_series(color=style.BLUE, marker="D", linestyle="-")),
            ("MONOLITHIC", xm, ym, style.style_series(color=style.ORANGE, marker="s", linestyle="--")),
        ],
        [(p, 0)],
    )
    style.save(fig, f"P{p}_convergence", FIGURES, formats)


def fig_p1p1(formats, xlabel):
    levels = range(2, 7)
    xl, yl = collect(lambda k: f"nc_p1_lag_k{k}", levels)
    xr, yr = collect(lambda k: f"nc_p1_rbf1_k{k}", levels)
    if len(xl) < 2:
        return print("  P1P1_slave_convergence_Lagrange_vs_RBF: not enough results, skipped")
    fig, ax = style.new_axes()
    convergence_axes(
        ax,
        xlabel,
        "Error (Broken-norm)",
        [
            ("Lagrange", xl, yl, style.style_series(color="black", marker="o", linestyle="-")),
            ("RBF, $r_f=1$", xr, yr, style.style_series(color=style.BLUE, marker="s", linestyle="--")),
        ],
        [(1, 0)],
        legend_loc="lower right",
    )
    style.save(fig, "P1P1_slave_convergence_Lagrange_vs_RBF", FIGURES, formats)


def fig_p2p3(formats, xlabel):
    levels = range(2, 7)
    xl, yl = collect(lambda k: f"nc_p23_lag_k{k}", levels)
    if len(xl) < 2:
        return print("  P2P3_convergence_Lagrange_vs_RBF: not enough results, skipped")
    look = {
        1: (style.YELLOW, "s", "--"),
        2: (style.ORANGE, "D", "-."),
        5: (style.BLUE, "^", ":"),
        10: (style.PURPLE, "v", "-"),
    }
    curves = [("Lagrange", xl, yl, style.style_series(color="black", marker="o", linestyle="-"))]
    for rf in RADIUS_FACTORS:
        x, y = collect(lambda k: f"nc_p23_rbf{rf}_k{k}", levels)
        if len(x):
            color, marker, ls = look[rf]
            curves.append((f"RBF, $r_f={rf}$", x, y, style.style_series(color=color, marker=marker, linestyle=ls)))
    fig, ax = style.new_axes()
    convergence_axes(ax, xlabel, "Error (Broken norm)", curves, [(2, 0)], legend_loc="lower right")
    style.save(fig, "P2P3_convergence_Lagrange_vs_RBF", FIGURES, formats)


def interpolation_cost(result):
    """The paper's 'CPU time for the RBF interpolation and normal-derivative
    evaluation': the wall time (max over ranks) of the RBF interpolations
    (Phi solve + evaluation) and of the normal-derivative evaluations."""
    t = result["timings"]
    sections = ("interpolate: solve Phi", "interpolate: evaluate at points", "normal derivative")
    return sum(t.get(s, {}).get("wall_max", 0.0) for s in sections)


def fig_radius_cost(formats, xlabel):
    levels = [k for k in range(2, 7) if load(f"nc_p23_rbf1_k{k}")]
    if not levels:
        return print("  RBF_radius_vs_time_hslave: no results, skipped")
    width = 0.2
    cost = {
        rf: [
            interpolation_cost(load(f"nc_p23_rbf{rf}_k{k}")) if load(f"nc_p23_rbf{rf}_k{k}") else np.nan
            for k in levels
        ]
        for rf in RADIUS_FACTORS
    }
    positive = np.array([c for row in cost.values() for c in row if c > 0])
    if not len(positive):
        return print("  RBF_radius_vs_time_hslave: no timings, skipped")
    # y range: whole decades around the data (the paper's is 0.1 s ... 1e4 s)
    ylim = (10 ** np.floor(np.log10(positive.min())), 10 ** np.ceil(np.log10(positive.max() * 3)))
    for gray in (False, True):
        fig, ax = style.new_axes()
        for j, rf in enumerate(RADIUS_FACTORS):
            heights = cost[rf]
            color = ("0.25", "0.5", "0.75", "1.0")[j] if gray else (style.BLUE, style.YELLOW, style.GREEN, style.ORANGE)[j]
            ax.bar(
                np.arange(len(levels)) + (j - 1.5) * width,
                heights,
                width,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                label=f"$r_f={rf}$",
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_xticks(range(len(levels)), [f"1/{2 ** (k - 1)}" for k in levels])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CPU time (s)")
        ax.set_ylim(*ylim)
        style.grid(ax, log_y=True)
        ax.grid(False, axis="x")
        ax.legend(loc="upper left")
        style.save(fig, "RBF_radius_vs_time_hslave_gray" if gray else "RBF_radius_vs_time_hslave", FIGURES, formats)


def fig_gmres(formats):
    dofs, with_pc, without_pc = [], [], []
    for k in range(2, 7):
        a, b = load(f"gmres_pc_k{k}"), load(f"gmres_nopc_k{k}")
        if a and b:
            dofs.append(a["subdomains"]["master"]["n_dofs"])
            with_pc.append(a["gmres_iterations"])
            without_pc.append(b["gmres_iterations"])
    if len(dofs) < 2:
        return print("  GMRES_iterations_precond_vs_unprecond: not enough results, skipped")
    fig, ax = style.new_axes()
    ax.semilogx(dofs, with_pc, label="GMRES + preconditioner", **style.style_series(color=style.BLUE, marker="o", linestyle="-"))
    ax.semilogx(dofs, without_pc, label="GMRES (no preconditioner)", **style.style_series(color=style.ORANGE, marker="s", linestyle="--"))
    style.grid(ax, log_x=True)
    ax.set_xlabel("DOFs (master mesh)")
    ax.set_ylabel("GMRES iterations")
    ax.set_ylim(0, max(100, 1.1 * max(without_pc)))
    ax.legend(loc="upper left")
    style.save(fig, "GMRES_iterations_precond_vs_unprecond", FIGURES, formats)


def fig_hybrid(formats, xlabel):
    levels = range(2, 7)
    for tag, filename, labels in (
        ("hexmaster", "P1P2_slave_convergence", ("P1", "P2")),
        ("tetmaster", "P1P1_P2P2_slave_convergence", ("P1$-$P1", "P2$-$P2")),
    ):
        x1, y1 = collect(lambda k: f"hyb_{tag}_p1_k{k}", levels)
        x2, y2 = collect(lambda k: f"hyb_{tag}_p2_k{k}", levels)
        if len(x1) < 2 or len(x2) < 2:
            print(f"  {filename}: not enough results, skipped")
            continue
        fig, ax = style.new_axes()
        convergence_axes(
            ax,
            xlabel,
            "Error (Broken Norm)" if tag == "hexmaster" else "Error (Broken-norm)",
            [
                (labels[0], x1, y1, style.style_series(color="black", marker="o", linestyle="-")),
                (labels[1], x2, y2, style.style_series(color=style.BLUE, marker="s", linestyle="--")),
            ],
            [(1, 0), (2, 1)],
            legend_loc="lower right",
        )
        style.save(fig, filename, FIGURES, formats)


def plot_all(formats, paper_labels=False):
    xlabel = NC_LABEL_PAPER if paper_labels else NC_LABEL
    fig_conforming(1, formats)
    fig_conforming(2, formats)
    fig_p1p1(formats, xlabel)
    fig_p2p3(formats, xlabel)
    fig_radius_cost(formats, xlabel)
    fig_gmres(formats)
    fig_hybrid(formats, xlabel)


# ---------------------------------------------------------------------------
def run_all(args):
    keys = FIGURE_KEYS if args.figures == "all" else args.figures.split(",")
    todo = []
    for key in keys:
        todo += cases(key, args.max_level)
    seen, unique = set(), []
    for name, sections in todo:
        if name not in seen:
            seen.add(name)
            unique.append((name, sections))
    print(f"{len(unique)} configurations (np={args.np}, levels up to {args.max_level}); results in {RESULTS}")
    for name, sections in unique:
        run_case(
            name,
            sections,
            RESULTS,
            np=args.np,
            exe=args.exe,
            timeout=args.timeout,
            force=args.force,
            retry_failed=args.retry_failed,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "plot", "all"))
    parser.add_argument("--np", type=int, default=4, help="MPI ranks per run (default 4)")
    parser.add_argument("--max-level", type=int, default=4, help="finest refinement level (paper: 6; default 4)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds allowed per run (default 1800)")
    parser.add_argument("--figures", default="all", help=f"comma-separated subset of {','.join(FIGURE_KEYS)}")
    parser.add_argument("--exe", help="path of the coupled_diffusion executable")
    parser.add_argument("--force", action="store_true", help="rerun even if results exist")
    parser.add_argument("--retry-failed", action="store_true", help="rerun cases that failed or timed out before")
    parser.add_argument("--formats", default="pdf,png", help="output formats (default pdf,png; the paper uses eps)")
    parser.add_argument("--paper-labels", action="store_true", help="label the x axis of the non-conforming figures h_2, as the paper does")
    args = parser.parse_args()

    if args.command in ("plot", "all") and style is None:
        sys.exit("plot needs numpy and matplotlib: pip install -r requirements.txt")
    if args.command in ("run", "all"):
        run_all(args)
    if args.command in ("plot", "all"):
        plot_all(tuple(args.formats.split(",")), args.paper_labels)


if __name__ == "__main__":
    main()
