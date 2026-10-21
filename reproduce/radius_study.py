#!/usr/bin/env python3
"""Figure of the RBF radius study (Geometry-B, P2/P2, 768 processes):

    python radius_study.py [--formats png,pdf,eps]

reads results/scalability/radius_study/shell{54,65}_rf<r_f>_np768.json (see
the README there) and writes figures/radius_study/RBF_radius_study.*:
(a) broken H1 error vs r_f on the two shell meshes, (b) cost on the finer
mesh vs r_f: one-time setup of the interpolation operators (dominated by the
factorization of Phi) and the interface solve (Step 3).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ROOT  # noqa: E402

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import style
except ImportError:
    sys.exit("needs numpy and matplotlib: pip install -r requirements.txt")

RESULTS = ROOT / "results" / "scalability" / "radius_study"
FIGURES = ROOT / "figures" / "radius_study"
MESHES = {
    "shell54": r"shell 5/4 ($1.8\times10^{6}$ DoFs)",
    "shell65": r"shell 6/5 ($1.4\times10^{7}$ DoFs)",
}


def timer(result, name):
    return sum(v["wall_max"] for k, v in result["timings"].items() if k.strip() == name)


def load(mesh):
    """{r_f: result} of one mesh."""
    runs = {}
    for path in RESULTS.glob(f"{mesh}_rf*_np768.json"):
        r = json.loads(path.read_text())
        runs[r["rbf_radius_factor"]] = r
    return dict(sorted(runs.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", default="png,pdf")
    formats = tuple(parser.parse_args().formats.split(","))

    style.setup()
    fig, (ax_err, ax_cost) = plt.subplots(1, 2, figsize=(2 * style.ACCURACY_FIGSIZE[0], style.ACCURACY_FIGSIZE[1]),
                                          constrained_layout=True)

    # (a) accuracy
    for (mesh, label), color, marker in zip(MESHES.items(), (style.BLUE, style.ORANGE), ("o", "s")):
        runs = load(mesh)
        rf = np.array(list(runs))
        err = np.array([r["broken_H1_error"] for r in runs.values()])
        ax_err.loglog(rf, err, label=label, **style.style_series(color=color, marker=marker, linestyle="-"))
    ax_err.set_xlabel(r"radius factor $r_f$")
    ax_err.set_ylabel(r"broken $H^1$ error")
    ax_err.set_title("(a) accuracy", fontsize=11)
    style.grid(ax_err, log_x=True, log_y=True)
    ax_err.legend(loc="lower left")

    # (b) cost on the finer mesh, 768 processes
    runs = load("shell65")
    rf = np.array(list(runs))
    setup_t = np.array([timer(r, "setup: RBF operators (Phi, AMG, scaling)") + timer(r, "Setup destination points")
                        for r in runs.values()])
    step3 = np.array([timer(r, "Step 3: interface solve (total)") for r in runs.values()])
    ax_cost.loglog(rf, setup_t, label="interpolation operators (once)",
                   **style.style_series(color=style.ORANGE, marker="s", linestyle="-."))
    ax_cost.loglog(rf, step3, label="interface solve (Step 3)",
                   **style.style_series(color=style.GREEN, marker="o", linestyle="-"))
    # Phi solved iteratively instead (results/scalability/galileo100-phi-{amg,none}/,
    # r_f = 50, 768 processes): the first solve with the master Phi had not
    # finished when the 1 h job limit was reached -- a lower bound.
    ax_cost.annotate("", xy=(50, 9000), xytext=(50, 3600),
                     arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    ax_cost.plot([50], [3600], marker="v", color="black", markerfacecolor="white", linestyle="none")
    ax_cost.text(44, 3600, "one $\\Phi$ solve with\nCG or AMG-CG: $>1$ h", ha="right", va="center", fontsize=9)
    ax_cost.set_xlabel(r"radius factor $r_f$")
    ax_cost.set_ylabel("time on 768 processes (s)")
    ax_cost.set_title(r"(b) cost, shell 6/5", fontsize=11)
    style.grid(ax_cost, log_x=True, log_y=True)
    ax_cost.set_ylim(10, 1.5e4)
    ax_cost.legend(loc="center left")

    for ax in (ax_err, ax_cost):
        ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    style.save(fig, "RBF_radius_study", FIGURES, formats)


if __name__ == "__main__":
    main()
