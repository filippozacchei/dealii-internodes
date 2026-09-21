#!/usr/bin/env python3
"""Strong-scalability figures of the paper (the five tests of Table 4).

On a cluster (SLURM):
    python scalability.py prepare --account <account> --partition <partition> \\
        [--modules "module load ..."] [--tests 1,2,3,4,5] [--cores 48,96,192,240,288,384,480,768]
    bash results/scalability/submit_all.sh          # afterwards
    python scalability.py plot

`prepare` writes, for every test and core count, a parameter file and a batch
script (results/scalability/testN_npP.{prm,sh}); each job writes
testN_npP.json. Nothing is run by `prepare` itself.

On a laptop, a scaled-down version of the same pipeline (every refinement
level lowered by --levels-down, few ranks) checks the scripts end to end; its
timings say nothing about scalability:
    python scalability.py run --tests 1,2 --cores 1,2,4 --levels-down 3
    python scalability.py plot

The per-test configuration lives in testN.prm (this directory).

The phases plotted are those of the paper:
  Assembly Interpolation Operators   destination points + RBF matrix setup
  Assembly Internal Operators        subdomain matrices ("Step 0")
  Step 1 / 2 / 3 / 4                 as in Algorithm 3 of the paper
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from lib import ROOT, read_prm, run_case  # noqa: E402

RESULTS = ROOT / "results" / "scalability"
FIGURES = ROOT / "figures" / "scalability"

CORES = (48, 96, 192, 240, 288, 384, 480, 768)  # the paper's counts; p0 = 48

# File names of the paper's figures (without the _scalability / _total suffix).
TESTS = {
    1: {"figure": "Lagrange_Cubi_p1r8-p1r7_galileo", "suffix": "_bis"},
    2: {"figure": "RBF_Cubi_p1r8-p1r7_r=1_galileo", "suffix": ""},
    3: {"figure": "RBF_Cubi_p2r7-p4r5_r=1_galileo", "suffix": ""},
    4: {"figure": "RBF_Shell_p1r7-p1r6_r=05_galileo", "suffix": ""},
    5: {"figure": "RBF_Shell_p2r6-p2r5_r=10_galileo", "suffix": ""},
}

# (label, timer sections of the results file that are added up)
PHASES = [
    (
        "Assembly Interpolation Operators",
        ("Setup destination points", "setup: RBF operators (Phi, AMG, scaling)"),
    ),
    ("Assembly Internal Operators", ("Step 0: assembly (master/slave)",)),
    ("Step 1", ("Step 1: u^(f) on master/slave",)),
    ("Step 2", ("Step 2: interface residual chi",)),
    ("Step 3", ("Step 3: interface solve (total)",)),
    ("Step 4", ("Step 4: u^(lambda) on master/slave",)),
]

# Line styles of the paper's figures.
LOOK = {
    "Ideal": dict(color=style.GRAY, linestyle="--", linewidth=1.6),
    "Assembly Interpolation Operators": dict(color=style.ORANGE, marker="s", linestyle="-."),
    "Assembly Internal Operators": dict(color="black", marker="o", linestyle="-"),
    "Step 1": dict(color=style.BLUE, marker=">", linestyle="--"),
    "Step 2": dict(color=style.ORANGE, marker="^", linestyle=":"),
    "Step 3": dict(color=style.GREEN, marker="s", linestyle="-."),
    "Step 4": dict(color=style.PURPLE, marker="D", linestyle="--"),
}


# ---------------------------------------------------------------------------
def test_sections(n, levels_down=0):
    """Parameter sections of test n from testN.prm, with all refinement levels
    lowered by levels_down (0 = the paper's)."""
    sections = read_prm(ROOT / f"test{n}.prm")
    for key in ("Global refinement master", "Global refinement slave", "Shell refinement master", "Shell refinement slave"):
        for entries in sections.values():
            if key in entries:
                entries[key] = max(0, int(entries[key]) - levels_down)
    return sections


def job_script(name, np_, prm, args):
    nodes = -(-np_ // args.cores_per_node)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={min(np_, args.cores_per_node)}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --output={RESULTS / name}.log",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    if args.modules:
        lines.append(args.modules)
    lines.append(f"{args.launcher} {args.exe} {prm}")
    return "\n".join(lines) + "\n"


def prepare(args):
    from lib import DEFAULT_EXE, prm_text

    args.exe = args.exe or DEFAULT_EXE
    RESULTS.mkdir(parents=True, exist_ok=True)
    submit = ["#!/bin/bash", f"cd {ROOT}"]
    for n in map(int, args.tests.split(",")):
        for p in map(int, args.cores.split(",")):
            name = f"test{n}_np{p}"
            sections = test_sections(n, args.levels_down)
            sections.setdefault("Run", {})["Results file"] = str(RESULTS / f"{name}.json")
            prm = RESULTS / f"{name}.prm"
            prm.write_text(prm_text(sections))
            script = RESULTS / f"{name}.sh"
            script.write_text(job_script(name, p, prm, args))
            submit.append(f"sbatch {script}")
    (RESULTS / "submit_all.sh").write_text("\n".join(submit) + "\n")
    print(f"Wrote {len(submit) - 2} jobs to {RESULTS}; submit them with: bash {RESULTS / 'submit_all.sh'}")


def run_local(args):
    for n in map(int, args.tests.split(",")):
        print(f"Test {n} (levels lowered by {args.levels_down})")
        for p in map(int, args.cores.split(",")):
            run_case(
                f"test{n}_np{p}",
                test_sections(n, args.levels_down),
                RESULTS,
                np=p,
                exe=args.exe,
                timeout=args.timeout,
                force=args.force,
                retry_failed=args.retry_failed,
            )


# ---------------------------------------------------------------------------
def load_test(n):
    """{cores: results} for test n."""
    found = {}
    for path in RESULTS.glob(f"test{n}_np*.json"):
        p = int(path.stem.split("_np")[1])
        found[p] = json.loads(path.read_text())
    return dict(sorted(found.items()))


def phase_times(result):
    t = result["timings"]
    return {label: sum(t.get(s, {}).get("wall_max", 0.0) for s in sections) for label, sections in PHASES}


def plot_test(n, formats):
    data = load_test(n)
    if len(data) < 2:
        return print(f"  test {n}: fewer than two core counts, skipped")
    cores = np.array(list(data))
    p0 = cores[0]
    times = {label: np.array([phase_times(r)[label] for r in data.values()]) for label, _ in PHASES}
    name = TESTS[n]["figure"]
    suffix = TESTS[n]["suffix"]

    for kind in ("scalability", "total"):
        fig, ax = style.new_axes(style.SCALABILITY_FIGSIZE)
        ideal = cores / p0 if kind == "scalability" else p0 / cores
        ax.loglog(cores, ideal, label="Ideal", **LOOK["Ideal"])
        for label, _ in PHASES:
            t = times[label]
            valid = t > 0
            if not valid.any():
                continue
            y = t[valid][0] / t[valid] if kind == "scalability" else t[valid]
            look = dict(LOOK[label])
            ax.loglog(cores[valid], y, label=label, **style.style_series(**look))
        style.grid(ax, log_x=True, log_y=True)
        ax.set_xticks(cores, [str(c) for c in cores])
        ax.xaxis.set_minor_locator(mpl_null())
        ax.set_xlim(cores[0] / 1.06, cores[-1] * 1.25)
        ax.set_xlabel(r"$N_{\mathrm{cores}}$")
        if kind == "scalability":
            ax.set_ylabel(r"$T(p_0)/T(p)$")
            ax.set_ylim(1, 1.15 * cores[-1] / p0)
            ax.legend(loc="upper left")
            out = f"{name}_scalability{suffix}"
        else:
            ax.set_ylabel("Time (s)")
            ax.legend(loc="lower left")
            out = f"{name}_total{suffix}"
        style.save(fig, out, FIGURES, formats)


def mpl_null():
    from matplotlib.ticker import NullLocator

    return NullLocator()


def plot_all(args):
    for n in map(int, args.tests.split(",")):
        plot_test(n, tuple(args.formats.split(",")))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "run", "plot"))
    parser.add_argument("--tests", default="1,2,3,4,5")
    parser.add_argument("--cores", default=",".join(map(str, CORES)))
    parser.add_argument("--levels-down", type=int, default=0, help="lower every refinement level by this (0 = the paper's)")
    parser.add_argument("--exe", help="path of the coupled_diffusion executable")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per local run")
    cluster = parser.add_argument_group("prepare (SLURM)")
    cluster.add_argument("--account")
    cluster.add_argument("--partition")
    cluster.add_argument("--modules", help='line(s) put before the launch, e.g. "module load openmpi dealii"')
    cluster.add_argument("--cores-per-node", type=int, default=48)
    cluster.add_argument("--time-limit", default="04:00:00")
    cluster.add_argument("--launcher", default="srun", help="MPI launcher in the batch scripts (default srun)")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run_local(args)
    else:
        plot_all(args)


if __name__ == "__main__":
    main()
