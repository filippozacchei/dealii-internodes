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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# numpy and matplotlib are only needed by `plot`; `prepare` and `run` use the
# standard library only, so they work on a bare cluster Python.
try:
    import numpy as np
    import style
except ImportError:
    np = style = None

from lib import ROOT, read_prm, run_case  # noqa: E402

def _results_dir(tag=None):
    return ROOT / "results" / (f"scalability_{tag}" if tag else "scalability")


RESULTS = _results_dir()
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
    (
        "Assembly Internal Operators",
        ("Step 0: assembly (master/slave)", "Step 0: homogeneous-phase data update"),
    ),
    ("Step 1", ("Step 1: u^(f) on master/slave",)),
    ("Step 2", ("Step 2: interface residual chi",)),
    ("Step 3", ("Step 3: interface solve (total)",)),
    ("Step 4", ("Step 4: u^(lambda) on master/slave",)),
]

def _look():
    """Line styles of the paper's figures."""
    return {
        "Ideal": dict(color=style.GRAY, linestyle="--", linewidth=1.6),
        "Assembly Interpolation Operators": dict(color=style.ORANGE, marker="s", linestyle="-."),
        "Assembly Internal Operators": dict(color="black", marker="o", linestyle="-"),
        "Step 1": dict(color=style.BLUE, marker=">", linestyle="--"),
        "Step 2": dict(color=style.ORANGE, marker="^", linestyle=":"),
        "Step 3": dict(color=style.GREEN, marker="s", linestyle="-."),
        "Step 4": dict(color=style.PURPLE, marker="D", linestyle="--"),
        "Whole run": dict(color="0.35", marker="x", linestyle="-"),
    }


# ---------------------------------------------------------------------------
def _run_name(n, p, k=1, repeats=1):
    """File stem of run k (1-based) of test n on p cores; the suffix _rK is
    only used when there are several runs of each case."""
    return f"test{n}_np{p}" + (f"_r{k}" if repeats > 1 else "")


def apply_overrides(sections, overrides):
    """Applies --set "Section/Name=value" overrides to the parameter sections."""
    for item in overrides:
        key, equals, value = item.partition("=")
        section, slash, name = key.partition("/")
        if not (equals and slash and section.strip() and name.strip()):
            sys.exit(f'--set expects "Section/Name=value", got {item!r}')
        sections.setdefault(section.strip(), {})[name.strip()] = value.strip()
    return sections


def test_sections(n, levels_down=0):
    """Parameter sections of test n from testN.prm, with all refinement levels
    lowered by levels_down (0 = the paper's)."""
    sections = read_prm(ROOT / f"test{n}.prm")
    for key in ("Global refinement master", "Global refinement slave", "Shell refinement master", "Shell refinement slave"):
        for entries in sections.values():
            if key in entries:
                entries[key] = max(0, int(entries[key]) - levels_down)
    return sections


def job_script(name, np_, prm, args, results):
    nodes = -(-np_ // args.cores_per_node)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={min(np_, args.cores_per_node)}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --output={results / name}.log",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    # sbatch passes on the submitting shell's environment; a SLURM_CPUS_PER_TASK
    # left over from an earlier `salloc --cpus-per-task=...` there would make
    # srun ask for (ntasks x that) cores ("More processors requested than
    # permitted"), so drop it.
    lines.append("unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK")
    if args.modules:
        lines.append(args.modules)
    lines.append(f"{args.launcher} {args.exe} {prm}")
    return "\n".join(lines) + "\n"


def prepare(args):
    from lib import DEFAULT_EXE, prm_text

    args.exe = args.exe or DEFAULT_EXE
    results = _results_dir(args.tag)
    results.mkdir(parents=True, exist_ok=True)
    submit = ["#!/bin/bash", f"cd {ROOT}"]
    for n in map(int, args.tests.split(",")):
        for p in map(int, args.cores.split(",")):
            for k in range(1, args.repeats + 1):
                name = _run_name(n, p, k, args.repeats)
                sections = apply_overrides(test_sections(n, args.levels_down), args.set)
                sections.setdefault("Run", {})["Results file"] = str(results / f"{name}.json")
                prm = results / f"{name}.prm"
                prm.write_text(prm_text(sections))
                script = results / f"{name}.sh"
                script.write_text(job_script(name, p, prm, args, results))
                submit.append(f"sbatch {script}")
    (results / "submit_all.sh").write_text("\n".join(submit) + "\n")
    print(f"Wrote {len(submit) - 2} jobs to {results}; submit them with: bash {results / 'submit_all.sh'}")


def run_local(args):
    results = _results_dir(args.tag)
    for n in map(int, args.tests.split(",")):
        print(f"Test {n} (levels lowered by {args.levels_down})")
        for p in map(int, args.cores.split(",")):
            for k in range(1, args.repeats + 1):
                run_case(
                    _run_name(n, p, k, args.repeats),
                    apply_overrides(test_sections(n, args.levels_down), args.set),
                    results,
                    np=p,
                    exe=args.exe,
                    timeout=args.timeout,
                    force=args.force,
                    retry_failed=args.retry_failed,
                )


# ---------------------------------------------------------------------------
def load_test(n, tag=None):
    """{cores: [results of every run]} for test n (testN_npP.json, or
    testN_npP_rK.json for repeated runs)."""
    found = {}
    for path in _results_dir(tag).glob(f"test{n}_np*.json"):
        m = re.fullmatch(rf"test{n}_np(\d+)(?:_r\d+)?", path.stem)
        if m:
            found.setdefault(int(m.group(1)), []).append(json.loads(path.read_text()))
    return dict(sorted(found.items()))


def phase_times(result):
    t = result["timings"]
    return {label: sum(t.get(s, {}).get("wall_max", 0.0) for s in sections) for label, sections in PHASES}


def mean_phase_times(runs):
    """Phase times averaged over the runs of one case."""
    per_run = [phase_times(r) for r in runs]
    return {label: float(np.mean([t[label] for t in per_run])) for label, _ in PHASES}


def spread(values):
    """Relative standard deviation of the values (0 for a single one)."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values) / np.mean(values)) if len(values) > 1 and np.mean(values) > 0 else 0.0


def plot_test(n, formats, tag=None, with_total=False):
    data = load_test(n, tag)
    if len(data) < 2:
        return print(f"  test {n}: fewer than two core counts, skipped")
    LOOK = _look()
    cores = np.array(list(data))
    p0 = cores[0]
    times = {label: np.array([mean_phase_times(runs)[label] for runs in data.values()]) for label, _ in PHASES}
    series = [label for label, _ in PHASES]

    n_runs = {p: len(runs) for p, runs in data.items()}
    if max(n_runs.values()) > 1:
        worst = max(
            (
                spread([phase_times(r)[label] for r in runs])
                for runs in data.values()
                for label, _ in PHASES
                if len(runs) > 1 and np.mean([phase_times(r)[label] for r in runs]) > 0.05
            ),
            default=None,
        )
        print(f"  test {n}: runs per core count {sorted(set(n_runs.values()))}; times are means"
              + (f"; largest relative standard deviation of a phase above 0.05 s: {100 * worst:.0f}%" if worst is not None else ""))

    if with_total:
        totals = [[r.get("total_wall") for r in runs] for runs in data.values()]
        if all(t is not None for runs in totals for t in runs):
            times["Whole run"] = np.array([np.mean(runs) for runs in totals])
            series.append("Whole run")
        else:
            print(f"  test {n}: no total_wall in these results (older code), whole-run series skipped")
    name = TESTS[n]["figure"]
    suffix = TESTS[n]["suffix"] + ("_with_total" if "Whole run" in series else "")

    for kind in ("scalability", "total"):
        fig, ax = style.new_axes(style.SCALABILITY_FIGSIZE)
        ideal = cores / p0 if kind == "scalability" else p0 / cores
        ax.loglog(cores, ideal, label="Ideal", **LOOK["Ideal"])
        for label in series:
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
        plot_test(n, tuple(args.formats.split(",")), args.tag, args.with_total)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "run", "plot"))
    parser.add_argument("--tests", default="1,2,3,4,5")
    parser.add_argument("--cores", default=",".join(map(str, CORES)))
    parser.add_argument("--levels-down", type=int, default=0, help="lower every refinement level by this (0 = the paper's)")
    parser.add_argument("--tag", default=None, help="keep results in results/scalability_<tag> instead of results/scalability (e.g. --tag smoke), so a smoke test cannot collide with or overwrite the real sweep")
    parser.add_argument("--exe", help="path of the coupled_diffusion executable")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="Section/Name=value", help="override a parameter of the test files (repeatable), e.g. --set \"Solver/Subdomain solve reduction=1e-8\"; use --tag to keep such a study apart")
    parser.add_argument("--repeats", type=int, default=1, help="runs of every case (prepare/run); the results are testN_npP_rK.* and plot averages the times")
    parser.add_argument("--with-total", action="store_true", help="plot: also draw the wall-clock time of the whole program (setup and mesh included, not only the phases of the paper)")
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
        if style is None:
            sys.exit("plot needs numpy and matplotlib: pip install -r requirements.txt")
        plot_all(args)


if __name__ == "__main__":
    main()
