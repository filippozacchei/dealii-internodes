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

sys.path.insert(0, str(Path(__file__).resolve().parent))

# numpy and matplotlib are only needed by `plot`; `prepare` and `run` use the
# standard library only, so they work on a bare cluster Python.
try:
    import numpy as np
    import style
except ImportError:
    np = style = None

from lib import ROOT, add_cluster_args, job_script, read_prm, run_case, write_submit_all  # noqa: E402

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
    6: {"figure": "RBF_Shell_p2r7-p2r6_r=10_galileo", "suffix": ""},  # Test 5 refined once more; not in the paper's Table 1
}

# (label, timer sections of the results file that are added up)
PHASES = [
    (
        "Assembly Interpolation Operators",
        (
            "Setup destination points",
            # Older keys (AMG, then a one-time direct factorization that
            # turned out to blow up in fill-in at large radius -- see git
            # history), kept so already-cached results still plot.
            "setup: RBF operators (Phi, AMG, scaling)",
            "setup: RBF operators (Phi, direct factor, scaling)",
            "setup: RBF operators (Phi, ILUT, scaling)",
        ),
    ),
    ("Assembly Internal Operators", ("Step 0: assembly (master/slave)",)),
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


def prepare(args):
    from lib import DEFAULT_EXE, prm_text

    args.exe = args.exe or DEFAULT_EXE
    results = _results_dir(args.tag)
    results.mkdir(parents=True, exist_ok=True)
    names = []
    repeats = [None] if args.repeats <= 1 and args.repeat_start <= 1 else \
        list(range(args.repeat_start, args.repeat_start + args.repeats))
    for n in map(int, args.tests.split(",")):
        for p in map(int, args.cores.split(",")):
          for k in repeats:
            name = f"test{n}_np{p}" + (f"_r{k}" if k is not None else "")
            sections = test_sections(n, args.levels_down)
            sections.setdefault("Run", {})["Results file"] = str(results / f"{name}.json")
            prm = results / f"{name}.prm"
            prm.write_text(prm_text(sections))
            script = results / f"{name}.sh"
            script.write_text(job_script(name, p, prm, args, results))
            names.append(name)
    submit = write_submit_all(results, names)
    print(f"Wrote {len(names)} jobs to {results}; submit them with: bash {submit}")


def run_local(args):
    results = _results_dir(args.tag)
    for n in map(int, args.tests.split(",")):
        print(f"Test {n} (levels lowered by {args.levels_down})")
        for p in map(int, args.cores.split(",")):
            run_case(
                f"test{n}_np{p}",
                test_sections(n, args.levels_down),
                results,
                np=p,
                exe=args.exe,
                timeout=args.timeout,
                force=args.force,
                retry_failed=args.retry_failed,
            )


# ---------------------------------------------------------------------------
def load_test(n, tag=None):
    """{cores: [results of every repeat]} for test n. A plain testN_npP.json
    counts as one repeat, testN_npP_rK.json as the others. Refuses to mix
    runs built from different commits (results files record it)."""
    found = {}
    for path in _results_dir(tag).glob(f"test{n}_np*.json"):
        p = int(path.stem.split("_np")[1].split("_r")[0])
        found.setdefault(p, []).append(json.loads(path.read_text()))
    versions = {r.get("git_version", "unknown") for runs in found.values() for r in runs}
    if len(versions) > 1:
        sys.exit(f"test {n} ({tag}): results from different commits {sorted(versions)}; not mixing them")
    return dict(sorted(found.items()))


def median_min_max(values):
    v = np.sort(np.asarray(values, dtype=float))
    return np.median(v), v[0], v[-1]


def phase_times(result):
    t = result["timings"]
    return {label: sum(t.get(s, {}).get("wall_max", 0.0) for s in sections) for label, sections in PHASES}


def plot_test(n, formats, tag=None):
    data = load_test(n, tag)
    if len(data) < 2:
        return print(f"  test {n}: fewer than two core counts, skipped")
    LOOK = _look()
    cores = np.array(list(data))
    p0 = cores[0]
    n_repeats = {p: len(runs) for p, runs in data.items()}
    # per phase: median / min / max over the repeats at each core count
    stats = {
        label: np.array([median_min_max([phase_times(r)[label] for r in runs]) for runs in data.values()])
        for label, _ in PHASES
    }
    times = {label: stats[label][:, 0] for label, _ in PHASES}
    if max(n_repeats.values()) > 1:
        print(f"  test {n}: repeats per core count {sorted(set(n_repeats.values()))}; largest spreads (max-min)/median:")
        spreads = sorted(
            ((st[i, 2] - st[i, 1]) / st[i, 0], label, int(cores[i]))
            for label, st in stats.items() for i in range(len(cores)) if st[i, 0] > 0.05
        )[::-1][:3]
        for sp, label, p in spreads:
            print(f"     {sp * 100:5.1f}%  {label} @ {p} cores")
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
            st = stats[label][valid]
            if kind == "scalability":
                # median T(p0) over the repeats at p, with the range of the repeats at p
                y, lo, hi = t[valid][0] / st[:, 0], t[valid][0] / st[:, 2], t[valid][0] / st[:, 1]
            else:
                y, lo, hi = st[:, 0], st[:, 1], st[:, 2]
            look = dict(LOOK[label])
            ax.loglog(cores[valid], y, label=label, **style.style_series(**look))
            if (hi > lo).any():
                ax.errorbar(cores[valid], y, yerr=[y - lo, hi - y], fmt="none",
                            ecolor=look.get("color", "black"), elinewidth=0.8, capsize=2)
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
        plot_test(n, tuple(args.formats.split(",")), args.tag)


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
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per local run")
    parser.add_argument("--repeats", type=int, default=1, help="prepare: number of repeated runs per test/core count (named testN_npP_rK)")
    parser.add_argument("--repeat-start", type=int, default=1, help="prepare: first repeat index K (e.g. 2 to add repeats next to an existing testN_npP run)")
    add_cluster_args(parser)
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
