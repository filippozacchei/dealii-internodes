"""Shared helpers of the reproduction scripts: run one configuration of the
coupled_diffusion example and cache its JSON results.

Nothing here is specific to a figure. A *case* is a parameter file (given as
nested dicts, one per subsection) plus a number of MPI ranks; running it
writes <outdir>/<name>.prm, <name>.log and <name>.json (the example's
"Run/Results file"). Cases whose JSON already exists are not rerun unless
force=True, so the scripts can be interrupted and restarted.
"""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: Default location of the example executable (after `cmake --build build`).
DEFAULT_EXE = ROOT.parent / "build" / "examples" / "coupled_diffusion" / "coupled_diffusion"

#: Command that launches an MPI program; {np} is replaced by the rank count.
#: Override with the INTERNODES_LAUNCHER environment variable, e.g.
#: "srun -n {np}" inside a SLURM allocation.
DEFAULT_LAUNCHER = "mpirun -np {np}"


def executable(exe=None):
    path = Path(exe or os.environ.get("INTERNODES_EXE", DEFAULT_EXE))
    if not path.exists():
        raise SystemExit(
            f"Executable not found: {path}\n"
            "Build the project first (see the top-level README) or pass --exe."
        )
    return path


def prm_text(sections):
    """{'Geometry': {'Type': 'adjacent_boxes'}, ...} -> deal.II .prm text."""
    lines = []
    for section, entries in sections.items():
        lines.append(f"subsection {section}")
        lines += [f"  set {key} = {value}" for key, value in entries.items()]
        lines += ["end", ""]
    return "\n".join(lines)


def read_prm(path):
    """Parse a (flat, one level of subsections) .prm file into
    {'Section': {'Key': 'value'}} so that a case can be modified in Python."""
    sections, current = {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("subsection"):
            current = line[len("subsection"):].strip()
            sections[current] = {}
        elif line == "end":
            current = None
        elif line.startswith("set") and current is not None:
            key, value = line[3:].split("=", 1)
            sections[current][key.strip()] = value.strip()
    return sections


def box_case(
    master_refinement,
    slave_refinement=None,
    master_degree=1,
    slave_degree=None,
    rbf_factor=0.0,
    master_cells="hex",
    slave_cells="hex",
    preconditioner=True,
    monolithic=False,
    gmres_reduction=1e-8,
):
    """Parameter sections of the paper's Geometry-A test: two cubes of side 2,
    (-2,0)x(-1,1)^2 and (0,2)x(-1,1)^2, each a single coarse cell refined
    globally, so that the mesh size is h = 2 / 2^refinement.

    rbf_factor = 0 selects Lagrange interpolation, r_f > 0 the RL-RBF
    interpolation with r = r_f * h_avg. gmres_reduction is the relative
    residual reduction at which the interface GMRES stops (the paper's
    1e-8); the absolute tolerance is set negligibly small.
    """
    slave_refinement = master_refinement if slave_refinement is None else slave_refinement
    slave_degree = master_degree if slave_degree is None else slave_degree
    return {
        "Geometry": {
            "Type": "adjacent_boxes",
            "Subdivisions master": "1,1,1",
            "Subdivisions slave": "1,1,1",
            "Global refinement master": master_refinement,
            "Global refinement slave": slave_refinement,
            "Cell type master": master_cells,
            "Cell type slave": slave_cells,
        },
        "Discretization": {
            "Master degree": master_degree,
            "Slave degree": slave_degree,
            "RBF radius": 0.0,
            "RBF radius factor": rbf_factor,
        },
        "Solver": {
            "GMRES tolerance": 1e-14,
            "GMRES reduction": gmres_reduction,
            "GMRES max iterations": 2000,
            "Use Schur preconditioner": "true" if preconditioner else "false",
        },
        "Run": {"Mode": "monolithic" if monolithic else "coupled"},
    }


def run_case(
    name,
    sections,
    outdir,
    np=1,
    exe=None,
    timeout=None,
    force=False,
    retry_failed=False,
    verbose=True,
):
    """Run one configuration; return its parsed JSON results, or None if it
    failed or exceeded `timeout` seconds (recorded in <name>.status)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_file = outdir / f"{name}.json"
    status_file = outdir / f"{name}.status"

    if result_file.exists() and not force:
        return json.loads(result_file.read_text())
    if status_file.exists() and not (force or retry_failed):
        if verbose:
            print(f"  skip  {name}: {status_file.read_text().strip()}")
        return None

    sections = {k: dict(v) for k, v in sections.items()}
    sections.setdefault("Run", {})["Results file"] = str(result_file)
    prm_file = outdir / f"{name}.prm"
    prm_file.write_text(prm_text(sections))

    launcher = os.environ.get("INTERNODES_LAUNCHER", DEFAULT_LAUNCHER).format(np=np)
    command = shlex.split(launcher) + [str(executable(exe)), str(prm_file)]

    start = time.time()
    if result_file.exists():
        result_file.unlink()
    status_file.unlink(missing_ok=True)
    try:
        with open(outdir / f"{name}.log", "w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True
            )
    except subprocess.TimeoutExpired:
        status_file.write_text(f"timeout after {timeout} s\n")
        if verbose:
            print(f"  TIMEOUT {name} (>{timeout} s)")
        return None
    except subprocess.CalledProcessError as exc:
        status_file.write_text(f"failed with exit code {exc.returncode}, see {name}.log\n")
        if verbose:
            print(f"  FAILED  {name} (exit code {exc.returncode}), see {name}.log")
        return None

    if verbose:
        print(f"  done  {name}  ({time.time() - start:.1f} s, np={np})")
    return json.loads(result_file.read_text())


def timing(results, section, default=0.0):
    """Wall time (max over ranks) of a timed section of a results dict."""
    return results["timings"].get(section, {}).get("wall_max", default)
