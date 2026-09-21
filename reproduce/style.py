"""Plot style of the paper's figures: MATLAB-like axes (full box, ticks
pointing inwards, solid major and dotted minor grid), Computer Modern text,
unfilled markers, and the slope triangles of the convergence plots.

Only matplotlib and numpy are needed (no LaTeX installation): text is set in
matplotlib's bundled Computer Modern (cmr10) and mathtext.
"""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

# MATLAB's default line colours, as used in the paper's figures.
BLUE = "#0072BD"
ORANGE = "#D95319"
YELLOW = "#EDB120"
PURPLE = "#7E2F8E"
GREEN = "#77AC30"
GRAY = "#8c8c8c"

#: Size of the accuracy figures (inches): the EPS bounding boxes of the paper
#: are 334 x ~250 pt.
ACCURACY_FIGSIZE = (334 / 72.0, 250 / 72.0)
#: Size of the scalability figures: the PNGs of the paper are 700 x 525 px.
SCALABILITY_FIGSIZE = (7.0, 5.25)


def setup():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "lines.markeredgewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "legend.frameon": False,
            "legend.handlelength": 2.6,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_axes(figsize=ACCURACY_FIGSIZE):
    setup()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def style_series(**kwargs):
    """Line style of a data series: unfilled marker of the line's colour."""
    color = kwargs.pop("color")
    return dict(color=color, markerfacecolor="white", markeredgecolor=color, **kwargs)


def grid(ax, log_x=False, log_y=False):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#c8c8c8", linestyle="-", linewidth=1.2)
    ax.grid(True, which="minor", color="#c8c8c8", linestyle=":", linewidth=0.9)
    for axis, is_log in ((ax.xaxis, log_x), (ax.yaxis, log_y)):
        if is_log:
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
            axis.set_minor_formatter(NullFormatter())
            axis.set_tick_params(which="minor", length=0)
        else:
            axis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            axis.set_tick_params(which="minor", length=0)


def decade_limits(values, pad=0.25):
    """Axis limits: the data range extended by `pad` decades on both sides,
    then rounded outward to the next 1-2-5 step so the frame looks tidy."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)


def slope_triangle(ax, x0, y0, slope, factor=2.0, label=None, label_side="left"):
    """Right triangle indicating a convergence rate, drawn as in the paper: the
    hypotenuse goes from (x0, y0) up to the right with the given slope on
    the log-log plot, the right angle sits at the top left.

    (x0, y0): lower-left corner in data coordinates; `factor`: horizontal
    extent as a multiplicative factor.
    """
    x1 = x0 * factor
    y1 = y0 * factor**slope
    ax.plot([x0, x0, x1, x0], [y0, y1, y1, y0], color="black", linewidth=1.4, solid_joinstyle="miter")
    label = str(slope) if label is None else label
    ax.text(
        x0 / 1.12 if label_side == "left" else x1 * 1.05,
        np.sqrt(y0 * y1),
        label,
        ha="right" if label_side == "left" else "left",
        va="center",
        fontsize=10,
    )


def set_log_ticks(ax, axis, powers):
    """Major ticks at the given powers of ten, labelled 10^p."""
    ticks = [10.0**p for p in powers]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))


def save(fig, name, outdir, formats=("pdf", "png")):
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  wrote {outdir / name}.{{{','.join(formats)}}}")
