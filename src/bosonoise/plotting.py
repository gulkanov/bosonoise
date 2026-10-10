"""Shared figure style (validated categorical palette, recessive axes/grid).

Categorical slots 1-3 (blue, orange, aqua) pass all-pairs CVD / normal-vision checks on
a light surface (validated with the dataviz validator, 2026-09-28); the aqua slot is
below 3:1 contrast, so every figure carries a legend. Series are assigned in fixed
order and never cycled; sequential encodings use the single-hue blue ramp.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import ListedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# ordinal blue ramp, darkest first (step 700 -> 250); lightest step still >= 2:1 on light
BLUE_ORDINAL = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#86b6ef"]


def ordinal_cmap(n: int) -> ListedColormap:
    return ListedColormap(BLUE_ORDINAL[:n], name=f"blue_ordinal_{n}")


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelcolor": TEXT_PRIMARY,
        "ytick.labelcolor": TEXT_PRIMARY,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })
