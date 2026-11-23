"""The chart style shared by every figure in reports/figures/ (PNG, 200 dpi).

The colours are slots of a palette checked for colour-blind separation. Each
scorer keeps its colour in every chart (`src.evaluate.SCORER_COLORS`).
"""

from __future__ import annotations

from pathlib import Path

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style_axis(ax, grid: str = "y") -> None:
    """Recessive axes: no box, a light grid behind the data, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    return fig
