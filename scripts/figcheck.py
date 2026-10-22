"""Deterministic layout checks for matplotlib figures (tests in tests/test_figcheck.py).

Run on the rendered figure, so positions are what a reader sees:

  * no two visible text labels overlap or nearly touch — at least ``min_gap_pt`` points
    apart (titles, axis labels, tick labels in view, annotations, legend entries);
  * no legend covers a plotted data point or line vertex.

    from figcheck import assert_layout
    assert_layout(fig)          # before fig.savefig(...)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox


def _ticks_in_view(fig: Figure) -> set[int]:
    """ids of tick labels whose tick lies inside the axis view (others are not drawn)."""
    keep = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if lo <= tick.get_loc() <= hi:
                    keep.update({id(tick.label1), id(tick.label2)})
    return keep


def _texts(fig: Figure) -> list[Text]:
    tick_labels = {id(label) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                   for t in axis.get_major_ticks() + axis.get_minor_ticks() for label in (t.label1, t.label2)}
    in_view = _ticks_in_view(fig)
    out = []
    for t in fig.findobj(Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        if id(t) in tick_labels and id(t) not in in_view:
            continue
        out.append(t)
    return out


def _data_points(fig: Figure) -> np.ndarray:
    """Display coordinates of every line vertex and scatter point in the axes."""
    pts = []
    for ax in fig.axes:
        for line in ax.lines:
            xy = line.get_xydata()
            if len(xy):
                pts.append(line.get_transform().transform(xy))
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) and coll.get_offset_transform() is not None:
                pts.append(coll.get_offset_transform().transform(offsets))
    return np.vstack(pts) if pts else np.empty((0, 2))


def layout_problems(fig: Figure, min_gap_pt: float = 3.0) -> list[str]:
    """Every layout problem found, as readable messages (empty list = clean)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    texts = _texts(fig)
    half_gap_px = min_gap_pt * fig.dpi / 72 / 2  # in points, so it does not depend on the dpi
    boxes = [t.get_window_extent(renderer).padded(half_gap_px) for t in texts]
    for (ta, ba), (tb, bb) in combinations(zip(texts, boxes), 2):
        if ba.width > 0 and bb.width > 0 and ba.overlaps(bb):
            problems.append(f"text overlap or crowding: {ta.get_text()!r} and {tb.get_text()!r}")
    points = _data_points(fig)
    for legend in fig.findobj(Legend):
        box: Bbox = legend.get_window_extent(renderer)
        covered = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
                   & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1))
        if covered.any():
            problems.append(f"legend covers {int(covered.sum())} data point(s)")
    return problems


def spread_labels(ax, labels: list[tuple[float, float, str]], min_sep_pt: float | None = None,
                  dx_pt: float = 6.0, **text_kw) -> None:
    """Direct labels at the end of lines, (x, y, text) in data units, nudged vertically so
    neighbours are at least ``min_sep_pt`` apart (default: the line height of the font
    plus the 3 pt gap `layout_problems` requires). Call after the layout is final
    (after ``tight_layout``): positions are fixed in points from the rendered axes."""
    if not labels:
        return
    if min_sep_pt is None:
        size = FontProperties(size=text_kw.get("fontsize", text_kw.get("size"))).get_size_in_points()
        min_sep_pt = 1.25 * size + 3.5
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    ys = np.array([ax.transData.transform((x, y))[1] * to_pt for x, y, _ in labels])
    order = np.argsort(ys)
    placed = ys[order].copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_sep_pt)
    placed -= (placed - ys[order]).mean()  # keep the group centred on the lines
    for k, i in enumerate(order):
        x, y, text = labels[i]
        ax.annotate(text, (x, y), xytext=(dx_pt, placed[k] - ys[i]), textcoords="offset points",
                    va="center", **text_kw)


def assert_layout(fig: Figure) -> None:
    problems = layout_problems(fig)
    if problems:
        raise AssertionError("figure layout:\n  " + "\n  ".join(problems))
