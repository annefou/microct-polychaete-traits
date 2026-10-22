"""The layout check must flag what a reader would see as broken, and pass a clean figure."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tests only: no display needed (never in notebooks)
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from figcheck import assert_layout, layout_problems  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_clean_figure_passes():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [1, 2, 3], label="a")
    ax.set_title("A title")
    ax.set_xlabel("x")
    ax.legend(loc="upper left")
    assert layout_problems(fig) == []


def test_overlapping_annotations_are_flagged():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1])
    ax.annotate("first label", (0.5, 0.5))
    ax.annotate("second label", (0.5, 0.5))
    assert any("text overlap" in p for p in layout_problems(fig))


def test_colliding_tick_labels_are_flagged():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xscale("log")
    ax.set_xlim(1.7, 300)
    ax.set_xticks([20, 22, 25], labels=["20", "22", "25"])
    assert any("text overlap" in p for p in layout_problems(fig))


def test_ticks_outside_the_view_are_ignored():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([0, 0.5, 1, 50, 50.01], labels=["0", "0.5", "1", "x", "y"])
    ax.set_xlim(0, 1)
    assert layout_problems(fig) == []


def test_legend_over_data_is_flagged():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([0.05, 0.5], [0.95, 0.5], label="points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    assert any("legend covers" in p for p in layout_problems(fig))


def test_assert_layout_raises_with_messages():
    fig, ax = plt.subplots()
    ax.annotate("same", (0.5, 0.5))
    ax.annotate("same", (0.5, 0.5))
    with pytest.raises(AssertionError, match="text overlap"):
        assert_layout(fig)


def test_spread_labels_separates_close_line_labels():
    from figcheck import spread_labels

    fig, ax = plt.subplots(figsize=(6, 4))
    for y in (1.0, 1.02, 1.04):
        ax.plot([0, 1], [0, y])
    ax.set_xlim(0, 2)
    fig.tight_layout()
    spread_labels(ax, [(1, 1.0, "first line"), (1, 1.02, "second line"), (1, 1.04, "third line")])
    assert layout_problems(fig) == []
