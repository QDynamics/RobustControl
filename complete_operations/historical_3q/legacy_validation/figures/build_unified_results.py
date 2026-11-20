from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


parser = argparse.ArgumentParser(description="Rebuild the vector summary in a fresh output directory.")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
HERE = args.output_dir.resolve()
HERE.mkdir(parents=True, exist_ok=False)

FONT_PATHS = [Path(r"C:\Windows\Fonts") / name for name in ("times.ttf", "timesi.ttf", "timesbd.ttf")]
for path in FONT_PATHS:
    if not path.is_file():
        raise FileNotFoundError(path)
    font_manager.fontManager.addfont(str(path))
FONT_FAMILY = font_manager.FontProperties(fname=str(FONT_PATHS[0])).get_name()

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "mathtext.sf": FONT_FAMILY,
        "mathtext.fallback": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
# Build these after configuring mathtext; otherwise they cache DejaVu math.
FONTS = {
    "regular": font_manager.FontProperties(family=FONT_FAMILY, math_fontfamily="custom"),
    "bold": font_manager.FontProperties(family=FONT_FAMILY, weight="bold", math_fontfamily="custom"),
    "italic": font_manager.FontProperties(family=FONT_FAMILY, style="italic", math_fontfamily="custom"),
}

INK = "#24313D"
MUTED = "#65717B"
RULE = "#D6DDE2"
BLUE = "#2D6FB7"
BLUE_LIGHT = "#EDF3FA"
TEAL = "#258A86"
TEAL_LIGHT = "#EDF7F5"
ORANGE = "#CB702F"
ORANGE_LIGHT = "#FFF3E9"

# Coordinates are millimetres, with an equal aspect ratio: neither lettering
# nor circular nodes need any compensating stretch.
WIDTH, HEIGHT = 180.0, 88.0
fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
ax.axis("off")
TEXT_ITEMS = []


def label(x, y, text, *, size=8.5, color=INK, weight="regular", ha="center"):
    artist = ax.text(
        x, y, text, fontproperties=FONTS[weight], fontsize=size,
        color=color, ha=ha, va="center", linespacing=1.15, zorder=6,
    )
    TEXT_ITEMS.append(artist)
    return artist


def box(x, y, w, h, *, face="white", edge=RULE, lw=0.7):
    # No hidden padding: the specified rectangle is its true visible extent.
    artist = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=1,
    )
    ax.add_patch(artist)


def arrow(start, end, *, color=MUTED, lw=0.9, head=8):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=head,
        color=color, linewidth=lw, shrinkA=1.2, shrinkB=1.2, zorder=3,
    ))


def header(x, text):
    label(x, 83.3, text, size=9.2, weight="bold", ha="left")


def network(points, edges, *, special=None):
    for i, j in edges:
        a, b = points[i], points[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.0, zorder=2)
    if special is not None:
        a, b = (points[index] for index in special)
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-", connectionstyle="arc3,rad=-0.75",
            color=ORANGE, linewidth=1.6, shrinkA=0, shrinkB=0, zorder=3,
        ))
    for xy in points:
        ax.add_patch(Circle(xy, radius=1.25, facecolor="white", edgecolor=BLUE, lw=1.1, zorder=4))


# Open panels replace the nested outer frames of the earlier figure.
for x in (60.5, 121.5):
    ax.plot([x, x], [4, 85], color=RULE, lw=0.6, zorder=0)
for left, right in ((2, 58), (64, 118), (125, 179)):
    ax.plot([left, right], [78.8, 78.8], color=RULE, lw=0.6, zorder=0)

# (a) Classification: split the nominal algebra, then read off robust targets.
header(2, "(a)  Unified classification")
label(30, 72, "Nominal algebra", color=MUTED)
label(30, 64.3, r"$g=z\oplus s_1\oplus s_2\oplus\cdots$", size=10.1)
arrow((26, 59.5), (16, 54.5), color=ORANGE)
arrow((34, 59.5), (45, 54.5), color=BLUE)

box(2, 37, 26, 17, face=ORANGE_LIGHT, edge=ORANGE)
box(32, 37, 26, 17, face=BLUE_LIGHT, edge=BLUE)
label(15, 49.2, "Center", color=ORANGE, weight="bold")
label(15, 42.1, r"$C_{\rm err}t=0$", size=9.4)
label(45, 49.2, "Simple factors", color=BLUE, weight="bold")
label(45, 42.1, r"$s_j\otimes A_q(W_j)$", size=9.1)

arrow((15, 36.7), (24, 31.4), color=ORANGE)
arrow((45, 36.7), (36, 31.4), color=BLUE)
box(5, 13, 50, 18, face=TEAL_LIGHT, edge=TEAL, lw=0.9)
label(30, 26.3, "Robust targets", color=TEAL, weight="bold")
label(30, 19.3, r"$K_{\rm rob}=\exp(E)\,S_{\rm der}$", size=10.1)
label(30, 5.9, "Every fixed finite order $q$", size=8.4, color=MUTED)

# (b) Distinct physical symmetries, the same target-classification procedure.
header(64, "(b)  Two control families")
network([(69, 70), (80, 70), (69, 60), (80, 60)],
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
label(86, 68.6, "Collective spins", weight="bold", ha="left")
label(86, 62.7, "Rotations + twisting", size=8.1, color=MUTED, ha="left")
label(91, 49.8, r"$\prod_{j>0}\,\mathrm{SU}(2j+1)$", size=10.2, color=BLUE)
ax.plot([64, 118], [40.7, 40.7], color=RULE, lw=0.6, zorder=0)

network([(66.5, 30), (72, 30), (77.5, 30), (83, 30)],
        [(0, 1), (1, 2), (2, 3)], special=(0, 1))
label(88, 32.6, "Excitation chain", weight="bold", ha="left")
label(88, 26.7, "Z, exchange + ZZ", size=8.1, color=MUTED, ha="left")
label(91, 12.9, r"$\prod_{k=1}^{N-1}\,\mathrm{SU}(d_k),\quad d_k=\binom{N}{k}$",
      size=9.5, color=BLUE)

# (c) The target stays fixed; one added ideal axis changes feasibility.
header(125, "(c)  Resource boundary")
label(152, 72, "Fixed two-qubit target", color=MUTED)
label(152, 64.3, r"$U_\star=e^{-i\pi ZZ/4}$", size=10.1)

box(125, 42, 54, 17, face=ORANGE_LIGHT, edge=ORANGE)
label(152, 55, "Positive ZZ; local Z + exchange", size=8.3)
label(137.5, 46.7, r"$\kappa\geq\pi/4$", size=10.6, color=ORANGE)
label(162.7, 46.7, "Empty robust fibre", size=8.4, color=ORANGE)

arrow((135, 40.5), (135, 29.5), color=TEAL, lw=1.1, head=9)
label(142, 35, r"Add ideal $Y_2$", color=TEAL, ha="left")

box(125, 8, 54, 20.5, face=TEAL_LIGHT, edge=TEAL)
label(152, 23.7, "First-order robust (SK1)", size=8.5, color=TEAL, weight="bold")
label(137.5, 14.9, r"$K=0$", size=10.6, color=TEAL)
label(163, 14.9, "$9\\times$ native\nZZ action", size=8.3)

# Measure actual typeset extents as a layout diagnostic, in addition to visual
# review of the exported PDF. This is not a scientific/numerical validation.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
extents = [item.get_window_extent(renderer) for item in TEXT_ITEMS]
overlaps = []
for i, first in enumerate(extents):
    for j in range(i + 1, len(extents)):
        second = extents[j]
        if first.overlaps(second):
            overlaps.append([TEXT_ITEMS[i].get_text(), TEXT_ITEMS[j].get_text()])
outside = [item.get_text() for item, bounds in zip(TEXT_ITEMS, extents)
           if not (fig.bbox.contains(bounds.x0, bounds.y0) and fig.bbox.contains(bounds.x1, bounds.y1))]
layout = {
    "canvas_mm": [WIDTH, HEIGHT],
    "font_family": FONT_FAMILY,
    "text_artist_count": len(TEXT_ITEMS),
    "text_overlap_pairs": overlaps,
    "text_outside_canvas": outside,
}
(HERE / "layout_diagnostics.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")

metadata = {
    "Title": "Unified robust-control classification and resource boundary",
    "Author": "Composable Robust Quantum Control project",
    "Subject": "Vector summary: classification, control families, and resource boundary",
    "Keywords": "quantum control, response algebra, robust targets, resource boundary",
}
fig.savefig(HERE / "unified_results.pdf", metadata=metadata)
fig.savefig(HERE / "unified_results.svg", metadata={"Title": metadata["Title"]})
fig.savefig(HERE / "unified_results_preview.png", dpi=300)
plt.close(fig)
print(json.dumps({"output_dir": str(HERE), "layout": layout}, indent=2))
