```python
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from shapely import geometry
import networkx as nx
from snippets.plot import arrow_path, get_anchor, rounded_path, label_axes


mpl.style.use("../.mplstyle")
figwidth, figheight = mpl.rcParams["figure.figsize"]
```

```python
fig, ax = plt.subplots()

vspace = 1
hspace = 2.1
kwargs = {
    "ha": "center",
    "va": "center", 
    "fontsize": "small",
}
fkwargs = kwargs | {
    "bbox": {
        "boxstyle": "round,pad=0.5",
        "color": "C0",
        "alpha": 0.5,
    }
}
vkwargs = kwargs | {
    "bbox": {
        "boxstyle": "round,pad=0.5",
        "color": "silver",
        "alpha": 0.5,
    }
}
ckwargs = kwargs | {
    "bbox": {
        "boxstyle": "round,pad=0.5",
        "color": "C1",
        "alpha": 0.5,
    }
}
dkwargs = kwargs | {
    "bbox": {
        "boxstyle": "round,pad=0.5",
        "color": "C2",
        "alpha": 0.5,
    }
}

texts = {
    "prior": (0, 0, "prior $\\pi(\\theta)$", fkwargs),
    "params": (0, -vspace, "parameters\n$\\theta\\in\\mathbb{R}^p\\sim\\pi$", vkwargs),
    "simulator": (0, -2 * vspace, "simulator\n$g\\left(z\\mid\\theta\\right)$", fkwargs),
    "simulated_data": (0, -3 * vspace, "simulated data\n$z\\in\\mathbb{D}\\sim g\\left(z\\mid\\theta\\right)$", dkwargs),
    "observed_data": (2 * hspace, -3 * vspace, "observed data\n$y\\in\\mathbb{D}$", dkwargs),
    "compressor": (hspace, -4 * vspace, "compressor\n$t(\\cdot): \\mathbb{D}\\rightarrow\\mathbb{R}^q$", ckwargs),
    "simulated_summaries": (0, -5 * vspace, "simulated summaries\n$t(z)\\in\\mathbb{R}^q$", vkwargs),
    "observed_summaries": (2 * hspace, -5 * vspace, "observed summaries\n$t(y)\\in\\mathbb{R}^q$", vkwargs),
    "abc": (2 * hspace, -6 * vspace, "approximate Bayesian\ncomputation", fkwargs),
    "mdn-compressed-samples": (2 * hspace, -7 * vspace, "MDN-compressed samples\n$\\tilde\\theta\\in\\mathbb{R}^p\\sim \\tilde f\\left(\\theta\\mid t(y)\\right)$", vkwargs),
    "estimator": (0, -6 * vspace, "mixture density\nnetwork $h:\\mathbb{R}^q\\rightarrow\\mathcal{F}$", fkwargs),
    "estimate": (0, -7 * vspace, "density estimate\n$\\hat f\\left(\\theta\\mid t(z)\\right)\\in \\mathcal{F}$", vkwargs),
    "loss": (- hspace / 2, -8 * vspace, "NLP loss", fkwargs), 
}

elements = {}
for key, (x, y, text, kwargs) in texts.items():
    elements[key] = ax.text(x, y, text, **kwargs, in_layout=True)

ax.set_axis_off()
if True:
    # Set the limits large initially to ensure all text is within the bounds.
    xy = np.asarray([(x, y) for x, y, *_ in texts.values()])
    xmin, ymin = xy.min(axis=0) - 1
    xmax, ymax = xy.max(axis=0) + 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    
    # Then adjust based on the actual extent of the box containing the text.
    fig.tight_layout()
    fig.draw_without_rendering()
    transform = ax.transData.inverted()
    extents = np.asarray([transform.transform(element.get_bbox_patch().get_window_extent()) 
                          for element in elements.values()])
    xmin, ymin = extents.min(axis=0)[0]
    xmax, ymax = extents.max(axis=0)[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
else:
    ax.set_xlim(-2.5, 7)
    ax.set_ylim(-8.5, 0.5)

ax.set_xlim(right=8)
fig.draw_without_rendering()

# Plot connections between the different boxes.
connections = [
    # Simulated.
    [("prior", 6), ("params", 12)],
    [("params", 6), ("simulator", 12)],
    [("simulator", 6), ("simulated_data", 12)],
    [
        ("simulated_data", 3), 
        (get_anchor(elements["compressor"], 11).x, get_anchor(elements["simulated_data"], 3).y),
        ("compressor", 11),
    ],
    [
        ("compressor", 7), 
        (get_anchor(elements["compressor"], 7).x, get_anchor(elements["simulated_summaries"], 2.75).y),
        ("simulated_summaries", 2.75),
    ],
    [
        ("simulated_summaries", 3.25),
        (get_anchor(elements["compressor"], 6).x, get_anchor(elements["simulated_summaries"], 3.25).y),
        (get_anchor(elements["compressor"], 6).x, get_anchor(elements["abc"], 9).y),
        ("abc", 9),
        
    ],
    [("simulated_summaries", 6), ("estimator", 12)],
    [("estimator", 6), ("estimate", 12)],
    [
        ("params", 9),
        (get_anchor(elements["loss"], 10).x, get_anchor(elements["params"], 9).y),
        ("loss", 10),
    ],
    [
        ("estimate", 6),
        (get_anchor(elements["estimate"], 6).x, get_anchor(elements["loss"], 3).y),
        ("loss", 3),
    ],

    # Observed.
    [
        ("observed_data", 9), 
        (get_anchor(elements["compressor"], 1).x, get_anchor(elements["observed_data"], 9).y),
        ("compressor", 1),
    ],
    [
        ("compressor", 5), 
        (get_anchor(elements["compressor"], 5).x, get_anchor(elements["observed_summaries"], 9).y),
        ("observed_summaries", 9),
    ],
    [("observed_summaries", 6), ("abc", 12)],
    [
        ("params", 3),
        (get_anchor(elements["abc"], 3).x + 0.5, get_anchor(elements["params"], 3).y),
        (get_anchor(elements["abc"], 3).x + 0.5, get_anchor(elements["abc"], 3).y),
        ("abc", 3),
    ],
    [("abc", 6), ("mdn-compressed-samples", 12)],
]

for connection in connections:
    connection = [get_anchor(elements[x], y) if isinstance(x, str) else (x, y) for x, y in connection]
    path = rounded_path(connection, 0.1, shrink=0.05)
    patch = mpl.patches.PathPatch(path, fc="none", ec="gray", zorder=-2)
    ax.add_patch(patch)
    arrow_path_ = arrow_path(path, 0.06)
    arrow_patch = mpl.patches.PathPatch(arrow_path_, color="gray")
    ax.add_patch(arrow_patch)

# Add a table.
mpl.table.Cell.PAD = 0.1
table = ax.table([
    ["benchmark", "samples $\\mathbb{R}^{n\\times 2}$", "MLP(2,16,16,1),\nmean-pool"],
    ["population\ngenetics", "expert\nsummaries $\\mathbb{R}^{7}$", "MLP(7,16,16,2)"],
    ["growing\ntrees", "trees $\\mathbb{T}$", "GIN(1,8,8), GIN(8,8,1),\nmean-pool"],
], loc="upper right", cellLoc="left", colLabels=["experiment", "data $\\mathbb{D}$", "compressor $t(\\cdot)$"])
table.auto_set_font_size(False)
table.set_fontsize("small")
table.auto_set_column_width([0, 1, 2])
fig.draw_without_rendering()

# Add edges like booktabs in latex would.
celld = table.properties()["celld"]
colors = {(0, 1): mpl.colors.to_rgba("C2", 0.5), (0, 2): mpl.colors.to_rgba("C1", 0.5)}
for (row, col), cell in celld.items():
    if row == 0:
        cell.visible_edges = "T"
    elif row == 1:
        cell.visible_edges = "T"
        # Same relative size as booktabs (https://tex.stackexchange.com/a/156124/12335).
        cell.set_linewidth(5 / 8)
    elif row == 3:
        cell.visible_edges = "B"
    else:
        cell.visible_edges = ""
    original_height = cell.get_height()
    height = 1.4 * original_height
    cell.set_height(height)

    color = colors.get((row, col), mpl.colors.to_rgba("w", 0.65))
    x = cell.get_x()
    y = cell.get_y() - (row + 1) * (height - original_height)
    phantom = mpl.patches.Rectangle(
        (x, y), cell.get_width(), cell.get_height(), transform=ax.transAxes, 
        facecolor=color, zorder=-1)
    ax.add_patch(phantom)

fig.savefig("../workspace/figures/experiments.pdf")
```
