```python
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from shapely import geometry
import networkx as nx
from snippets.plot import rounded_path, label_axes, get_anchor, arrow_path


mpl.style.use("../.mplstyle")
figwidth, figheight = mpl.rcParams["figure.figsize"]
```

# Illustration of different methods and classes of summaries

```python
fig, axes = plt.subplots(1, 2, figsize=(figwidth, 2.55), width_ratios=(3.2, 2))

ax = axes[1]
sets = {
    "all": {
        "text": r"$\mathcal{U}$",
        "textxy": (8, 0),
        "xy": [
            (0, 0),
            (10, 0),
            (10, 11),
            (0, 9),
            (0, 0),
        ],
        "kwargs": {
            "facecolor": "none",
            "edgecolor": "C7",
        },
    },
    "sufficient": {
        "text": r"$\mathcal{S}$",
        "textxy": (8, 6),
        "xy": [
            (5, 4),
            (9.5, 5),
            (8, 8),
            (5, 6),
            (5, 4),
        ],
        "kwargs": {
            "edgecolor": "C0",
            "facecolor": mpl.colors.to_rgba("C0", 0.2),
            "ls": "--",
        }
    },
    "lossless": {
        "text": r"$\mathcal{L}$",
        "textxy": (7, 9),
        "xy": [
            (3, 3),
            (10, 3),
            (9, 10),
            (3, 8),
            (3, 3),
        ],
        "kwargs": {
            "facecolor": mpl.colors.to_rgba("C1", 0.2),
            "edgecolor": "C1",
        },
    },
    "considered": {
        "text": r"$\mathcal{T}$",
        "textxy": (2, 1),
        "xy": [
            (1, 1),
            (7, 2),
            (2, 9),
            (1, 1),
        ],
        "kwargs": {
            "edgecolor": "k",
            "facecolor": "none",
        },
    },
}
sets.pop("all")

# Interpolate the points to get smooth sets (https://stackoverflow.com/a/27650158/1150961).
for value in sets.values():
    xy = np.transpose(value["xy"])
    zs = []
    for z in xy:
        # Pad the data for circular boundary conditions.
        orig_len = len(z)
        z = np.concatenate([z[-3:-1], z, z[1:3]])
        t = np.arange(len(z))
        # Values at which to evaluate starting at 2 due to padding.
        ti = np.linspace(2, orig_len + 1, 20 * orig_len)
        interpolated = interpolate.interp1d(t, z, kind='quadratic')(ti)
        zs.append(interpolated)
    value["xy"] = np.transpose(zs)
    value["geometry"] = geometry.Polygon(value["xy"])

# Get the intersection of lossless and considered statistics. They are optimal.
sets["optimal"] = {
    "text": r"$\mathcal{O}$",
    "textxy": (3.5, 7),
    "xy": np.transpose((sets["lossless"]["geometry"] & sets["considered"]["geometry"]).boundary.xy),
    "kwargs": {
        "facecolor": mpl.colors.to_rgba("C2", 0.2),
        "edgecolor": "C3",
        "linewidth": 3,
    },
}

# Plot all the sets.
for key, value in sets.items():
    kwargs = value.get("kwargs", {})
    kwargs["facecolor"] = "none"
    patch = mpl.patches.Polygon(value["xy"], **kwargs)
    ax.add_patch(patch)
    textkwargs = {
        "color": kwargs["edgecolor"],
        "ha": "center",
        "va": "center",
        "fontsize": 12,
    }
    textkwargs.update(value.get("textkwargs", {}))
    ax.text(*value["textxy"], value["text"], **textkwargs, zorder=9)

ax.set_xlim(0, 11)
ax.set_ylim(-3, 10.6)
ax.set_aspect("equal")
ax.set_axis_off()


# Show the relationships between different methods.
ax = axes[0]
hspace = 2.5
info_vspace = 0.7
texts = {
    "epe": (0, 2 * info_vspace, "Min. expected\nposterior entropy"),
    "cpde": (0, 1 * info_vspace, "Cond. posterior\ndensity estimation"),
    "kl": (0, 0, "Min. expected\nKL divergence"),
    "mi": (0, -1 * info_vspace, "Max. mutual\ninformation"),
    "surprise": (0, -2 * info_vspace, "Max. expected\nsurprise"),
    "approx": (hspace, 2, "Approximate\nsufficiency"),
    "cpe": (hspace, 1, "Min. cond.\nposterior density"),
    "selection": (hspace, 0, "Probabilistic\nmodel selection"),
    "fisher": (hspace, -1, "Max. Fisher\ninformation"),
    "risk": (hspace, -2, "Min. $L^2$\nrisk"),
}
boxstyle = "round,pad=0.5"
bbox = {
    "boxstyle": boxstyle,
    "facecolor": "w",
}

elements = {}
for key, (x, y, text) in texts.items():
    element = ax.text(x, y, text, ha="center", va="center", fontsize="small", bbox=bbox)
    elements[key] = element

hpad = 1.2
ax.set_xlim(-hpad, hspace + hpad)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")

ax.set_axis_off()
fig.tight_layout()
fig.draw_without_rendering()

scale = ax.transAxes.get_matrix()[0, 0] / ax.transData.get_matrix()[0, 0]
width = 2
info = mpl.patches.FancyBboxPatch(
    (-width / 2, -3.25 * info_vspace), width, 5.75 * info_vspace, boxstyle=boxstyle, mutation_scale=1 / scale,
    color="C0", alpha=0.2,
)
ax.add_patch(info)
ax.text(0, -2, "information-theoretic\napproaches", color="C0", ha="center", va="center", fontsize="small")

fig.draw_without_rendering()

info_right = get_anchor(info, 3).x
paths = [
    ("--", [
        get_anchor(info, 0),
        (get_anchor(info, 0).x, get_anchor(elements["approx"], 9).y),
        get_anchor(elements["approx"], 9)
    ]),
    ("--", [
        (info_right, get_anchor(elements["cpe"], 9).y),
        get_anchor(elements["cpe"], 9),
    ]),
    ("--", [
        (info_right, get_anchor(elements["selection"], 9).y),
        get_anchor(elements["selection"], 9),
    ]),
    ("-", [
        (info_right, get_anchor(elements["fisher"], 9).y),
        get_anchor(elements["fisher"], 9),
    ]),
    ("-", [
        (info_right, get_anchor(elements["risk"], 9).y),
        get_anchor(elements["risk"], 9),
    ]),
    ("-", [
        get_anchor(elements["risk"], 0),
        get_anchor(elements["fisher"], 6),
    ]),
]
for ls, path in paths:
    path = rounded_path(path, 0.1, shrink=0.05)
    patch = mpl.patches.PathPatch(path, facecolor="none", edgecolor="gray", ls=ls)
    ax.add_patch(patch)
    arrow = mpl.patches.PathPatch(arrow_path(path, 0.075), color="gray")
    ax.add_patch(arrow)

    # Add the bidirectional arrow for large-sample correspondence.
    if ls == "-":
        arrow = mpl.patches.PathPatch(arrow_path(path, 0.075, backward=True), color="gray")
        ax.add_patch(arrow)

# Add the legend for different types of connections.
handles = [
    mpl.lines.Line2D([], [], color="gray", ls="--"),
    mpl.lines.Line2D([], [], color="gray"),
]
labels = [
    "special case",
    "large-sample limit",
]
ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(.9, 0))

label_axes(axes)
fig.savefig("../workspace/figures/illustration.pdf")
```
