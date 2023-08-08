```python
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from snippets.plot import label_axes, rounded_path
import pandas as pd

mpl.style.use("../.mplstyle")
```

```python
statistics = pd.read_csv("../workspace/coalescent/evaluation.csv")
statistics
```

```python
fig, axes = plt.subplots(1, 2, width_ratios=[2, 3])

# A reasonably hacky approach to generating the illustrative model for the coalescent...
ax = axes[0]
# Size of each individual (square or circle).
size = 0.65
# Sex indicator for individuals in each layer, corresponding to generations. We tack on an
# individual with "square" sex to each layer to illustrate that the population is much larger.
layers = [
    [0, 1, 0, 1, ],
    [1, 0, 0, 1, ],
    [1, 1, 0, 1, ],
]

for i, layer in enumerate(layers):
    for j, sex in enumerate(layer):
        kwargs = {'facecolor': 'none', 'edgecolor': 'black'}
        # Use different shapes by sex.
        if sex:
            patch = mpl.patches.Circle((i, -j), size / 2, **kwargs)
        else:
            patch = mpl.patches.Rectangle((i - size / 2, - j - size / 2), size, size, **kwargs)
        ax.add_patch(patch)

ax.set_ylim(- len(layer) - 1, 1)
ax.set_xlim(-.5, len(layers)-.5)
ax.set_aspect('equal')

# Relationships in each layer. Each tuple corresponds to one of the individuals in `layers`. E.g.,
# a tuple (i, j) at the second position indicates that the person with index 1 in the layer has
# parents i and j in the previous layer.
relationships = [
    [(0, 1), (2, 3), (2, 3), (2, 3)],
    [(0, 1), (0, 1), (3, 4), (3, 4)],
]

# Draw the relationship lines.
for i, layer in enumerate(relationships):
    for j, parents in enumerate(layer):
        for sign in [-1, 1]:
            xy = np.asarray([
                (i - size / 2 + 1, j),
                (i + .5, j),
                (i + .5, np.mean(parents)),
                (i, np.mean(parents)),
                (i, np.mean(parents) + (0.5 - size / 2) * sign),
            ]) * [1, -1]
            path = rounded_path(xy, 0.1, 0.05)
            patch = mpl.patches.PathPatch(path, edgecolor="gray", facecolor="none")
            ax.add_patch(patch)

# Height and which of each illustrated chromosome.
csize = 0.75 * size
cheight = 0.2 * size
# Color of chromosomes by individual. Individuals are identified by their position in the grid. The
# value comprises two "chromosomes", each having one or more different genes (indicated by different
# colors).
chromosomes_by_individual = {
    (0, 0): [['C0'], ['C1']],
    (0, 1): [['C4'], ['C7']],
    (0, 2): [['C2'], ['C7']],
    (0, 3): [['C3'], ['C7']],
    (1, 0): [['C0', 'C1'], ['C4']],
    (1, 1): [['C2'], ['C3']],
    (2, 0): [['C0', 'C1'], ['C2']]
}

# Draw the chromosome patches.
for (x, y), chromosomes in chromosomes_by_individual.items():
    for i, colors in enumerate(chromosomes):
        for j, color in enumerate(colors):
            patch = mpl.patches.Rectangle(
                (x - csize / 2 + j / len(colors) * csize, - y - i * cheight),
                csize / len(colors), cheight,
                facecolor=color, alpha=.75)
            ax.add_patch(patch)
        patch = mpl.patches.Rectangle(
            (x - csize / 2, - y - i * cheight),
            csize, cheight,
            facecolor='none', edgecolor='k'
        )
        ax.add_patch(patch)

# Add on a random mutation.
ax.scatter([1, 2], [-1 + cheight / 2, -cheight / 2], marker='.', color='k').set_edgecolor('w')

# Illustrate the direction of time.
y = - len(layer) - .5
ax.arrow(- size / 2, y, len(layers) - 1 + size, 0,
         linewidth=1, head_width=.1, length_includes_head=True, facecolor='k')
ax.text(len(layers) / 2 - .5, y - .1, r'generations', va='top', ha='center')

# Plot the semi-transparent individuals illustrating the larger population (squares are easier to
# plot). This isn't pretty, but it does the trick.
segments = []
z = []
for i in range(len(layers)):
    left = i - size / 2
    right = i + size / 2
    top = -len(layer) + size / 2
    segments.append([(left, top), (right, top)])
    z.append(0.)
    for x in [left, right]:
        previous = None
        for y in np.linspace(0, 1, 10):
            if previous is not None:
                segments.append([(x, top - y * size / 2), (x, top - previous * size / 2)])
                z.append(y)
            previous = y

collection = mpl.collections.LineCollection(segments, array=z, cmap='binary_r', lw=1)
ax.add_collection(collection)
ax.yaxis.set_ticks([])
ax.xaxis.set_ticks([])
ax.set_axis_off()

ax = axes[1]
metadata = {
    "mdn-CoalescentMixtureDensityConfig.pkl": ("MDN", "o"),
    "CoalescentNeuralConfig-CoalescentMixtureDensityConfig.pkl": ("MDN compression", "s"),
    "CoalescentNeuralConfig-CoalescentPosteriorMeanConfig.pkl": ("nonlinear regression", "X"),
    "CoalescentLinearPosteriorMeanConfig.pkl": ("linear regression", "v"),
    "CoalescentMinimumConditionalEntropyConfig.pkl": ("minimum CPE", "D"),
    "CoalescentExpertSummaryConfig.pkl": ("expert summaries", "<"),
    "CoalescentPLSConfig.pkl": ("PLS", "p"),
}

for _, row in statistics.iterrows():
    if row.path == "PriorConfig.pkl":
        continue
    label, marker = metadata[row.path]
    ax.errorbar(row.nlp, row.rmse, row.rmse_err, row.nlp_err, label=label, marker=marker,
                markeredgecolor="w")

ax.legend(loc="lower right")
ax.set_xlabel("negative log probability")
ax.set_ylabel("root mean squared error")

label_axes(axes)

fig.tight_layout()
fig.savefig('../workspace/figures/coalescent.pdf')
```
