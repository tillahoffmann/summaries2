```python
from IPython.display import display
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

mpl.style.use("../.mplstyle")
figwidth, figheight = mpl.rcParams["figure.figsize"]
```

```python
name_normalizations = {
    "PriorConfig.pkl": "Prior",
    
    "BenchmarkStanConfig.pkl": "Likelihood-based",
    "mdn-BenchmarkMixtureDensityConfig.pkl": "<print>",
    "BenchmarkLinearPosteriorMeanConfig.pkl": "Linear regression",
    "BenchmarkNeuralConfig-BenchmarkPosteriorMeanConfig.pkl": "Nonlinear regression",
    "BenchmarkNeuralConfig-BenchmarkMixtureDensityConfig.pkl": "MDN compression",
    "BenchmarkMinimumConditionalEntropyConfig.pkl": "Minimum CPE",
    "mdn-BenchmarkMixtureDensityConfigReduced.pkl": "MDN",
    "BenchmarkNeuralConfig-BenchmarkMixtureDensityConfigReduced.pkl": "<hide>",
    "BenchmarkExpertSummaryConfig.pkl": "Expert summaries",
    "BenchmarkPLSConfig.pkl": "PLS",
    "BenchmarkApproximateSufficiencyConfig.pkl": "Approx. sufficiency",
    "BenchmarkApproximateSufficiencyLikelihoodRatioConfig.pkl": "Approx. sufficiency (LR)",

    "CoalescentExpertSummaryConfig.pkl": "Expert summaries", 
    "mdn-CoalescentMixtureDensityConfig.pkl": "MDN", 
    "CoalescentNeuralConfig-CoalescentMixtureDensityConfig.pkl": "MDN compression", 
    "CoalescentLinearPosteriorMeanConfig.pkl": "Linear regression", 
    "CoalescentMinimumConditionalEntropyConfig.pkl": "Minimum CPE", 
    "CoalescentNeuralConfig-CoalescentPosteriorMeanConfig.pkl": "Nonlinear regression",
    "CoalescentPLSConfig.pkl": "PLS",

    "TreeKernelExpertSummaryConfig.pkl": "Expert summaries", 
    "TreeKernelHistorySamplerConfig.pkl": "Likelihood-based", 
    "TreeKernelNeuralConfig-TreeMixtureDensityConfig.pkl": "MDN compression",
    "mdn-TreeMixtureDensityConfig.pkl": "MDN",
    "TreeKernelNeuralConfig-TreePosteriorMeanConfig.pkl": "Nonlinear regression",
    "TreeKernelLinearPosteriorMeanConfig.pkl": "Linear regression",
    "TreeKernelPLSConfig.pkl": "PLS",
    "TreeKernelMinimumConditionalEntropyConfig.pkl": "Minimum CPE",
    "TreeKernelApproximateSufficiencyConfig.pkl": "Approx. sufficiency",
    "TreeKernelApproximateSufficiencyLikelihoodRatioConfig.pkl": "Approx. sufficiency (LR)",
}

root = Path("../workspace/")
statistics = {
    "benchmark": pd.read_csv(root / "benchmark-small/evaluation.csv"),
    "coalescent": pd.read_csv(root / "coalescent/evaluation.csv"),
    "tree": pd.read_csv(root / "tree-small/evaluation.csv"),
}
for key, value in statistics.items():
    # Normalize the names and remove "hidden" configurations.
    value["name"] = value.path.map(name_normalizations)
    missing_names = set(value.path[value.name.isnull()])
    if missing_names:
        raise ValueError(missing_names)
    print(value[value.name == "<print>"])
    value = value[~value.name.isin(["<hide>", "<print>"])].copy()
    value["experiment"] = key
    del value["path"]

    # Round to the nearest digit *for each* combination of statistics and experiment.
    for stat in ["rmise", "nlp"]:
        value[f"{stat}_digits"] = digits = math.ceil(-math.log10(2 * value[f"{stat}_err"].min()))
    
    statistics[key] = value

order = [
    "Likelihood-based", "MDN compression", "MDN", "Nonlinear regression", "Linear regression", 
    "Minimum CPE", "Approx. sufficiency (LR)", "Approx. sufficiency", "Expert summaries", "PLS", 
    "Prior",
]
statistics = pd.concat(statistics.values())
parts = []
for key, subset in statistics.sort_values("nlp").groupby("experiment"):
    formatted = {
        "method": subset.name, 
    }
    for stat in ["nlp", "rmise"]:
        best, best_err = subset.sort_values(stat).iloc[0][[stat, f"{stat}_err"]]
        delta = subset[stat] - best
        err = np.sqrt(best_err ** 2 + subset[f"{stat}_err"] ** 2)
        within_best = delta < err
        column = []
        for w, v, e, d in zip(within_best, subset[stat], subset[f"{stat}_err"], subset[f"{stat}_digits"]):
            v = np.round(v, d)
            e = np.round(e, d)
            cell = f"{v:.{d}f} \\pm {e:.{d}f}"
            if w:
                cell = f"\\mathbf{{{cell}}}"
            column.append(f"${cell}$")
        formatted[f"{key}_{stat}"] = column
    formatted = pd.DataFrame(formatted)
    missing = set(formatted["method"]) - set(order)
    if missing:
        raise ValueError(missing)

    # Reorder for consistency.
    display(formatted)
    formatted = formatted.set_index("method").reindex(order).loc[order]
    parts.append(formatted)

formatted = pd.concat(parts, axis=1)
```

```python
display(formatted)
print(formatted.to_latex())
```

```python
# Show the expected properties under the prior. The rmise based on a sample is biased.
priors = {
    "benchmark": stats.norm(np.zeros(1), np.ones(1)),
    "coalescent": stats.uniform(np.zeros(2), 10 * np.ones(2)),
    "tree": stats.uniform(np.zeros(1), 2 * np.ones(2)),
}

for experiment, prior in priors.items():
    p, = prior.mean().shape
    ref = prior.rvs([10000, p])
    samples = prior.rvs([ref.shape[0], 1000, p])

    nlp = - prior.logpdf(samples).sum(axis=-1).mean(axis=-1)
    mise = np.square(ref[:, None, :] - samples).sum(axis=-1).mean(axis=-1)
    rmise = np.sqrt(mise)
    print(f"{experiment}: mise={mise.mean():.3f}, rmise={rmise.mean():.3f}, nlp={nlp.mean():.3f}")
```

```python
# Show relative performance of the different methods.
exclude_prior = True

normalized = []
columns = []
for experiment, subset in statistics.sort_values("nlp").groupby("experiment"):
    # Slicing the end of the order to exclude the prior because it has poor metrics.
    subset = subset.set_index("name").reindex(order[:-1] if exclude_prior else order)
    for key in ["rmise", "nlp"]:
        value = subset[key].values
        value -= np.nanmin(value)
        value /= np.nanmax(value)
        normalized.append(value)
        
        columns.append(f"{experiment}-{key}")
        
normalized = np.transpose(normalized)
        
fig, ax = plt.subplots()
im = ax.imshow(normalized, origin="upper")

ax.set_xticks(np.arange(len(columns)))
ax.set_xticklabels(columns, rotation=30, ha="left")
ax.xaxis.tick_top()

ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(order)
fig.colorbar(im, ax=ax, label="relative loss")

```

```python
plt.plot(np.nanmean(normalized[:, 3::2], axis=1), marker="o")
plt.xticks(np.arange(len(order)), order, rotation=90)
plt.grid()
pass
```

```python
fig = plt.figure(figsize=(figwidth, 0.9 * figheight))
gs = fig.add_gridspec(2, 2)

# Declare how each point should be visualized.
metadata = {
    "Likelihood-based": ("k", "o"),
    "Expert summaries": ("C0", ">"),
    "MDN compression": ("C1", "s"),
    "Linear regression": ("C2", "v"),
    "Nonlinear regression": ("C3", "^"),
    "MDN": ("C4", "D"),
    "Minimum CPE": ("C5", "<"),
    "PLS": ("C6", "p"),
}

# Declare points that should be indicated as off-the-charts.
off_the_charts = {
    ("tree", "PLS"): (0.875, 0.95),
    ("tree", "Expert summaries"): (0.95, 0.95),
}

# Iterate over experiments and results.
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
items = zip(
    axes, 
    "abc",
    statistics.groupby("experiment"),
)
artists = {}
for ax, axis_label, (experiment, subset) in items:
    for _, row in subset.iterrows():
        # Complain if we don't recognize the experiment.
        name = row["name"]
        if name not in metadata:
            print(row.experiment, name)
            continue
            
        color, marker = metadata[name]
        
        # Add the experiment-method combination.
        if xy := off_the_charts.get((experiment, name)):
            ax.scatter(*xy, transform=ax.transAxes, facecolor=color, edgecolor="w", 
                       marker=marker, zorder=10)
            ax.arrow(*xy, 0.05, 0.1, transform=ax.transAxes, clip_on=False, color=color, 
                     zorder=9, head_width=0.02)
        else:
            artist = ax.errorbar(row.nlp, row.rmise, row.rmise_err, row.nlp_err, ls="none",
                                 label=name, marker=marker, color=color, markeredgecolor="w")
        artists.setdefault(name, artist)
        
    ax.text(0.05, 0.95, f"({axis_label}) {experiment}", transform=ax.transAxes, va="top")
    
ax = axes[2]
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
ax.yaxis.major.formatter.set_powerlimits((0, 0))
ax.yaxis.major.formatter.set_useMathText(True)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    
# Add axis labels.
for ax in axes[:2]:
    ax.set_ylabel("root mean integrated\nsquared error")
for ax in axes[1:]:
    ax.set_xlabel("negative log probability")
    
# Add legend in separate plot.
ax = fig.add_subplot(gs[0, 1])
labels, handles = zip(*artists.items())
ax.legend(handles, labels)
ax.set_axis_off()

fig.tight_layout()
fig.savefig("../workspace/figures/evaluation.pdf")
```
