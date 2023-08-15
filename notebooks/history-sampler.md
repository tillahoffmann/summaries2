```python
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from summaries.util import load_pickle


mpl.style.use("../.mplstyle")
```

```python
results = {
    "small": load_pickle("../workspace/tree-small/samples/TreeKernelHistorySamplerConfig.pkl"),
    "large": load_pickle("../workspace/tree-large/samples/TreeKernelHistorySamplerConfig.pkl"),
}
```

```python
fig, axes = plt.subplots(2, 2, sharex=True, sharey="row")
for col, (key, result) in zip(axes.T, results.items()):
    ax1, ax2 = col
    ax1.scatter(result["params"].ravel(), result["spearman"], marker=".")
    residuals = (result["samples"] - result["params"][:, None]).squeeze()
    residuals /= np.std(residuals, axis=1, keepdims=True)
    ax2.scatter(result["params"].ravel(), residuals.mean(axis=1).ravel(), marker=".")
    ax2.axhline(0, color="k", ls="--", zorder=9)
    ax2.set_xlabel(r"kernel exponent $\gamma$")
    
ax1, ax2 = axes[:, 0]
ax1.set_ylabel("Spearman rank\ncorrelation")
ax2.set_ylabel("$z$-score")

ax1, ax2 = axes[0]
ax1.set_title("$n=100$")
ax2.set_title("$n=748$")

fig.tight_layout()
fig.savefig("../workspace/figures/tree-history.pdf")
```
