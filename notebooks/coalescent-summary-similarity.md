```python
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy import stats
from summaries.util import load_pickle
import torch
from tqdm.notebook import tqdm
```

```python
# Load data and transformers and obtain summaries.
train = load_pickle("../workspace/coalescent/data/train.pkl")
transformers = {config: load_pickle(f"../workspace/coalescent/transformers/{config}.pkl") for config 
                in ["CoalescentMixtureDensityConfig", "CoalescentPosteriorMeanConfig"]}

with torch.no_grad():
    summaries = {config: transformer["transformer"].transform(train["data"]) for config, transformer 
                 in transformers.items()}
```

```python
# Compute the discrepancy measure for the summaries.
a, b = summaries.values()
mtx1, mtx2, disparity = procrustes(a, b)
disparity
```

```python
# Compare with disparity for random points.
disparities = np.asarray([procrustes(a, b[np.random.permutation(b.shape[0])])[-1] 
                          for _ in tqdm(range(100))])

# Show the distribution of random disparities on the log10 scale.
plt.hist(np.log10(1 - disparities))
plt.axvline(np.log10(1 - disparity))
disparities.mean()
```

```python
# Plot the summaries for visual inspection.
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
step = 1000
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.scatter(mtx1[::step, i], mtx2[::step, i], c=train["params"][::step, j])
        ax.xaxis.major.formatter.set_powerlimits((0, 0))
        ax.xaxis.major.formatter.set_useMathText(True)
        ax.yaxis.major.formatter.set_powerlimits((0, 0))
        ax.yaxis.major.formatter.set_useMathText(True)
    print(i, np.corrcoef(mtx1[:, i], mtx2[:, i])[0, 1])
    
fig.tight_layout()
```
