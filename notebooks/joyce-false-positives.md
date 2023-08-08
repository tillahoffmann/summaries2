This notebook illustrates that Joyce and Marjoram's (2008) approximate sufficiency method is more likely to incorrectly reject the null hypothesis as the number of bins of the histogram increases. This is a result of multiple hypothesis testing without correction. We do not apply for our manuscript because it is only suitable for one-dimensional parameters (density estimation using histograms in more than one dimension requires a lot of samples). We thus also do not dig into this particular issue of the method because it seems unnecessary.

```python
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
from snippets.plot import plot_band
from tqdm.notebook import tqdm
from summaries.transformers import ApproximateSufficiencyTransformer

mpl.style.use("../.mplstyle")
```

```python
n_samples = 1_000
n_examples = 1_000_000
n_observed = 500
observed_data = np.random.uniform(0, 1, (n_observed, 1))
simulated_data = np.random.uniform(0, 1, (n_examples, 1))
simulated_params = np.random.uniform(0, 1, (n_examples, 1))
bins = np.arange(2, 20, 2)
statistics = {}
range_ = (0, 1)
alpha = 0.05

for observed in tqdm(observed_data):
    for n in bins:
        kwargs = {
            "observed_data": observed,
            "range_": (0, 1),
            "n_samples": n_samples,
            "bins": n,
        }
        
        # Use the standard setup.
        transformer = ApproximateSufficiencyTransformer(**kwargs)
        transformer.fit(simulated_data, simulated_params)
        statistics.setdefault("approximate_sufficiency", []).append(bool(transformer.features_))
        
        # Use the likelihood ratio test.
        transformer = ApproximateSufficiencyTransformer(**kwargs, likelihood_ratio=True)
        transformer.fit(simulated_data, simulated_params)
        statistics.setdefault("likelihood_ratio", []).append(bool(transformer.features_))
        
statistics = {key: np.reshape(value, (n_observed, bins.size)) for key, value in statistics.items()}
```

```python
fig, ax = plt.subplots()

labels = {
    "likelihood_ratio": "likelihood ratio",
    "approximate_sufficiency": "Joyce and Marjoram (2008)",
}

for key, value in statistics.items():
    mean = value.mean(axis=0)
    err = 1.96 * value.std(axis=0) / np.sqrt(value.shape[0] - 1)
    line, = ax.plot(bins, mean, label=labels[key])
    ax.fill_between(bins, mean - err, mean + err, color=line.get_color(), alpha=0.2)
    
ax.legend()
ax.axhline(alpha, color="k", ls=":")
ax.set_xlabel("number of bins")
ax.set_ylabel("false discovery rate")
fig.tight_layout()
fig.savefig("../workspace/figures/joyce.pdf")
```
