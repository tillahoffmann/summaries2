```python
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from scipy import optimize, stats
from summaries.experiments import benchmark
from summaries.util import load_pickle
import torch


mpl.style.use("../.mplstyle")
```

```python
def evaluate_log_likelihood(x, theta):
    """
    Evaluate the log likelihood for the benchmark problem.
    """
    assert theta.shape[-1] == 1
    locs = theta.tanh()[..., None]
    locs = torch.stack([locs, - locs], axis=-1)
    scale = (1 - locs ** 2).sqrt()

    components = torch.distributions.Normal(locs, scale)
    assert components.batch_shape == theta.shape + (1, 2,)

    dist = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(torch.ones((1, 2)) / 2),
        components,
    )
    dist = torch.distributions.Independent(dist, 2)
    assert dist.batch_shape == theta.shape[:-1] and dist.event_shape == (1, 1)
    return dist.log_prob(x[:, :1])


def evaluate_log_joint(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate the log-joint distribution.
    """
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(1), 1), 1)
    return evaluate_log_likelihood(x, theta) + prior.log_prob(theta)
```

```python
root = Path("../workspace/benchmark-small")

observed = load_pickle(root / "data/test.pkl")
mdnabc = load_pickle(root / "samples/BenchmarkNeuralConfig-BenchmarkMixtureDensityConfigReduced.pkl")
transformer = load_pickle(root / "transformers/BenchmarkMixtureDensityConfigReduced.pkl")["transformer"]
```

```python
# Show the range over which we see summaries.
summaries = transformer.transform(observed["data"]).ravel().detach()
plt.hist(summaries)
```

```python
# Get the data, parameters, and posterior samples.
idx = 3
dtype = torch.get_default_dtype()
theta = torch.as_tensor(observed["params"][idx], dtype=dtype)
X = torch.as_tensor(observed["data"][idx], dtype=dtype)
x = X[:, 0]
samples = mdnabc["samples"][idx].squeeze()

fig, axes = plt.subplots(2, 2, sharex="col", sharey="row")

ax = axes[0, 0] # ----- ----- ----- -----

# Show the likelihood with samples as a rug plot.
lin = 3 * torch.linspace(-1, 1, 200)
ax.plot(lin, evaluate_log_likelihood(lin[:, None, None], theta).exp())
ax.scatter(x, torch.zeros_like(x), marker="|", color="k", zorder=2)
ax.set_ylabel(r"likelihood $g\left(y\mid\theta\right)$")
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, va="top")

ax = axes[0, 1] # ----- ----- ----- -----

# Show the ABC samples obtained using MDN-compression.
vmax = np.max(np.abs(samples))
ax.hist(samples, density=True, range=(-vmax, vmax), bins=15, color="silver", label="ABC")

# Show the exact posterior obtained by numerical integration.
log_joint = evaluate_log_joint(X, lin[:, None])
joint = (log_joint - log_joint.max()).exp()
joint = joint / np.trapz(joint, lin)
ax.plot(lin, joint, label="exact")

# Show the mixture density network approximation.
with torch.no_grad():
    distribution = transformer(X)
    log_posterior = distribution.log_prob(lin[:, None])
ax.plot(lin, log_posterior.exp(), label="MDN")

ax.axvline(theta, color="k", ls=":")
ax.set_ylabel(r"posterior $f\left(\theta\mid y_0\right)$")
ax.legend()
ax.text(theta + 0.1, 0.95, r"$\theta_0$", va="top", transform=ax.get_xaxis_transform())
ax.set_ylim(top=0.6)
ax.text(0.05, 0.95, "(b)", transform=ax.transAxes, va="top")

ax = axes[1, 0]  # ----- ----- ----- -----

# Show the summary statistics function and scatter of observed data.
with torch.no_grad():
    features = lin[:, None, None]
    features = torch.concatenate([features, torch.zeros_like(features)], axis=-1)
    transformed = transformer.transform(features)
    transformed_pts = transformer.transform(X[:, None])
    summary = transformer.transform(X)
line, = ax.plot(lin, transformed, color="C1", label="compressor")
ax.scatter(x, transformed_pts, edgecolor="w", facecolor=line.get_color(), zorder=9, clip_on=False)

# Fit to the curve using the first few moments.
def func(x, *params):
    moments = 2 * np.arange(len(params))
    features = np.asarray(x)[:, None] ** moments
    return features @ params

target = transformed.squeeze()
sigma = 1 / np.sqrt(stats.norm(0, 1).pdf(lin))
np.random.seed(0)
params, _ = optimize.curve_fit(func, lin, target, np.random.normal(0, 1e-2, 4), sigma=sigma)
ax.plot(lin, func(lin, *params), ls="--", color="C2", label="fit")

ax.axhline(summary, color="k", ls=":")
ax.set_xlabel(r"data $y$")
ax.set_ylabel(r"summary $t(y)$")
ax.text(0.95, summary + 0.1, r"$t(y_0)$", ha="right", transform=ax.get_yaxis_transform())
ax.text(0.05, 0.95, "(c)", transform=ax.transAxes, va="top")
ax.legend(ncol=2, loc="upper right")

ax = axes[1, 1]  # ----- ----- ----- -----

# We first get a batch of distributions where each element corresponds to one value of the summary
# statistic. Then, for each value of the parameter, we evaluate the batch of distributions.
tmin = - 0.5
tmax = 3.5
ts = torch.linspace(tmin, tmax, 201)
with torch.no_grad():
    dists = transformer(data=None, transformed=ts[:, None])
density = torch.vstack([dists.log_prob(theta) for theta in lin[:, None]]).exp()

cmap = mpl.cm.viridis.copy()
ax.imshow(density.T, extent=(lin.min(), lin.max(), ts.min(), ts.max()), aspect="auto", cmap=cmap)
ax.axhline(summary, color="w", ls=":")
ax.axvline(theta, color="w", ls=":")
ax.set_xlabel(r"parameter $\theta$")
ax.set_ylabel(r"summary $t(y)$")

ax.text(theta + 0.1, 0.95, r"$\theta_0$", color="w", transform=ax.get_xaxis_transform(), va="top")
ax.text(0.95, summary + 0.1, r"$t(y_0)$", color="w", ha="right",
        transform=ax.get_yaxis_transform())
ax.text(0.05, 0.95, "(d)", transform=ax.transAxes, va="top", color="w")
ax.set_ylim(tmin, tmax)

fig.tight_layout()
fig.savefig("../workspace/figures/benchmark.pdf", bbox_inches="tight")

theta
```
