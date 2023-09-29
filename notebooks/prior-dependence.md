```python
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from snippets.plot import label_axes
from summaries.entropy import estimate_mutual_information
from scipy import stats

mpl.style.use("../.mplstyle")
```

```python
def generate_data(m: int, n: int, scale: float) -> dict:
    """
    Generate synthetic data that exemplifies the sensitivity of mutual information to prior choice.

    Args:
        m: Number of independent samples for estimating the entropy.
        n: Number of observations per sample.
        scale: Scale of each prior distribution.
    """
    mus = {'left': -1, 'right': 1}
    results = {}
    for key, mu in mus.items():
        # Sample from the prior and likelihood.
        theta = np.random.normal(mu, scale, m)
        left = np.random.normal(0, np.exp(theta[:, None] / 2), (m, n))
        right = np.random.normal(theta[:, None], 1, (m, n))
        x = np.where(theta[:, None] < 0, left, right)

        # Evaluate the summary statistics.
        mean = x.mean(axis=-1)
        log_var = np.log(x.var(axis=-1))

        # Store the results in a dictionary for later plotting.
        results[key] = {
            'mu': mu,
            'theta': theta,
            'x': x,
            'mean': mean,
            'log_var': log_var,
            'mi_mean': estimate_mutual_information(theta[:, None], mean[:, None]),
            'mi_log_var': estimate_mutual_information(theta[:, None], log_var[:, None]),
        }

    return results


np.random.seed(0)

m = 100_000  # Number of independent samples for estimating the entropy.
n = 100  # Number of observations per sample.
scale = 0.25  # Scale of each prior distribution.
num_points = 200  # Number of points in the figure (we sample more for MI estimates).

results = generate_data(m, n, scale)

fig, axes = plt.subplots(2, 2, sharex=True)

# Show the two priors.
ax = axes[1, 0]
for result in results.values():
    mu = result['mu']
    lin = mu + np.linspace(-1, 1, 100) * 3 * scale
    prior = stats.norm(mu, scale)
    label = fr'$\theta\sim\mathrm{{Normal}}\left({mu}, 0.1\right)$'
    line, = ax.plot(lin, prior.pdf(lin), label=label)
    ax.axvline(mu, color=line.get_color(), ls='--')
ax.set_ylabel(r'prior $\pi(\theta)$')

# Show the likelihood parameters as a function of the parameter.
ax = axes[0, 0]
lin = np.linspace(-1, 1, 100) * (1 + 3 * scale)
ax.plot(lin, np.maximum(0, lin), label=r'location', color='k')
ax.plot(lin, np.minimum(np.exp(lin / 2), 1), label=r'scale', color='k', ls='--')
ax.set_ylabel('likelihood parameters')

# Plot the scatter of summaries against parameter value for both priors.
step = m // num_points  # Only plot `num_points` for better visualisation.
for key, result in results.items():
    for ax, s in zip(axes[:, 1], ['mean', 'log_var']):
        mi = result[f"mi_{s}"].mean()
        # Very close to zero, we may end up with negative results. We manually fix that to avoid
        # "-0.00" instead of "0.00" in the figure. This is a hack but consistent with the sklearn
        # implementation for mutual information (see https://bit.ly/3NXRn5r).
        if abs(mi) < 1e-3:
            mi = abs(mi)
        ax.scatter(result['theta'][::step], result[s][::step], marker='.', alpha=.5,
                    label=fr'${key.title()}$ ($\hat{{I}}={mi:.2f}$)')

# Set axes labels and label each panel.
axes[0, 1].set_ylabel(r'summary $\bar y$')
axes[1, 1].set_ylabel(r'summary $\log\mathrm{var}\,y$')
[ax.set_xlabel(r'parameter $\theta$') for ax in axes[1]]

axes[0, 0].legend()
[ax.legend(handletextpad=0, loc=loc)
    for ax, loc in zip(axes[:, 1], ['upper left', 'lower right'])]

label_axes(axes[0], loc='bottom right')
label_axes(axes[1], loc='top left', label_offset=2)

fig.tight_layout()
fig.savefig('../workspace/figures/prior-dependence.pdf')
```
