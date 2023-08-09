```python
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from operator import sub
from scipy import stats
from snippets.plot import label_axes


mpl.style.use("../.mplstyle")
```

```python
def get_aspect(ax):
    """
    Get the actual aspect ratio of the figure so we can plot arrows orthogonal to curves.
    """
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def evaluate_entropy(a, b):
    "Evaluate the entropy of a gamma distribution."
    return stats.gamma(a, scale=1 / b).entropy()


def evaluate_posterior_params(a, b, n, second_moment):
    "Evaluate the parameters of a gamma posterior distribution (normal likelihood with known mean)."
    return a + n / 2, b + n * second_moment / 2


def evaluate_posterior_entropy(a, b, n, second_moment):
    "Evaluate the entropy of a gamma posterior distribution."
    return evaluate_entropy(*evaluate_posterior_params(a, b, n, second_moment))


def evaluate_entropy_gain(a, b, n, second_moment):
    "Evaluate the gain in entropy in light of data."
    return evaluate_posterior_entropy(a, b, n, second_moment) - evaluate_entropy(a, b)


n = 4  # Number of observations.
b = 1  # Scale parameter of the gamma distribution.

# Build a grid over the shape parameter and the realized second moment to evaluate the entropy gain.
a = np.linspace(1, 4, 100)
second_moment = np.linspace(.2, .725, 101)
aa, ss = np.meshgrid(a, second_moment)
entropy_gain = evaluate_entropy_gain(aa, b, n, ss)

fig, axes = plt.subplots(1, 2)

# Plot entropy gain with "centered" colorbar.
ax = axes[0]
vmax = np.abs(entropy_gain).max()
mappable = ax.pcolormesh(a, second_moment, entropy_gain, vmax=vmax, vmin=-vmax,
                         cmap='coolwarm', rasterized=True)
cs = ax.contour(a, second_moment, entropy_gain, levels=[0], colors='k', linestyles=':')

# Plot the expected second moment.
ax.plot(a, 1 / a, color='k', ls='--',
        label='expected second\nmoment 'r'$\left\langle t\right\rangle$')

# Consider a particular example.
a0 = 1.5
sm0 = 0.3
pts = ax.scatter(a0, sm0, color='k', marker='o', zorder=9, label='Example point')
pts.set_edgecolor('w')

arrowprops = {
    'arrowstyle': '-|>',
    'connectionstyle': 'arc3,rad=-.3',
    'patchB': pts,
    'facecolor': 'k',
}
bbox = {
    'boxstyle': 'round',
    'edgecolor': '0.8',
    'facecolor': '1.0',
    'alpha': 0.8,
}

handles, labels = ax.get_legend_handles_labels()
handles = [cs.legend_elements()[0][0]] + handles
labels = [r'$\Delta=0$'] + labels
ax.legend(handles, labels, loc='upper right')

ax.set_xlabel(r'prior mean $a$')
ax.set_ylabel(r'second moment $t=n^{-1}\sum_{i=1}^n y_i^2$')
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
cb = fig.colorbar(mappable, location='top', ax=ax)
cb.set_label('entropy gain\n'r'$\Delta=H\{f(\theta\mid y)\} - H\{\pi(\theta)\}$')
ax.set_xlim(a[0], a[-1])
ax.set_ylim(second_moment[0], second_moment[-1])


# Show the posterior if we use the absolute value of \theta as the precision.
ax = axes[1]
ax.set_ylabel(r'posterior $p(\theta\mid y_0)$')
ax.set_xlabel(r'parameter $\theta$')

mle = 1 / sm0
ax.axvline(mle, color='k', ls='--')
ax.text(mle + 0.3, 0.95, r"$\widehat{\vert\theta\vert}$",
        transform=ax.get_xaxis_transform(), va="top")
a1, b1 = evaluate_posterior_params(a0, b, n, sm0)
posterior = stats.gamma(a1, scale=1 / b1)
xmax = posterior.ppf(0.99)
lin = np.linspace(-xmax, xmax, 101)
# Posterior needs a factor of 0.5 to be normalized because we have the left and right branch.
ax.plot(lin, posterior.pdf(np.abs(lin)) / 2)
ax.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

label_axes(axes)
fig.tight_layout()

# Draw arrows. This needs to happen after tight layout to get the right aspect.
ax = axes[0]
# First get the curve, then find the index of a vertex close to the reference position x0.
path, = cs.collections[0].get_paths()
x0 = 2.05
i = np.argmax(path.vertices[:, 0] < x0)
# Compute the normals to the curve at the position we've identified.
x, y = path.vertices.T
aspect = get_aspect(ax)
dy = (y[i + 1] - y[i]) * aspect ** 2
dx = x[i + 1] - x[i]
scale = .3 / np.sqrt(dx ** 2 + dy ** 2)

# Draw the arrows pointing to the increasing and decreasing regions for black and white printing.
arrowprops = {'arrowstyle': '<|-', 'facecolor': 'k'}
pt = (x[i] - scale * dy, y[i] + scale * dx)
ax.annotate('', (x[i], y[i]), pt, arrowprops=arrowprops)
ax.text(*pt, r"$\Delta>0$", ha='right', va='center')

pt = (x[i] + scale * dy, y[i] - scale * dx)
ax.annotate('', (x[i], y[i]), pt, arrowprops=arrowprops)
ax.text(*pt, r"$\Delta<0$", ha='left', va='center')

fig.savefig('../workspace/figures/tough-nuts.pdf')
```

```python
# Generate numbers for the text...
prior_entropy = evaluate_entropy(a0, 1)
print(f"prior entropy: {prior_entropy:.2f}")
print(f"posterior entropy: {evaluate_posterior_entropy(a0, 1, n, sm0):.2f}")

# For the expected posterior entropy, we need to sample a bunch from the prior.
N = 100_000
theta = np.random.gamma(a0, 1, N)
z = np.random.normal(0, 1 / np.sqrt(theta[:, None]), (N, n))
sm = np.square(z).mean(axis=1)

# Evaluate the posterior parameters and evaluate the entropies.
a, b = evaluate_posterior_params(a0, 1, n, sm)
epes = evaluate_entropy(a, b)
print(f"expected posterior entropy: {epes.mean():.2f} +- {epes.std() / np.sqrt(N - 1):.2f}")

# Evaluate the fraction of times when the entropy increased.
print(f"fraction of entropy increases: {(epes > prior_entropy).mean()}")
```
