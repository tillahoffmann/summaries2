from pathlib import Path
import pickle
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.scripts.infer_graph_posterior import __main__, TreeKernelPosterior
from summaries.scripts.simulate_data import __main__ as __main__simulate_data, GraphSimulationConfig
from unittest import mock


@pytest.mark.filterwarnings("ignore:divide by zero encountered in power")
def test_infer_graph_posterior(tmp_path: Path) -> None:
    n = 23
    observed_path = tmp_path / "observed.pkl"
    output_path = tmp_path / "output.pkl"

    with mock.patch.object(GraphSimulationConfig, "N_NODES", 23):
        __main__simulate_data([f"--n-samples={n}", "GraphSimulationConfig", str(observed_path)])
        __main__([str(observed_path), str(output_path)])

    with observed_path.open("rb") as fp:
        observed = pickle.load(fp)

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    # Check true and inferred are correlated.
    assert observed["params"].shape == (n, 1)
    assert output["map_estimate"].shape == (n, 1)
    pearsonr = stats.pearsonr(observed["params"].ravel(), output["map_estimate"].ravel())
    assert pearsonr.statistic > 0.5
    assert pearsonr.pvalue < 1e-3


def test_posteriot_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1))._log_target(0.5)
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1)).log_prob(0.5)
