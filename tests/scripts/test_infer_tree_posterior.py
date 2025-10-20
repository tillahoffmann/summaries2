from pathlib import Path
import pickle
import pytest
from scipy import stats
from summaries.scripts.infer_tree_posterior import __main__
from summaries.scripts.simulate_data import __main__ as __main__simulate_data


@pytest.mark.filterwarnings("ignore:divide by zero encountered in power")
def test_infer_tree_posterior(tmp_path: Path) -> None:
    n = 23
    observed_path = tmp_path / "observed.pkl"
    output_path = tmp_path / "output.pkl"

    __main__simulate_data(
        [
            f"--n-samples={n}",
            "--n-observations=17",
            "TreeSimulationConfig",
            str(observed_path),
        ]
    )
    __main__(["--n-samples=13", str(observed_path), str(output_path)])

    with observed_path.open("rb") as fp:
        observed = pickle.load(fp)

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    # Check true and inferred are correlated.
    assert observed["params"].shape == (n, 1)
    assert output["map_estimate"].shape == (n, 1)
    pearsonr = stats.pearsonr(
        observed["params"].ravel(), output["map_estimate"].ravel()
    )
    assert pearsonr.statistic > 0.5
    assert pearsonr.pvalue < 1e-2

    # Check sample shape.
    assert output["samples"].shape == (n, 13, 1)

    pytest.shared.assert_pickle_loadable(output_path)
