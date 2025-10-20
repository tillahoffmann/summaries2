"""Test for MDN tree inference."""

import pickle
from pathlib import Path
import torch
from summaries.experiments.tree import TreePosteriorMixtureDensityTransformer
from summaries.scripts.infer_mdn import __main__


def test_infer_mdn_tree(tmp_path: Path) -> None:
    output_path = tmp_path / "output.pkl"
    mdn_path = tmp_path / "mdn.pkl"
    observed_path = tmp_path / "observed.pkl"

    observed_data = torch.randint(0, 10, (7, 9))
    params = torch.randn(7, 2)

    with observed_path.open("wb") as fp:
        pickle.dump(
            {
                "data": observed_data,
                "params": params,
            },
            fp,
        )

    with mdn_path.open("wb") as fp:
        pickle.dump(
            {
                "transformer": TreePosteriorMixtureDensityTransformer(),
            },
            fp,
        )

    __main__(
        [
            "--n-samples=17",
            "--loader=tree",
            str(mdn_path),
            str(observed_path),
            str(output_path),
        ]
    )

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    assert output["samples"].shape == (observed_data.shape[0], 17, 1)
    torch.testing.assert_close(output["params"], params)
