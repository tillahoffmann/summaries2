import numpy as np
from pathlib import Path
import pickle
from summaries.scripts.infer_mdn import __main__
from torch import nn, Tensor
from torch.distributions import Normal


class DummyModule(nn.Module):
    def forward(self, x: Tensor) -> None:
        return Normal(x.mean(axis=1, keepdims=True), 1)


def test_infer_mdn(tmp_path: Path, observed_data: np.ndarray) -> None:
    output_path = tmp_path / "output.pkl"
    mdn_path = tmp_path / "mdn.pkl"
    observed_path = tmp_path / "observed.pkl"

    with observed_path.open("wb") as fp:
        pickle.dump({
            "data": observed_data,
        }, fp)

    with mdn_path.open("wb") as fp:
        pickle.dump({
            "transformer": DummyModule(),
        }, fp)

    __main__(["--n-samples=17", str(mdn_path), str(observed_path), str(output_path)])

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    assert output["samples"].shape == (observed_data.shape[0], 17, 1)