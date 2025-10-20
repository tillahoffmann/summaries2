import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from typing import Any, Dict

from ..experiments.benchmark import simulate_benchmark
from ..experiments.tree import compress_tree, simulate_tree


class SimulationArgs:
    n_samples: int | None
    n_observations: int | None
    seed: int | None
    config: str
    output: Path


class SimulationConfig:
    def __init__(self, args: SimulationArgs) -> None:
        self.args = args

    def simulate(self) -> Dict[str, Any]:
        raise NotImplementedError


class TreeSimulationConfig(SimulationConfig):
    N_SAMPLES = 100
    N_NODES = 748
    PRIOR = stats.uniform(0, 2)

    def simulate(self) -> Dict[str, Any]:
        n_samples = self.args.n_samples or self.N_SAMPLES
        n_nodes = self.args.n_observations or self.N_NODES
        random_state = np.random.RandomState(self.args.seed)
        simulations = {}
        for _ in tqdm(range(n_samples)):
            gamma = self.PRIOR.rvs(random_state=random_state)
            tree = simulate_tree(n_nodes, gamma, seed=random_state)
            simulations.setdefault("data", []).append(compress_tree(tree))
            simulations.setdefault("params", []).append(gamma)

        # Reshape to match the shape expected by subsequent steps.
        simulations["data"] = np.asarray(simulations["data"], dtype=np.int16)
        simulations["params"] = np.asarray(simulations["params"]).reshape((n_samples, 1))
        return simulations


class BenchmarkSimulationConfig(SimulationConfig):
    N_SAMPLES = 100
    N_OBSERVATIONS = 10
    PRIOR = stats.norm(0, 1)

    def simulate(self) -> Dict[str, Any]:
        n_samples = self.args.n_samples or self.N_SAMPLES
        n_observations = self.args.n_observations or self.N_OBSERVATIONS
        random_state = np.random.RandomState(self.args.seed)

        params = self.PRIOR.rvs([n_samples, 1], random_state=random_state)
        data = simulate_benchmark(params, n_observations, random_state=random_state)

        return {
            "params": params,
            "data": data,
        }
