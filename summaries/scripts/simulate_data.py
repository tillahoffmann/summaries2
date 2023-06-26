import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Any, Dict, List

from ..experiments.graph import compress_graph, simulate_graph


class Args:
    n_samples: int | None
    seed: int | None
    config: str
    output: Path


class SimulationConfig:
    def __init__(self, args: Args) -> None:
        self.args = args

    def simulate(self) -> Dict[str, Any]:
        raise NotImplementedError


class GraphSimulationConfig(SimulationConfig):
    N_SAMPLES = 100

    def simulate(self) -> Dict[str, Any]:
        n_samples = self.args.n_samples or self.N_SAMPLES
        random_state = np.random.RandomState(self.args.seed)
        simulations = {}
        for _ in tqdm(range(n_samples)):
            graph, gamma = simulate_graph(748, seed=random_state)
            simulations.setdefault("data", []).append(compress_graph(graph))
            simulations.setdefault("params", []).append(gamma)

        # Reshape to match the shape expected by subsequent steps.
        simulations["data"] = np.asarray(simulations["data"], dtype=np.int16)
        simulations["params"] = np.asarray(simulations["params"]).reshape((n_samples, 1))
        return simulations


SIMULATION_CONFIGS = [
    GraphSimulationConfig,
]
SIMULATION_CONFIGS = {config.__name__: config for config in SIMULATION_CONFIGS}


def __main__(argv: List[str] | None = None) -> None:
    start = datetime.now()
    parser = argparse.ArgumentParser("simulate_data")
    parser.add_argument("--n-samples", type=int, help="override the number of samples generated")
    parser.add_argument("--seed", type=int, help="random number generator seed")
    parser.add_argument("config", help="configuration for simulating data",
                        choices=SIMULATION_CONFIGS)
    parser.add_argument("output", type=Path, help="path to output file")
    args: Args = parser.parse_args(argv)

    config: SimulationConfig = SIMULATION_CONFIGS[args.config](args)
    simulations = config.simulate()

    with args.output.open("wb") as fp:
        pickle.dump({
            "start": start,
            "end": datetime.now(),
            "args": vars(args),
            **simulations
        }, fp)


if __name__ == "__main__":
    __main__()
