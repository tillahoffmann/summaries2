import argparse
from datetime import datetime
from pathlib import Path
import pickle
from typing import List

from .configs import SimulationArgs, SimulationConfig, TreeSimulationConfig


SIMULATION_CONFIGS = [
    TreeSimulationConfig,
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
    args: SimulationArgs = parser.parse_args(argv)

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
