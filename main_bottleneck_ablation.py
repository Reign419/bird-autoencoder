"""Deprecated compatibility entrypoint for Stage 1 topology ablations."""

import argparse
import warnings


def main(config_path="config_bottleneck_ablation.json"):
    warnings.warn(
        "main_bottleneck_ablation.py is deprecated; use main_experiment.py "
        "--config configs/topology_ablation.json instead.",
        FutureWarning,
        stacklevel=2,
    )
    from main_experiment import main as run_stage1

    return run_stage1(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_bottleneck_ablation.json")
    arguments = parser.parse_args()
    main(arguments.config)
