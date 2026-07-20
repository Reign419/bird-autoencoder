"""Deprecated compatibility entrypoint for Stage 1 experiments.

Use ``python main_experiment.py --config ...`` for new runs.  The legacy
``model_name`` plus ``latent_dims`` config format is translated by
``stage1_experiments.py`` before the shared runner starts.
"""

import argparse
import warnings


def main(config_path="config.json"):
    warnings.warn(
        "main.py is deprecated; use main_experiment.py --config instead.",
        FutureWarning,
        stacklevel=2,
    )
    from main_experiment import main as run_stage1

    return run_stage1(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    main(arguments.config)
