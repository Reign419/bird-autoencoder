"""Safety-checked entry point for Stage 2 factorized experiments.

Use this launcher instead of calling ``main_factorized.py`` directly. Official
CUB test evaluation is disabled unless the config explicitly releases it and
every experiment name is marked confirmatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_official_test_release(config: dict) -> None:
    evaluate = bool(config.get("evaluate_official_test", False))
    release = bool(config.get("official_test_release", False))
    experiments = config.get("experiments", [])
    names = [str(item.get("name", "")) for item in experiments]

    if release and not evaluate:
        raise ValueError(
            "official_test_release=true is inconsistent with "
            "evaluate_official_test=false"
        )
    if not evaluate:
        return
    if not release:
        raise PermissionError(
            "Official test evaluation is locked. Set both "
            "evaluate_official_test=true and official_test_release=true only "
            "after the confirmatory protocol is frozen."
        )
    if not names or any("confirmatory" not in name.lower() for name in names):
        raise PermissionError(
            "Official test release requires every experiment name to contain "
            "'confirmatory'."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_official_test_release(config)

    # Import TensorFlow and the training pipeline only after the irreversible
    # official-test guard has passed.
    from main_factorized import main as run_main

    run_main(args.config)


if __name__ == "__main__":
    main()
