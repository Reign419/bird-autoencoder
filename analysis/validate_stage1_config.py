"""Validate and expand a Stage 1 config without importing TensorFlow."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1_experiments import experiment_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    table = pd.DataFrame(experiment_rows(config))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
