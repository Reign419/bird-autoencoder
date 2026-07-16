"""Create a compact Markdown table from an aggregated mean/std CSV."""

import argparse
from pathlib import Path

import pandas as pd


def format_mean_std(frame, metric, digits=4):
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in frame or std_column not in frame:
        return None
    return frame.apply(
        lambda row: f"{row[mean_column]:.{digits}f} ± {row[std_column]:.{digits}f}",
        axis=1,
    )


def to_markdown(frame):
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.fillna("").iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows]) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mean_std_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.mean_std_csv)
    table = pd.DataFrame()
    for column in ("experiment_name", "model", "variant", "effective_latent_size"):
        if column in frame:
            table[column] = frame[column]
    for metric, digits in (("best_val_ssim", 4), ("best_val_psnr", 3), ("best_val_mse", 5)):
        formatted = format_mean_std(frame, metric, digits)
        if formatted is not None:
            table[metric] = formatted

    markdown = to_markdown(table)
    output = Path(args.output) if args.output else Path(args.mean_std_csv).with_suffix(".md")
    output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
