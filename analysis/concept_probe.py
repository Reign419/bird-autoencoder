"""Linear m -> concept leakage probe using saved train/validation latents."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", required=True)
    parser.add_argument("--validation-latents", required=True)
    parser.add_argument("--attribute-definitions", required=True)
    parser.add_argument("--output", default="concept_probe.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    train = np.load(args.train_latents)
    validation = np.load(args.validation_latents)
    train_x = train["residual"].reshape(len(train["residual"]), -1)
    validation_x = validation["residual"].reshape(len(validation["residual"]), -1)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    validation_x = scaler.transform(validation_x)
    definitions = pd.read_csv(args.attribute_definitions).set_index("attribute_id")
    attribute_ids = train["attribute_ids"].astype(int)
    rows = []

    for column, attribute_id in enumerate(attribute_ids):
        train_mask = train["weights"][:, column] > 0
        validation_mask = validation["weights"][:, column] > 0
        train_y = train["labels"][train_mask, column]
        validation_y = validation["labels"][validation_mask, column]
        if np.unique(train_y).size < 2 or np.unique(validation_y).size < 2:
            ap = balanced = np.nan
        else:
            probe = SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=args.max_iter,
                tol=1e-4,
                random_state=args.seed,
            )
            probe.fit(
                train_x[train_mask],
                train_y,
                sample_weight=train["weights"][train_mask, column],
            )
            score = probe.predict_proba(validation_x[validation_mask])[:, 1]
            prediction = (score >= 0.5).astype(np.uint8)
            ap = average_precision_score(validation_y, score)
            balanced = balanced_accuracy_score(validation_y, prediction)
        definition = definitions.loc[attribute_id]
        prevalence = float(validation_y.mean()) if len(validation_y) else np.nan
        rows.append(
            {
                "attribute_id": attribute_id,
                "group": definition.group,
                "value": definition.value,
                "validation_visible": int(validation_mask.sum()),
                "prevalence_baseline_ap": prevalence,
                "probe_average_precision": ap,
                "probe_ap_lift": ap - prevalence if np.isfinite(ap) else np.nan,
                "probe_balanced_accuracy": balanced,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    group_table = (
        table.groupby("group", sort=False)
        .agg(
            n_attributes=("attribute_id", "size"),
            macro_probe_ap=("probe_average_precision", "mean"),
            macro_probe_ap_lift=("probe_ap_lift", "mean"),
            macro_probe_balanced_accuracy=("probe_balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    group_table.to_csv(output.with_name(output.stem + "_groups.csv"), index=False)
    print(group_table.to_string(index=False))


if __name__ == "__main__":
    main()
