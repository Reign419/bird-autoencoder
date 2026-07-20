"""Backward-compatible linear/nonlinear m -> concept leakage probes.

The historical ``concept_probe.csv`` and ``concept_probe_groups.csv`` schemas
remain unchanged.  Detailed real/null tables are written beside them so that
new diagnostics do not break existing aggregation or report references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings


LEGACY_COLUMNS = [
    "attribute_id",
    "group",
    "value",
    "validation_visible",
    "prevalence_baseline_ap",
    "probe_average_precision",
    "probe_ap_lift",
    "probe_balanced_accuracy",
]


def parse_probe_types(value):
    aliases = {"nonlinear": "mlp", "linear": "linear", "mlp": "mlp"}
    result = []
    for token in str(value).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise argparse.ArgumentTypeError(
                "probe types must be a comma-separated subset of linear,mlp"
            )
        resolved = aliases[token]
        if resolved not in result:
            result.append(resolved)
    if not result:
        raise argparse.ArgumentTypeError("at least one probe type is required")
    return tuple(result)


def parse_hidden_layers(value):
    try:
        layers = tuple(int(token.strip()) for token in str(value).split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("MLP hidden layers must be comma-separated integers") from exc
    if not layers or any(width <= 0 for width in layers):
        raise argparse.ArgumentTypeError("MLP hidden layers must be positive")
    return layers


def load_latents(path):
    data = np.load(path)
    required = {"residual", "labels", "weights", "attribute_ids"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
    residual = data["residual"]
    if residual.size == 0:
        raise ValueError(f"{path} contains an empty residual array")
    return {
        "residual": residual.reshape(len(residual), -1),
        "labels": data["labels"],
        "weights": data["weights"],
        "attribute_ids": data["attribute_ids"].astype(int),
    }


def validate_latent_pair(train, evaluation):
    if not np.array_equal(train["attribute_ids"], evaluation["attribute_ids"]):
        raise ValueError("train and evaluation latent files use different attribute_ids")
    concept_dim = len(train["attribute_ids"])
    for name, data in (("train", train), ("evaluation", evaluation)):
        if data["labels"].shape[1] != concept_dim or data["weights"].shape[1] != concept_dim:
            raise ValueError(f"{name} labels/weights do not match attribute_ids")


def balanced_sample_weights(labels, certainty_weights):
    labels = np.asarray(labels, dtype=np.uint8)
    certainty_weights = np.asarray(certainty_weights, dtype=np.float64)
    positive = max(int(labels.sum()), 1)
    negative = max(int((1 - labels).sum()), 1)
    count = len(labels)
    class_weights = np.where(
        labels > 0,
        count / (2.0 * positive),
        count / (2.0 * negative),
    )
    return certainty_weights * class_weights


def build_probe(kind, args, seed):
    if kind == "linear":
        return SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=args.linear_alpha,
            max_iter=args.max_iter,
            tol=1e-4,
            random_state=seed,
        )
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=args.mlp_hidden_layers,
            activation="relu",
            solver="adam",
            alpha=args.mlp_alpha,
            batch_size=args.mlp_batch_size,
            learning_rate_init=args.mlp_learning_rate,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            validation_fraction=args.mlp_validation_fraction,
            n_iter_no_change=args.mlp_patience,
            random_state=seed,
        )
    raise ValueError(f"Unknown probe type {kind!r}")


def fit_scores(kind, train_x, train_y, train_weights, evaluation_x, args, seed):
    probe = build_probe(kind, args, seed)
    sample_weight = train_weights
    if kind == "mlp":
        sample_weight = balanced_sample_weights(train_y, train_weights)
        # MLPClassifier reserves validation_fraction before batching when
        # early_stopping=True, so clip against the effective fit split.
        fit_samples = max(
            1,
            int(np.floor(len(train_y) * (1.0 - args.mlp_validation_fraction))),
        )
        probe.set_params(batch_size=min(args.mlp_batch_size, fit_samples))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        probe.fit(train_x, train_y, sample_weight=sample_weight)
    return probe.predict_proba(evaluation_x)[:, 1]


def score_predictions(labels, scores, sample_weight):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    prediction = (scores >= 0.5).astype(np.uint8)
    return {
        "average_precision": float(
            average_precision_score(labels, scores, sample_weight=sample_weight)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction, sample_weight=sample_weight)
        ),
        "roc_auc": float(roc_auc_score(labels, scores, sample_weight=sample_weight)),
    }


def evaluate_attribute(
    column,
    attribute_id,
    definition,
    kind,
    train,
    evaluation,
    train_x,
    evaluation_x,
    args,
):
    train_mask = train["weights"][:, column] > 0
    evaluation_mask = evaluation["weights"][:, column] > 0
    train_y = train["labels"][train_mask, column].astype(np.uint8)
    evaluation_y = evaluation["labels"][evaluation_mask, column].astype(np.uint8)
    train_weight = train["weights"][train_mask, column]
    evaluation_weight = evaluation["weights"][evaluation_mask, column]
    prevalence = float(evaluation_y.mean()) if len(evaluation_y) else np.nan
    base = {
        "probe_type": kind,
        "evaluation_split": args.evaluation_split,
        "attribute_id": int(attribute_id),
        "group": definition.group,
        "value": definition.value,
        "train_visible": int(train_mask.sum()),
        "evaluation_visible": int(evaluation_mask.sum()),
        "prevalence_baseline_ap": prevalence,
    }
    invalid = np.unique(train_y).size < 2 or np.unique(evaluation_y).size < 2
    if invalid:
        real = {
            **base,
            "probe_average_precision": np.nan,
            "probe_ap_lift": np.nan,
            "probe_balanced_accuracy": np.nan,
            "probe_roc_auc": np.nan,
        }
        return real, []

    real_scores = fit_scores(
        kind,
        train_x[train_mask],
        train_y,
        train_weight,
        evaluation_x[evaluation_mask],
        args,
        args.seed + int(attribute_id),
    )
    metrics = score_predictions(evaluation_y, real_scores, evaluation_weight)
    real = {
        **base,
        "probe_average_precision": metrics["average_precision"],
        "probe_ap_lift": metrics["average_precision"] - prevalence,
        "probe_balanced_accuracy": metrics["balanced_accuracy"],
        "probe_roc_auc": metrics["roc_auc"],
    }

    null_rows = []
    for repeat in range(args.null_repeats):
        null_seed = args.null_seed + repeat * 100_003 + int(attribute_id)
        rng = np.random.default_rng(null_seed)
        # Shuffle the complete supervision record. Keeping certainty weights
        # attached to their labels preserves the observed label/certainty
        # distribution while breaking its relationship to m.
        permutation = rng.permutation(len(train_y))
        shuffled_y = train_y[permutation]
        shuffled_weight = train_weight[permutation]
        null_scores = fit_scores(
            kind,
            train_x[train_mask],
            shuffled_y,
            shuffled_weight,
            evaluation_x[evaluation_mask],
            args,
            null_seed,
        )
        null_metrics = score_predictions(evaluation_y, null_scores, evaluation_weight)
        null_rows.append(
            {
                **base,
                "null_repeat": repeat,
                "null_seed": null_seed,
                "probe_average_precision": null_metrics["average_precision"],
                "probe_ap_lift": null_metrics["average_precision"] - prevalence,
                "probe_balanced_accuracy": null_metrics["balanced_accuracy"],
                "probe_roc_auc": null_metrics["roc_auc"],
            }
        )
    return real, null_rows


def benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def attach_null_statistics(real, null):
    real = real.copy()
    for metric in ("probe_average_precision", "probe_balanced_accuracy", "probe_roc_auc"):
        prefix = metric.removeprefix("probe_")
        if null.empty:
            real[f"null_{prefix}_mean"] = np.nan
            real[f"null_{prefix}_std"] = np.nan
            real[f"{prefix}_lift_over_null"] = np.nan
            real[f"{prefix}_empirical_p"] = np.nan
            real[f"{prefix}_fdr_q"] = np.nan
            continue
        grouped = null.groupby("attribute_id")[metric]
        null_mean = grouped.mean()
        null_std = grouped.std(ddof=1)
        real[f"null_{prefix}_mean"] = real.attribute_id.map(null_mean)
        real[f"null_{prefix}_std"] = real.attribute_id.map(null_std)
        real[f"{prefix}_lift_over_null"] = real[metric] - real[f"null_{prefix}_mean"]
        null_values = {key: value.to_numpy() for key, value in grouped}
        p_values = []
        for row in real.itertuples(index=False):
            values = null_values.get(row.attribute_id, np.asarray([]))
            observed = getattr(row, metric)
            if not len(values) or not np.isfinite(observed):
                p_values.append(np.nan)
            else:
                p_values.append((1.0 + np.sum(values >= observed)) / (len(values) + 1.0))
        real[f"{prefix}_empirical_p"] = p_values
        real[f"{prefix}_fdr_q"] = benjamini_hochberg(p_values)
    return real


def run_probe(kind, train, evaluation, definitions, args):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train["residual"])
    evaluation_x = scaler.transform(evaluation["residual"])
    jobs = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(evaluate_attribute)(
            column,
            attribute_id,
            definitions.loc[attribute_id],
            kind,
            train,
            evaluation,
            train_x,
            evaluation_x,
            args,
        )
        for column, attribute_id in enumerate(train["attribute_ids"])
    )
    real = pd.DataFrame([item[0] for item in jobs])
    null = pd.DataFrame([row for item in jobs for row in item[1]])
    return attach_null_statistics(real, null), null


def legacy_tables(linear_table):
    legacy = linear_table.rename(columns={"evaluation_visible": "validation_visible"})
    legacy = legacy[LEGACY_COLUMNS].copy()
    groups = (
        legacy.groupby("group", sort=False)
        .agg(
            n_attributes=("attribute_id", "size"),
            macro_probe_ap=("probe_average_precision", "mean"),
            macro_probe_ap_lift=("probe_ap_lift", "mean"),
            macro_probe_balanced_accuracy=("probe_balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    return legacy, groups


def output_path(base, suffix):
    return base.with_name(base.stem + suffix + base.suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", required=True)
    parser.add_argument("--validation-latents", required=True)
    parser.add_argument("--test-latents")
    parser.add_argument("--evaluation-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--attribute-definitions", required=True)
    parser.add_argument("--output", default="concept_probe.csv")
    parser.add_argument("--probe-types", type=parse_probe_types, default=("linear",))
    parser.add_argument("--null-repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-seed", type=int, default=1042)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--linear-alpha", type=float, default=0.0001)
    parser.add_argument("--mlp-hidden-layers", type=parse_hidden_layers, default=(128, 64))
    parser.add_argument("--mlp-alpha", type=float, default=0.0001)
    parser.add_argument("--mlp-batch-size", type=int, default=128)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.null_repeats < 0:
        parser.error("--null-repeats must be non-negative")
    if args.evaluation_split == "test" and not args.test_latents:
        parser.error("--test-latents is required when --evaluation-split=test")

    train = load_latents(args.train_latents)
    evaluation_path = (
        args.test_latents if args.evaluation_split == "test" else args.validation_latents
    )
    evaluation = load_latents(evaluation_path)
    validate_latent_pair(train, evaluation)
    definitions = pd.read_csv(args.attribute_definitions).set_index("attribute_id")
    missing = sorted(set(train["attribute_ids"]).difference(definitions.index))
    if missing:
        raise KeyError(f"Missing attribute definitions for IDs: {missing}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tables = {}
    for kind in args.probe_types:
        real, null = run_probe(kind, train, evaluation, definitions, args)
        real.to_csv(output_path(output, f"_{kind}"), index=False)
        if not null.empty:
            null.to_csv(output_path(output, f"_{kind}_null"), index=False)
        tables[kind] = real

    # Preserve historical output references even when only MLP was requested:
    # the linear probe is still produced for concept_probe.csv and aggregation.
    if "linear" not in tables:
        linear, null = run_probe("linear", train, evaluation, definitions, args)
        linear.to_csv(output_path(output, "_linear"), index=False)
        if not null.empty:
            null.to_csv(output_path(output, "_linear_null"), index=False)
        tables["linear"] = linear
    legacy, groups = legacy_tables(tables["linear"])
    legacy.to_csv(output, index=False)
    groups.to_csv(output.with_name(output.stem + "_groups.csv"), index=False)

    comparison = pd.concat(tables.values(), ignore_index=True)
    comparison.to_csv(output_path(output, "_comparison"), index=False)
    summary = {
        "evaluation_split": args.evaluation_split,
        "probe_types": list(tables),
        "null_repeats": args.null_repeats,
        "train_samples": len(train["residual"]),
        "evaluation_samples": len(evaluation["residual"]),
        "concept_dim": len(train["attribute_ids"]),
        "legacy_output": str(output),
        "legacy_group_output": str(output.with_name(output.stem + "_groups.csv")),
    }
    output_path(output, "_summary").with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
