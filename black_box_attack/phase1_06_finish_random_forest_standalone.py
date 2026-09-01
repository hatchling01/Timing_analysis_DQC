#!/usr/bin/env python3
"""
Complete only the Phase 1.6 random-forest analysis from existing samples.

This script is fully standalone. It does NOT import
phase1_06_unknown_placement_robustness.py and therefore does not execute or
reuse the previously broken class_weight code path.

It reads:
    blackbox_window_results/
      phase1_06_unknown_placement_robustness/
        unknown_placement_samples.csv

It writes:
    unknown_placement_random_forest_metrics.csv
    unknown_placement_random_forest_predictions.csv

No placement, allocation, scheduling, or probe simulation is rerun.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_06_unknown_placement_robustness"
)

SAMPLES_FILENAME = "unknown_placement_samples.csv"
METRICS_FILENAME = "unknown_placement_random_forest_metrics.csv"
PREDICTIONS_FILENAME = "unknown_placement_random_forest_predictions.csv"

CLASSIFICATION_FOLDS = 5
GLOBAL_SEED = 20260731
N_ESTIMATORS = 300


def stable_seed(*parts: Any) -> int:
    """Return a deterministic nonnegative seed."""

    text = "|".join(str(part) for part in parts)
    return (
        GLOBAL_SEED
        + zlib.crc32(text.encode("utf-8"))
    ) & 0x7FFFFFFF


def fold_for_group(
    group: str,
    folds: int = CLASSIFICATION_FOLDS,
) -> int:
    """Assign one physical trial to a stable cross-validation fold."""

    return zlib.crc32(
        group.encode("utf-8")
    ) % folds


def normalize_binary_labels(series: pd.Series) -> np.ndarray:
    """Convert bool, numeric, or textual victim-presence labels to 0/1."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).to_numpy(dtype=int)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="raise",
        ).astype(int)

        invalid = ~numeric.isin([0, 1])
        if invalid.any():
            bad_values = sorted(
                numeric[invalid].unique().tolist()
            )
            raise ValueError(
                "victim_presence contains non-binary numeric values: "
                f"{bad_values}"
            )

        return numeric.to_numpy(dtype=int)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": 0,
        "false": 0,
        "no": 0,
        "absent": 0,
        "victim_absent": 0,
        "1": 1,
        "true": 1,
        "yes": 1,
        "present": 1,
        "victim_present": 1,
    }

    unknown = sorted(
        set(normalized.unique())
        - set(mapping)
    )

    if unknown:
        raise ValueError(
            "Could not interpret victim_presence values: "
            f"{unknown}"
        )

    return normalized.map(mapping).to_numpy(dtype=int)


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Compute balanced per-sample weights without sklearn class_weight."""

    unique_classes, class_counts = np.unique(
        labels,
        return_counts=True,
    )

    if len(unique_classes) < 2:
        return np.ones(
            len(labels),
            dtype=float,
        )

    total = float(len(labels))
    number_of_classes = float(
        len(unique_classes)
    )

    weights_by_class = {
        int(class_label): (
            total
            / (
                number_of_classes
                * float(class_count)
            )
        )
        for class_label, class_count
        in zip(unique_classes, class_counts)
    }

    return np.asarray(
        [
            weights_by_class[int(label)]
            for label in labels
        ],
        dtype=float,
    )


def find_samples_file() -> Path:
    """Locate the existing Phase 1.6 sample table."""

    candidates = [
        OUTPUT_DIR / SAMPLES_FILENAME,
        Path(SAMPLES_FILENAME),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(
        f"  - {candidate.resolve()}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not find unknown_placement_samples.csv. "
        "Searched:\n"
        f"{searched}"
    )


def validate_samples(samples: pd.DataFrame) -> list[str]:
    """Validate required metadata and return feature-column names."""

    required_columns = {
        "physical_trial_id",
        "knowledge_level",
        "probe_strategy",
        "victim_presence",
    }

    missing = sorted(
        required_columns
        - set(samples.columns)
    )

    if missing:
        raise KeyError(
            "Sample table is missing required columns: "
            f"{missing}"
        )

    feature_columns = [
        column
        for column in samples.columns
        if (
            column.startswith("f_")
            or column.startswith("k_")
        )
    ]

    if not feature_columns:
        raise RuntimeError(
            "No feature columns beginning with f_ or k_ were found."
        )

    return feature_columns


def run_random_forest(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped five-fold victim-presence classification."""

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required. Install it with: "
            "mamba install -c conda-forge scikit-learn"
        ) from error

    feature_columns = validate_samples(
        samples
    )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    grouped = samples.groupby(
        [
            "knowledge_level",
            "probe_strategy",
        ],
        observed=True,
        sort=True,
    )

    for (
        knowledge_level,
        probe_strategy,
    ), group in grouped:
        data = group.reset_index(
            drop=True
        ).copy()

        # Convert labels once for the complete group, then retain the values
        # by row index in every fold.
        data["_binary_label"] = (
            normalize_binary_labels(
                data["victim_presence"]
            )
        )

        true_all: list[int] = []
        predicted_all: list[int] = []
        group_prediction_rows: list[
            dict[str, Any]
        ] = []

        for fold in range(
            CLASSIFICATION_FOLDS
        ):
            test_mask = data[
                "physical_trial_id"
            ].map(
                lambda value: (
                    fold_for_group(
                        str(value)
                    )
                    == fold
                )
            )

            train = data[
                ~test_mask
            ].copy()

            test = data[
                test_mask
            ].copy()

            if train.empty or test.empty:
                continue

            x_train = (
                train[feature_columns]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            x_test = (
                test[feature_columns]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            y_train = train[
                "_binary_label"
            ].to_numpy(dtype=int)

            y_test = test[
                "_binary_label"
            ].to_numpy(dtype=int)

            unique_train_classes = np.unique(
                y_train
            )

            if len(unique_train_classes) < 2:
                # A degenerate training fold can only support a constant
                # classifier. This avoids an invalid RandomForest fit.
                predicted = np.full(
                    shape=len(y_test),
                    fill_value=int(
                        unique_train_classes[0]
                    ),
                    dtype=int,
                )
            else:
                classifier = (
                    RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        max_features="sqrt",
                        class_weight=None,
                        random_state=stable_seed(
                            knowledge_level,
                            probe_strategy,
                            fold,
                        ),
                        n_jobs=-1,
                    )
                )

                classifier.fit(
                    x_train,
                    y_train,
                    sample_weight=(
                        balanced_sample_weights(
                            y_train
                        )
                    ),
                )

                predicted = classifier.predict(
                    x_test
                ).astype(int)

            true_all.extend(
                y_test.tolist()
            )

            predicted_all.extend(
                predicted.tolist()
            )

            for row_position, (
                row_index,
                row,
            ) in enumerate(
                test.iterrows()
            ):
                true_value = int(
                    y_test[row_position]
                )

                predicted_value = int(
                    predicted[row_position]
                )

                group_prediction_rows.append(
                    {
                        "physical_trial_id": row[
                            "physical_trial_id"
                        ],
                        "knowledge_level": (
                            knowledge_level
                        ),
                        "probe_strategy": (
                            probe_strategy
                        ),
                        "true_label": true_value,
                        "predicted_label": (
                            predicted_value
                        ),
                        "correct": (
                            true_value
                            == predicted_value
                        ),
                        "fold": fold,
                    }
                )

        if not true_all:
            continue

        true_array = np.asarray(
            true_all,
            dtype=int,
        )

        predicted_array = np.asarray(
            predicted_all,
            dtype=int,
        )

        matrix = confusion_matrix(
            true_array,
            predicted_array,
            labels=[0, 1],
        )

        true_negative = int(
            matrix[0, 0]
        )
        false_positive = int(
            matrix[0, 1]
        )
        false_negative = int(
            matrix[1, 0]
        )
        true_positive = int(
            matrix[1, 1]
        )

        specificity = (
            true_negative
            / (
                true_negative
                + false_positive
            )
            if (
                true_negative
                + false_positive
            )
            > 0
            else 0.0
        )

        metric_rows.append(
            {
                "task": "victim_presence",
                "method": "random_forest",
                "knowledge_level": (
                    knowledge_level
                ),
                "probe_strategy": (
                    probe_strategy
                ),
                "sample_count": int(
                    len(true_array)
                ),
                "feature_count": int(
                    len(feature_columns)
                ),
                "fold_count": int(
                    CLASSIFICATION_FOLDS
                ),
                "accuracy": float(
                    accuracy_score(
                        true_array,
                        predicted_array,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        true_array,
                        predicted_array,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        true_array,
                        predicted_array,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "victim_present_precision": float(
                    precision_score(
                        true_array,
                        predicted_array,
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "victim_present_recall": float(
                    recall_score(
                        true_array,
                        predicted_array,
                        pos_label=1,
                        zero_division=0,
                    )
                ),
                "specificity": float(
                    specificity
                ),
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            }
        )

        prediction_rows.extend(
            group_prediction_rows
        )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
    )


def main() -> None:
    samples_path = find_samples_file()

    output_directory = (
        samples_path.parent
    )

    samples = pd.read_csv(
        samples_path
    )

    print(
        "Loaded existing Phase 1.6 samples:"
    )
    print(
        f"  {samples_path}"
    )
    print(
        f"Rows: {len(samples):,}"
    )

    metrics, predictions = (
        run_random_forest(
            samples
        )
    )

    metrics_path = (
        output_directory
        / METRICS_FILENAME
    )

    predictions_path = (
        output_directory
        / PREDICTIONS_FILENAME
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "\nRandom-forest analysis completed."
    )
    print(
        "No physical simulation was rerun."
    )
    print(
        f"Metrics:     {metrics_path}"
    )
    print(
        f"Predictions: {predictions_path}"
    )

    if not metrics.empty:
        print(
            "\nSummary:"
        )
        print(
            metrics[
                [
                    "knowledge_level",
                    "probe_strategy",
                    "sample_count",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
