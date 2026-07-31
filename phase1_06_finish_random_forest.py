#!/usr/bin/env python3
"""
Finish only the Phase 1.6 random-forest analysis from existing samples.

This script does not rerun placement, allocation, scheduling, or probe
simulations. It reads unknown_placement_samples.csv produced by the main
Phase 1.6 script and writes the two random-forest result files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import phase1_06_unknown_placement_robustness as phase1_06


def main() -> None:
    output_dir = Path(phase1_06.OUTPUT_DIR)
    samples_path = output_dir / "unknown_placement_samples.csv"

    if not samples_path.exists():
        raise FileNotFoundError(
            "Existing Phase 1.6 samples were not found at: "
            f"{samples_path.resolve()}"
        )

    samples = pd.read_csv(samples_path)

    metrics, predictions = phase1_06.optional_random_forest(samples)

    metrics_path = output_dir / "unknown_placement_random_forest_metrics.csv"
    predictions_path = (
        output_dir / "unknown_placement_random_forest_predictions.csv"
    )

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("Random-forest analysis completed without rerunning simulation.")
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
