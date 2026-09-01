#!/usr/bin/env python3
"""
phase1_03_postprocess.py

Correct and extend the Phase 1.3 communication-qubit-allocation analysis
without rerunning the simulator.

This script reads the existing Phase 1.3 CSV files and produces:

1. Delay-only, signed-timing, failure-only, and combined leakage metrics.
2. Victim slowdown computed only for successful victim executions.
3. Communication-qubit service, reset, reservation, and allocation-hold
   utilization reported as separate components.
4. Aggregate resource demand explicitly labeled as non-physical and allowed
   to exceed one when component intervals overlap.
5. Corrected EPR unused/wastage accounting.
6. A genuine non-ML paired-change detector evaluated against cross-trial
   attacker-only negative controls.
7. A genuine victim-presence random-forest classifier using only raw
   attacker-visible traces, evaluated on held-out allocation configurations.

No simulator is invoked.

Expected input files
--------------------
communication_qubit_attacker_comparison.csv
communication_qubit_attacker_request_log.csv
communication_qubit_trial_summary.csv

Default output directory
------------------------
blackbox_window_results/
    phase1_03_communication_qubit_allocation/
        postprocessed/

Run
---
python phase1_03_postprocess.py
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Integrated settings
# ============================================================

PHASE_DIR = (
    Path("blackbox_window_results")
    / "phase1_03_communication_qubit_allocation"
)

OUTPUT_DIR = PHASE_DIR / "postprocessed"

# Deterministic simulator output can use zero. Increase this value when
# post-processing measurements with a known timestamp resolution/noise floor.
TIMING_CHANGE_THRESHOLD_NS = 0.0

RUN_RANDOM_FOREST_CLASSIFICATION = True
RANDOM_FOREST_TREES = 60
CLASSIFICATION_FOLDS = 3
RANDOM_SEED = 2026

SAVE_CORRECTED_PROBE_TABLE = False
SAVE_TRACE_FEATURE_TABLE = True

INPUT_FILENAMES = {
    "comparison": "communication_qubit_attacker_comparison.csv",
    "request_log": "communication_qubit_attacker_request_log.csv",
    "trial_summary": "communication_qubit_trial_summary.csv",
}


# ============================================================
# Input discovery and validation
# ============================================================


def find_input_file(filename: str) -> Path:
    """Find an input file without requiring command-line arguments."""

    direct_candidates = [
        PHASE_DIR / filename,
        Path.cwd() / filename,
        Path("/mnt/data") / filename,
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate.resolve()

    search_roots = [
        PHASE_DIR,
        Path.cwd(),
        Path("/mnt/data"),
    ]

    matches: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob(filename):
            if OUTPUT_DIR in path.parents:
                continue
            matches.append(path.resolve())

    unique_matches = sorted(set(matches), key=lambda path: str(path))

    if not unique_matches:
        raise FileNotFoundError(
            f"Could not find required Phase 1.3 file: {filename}"
        )

    if len(unique_matches) > 1:
        print(
            f"Warning: multiple copies of {filename} were found; "
            f"using {unique_matches[0]}"
        )

    return unique_matches[0]


def require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    table_name: str,
) -> None:
    """Raise a clear error when an expected schema field is absent."""

    missing = [column for column in required if column not in dataframe.columns]

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def boolean_series(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV values safely."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def safe_divide(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
) -> Any:
    """Divide while returning NaN where the denominator is zero."""

    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)

    result = np.full(
        np.broadcast(numerator_array, denominator_array).shape,
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=denominator_array != 0,
    )

    if result.ndim == 0:
        return float(result)

    return result


# ============================================================
# Probe-level correction
# ============================================================


def correct_probe_comparisons(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create mutually separated probe-level leakage outcomes."""

    required_columns = [
        "victim_qasm",
        "subexperiment",
        "configuration_id",
        "trial_id",
        "logical_event_id",
        "baseline_completed",
        "combined_completed",
        "combined_failed",
        "baseline_request_failed",
        "combined_request_failed",
        "baseline_turnaround_time_ns",
        "combined_turnaround_time_ns",
        "baseline_communication_qubit_queue_delay_ns",
        "combined_communication_qubit_queue_delay_ns",
        "baseline_request_redirected",
        "combined_request_redirected",
        "baseline_source_communication_qubit",
        "combined_source_communication_qubit",
        "baseline_target_communication_qubit",
        "combined_target_communication_qubit",
    ]

    require_columns(
        comparison,
        required_columns,
        "communication_qubit_attacker_comparison.csv",
    )

    corrected = comparison.copy()

    boolean_columns = [
        "baseline_completed",
        "combined_completed",
        "combined_failed",
        "baseline_request_failed",
        "combined_request_failed",
        "baseline_request_redirected",
        "combined_request_redirected",
        "baseline_used_prefetched_epr",
        "combined_used_prefetched_epr",
        "allocation_changed",
    ]

    for column in boolean_columns:
        if column in corrected.columns:
            corrected[column] = boolean_series(corrected[column])

    corrected["victim_tag"] = (
        corrected["victim_qasm"]
        .astype(str)
        .map(lambda value: Path(value).stem)
    )

    corrected["baseline_failed_corrected"] = (
        corrected["baseline_request_failed"]
        | ~corrected["baseline_completed"]
    )

    corrected["combined_failed_corrected"] = (
        corrected["combined_request_failed"]
        | ~corrected["combined_completed"]
        | corrected["combined_failed"]
    )

    corrected["both_completed"] = (
        corrected["baseline_completed"]
        & corrected["combined_completed"]
        & ~corrected["baseline_failed_corrected"]
        & ~corrected["combined_failed_corrected"]
    )

    corrected["new_failure_due_to_victim"] = (
        ~corrected["baseline_failed_corrected"]
        & corrected["combined_failed_corrected"]
    )

    corrected["failure_recovered_with_victim"] = (
        corrected["baseline_failed_corrected"]
        & ~corrected["combined_failed_corrected"]
    )

    corrected["both_failed"] = (
        corrected["baseline_failed_corrected"]
        & corrected["combined_failed_corrected"]
    )

    signed_latency_change = (
        pd.to_numeric(
            corrected["combined_turnaround_time_ns"],
            errors="coerce",
        )
        - pd.to_numeric(
            corrected["baseline_turnaround_time_ns"],
            errors="coerce",
        )
    )

    corrected["signed_turnaround_change_ns"] = signed_latency_change.where(
        corrected["both_completed"]
    )

    corrected["absolute_turnaround_change_ns"] = (
        corrected["signed_turnaround_change_ns"].abs()
    )

    signed_queue_change = (
        pd.to_numeric(
            corrected[
                "combined_communication_qubit_queue_delay_ns"
            ],
            errors="coerce",
        )
        - pd.to_numeric(
            corrected[
                "baseline_communication_qubit_queue_delay_ns"
            ],
            errors="coerce",
        )
    )

    corrected["signed_cq_queue_change_ns"] = signed_queue_change.where(
        corrected["both_completed"]
    )

    corrected["absolute_cq_queue_change_ns"] = (
        corrected["signed_cq_queue_change_ns"].abs()
    )

    threshold = float(TIMING_CHANGE_THRESHOLD_NS)

    corrected["positive_delay_detected"] = (
        corrected["both_completed"]
        & (
            corrected["signed_turnaround_change_ns"]
            > threshold
        )
    )

    corrected["negative_speedup_detected"] = (
        corrected["both_completed"]
        & (
            corrected["signed_turnaround_change_ns"]
            < -threshold
        )
    )

    corrected["signed_timing_change_detected"] = (
        corrected["positive_delay_detected"]
        | corrected["negative_speedup_detected"]
    )

    corrected["failure_state_change_detected"] = (
        corrected["new_failure_due_to_victim"]
        | corrected["failure_recovered_with_victim"]
    )

    corrected["redirect_state_change_detected"] = (
        corrected["baseline_request_redirected"]
        != corrected["combined_request_redirected"]
    )

    source_changed = (
        corrected["baseline_source_communication_qubit"]
        .fillna("<none>")
        .astype(str)
        != corrected["combined_source_communication_qubit"]
        .fillna("<none>")
        .astype(str)
    )

    target_changed = (
        corrected["baseline_target_communication_qubit"]
        .fillna("<none>")
        .astype(str)
        != corrected["combined_target_communication_qubit"]
        .fillna("<none>")
        .astype(str)
    )

    corrected["communication_qubit_assignment_changed"] = (
        source_changed | target_changed
    )

    corrected["delay_or_failure_detected"] = (
        corrected["positive_delay_detected"]
        | corrected["new_failure_due_to_victim"]
    )

    corrected["signed_timing_or_failure_detected"] = (
        corrected["signed_timing_change_detected"]
        | corrected["failure_state_change_detected"]
    )

    corrected["full_attacker_observable_change"] = (
        corrected["signed_timing_or_failure_detected"]
        | corrected["redirect_state_change_detected"]
        | corrected["communication_qubit_assignment_changed"]
    )

    return corrected


# ============================================================
# Corrected trial metrics
# ============================================================


def aggregate_corrected_trials(
    corrected_probes: pd.DataFrame,
    original_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate corrected probe outcomes and merge simulator metadata."""

    group_columns = [
        "victim_qasm",
        "victim_tag",
        "subexperiment",
        "configuration_id",
        "trial_id",
        "allocation_policy",
        "communication_qubits_per_module",
        "tenants_sharing_module",
        "reservation_duration_ns",
        "reset_time_ns",
        "epr_prefetch_enabled",
        "allocation_coordination",
        "allocation_granularity",
    ]

    available_group_columns = [
        column
        for column in group_columns
        if column in corrected_probes.columns
    ]

    grouped = corrected_probes.groupby(
        available_group_columns,
        dropna=False,
        observed=True,
    )

    corrected_trials = grouped.agg(
        attacker_probe_count_corrected=(
            "logical_event_id",
            "count",
        ),
        comparable_completed_probe_count=(
            "both_completed",
            "sum",
        ),
        positive_delay_probe_count=(
            "positive_delay_detected",
            "sum",
        ),
        negative_speedup_probe_count=(
            "negative_speedup_detected",
            "sum",
        ),
        signed_timing_change_probe_count=(
            "signed_timing_change_detected",
            "sum",
        ),
        new_failure_probe_count=(
            "new_failure_due_to_victim",
            "sum",
        ),
        recovered_failure_probe_count=(
            "failure_recovered_with_victim",
            "sum",
        ),
        failure_state_change_probe_count=(
            "failure_state_change_detected",
            "sum",
        ),
        redirect_change_probe_count=(
            "redirect_state_change_detected",
            "sum",
        ),
        allocation_change_probe_count=(
            "communication_qubit_assignment_changed",
            "sum",
        ),
        delay_or_failure_probe_count=(
            "delay_or_failure_detected",
            "sum",
        ),
        signed_timing_or_failure_probe_count=(
            "signed_timing_or_failure_detected",
            "sum",
        ),
        full_observable_change_probe_count=(
            "full_attacker_observable_change",
            "sum",
        ),
        mean_signed_turnaround_change_ns=(
            "signed_turnaround_change_ns",
            "mean",
        ),
        median_signed_turnaround_change_ns=(
            "signed_turnaround_change_ns",
            "median",
        ),
        mean_absolute_turnaround_change_ns=(
            "absolute_turnaround_change_ns",
            "mean",
        ),
        max_absolute_turnaround_change_ns=(
            "absolute_turnaround_change_ns",
            "max",
        ),
        total_absolute_turnaround_change_ns=(
            "absolute_turnaround_change_ns",
            "sum",
        ),
        mean_signed_cq_queue_change_ns=(
            "signed_cq_queue_change_ns",
            "mean",
        ),
        mean_absolute_cq_queue_change_ns=(
            "absolute_cq_queue_change_ns",
            "mean",
        ),
    ).reset_index()

    probe_denominator = corrected_trials[
        "attacker_probe_count_corrected"
    ].astype(float)

    comparable_denominator = corrected_trials[
        "comparable_completed_probe_count"
    ].astype(float)

    probability_columns = {
        "positive_delay_detection_probability": (
            "positive_delay_probe_count",
            probe_denominator,
        ),
        "negative_speedup_detection_probability": (
            "negative_speedup_probe_count",
            probe_denominator,
        ),
        "signed_timing_change_probability": (
            "signed_timing_change_probe_count",
            probe_denominator,
        ),
        "failure_only_detection_probability": (
            "new_failure_probe_count",
            probe_denominator,
        ),
        "failure_state_change_probability": (
            "failure_state_change_probe_count",
            probe_denominator,
        ),
        "delay_or_failure_detection_probability": (
            "delay_or_failure_probe_count",
            probe_denominator,
        ),
        "signed_timing_or_failure_detection_probability": (
            "signed_timing_or_failure_probe_count",
            probe_denominator,
        ),
        "full_observable_change_probability": (
            "full_observable_change_probe_count",
            probe_denominator,
        ),
        "comparable_probe_fraction": (
            "comparable_completed_probe_count",
            probe_denominator,
        ),
        "positive_delay_probability_among_comparable": (
            "positive_delay_probe_count",
            comparable_denominator,
        ),
        "signed_timing_change_probability_among_comparable": (
            "signed_timing_change_probe_count",
            comparable_denominator,
        ),
    }

    for output_column, (
        numerator_column,
        denominator,
    ) in probability_columns.items():
        corrected_trials[output_column] = safe_divide(
            corrected_trials[numerator_column],
            denominator,
        )

    original = original_trials.copy()
    original["victim_tag"] = (
        original["victim_qasm"]
        .astype(str)
        .map(lambda value: Path(value).stem)
    )

    merge_keys = [
        "victim_qasm",
        "victim_tag",
        "subexperiment",
        "configuration_id",
        "trial_id",
    ]

    metadata_columns = [
        column
        for column in original.columns
        if column not in corrected_trials.columns
        or column in merge_keys
    ]

    original_metadata = original[metadata_columns].copy()

    merged = corrected_trials.merge(
        original_metadata,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    boolean_original_columns = [
        "attacker_baseline_rejected",
        "attacker_combined_rejected",
        "victim_only_rejected",
        "victim_combined_rejected",
        "victim_failed_in_combined",
        "epr_prefetch_enabled",
    ]

    for column in boolean_original_columns:
        if column in merged.columns:
            merged[column] = boolean_series(merged[column])

    merged["successful_victim_execution"] = (
        ~merged["victim_only_rejected"]
        & ~merged["victim_combined_rejected"]
        & ~merged["victim_failed_in_combined"]
        & (
            pd.to_numeric(
                merged["victim_combined_failed_request_count"],
                errors="coerce",
            ).fillna(0)
            == 0
        )
        & (
            pd.to_numeric(
                merged["victim_only_completion_time_ns"],
                errors="coerce",
            )
            > 0
        )
    )

    corrected_slowdown = safe_divide(
        pd.to_numeric(
            merged["combined_victim_completion_time_ns"],
            errors="coerce",
        ),
        pd.to_numeric(
            merged["victim_only_completion_time_ns"],
            errors="coerce",
        ),
    )

    merged["successful_victim_slowdown_ratio"] = np.where(
        merged["successful_victim_execution"],
        corrected_slowdown,
        np.nan,
    )

    merged["invalid_failed_execution_slowdown"] = np.where(
        merged["successful_victim_execution"],
        np.nan,
        pd.to_numeric(
            merged["victim_slowdown_ratio"],
            errors="coerce",
        ),
    )

    merged["failed_execution_apparent_speedup"] = (
        ~merged["successful_victim_execution"]
        & (
            pd.to_numeric(
                merged["victim_slowdown_ratio"],
                errors="coerce",
            )
            < 1.0
        )
    )

    merged["victim_failure_indicator"] = (
        ~merged["successful_victim_execution"]
    ).astype(int)

    # Preserve physically interpretable utilization components separately.
    merged["cq_service_utilization"] = pd.to_numeric(
        merged["communication_qubit_utilization"],
        errors="coerce",
    )

    merged["cq_reset_occupancy_fraction"] = pd.to_numeric(
        merged["reset_utilization"],
        errors="coerce",
    )

    merged["cq_reservation_occupancy_fraction"] = pd.to_numeric(
        merged["reservation_utilization"],
        errors="coerce",
    )

    merged["cq_allocation_hold_fraction"] = pd.to_numeric(
        merged["allocation_hold_utilization"],
        errors="coerce",
    )

    merged["aggregate_resource_demand_nonphysical"] = (
        merged["cq_service_utilization"].fillna(0.0)
        + merged["cq_reset_occupancy_fraction"].fillna(0.0)
        + merged["cq_reservation_occupancy_fraction"].fillna(0.0)
        + merged["cq_allocation_hold_fraction"].fillna(0.0)
    )

    merged["aggregate_resource_demand_exceeds_one"] = (
        merged["aggregate_resource_demand_nonphysical"]
        > 1.0
    )

    prefetched = pd.to_numeric(
        merged["epr_prefetched_pair_count"],
        errors="coerce",
    ).fillna(0.0)

    used = pd.to_numeric(
        merged["epr_used_pair_count"],
        errors="coerce",
    ).fillna(0.0)

    wasted = pd.to_numeric(
        merged["epr_wasted_pair_count"],
        errors="coerce",
    ).fillna(0.0)

    stranded = pd.to_numeric(
        merged["epr_stranded_pair_count"],
        errors="coerce",
    ).fillna(0.0)

    merged["epr_unused_pair_count_corrected"] = wasted + stranded

    merged["epr_unused_fraction_corrected"] = safe_divide(
        merged["epr_unused_pair_count_corrected"],
        prefetched,
    )

    merged["epr_unaccounted_pair_count"] = (
        prefetched
        - used
        - wasted
        - stranded
    )

    return merged


# ============================================================
# Aggregate summaries
# ============================================================


def summarize_trials(
    trials: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Produce corrected aggregate statistics for selected dimensions."""

    available_groups = [
        column
        for column in group_columns
        if column in trials.columns
    ]

    grouped = trials.groupby(
        available_groups,
        dropna=False,
        observed=True,
    )

    summary = grouped.agg(
        trial_count=(
            "trial_id",
            "count",
        ),
        successful_victim_trial_count=(
            "successful_victim_execution",
            "sum",
        ),
        victim_failure_rate=(
            "victim_failure_indicator",
            "mean",
        ),
        mean_positive_delay_detection_probability=(
            "positive_delay_detection_probability",
            "mean",
        ),
        mean_negative_speedup_detection_probability=(
            "negative_speedup_detection_probability",
            "mean",
        ),
        mean_signed_timing_change_probability=(
            "signed_timing_change_probability",
            "mean",
        ),
        mean_failure_only_detection_probability=(
            "failure_only_detection_probability",
            "mean",
        ),
        mean_delay_or_failure_detection_probability=(
            "delay_or_failure_detection_probability",
            "mean",
        ),
        mean_signed_timing_or_failure_detection_probability=(
            "signed_timing_or_failure_detection_probability",
            "mean",
        ),
        mean_full_observable_change_probability=(
            "full_observable_change_probability",
            "mean",
        ),
        mean_signed_turnaround_change_ns=(
            "mean_signed_turnaround_change_ns",
            "mean",
        ),
        mean_absolute_turnaround_change_ns=(
            "mean_absolute_turnaround_change_ns",
            "mean",
        ),
        mean_total_absolute_turnaround_change_ns=(
            "total_absolute_turnaround_change_ns",
            "mean",
        ),
        mean_signed_cq_queue_change_ns=(
            "mean_signed_cq_queue_change_ns",
            "mean",
        ),
        mean_absolute_cq_queue_change_ns=(
            "mean_absolute_cq_queue_change_ns",
            "mean",
        ),
        mean_successful_victim_slowdown_ratio=(
            "successful_victim_slowdown_ratio",
            "mean",
        ),
        max_successful_victim_slowdown_ratio=(
            "successful_victim_slowdown_ratio",
            "max",
        ),
        mean_request_failure_rate=(
            "combined_request_failure_rate",
            "mean",
        ),
        mean_cq_service_utilization=(
            "cq_service_utilization",
            "mean",
        ),
        mean_cq_reset_occupancy_fraction=(
            "cq_reset_occupancy_fraction",
            "mean",
        ),
        mean_cq_reservation_occupancy_fraction=(
            "cq_reservation_occupancy_fraction",
            "mean",
        ),
        mean_cq_allocation_hold_fraction=(
            "cq_allocation_hold_fraction",
            "mean",
        ),
        mean_aggregate_resource_demand_nonphysical=(
            "aggregate_resource_demand_nonphysical",
            "mean",
        ),
        fraction_aggregate_resource_demand_exceeds_one=(
            "aggregate_resource_demand_exceeds_one",
            "mean",
        ),
        mean_epr_unused_fraction=(
            "epr_unused_fraction_corrected",
            "mean",
        ),
        total_epr_prefetched_pairs=(
            "epr_prefetched_pair_count",
            "sum",
        ),
        total_epr_used_pairs=(
            "epr_used_pair_count",
            "sum",
        ),
        total_epr_unused_pairs=(
            "epr_unused_pair_count_corrected",
            "sum",
        ),
        mean_grant_fairness=(
            "jain_grant_fairness",
            "mean",
        ),
        mean_inverse_wait_fairness=(
            "jain_inverse_wait_fairness",
            "mean",
        ),
    ).reset_index()

    summary["successful_victim_trial_fraction"] = safe_divide(
        summary["successful_victim_trial_count"],
        summary["trial_count"],
    )

    summary["aggregate_epr_unused_fraction"] = safe_divide(
        summary["total_epr_unused_pairs"],
        summary["total_epr_prefetched_pairs"],
    )

    return summary


# ============================================================
# Genuine non-ML detector evaluation
# ============================================================


def create_nonml_rule_dataset(
    corrected_probes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate a simple paired-change detector.

    Positive examples compare combined execution against its own attacker-only
    baseline. Negative controls compare attacker-only traces from different
    trials of the same workload and allocation configuration.
    """

    key_without_trial = [
        "victim_qasm",
        "victim_tag",
        "subexperiment",
        "configuration_id",
        "allocation_policy",
        "communication_qubits_per_module",
        "tenants_sharing_module",
        "reservation_duration_ns",
        "reset_time_ns",
        "epr_prefetch_enabled",
        "allocation_coordination",
        "allocation_granularity",
    ]

    available_keys = [
        column
        for column in key_without_trial
        if column in corrected_probes.columns
    ]

    sample_rows: list[dict[str, Any]] = []

    # Positive combined-versus-baseline examples.
    positive_group_columns = available_keys + ["trial_id"]

    for group_values, group in corrected_probes.groupby(
        positive_group_columns,
        dropna=False,
        observed=True,
    ):
        metadata = dict(zip(positive_group_columns, group_values))

        signed_change = pd.to_numeric(
            group["signed_turnaround_change_ns"],
            errors="coerce",
        )

        detected = bool(
            group["signed_timing_or_failure_detected"].any()
            or group["redirect_state_change_detected"].any()
        )

        sample_rows.append(
            {
                **metadata,
                "sample_type": "combined_vs_baseline",
                "true_victim_present": 1,
                "rule_detected_victim": int(detected),
                "changed_probe_count": int(
                    group[
                        "signed_timing_or_failure_detected"
                    ].sum()
                ),
                "failure_transition_count": int(
                    group["failure_state_change_detected"].sum()
                ),
                "mean_absolute_trace_change_ns": float(
                    signed_change.abs().mean()
                ),
                "max_absolute_trace_change_ns": float(
                    signed_change.abs().max()
                ),
            }
        )

    # Cross-trial attacker-only negative controls.
    baseline_columns = [
        *available_keys,
        "trial_id",
        "logical_event_id",
        "baseline_completed",
        "baseline_failed_corrected",
        "baseline_request_redirected",
        "baseline_turnaround_time_ns",
    ]

    baselines = corrected_probes[baseline_columns].copy()

    for group_values, group in baselines.groupby(
        available_keys,
        dropna=False,
        observed=True,
    ):
        metadata = dict(zip(available_keys, group_values))
        trial_ids = sorted(group["trial_id"].unique())

        if len(trial_ids) < 2:
            continue

        by_trial = {
            trial_id: (
                group[group["trial_id"] == trial_id]
                .sort_values("logical_event_id")
                .reset_index(drop=True)
            )
            for trial_id in trial_ids
        }

        for index, trial_id in enumerate(trial_ids):
            reference_trial_id = trial_ids[(index + 1) % len(trial_ids)]

            first = by_trial[trial_id][
                [
                    "logical_event_id",
                    "baseline_completed",
                    "baseline_failed_corrected",
                    "baseline_request_redirected",
                    "baseline_turnaround_time_ns",
                ]
            ].rename(
                columns={
                    "baseline_completed": "first_completed",
                    "baseline_failed_corrected": "first_failed",
                    "baseline_request_redirected": "first_redirected",
                    "baseline_turnaround_time_ns": "first_turnaround",
                }
            )

            second = by_trial[reference_trial_id][
                [
                    "logical_event_id",
                    "baseline_completed",
                    "baseline_failed_corrected",
                    "baseline_request_redirected",
                    "baseline_turnaround_time_ns",
                ]
            ].rename(
                columns={
                    "baseline_completed": "second_completed",
                    "baseline_failed_corrected": "second_failed",
                    "baseline_request_redirected": "second_redirected",
                    "baseline_turnaround_time_ns": "second_turnaround",
                }
            )

            paired = first.merge(
                second,
                on="logical_event_id",
                how="outer",
                validate="one_to_one",
            )

            for column in [
                "first_completed",
                "first_failed",
                "first_redirected",
                "second_completed",
                "second_failed",
                "second_redirected",
            ]:
                paired[column] = boolean_series(paired[column])

            comparable = (
                paired["first_completed"]
                & paired["second_completed"]
                & ~paired["first_failed"]
                & ~paired["second_failed"]
            )

            signed_change = (
                pd.to_numeric(
                    paired["second_turnaround"],
                    errors="coerce",
                )
                - pd.to_numeric(
                    paired["first_turnaround"],
                    errors="coerce",
                )
            ).where(comparable)

            timing_change = (
                signed_change.abs()
                > TIMING_CHANGE_THRESHOLD_NS
            ).fillna(False)

            failure_change = (
                paired["first_failed"]
                != paired["second_failed"]
            )

            redirect_change = (
                paired["first_redirected"]
                != paired["second_redirected"]
            )

            detected = bool(
                timing_change.any()
                or failure_change.any()
                or redirect_change.any()
            )

            sample_rows.append(
                {
                    **metadata,
                    "trial_id": trial_id,
                    "reference_trial_id": reference_trial_id,
                    "sample_type": "baseline_vs_baseline",
                    "true_victim_present": 0,
                    "rule_detected_victim": int(detected),
                    "changed_probe_count": int(
                        timing_change.sum()
                        + failure_change.sum()
                    ),
                    "failure_transition_count": int(
                        failure_change.sum()
                    ),
                    "mean_absolute_trace_change_ns": float(
                        signed_change.abs().mean()
                    ),
                    "max_absolute_trace_change_ns": float(
                        signed_change.abs().max()
                    ),
                }
            )

    return pd.DataFrame(sample_rows)


def binary_classification_metrics(
    predictions: pd.DataFrame,
    group_columns: list[str],
    prediction_column: str,
) -> pd.DataFrame:
    """Calculate real binary classification metrics."""

    available_groups = [
        column
        for column in group_columns
        if column in predictions.columns
    ]

    rows: list[dict[str, Any]] = []

    if available_groups:
        iterator = predictions.groupby(
            available_groups,
            dropna=False,
            observed=True,
        )
    else:
        iterator = [((), predictions)]

    for group_values, group in iterator:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        metadata = dict(zip(available_groups, group_values))

        truth = group["true_victim_present"].astype(int).to_numpy()
        prediction = group[prediction_column].astype(int).to_numpy()

        tp = int(np.sum((truth == 1) & (prediction == 1)))
        tn = int(np.sum((truth == 0) & (prediction == 0)))
        fp = int(np.sum((truth == 0) & (prediction == 1)))
        fn = int(np.sum((truth == 1) & (prediction == 0)))

        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        accuracy = (tp + tn) / len(group) if len(group) else np.nan

        balanced_accuracy = (
            np.nanmean([sensitivity, specificity])
            if not (
                np.isnan(sensitivity)
                and np.isnan(specificity)
            )
            else np.nan
        )

        f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if (
                not np.isnan(precision)
                and not np.isnan(sensitivity)
                and precision + sensitivity > 0
            )
            else np.nan
        )

        rows.append(
            {
                **metadata,
                "sample_count": int(len(group)),
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "sensitivity_recall": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ML victim-presence classification
# ============================================================


def build_trace_feature_table(
    request_log: pd.DataFrame,
) -> pd.DataFrame:
    """Convert each attacker execution trace into one feature vector."""

    required_columns = [
        "victim_qasm",
        "subexperiment",
        "configuration_id",
        "trial_id",
        "execution_mode",
        "logical_event_id",
        "turnaround_time_ns",
        "communication_qubit_queue_delay_ns",
        "waiting_time_ns",
        "allocation_conflict_checks",
        "source_reuse_delay_ns",
        "target_reuse_delay_ns",
        "request_failed",
        "request_redirected",
        "used_prefetched_epr",
    ]

    require_columns(
        request_log,
        required_columns,
        "communication_qubit_attacker_request_log.csv",
    )

    log = request_log.copy()

    for column in [
        "request_failed",
        "request_redirected",
        "used_prefetched_epr",
    ]:
        log[column] = boolean_series(log[column])

    log["victim_tag"] = (
        log["victim_qasm"]
        .astype(str)
        .map(lambda value: Path(value).stem)
    )

    group_columns = [
        "victim_qasm",
        "victim_tag",
        "subexperiment",
        "configuration_id",
        "trial_id",
        "execution_mode",
        "allocation_policy",
        "communication_qubits_per_module",
        "tenants_sharing_module",
        "reservation_duration_ns",
        "reset_time_ns",
        "epr_prefetch_enabled",
        "allocation_coordination",
        "allocation_granularity",
    ]

    group_columns = [
        column
        for column in group_columns
        if column in log.columns
    ]

    grouped = log.groupby(
        group_columns,
        dropna=False,
        observed=True,
    )

    aggregate = grouped.agg(
        bb_request_count=("logical_event_id", "count"),
        bb_failure_fraction=("request_failed", "mean"),
        bb_redirect_fraction=("request_redirected", "mean"),
        bb_prefetched_epr_use_fraction=("used_prefetched_epr", "mean"),
        bb_turnaround_mean_ns=("turnaround_time_ns", "mean"),
        bb_turnaround_std_ns=("turnaround_time_ns", "std"),
        bb_turnaround_median_ns=("turnaround_time_ns", "median"),
        bb_turnaround_max_ns=("turnaround_time_ns", "max"),
        bb_queue_mean_ns=(
            "communication_qubit_queue_delay_ns",
            "mean",
        ),
        bb_queue_std_ns=(
            "communication_qubit_queue_delay_ns",
            "std",
        ),
        bb_queue_max_ns=(
            "communication_qubit_queue_delay_ns",
            "max",
        ),
        bb_wait_mean_ns=("waiting_time_ns", "mean"),
        bb_wait_std_ns=("waiting_time_ns", "std"),
        bb_wait_max_ns=("waiting_time_ns", "max"),
        bb_conflict_mean=("allocation_conflict_checks", "mean"),
        bb_conflict_sum=("allocation_conflict_checks", "sum"),
        bb_source_reuse_mean_ns=("source_reuse_delay_ns", "mean"),
        bb_target_reuse_mean_ns=("target_reuse_delay_ns", "mean"),
    ).reset_index()

    # Vectorized group quantiles avoid thousands of Python lambda calls.
    quantile_source = log.copy()
    quantile_source["turnaround_time_ns"] = pd.to_numeric(
        quantile_source["turnaround_time_ns"],
        errors="coerce",
    )
    quantile_source["communication_qubit_queue_delay_ns"] = pd.to_numeric(
        quantile_source["communication_qubit_queue_delay_ns"],
        errors="coerce",
    )

    quantile_rows = (
        quantile_source.groupby(
            group_columns,
            dropna=False,
            observed=True,
        )[[
            "turnaround_time_ns",
            "communication_qubit_queue_delay_ns",
        ]]
        .quantile(0.9)
        .rename(
            columns={
                "turnaround_time_ns": "bb_turnaround_p90_ns",
                "communication_qubit_queue_delay_ns": "bb_queue_p90_ns",
            }
        )
        .reset_index()
    )

    aggregate = aggregate.merge(
        quantile_rows,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    # Preserve the full 48-probe shape. Failed numeric values become -1,
    # while a separate raw failure vector preserves status explicitly.
    raw_specs = [
        (
            "turnaround_time_ns",
            "bb_probe_turnaround",
            -1.0,
        ),
        (
            "communication_qubit_queue_delay_ns",
            "bb_probe_queue",
            -1.0,
        ),
        (
            "request_failed",
            "bb_probe_failed",
            1.0,
        ),
    ]

    index_columns = group_columns

    for value_column, prefix, fill_value in raw_specs:
        pivot = log.pivot_table(
            index=index_columns,
            columns="logical_event_id",
            values=value_column,
            aggfunc="first",
        )

        pivot = pivot.sort_index(axis=1)

        pivot.columns = [
            f"{prefix}_{int(column):03d}"
            for column in pivot.columns
        ]

        pivot = pivot.reset_index()

        feature_columns = [
            column
            for column in pivot.columns
            if column.startswith(prefix)
        ]

        for column in feature_columns:
            if prefix == "bb_probe_failed":
                pivot[column] = boolean_series(pivot[column]).astype(float)
            else:
                pivot[column] = pd.to_numeric(
                    pivot[column],
                    errors="coerce",
                )

        pivot[feature_columns] = pivot[feature_columns].fillna(fill_value)

        aggregate = aggregate.merge(
            pivot,
            on=index_columns,
            how="left",
            validate="one_to_one",
        )

    aggregate["true_victim_present"] = (
        aggregate["execution_mode"].astype(str)
        == "combined"
    ).astype(int)

    aggregate["classification_group"] = aggregate[
        "configuration_id"
    ].astype(str)

    return aggregate


def run_random_forest_classification(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify victim presence on held-out allocation configurations."""

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    except ImportError:
        print(
            "scikit-learn is not installed. Random-forest classification "
            "will be skipped; corrected non-ML results will still be saved."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    feature_columns = [
        column
        for column in feature_table.columns
        if column.startswith("bb_")
    ]

    features = (
        feature_table[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    labels = feature_table["true_victim_present"].astype(int).to_numpy()
    groups = feature_table["classification_group"].astype(str).to_numpy()

    unique_group_count = len(np.unique(groups))
    split_count = min(CLASSIFICATION_FOLDS, unique_group_count)

    if split_count < 2:
        print("Not enough independent configurations for ML classification.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    classifier = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_TREES,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    cross_validation = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    predictions = cross_val_predict(
        classifier,
        features,
        labels,
        groups=groups,
        cv=cross_validation,
        n_jobs=-1,
        method="predict",
    )

    prediction_table = feature_table[
        [
            "victim_qasm",
            "victim_tag",
            "subexperiment",
            "configuration_id",
            "trial_id",
            "execution_mode",
            "allocation_policy",
            "communication_qubits_per_module",
            "tenants_sharing_module",
            "reservation_duration_ns",
            "reset_time_ns",
            "epr_prefetch_enabled",
            "allocation_coordination",
            "allocation_granularity",
            "true_victim_present",
        ]
    ].copy()

    prediction_table["predicted_victim_present"] = predictions.astype(int)
    prediction_table["correct"] = (
        prediction_table["true_victim_present"]
        == prediction_table["predicted_victim_present"]
    )

    metric_frames = []

    overall_metrics = binary_classification_metrics(
        prediction_table,
        [],
        "predicted_victim_present",
    )
    overall_metrics.insert(0, "scope", "overall")
    metric_frames.append(overall_metrics)

    for scope, columns in [
        ("subexperiment", ["subexperiment"]),
        ("policy", ["subexperiment", "allocation_policy"]),
        (
            "capacity",
            [
                "subexperiment",
                "communication_qubits_per_module",
            ],
        ),
    ]:
        metrics = binary_classification_metrics(
            prediction_table,
            columns,
            "predicted_victim_present",
        )
        metrics.insert(0, "scope", scope)
        metric_frames.append(metrics)

    metrics_table = pd.concat(metric_frames, ignore_index=True)

    classifier.fit(features, labels)

    importance_table = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    overall_matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    plt.figure(figsize=(7, 6))
    plt.imshow(overall_matrix, aspect="equal")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], ["No victim", "Victim present"])
    plt.yticks([0, 1], ["No victim", "Victim present"])
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Phase 1.3 Victim-Presence Classification")

    for row_index in range(2):
        for column_index in range(2):
            plt.text(
                column_index,
                row_index,
                str(overall_matrix[row_index, column_index]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "victim_presence_random_forest_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    top_importance = importance_table.head(20).sort_values("importance")

    axis = top_importance.plot(
        kind="barh",
        x="feature",
        y="importance",
        legend=False,
        figsize=(11, 8),
    )
    axis.set_xlabel("Random-forest feature importance")
    axis.set_ylabel("Attacker-visible trace feature")
    axis.set_title("Phase 1.3 Victim-Presence Features")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "victim_presence_random_forest_feature_importance.png",
        dpi=300,
    )
    plt.close()

    return prediction_table, metrics_table, importance_table


# ============================================================
# Plotting corrected aggregate results
# ============================================================


def save_corrected_plots(
    policy_summary: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    reset_summary: pd.DataFrame,
) -> None:
    """Save a concise set of corrected Phase 1.3 plots."""

    if not capacity_summary.empty:
        data = capacity_summary.sort_values(
            [
                "subexperiment",
                "communication_qubits_per_module",
            ]
        )

        for subexperiment, subset in data.groupby(
            "subexperiment",
            observed=True,
        ):
            plt.figure(figsize=(9, 5))
            plt.plot(
                subset["communication_qubits_per_module"],
                subset[
                    "mean_signed_timing_change_probability"
                ],
                marker="o",
                label="Any signed timing change",
            )
            plt.plot(
                subset["communication_qubits_per_module"],
                subset[
                    "mean_failure_only_detection_probability"
                ],
                marker="o",
                label="New request failure",
            )
            plt.xlabel("Communication qubits per module")
            plt.ylabel("Attacker-observed probability")
            plt.title(
                "Corrected Leakage versus Communication-Qubit Capacity\n"
                f"{subexperiment}"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR
                / f"corrected_capacity_leakage_{subexperiment}.png",
                dpi=300,
            )
            plt.close()

    if not policy_summary.empty:
        core = policy_summary[
            policy_summary["subexperiment"]
            == "core_policy_capacity_tenancy"
        ].copy()

        if not core.empty:
            core = core.sort_values(
                "mean_signed_timing_or_failure_detection_probability",
                ascending=False,
            )

            axis = core.plot(
                kind="bar",
                x="allocation_policy",
                y=[
                    "mean_signed_timing_change_probability",
                    "mean_failure_only_detection_probability",
                ],
                figsize=(13, 6),
            )
            axis.set_xlabel("Communication-qubit allocation policy")
            axis.set_ylabel("Mean attacker-observed probability")
            axis.set_title(
                "Phase 1.3: Timing Leakage versus Failure Leakage"
            )
            axis.tick_params(axis="x", rotation=35)
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / "corrected_policy_timing_vs_failure.png",
                dpi=300,
            )
            plt.close()

    if not reset_summary.empty:
        data = reset_summary.sort_values(
            [
                "reservation_duration_ns",
                "reset_time_ns",
            ]
        )

        for reservation, subset in data.groupby(
            "reservation_duration_ns",
            observed=True,
        ):
            plt.figure(figsize=(9, 5))
            plt.plot(
                subset["reset_time_ns"],
                subset["mean_absolute_turnaround_change_ns"],
                marker="o",
            )
            plt.xlabel("Communication-qubit reset time (ns)")
            plt.ylabel("Mean absolute attacker timing change (ns)")
            plt.title(
                "Reset-Induced Timing Leakage\n"
                f"Reservation duration = {reservation} ns"
            )
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR
                / f"corrected_reset_leakage_reservation_{reservation}ns.png",
                dpi=300,
            )
            plt.close()


# ============================================================
# Main
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_paths = {
        key: find_input_file(filename)
        for key, filename in INPUT_FILENAMES.items()
    }

    print("Resolved Phase 1.3 input files:")
    for key, path in input_paths.items():
        print(f"  {key}: {path}")

    comparison = pd.read_csv(
        input_paths["comparison"],
        low_memory=False,
    )

    request_log = pd.read_csv(
        input_paths["request_log"],
        low_memory=False,
    )

    original_trials = pd.read_csv(
        input_paths["trial_summary"],
        low_memory=False,
    )

    print("\nCorrecting probe-level leakage outcomes...")
    corrected_probes = correct_probe_comparisons(comparison)

    if SAVE_CORRECTED_PROBE_TABLE:
        corrected_probes.to_csv(
            OUTPUT_DIR / "corrected_probe_comparisons.csv",
            index=False,
        )

    print("Aggregating corrected trial metrics...")
    corrected_trials = aggregate_corrected_trials(
        corrected_probes,
        original_trials,
    )

    corrected_trials.to_csv(
        OUTPUT_DIR / "corrected_trial_summary.csv",
        index=False,
    )

    summary_specs = {
        "corrected_configuration_summary.csv": [
            "subexperiment",
            "configuration_id",
            "allocation_policy",
            "communication_qubits_per_module",
            "tenants_sharing_module",
            "reservation_duration_ns",
            "reset_time_ns",
            "epr_prefetch_enabled",
            "allocation_coordination",
            "allocation_granularity",
        ],
        "corrected_policy_summary.csv": [
            "subexperiment",
            "allocation_policy",
        ],
        "corrected_capacity_summary.csv": [
            "subexperiment",
            "communication_qubits_per_module",
        ],
        "corrected_tenancy_summary.csv": [
            "subexperiment",
            "tenants_sharing_module",
        ],
        "corrected_reservation_reset_summary.csv": [
            "subexperiment",
            "reservation_duration_ns",
            "reset_time_ns",
        ],
        "corrected_coordination_summary.csv": [
            "subexperiment",
            "allocation_coordination",
        ],
        "corrected_granularity_summary.csv": [
            "subexperiment",
            "allocation_granularity",
        ],
        "corrected_epr_summary.csv": [
            "subexperiment",
            "epr_prefetch_enabled",
        ],
        "corrected_failure_summary.csv": [
            "subexperiment",
            "allocation_policy",
            "communication_qubits_per_module",
        ],
        "corrected_utilization_summary.csv": [
            "subexperiment",
            "allocation_policy",
            "communication_qubits_per_module",
        ],
    }

    generated_summaries: dict[str, pd.DataFrame] = {}

    for filename, group_columns in summary_specs.items():
        summary = summarize_trials(
            corrected_trials,
            group_columns,
        )
        summary.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )
        generated_summaries[filename] = summary

    print("Building genuine non-ML negative controls...")
    nonml_predictions = create_nonml_rule_dataset(corrected_probes)

    nonml_predictions.to_csv(
        OUTPUT_DIR / "nonml_paired_change_predictions.csv",
        index=False,
    )

    nonml_metric_frames = []

    overall_nonml = binary_classification_metrics(
        nonml_predictions,
        [],
        "rule_detected_victim",
    )
    overall_nonml.insert(0, "scope", "overall")
    nonml_metric_frames.append(overall_nonml)

    for scope, columns in [
        ("subexperiment", ["subexperiment"]),
        ("policy", ["subexperiment", "allocation_policy"]),
        (
            "capacity",
            [
                "subexperiment",
                "communication_qubits_per_module",
            ],
        ),
    ]:
        metrics = binary_classification_metrics(
            nonml_predictions,
            columns,
            "rule_detected_victim",
        )
        metrics.insert(0, "scope", scope)
        nonml_metric_frames.append(metrics)

    nonml_metrics = pd.concat(
        nonml_metric_frames,
        ignore_index=True,
    )

    nonml_metrics.to_csv(
        OUTPUT_DIR / "nonml_paired_change_metrics.csv",
        index=False,
    )

    ml_predictions = pd.DataFrame()
    ml_metrics = pd.DataFrame()
    feature_importance = pd.DataFrame()

    if RUN_RANDOM_FOREST_CLASSIFICATION:
        print("Building attacker-visible trace features...")
        trace_features = build_trace_feature_table(request_log)

        if SAVE_TRACE_FEATURE_TABLE:
            trace_features.to_csv(
                OUTPUT_DIR / "victim_presence_trace_features.csv",
                index=False,
            )

        print(
            "Running random-forest victim-presence classification on "
            "held-out allocation configurations..."
        )

        (
            ml_predictions,
            ml_metrics,
            feature_importance,
        ) = run_random_forest_classification(trace_features)

        if not ml_predictions.empty:
            ml_predictions.to_csv(
                OUTPUT_DIR / "victim_presence_random_forest_predictions.csv",
                index=False,
            )

        if not ml_metrics.empty:
            ml_metrics.to_csv(
                OUTPUT_DIR / "victim_presence_random_forest_metrics.csv",
                index=False,
            )

        if not feature_importance.empty:
            feature_importance.to_csv(
                OUTPUT_DIR / "victim_presence_random_forest_feature_importance.csv",
                index=False,
            )

    save_corrected_plots(
        generated_summaries["corrected_policy_summary.csv"],
        generated_summaries["corrected_capacity_summary.csv"],
        generated_summaries[
            "corrected_reservation_reset_summary.csv"
        ],
    )

    overall_summary = {
        "timing_change_threshold_ns": TIMING_CHANGE_THRESHOLD_NS,
        "input_files": {
            key: str(path)
            for key, path in input_paths.items()
        },
        "probe_row_count": int(len(corrected_probes)),
        "trial_count": int(len(corrected_trials)),
        "successful_victim_trial_count": int(
            corrected_trials["successful_victim_execution"].sum()
        ),
        "victim_failure_trial_count": int(
            (~corrected_trials["successful_victim_execution"]).sum()
        ),
        "mean_positive_delay_detection_probability": float(
            corrected_trials[
                "positive_delay_detection_probability"
            ].mean()
        ),
        "mean_negative_speedup_detection_probability": float(
            corrected_trials[
                "negative_speedup_detection_probability"
            ].mean()
        ),
        "mean_failure_only_detection_probability": float(
            corrected_trials[
                "failure_only_detection_probability"
            ].mean()
        ),
        "mean_signed_timing_or_failure_detection_probability": float(
            corrected_trials[
                "signed_timing_or_failure_detection_probability"
            ].mean()
        ),
        "mean_successful_victim_slowdown_ratio": float(
            corrected_trials[
                "successful_victim_slowdown_ratio"
            ].mean()
        ),
        "nonml_overall_metrics": (
            overall_nonml.iloc[0].to_dict()
            if not overall_nonml.empty
            else {}
        ),
        "random_forest_overall_metrics": (
            ml_metrics[
                ml_metrics["scope"] == "overall"
            ].iloc[0].to_dict()
            if (
                not ml_metrics.empty
                and "scope" in ml_metrics.columns
                and (
                    ml_metrics["scope"] == "overall"
                ).any()
            )
            else {}
        ),
        "interpretation_notes": [
            "Positive delay, negative speedup, and request-failure leakage are reported separately.",
            "Victim slowdown excludes executions with rejected or failed victim requests.",
            "Aggregate resource demand is not physical utilization and may exceed one because its components overlap.",
            "The non-ML detector uses cross-trial attacker-only traces as negative controls.",
            "The random forest uses only attacker-visible traces and holds out complete allocation configurations.",
        ],
    }

    with (
        OUTPUT_DIR / "phase1_03_postprocessing_summary.json"
    ).open("w", encoding="utf-8") as output_file:
        json.dump(
            overall_summary,
            output_file,
            indent=2,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )

    print("\n=== Corrected Phase 1.3 summary ===")
    print(
        f"Probe comparisons: {len(corrected_probes):,}"
    )
    print(
        f"Trials: {len(corrected_trials):,}"
    )
    print(
        "Successful victim trials: "
        f"{int(corrected_trials['successful_victim_execution'].sum()):,}"
    )
    print(
        "Victim failure/rejection trials excluded from slowdown: "
        f"{int((~corrected_trials['successful_victim_execution']).sum()):,}"
    )
    print(
        "Mean positive-delay detection probability: "
        f"{corrected_trials['positive_delay_detection_probability'].mean():.6f}"
    )
    print(
        "Mean negative-speedup detection probability: "
        f"{corrected_trials['negative_speedup_detection_probability'].mean():.6f}"
    )
    print(
        "Mean new-failure detection probability: "
        f"{corrected_trials['failure_only_detection_probability'].mean():.6f}"
    )
    print(
        "Mean successful victim slowdown ratio: "
        f"{corrected_trials['successful_victim_slowdown_ratio'].mean():.6f}"
    )

    if not overall_nonml.empty:
        print("\nNon-ML paired-change detector:")
        print(
            overall_nonml[
                [
                    "accuracy",
                    "balanced_accuracy",
                    "sensitivity_recall",
                    "specificity",
                    "precision",
                    "f1",
                ]
            ].to_string(index=False)
        )

    if not ml_metrics.empty:
        overall_ml = ml_metrics[
            ml_metrics["scope"] == "overall"
        ]
        if not overall_ml.empty:
            print("\nRandom-forest victim-presence classifier:")
            print(
                overall_ml[
                    [
                        "accuracy",
                        "balanced_accuracy",
                        "sensitivity_recall",
                        "specificity",
                        "precision",
                        "f1",
                    ]
                ].to_string(index=False)
            )

    print("\nSaved corrected outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
    )
    main()
