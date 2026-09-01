#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_random_victim_start_sweep.py

Knob 6: random victim-start uncertainty.

Dependency
----------
run_attack_tier1_p1_static_blackbox_observation_window_sweep.py

Threat model
------------
The attacker uses one fixed estimated victim-window start. The victim's true
start varies randomly around that estimate.

Fixed
-----
- Probe 3 light-periodic probe
- uniform 420 ns inter-probe spacing
- 20,000 ns observation duration
- attacker estimated start = 20,000 ns
- P1 disjoint placement
- static-distributed victim execution
- one serialized hub-service slot
- victim-first timestamp tie breaking

Varied
------
The victim's true start is sampled uniformly from:

- exact:      0 ns uncertainty
- ±0.5 us:  ±500 ns
- ±1 us:   ±1,000 ns
- ±2.5 us: ±2,500 ns
- ±5 us:   ±5,000 ns
- ±10 us: ±10,000 ns

The same random start offsets are reused for every victim workload.

Outputs
-------
blackbox_window_results/random_victim_start/

    random_victim_start_trials.csv
    random_victim_start_aggregate_by_victim.csv
    random_victim_start_overall_summary.csv
    random_victim_start_offsets.csv

    random_victim_start_signal.png
    random_victim_start_signal_retention.png
    random_victim_start_detection_probability.png
    random_victim_start_contention_fraction.png
    random_victim_start_victim_slowdown.png

    uncertainty_exact/
    uncertainty_0p5us/
    uncertainty_1us/
    uncertainty_2p5us/
    uncertainty_5us/
    uncertainty_10us/
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import run_atack_tier1_p1_static_blackbox_observation_window_sweep as base


# ============================================================
# Output and experiment configuration
# ============================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "random_victim_start"
)

# The epoch is shifted forward so that a victim starting
# 10 us early still has a non-negative start time.
ATTACKER_ESTIMATED_START_NS = 20_000

OBSERVATION_DURATION_NS = 20_000

PROBE_ROUND_PERIOD_NS = 420

NUM_TRIALS_PER_UNCERTAINTY = 20

RANDOM_SEED = 27

# Save detailed request-level files for every repeated trial.
#
# Set this to False if only summary CSVs are needed.
SAVE_REQUEST_LEVEL_FILES = True


UNCERTAINTY_CONFIGS = [
    {
        "uncertainty_name": "uncertainty_exact",
        "uncertainty_half_width_ns": 0,
    },
    {
        "uncertainty_name": "uncertainty_0p5us",
        "uncertainty_half_width_ns": 500,
    },
    {
        "uncertainty_name": "uncertainty_1us",
        "uncertainty_half_width_ns": 1_000,
    },
    {
        "uncertainty_name": "uncertainty_2p5us",
        "uncertainty_half_width_ns": 2_500,
    },
    {
        "uncertainty_name": "uncertainty_5us",
        "uncertainty_half_width_ns": 5_000,
    },
    {
        "uncertainty_name": "uncertainty_10us",
        "uncertainty_half_width_ns": 10_000,
    },
]

UNCERTAINTY_ORDER = [
    config["uncertainty_name"]
    for config in UNCERTAINTY_CONFIGS
]


# Configure the imported architecture helpers.
base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
    ATTACKER_ESTIMATED_START_NS
)

base.PROBE_ROUND_PERIOD_NS = (
    PROBE_ROUND_PERIOD_NS
)


# ============================================================
# Random start-offset generation
# ============================================================

def generate_trial_offsets() -> dict[str, list[int]]:
    """
    Generate repeatable victim-start offsets.

    Offset definition
    -----------------
    victim_start_offset_ns =
        victim_true_start_ns - attacker_estimated_start_ns

    Negative:
        victim begins earlier than the attacker expects.

    Positive:
        victim begins later than the attacker expects.

    Offsets are generated in 5 ns increments to match the
    simulator's event-timing granularity.
    """

    offsets_by_uncertainty: dict[
        str,
        list[int],
    ] = {}

    for config_index, config in enumerate(
        UNCERTAINTY_CONFIGS
    ):
        uncertainty_name = (
            config["uncertainty_name"]
        )

        half_width_ns = int(
            config[
                "uncertainty_half_width_ns"
            ]
        )

        if half_width_ns == 0:
            offsets = [
                0
                for _ in range(
                    NUM_TRIALS_PER_UNCERTAINTY
                )
            ]

        else:
            random_generator = random.Random(
                RANDOM_SEED
                + config_index
            )

            offsets = [
                random_generator.randrange(
                    -half_width_ns,
                    half_width_ns + 1,
                    5,
                )
                for _ in range(
                    NUM_TRIALS_PER_UNCERTAINTY
                )
            ]

        offsets_by_uncertainty[
            uncertainty_name
        ] = offsets

    return offsets_by_uncertainty


def create_offset_dataframe(
    offsets_by_uncertainty:
    dict[str, list[int]],
) -> pd.DataFrame:
    """Store the exact random offsets used."""

    rows: list[dict] = []

    for config in UNCERTAINTY_CONFIGS:
        uncertainty_name = (
            config["uncertainty_name"]
        )

        half_width_ns = int(
            config[
                "uncertainty_half_width_ns"
            ]
        )

        offsets = offsets_by_uncertainty[
            uncertainty_name
        ]

        for trial_id, offset_ns in enumerate(
            offsets
        ):
            rows.append(
                {
                    "uncertainty_name": (
                        uncertainty_name
                    ),
                    "uncertainty_half_width_ns": (
                        half_width_ns
                    ),
                    "trial_id": trial_id,
                    "victim_start_offset_ns": (
                        offset_ns
                    ),
                    "attacker_start_error_ns": (
                        -offset_ns
                    ),
                    "victim_true_start_ns": (
                        ATTACKER_ESTIMATED_START_NS
                        + offset_ns
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Timing helpers
# ============================================================

def victim_completion_time_ns(
    victim_ground_truth: pd.DataFrame,
    fallback_start_ns: int,
) -> float:
    """Return the victim's final absolute completion time."""

    if victim_ground_truth.empty:
        return float(
            fallback_start_ns
        )

    return float(
        victim_ground_truth[
            "completion_time_ns"
        ].max()
    )


def victim_duration_ns(
    victim_ground_truth: pd.DataFrame,
    victim_start_ns: int,
) -> float:
    """Return victim duration relative to its true start."""

    return (
        victim_completion_time_ns(
            victim_ground_truth,
            victim_start_ns,
        )
        - victim_start_ns
    )


# ============================================================
# Reusable victim preparation
# ============================================================

def prepare_victim(
    victim_qasm: str,
) -> dict:
    """
    Prepare all deterministic controls for one victim.

    The attacker-only control and victim-only duration do not
    need to be recomputed for every random-start trial.
    """

    (
        victim_trace,
        victim_mapping,
        victim_num_qubits,
        victim_cross_operations,
    ) = base.extract_static_victim_trace(
        victim_qasm
    )

    window_config = {
        "window_name": (
            "random_victim_start"
        ),
        "observation_duration_ns": (
            OBSERVATION_DURATION_NS
        ),
    }

    (
        attacker_schedule,
        attacker_mapping,
        cross_step_metadata,
        schedule_metadata,
    ) = base.build_observation_window_schedule(
        window_config,
        victim_num_qubits,
    )

    # --------------------------------------------------------
    # Attacker-only calibration
    # --------------------------------------------------------

    attacker_only_architecture = (
        base.build_architecture(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    base.execute_timed_schedule(
        attacker_only_architecture,
        copy.deepcopy(
            attacker_schedule
        ),
    )

    attacker_only_observations = (
        base.collect_attacker_observations(
            attacker_only_architecture,
            cross_step_metadata,
            "attacker_only",
        )
    )

    # --------------------------------------------------------
    # Victim-only duration calibration
    # --------------------------------------------------------

    base.VICTIM_TRUE_START_NS = (
        ATTACKER_ESTIMATED_START_NS
    )

    nominal_victim_schedule = (
        base.schedule_victim_events(
            victim_trace
        )
    )

    victim_only_architecture = (
        base.build_architecture(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    base.execute_timed_schedule(
        victim_only_architecture,
        copy.deepcopy(
            nominal_victim_schedule
        ),
    )

    victim_only_ground_truth = (
        base.collect_victim_ground_truth(
            victim_only_architecture,
            "victim_only",
        )
    )

    baseline_victim_duration_ns = (
        victim_duration_ns(
            victim_only_ground_truth,
            ATTACKER_ESTIMATED_START_NS,
        )
    )

    return {
        "victim_trace": victim_trace,
        "victim_mapping": victim_mapping,
        "victim_num_qubits": (
            victim_num_qubits
        ),
        "victim_cross_operations": (
            victim_cross_operations
        ),
        "attacker_schedule": (
            attacker_schedule
        ),
        "attacker_mapping": (
            attacker_mapping
        ),
        "cross_step_metadata": (
            cross_step_metadata
        ),
        "schedule_metadata": (
            schedule_metadata
        ),
        "attacker_only_observations": (
            attacker_only_observations
        ),
        "attacker_only_architecture": (
            attacker_only_architecture
        ),
        "victim_only_ground_truth": (
            victim_only_ground_truth
        ),
        "victim_only_architecture": (
            victim_only_architecture
        ),
        "baseline_victim_duration_ns": (
            baseline_victim_duration_ns
        ),
    }


# ============================================================
# One random-start trial
# ============================================================

def run_one_trial(
    *,
    victim_qasm: str,
    prepared: dict,
    uncertainty_config: dict,
    trial_id: int,
    victim_start_offset_ns: int,
) -> dict:
    """Run one victim with one randomly sampled start."""

    victim_tag = base.safe_tag(
        victim_qasm
    )

    uncertainty_name = (
        uncertainty_config[
            "uncertainty_name"
        ]
    )

    uncertainty_half_width_ns = int(
        uncertainty_config[
            "uncertainty_half_width_ns"
        ]
    )

    victim_true_start_ns = (
        ATTACKER_ESTIMATED_START_NS
        + victim_start_offset_ns
    )

    attacker_start_error_ns = (
        ATTACKER_ESTIMATED_START_NS
        - victim_true_start_ns
    )

    if victim_true_start_ns < 0:
        raise ValueError(
            "Random victim start became negative."
        )

    print(
        "\n=== Random victim start: "
        f"{uncertainty_name} | "
        f"trial {trial_id:02d} | "
        f"offset {victim_start_offset_ns:+d} ns | "
        f"{victim_qasm} ==="
    )

    # This is the only execution variable changed
    # for the current trial.
    base.VICTIM_TRUE_START_NS = (
        victim_true_start_ns
    )

    victim_schedule = (
        base.schedule_victim_events(
            prepared["victim_trace"]
        )
    )

    victim_on_architecture = (
        base.build_architecture(
            prepared["victim_mapping"],
            prepared["victim_num_qubits"],
            prepared["attacker_mapping"],
        )
    )

    merged_schedule = (
        copy.deepcopy(
            victim_schedule
        )
        + copy.deepcopy(
            prepared[
                "attacker_schedule"
            ]
        )
    )

    base.execute_timed_schedule(
        victim_on_architecture,
        merged_schedule,
    )

    victim_present_observations = (
        base.collect_attacker_observations(
            victim_on_architecture,
            prepared[
                "cross_step_metadata"
            ],
            "victim_present",
        )
    )

    victim_on_ground_truth = (
        base.collect_victim_ground_truth(
            victim_on_architecture,
            "victim_present",
        )
    )

    compared = (
        base.compare_attacker_runs(
            prepared[
                "attacker_only_observations"
            ],
            victim_present_observations,
        )
    )

    baseline_victim_duration_ns = float(
        prepared[
            "baseline_victim_duration_ns"
        ]
    )

    victim_on_duration_ns = (
        victim_duration_ns(
            victim_on_ground_truth,
            victim_true_start_ns,
        )
    )

    victim_slowdown_ns = (
        victim_on_duration_ns
        - baseline_victim_duration_ns
    )

    victim_slowdown_ratio = (
        victim_on_duration_ns
        / baseline_victim_duration_ns
        if baseline_victim_duration_ns > 0
        else 1.0
    )

    baseline_victim_completion_ns = (
        victim_true_start_ns
        + baseline_victim_duration_ns
    )

    attacker_window_start_ns = (
        ATTACKER_ESTIMATED_START_NS
    )

    attacker_window_end_ns = (
        ATTACKER_ESTIMATED_START_NS
        + OBSERVATION_DURATION_NS
    )

    overlap_start_ns = max(
        attacker_window_start_ns,
        victim_true_start_ns,
    )

    overlap_end_ns = min(
        attacker_window_end_ns,
        baseline_victim_completion_ns,
    )

    window_victim_overlap_ns = max(
        0.0,
        overlap_end_ns
        - overlap_start_ns,
    )

    release_times = compared[
        "request_release_time_ns"
    ]

    pre_victim_probe_count = int(
        (
            release_times
            < victim_true_start_ns
        ).sum()
    )

    useful_probe_count = int(
        (
            (
                release_times
                >= victim_true_start_ns
            )
            & (
                release_times
                <= baseline_victim_completion_ns
            )
        ).sum()
    )

    post_victim_probe_count = int(
        (
            release_times
            > baseline_victim_completion_ns
        ).sum()
    )

    total_probe_count = int(
        len(compared)
    )

    useful_probe_fraction = (
        useful_probe_count
        / total_probe_count
        if total_probe_count > 0
        else 0.0
    )

    delayed_probe_count = int(
        compared[
            "victim_contention_observed"
        ].sum()
    )

    total_excess_latency_ns = float(
        compared[
            "excess_turnaround_time_ns"
        ].sum()
    )

    detection_observed = (
        delayed_probe_count > 0
    )

    summary = {
        "victim_qasm": victim_qasm,
        "victim_tag": victim_tag,
        "knob": (
            "random_victim_start"
        ),
        "uncertainty_name": (
            uncertainty_name
        ),
        "uncertainty_half_width_ns": (
            uncertainty_half_width_ns
        ),
        "trial_id": trial_id,
        "random_seed": RANDOM_SEED,
        "victim_start_offset_ns": (
            victim_start_offset_ns
        ),
        "attacker_start_error_ns": (
            attacker_start_error_ns
        ),
        "victim_true_start_ns": (
            victim_true_start_ns
        ),
        "attacker_estimated_start_ns": (
            ATTACKER_ESTIMATED_START_NS
        ),
        "attacker_window_end_ns": (
            attacker_window_end_ns
        ),
        "observation_duration_ns": (
            OBSERVATION_DURATION_NS
        ),
        "probe_name": (
            "probe_3_light_periodic"
        ),
        "spacing_pattern": "uniform",
        "probe_round_period_ns": (
            PROBE_ROUND_PERIOD_NS
        ),
        "placement": "P1_disjoint",
        "workload_type": (
            "static_distributed"
        ),
        "hub_max_concurrent_transfers": (
            base.HUB_MAX_CONCURRENT_TRANSFERS
        ),
        "total_attacker_remote_requests": int(
            len(compared)
        ),
        "baseline_avg_waiting_time_ns": float(
            prepared[
                "attacker_only_observations"
            ][
                "waiting_time_ns"
            ].mean()
        ),
        "baseline_max_waiting_time_ns": float(
            prepared[
                "attacker_only_observations"
            ][
                "waiting_time_ns"
            ].max()
        ),
        "avg_excess_waiting_time_ns": float(
            compared[
                "excess_waiting_time_ns"
            ].mean()
        ),
        "avg_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].mean()
        ),
        "median_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].median()
        ),
        "max_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].max()
        ),
        "total_excess_turnaround_time_ns": (
            total_excess_latency_ns
        ),
        "delayed_probe_count": (
            delayed_probe_count
        ),
        "contention_observed_fraction": float(
            compared[
                "victim_contention_observed"
            ].mean()
        ),
        "detection_observed": bool(
            detection_observed
        ),
        "victim_only_duration_ns": (
            baseline_victim_duration_ns
        ),
        "victim_on_duration_ns": (
            victim_on_duration_ns
        ),
        "victim_slowdown_ns": (
            victim_slowdown_ns
        ),
        "victim_slowdown_ratio": (
            victim_slowdown_ratio
        ),
        "window_victim_overlap_ns_evaluator_only": (
            window_victim_overlap_ns
        ),
        "pre_victim_probe_count_evaluator_only": (
            pre_victim_probe_count
        ),
        "useful_probe_count_evaluator_only": (
            useful_probe_count
        ),
        "post_victim_probe_count_evaluator_only": (
            post_victim_probe_count
        ),
        "useful_probe_fraction_evaluator_only": (
            useful_probe_fraction
        ),
        "victim_on_hub_makespan_ns": int(
            victim_on_architecture
            .hub.current_time_ns
        ),
    }

    # --------------------------------------------------------
    # Save detailed trial files
    # --------------------------------------------------------

    if SAVE_REQUEST_LEVEL_FILES:
        trial_directory = (
            OUTPUT_DIR
            / uncertainty_name
            / victim_tag
            / f"trial_{trial_id:02d}"
        )

        trial_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        compared.to_csv(
            trial_directory
            / "attacker_observations.csv",
            index=False,
        )

        victim_on_ground_truth.to_csv(
            trial_directory
            / "victim_ground_truth.csv",
            index=False,
        )

        with (
            trial_directory
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                summary,
                output_file,
                indent=2,
            )

    return summary


# ============================================================
# Aggregation
# ============================================================

def aggregate_by_victim(
    trial_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate repeated trials per victim and uncertainty."""

    grouped = trial_dataframe.groupby(
        [
            "victim_tag",
            "uncertainty_name",
            "uncertainty_half_width_ns",
        ],
        observed=True,
        sort=False,
    )

    aggregate = grouped.agg(
        trial_count=(
            "trial_id",
            "count",
        ),
        sampled_offset_mean_ns=(
            "victim_start_offset_ns",
            "mean",
        ),
        sampled_offset_std_ns=(
            "victim_start_offset_ns",
            "std",
        ),
        sampled_offset_min_ns=(
            "victim_start_offset_ns",
            "min",
        ),
        sampled_offset_max_ns=(
            "victim_start_offset_ns",
            "max",
        ),
        avg_signal_mean_ns=(
            "avg_excess_turnaround_time_ns",
            "mean",
        ),
        avg_signal_std_ns=(
            "avg_excess_turnaround_time_ns",
            "std",
        ),
        total_signal_mean_ns=(
            "total_excess_turnaround_time_ns",
            "mean",
        ),
        total_signal_std_ns=(
            "total_excess_turnaround_time_ns",
            "std",
        ),
        total_signal_min_ns=(
            "total_excess_turnaround_time_ns",
            "min",
        ),
        total_signal_max_ns=(
            "total_excess_turnaround_time_ns",
            "max",
        ),
        contention_fraction_mean=(
            "contention_observed_fraction",
            "mean",
        ),
        contention_fraction_std=(
            "contention_observed_fraction",
            "std",
        ),
        detection_probability=(
            "detection_observed",
            "mean",
        ),
        delayed_probe_count_mean=(
            "delayed_probe_count",
            "mean",
        ),
        victim_slowdown_ratio_mean=(
            "victim_slowdown_ratio",
            "mean",
        ),
        victim_slowdown_ratio_std=(
            "victim_slowdown_ratio",
            "std",
        ),
        victim_slowdown_ratio_max=(
            "victim_slowdown_ratio",
            "max",
        ),
        useful_probe_fraction_mean=(
            "useful_probe_fraction_evaluator_only",
            "mean",
        ),
        useful_probe_fraction_std=(
            "useful_probe_fraction_evaluator_only",
            "std",
        ),
        overlap_mean_ns=(
            "window_victim_overlap_ns_evaluator_only",
            "mean",
        ),
    ).reset_index()

    exact_signal = (
        aggregate[
            aggregate[
                "uncertainty_name"
            ] == "uncertainty_exact"
        ][
            [
                "victim_tag",
                "total_signal_mean_ns",
            ]
        ]
        .rename(
            columns={
                "total_signal_mean_ns": (
                    "exact_total_signal_mean_ns"
                )
            }
        )
    )

    aggregate = aggregate.merge(
        exact_signal,
        on="victim_tag",
        how="left",
        validate="many_to_one",
    )

    aggregate[
        "signal_retention_vs_exact"
    ] = (
        aggregate[
            "total_signal_mean_ns"
        ]
        / aggregate[
            "exact_total_signal_mean_ns"
        ].replace(0, pd.NA)
    )

    return aggregate


def aggregate_overall(
    victim_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate robustness across all victim workloads."""

    grouped = victim_aggregate.groupby(
        [
            "uncertainty_name",
            "uncertainty_half_width_ns",
        ],
        observed=True,
        sort=False,
    )

    overall = grouped.agg(
        workload_count=(
            "victim_tag",
            "count",
        ),
        average_signal_retention=(
            "signal_retention_vs_exact",
            "mean",
        ),
        minimum_signal_retention=(
            "signal_retention_vs_exact",
            "min",
        ),
        average_detection_probability=(
            "detection_probability",
            "mean",
        ),
        minimum_detection_probability=(
            "detection_probability",
            "min",
        ),
        average_contention_fraction=(
            "contention_fraction_mean",
            "mean",
        ),
        average_victim_slowdown_ratio=(
            "victim_slowdown_ratio_mean",
            "mean",
        ),
        maximum_victim_slowdown_ratio=(
            "victim_slowdown_ratio_max",
            "max",
        ),
        average_useful_probe_fraction=(
            "useful_probe_fraction_mean",
            "mean",
        ),
    ).reset_index()

    return overall


# ============================================================
# Plotting
# ============================================================

def prepare_uncertainty_axis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Sort aggregate data by uncertainty width."""

    return dataframe.sort_values(
        [
            "uncertainty_half_width_ns",
            "victim_tag",
        ]
    )


def save_workload_line_plot(
    aggregate: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot one aggregate metric for every victim."""

    plt.figure(
        figsize=(12, 6)
    )

    ordered = prepare_uncertainty_axis(
        aggregate
    )

    for victim_tag in sorted(
        ordered[
            "victim_tag"
        ].unique()
    ):
        subset = ordered[
            ordered[
                "victim_tag"
            ] == victim_tag
        ]

        x_values = (
            subset[
                "uncertainty_half_width_ns"
            ]
            / 1_000.0
        )

        plt.plot(
            x_values,
            subset[metric],
            marker="o",
            linewidth=1.5,
            label=victim_tag,
        )

    plt.xlabel(
        "Victim-start uncertainty half-width (µs)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend(
        title="Victim workload"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


def save_overall_plot(
    overall: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot an overall robustness metric."""

    ordered = overall.sort_values(
        "uncertainty_half_width_ns"
    )

    x_values = (
        ordered[
            "uncertainty_half_width_ns"
        ]
        / 1_000.0
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        x_values,
        ordered[metric],
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel(
        "Victim-start uncertainty half-width (µs)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    offsets_by_uncertainty = (
        generate_trial_offsets()
    )

    offset_dataframe = (
        create_offset_dataframe(
            offsets_by_uncertainty
        )
    )

    offset_path = (
        OUTPUT_DIR
        / "random_victim_start_offsets.csv"
    )

    offset_dataframe.to_csv(
        offset_path,
        index=False,
    )

    trial_summaries: list[dict] = []

    for victim_qasm in base.VICTIM_QASMS:
        print(
            "\n======================================"
        )

        print(
            f"Preparing victim: {victim_qasm}"
        )

        print(
            "======================================"
        )

        prepared = prepare_victim(
            victim_qasm
        )

        for uncertainty_config in (
            UNCERTAINTY_CONFIGS
        ):
            uncertainty_name = (
                uncertainty_config[
                    "uncertainty_name"
                ]
            )

            offsets = (
                offsets_by_uncertainty[
                    uncertainty_name
                ]
            )

            for trial_id, offset_ns in enumerate(
                offsets
            ):
                summary = run_one_trial(
                    victim_qasm=(
                        victim_qasm
                    ),
                    prepared=prepared,
                    uncertainty_config=(
                        uncertainty_config
                    ),
                    trial_id=trial_id,
                    victim_start_offset_ns=(
                        offset_ns
                    ),
                )

                trial_summaries.append(
                    summary
                )

    # --------------------------------------------------------
    # Save trial-level results
    # --------------------------------------------------------

    trial_dataframe = pd.DataFrame(
        trial_summaries
    )

    trial_dataframe[
        "uncertainty_name"
    ] = pd.Categorical(
        trial_dataframe[
            "uncertainty_name"
        ],
        categories=UNCERTAINTY_ORDER,
        ordered=True,
    )

    trial_dataframe = (
        trial_dataframe
        .sort_values(
            [
                "uncertainty_name",
                "victim_tag",
                "trial_id",
            ]
        )
        .reset_index(drop=True)
    )

    trial_path = (
        OUTPUT_DIR
        / "random_victim_start_trials.csv"
    )

    trial_dataframe.to_csv(
        trial_path,
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate by victim
    # --------------------------------------------------------

    victim_aggregate = (
        aggregate_by_victim(
            trial_dataframe
        )
    )

    victim_aggregate[
        "uncertainty_name"
    ] = pd.Categorical(
        victim_aggregate[
            "uncertainty_name"
        ],
        categories=UNCERTAINTY_ORDER,
        ordered=True,
    )

    victim_aggregate = (
        victim_aggregate
        .sort_values(
            [
                "uncertainty_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    victim_aggregate_path = (
        OUTPUT_DIR
        / (
            "random_victim_start_"
            "aggregate_by_victim.csv"
        )
    )

    victim_aggregate.to_csv(
        victim_aggregate_path,
        index=False,
    )

    # --------------------------------------------------------
    # Overall aggregate
    # --------------------------------------------------------

    overall = aggregate_overall(
        victim_aggregate
    )

    overall[
        "uncertainty_name"
    ] = pd.Categorical(
        overall[
            "uncertainty_name"
        ],
        categories=UNCERTAINTY_ORDER,
        ordered=True,
    )

    overall = (
        overall
        .sort_values(
            "uncertainty_name"
        )
        .reset_index(drop=True)
    )

    overall_path = (
        OUTPUT_DIR
        / (
            "random_victim_start_"
            "overall_summary.csv"
        )
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_workload_line_plot(
        victim_aggregate,
        metric=(
            "total_signal_mean_ns"
        ),
        ylabel=(
            "Mean cumulative victim-induced latency (ns)"
        ),
        title=(
            "Random Victim Start: "
            "Collected Timing Signal"
        ),
        filename=(
            "random_victim_start_signal.png"
        ),
    )

    save_workload_line_plot(
        victim_aggregate,
        metric=(
            "signal_retention_vs_exact"
        ),
        ylabel=(
            "Signal retention relative to exact start"
        ),
        title=(
            "Random Victim Start: "
            "Signal Retention"
        ),
        filename=(
            "random_victim_start_"
            "signal_retention.png"
        ),
    )

    save_workload_line_plot(
        victim_aggregate,
        metric=(
            "detection_probability"
        ),
        ylabel=(
            "Probability of observing contention"
        ),
        title=(
            "Random Victim Start: "
            "Detection Probability"
        ),
        filename=(
            "random_victim_start_"
            "detection_probability.png"
        ),
    )

    save_workload_line_plot(
        victim_aggregate,
        metric=(
            "contention_fraction_mean"
        ),
        ylabel=(
            "Mean fraction of probes delayed"
        ),
        title=(
            "Random Victim Start: "
            "Contention Coverage"
        ),
        filename=(
            "random_victim_start_"
            "contention_fraction.png"
        ),
    )

    save_workload_line_plot(
        victim_aggregate,
        metric=(
            "victim_slowdown_ratio_mean"
        ),
        ylabel=(
            "Mean victim completion-time ratio"
        ),
        title=(
            "Random Victim Start: "
            "Victim Slowdown"
        ),
        filename=(
            "random_victim_start_"
            "victim_slowdown.png"
        ),
    )

    save_overall_plot(
        overall,
        metric=(
            "average_signal_retention"
        ),
        ylabel=(
            "Average signal retention"
        ),
        title=(
            "Overall Robustness to "
            "Random Victim-Start Uncertainty"
        ),
        filename=(
            "random_victim_start_"
            "overall_retention.png"
        ),
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        "\n=== Random victim-start "
        "aggregate by workload ==="
    )

    display_columns = [
        "victim_tag",
        "uncertainty_name",
        "trial_count",
        "sampled_offset_min_ns",
        "sampled_offset_max_ns",
        "total_signal_mean_ns",
        "total_signal_std_ns",
        "signal_retention_vs_exact",
        "contention_fraction_mean",
        "detection_probability",
        "victim_slowdown_ratio_mean",
        "victim_slowdown_ratio_max",
        "useful_probe_fraction_mean",
    ]

    print(
        victim_aggregate[
            display_columns
        ].to_string(index=False)
    )

    print(
        "\n=== Overall random-start "
        "robustness ==="
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\nSaved all results to: "
        f"{OUTPUT_DIR}"
    )

    print(
        "Trial results: "
        f"{trial_path}"
    )

    print(
        "Per-victim aggregate: "
        f"{victim_aggregate_path}"
    )

    print(
        "Overall summary: "
        f"{overall_path}"
    )

    print(
        "Random offsets: "
        f"{offset_path}"
    )


if __name__ == "__main__":
    main()
