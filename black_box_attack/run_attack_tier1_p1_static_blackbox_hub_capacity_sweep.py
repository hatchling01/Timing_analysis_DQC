#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_hub_capacity_sweep.py

Knob 7: hub concurrent-transfer capacity.

Dependency
----------
run_attack_tier1_p1_static_blackbox_observation_window_sweep.py

Fixed
-----
- Probe 3 light-periodic probe
- uniform inter-probe spacing
- one probe every 420 ns
- 20,000 ns observation window
- exact victim-window start estimate
- static-distributed victim execution
- P1 disjoint placement
- victim modules: 0, 1, 2
- attacker modules: 3, 4
- victim-first tie breaking

Varied
------
- capacity 1: serialized hub
- capacity 2: original dual-slot hub
- capacity 3
- capacity 4

Purpose
-------
This is an architectural sensitivity experiment.

Under P1 placement, the attacker and victim occupy disjoint modules. If the hub
has two or more independent service slots, the attacker may be served in
parallel with the victim. Capacity >= 2 therefore acts as an important negative
control for the shared-service timing channel.

Outputs
-------
blackbox_window_results/hub_capacity/

    hub_capacity_summary.csv
    hub_capacity_schedule_summary.csv

    hub_capacity_avg_excess_latency.png
    hub_capacity_total_excess_latency.png
    hub_capacity_contention_fraction.png
    hub_capacity_victim_slowdown.png
    hub_capacity_attacker_self_wait.png
    hub_capacity_signal_retention.png

    capacity_1_serialized/
    capacity_2_original/
    capacity_3/
    capacity_4/
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import run_atack_tier1_p1_static_blackbox_observation_window_sweep as base


# ============================================================
# Output and fixed experiment settings
# ============================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "hub_capacity"
)

VICTIM_TRUE_START_NS = 1_000

ATTACKER_ESTIMATED_START_NS = 1_000

OBSERVATION_DURATION_NS = 20_000

PROBE_ROUND_PERIOD_NS = 420


CAPACITY_CONFIGS = [
    {
        "capacity_name": (
            "capacity_1_serialized"
        ),
        "hub_capacity": 1,
    },
    {
        "capacity_name": (
            "capacity_2_original"
        ),
        "hub_capacity": 2,
    },
    {
        "capacity_name": (
            "capacity_3"
        ),
        "hub_capacity": 3,
    },
    {
        "capacity_name": (
            "capacity_4"
        ),
        "hub_capacity": 4,
    },
]

CAPACITY_ORDER = [
    configuration["capacity_name"]
    for configuration in CAPACITY_CONFIGS
]


# Configure imported helper functions.
base.VICTIM_TRUE_START_NS = (
    VICTIM_TRUE_START_NS
)

base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
    ATTACKER_ESTIMATED_START_NS
)

base.PROBE_ROUND_PERIOD_NS = (
    PROBE_ROUND_PERIOD_NS
)


# ============================================================
# Timing helpers
# ============================================================

def final_victim_completion_ns(
    victim_ground_truth: pd.DataFrame,
) -> float:
    """Return the absolute final victim completion time."""

    if victim_ground_truth.empty:
        return float(
            VICTIM_TRUE_START_NS
        )

    return float(
        victim_ground_truth[
            "completion_time_ns"
        ].max()
    )


def victim_duration_ns(
    victim_ground_truth: pd.DataFrame,
) -> float:
    """Return victim duration relative to its true start."""

    return (
        final_victim_completion_ns(
            victim_ground_truth
        )
        - VICTIM_TRUE_START_NS
    )


# ============================================================
# Summary generation
# ============================================================

def create_summary(
    *,
    victim_qasm: str,
    capacity_config: dict,
    victim_trace: list[dict],
    victim_cross_operations: int,
    schedule_metadata: dict,
    attacker_only: pd.DataFrame,
    victim_present: pd.DataFrame,
    compared: pd.DataFrame,
    victim_only_ground_truth: pd.DataFrame,
    victim_on_ground_truth: pd.DataFrame,
    attacker_only_architecture,
    victim_only_architecture,
    victim_on_architecture,
) -> dict:
    """Create one workload/capacity summary."""

    hub_capacity = int(
        capacity_config[
            "hub_capacity"
        ]
    )

    victim_only_duration = (
        victim_duration_ns(
            victim_only_ground_truth
        )
    )

    victim_on_duration = (
        victim_duration_ns(
            victim_on_ground_truth
        )
    )

    victim_slowdown_ns = (
        victim_on_duration
        - victim_only_duration
    )

    victim_slowdown_ratio = (
        victim_on_duration
        / victim_only_duration
        if victim_only_duration > 0
        else 1.0
    )

    delayed_probe_count = int(
        compared[
            "victim_contention_observed"
        ].sum()
    )

    total_probe_count = int(
        len(compared)
    )

    total_excess_latency_ns = float(
        compared[
            "excess_turnaround_time_ns"
        ].sum()
    )

    positive_excess_latency_ns = float(
        compared.loc[
            compared[
                "excess_turnaround_time_ns"
            ] > 0,
            "excess_turnaround_time_ns",
        ].sum()
    )

    negative_excess_latency_ns = float(
        compared.loc[
            compared[
                "excess_turnaround_time_ns"
            ] < 0,
            "excess_turnaround_time_ns",
        ].sum()
    )

    victim_only_avg_waiting_ns = (
        float(
            victim_only_ground_truth[
                "waiting_time_ns"
            ].mean()
        )
        if not victim_only_ground_truth.empty
        else 0.0
    )

    victim_on_avg_waiting_ns = (
        float(
            victim_on_ground_truth[
                "waiting_time_ns"
            ].mean()
        )
        if not victim_on_ground_truth.empty
        else 0.0
    )

    victim_only_max_waiting_ns = (
        float(
            victim_only_ground_truth[
                "waiting_time_ns"
            ].max()
        )
        if not victim_only_ground_truth.empty
        else 0.0
    )

    victim_on_max_waiting_ns = (
        float(
            victim_on_ground_truth[
                "waiting_time_ns"
            ].max()
        )
        if not victim_on_ground_truth.empty
        else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "victim_tag": base.safe_tag(
            victim_qasm
        ),
        "knob": "hub_capacity",
        "capacity_name": (
            capacity_config[
                "capacity_name"
            ]
        ),
        "hub_max_concurrent_transfers": (
            hub_capacity
        ),
        "probe_name": (
            "probe_3_light_periodic"
        ),
        "spacing_pattern": "uniform",
        "probe_round_period_ns": (
            PROBE_ROUND_PERIOD_NS
        ),
        "observation_duration_ns": (
            OBSERVATION_DURATION_NS
        ),
        "victim_true_start_ns": (
            VICTIM_TRUE_START_NS
        ),
        "attacker_estimated_start_ns": (
            ATTACKER_ESTIMATED_START_NS
        ),
        "workload_type": (
            "static_distributed"
        ),
        "placement": "P1_disjoint",
        "threat_model": (
            "blackbox_with_coarse_"
            "window_knowledge"
        ),
        "within_round_event_spacing_ns": (
            base.WITHIN_ROUND_EVENT_SPACING_NS
        ),
        "total_probe_rounds": int(
            schedule_metadata[
                "total_probe_rounds"
            ]
        ),
        "total_attacker_events": int(
            schedule_metadata[
                "total_attacker_events"
            ]
        ),
        "total_attacker_remote_requests": int(
            schedule_metadata[
                "total_attacker_remote_requests"
            ]
        ),
        "first_remote_probe_release_ns": int(
            schedule_metadata[
                "first_remote_probe_release_ns"
            ]
        ),
        "last_remote_probe_release_ns": int(
            schedule_metadata[
                "last_remote_probe_release_ns"
            ]
        ),
        "realized_remote_probe_rate_per_us": (
            float(
                schedule_metadata[
                    "realized_remote_probe_rate_per_us"
                ]
            )
        ),
        "victim_total_events": (
            len(victim_trace)
        ),
        "victim_cross_module_ops_"
        "evaluator_only": (
            victim_cross_operations
        ),
        "victim_completed_remote_requests": (
            int(
                len(
                    victim_on_ground_truth
                )
            )
        ),
        "baseline_avg_waiting_time_ns": (
            float(
                attacker_only[
                    "waiting_time_ns"
                ].mean()
            )
        ),
        "baseline_max_waiting_time_ns": (
            float(
                attacker_only[
                    "waiting_time_ns"
                ].max()
            )
        ),
        "baseline_avg_turnaround_time_ns": (
            float(
                attacker_only[
                    "turnaround_time_ns"
                ].mean()
            )
        ),
        "baseline_waited_fraction": (
            float(
                (
                    attacker_only[
                        "waiting_time_ns"
                    ] > 0
                ).mean()
            )
        ),
        "victim_on_avg_waiting_time_ns": (
            float(
                victim_present[
                    "waiting_time_ns"
                ].mean()
            )
        ),
        "victim_on_max_waiting_time_ns": (
            float(
                victim_present[
                    "waiting_time_ns"
                ].max()
            )
        ),
        "victim_on_avg_turnaround_time_ns": (
            float(
                victim_present[
                    "turnaround_time_ns"
                ].mean()
            )
        ),
        "avg_excess_waiting_time_ns": (
            float(
                compared[
                    "excess_waiting_time_ns"
                ].mean()
            )
        ),
        "avg_excess_turnaround_time_ns": (
            float(
                compared[
                    "excess_turnaround_time_ns"
                ].mean()
            )
        ),
        "median_excess_turnaround_time_ns": (
            float(
                compared[
                    "excess_turnaround_time_ns"
                ].median()
            )
        ),
        "max_excess_turnaround_time_ns": (
            float(
                compared[
                    "excess_turnaround_time_ns"
                ].max()
            )
        ),
        "min_excess_turnaround_time_ns": (
            float(
                compared[
                    "excess_turnaround_time_ns"
                ].min()
            )
        ),
        "total_excess_turnaround_time_ns": (
            total_excess_latency_ns
        ),
        "positive_excess_turnaround_time_ns": (
            positive_excess_latency_ns
        ),
        "negative_excess_turnaround_time_ns": (
            negative_excess_latency_ns
        ),
        "delayed_probe_count": (
            delayed_probe_count
        ),
        "contention_observed_fraction": (
            delayed_probe_count
            / total_probe_count
            if total_probe_count > 0
            else 0.0
        ),
        "intertenant_signal_observed": (
            delayed_probe_count > 0
        ),
        "victim_only_avg_waiting_time_ns": (
            victim_only_avg_waiting_ns
        ),
        "victim_only_max_waiting_time_ns": (
            victim_only_max_waiting_ns
        ),
        "victim_on_avg_remote_waiting_time_ns": (
            victim_on_avg_waiting_ns
        ),
        "victim_on_max_remote_waiting_time_ns": (
            victim_on_max_waiting_ns
        ),
        "victim_only_duration_ns": (
            victim_only_duration
        ),
        "victim_on_duration_ns": (
            victim_on_duration
        ),
        "victim_slowdown_ns": (
            victim_slowdown_ns
        ),
        "victim_slowdown_ratio": (
            victim_slowdown_ratio
        ),
        "attacker_only_hub_makespan_ns": (
            int(
                attacker_only_architecture
                .hub.current_time_ns
            )
        ),
        "victim_only_hub_makespan_ns": (
            int(
                victim_only_architecture
                .hub.current_time_ns
            )
        ),
        "victim_on_hub_makespan_ns": (
            int(
                victim_on_architecture
                .hub.current_time_ns
            )
        ),
    }


# ============================================================
# Request-level plots
# ============================================================

def save_request_level_plots(
    compared: pd.DataFrame,
    victim_tag: str,
    capacity_name: str,
    output_directory: Path,
) -> None:
    """Save timing traces for one capacity."""

    release_times = compared[
        "request_release_time_ns"
    ]

    plt.figure(
        figsize=(13, 6)
    )

    plt.plot(
        release_times,
        compared[
            "victim_on_turnaround_time_ns"
        ],
        marker="o",
        markersize=2.5,
        linewidth=1,
        label="Victim present",
    )

    plt.plot(
        release_times,
        compared[
            "baseline_turnaround_time_ns"
        ],
        linewidth=1,
        label="Attacker only",
    )

    plt.xlabel(
        "Attacker remote-probe "
        "release time (ns)"
    )

    plt.ylabel(
        "Remote-request "
        "turnaround time (ns)"
    )

    plt.title(
        "Hub capacity — "
        f"{capacity_name}: {victim_tag}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory
        / f"{victim_tag}_timing_trace.png",
        dpi=300,
    )

    plt.close()

    plt.figure(
        figsize=(13, 5)
    )

    plt.plot(
        release_times,
        compared[
            "excess_turnaround_time_ns"
        ],
        marker="o",
        markersize=2.5,
        linewidth=1,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Attacker remote-probe "
        "release time (ns)"
    )

    plt.ylabel(
        "Victim-induced "
        "excess latency (ns)"
    )

    plt.title(
        "Victim-induced delay — "
        f"{capacity_name}: {victim_tag}"
    )

    plt.tight_layout()

    plt.savefig(
        output_directory
        / f"{victim_tag}_excess_latency.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Cross-capacity plots
# ============================================================

def save_comparison_plot(
    summary_dataframe: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Save a workload-by-capacity plot."""

    pivot = (
        summary_dataframe.pivot(
            index="victim_tag",
            columns="capacity_name",
            values=metric,
        )
        .reindex(
            columns=CAPACITY_ORDER
        )
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(14, 6),
    )

    axis.set_xlabel(
        "Victim workload"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.legend(
        title="Hub capacity",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# One workload/capacity experiment
# ============================================================

def run_one_configuration(
    victim_qasm: str,
    capacity_config: dict,
) -> dict:
    """
    Run:
    1. victim only;
    2. attacker only;
    3. victim and attacker together.
    """

    victim_tag = base.safe_tag(
        victim_qasm
    )

    capacity_name = (
        capacity_config[
            "capacity_name"
        ]
    )

    hub_capacity = int(
        capacity_config[
            "hub_capacity"
        ]
    )

    output_directory = (
        OUTPUT_DIR
        / capacity_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Hub-capacity sweep: "
        f"{capacity_name} | "
        f"{victim_qasm} ==="
    )

    # This is the only architectural parameter
    # changed in this sweep.
    base.HUB_MAX_CONCURRENT_TRANSFERS = (
        hub_capacity
    )

    (
        victim_trace,
        victim_mapping,
        victim_num_qubits,
        victim_cross_operations,
    ) = base.extract_static_victim_trace(
        victim_qasm
    )

    victim_schedule = (
        base.schedule_victim_events(
            victim_trace
        )
    )

    window_config = {
        "window_name": (
            capacity_name
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
    # Victim-only control
    # --------------------------------------------------------

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
            victim_schedule
        ),
    )

    victim_only_ground_truth = (
        base.collect_victim_ground_truth(
            victim_only_architecture,
            "victim_only",
        )
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

    attacker_only = (
        base.collect_attacker_observations(
            attacker_only_architecture,
            cross_step_metadata,
            "attacker_only",
        )
    )

    # --------------------------------------------------------
    # Victim and attacker together
    # --------------------------------------------------------

    victim_on_architecture = (
        base.build_architecture(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    merged_schedule = (
        copy.deepcopy(
            victim_schedule
        )
        + copy.deepcopy(
            attacker_schedule
        )
    )

    base.execute_timed_schedule(
        victim_on_architecture,
        merged_schedule,
    )

    victim_present = (
        base.collect_attacker_observations(
            victim_on_architecture,
            cross_step_metadata,
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
            attacker_only,
            victim_present,
        )
    )

    summary = create_summary(
        victim_qasm=victim_qasm,
        capacity_config=capacity_config,
        victim_trace=victim_trace,
        victim_cross_operations=(
            victim_cross_operations
        ),
        schedule_metadata=(
            schedule_metadata
        ),
        attacker_only=attacker_only,
        victim_present=victim_present,
        compared=compared,
        victim_only_ground_truth=(
            victim_only_ground_truth
        ),
        victim_on_ground_truth=(
            victim_on_ground_truth
        ),
        attacker_only_architecture=(
            attacker_only_architecture
        ),
        victim_only_architecture=(
            victim_only_architecture
        ),
        victim_on_architecture=(
            victim_on_architecture
        ),
    )

    # --------------------------------------------------------
    # Save per-configuration files
    # --------------------------------------------------------

    compared.to_csv(
        output_directory
        / (
            f"{victim_tag}_"
            "attacker_observations.csv"
        ),
        index=False,
    )

    victim_only_ground_truth.to_csv(
        output_directory
        / (
            f"{victim_tag}_"
            "victim_only_ground_truth.csv"
        ),
        index=False,
    )

    victim_on_ground_truth.to_csv(
        output_directory
        / (
            f"{victim_tag}_"
            "victim_on_ground_truth.csv"
        ),
        index=False,
    )

    summary_path = (
        output_directory
        / f"{victim_tag}_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
        )

    save_request_level_plots(
        compared,
        victim_tag,
        capacity_name,
        output_directory,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return summary


# ============================================================
# Postprocessing
# ============================================================

def add_capacity_one_retention(
    summary_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize every capacity's timing signal to
    capacity 1 for the same victim workload.
    """

    capacity_one = (
        summary_dataframe[
            summary_dataframe[
                "hub_max_concurrent_transfers"
            ] == 1
        ][
            [
                "victim_tag",
                "total_excess_turnaround_time_ns",
                "avg_excess_turnaround_time_ns",
                "contention_observed_fraction",
            ]
        ]
        .rename(
            columns={
                "total_excess_turnaround_time_ns": (
                    "capacity_1_total_signal_ns"
                ),
                "avg_excess_turnaround_time_ns": (
                    "capacity_1_avg_signal_ns"
                ),
                "contention_observed_fraction": (
                    "capacity_1_contention_fraction"
                ),
            }
        )
    )

    merged = summary_dataframe.merge(
        capacity_one,
        on="victim_tag",
        how="left",
        validate="many_to_one",
    )

    total_denominator = merged[
        "capacity_1_total_signal_ns"
    ].replace(
        0,
        pd.NA,
    )

    average_denominator = merged[
        "capacity_1_avg_signal_ns"
    ].replace(
        0,
        pd.NA,
    )

    coverage_denominator = merged[
        "capacity_1_contention_fraction"
    ].replace(
        0,
        pd.NA,
    )

    merged[
        "total_signal_retention_vs_capacity_1"
    ] = (
        merged[
            "total_excess_turnaround_time_ns"
        ]
        / total_denominator
    )

    merged[
        "avg_signal_retention_vs_capacity_1"
    ] = (
        merged[
            "avg_excess_turnaround_time_ns"
        ]
        / average_denominator
    )

    merged[
        "contention_retention_vs_capacity_1"
    ] = (
        merged[
            "contention_observed_fraction"
        ]
        / coverage_denominator
    )

    return merged


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict] = []

    for capacity_config in CAPACITY_CONFIGS:
        for victim_qasm in base.VICTIM_QASMS:
            summary = (
                run_one_configuration(
                    victim_qasm,
                    capacity_config,
                )
            )

            summaries.append(
                summary
            )

    summary_dataframe = pd.DataFrame(
        summaries
    )

    summary_dataframe = (
        add_capacity_one_retention(
            summary_dataframe
        )
    )

    summary_dataframe[
        "capacity_name"
    ] = pd.Categorical(
        summary_dataframe[
            "capacity_name"
        ],
        categories=CAPACITY_ORDER,
        ordered=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "capacity_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = (
        OUTPUT_DIR
        / "hub_capacity_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    schedule_summary = (
        summary_dataframe[
            [
                "capacity_name",
                "hub_max_concurrent_transfers",
                "observation_duration_ns",
                "probe_round_period_ns",
                "total_attacker_remote_requests",
                "first_remote_probe_release_ns",
                "last_remote_probe_release_ns",
                "realized_remote_probe_rate_per_us",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "hub_max_concurrent_transfers"
        )
        .reset_index(drop=True)
    )

    schedule_path = (
        OUTPUT_DIR
        / "hub_capacity_schedule_summary.csv"
    )

    schedule_summary.to_csv(
        schedule_path,
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "avg_excess_turnaround_time_ns"
        ),
        ylabel=(
            "Average victim-induced latency (ns)"
        ),
        title=(
            "Hub Capacity: "
            "Average Victim-Induced Latency"
        ),
        filename=(
            "hub_capacity_"
            "avg_excess_latency.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "total_excess_turnaround_time_ns"
        ),
        ylabel=(
            "Cumulative victim-induced latency (ns)"
        ),
        title=(
            "Hub Capacity: "
            "Total Collected Timing Signal"
        ),
        filename=(
            "hub_capacity_"
            "total_excess_latency.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "contention_observed_fraction"
        ),
        ylabel=(
            "Fraction of attacker probes delayed"
        ),
        title=(
            "Hub Capacity: "
            "Contention Observation Rate"
        ),
        filename=(
            "hub_capacity_"
            "contention_fraction.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "victim_slowdown_ratio"
        ),
        ylabel=(
            "Victim completion-time ratio"
        ),
        title=(
            "Hub Capacity: "
            "Victim Slowdown"
        ),
        filename=(
            "hub_capacity_"
            "victim_slowdown.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "baseline_avg_waiting_time_ns"
        ),
        ylabel=(
            "Attacker-only average waiting time (ns)"
        ),
        title=(
            "Hub Capacity: "
            "Attacker Self-Contention"
        ),
        filename=(
            "hub_capacity_"
            "attacker_self_wait.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "total_signal_retention_vs_capacity_1"
        ),
        ylabel=(
            "Timing-signal retention "
            "relative to capacity 1"
        ),
        title=(
            "Hub Capacity: "
            "Signal Retention"
        ),
        filename=(
            "hub_capacity_"
            "signal_retention.png"
        ),
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    display_columns = [
        "victim_tag",
        "capacity_name",
        "hub_max_concurrent_transfers",
        "total_attacker_remote_requests",
        "baseline_avg_waiting_time_ns",
        "avg_excess_turnaround_time_ns",
        "total_excess_turnaround_time_ns",
        "max_excess_turnaround_time_ns",
        "delayed_probe_count",
        "contention_observed_fraction",
        "total_signal_retention_vs_capacity_1",
        "victim_only_duration_ns",
        "victim_on_duration_ns",
        "victim_slowdown_ratio",
        "intertenant_signal_observed",
    ]

    print(
        "\n=== Combined hub-capacity "
        "summary ==="
    )

    print(
        summary_dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )

    overall_summary = (
        summary_dataframe.groupby(
            [
                "capacity_name",
                "hub_max_concurrent_transfers",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            workload_count=(
                "victim_tag",
                "count",
            ),
            average_excess_latency_ns=(
                "avg_excess_turnaround_time_ns",
                "mean",
            ),
            average_total_signal_ns=(
                "total_excess_turnaround_time_ns",
                "mean",
            ),
            average_contention_fraction=(
                "contention_observed_fraction",
                "mean",
            ),
            workloads_with_signal=(
                "intertenant_signal_observed",
                "sum",
            ),
            average_signal_retention=(
                "total_signal_retention_vs_capacity_1",
                "mean",
            ),
            average_victim_slowdown_ratio=(
                "victim_slowdown_ratio",
                "mean",
            ),
            maximum_victim_slowdown_ratio=(
                "victim_slowdown_ratio",
                "max",
            ),
            attacker_self_wait_ns=(
                "baseline_avg_waiting_time_ns",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "hub_max_concurrent_transfers"
        )
    )

    overall_path = (
        OUTPUT_DIR
        / "hub_capacity_overall_summary.csv"
    )

    overall_summary.to_csv(
        overall_path,
        index=False,
    )

    print(
        "\n=== Overall architectural "
        "capacity summary ==="
    )

    print(
        overall_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved all results to: "
        f"{OUTPUT_DIR}"
    )

    print(
        "Combined summary: "
        f"{summary_path}"
    )

    print(
        "Overall summary: "
        f"{overall_path}"
    )

    print(
        "Schedule summary: "
        f"{schedule_path}"
    )


if __name__ == "__main__":
    main()
