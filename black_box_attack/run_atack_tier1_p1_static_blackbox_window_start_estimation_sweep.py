#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_window_start_estimation_sweep.py

Knob 5: deterministic window-start estimation error.

Dependency:
    run_attack_tier1_p1_static_blackbox_observation_window_sweep.py

Fixed:
- Probe 3
- uniform 420 ns spacing
- 20,000 ns observation duration
- P1 disjoint placement
- one serialized shared hub-service slot

Varied:
- {-10, -5, -2.5, -1, 0, +1, +2.5, +5, +10} us start error

Negative error means the attacker starts early.
Positive error means the attacker starts late.

Outputs:
blackbox_window_results/window_start_estimation/
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import run_atack_tier1_p1_static_blackbox_observation_window_sweep as base


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "window_start_estimation"
)

# Shift the entire experiment epoch so that a -10 us
# estimate remains non-negative.
#
# Relative victim/attacker timing is unchanged.
VICTIM_TRUE_START_NS = 20_000

OBSERVATION_DURATION_NS = 20_000
PROBE_ROUND_PERIOD_NS = 420

START_CONFIGS = [
    {
        "start_name": "start_m10us",
        "start_error_ns": -10_000,
    },
    {
        "start_name": "start_m5us",
        "start_error_ns": -5_000,
    },
    {
        "start_name": "start_m2p5us",
        "start_error_ns": -2_500,
    },
    {
        "start_name": "start_m1us",
        "start_error_ns": -1_000,
    },
    {
        "start_name": "start_exact",
        "start_error_ns": 0,
    },
    {
        "start_name": "start_p1us",
        "start_error_ns": 1_000,
    },
    {
        "start_name": "start_p2p5us",
        "start_error_ns": 2_500,
    },
    {
        "start_name": "start_p5us",
        "start_error_ns": 5_000,
    },
    {
        "start_name": "start_p10us",
        "start_error_ns": 10_000,
    },
]

START_ORDER = [
    config["start_name"]
    for config in START_CONFIGS
]

# Configure the imported helper functions.
base.VICTIM_TRUE_START_NS = (
    VICTIM_TRUE_START_NS
)

base.PROBE_ROUND_PERIOD_NS = (
    PROBE_ROUND_PERIOD_NS
)


# ============================================================
# Summary helpers
# ============================================================

def final_victim_completion_ns(
    dataframe: pd.DataFrame,
) -> float:
    """Return the absolute final victim completion time."""

    if dataframe.empty:
        return float(
            VICTIM_TRUE_START_NS
        )

    return float(
        dataframe[
            "completion_time_ns"
        ].max()
    )


def create_summary(
    *,
    victim_qasm: str,
    start_config: dict,
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
    """Create one victim/start-error result summary."""

    victim_only_completion = (
        final_victim_completion_ns(
            victim_only_ground_truth
        )
    )

    victim_on_completion = (
        final_victim_completion_ns(
            victim_on_ground_truth
        )
    )

    victim_only_duration = (
        victim_only_completion
        - VICTIM_TRUE_START_NS
    )

    victim_on_duration = (
        victim_on_completion
        - VICTIM_TRUE_START_NS
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

    estimated_start = int(
        schedule_metadata[
            "observation_window_start_ns"
        ]
    )

    estimated_end = int(
        schedule_metadata[
            "observation_window_end_ns"
        ]
    )

    window_overlap = max(
        0.0,
        min(
            estimated_end,
            victim_only_completion,
        )
        - max(
            estimated_start,
            VICTIM_TRUE_START_NS,
        ),
    )

    release_times = compared[
        "request_release_time_ns"
    ]

    pre_victim_count = int(
        (
            release_times
            < VICTIM_TRUE_START_NS
        ).sum()
    )

    useful_count = int(
        (
            (
                release_times
                >= VICTIM_TRUE_START_NS
            )
            & (
                release_times
                <= victim_only_completion
            )
        ).sum()
    )

    post_victim_count = int(
        (
            release_times
            > victim_only_completion
        ).sum()
    )

    total_probe_count = int(
        len(compared)
    )

    useful_fraction = (
        useful_count
        / total_probe_count
        if total_probe_count
        else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "victim_tag": base.safe_tag(
            victim_qasm
        ),
        "knob": (
            "window_start_estimation"
        ),
        "start_name": (
            start_config[
                "start_name"
            ]
        ),
        "start_error_ns": int(
            start_config[
                "start_error_ns"
            ]
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
        "workload_type": (
            "static_distributed"
        ),
        "placement": "P1_disjoint",
        "threat_model": (
            "blackbox_with_coarse_"
            "window_knowledge"
        ),
        "hub_max_concurrent_transfers": (
            base.HUB_MAX_CONCURRENT_TRANSFERS
        ),
        "victim_true_start_ns": (
            VICTIM_TRUE_START_NS
        ),
        "within_round_event_spacing_ns": (
            base.WITHIN_ROUND_EVENT_SPACING_NS
        ),
        "estimated_window_start_ns": (
            estimated_start
        ),
        "estimated_window_end_ns": (
            estimated_end
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
        "victim_total_events": len(
            victim_trace
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
        "median_excess_"
        "turnaround_time_ns": (
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
        "total_excess_turnaround_time_ns": (
            float(
                compared[
                    "excess_turnaround_time_ns"
                ].sum()
            )
        ),
        "delayed_probe_count": int(
            compared[
                "victim_contention_observed"
            ].sum()
        ),
        "contention_observed_fraction": (
            float(
                compared[
                    "victim_contention_observed"
                ].mean()
            )
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
        "window_victim_overlap_ns_"
        "evaluator_only": (
            window_overlap
        ),
        "pre_victim_probe_count_"
        "evaluator_only": (
            pre_victim_count
        ),
        "useful_probe_count_"
        "evaluator_only": (
            useful_count
        ),
        "post_victim_probe_count_"
        "evaluator_only": (
            post_victim_count
        ),
        "useful_probe_fraction_"
        "evaluator_only": (
            useful_fraction
        ),
        "attacker_only_hub_makespan_ns": int(
            attacker_only_architecture
            .hub.current_time_ns
        ),
        "victim_only_hub_makespan_ns": int(
            victim_only_architecture
            .hub.current_time_ns
        ),
        "victim_on_hub_makespan_ns": int(
            victim_on_architecture
            .hub.current_time_ns
        ),
    }


# ============================================================
# Plotting
# ============================================================

def save_request_level_plots(
    compared: pd.DataFrame,
    victim_tag: str,
    start_name: str,
    output_directory: Path,
) -> None:
    """Save request-level timing plots."""

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

    plt.axvline(
        VICTIM_TRUE_START_NS,
        linestyle="--",
        linewidth=1,
        label="True victim start",
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
        "Window-start estimate — "
        f"{start_name}: {victim_tag}"
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

    plt.axvline(
        VICTIM_TRUE_START_NS,
        linestyle="--",
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
        f"{start_name}: {victim_tag}"
    )

    plt.tight_layout()

    plt.savefig(
        output_directory
        / f"{victim_tag}_excess_latency.png",
        dpi=300,
    )

    plt.close()


def save_comparison_plot(
    summary_dataframe: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Save a workload-by-start-error plot."""

    pivot = (
        summary_dataframe.pivot(
            index="victim_tag",
            columns="start_name",
            values=metric,
        )
        .reindex(
            columns=START_ORDER
        )
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(15, 6),
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
        title="Start estimate",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# One experiment
# ============================================================

def run_one_configuration(
    victim_qasm: str,
    start_config: dict,
) -> dict:
    """Run one victim/start-error configuration."""

    victim_tag = base.safe_tag(
        victim_qasm
    )

    start_name = (
        start_config[
            "start_name"
        ]
    )

    start_error_ns = int(
        start_config[
            "start_error_ns"
        ]
    )

    estimated_start_ns = (
        VICTIM_TRUE_START_NS
        + start_error_ns
    )

    output_directory = (
        OUTPUT_DIR / start_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Window-start estimation: "
        f"{start_name} | "
        f"{victim_qasm} ==="
    )

    # This is the only attacker parameter
    # changed by this sweep.
    base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
        estimated_start_ns
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
        "window_name": start_name,
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

    base.execute_timed_schedule(
        victim_on_architecture,
        copy.deepcopy(
            victim_schedule
        )
        + copy.deepcopy(
            attacker_schedule
        ),
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
        start_config=start_config,
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
    # Save outputs
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

    with (
        output_directory
        / f"{victim_tag}_summary.json"
    ).open(
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
        start_name,
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
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = [
        run_one_configuration(
            victim_qasm,
            start_config,
        )
        for start_config in START_CONFIGS
        for victim_qasm in base.VICTIM_QASMS
    ]

    summary_dataframe = (
        pd.DataFrame(
            summaries
        )
    )

    summary_dataframe[
        "start_name"
    ] = pd.Categorical(
        summary_dataframe[
            "start_name"
        ],
        categories=START_ORDER,
        ordered=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "start_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = (
        OUTPUT_DIR
        / "window_start_estimation_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    schedule_summary = (
        summary_dataframe[
            [
                "start_name",
                "start_error_ns",
                "estimated_window_start_ns",
                "estimated_window_end_ns",
                "observation_duration_ns",
                "total_attacker_remote_requests",
                "first_remote_probe_release_ns",
                "last_remote_probe_release_ns",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "start_error_ns"
        )
        .reset_index(drop=True)
    )

    schedule_path = (
        OUTPUT_DIR
        / (
            "window_start_estimation_"
            "schedule_summary.csv"
        )
    )

    schedule_summary.to_csv(
        schedule_path,
        index=False,
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "avg_excess_turnaround_time_ns"
        ),
        ylabel=(
            "Average victim-induced latency (ns)"
        ),
        title=(
            "Window-Start Error: "
            "Average Victim-Induced Latency"
        ),
        filename=(
            "window_start_"
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
            "Window-Start Error: "
            "Total Collected Timing Signal"
        ),
        filename=(
            "window_start_"
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
            "Window-Start Error: "
            "Contention Observation Rate"
        ),
        filename=(
            "window_start_"
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
            "Window-Start Error: "
            "Victim Slowdown"
        ),
        filename=(
            "window_start_"
            "victim_slowdown.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "useful_probe_fraction_"
            "evaluator_only"
        ),
        ylabel=(
            "Fraction of probes released "
            "during victim activity"
        ),
        title=(
            "Window-Start Error: "
            "Useful Probe Fraction"
        ),
        filename=(
            "window_start_"
            "useful_probe_fraction.png"
        ),
    )

    display_columns = [
        "victim_tag",
        "start_name",
        "start_error_ns",
        "total_attacker_remote_requests",
        "baseline_avg_waiting_time_ns",
        "avg_excess_turnaround_time_ns",
        "total_excess_turnaround_time_ns",
        "max_excess_turnaround_time_ns",
        "delayed_probe_count",
        "contention_observed_fraction",
        "victim_slowdown_ratio",
        "pre_victim_probe_count_evaluator_only",
        "useful_probe_count_evaluator_only",
        "post_victim_probe_count_evaluator_only",
        "useful_probe_fraction_evaluator_only",
    ]

    print(
        "\n=== Combined window-start "
        "estimation summary ==="
    )

    print(
        summary_dataframe[
            display_columns
        ].to_string(index=False)
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
        "Schedule summary: "
        f"{schedule_path}"
    )


if __name__ == "__main__":
    main()
