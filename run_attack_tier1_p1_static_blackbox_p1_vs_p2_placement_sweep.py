#!/usr/bin/env python3
"""
run_attack_tier1_static_blackbox_p1_vs_p2_placement_sweep.py

Final placement experiment: P1 disjoint versus P2 one-module overlap.

Dependency:
    run_attack_tier1_p1_static_blackbox_observation_window_sweep.py

Fixed attack configuration:
- Probe 3 light-periodic probe
- uniform 420 ns spacing
- 20,000 ns observation window
- exact window-start estimate
- static-distributed victim execution

Placement/capacity matrix:
- P1 disjoint, capacity 1
- P1 disjoint, capacity 2
- P2 one-module overlap, capacity 1
- P2 one-module overlap, capacity 2

P1:
    victim   = module_0, module_1, module_2
    attacker = module_3, module_4

P2:
    victim   = module_0, module_1, module_2
    attacker = module_2, module_3
    overlap  = module_2

The 2x2 design separates shared-hub serialization from direct module overlap.

Outputs:
    blackbox_window_results/placement/
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
)

import run_atack_tier1_p1_static_blackbox_observation_window_sweep as base


# ============================================================
# Fixed configuration
# ============================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "placement"
)

VICTIM_TRUE_START_NS = 1_000
ATTACKER_ESTIMATED_START_NS = 1_000
OBSERVATION_DURATION_NS = 20_000
PROBE_ROUND_PERIOD_NS = 420

LINK_LATENCY_NS = 10
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80
VICTIM_EVENT_TICK_NS = 5


EXPERIMENT_CONFIGS = [
    {
        "config_name": (
            "p1_disjoint_capacity_1"
        ),
        "placement_name": (
            "P1_disjoint"
        ),
        "placement_short": "P1",
        "hub_capacity": 1,
        "attacker_modules": [
            "module_3",
            "module_4",
        ],
        "shared_modules": [],
        "needs_anchor": False,
    },
    {
        "config_name": (
            "p1_disjoint_capacity_2"
        ),
        "placement_name": (
            "P1_disjoint"
        ),
        "placement_short": "P1",
        "hub_capacity": 2,
        "attacker_modules": [
            "module_3",
            "module_4",
        ],
        "shared_modules": [],
        "needs_anchor": False,
    },
    {
        "config_name": (
            "p2_overlap_capacity_1"
        ),
        "placement_name": (
            "P2_one_module_overlap"
        ),
        "placement_short": "P2",
        "hub_capacity": 1,
        "attacker_modules": [
            "module_2",
            "module_3",
        ],
        "shared_modules": [
            "module_2",
        ],
        "needs_anchor": True,
    },
    {
        "config_name": (
            "p2_overlap_capacity_2"
        ),
        "placement_name": (
            "P2_one_module_overlap"
        ),
        "placement_short": "P2",
        "hub_capacity": 2,
        "attacker_modules": [
            "module_2",
            "module_3",
        ],
        "shared_modules": [
            "module_2",
        ],
        "needs_anchor": True,
    },
]

CONFIG_ORDER = [
    config["config_name"]
    for config in EXPERIMENT_CONFIGS
]


# Configure reusable helpers from the
# observation-window experiment.
base.VICTIM_TRUE_START_NS = (
    VICTIM_TRUE_START_NS
)

base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
    ATTACKER_ESTIMATED_START_NS
)

base.PROBE_ROUND_PERIOD_NS = (
    PROBE_ROUND_PERIOD_NS
)

base.OUTPUT_DIR = OUTPUT_DIR
base.WINDOW_ORDER = CONFIG_ORDER


# ============================================================
# Placement helpers
# ============================================================

def attacker_qubit_map(
    config: dict,
) -> dict[int, str]:
    """
    Map q0/q1 to one attacker module and
    q2/q3 to the other.
    """

    (
        first_module,
        second_module,
    ) = config["attacker_modules"]

    return {
        0: first_module,
        1: first_module,
        2: second_module,
        3: second_module,
    }


def configure_base(
    config: dict,
) -> dict[int, str]:
    """
    Set the two configuration-dependent
    simulator controls.
    """

    mapping = attacker_qubit_map(
        config
    )

    # The imported schedule builder calls
    # attacker_qubit_map() dynamically.
    base.attacker_qubit_map = (
        lambda mapping=mapping: dict(
            mapping
        )
    )

    base.HUB_MAX_CONCURRENT_TRANSFERS = int(
        config["hub_capacity"]
    )

    return mapping


def build_architecture(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    config: dict,
) -> tuple[
    FiveModuleLocalModularSuperconductingDQC,
    int | None,
]:
    """
    Build one P1/P2 and
    capacity-1/capacity-2 architecture.
    """

    attacker_mapping = (
        attacker_qubit_map(
            config
        )
    )

    combined_mapping = dict(
        victim_mapping
    )

    for attacker_qubit, module in (
        attacker_mapping.items()
    ):
        combined_mapping[
            victim_num_qubits
            + attacker_qubit
        ] = module

    # P2 otherwise leaves module_4 absent.
    # This unused anchor preserves the
    # five-module validator without creating
    # any event or request.
    anchor_qubit = None

    if config["needs_anchor"]:
        anchor_qubit = (
            victim_num_qubits
            + len(attacker_mapping)
        )

        combined_mapping[
            anchor_qubit
        ] = "module_4"

    expected_modules = {
        f"module_{index}"
        for index in range(5)
    }

    if (
        set(combined_mapping.values())
        != expected_modules
    ):
        raise RuntimeError(
            "Combined mapping does not cover "
            "all five modules: "
            f"{sorted(set(combined_mapping.values()))}"
        )

    architecture = (
        FiveModuleLocalModularSuperconductingDQC(
            qubit_to_module=(
                combined_mapping
            ),
            link_latency_ns=(
                LINK_LATENCY_NS
            ),
            hub_max_concurrent_transfers=int(
                config["hub_capacity"]
            ),
            hub_setup_latency_ns=(
                HUB_SETUP_LATENCY_NS
            ),
            hub_transfer_latency_ns=(
                HUB_TRANSFER_LATENCY_NS
            ),
            event_tick_ns=(
                VICTIM_EVENT_TICK_NS
            ),
        )
    )

    return (
        architecture,
        anchor_qubit,
    )


# ============================================================
# Placement-safe request collection
# ============================================================

def collect_attacker_observations(
    architecture:
    FiveModuleLocalModularSuperconductingDQC,
    cross_step_metadata:
    dict[int, dict],
    run_type: str,
) -> pd.DataFrame:
    """
    Collect attacker requests by trace-step
    identity.

    Source-module filtering cannot be used
    under P2 because module_2 belongs to
    both tenants.
    """

    rows = []

    for request in (
        architecture.hub.completed_requests
    ):
        metadata = (
            cross_step_metadata.get(
                request.original_event.step
            )
        )

        if metadata is None:
            continue

        rows.append(
            {
                "run_type": run_type,
                **metadata,
                "actual_arrival_time_ns": (
                    request.arrival_time_ns
                ),
                "service_start_time_ns": (
                    request.start_time_ns
                ),
                "completion_time_ns": (
                    request.end_time_ns
                ),
                "service_time_ns": (
                    request.service_time_ns
                ),
                "waiting_time_ns": (
                    request.waiting_time_ns
                ),
                "turnaround_time_ns": (
                    request.turnaround_time_ns
                ),
                "source_module": (
                    request.source_module
                ),
                "target_modules": ",".join(
                    request.target_modules
                ),
                "architecture_request_id": (
                    request.request_id
                ),
            }
        )

    observations = pd.DataFrame(
        rows
    )

    if observations.empty:
        raise RuntimeError(
            "No attacker requests completed "
            f"for {run_type}."
        )

    return (
        observations
        .sort_values(
            "attacker_request_id"
        )
        .reset_index(drop=True)
    )


def collect_victim_ground_truth(
    architecture:
    FiveModuleLocalModularSuperconductingDQC,
    cross_step_metadata:
    dict[int, dict],
    run_type: str,
) -> pd.DataFrame:
    """
    Collect victim requests by excluding
    all known attacker trace steps.
    """

    attacker_steps = set(
        cross_step_metadata
    )

    rows = []

    victim_requests = [
        request
        for request
        in architecture.hub.completed_requests
        if request.original_event.step
        not in attacker_steps
    ]

    for victim_request_id, request in enumerate(
        victim_requests
    ):
        rows.append(
            {
                "run_type": run_type,
                "victim_remote_event_id": (
                    victim_request_id
                ),
                "victim_trace_step": (
                    request.original_event.step
                ),
                "op_name": (
                    request.original_event.op_name
                ),
                "qubits": ",".join(
                    str(qubit)
                    for qubit
                    in request.original_event.qubits
                ),
                "source_module": (
                    request.source_module
                ),
                "target_modules": ",".join(
                    request.target_modules
                ),
                "arrival_time_ns": (
                    request.arrival_time_ns
                ),
                "service_start_time_ns": (
                    request.start_time_ns
                ),
                "completion_time_ns": (
                    request.end_time_ns
                ),
                "waiting_time_ns": (
                    request.waiting_time_ns
                ),
                "turnaround_time_ns": (
                    request.turnaround_time_ns
                ),
                "service_time_ns": (
                    request.service_time_ns
                ),
            }
        )

    columns = [
        "run_type",
        "victim_remote_event_id",
        "victim_trace_step",
        "op_name",
        "qubits",
        "source_module",
        "target_modules",
        "arrival_time_ns",
        "service_start_time_ns",
        "completion_time_ns",
        "waiting_time_ns",
        "turnaround_time_ns",
        "service_time_ns",
    ]

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        )
        .sort_values(
            [
                "arrival_time_ns",
                "victim_remote_event_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# One experiment
# ============================================================

def run_one_configuration(
    victim_qasm: str,
    config: dict,
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

    config_name = (
        config["config_name"]
    )

    output_directory = (
        OUTPUT_DIR
        / config_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Placement sweep: "
        f"{config_name} | "
        f"{victim_qasm} ==="
    )

    attacker_mapping = configure_base(
        config
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
            config_name
        ),
        "observation_duration_ns": (
            OBSERVATION_DURATION_NS
        ),
    }

    (
        attacker_schedule,
        schedule_attacker_mapping,
        cross_step_metadata,
        schedule_metadata,
    ) = (
        base.build_observation_window_schedule(
            window_config,
            victim_num_qubits,
        )
    )

    if (
        schedule_attacker_mapping
        != attacker_mapping
    ):
        raise RuntimeError(
            "Attacker schedule mapping "
            "does not match placement mapping."
        )

    # --------------------------------------------------------
    # Victim-only control
    # --------------------------------------------------------

    (
        victim_only_architecture,
        anchor_qubit,
    ) = build_architecture(
        victim_mapping,
        victim_num_qubits,
        config,
    )

    base.execute_timed_schedule(
        victim_only_architecture,
        copy.deepcopy(
            victim_schedule
        ),
    )

    victim_only_ground_truth = (
        collect_victim_ground_truth(
            victim_only_architecture,
            cross_step_metadata,
            "victim_only",
        )
    )

    # --------------------------------------------------------
    # Attacker-only calibration
    # --------------------------------------------------------

    (
        attacker_only_architecture,
        _,
    ) = build_architecture(
        victim_mapping,
        victim_num_qubits,
        config,
    )

    base.execute_timed_schedule(
        attacker_only_architecture,
        copy.deepcopy(
            attacker_schedule
        ),
    )

    attacker_only = (
        collect_attacker_observations(
            attacker_only_architecture,
            cross_step_metadata,
            "attacker_only",
        )
    )

    # --------------------------------------------------------
    # Victim and attacker together
    # --------------------------------------------------------

    (
        victim_on_architecture,
        _,
    ) = build_architecture(
        victim_mapping,
        victim_num_qubits,
        config,
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
        collect_attacker_observations(
            victim_on_architecture,
            cross_step_metadata,
            "victim_present",
        )
    )

    victim_on_ground_truth = (
        collect_victim_ground_truth(
            victim_on_architecture,
            cross_step_metadata,
            "victim_present",
        )
    )

    compared = (
        base.compare_attacker_runs(
            attacker_only,
            victim_present,
        )
    )

    summary = base.create_summary(
        victim_qasm=victim_qasm,
        window_config=window_config,
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

    # Replace the base script's P1-specific
    # labels with the actual placement.
    summary.update(
        {
            "knob": "placement",
            "config_name": config_name,
            "placement": (
                config["placement_name"]
            ),
            "placement_name": (
                config["placement_name"]
            ),
            "placement_short": (
                config["placement_short"]
            ),
            "hub_max_concurrent_transfers": int(
                config["hub_capacity"]
            ),
            "victim_modules": (
                "module_0,module_1,module_2"
            ),
            "attacker_modules": ",".join(
                config["attacker_modules"]
            ),
            "shared_modules": ",".join(
                config["shared_modules"]
            ),
            "shared_module_count": len(
                config["shared_modules"]
            ),
            "direct_module_overlap": bool(
                config["shared_modules"]
            ),
            "attacker_qubit_mapping": (
                json.dumps(
                    attacker_mapping,
                    sort_keys=True,
                )
            ),
            "dummy_anchor_qubit": (
                anchor_qubit
            ),
            "dummy_anchor_module": (
                "module_4"
                if anchor_qubit is not None
                else None
            ),
            "intertenant_signal_observed": bool(
                summary[
                    "delayed_probe_count"
                ] > 0
            ),
        }
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

    base.save_request_level_plots(
        compared,
        victim_tag,
        config_name,
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
# Cross-configuration metrics
# ============================================================

def add_comparison_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare P2 with P1 and
    capacity 2 with capacity 1.
    """

    result = dataframe.copy()

    # --------------------------------------------------------
    # P2 versus P1 at the same capacity
    # --------------------------------------------------------

    p1 = (
        result[
            result[
                "placement_short"
            ] == "P1"
        ][
            [
                "victim_tag",
                "hub_max_concurrent_transfers",
                "total_excess_turnaround_time_ns",
                "contention_observed_fraction",
            ]
        ]
        .rename(
            columns={
                "total_excess_turnaround_time_ns": (
                    "p1_same_capacity_"
                    "total_signal_ns"
                ),
                "contention_observed_fraction": (
                    "p1_same_capacity_"
                    "contention_fraction"
                ),
            }
        )
    )

    result = result.merge(
        p1,
        on=[
            "victim_tag",
            "hub_max_concurrent_transfers",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "signal_difference_vs_"
        "p1_same_capacity_ns"
    ] = (
        result[
            "total_excess_turnaround_time_ns"
        ]
        - result[
            "p1_same_capacity_total_signal_ns"
        ]
    )

    result[
        "contention_difference_vs_"
        "p1_same_capacity"
    ] = (
        result[
            "contention_observed_fraction"
        ]
        - result[
            "p1_same_capacity_"
            "contention_fraction"
        ]
    )

    result[
        "placement_restores_signal_over_p1"
    ] = (
        result[
            "intertenant_signal_observed"
        ]
        & (
            result[
                "p1_same_capacity_"
                "total_signal_ns"
            ] <= 0
        )
    )

    # --------------------------------------------------------
    # Capacity 2 versus capacity 1
    # for the same placement
    # --------------------------------------------------------

    capacity_one = (
        result[
            result[
                "hub_max_concurrent_transfers"
            ] == 1
        ][
            [
                "victim_tag",
                "placement_short",
                "total_excess_turnaround_time_ns",
                "contention_observed_fraction",
            ]
        ]
        .rename(
            columns={
                "total_excess_turnaround_time_ns": (
                    "capacity_1_total_signal_ns"
                ),
                "contention_observed_fraction": (
                    "capacity_1_"
                    "contention_fraction"
                ),
            }
        )
    )

    result = result.merge(
        capacity_one,
        on=[
            "victim_tag",
            "placement_short",
        ],
        how="left",
        validate="many_to_one",
    )

    result[
        "signal_retention_vs_capacity_1"
    ] = (
        result[
            "total_excess_turnaround_time_ns"
        ]
        / result[
            "capacity_1_total_signal_ns"
        ].where(
            result[
                "capacity_1_total_signal_ns"
            ] != 0
        )
    )

    result[
        "contention_retention_vs_capacity_1"
    ] = (
        result[
            "contention_observed_fraction"
        ]
        / result[
            "capacity_1_contention_fraction"
        ].where(
            result[
                "capacity_1_contention_fraction"
            ] != 0
        )
    )

    return result


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
            config,
        )
        for config in EXPERIMENT_CONFIGS
        for victim_qasm in base.VICTIM_QASMS
    ]

    summary_dataframe = (
        add_comparison_metrics(
            pd.DataFrame(
                summaries
            )
        )
    )

    summary_dataframe[
        "config_name"
    ] = pd.Categorical(
        summary_dataframe[
            "config_name"
        ],
        categories=CONFIG_ORDER,
        ordered=True,
    )

    summary_dataframe[
        "window_name"
    ] = pd.Categorical(
        summary_dataframe[
            "window_name"
        ],
        categories=CONFIG_ORDER,
        ordered=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "config_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save summaries
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "placement_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    schedule_summary = (
        summary_dataframe[
            [
                "config_name",
                "placement_name",
                "hub_max_concurrent_transfers",
                "victim_modules",
                "attacker_modules",
                "shared_modules",
                "shared_module_count",
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
            "config_name"
        )
        .reset_index(drop=True)
    )

    schedule_path = (
        OUTPUT_DIR
        / "placement_schedule_summary.csv"
    )

    schedule_summary.to_csv(
        schedule_path,
        index=False,
    )

    overall_summary = (
        summary_dataframe.groupby(
            [
                "config_name",
                "placement_name",
                "hub_max_concurrent_transfers",
                "shared_module_count",
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
            average_signal_retention_vs_capacity_1=(
                "signal_retention_vs_capacity_1",
                "mean",
            ),
            average_signal_difference_vs_p1_ns=(
                "signal_difference_vs_"
                "p1_same_capacity_ns",
                "mean",
            ),
            workloads_restored_over_p1=(
                "placement_restores_signal_over_p1",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "config_name"
        )
        .reset_index(drop=True)
    )

    overall_path = (
        OUTPUT_DIR
        / "placement_overall_summary.csv"
    )

    overall_summary.to_csv(
        overall_path,
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    base.save_comparison_plot(
        summary_dataframe,
        "avg_excess_turnaround_time_ns",
        "Average victim-induced latency (ns)",
        (
            "P1 vs P2: Average "
            "Victim-Induced Latency"
        ),
        "placement_avg_excess_latency.png",
    )

    base.save_comparison_plot(
        summary_dataframe,
        "total_excess_turnaround_time_ns",
        (
            "Cumulative victim-induced "
            "latency (ns)"
        ),
        (
            "P1 vs P2: Total Collected "
            "Timing Signal"
        ),
        "placement_total_excess_latency.png",
    )

    base.save_comparison_plot(
        summary_dataframe,
        "contention_observed_fraction",
        (
            "Fraction of attacker "
            "probes delayed"
        ),
        (
            "P1 vs P2: Contention "
            "Observation Rate"
        ),
        "placement_contention_fraction.png",
    )

    base.save_comparison_plot(
        summary_dataframe,
        "victim_slowdown_ratio",
        "Victim completion-time ratio",
        "P1 vs P2: Victim Slowdown",
        "placement_victim_slowdown.png",
    )

    base.save_comparison_plot(
        summary_dataframe,
        "baseline_avg_waiting_time_ns",
        (
            "Attacker-only average "
            "waiting time (ns)"
        ),
        (
            "P1 vs P2: Attacker "
            "Self-Contention"
        ),
        "placement_attacker_self_wait.png",
    )

    base.save_comparison_plot(
        summary_dataframe,
        "signal_retention_vs_capacity_1",
        (
            "Signal retention relative "
            "to capacity 1"
        ),
        (
            "P1 vs P2: Capacity-2 "
            "Signal Retention"
        ),
        (
            "placement_signal_retention_"
            "vs_serialized.png"
        ),
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    display_columns = [
        "victim_tag",
        "config_name",
        "hub_max_concurrent_transfers",
        "shared_modules",
        "baseline_avg_waiting_time_ns",
        "avg_excess_turnaround_time_ns",
        "total_excess_turnaround_time_ns",
        "max_excess_turnaround_time_ns",
        "delayed_probe_count",
        "contention_observed_fraction",
        "signal_retention_vs_capacity_1",
        (
            "signal_difference_vs_"
            "p1_same_capacity_ns"
        ),
        "placement_restores_signal_over_p1",
        "victim_slowdown_ratio",
    ]

    print(
        "\n=== Combined P1-vs-P2 "
        "placement summary ==="
    )

    print(
        summary_dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== Overall placement/"
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
