#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_window_baseline.py

Black-box attack with coarse victim timing knowledge.

Threat model
------------
The attacker knows:
- approximately when the victim job starts;
- an approximate observation window in which the victim is active.

The attacker does NOT know:
- victim QASM;
- victim operation count;
- victim cross-module operation count;
- victim event counter;
- victim stage boundaries;
- exact victim communication times.

Important architectural choice
------------------------------
HUB_MAX_CONCURRENT_TRANSFERS = 1 creates one genuinely shared hub service.
Both victim and attacker requests therefore compete for the same resource.

Set HUB_MAX_CONCURRENT_TRANSFERS = 2 later as a negative control. Under the
current P1 disjoint placement, two slots allow one victim transfer and one
attacker transfer to proceed simultaneously without queueing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)


# ============================================================
# Experiment configuration
# ============================================================

VICTIM_MODULES = ["module_0", "module_1", "module_2"]
ATTACKER_MODULES = ["module_3", "module_4"]

VICTIM_QASMS = [
    "square_root_n18.qasm",
    "qft_n18.qasm",
    "bv_n19.qasm",
    "dnn_n16.qasm",
    "sat_n11.qasm",
]

OUTPUT_DIR = Path("blackbox_window_results") / "baseline"


# ============================================================
# Architecture configuration
# ============================================================

LINK_LATENCY_NS = 10
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80

# One shared service slot creates genuine P1 contention.
#
# Change to 2 later to reproduce the no-contention negative control.
HUB_MAX_CONCURRENT_TRANSFERS = 1

VICTIM_EVENT_TICK_NS = 5


# ============================================================
# Victim timing
# ============================================================

# Evaluator-controlled true victim start.
VICTIM_TRUE_START_NS = 1_000


# ============================================================
# Attacker knowledge and schedule
# ============================================================

# The attacker estimates that the victim starts near 1000 ns.
#
# This is coarse scheduling knowledge, not internal victim-trace knowledge.
ATTACKER_ESTIMATED_WINDOW_START_NS = 1_000

# Fixed attacker-controlled observation duration.
#
# This is identical for all victim circuits and is not derived from the
# victim QASM or operation count.
ATTACKER_OBSERVATION_DURATION_NS = 20_000

# Time between the starts of consecutive Probe-3 rounds.
PROBE_PERIOD_NS = 420

# Time between operations inside one Probe-3 round.
WITHIN_ROUND_EVENT_SPACING_NS = 5

TIE_BREAK_POLICY = "victim_first"

ATTACKER_STEP_BASE = 10_000_000


# ============================================================
# Utility functions
# ============================================================

def safe_tag(path_string: str) -> str:
    """Create a filename-safe circuit identifier."""
    stem = Path(path_string).stem

    return "".join(
        character
        if character.isalnum() or character in "_-"
        else "_"
        for character in stem
    )


def build_subset_qubit_map(
    num_qubits: int,
    module_subset: list[str],
) -> dict[int, str]:
    """
    Map a circuit across the supplied modules using contiguous blocks.
    """
    if num_qubits < len(module_subset):
        raise ValueError(
            f"Need at least {len(module_subset)} qubits to populate "
            f"{module_subset}; received {num_qubits}."
        )

    block_size = (
        num_qubits + len(module_subset) - 1
    ) // len(module_subset)

    mapping: dict[int, str] = {}

    for qubit in range(num_qubits):
        module_index = min(
            qubit // block_size,
            len(module_subset) - 1,
        )

        mapping[qubit] = module_subset[module_index]

    # Fallback for unusual sizes that fail to populate every module.
    if set(mapping.values()) != set(module_subset):
        mapping = {
            qubit: module_subset[
                qubit % len(module_subset)
            ]
            for qubit in range(num_qubits)
        }

    return mapping


def attacker_qubit_map() -> dict[int, str]:
    """
    P1 attacker placement.

    Qubits 0 and 1 are placed on module_3.
    Qubits 2 and 3 are placed on module_4.
    """
    return {
        0: "module_3",
        1: "module_3",
        2: "module_4",
        3: "module_4",
    }


def combine_qubit_mappings(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    attacker_mapping: dict[int, str],
) -> dict[int, str]:
    """Combine victim and attacker qubit spaces."""
    combined_mapping = dict(victim_mapping)

    for attacker_qubit, module in attacker_mapping.items():
        global_qubit = attacker_qubit + victim_num_qubits
        combined_mapping[global_qubit] = module

    return combined_mapping


def build_architecture(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    attacker_mapping: dict[int, str],
) -> FiveModuleLocalModularSuperconductingDQC:
    """Create the existing five-module architecture."""
    combined_mapping = combine_qubit_mappings(
        victim_mapping,
        victim_num_qubits,
        attacker_mapping,
    )

    return FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=combined_mapping,
        link_latency_ns=LINK_LATENCY_NS,
        hub_max_concurrent_transfers=(
            HUB_MAX_CONCURRENT_TRANSFERS
        ),
        hub_setup_latency_ns=HUB_SETUP_LATENCY_NS,
        hub_transfer_latency_ns=HUB_TRANSFER_LATENCY_NS,
        event_tick_ns=VICTIM_EVENT_TICK_NS,
    )


# ============================================================
# Victim trace generation
# ============================================================

def extract_static_victim_trace(
    qasm_file: str,
) -> tuple[list[dict], dict[int, str], int, int]:
    """
    Parse the victim QASM to create simulator-side victim traffic.

    The cross-operation count is evaluator-only metadata and is never
    passed to the attacker schedule.
    """
    qasm_path = Path(qasm_file)

    if not qasm_path.exists():
        raise FileNotFoundError(
            f"QASM file does not exist: {qasm_path.resolve()}"
        )

    circuit = QuantumCircuit.from_qasm_file(
        str(qasm_path)
    )

    victim_mapping = build_subset_qubit_map(
        circuit.num_qubits,
        VICTIM_MODULES,
    )

    victim_trace: list[dict] = []
    cross_module_operation_count = 0

    for step_index, instruction in enumerate(circuit.data):
        operation = instruction.operation

        qubits = [
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        clbits = [
            circuit.find_bit(clbit).index
            for clbit in instruction.clbits
        ]

        if not qubits:
            continue

        touched_modules = sorted(
            {
                victim_mapping[qubit]
                for qubit in qubits
            }
        )

        is_cross_module = len(touched_modules) > 1

        if is_cross_module:
            cross_module_operation_count += 1

        victim_trace.append(
            {
                "step": step_index,
                "op_name": operation.name,
                "qubits": qubits,
                "clbits": clbits,
                "params": [
                    float(parameter)
                    if hasattr(parameter, "__float__")
                    else str(parameter)
                    for parameter in operation.params
                ],
                "placement_style": "static_distributed",
                "modules_touched": touched_modules,
                "modules": touched_modules,
                "is_cross_module": is_cross_module,
                "cross_module": is_cross_module,
                "communication_event": is_cross_module,
            }
        )

    return (
        victim_trace,
        victim_mapping,
        circuit.num_qubits,
        cross_module_operation_count,
    )


def schedule_victim_events(
    victim_trace: list[dict],
) -> list[dict]:
    """
    Assign the victim trace to its true simulator timeline.

    This timeline is hidden from the attacker.
    """
    scheduled_events: list[dict] = []

    for event_index, trace_entry in enumerate(victim_trace):
        release_time_ns = (
            VICTIM_TRUE_START_NS
            + event_index * VICTIM_EVENT_TICK_NS
        )

        scheduled_events.append(
            {
                "release_time_ns": release_time_ns,
                "tenant": "victim",
                "sequence_index": event_index,
                "entry": copy.deepcopy(trace_entry),
            }
        )

    return scheduled_events


# ============================================================
# Window-aligned Probe 3
# ============================================================

def build_window_aligned_probe3_schedule(
    victim_num_qubits: int,
) -> tuple[list[dict], dict[int, str], dict[int, dict]]:
    """
    Generate independently timed Probe-3 rounds inside the attacker's
    estimated victim-execution window.

    No victim trace, event count, or cross-operation count is read here.
    """
    attacker_mapping = attacker_qubit_map()

    probe_operations = [
        ("h", [0]),
        ("h", [1]),
        ("x", [2]),
        ("x", [3]),
        ("z", [0]),
        ("z", [2]),
        ("cx", [0, 2]),
    ]

    observation_end_ns = (
        ATTACKER_ESTIMATED_WINDOW_START_NS
        + ATTACKER_OBSERVATION_DURATION_NS
    )

    scheduled_events: list[dict] = []
    cross_step_to_probe: dict[int, dict] = {}

    probe_id = 0
    attacker_event_index = 0

    while True:
        round_start_ns = (
            ATTACKER_ESTIMATED_WINDOW_START_NS
            + probe_id * PROBE_PERIOD_NS
        )

        if round_start_ns >= observation_end_ns:
            break

        for operation_index, (
            operation_name,
            local_qubits,
        ) in enumerate(probe_operations):
            release_time_ns = (
                round_start_ns
                + operation_index
                * WITHIN_ROUND_EVENT_SPACING_NS
            )

            # Do not release events outside the observation window.
            if release_time_ns >= observation_end_ns:
                continue

            global_qubits = [
                local_qubit + victim_num_qubits
                for local_qubit in local_qubits
            ]

            touched_modules = sorted(
                {
                    attacker_mapping[local_qubit]
                    for local_qubit in local_qubits
                }
            )

            is_cross_module = len(touched_modules) > 1

            step = (
                ATTACKER_STEP_BASE
                + attacker_event_index
            )

            trace_entry = {
                "step": step,
                "op_name": operation_name,
                "qubits": global_qubits,
                "clbits": [],
                "params": [],
                "placement_style": "static_distributed",
                "modules_touched": touched_modules,
                "modules": touched_modules,
                "is_cross_module": is_cross_module,
                "cross_module": is_cross_module,
                "communication_event": is_cross_module,
            }

            scheduled_events.append(
                {
                    "release_time_ns": release_time_ns,
                    "tenant": "attacker",
                    "sequence_index": attacker_event_index,
                    "entry": trace_entry,
                }
            )

            if is_cross_module:
                cross_step_to_probe[step] = {
                    "probe_id": probe_id,
                    "round_start_time_ns": round_start_ns,
                    "request_release_time_ns": release_time_ns,
                }

            attacker_event_index += 1

        probe_id += 1

    return (
        scheduled_events,
        attacker_mapping,
        cross_step_to_probe,
    )


# ============================================================
# Event scheduler
# ============================================================

def scheduled_event_sort_key(
    scheduled_event: dict,
) -> tuple[int, int, int]:
    """Sort all victim and attacker events chronologically."""
    if TIE_BREAK_POLICY == "victim_first":
        tenant_priority = (
            0
            if scheduled_event["tenant"] == "victim"
            else 1
        )

    elif TIE_BREAK_POLICY == "attacker_first":
        tenant_priority = (
            0
            if scheduled_event["tenant"] == "attacker"
            else 1
        )

    else:
        raise ValueError(
            "TIE_BREAK_POLICY must be either "
            "'victim_first' or 'attacker_first'."
        )

    return (
        int(scheduled_event["release_time_ns"]),
        tenant_priority,
        int(scheduled_event["sequence_index"]),
    )


def advance_architecture_to_time(
    architecture,
    target_time_ns: int,
) -> None:
    """
    Advance the architecture while preserving intermediate request
    completions and hub admissions.
    """
    if target_time_ns < architecture.hub.current_time_ns:
        raise RuntimeError(
            "Attempted to move architecture time backwards: "
            f"{architecture.hub.current_time_ns} -> "
            f"{target_time_ns}"
        )

    while architecture.hub.current_time_ns < target_time_ns:
        active_completion_times = [
            request.end_time_ns
            for request in architecture.hub.active_requests
            if request.end_time_ns is not None
        ]

        if active_completion_times:
            next_completion_ns = min(
                active_completion_times
            )

            next_time_ns = min(
                next_completion_ns,
                target_time_ns,
            )
        else:
            next_time_ns = target_time_ns

        delta_ns = (
            next_time_ns
            - architecture.hub.current_time_ns
        )

        if delta_ns <= 0:
            delta_ns = 1

        architecture.advance_architecture_time(
            delta_ns=delta_ns
        )


def execute_timed_schedule(
    architecture,
    scheduled_events: list[dict],
) -> None:
    """Execute the merged wall-clock event schedule."""
    ordered_events = sorted(
        scheduled_events,
        key=scheduled_event_sort_key,
    )

    for scheduled_event in ordered_events:
        release_time_ns = int(
            scheduled_event["release_time_ns"]
        )

        advance_architecture_to_time(
            architecture,
            release_time_ns,
        )

        normalized_event = normalize_trace_entry(
            scheduled_event["entry"],
            "static_distributed",
        )

        architecture.route_trace_event(
            normalized_event
        )

    architecture.drain_hub()


# ============================================================
# Observation collection
# ============================================================

def collect_attacker_observations(
    architecture,
    cross_step_to_probe: dict[int, dict],
    run_type: str,
) -> pd.DataFrame:
    """Collect only attacker-visible remote-request timing."""
    rows: list[dict] = []

    attacker_requests = [
        request
        for request in architecture.hub.completed_requests
        if request.source_module in ATTACKER_MODULES
    ]

    for request in attacker_requests:
        trace_step = request.original_event.step
        probe_metadata = cross_step_to_probe.get(trace_step)

        if probe_metadata is None:
            raise KeyError(
                f"No probe metadata exists for attacker "
                f"trace step {trace_step}."
            )

        rows.append(
            {
                "run_type": run_type,
                "probe_id": probe_metadata["probe_id"],
                "round_start_time_ns": (
                    probe_metadata[
                        "round_start_time_ns"
                    ]
                ),
                "planned_request_release_time_ns": (
                    probe_metadata[
                        "request_release_time_ns"
                    ]
                ),
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
                "source_module": request.source_module,
                "target_modules": ",".join(
                    request.target_modules
                ),
                "request_id": request.request_id,
            }
        )

    observations = pd.DataFrame(rows)

    if observations.empty:
        raise RuntimeError(
            "No attacker remote requests were completed."
        )

    return (
        observations
        .sort_values("probe_id")
        .reset_index(drop=True)
    )


def collect_victim_ground_truth(
    architecture,
) -> pd.DataFrame:
    """
    Save evaluator-only victim remote-request timing.

    This file is not provided to the attacker analysis.
    """
    rows: list[dict] = []

    victim_requests = [
        request
        for request in architecture.hub.completed_requests
        if request.source_module in VICTIM_MODULES
    ]

    for remote_event_id, request in enumerate(
        victim_requests
    ):
        rows.append(
            {
                "victim_remote_event_id": remote_event_id,
                "victim_trace_step": (
                    request.original_event.step
                ),
                "op_name": request.original_event.op_name,
                "qubits": ",".join(
                    str(qubit)
                    for qubit
                    in request.original_event.qubits
                ),
                "source_module": request.source_module,
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

    if not rows:
        return pd.DataFrame(
            columns=[
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
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "arrival_time_ns",
                "victim_remote_event_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Baseline subtraction
# ============================================================

def compare_against_attacker_only_baseline(
    baseline_observations: pd.DataFrame,
    victim_on_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract attacker-only timing from victim-on timing."""
    baseline = baseline_observations[
        [
            "probe_id",
            "waiting_time_ns",
            "turnaround_time_ns",
            "completion_time_ns",
        ]
    ].rename(
        columns={
            "waiting_time_ns": (
                "baseline_waiting_time_ns"
            ),
            "turnaround_time_ns": (
                "baseline_turnaround_time_ns"
            ),
            "completion_time_ns": (
                "baseline_completion_time_ns"
            ),
        }
    )

    victim_on = victim_on_observations.drop(
        columns=["run_type"]
    ).rename(
        columns={
            "waiting_time_ns": (
                "victim_on_waiting_time_ns"
            ),
            "turnaround_time_ns": (
                "victim_on_turnaround_time_ns"
            ),
            "completion_time_ns": (
                "victim_on_completion_time_ns"
            ),
        }
    )

    combined = victim_on.merge(
        baseline,
        on="probe_id",
        how="inner",
        validate="one_to_one",
    )

    combined["excess_waiting_time_ns"] = (
        combined["victim_on_waiting_time_ns"]
        - combined["baseline_waiting_time_ns"]
    )

    combined["excess_turnaround_time_ns"] = (
        combined["victim_on_turnaround_time_ns"]
        - combined["baseline_turnaround_time_ns"]
    )

    combined["victim_contention_observed"] = (
        combined["excess_turnaround_time_ns"] > 0
    )

    return (
        combined
        .sort_values("probe_id")
        .reset_index(drop=True)
    )


# ============================================================
# Summary and plotting
# ============================================================

def create_summary(
    victim_qasm: str,
    victim_trace: list[dict],
    victim_cross_operations: int,
    baseline_observations: pd.DataFrame,
    victim_on_observations: pd.DataFrame,
    combined_observations: pd.DataFrame,
    victim_ground_truth: pd.DataFrame,
    baseline_architecture,
    victim_architecture,
) -> dict:
    """Create one summary record for a victim circuit."""
    return {
        "victim_qasm": victim_qasm,
        "workload_type": "static_distributed",
        "placement": "P1_disjoint",
        "threat_model": (
            "blackbox_with_coarse_window_knowledge"
        ),
        "probe": "probe3_R1_uniform",
        "hub_max_concurrent_transfers": (
            HUB_MAX_CONCURRENT_TRANSFERS
        ),
        "victim_true_start_ns": (
            VICTIM_TRUE_START_NS
        ),
        "attacker_estimated_window_start_ns": (
            ATTACKER_ESTIMATED_WINDOW_START_NS
        ),
        "attacker_observation_duration_ns": (
            ATTACKER_OBSERVATION_DURATION_NS
        ),
        "probe_period_ns": PROBE_PERIOD_NS,
        "within_round_event_spacing_ns": (
            WITHIN_ROUND_EVENT_SPACING_NS
        ),
        "total_probe_rounds": int(
            len(victim_on_observations)
        ),
        "victim_total_events": len(victim_trace),
        "victim_cross_module_ops_evaluator_only": (
            victim_cross_operations
        ),
        "victim_completed_remote_requests": int(
            len(victim_ground_truth)
        ),
        "baseline_avg_waiting_time_ns": float(
            baseline_observations[
                "waiting_time_ns"
            ].mean()
        ),
        "baseline_avg_turnaround_time_ns": float(
            baseline_observations[
                "turnaround_time_ns"
            ].mean()
        ),
        "victim_on_avg_waiting_time_ns": float(
            victim_on_observations[
                "waiting_time_ns"
            ].mean()
        ),
        "victim_on_avg_turnaround_time_ns": float(
            victim_on_observations[
                "turnaround_time_ns"
            ].mean()
        ),
        "victim_on_max_waiting_time_ns": float(
            victim_on_observations[
                "waiting_time_ns"
            ].max()
        ),
        "victim_on_waited_fraction": float(
            (
                victim_on_observations[
                    "waiting_time_ns"
                ]
                > 0
            ).mean()
        ),
        "avg_excess_waiting_time_ns": float(
            combined_observations[
                "excess_waiting_time_ns"
            ].mean()
        ),
        "avg_excess_turnaround_time_ns": float(
            combined_observations[
                "excess_turnaround_time_ns"
            ].mean()
        ),
        "max_excess_turnaround_time_ns": float(
            combined_observations[
                "excess_turnaround_time_ns"
            ].max()
        ),
        "contention_observed_fraction": float(
            combined_observations[
                "victim_contention_observed"
            ].mean()
        ),
        "baseline_hub_makespan_ns": int(
            baseline_architecture.hub.current_time_ns
        ),
        "victim_on_hub_makespan_ns": int(
            victim_architecture.hub.current_time_ns
        ),
    }


def save_timing_plots(
    combined_observations: pd.DataFrame,
    victim_tag: str,
) -> None:
    """Save raw and baseline-subtracted attacker timing plots."""
    release_times = combined_observations[
        "planned_request_release_time_ns"
    ]

    plt.figure(figsize=(13, 6))

    plt.plot(
        release_times,
        combined_observations[
            "victim_on_turnaround_time_ns"
        ],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Victim present",
    )

    plt.plot(
        release_times,
        combined_observations[
            "baseline_turnaround_time_ns"
        ],
        linewidth=1,
        label="Attacker only",
    )

    plt.xlabel("Attacker probe release time (ns)")
    plt.ylabel("Probe turnaround time (ns)")

    plt.title(
        f"Window-Aligned Black-Box Timing: "
        f"{victim_tag}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{victim_tag}_window_timing_trace.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(13, 5))

    plt.plot(
        release_times,
        combined_observations[
            "excess_turnaround_time_ns"
        ],
        marker="o",
        markersize=3,
        linewidth=1,
    )

    plt.axhline(0, linewidth=1)

    plt.xlabel("Attacker probe release time (ns)")
    plt.ylabel("Victim-induced excess latency (ns)")

    plt.title(
        f"Victim-Induced Probe Delay: "
        f"{victim_tag}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{victim_tag}_window_excess_latency.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# One victim experiment
# ============================================================

def run_one_victim(victim_qasm: str) -> dict:
    """Run attacker-only and victim-present executions."""
    victim_tag = safe_tag(victim_qasm)

    print(
        f"\n=== Window-aligned black-box run: "
        f"{victim_qasm} ==="
    )

    (
        victim_trace,
        victim_mapping,
        victim_num_qubits,
        victim_cross_operations,
    ) = extract_static_victim_trace(victim_qasm)

    (
        attacker_schedule,
        attacker_mapping,
        cross_step_to_probe,
    ) = build_window_aligned_probe3_schedule(
        victim_num_qubits
    )

    # --------------------------------------------------------
    # Attacker-only calibration
    # --------------------------------------------------------

    baseline_architecture = build_architecture(
        victim_mapping,
        victim_num_qubits,
        attacker_mapping,
    )

    execute_timed_schedule(
        baseline_architecture,
        copy.deepcopy(attacker_schedule),
    )

    baseline_observations = (
        collect_attacker_observations(
            baseline_architecture,
            cross_step_to_probe,
            "attacker_only",
        )
    )

    # --------------------------------------------------------
    # Victim-present execution
    # --------------------------------------------------------

    victim_architecture = build_architecture(
        victim_mapping,
        victim_num_qubits,
        attacker_mapping,
    )

    victim_schedule = schedule_victim_events(
        victim_trace
    )

    merged_schedule = (
        victim_schedule
        + copy.deepcopy(attacker_schedule)
    )

    execute_timed_schedule(
        victim_architecture,
        merged_schedule,
    )

    victim_on_observations = (
        collect_attacker_observations(
            victim_architecture,
            cross_step_to_probe,
            "victim_present",
        )
    )

    victim_ground_truth = (
        collect_victim_ground_truth(
            victim_architecture
        )
    )

    combined_observations = (
        compare_against_attacker_only_baseline(
            baseline_observations,
            victim_on_observations,
        )
    )

    summary = create_summary(
        victim_qasm=victim_qasm,
        victim_trace=victim_trace,
        victim_cross_operations=(
            victim_cross_operations
        ),
        baseline_observations=(
            baseline_observations
        ),
        victim_on_observations=(
            victim_on_observations
        ),
        combined_observations=(
            combined_observations
        ),
        victim_ground_truth=(
            victim_ground_truth
        ),
        baseline_architecture=(
            baseline_architecture
        ),
        victim_architecture=(
            victim_architecture
        ),
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    attacker_csv_path = (
        OUTPUT_DIR
        / f"{victim_tag}_attacker_observations.csv"
    )

    ground_truth_csv_path = (
        OUTPUT_DIR
        / f"{victim_tag}_victim_ground_truth.csv"
    )

    summary_json_path = (
        OUTPUT_DIR
        / f"{victim_tag}_summary.json"
    )

    combined_observations.to_csv(
        attacker_csv_path,
        index=False,
    )

    victim_ground_truth.to_csv(
        ground_truth_csv_path,
        index=False,
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
        )

    save_timing_plots(
        combined_observations,
        victim_tag,
    )

    print(json.dumps(summary, indent=2))

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[dict] = []

    for victim_qasm in VICTIM_QASMS:
        summaries.append(
            run_one_victim(victim_qasm)
        )

    summary_dataframe = pd.DataFrame(summaries)

    combined_summary_path = (
        OUTPUT_DIR
        / "blackbox_window_baseline_summary.csv"
    )

    summary_dataframe.to_csv(
        combined_summary_path,
        index=False,
    )

    print(
        "\n=== Combined window-aligned "
        "black-box summary ==="
    )

    print(
        summary_dataframe.to_string(index=False)
    )

    print(
        f"\nSaved summary to: "
        f"{combined_summary_path}"
    )


if __name__ == "__main__":
    main()
