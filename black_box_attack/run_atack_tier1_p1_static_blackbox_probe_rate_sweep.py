#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_probe_rate_sweep.py

Knob 2: Probe 3 rate.

The selected Probe 3 pattern is repeated uniformly throughout the same fixed
20,000 ns black-box observation window. Only its repetition rate changes.

Experiment separation
---------------------
- Probe-rate sweep:
    Change the average number of probes within the observation window.

- Later spacing sweep:
    Keep the selected average rate fixed, but change how those probes are
    distributed in time.

Fixed
-----
- Probe 3 light-periodic operation pattern
- static-distributed victim execution
- P1 disjoint placement
- one serialized shared hub-service slot
- true victim start = 1,000 ns
- estimated victim-window start = 1,000 ns
- observation duration = 20,000 ns
- 5 ns spacing between operations inside Probe 3
- victim-first tie breaking

Rate levels
-----------
- rate_0p25x_sparse: period 1,680 ns
- rate_0p50x_low:    period   840 ns
- rate_1p00x_base:   period   420 ns
- rate_2p00x_high:   period   210 ns
- rate_3p00x_dense:  period   140 ns
- rate_4p00x_sat:    period   105 ns

The 1.00x configuration is the previous Probe 3 baseline.

The 4.00x configuration is intentionally slightly faster than the
110 ns remote-request service time. It therefore acts as a saturation
and attacker-self-contention control.

Outputs
-------
blackbox_window_results/probe_rate/
    probe_rate_summary.csv
    probe_rate_avg_excess_latency.png
    probe_rate_contention_fraction.png
    probe_rate_victim_slowdown.png
    probe_rate_baseline_self_wait.png

    rate_0p25x_sparse/
    rate_0p50x_low/
    rate_1p00x_base/
    rate_2p00x_high/
    rate_3p00x_dense/
    rate_4p00x_sat/
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)


# ============================================================
# Victims and output directory
# ============================================================

VICTIM_MODULES = [
    "module_0",
    "module_1",
    "module_2",
]

ATTACKER_MODULES = [
    "module_3",
    "module_4",
]

VICTIM_QASMS = [
    "square_root_n18.qasm",
    "qft_n18.qasm",
    "dnn_n16.qasm",
    "sat_n11.qasm",
    "bv_n19.qasm",
]

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "probe_rate"
)


# ============================================================
# Architecture configuration
# ============================================================

LINK_LATENCY_NS = 10
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80

# One serialized shared hub service.
HUB_MAX_CONCURRENT_TRANSFERS = 1

VICTIM_EVENT_TICK_NS = 5


# ============================================================
# Black-box timing configuration
# ============================================================

# Evaluator-controlled true victim start.
VICTIM_TRUE_START_NS = 1_000

# Attacker's coarse estimate.
#
# Start-time error will be tested separately.
ATTACKER_ESTIMATED_WINDOW_START_NS = 1_000

# Fixed observation duration for every victim and rate.
ATTACKER_OBSERVATION_DURATION_NS = 20_000

# Operation spacing inside one Probe 3 round.
WITHIN_ROUND_EVENT_SPACING_NS = 5

# Deterministic ordering for equal timestamps.
TIE_BREAK_POLICY = "victim_first"

# Keep attacker steps separate from victim trace steps.
ATTACKER_STEP_BASE = 10_000_000


# ============================================================
# Probe 3 and rate configurations
# ============================================================

# Preserve the exact Probe 3 operation order used in the
# black-box baseline.
PROBE_3_OPERATIONS: list[
    tuple[str, list[int]]
] = [
    ("h", [0]),
    ("h", [1]),
    ("x", [2]),
    ("x", [3]),
    ("z", [0]),
    ("z", [2]),
    ("cx", [0, 2]),
]


RATE_CONFIGS = [
    {
        "rate_name": "rate_0p25x_sparse",
        "rate_multiplier": 0.25,
        "probe_round_period_ns": 1_680,
    },
    {
        "rate_name": "rate_0p50x_low",
        "rate_multiplier": 0.50,
        "probe_round_period_ns": 840,
    },
    {
        "rate_name": "rate_1p00x_base",
        "rate_multiplier": 1.00,
        "probe_round_period_ns": 420,
    },
    {
        "rate_name": "rate_2p00x_high",
        "rate_multiplier": 2.00,
        "probe_round_period_ns": 210,
    },
    {
        "rate_name": "rate_3p00x_dense",
        "rate_multiplier": 3.00,
        "probe_round_period_ns": 140,
    },
    {
        "rate_name": "rate_4p00x_sat",
        "rate_multiplier": 4.00,
        "probe_round_period_ns": 105,
    },
]

RATE_ORDER = [
    configuration["rate_name"]
    for configuration in RATE_CONFIGS
]


# ============================================================
# General utilities
# ============================================================

def safe_tag(value: str) -> str:
    """Create a filename-safe identifier."""

    stem = Path(value).stem

    return "".join(
        character
        if character.isalnum()
        or character in "_-"
        else "_"
        for character in stem
    )


def attacker_qubit_map() -> dict[int, str]:
    """
    P1 attacker placement.

    q0 and q1 -> module_3
    q2 and q3 -> module_4
    """

    return {
        0: "module_3",
        1: "module_3",
        2: "module_4",
        3: "module_4",
    }


def build_subset_qubit_map(
    num_qubits: int,
    module_subset: list[str],
) -> dict[int, str]:
    """
    Map victim qubits across modules using
    contiguous blocks.
    """

    if num_qubits < len(module_subset):
        raise ValueError(
            f"Need at least {len(module_subset)} "
            f"qubits to populate {module_subset}; "
            f"received {num_qubits}."
        )

    block_size = (
        num_qubits
        + len(module_subset)
        - 1
    ) // len(module_subset)

    mapping: dict[int, str] = {}

    for qubit in range(num_qubits):
        module_index = min(
            qubit // block_size,
            len(module_subset) - 1,
        )

        mapping[qubit] = (
            module_subset[module_index]
        )

    # Fallback for unusual circuit sizes.
    if set(mapping.values()) != set(
        module_subset
    ):
        mapping = {
            qubit: module_subset[
                qubit % len(module_subset)
            ]
            for qubit in range(num_qubits)
        }

    return mapping


def combine_qubit_mappings(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    attacker_mapping: dict[int, str],
) -> dict[int, str]:
    """
    Combine victim and attacker qubit spaces.
    """

    combined = dict(victim_mapping)

    for attacker_qubit, module in (
        attacker_mapping.items()
    ):
        global_qubit = (
            attacker_qubit
            + victim_num_qubits
        )

        combined[global_qubit] = module

    return combined


def build_architecture(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    attacker_mapping: dict[int, str],
) -> FiveModuleLocalModularSuperconductingDQC:
    """
    Build the five-module architecture.
    """

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
        hub_setup_latency_ns=(
            HUB_SETUP_LATENCY_NS
        ),
        hub_transfer_latency_ns=(
            HUB_TRANSFER_LATENCY_NS
        ),
        event_tick_ns=VICTIM_EVENT_TICK_NS,
    )


# ============================================================
# Victim trace generation
# ============================================================

def extract_static_victim_trace(
    qasm_file: str,
) -> tuple[
    list[dict],
    dict[int, str],
    int,
    int,
]:
    """
    Parse victim QASM on the evaluator side.

    Victim trace information is never used to
    choose attacker probe times.
    """

    qasm_path = Path(qasm_file)

    if not qasm_path.exists():
        raise FileNotFoundError(
            "QASM file does not exist: "
            f"{qasm_path.resolve()}"
        )

    circuit = QuantumCircuit.from_qasm_file(
        str(qasm_path)
    )

    victim_mapping = build_subset_qubit_map(
        circuit.num_qubits,
        VICTIM_MODULES,
    )

    victim_trace: list[dict] = []
    cross_operation_count = 0

    for step_index, instruction in enumerate(
        circuit.data
    ):
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

        is_cross_module = (
            len(touched_modules) > 1
        )

        if is_cross_module:
            cross_operation_count += 1

        victim_trace.append(
            {
                "step": step_index,
                "op_name": operation.name,
                "qubits": qubits,
                "clbits": clbits,
                "params": [
                    float(parameter)
                    if hasattr(
                        parameter,
                        "__float__",
                    )
                    else str(parameter)
                    for parameter
                    in operation.params
                ],
                "placement_style": (
                    "static_distributed"
                ),
                "modules_touched": (
                    touched_modules
                ),
                "modules": touched_modules,
                "is_cross_module": (
                    is_cross_module
                ),
                "cross_module": (
                    is_cross_module
                ),
                "communication_event": (
                    is_cross_module
                ),
            }
        )

    return (
        victim_trace,
        victim_mapping,
        circuit.num_qubits,
        cross_operation_count,
    )


def schedule_victim_events(
    victim_trace: list[dict],
) -> list[dict]:
    """
    Place the victim on its evaluator-controlled
    wall-clock timeline.
    """

    scheduled_events: list[dict] = []

    for event_index, trace_entry in enumerate(
        victim_trace
    ):
        release_time_ns = (
            VICTIM_TRUE_START_NS
            + event_index
            * VICTIM_EVENT_TICK_NS
        )

        scheduled_events.append(
            {
                "release_time_ns": (
                    release_time_ns
                ),
                "tenant": "victim",
                "sequence_index": (
                    event_index
                ),
                "entry": copy.deepcopy(
                    trace_entry
                ),
            }
        )

    return scheduled_events


# ============================================================
# Probe-rate schedule generation
# ============================================================

def build_probe_rate_schedule(
    rate_config: dict,
    victim_num_qubits: int,
) -> tuple[
    list[dict],
    dict[int, str],
    dict[int, dict],
    dict,
]:
    """
    Repeat Probe 3 uniformly at the selected rate.

    No victim event, communication count, or true
    victim duration is consulted.
    """

    attacker_mapping = (
        attacker_qubit_map()
    )

    period_ns = int(
        rate_config[
            "probe_round_period_ns"
        ]
    )

    if period_ns <= 0:
        raise ValueError(
            "probe_round_period_ns "
            "must be positive."
        )

    observation_end_ns = (
        ATTACKER_ESTIMATED_WINDOW_START_NS
        + ATTACKER_OBSERVATION_DURATION_NS
    )

    scheduled_events: list[dict] = []

    cross_step_metadata: dict[
        int,
        dict,
    ] = {}

    probe_round_id = 0
    attacker_event_index = 0
    attacker_request_id = 0

    while True:
        round_start_ns = (
            ATTACKER_ESTIMATED_WINDOW_START_NS
            + probe_round_id
            * period_ns
        )

        if round_start_ns >= observation_end_ns:
            break

        for operation_index, (
            operation_name,
            local_qubits,
        ) in enumerate(
            PROBE_3_OPERATIONS
        ):
            release_time_ns = (
                round_start_ns
                + operation_index
                * WITHIN_ROUND_EVENT_SPACING_NS
            )

            if (
                release_time_ns
                >= observation_end_ns
            ):
                continue

            global_qubits = [
                local_qubit
                + victim_num_qubits
                for local_qubit
                in local_qubits
            ]

            touched_modules = sorted(
                {
                    attacker_mapping[
                        local_qubit
                    ]
                    for local_qubit
                    in local_qubits
                }
            )

            is_cross_module = (
                len(touched_modules) > 1
            )

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
                "placement_style": (
                    "static_distributed"
                ),
                "modules_touched": (
                    touched_modules
                ),
                "modules": touched_modules,
                "is_cross_module": (
                    is_cross_module
                ),
                "cross_module": (
                    is_cross_module
                ),
                "communication_event": (
                    is_cross_module
                ),
            }

            scheduled_events.append(
                {
                    "release_time_ns": (
                        release_time_ns
                    ),
                    "tenant": "attacker",
                    "sequence_index": (
                        attacker_event_index
                    ),
                    "entry": trace_entry,
                }
            )

            if is_cross_module:
                cross_step_metadata[step] = {
                    "attacker_request_id": (
                        attacker_request_id
                    ),
                    "rate_name": (
                        rate_config[
                            "rate_name"
                        ]
                    ),
                    "rate_multiplier": (
                        rate_config[
                            "rate_multiplier"
                        ]
                    ),
                    "probe_round_period_ns": (
                        period_ns
                    ),
                    "probe_round_id": (
                        probe_round_id
                    ),
                    "round_start_time_ns": (
                        round_start_ns
                    ),
                    "request_release_time_ns": (
                        release_time_ns
                    ),
                }

                attacker_request_id += 1

            attacker_event_index += 1

        probe_round_id += 1

    observation_duration_us = (
        ATTACKER_OBSERVATION_DURATION_NS
        / 1_000.0
    )

    realized_rate = (
        len(cross_step_metadata)
        / observation_duration_us
    )

    schedule_metadata = {
        "rate_name": (
            rate_config["rate_name"]
        ),
        "rate_multiplier": (
            rate_config["rate_multiplier"]
        ),
        "probe_round_period_ns": (
            period_ns
        ),
        "total_probe_rounds": (
            probe_round_id
        ),
        "total_attacker_events": (
            len(scheduled_events)
        ),
        "total_attacker_remote_requests": (
            len(cross_step_metadata)
        ),
        "realized_remote_probe_rate_per_us": (
            realized_rate
        ),
    }

    return (
        scheduled_events,
        attacker_mapping,
        cross_step_metadata,
        schedule_metadata,
    )


# ============================================================
# Wall-clock event scheduler
# ============================================================

def scheduled_event_sort_key(
    scheduled_event: dict,
) -> tuple[int, int, int]:
    """
    Sort events by release time and deterministic
    tenant priority.
    """

    if TIE_BREAK_POLICY == "victim_first":
        tenant_priority = (
            0
            if scheduled_event["tenant"]
            == "victim"
            else 1
        )

    elif TIE_BREAK_POLICY == "attacker_first":
        tenant_priority = (
            0
            if scheduled_event["tenant"]
            == "attacker"
            else 1
        )

    else:
        raise ValueError(
            "TIE_BREAK_POLICY must be "
            "'victim_first' or "
            "'attacker_first'."
        )

    return (
        int(
            scheduled_event[
                "release_time_ns"
            ]
        ),
        tenant_priority,
        int(
            scheduled_event[
                "sequence_index"
            ]
        ),
    )


def advance_architecture_to_time(
    architecture:
    FiveModuleLocalModularSuperconductingDQC,
    target_time_ns: int,
) -> None:
    """
    Advance to a release timestamp while processing
    intermediate transfer completions.
    """

    if (
        target_time_ns
        < architecture.hub.current_time_ns
    ):
        raise RuntimeError(
            "Attempted to move architecture "
            "time backwards: "
            f"{architecture.hub.current_time_ns}"
            f" -> {target_time_ns}"
        )

    while (
        architecture.hub.current_time_ns
        < target_time_ns
    ):
        completion_times = [
            request.end_time_ns
            for request
            in architecture.hub.active_requests
            if request.end_time_ns is not None
        ]

        if completion_times:
            next_completion_ns = min(
                completion_times
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
    architecture:
    FiveModuleLocalModularSuperconductingDQC,
    scheduled_events: Iterable[dict],
) -> None:
    """
    Execute a merged victim-attacker timeline.
    """

    ordered_events = sorted(
        scheduled_events,
        key=scheduled_event_sort_key,
    )

    for scheduled_event in ordered_events:
        release_time_ns = int(
            scheduled_event[
                "release_time_ns"
            ]
        )

        advance_architecture_to_time(
            architecture,
            release_time_ns,
        )

        normalized_event = (
            normalize_trace_entry(
                scheduled_event["entry"],
                "static_distributed",
            )
        )

        architecture.route_trace_event(
            normalized_event
        )

    architecture.drain_hub()


# ============================================================
# Observation collection
# ============================================================

def collect_attacker_observations(
    architecture:
    FiveModuleLocalModularSuperconductingDQC,
    cross_step_metadata:
    dict[int, dict],
    run_type: str,
) -> pd.DataFrame:
    """
    Collect attacker-visible timing for every
    remote Probe 3 request.
    """

    rows: list[dict] = []

    for request in (
        architecture.hub.completed_requests
    ):
        if (
            request.source_module
            not in ATTACKER_MODULES
        ):
            continue

        trace_step = (
            request.original_event.step
        )

        metadata = (
            cross_step_metadata.get(
                trace_step
            )
        )

        if metadata is None:
            raise KeyError(
                "No probe metadata exists "
                "for attacker trace step "
                f"{trace_step}."
            )

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

    observations = pd.DataFrame(rows)

    if observations.empty:
        raise RuntimeError(
            "No attacker remote requests "
            f"completed for {run_type}."
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
    run_type: str,
) -> pd.DataFrame:
    """
    Collect evaluator-only victim request timing.
    """

    rows: list[dict] = []

    victim_requests = [
        request
        for request
        in architecture.hub.completed_requests
        if request.source_module
        in VICTIM_MODULES
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
# Attacker-only baseline subtraction
# ============================================================

def compare_attacker_runs(
    attacker_only: pd.DataFrame,
    victim_present: pd.DataFrame,
) -> pd.DataFrame:
    """
    Subtract the identical attacker-only schedule
    request by request.
    """

    baseline = attacker_only[
        [
            "attacker_request_id",
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

    victim_on = victim_present.drop(
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
        on="attacker_request_id",
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != len(
        attacker_only
    ):
        raise RuntimeError(
            "Attacker-only and victim-present "
            "request counts differ."
        )

    combined[
        "excess_waiting_time_ns"
    ] = (
        combined[
            "victim_on_waiting_time_ns"
        ]
        - combined[
            "baseline_waiting_time_ns"
        ]
    )

    combined[
        "excess_turnaround_time_ns"
    ] = (
        combined[
            "victim_on_turnaround_time_ns"
        ]
        - combined[
            "baseline_turnaround_time_ns"
        ]
    )

    combined[
        "victim_contention_observed"
    ] = (
        combined[
            "excess_turnaround_time_ns"
        ] > 0
    )

    return (
        combined
        .sort_values(
            "attacker_request_id"
        )
        .reset_index(drop=True)
    )


# ============================================================
# Summary metrics
# ============================================================

def victim_completion_offset_ns(
    victim_ground_truth: pd.DataFrame,
) -> float:
    """
    Return the victim's final remote completion
    relative to its true start.
    """

    if victim_ground_truth.empty:
        return 0.0

    return float(
        victim_ground_truth[
            "completion_time_ns"
        ].max()
        - VICTIM_TRUE_START_NS
    )


def create_summary(
    *,
    victim_qasm: str,
    rate_config: dict,
    victim_trace: list[dict],
    victim_cross_operations: int,
    schedule_metadata: dict,
    attacker_only: pd.DataFrame,
    victim_present: pd.DataFrame,
    compared: pd.DataFrame,
    victim_only_ground_truth:
    pd.DataFrame,
    victim_on_ground_truth:
    pd.DataFrame,
    attacker_only_architecture,
    victim_only_architecture,
    victim_on_architecture,
) -> dict:
    """
    Create one victim/rate summary.
    """

    victim_only_completion = (
        victim_completion_offset_ns(
            victim_only_ground_truth
        )
    )

    victim_on_completion = (
        victim_completion_offset_ns(
            victim_on_ground_truth
        )
    )

    victim_slowdown_ns = (
        victim_on_completion
        - victim_only_completion
    )

    if victim_only_completion > 0:
        victim_slowdown_ratio = (
            victim_on_completion
            / victim_only_completion
        )
    else:
        victim_slowdown_ratio = 1.0

    return {
        "victim_qasm": victim_qasm,
        "victim_tag": safe_tag(
            victim_qasm
        ),
        "probe_name": (
            "probe_3_light_periodic"
        ),
        "knob": "probe_rate",
        "rate_name": (
            rate_config["rate_name"]
        ),
        "rate_multiplier": (
            rate_config[
                "rate_multiplier"
            ]
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
            HUB_MAX_CONCURRENT_TRANSFERS
        ),
        "victim_true_start_ns": (
            VICTIM_TRUE_START_NS
        ),
        "attacker_estimated_"
        "window_start_ns": (
            ATTACKER_ESTIMATED_WINDOW_START_NS
        ),
        "attacker_observation_"
        "duration_ns": (
            ATTACKER_OBSERVATION_DURATION_NS
        ),
        "within_round_event_spacing_ns": (
            WITHIN_ROUND_EVENT_SPACING_NS
        ),
        **schedule_metadata,
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
        "victim_on_waited_fraction": (
            float(
                (
                    victim_present[
                        "waiting_time_ns"
                    ] > 0
                ).mean()
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
        "contention_observed_fraction": (
            float(
                compared[
                    "victim_contention_observed"
                ].mean()
            )
        ),
        "victim_only_completion_offset_ns": (
            victim_only_completion
        ),
        "victim_on_completion_offset_ns": (
            victim_on_completion
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
# Plotting
# ============================================================

def save_request_level_plots(
    compared: pd.DataFrame,
    victim_tag: str,
    rate_name: str,
    rate_output_dir: Path,
) -> None:
    """
    Save raw and baseline-subtracted timing plots.
    """

    release_times = compared[
        "request_release_time_ns"
    ]

    plt.figure(figsize=(13, 6))

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
        "Attacker request release time (ns)"
    )

    plt.ylabel(
        "Remote-request turnaround time (ns)"
    )

    plt.title(
        "Probe 3 rate — "
        f"{rate_name}: {victim_tag}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        rate_output_dir
        / f"{victim_tag}_timing_trace.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(13, 5))

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
        "Attacker request release time (ns)"
    )

    plt.ylabel(
        "Victim-induced excess latency (ns)"
    )

    plt.title(
        "Victim-induced delay — "
        f"{rate_name}: {victim_tag}"
    )

    plt.tight_layout()

    plt.savefig(
        rate_output_dir
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
    """
    Save a workload-by-rate grouped plot.
    """

    pivot = summary_dataframe.pivot(
        index="victim_tag",
        columns="rate_name",
        values=metric,
    )

    pivot = pivot.reindex(
        columns=RATE_ORDER
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(14, 6),
    )

    axis.set_xlabel(
        "Victim workload"
    )

    axis.set_ylabel(ylabel)
    axis.set_title(title)

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.legend(
        title="Probe rate",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# One victim/rate configuration
# ============================================================

def run_one_configuration(
    victim_qasm: str,
    rate_config: dict,
) -> dict:
    """
    Run victim-only, attacker-only, and
    victim-present controls.
    """

    victim_tag = safe_tag(
        victim_qasm
    )

    rate_name = (
        rate_config["rate_name"]
    )

    rate_output_dir = (
        OUTPUT_DIR / rate_name
    )

    rate_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Probe-rate sweep: "
        f"{rate_name} | "
        f"{victim_qasm} ==="
    )

    (
        victim_trace,
        victim_mapping,
        victim_num_qubits,
        victim_cross_operations,
    ) = extract_static_victim_trace(
        victim_qasm
    )

    victim_schedule = (
        schedule_victim_events(
            victim_trace
        )
    )

    (
        attacker_schedule,
        attacker_mapping,
        cross_step_metadata,
        schedule_metadata,
    ) = build_probe_rate_schedule(
        rate_config,
        victim_num_qubits,
    )

    # --------------------------------------------------------
    # Victim-only control
    # --------------------------------------------------------

    victim_only_architecture = (
        build_architecture(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    execute_timed_schedule(
        victim_only_architecture,
        copy.deepcopy(
            victim_schedule
        ),
    )

    victim_only_ground_truth = (
        collect_victim_ground_truth(
            victim_only_architecture,
            "victim_only",
        )
    )

    # --------------------------------------------------------
    # Attacker-only calibration
    # --------------------------------------------------------

    attacker_only_architecture = (
        build_architecture(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    execute_timed_schedule(
        attacker_only_architecture,
        copy.deepcopy(
            attacker_schedule
        ),
    )

    attacker_only_observations = (
        collect_attacker_observations(
            attacker_only_architecture,
            cross_step_metadata,
            "attacker_only",
        )
    )

    # --------------------------------------------------------
    # Victim and attacker together
    # --------------------------------------------------------

    victim_on_architecture = (
        build_architecture(
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

    execute_timed_schedule(
        victim_on_architecture,
        merged_schedule,
    )

    victim_present_observations = (
        collect_attacker_observations(
            victim_on_architecture,
            cross_step_metadata,
            "victim_present",
        )
    )

    victim_on_ground_truth = (
        collect_victim_ground_truth(
            victim_on_architecture,
            "victim_present",
        )
    )

    compared = compare_attacker_runs(
        attacker_only_observations,
        victim_present_observations,
    )

    summary = create_summary(
        victim_qasm=victim_qasm,
        rate_config=rate_config,
        victim_trace=victim_trace,
        victim_cross_operations=(
            victim_cross_operations
        ),
        schedule_metadata=(
            schedule_metadata
        ),
        attacker_only=(
            attacker_only_observations
        ),
        victim_present=(
            victim_present_observations
        ),
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
        rate_output_dir
        / (
            f"{victim_tag}_"
            "attacker_observations.csv"
        ),
        index=False,
    )

    victim_only_ground_truth.to_csv(
        rate_output_dir
        / (
            f"{victim_tag}_"
            "victim_only_ground_truth.csv"
        ),
        index=False,
    )

    victim_on_ground_truth.to_csv(
        rate_output_dir
        / (
            f"{victim_tag}_"
            "victim_on_ground_truth.csv"
        ),
        index=False,
    )

    summary_path = (
        rate_output_dir
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
        rate_name,
        rate_output_dir,
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

    summaries: list[dict] = []

    for rate_config in RATE_CONFIGS:
        for victim_qasm in VICTIM_QASMS:
            summary = (
                run_one_configuration(
                    victim_qasm,
                    rate_config,
                )
            )

            summaries.append(summary)

    summary_dataframe = pd.DataFrame(
        summaries
    )

    summary_dataframe[
        "rate_name"
    ] = pd.Categorical(
        summary_dataframe[
            "rate_name"
        ],
        categories=RATE_ORDER,
        ordered=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "rate_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = (
        OUTPUT_DIR
        / "probe_rate_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "avg_excess_"
            "turnaround_time_ns"
        ),
        ylabel=(
            "Average victim-induced "
            "latency (ns)"
        ),
        title=(
            "Probe 3 Rate: Average "
            "Victim-Induced Latency"
        ),
        filename=(
            "probe_rate_"
            "avg_excess_latency.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "contention_observed_fraction"
        ),
        ylabel=(
            "Fraction of attacker "
            "requests delayed"
        ),
        title=(
            "Probe 3 Rate: Contention "
            "Observation Rate"
        ),
        filename=(
            "probe_rate_"
            "contention_fraction.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric="victim_slowdown_ratio",
        ylabel=(
            "Victim completion-time ratio"
        ),
        title=(
            "Probe 3 Rate: "
            "Victim Slowdown"
        ),
        filename=(
            "probe_rate_"
            "victim_slowdown.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "baseline_avg_waiting_time_ns"
        ),
        ylabel=(
            "Attacker-only average "
            "waiting time (ns)"
        ),
        title=(
            "Probe 3 Rate: "
            "Attacker Self-Contention"
        ),
        filename=(
            "probe_rate_"
            "baseline_self_wait.png"
        ),
    )

    print(
        "\n=== Combined probe-rate "
        "summary ==="
    )

    display_columns = [
        "victim_tag",
        "rate_name",
        "probe_round_period_ns",
        "total_attacker_remote_requests",
        "realized_remote_probe_rate_per_us",
        "baseline_avg_waiting_time_ns",
        "avg_excess_turnaround_time_ns",
        "max_excess_turnaround_time_ns",
        "contention_observed_fraction",
        "victim_slowdown_ratio",
    ]

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


if __name__ == "__main__":
    main()
