#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_observation_window_sweep.py

Knob 4: observation-window duration.

Fixed
-----
- Probe 3 light-periodic pattern
- uniform inter-probe spacing
- one remote probe every 420 ns
- static-distributed victim execution
- P1 disjoint placement
- one serialized shared hub-service slot
- victim true start = 1,000 ns
- attacker estimated start = 1,000 ns
- 5 ns spacing between Probe 3 operations
- victim-first tie breaking

Varied
------
- 5,000 ns observation window
- 10,000 ns observation window
- 20,000 ns observation window: selected baseline
- 30,000 ns observation window
- 40,000 ns observation window

Important
---------
These durations are absolute attacker-selected values. They are not calculated
from the victim QASM, victim event count, victim communication count, or true
victim completion time.

Outputs
-------
blackbox_window_results/observation_window/
    observation_window_summary.csv
    observation_window_schedule_summary.csv
    observation_window_avg_excess_latency.png
    observation_window_contention_fraction.png
    observation_window_victim_slowdown.png
    observation_window_total_excess_latency.png
    observation_window_probe_count.png

    window_5us_short/
    window_10us_medium/
    window_20us_baseline/
    window_30us_long/
    window_40us_extended/
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
    / "observation_window"
)


# ============================================================
# Architecture configuration
# ============================================================

LINK_LATENCY_NS = 10
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80

HUB_MAX_CONCURRENT_TRANSFERS = 1

VICTIM_EVENT_TICK_NS = 5


# ============================================================
# Black-box timing configuration
# ============================================================

VICTIM_TRUE_START_NS = 1_000

ATTACKER_ESTIMATED_WINDOW_START_NS = 1_000

# Selected rate and spacing from the preceding experiments.
PROBE_ROUND_PERIOD_NS = 420

WITHIN_ROUND_EVENT_SPACING_NS = 5

TIE_BREAK_POLICY = "victim_first"

ATTACKER_STEP_BASE = 10_000_000


# ============================================================
# Observation-window configurations
# ============================================================

WINDOW_CONFIGS = [
    {
        "window_name": "window_5us_short",
        "observation_duration_ns": 5_000,
    },
    {
        "window_name": "window_10us_medium",
        "observation_duration_ns": 10_000,
    },
    {
        "window_name": "window_20us_baseline",
        "observation_duration_ns": 20_000,
    },
    {
        "window_name": "window_30us_long",
        "observation_duration_ns": 30_000,
    },
    {
        "window_name": "window_40us_extended",
        "observation_duration_ns": 40_000,
    },
]

WINDOW_ORDER = [
    configuration["window_name"]
    for configuration in WINDOW_CONFIGS
]


# ============================================================
# Fixed Probe 3 definition
# ============================================================

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

# Probe 3's remote CX is the seventh operation.
REMOTE_REQUEST_OFFSET_NS = (
    (len(PROBE_3_OPERATIONS) - 1)
    * WITHIN_ROUND_EVENT_SPACING_NS
)


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
    """Map victim qubits across modules using contiguous blocks."""

    if num_qubits < len(module_subset):
        raise ValueError(
            f"Need at least {len(module_subset)} qubits "
            f"to populate {module_subset}; "
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

        mapping[qubit] = module_subset[
            module_index
        ]

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
    """Combine victim and attacker qubit namespaces."""

    combined_mapping = dict(
        victim_mapping
    )

    for attacker_qubit, module in (
        attacker_mapping.items()
    ):
        global_qubit = (
            victim_num_qubits
            + attacker_qubit
        )

        combined_mapping[
            global_qubit
        ] = module

    return combined_mapping


def build_architecture(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    attacker_mapping: dict[int, str],
) -> FiveModuleLocalModularSuperconductingDQC:
    """Build the existing five-module architecture."""

    combined_mapping = (
        combine_qubit_mappings(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    return (
        FiveModuleLocalModularSuperconductingDQC(
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
            event_tick_ns=(
                VICTIM_EVENT_TICK_NS
            ),
        )
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
    Parse the victim QASM on the evaluator side.

    Victim properties are never used to choose the
    attacker observation-window duration.
    """

    qasm_path = Path(
        qasm_file
    )

    if not qasm_path.exists():
        raise FileNotFoundError(
            "QASM file does not exist: "
            f"{qasm_path.resolve()}"
        )

    circuit = (
        QuantumCircuit.from_qasm_file(
            str(qasm_path)
        )
    )

    victim_mapping = (
        build_subset_qubit_map(
            circuit.num_qubits,
            VICTIM_MODULES,
        )
    )

    victim_trace: list[dict] = []

    cross_operation_count = 0

    for step_index, instruction in enumerate(
        circuit.data
    ):
        operation = (
            instruction.operation
        )

        qubits = [
            circuit.find_bit(qubit).index
            for qubit
            in instruction.qubits
        ]

        clbits = [
            circuit.find_bit(clbit).index
            for clbit
            in instruction.clbits
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
                    (
                        float(parameter)
                        if hasattr(
                            parameter,
                            "__float__",
                        )
                        else str(parameter)
                    )
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
    """Place victim events on the evaluator-controlled timeline."""

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
# Observation-window attacker schedule
# ============================================================

def build_observation_window_schedule(
    window_config: dict,
    victim_num_qubits: int,
) -> tuple[
    list[dict],
    dict[int, str],
    dict[int, dict],
    dict,
]:
    """
    Generate uniformly spaced Probe 3 rounds inside
    the selected absolute observation window.

    Only remote probes whose CX release occurs inside
    the window are included.
    """

    attacker_mapping = (
        attacker_qubit_map()
    )

    duration_ns = int(
        window_config[
            "observation_duration_ns"
        ]
    )

    if duration_ns <= (
        REMOTE_REQUEST_OFFSET_NS
    ):
        raise ValueError(
            "Observation duration is too short "
            "to contain one complete Probe 3 round."
        )

    window_start_ns = (
        ATTACKER_ESTIMATED_WINDOW_START_NS
    )

    window_end_ns = (
        window_start_ns
        + duration_ns
    )

    first_remote_release_ns = (
        window_start_ns
        + REMOTE_REQUEST_OFFSET_NS
    )

    remote_release_times_ns: list[int] = []

    remote_release_ns = (
        first_remote_release_ns
    )

    while remote_release_ns < window_end_ns:
        remote_release_times_ns.append(
            remote_release_ns
        )

        remote_release_ns += (
            PROBE_ROUND_PERIOD_NS
        )

    if not remote_release_times_ns:
        raise RuntimeError(
            "No complete remote probes fit "
            "inside the observation window."
        )

    scheduled_events: list[dict] = []

    cross_step_metadata: dict[
        int,
        dict,
    ] = {}

    attacker_event_index = 0

    for probe_id, remote_release_ns in enumerate(
        remote_release_times_ns
    ):
        round_start_ns = (
            remote_release_ns
            - REMOTE_REQUEST_OFFSET_NS
        )

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

            trace_step = (
                ATTACKER_STEP_BASE
                + attacker_event_index
            )

            trace_entry = {
                "step": trace_step,
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
                cross_step_metadata[
                    trace_step
                ] = {
                    "attacker_request_id": (
                        probe_id
                    ),
                    "probe_id": probe_id,
                    "window_name": (
                        window_config[
                            "window_name"
                        ]
                    ),
                    "observation_duration_ns": (
                        duration_ns
                    ),
                    "round_start_time_ns": (
                        round_start_ns
                    ),
                    "request_release_time_ns": (
                        remote_release_ns
                    ),
                }

            attacker_event_index += 1

    observation_duration_us = (
        duration_ns / 1_000.0
    )

    realized_rate_per_us = (
        len(remote_release_times_ns)
        / observation_duration_us
    )

    schedule_metadata = {
        "window_name": (
            window_config[
                "window_name"
            ]
        ),
        "observation_duration_ns": (
            duration_ns
        ),
        "observation_window_start_ns": (
            window_start_ns
        ),
        "observation_window_end_ns": (
            window_end_ns
        ),
        "probe_round_period_ns": (
            PROBE_ROUND_PERIOD_NS
        ),
        "total_probe_rounds": (
            len(remote_release_times_ns)
        ),
        "total_attacker_events": (
            len(scheduled_events)
        ),
        "total_attacker_remote_requests": (
            len(cross_step_metadata)
        ),
        "first_remote_probe_release_ns": (
            remote_release_times_ns[0]
        ),
        "last_remote_probe_release_ns": (
            remote_release_times_ns[-1]
        ),
        "realized_remote_probe_rate_per_us": (
            realized_rate_per_us
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
    """Sort events chronologically."""

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
    intermediate hub completions.
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
            if request.end_time_ns
            is not None
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
            next_time_ns = (
                target_time_ns
            )

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
    """Execute a merged wall-clock schedule."""

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
    """Collect attacker-visible remote-request timing."""

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
                "No attacker metadata exists "
                f"for trace step {trace_step}."
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

    observations = pd.DataFrame(
        rows
    )

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
    """Collect evaluator-only victim request timing."""

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
# Baseline subtraction
# ============================================================

def compare_attacker_runs(
    attacker_only: pd.DataFrame,
    victim_present: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract attacker-only timing request by request."""

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
    victim_ground_truth:
    pd.DataFrame,
) -> float:
    """
    Return final victim remote completion relative
    to the true victim start.
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
    window_config: dict,
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
    """Create one workload/window summary."""

    duration_ns = int(
        window_config[
            "observation_duration_ns"
        ]
    )

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

        window_to_victim_duration_ratio = (
            duration_ns
            / victim_only_completion
        )

    else:
        victim_slowdown_ratio = 1.0
        window_to_victim_duration_ratio = 0.0

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

    probes_released_before_victim_completion = int(
        (
            compared[
                "request_release_time_ns"
            ]
            <= (
                VICTIM_TRUE_START_NS
                + victim_only_completion
            )
        ).sum()
    )

    probe_count = int(
        len(compared)
    )

    active_window_probe_fraction = (
        probes_released_before_victim_completion
        / probe_count
        if probe_count > 0
        else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "victim_tag": safe_tag(
            victim_qasm
        ),
        "knob": (
            "observation_window"
        ),
        "window_name": (
            window_config[
                "window_name"
            ]
        ),
        "observation_duration_ns": (
            duration_ns
        ),
        "probe_name": (
            "probe_3_light_periodic"
        ),
        "probe_round_period_ns": (
            PROBE_ROUND_PERIOD_NS
        ),
        "spacing_pattern": (
            "uniform"
        ),
        "workload_type": (
            "static_distributed"
        ),
        "placement": (
            "P1_disjoint"
        ),
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
        "attacker_estimated_window_start_ns": (
            ATTACKER_ESTIMATED_WINDOW_START_NS
        ),
        "within_round_event_spacing_ns": (
            WITHIN_ROUND_EVENT_SPACING_NS
        ),
        **schedule_metadata,
        "victim_total_events": (
            len(victim_trace)
        ),
        "victim_cross_module_ops_evaluator_only": (
            victim_cross_operations
        ),
        "victim_completed_remote_requests": int(
            len(victim_on_ground_truth)
        ),
        "baseline_avg_waiting_time_ns": float(
            attacker_only[
                "waiting_time_ns"
            ].mean()
        ),
        "baseline_max_waiting_time_ns": float(
            attacker_only[
                "waiting_time_ns"
            ].max()
        ),
        "baseline_avg_turnaround_time_ns": float(
            attacker_only[
                "turnaround_time_ns"
            ].mean()
        ),
        "victim_on_avg_waiting_time_ns": float(
            victim_present[
                "waiting_time_ns"
            ].mean()
        ),
        "victim_on_max_waiting_time_ns": float(
            victim_present[
                "waiting_time_ns"
            ].max()
        ),
        "victim_on_avg_turnaround_time_ns": float(
            victim_present[
                "turnaround_time_ns"
            ].mean()
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
        "window_to_victim_duration_ratio_evaluator_only": (
            window_to_victim_duration_ratio
        ),
        "probes_before_victim_completion_evaluator_only": (
            probes_released_before_victim_completion
        ),
        "active_window_probe_fraction_evaluator_only": (
            active_window_probe_fraction
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
    window_name: str,
    window_output_dir: Path,
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

    plt.xlabel(
        "Attacker remote-probe "
        "release time (ns)"
    )

    plt.ylabel(
        "Remote-request "
        "turnaround time (ns)"
    )

    plt.title(
        "Observation window — "
        f"{window_name}: {victim_tag}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        window_output_dir
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
        "Victim-induced excess latency (ns)"
    )

    plt.title(
        "Victim-induced delay — "
        f"{window_name}: {victim_tag}"
    )

    plt.tight_layout()

    plt.savefig(
        window_output_dir
        / f"{victim_tag}_excess_latency.png",
        dpi=300,
    )

    plt.close()


def save_comparison_plot(
    summary_dataframe:
    pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Save a workload-by-window comparison plot."""

    pivot = (
        summary_dataframe.pivot(
            index="victim_tag",
            columns="window_name",
            values=metric,
        )
        .reindex(
            columns=WINDOW_ORDER
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
        title="Observation window",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# One victim/window experiment
# ============================================================

def run_one_configuration(
    victim_qasm: str,
    window_config: dict,
) -> dict:
    """
    Run:
    1. victim only;
    2. attacker only;
    3. victim and attacker together.
    """

    victim_tag = safe_tag(
        victim_qasm
    )

    window_name = (
        window_config[
            "window_name"
        ]
    )

    window_output_dir = (
        OUTPUT_DIR
        / window_name
    )

    window_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Observation-window sweep: "
        f"{window_name} | "
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
    ) = build_observation_window_schedule(
        window_config,
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
    # Victim-present execution
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

    compared = (
        compare_attacker_runs(
            attacker_only_observations,
            victim_present_observations,
        )
    )

    summary = create_summary(
        victim_qasm=victim_qasm,
        window_config=window_config,
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
    # Save per-configuration outputs
    # --------------------------------------------------------

    compared.to_csv(
        window_output_dir
        / (
            f"{victim_tag}_"
            "attacker_observations.csv"
        ),
        index=False,
    )

    victim_only_ground_truth.to_csv(
        window_output_dir
        / (
            f"{victim_tag}_"
            "victim_only_ground_truth.csv"
        ),
        index=False,
    )

    victim_on_ground_truth.to_csv(
        window_output_dir
        / (
            f"{victim_tag}_"
            "victim_on_ground_truth.csv"
        ),
        index=False,
    )

    summary_path = (
        window_output_dir
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
        window_name,
        window_output_dir,
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

    for window_config in WINDOW_CONFIGS:
        for victim_qasm in VICTIM_QASMS:
            summary = (
                run_one_configuration(
                    victim_qasm,
                    window_config,
                )
            )

            summaries.append(
                summary
            )

    summary_dataframe = pd.DataFrame(
        summaries
    )

    summary_dataframe[
        "window_name"
    ] = pd.Categorical(
        summary_dataframe[
            "window_name"
        ],
        categories=WINDOW_ORDER,
        ordered=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "window_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = (
        OUTPUT_DIR
        / "observation_window_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    schedule_summary = (
        summary_dataframe[
            [
                "window_name",
                "observation_duration_ns",
                "total_probe_rounds",
                "total_attacker_remote_requests",
                "first_remote_probe_release_ns",
                "last_remote_probe_release_ns",
                "realized_remote_probe_rate_per_us",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "observation_duration_ns"
        )
        .reset_index(drop=True)
    )

    schedule_path = (
        OUTPUT_DIR
        / "observation_window_schedule_summary.csv"
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
            "Observation Window: "
            "Average Victim-Induced Latency"
        ),
        filename=(
            "observation_window_"
            "avg_excess_latency.png"
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
            "Observation Window: "
            "Contention Observation Rate"
        ),
        filename=(
            "observation_window_"
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
            "Observation Window: "
            "Victim Slowdown"
        ),
        filename=(
            "observation_window_"
            "victim_slowdown.png"
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
            "Observation Window: "
            "Total Collected Timing Signal"
        ),
        filename=(
            "observation_window_"
            "total_excess_latency.png"
        ),
    )

    save_comparison_plot(
        summary_dataframe,
        metric=(
            "total_attacker_remote_requests"
        ),
        ylabel=(
            "Number of remote probes"
        ),
        title=(
            "Observation Window: "
            "Probe Count"
        ),
        filename=(
            "observation_window_"
            "probe_count.png"
        ),
    )

    print(
        "\n=== Combined observation-window "
        "summary ==="
    )

    display_columns = [
        "victim_tag",
        "window_name",
        "observation_duration_ns",
        "total_attacker_remote_requests",
        "baseline_avg_waiting_time_ns",
        "avg_excess_turnaround_time_ns",
        "total_excess_turnaround_time_ns",
        "max_excess_turnaround_time_ns",
        "delayed_probe_count",
        "contention_observed_fraction",
        "victim_slowdown_ratio",
        "active_window_probe_fraction_evaluator_only",
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

    print(
        "Schedule summary: "
        f"{schedule_path}"
    )


if __name__ == "__main__":
    main()
