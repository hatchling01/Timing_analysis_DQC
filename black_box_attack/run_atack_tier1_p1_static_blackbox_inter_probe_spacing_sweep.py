#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_inter_probe_spacing_sweep.py

Knob 3: inter-probe spacing.

Fixed:
- Probe 3 light-periodic probe
- 1.00x average rate
- 48 remote probes
- 20,000 ns observation window
- static-distributed victim execution
- P1 disjoint placement
- one serialized shared hub-service slot

Varied:
- uniform spacing
- mild alternating spacing
- strong alternating spacing
- four-probe bursts
- deterministic jittered spacing

Fairness:
Every pattern uses exactly 48 remote probes, the same first and last remote
release times, and the same mean inter-probe interval of 420 ns.

Outputs:
blackbox_window_results/inter_probe_spacing/
"""

from __future__ import annotations

import copy
import json
import random
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
# Configuration
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
    / "inter_probe_spacing"
)

LINK_LATENCY_NS = 10
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80

HUB_MAX_CONCURRENT_TRANSFERS = 1
VICTIM_EVENT_TICK_NS = 5

VICTIM_TRUE_START_NS = 1_000

ATTACKER_ESTIMATED_WINDOW_START_NS = 1_000
ATTACKER_OBSERVATION_DURATION_NS = 20_000

TIE_BREAK_POLICY = "victim_first"
ATTACKER_STEP_BASE = 10_000_000


# ============================================================
# Fixed Probe 3 configuration
# ============================================================

PROBE_3_OPERATIONS = [
    ("h", [0]),
    ("h", [1]),
    ("x", [2]),
    ("x", [3]),
    ("z", [0]),
    ("z", [2]),
    ("cx", [0, 2]),
]

WITHIN_ROUND_EVENT_SPACING_NS = 5

# Probe 3's remote CX is operation index 6.
# It is released 30 ns after the beginning of the round.
REMOTE_REQUEST_OFFSET_NS = (
    (len(PROBE_3_OPERATIONS) - 1)
    * WITHIN_ROUND_EVENT_SPACING_NS
)

NUM_REMOTE_PROBES = 48

BASE_INTER_PROBE_INTERVAL_NS = 420

FIRST_REMOTE_PROBE_RELEASE_NS = (
    ATTACKER_ESTIMATED_WINDOW_START_NS
    + REMOTE_REQUEST_OFFSET_NS
)

TOTAL_REMOTE_PROBE_SPAN_NS = (
    (NUM_REMOTE_PROBES - 1)
    * BASE_INTER_PROBE_INTERVAL_NS
)

LAST_REMOTE_PROBE_RELEASE_NS = (
    FIRST_REMOTE_PROBE_RELEASE_NS
    + TOTAL_REMOTE_PROBE_SPAN_NS
)

OBSERVATION_END_NS = (
    ATTACKER_ESTIMATED_WINDOW_START_NS
    + ATTACKER_OBSERVATION_DURATION_NS
)


SPACING_NAMES = [
    "spacing_uniform",
    "spacing_alternating_mild",
    "spacing_alternating_strong",
    "spacing_burst4",
    "spacing_jittered",
]


if (
    LAST_REMOTE_PROBE_RELEASE_NS
    >= OBSERVATION_END_NS
):
    raise RuntimeError(
        "The fixed 48-probe schedule does not "
        "fit inside the observation window."
    )


# ============================================================
# Spacing-pattern generation
# ============================================================

def validate_intervals(
    spacing_name: str,
    intervals_ns: list[int],
) -> None:
    """Check that every pattern is a fair comparison."""

    expected_count = (
        NUM_REMOTE_PROBES - 1
    )

    if len(intervals_ns) != expected_count:
        raise ValueError(
            f"{spacing_name}: expected "
            f"{expected_count} intervals, "
            f"received {len(intervals_ns)}."
        )

    if any(
        interval <= 0
        for interval in intervals_ns
    ):
        raise ValueError(
            f"{spacing_name}: all intervals "
            "must be positive."
        )

    actual_span_ns = sum(
        intervals_ns
    )

    if (
        actual_span_ns
        != TOTAL_REMOTE_PROBE_SPAN_NS
    ):
        raise ValueError(
            f"{spacing_name}: interval sum is "
            f"{actual_span_ns} ns; expected "
            f"{TOTAL_REMOTE_PROBE_SPAN_NS} ns."
        )


def build_uniform_intervals() -> list[int]:
    """
    Uniform baseline.

    Every remote probe is separated by 420 ns.
    """

    return [
        BASE_INTER_PROBE_INTERVAL_NS
        for _ in range(
            NUM_REMOTE_PROBES - 1
        )
    ]


def build_alternating_intervals(
    short_interval_ns: int,
    long_interval_ns: int,
) -> list[int]:
    """
    Alternate short and long intervals.

    Each short/long pair must average 420 ns.
    """

    required_pair_sum_ns = (
        2 * BASE_INTER_PROBE_INTERVAL_NS
    )

    if (
        short_interval_ns
        + long_interval_ns
        != required_pair_sum_ns
    ):
        raise ValueError(
            "Alternating intervals must "
            "average 420 ns."
        )

    intervals: list[int] = []

    # 23 pairs produce 46 intervals.
    for _ in range(23):
        intervals.extend(
            [
                short_interval_ns,
                long_interval_ns,
            ]
        )

    # The 47th interval remains 420 ns.
    intervals.append(
        BASE_INTER_PROBE_INTERVAL_NS
    )

    return intervals


def build_burst4_intervals() -> list[int]:
    """
    Place probes in groups of four.

    Pattern:
        140, 140, 140, 1260 ns

    The first three intervals form the burst.
    The 1260 ns gap restores the 420 ns mean.
    """

    intervals: list[int] = []

    # Eleven complete four-interval cycles.
    for _ in range(11):
        intervals.extend(
            [
                140,
                140,
                140,
                1_260,
            ]
        )

    # 44 intervals have now been generated.
    # The remaining three must sum to 1260 ns.
    intervals.extend(
        [
            140,
            140,
            980,
        ]
    )

    return intervals


def build_jittered_intervals() -> list[int]:
    """
    Generate deterministic irregular spacing.

    Complementary interval pairs preserve the average:

        (420 - delta) + (420 + delta) = 840 ns

    The fixed random seed makes the experiment repeatable.
    """

    random_generator = random.Random(
        7
    )

    intervals: list[int] = []

    for _ in range(23):
        delta_ns = (
            random_generator.randrange(
                40,
                281,
                10,
            )
        )

        intervals.extend(
            [
                (
                    BASE_INTER_PROBE_INTERVAL_NS
                    - delta_ns
                ),
                (
                    BASE_INTER_PROBE_INTERVAL_NS
                    + delta_ns
                ),
            ]
        )

    random_generator.shuffle(
        intervals
    )

    intervals.append(
        BASE_INTER_PROBE_INTERVAL_NS
    )

    return intervals


def get_spacing_intervals(
    spacing_name: str,
) -> list[int]:
    """Return the selected interval sequence."""

    if spacing_name == "spacing_uniform":
        intervals = (
            build_uniform_intervals()
        )

    elif (
        spacing_name
        == "spacing_alternating_mild"
    ):
        intervals = (
            build_alternating_intervals(
                short_interval_ns=280,
                long_interval_ns=560,
            )
        )

    elif (
        spacing_name
        == "spacing_alternating_strong"
    ):
        intervals = (
            build_alternating_intervals(
                short_interval_ns=140,
                long_interval_ns=700,
            )
        )

    elif spacing_name == "spacing_burst4":
        intervals = (
            build_burst4_intervals()
        )

    elif spacing_name == "spacing_jittered":
        intervals = (
            build_jittered_intervals()
        )

    else:
        raise ValueError(
            "Unknown spacing pattern: "
            f"{spacing_name}"
        )

    validate_intervals(
        spacing_name,
        intervals,
    )

    return intervals


def build_remote_release_times(
    spacing_name: str,
) -> tuple[list[int], list[int]]:
    """
    Convert the inter-probe intervals into
    48 remote-CX release times.
    """

    intervals_ns = (
        get_spacing_intervals(
            spacing_name
        )
    )

    release_times_ns = [
        FIRST_REMOTE_PROBE_RELEASE_NS
    ]

    for interval_ns in intervals_ns:
        next_release_ns = (
            release_times_ns[-1]
            + interval_ns
        )

        release_times_ns.append(
            next_release_ns
        )

    if (
        len(release_times_ns)
        != NUM_REMOTE_PROBES
    ):
        raise RuntimeError(
            f"{spacing_name}: generated "
            f"{len(release_times_ns)} probes; "
            f"expected {NUM_REMOTE_PROBES}."
        )

    if (
        release_times_ns[-1]
        != LAST_REMOTE_PROBE_RELEASE_NS
    ):
        raise RuntimeError(
            f"{spacing_name}: last release "
            f"is {release_times_ns[-1]} ns; "
            f"expected "
            f"{LAST_REMOTE_PROBE_RELEASE_NS} ns."
        )

    return (
        release_times_ns,
        intervals_ns,
    )


# ============================================================
# Architecture and mappings
# ============================================================

def safe_tag(value: str) -> str:
    """Create a filename-safe identifier."""

    stem = Path(value).stem

    return "".join(
        character
        if (
            character.isalnum()
            or character in "_-"
        )
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
    """Map victim qubits across modules."""

    if (
        num_qubits
        < len(module_subset)
    ):
        raise ValueError(
            f"Need at least "
            f"{len(module_subset)} qubits; "
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

    if (
        set(mapping.values())
        != set(module_subset)
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
    """Combine victim and attacker qubits."""

    combined_mapping = dict(
        victim_mapping
    )

    for attacker_qubit, module in (
        attacker_mapping.items()
    ):
        global_attacker_qubit = (
            attacker_qubit
            + victim_num_qubits
        )

        combined_mapping[
            global_attacker_qubit
        ] = module

    return combined_mapping


def build_architecture(
    victim_mapping: dict[int, str],
    victim_num_qubits: int,
    attacker_mapping: dict[int, str],
) -> FiveModuleLocalModularSuperconductingDQC:
    """Construct the five-module system."""

    combined_mapping = (
        combine_qubit_mappings(
            victim_mapping,
            victim_num_qubits,
            attacker_mapping,
        )
    )

    return (
        FiveModuleLocalModularSuperconductingDQC(
            qubit_to_module=(
                combined_mapping
            ),
            link_latency_ns=(
                LINK_LATENCY_NS
            ),
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
# Victim trace
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

    The victim trace is not used to construct
    the attacker spacing schedule.
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
                "modules": (
                    touched_modules
                ),
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
    """Place victim events on the timeline."""

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
# Attacker schedule
# ============================================================

def build_spacing_schedule(
    spacing_name: str,
    victim_num_qubits: int,
) -> tuple[
    list[dict],
    dict[int, str],
    dict[int, dict],
    dict,
]:
    """
    Construct the 48 Probe 3 rounds.

    The provided spacing sequence refers directly
    to the remote CX release times.
    """

    attacker_mapping = (
        attacker_qubit_map()
    )

    (
        remote_release_times_ns,
        intervals_ns,
    ) = build_remote_release_times(
        spacing_name
    )

    scheduled_events: list[dict] = []

    cross_step_metadata: dict[
        int,
        dict,
    ] = {}

    attacker_event_index = 0

    for probe_id, (
        remote_release_time_ns
    ) in enumerate(
        remote_release_times_ns
    ):
        round_start_time_ns = (
            remote_release_time_ns
            - REMOTE_REQUEST_OFFSET_NS
        )

        for operation_index, (
            operation_name,
            local_qubits,
        ) in enumerate(
            PROBE_3_OPERATIONS
        ):
            release_time_ns = (
                round_start_time_ns
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

            step = (
                ATTACKER_STEP_BASE
                + attacker_event_index
            )

            trace_entry = {
                "step": step,
                "op_name": (
                    operation_name
                ),
                "qubits": (
                    global_qubits
                ),
                "clbits": [],
                "params": [],
                "placement_style": (
                    "static_distributed"
                ),
                "modules_touched": (
                    touched_modules
                ),
                "modules": (
                    touched_modules
                ),
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
                preceding_interval_ns = (
                    intervals_ns[
                        probe_id - 1
                    ]
                    if probe_id > 0
                    else None
                )

                cross_step_metadata[
                    step
                ] = {
                    "attacker_request_id": (
                        probe_id
                    ),
                    "probe_id": (
                        probe_id
                    ),
                    "spacing_name": (
                        spacing_name
                    ),
                    "round_start_time_ns": (
                        round_start_time_ns
                    ),
                    "request_release_time_ns": (
                        remote_release_time_ns
                    ),
                    "preceding_interval_ns": (
                        preceding_interval_ns
                    ),
                }

            attacker_event_index += 1

    interval_series = pd.Series(
        intervals_ns,
        dtype=float,
    )

    interval_mean_ns = float(
        interval_series.mean()
    )

    interval_std_ns = float(
        interval_series.std(
            ddof=0
        )
    )

    schedule_metadata = {
        "spacing_name": spacing_name,
        "total_probe_rounds": (
            NUM_REMOTE_PROBES
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
        "mean_inter_probe_interval_ns": (
            interval_mean_ns
        ),
        "min_inter_probe_interval_ns": int(
            interval_series.min()
        ),
        "max_inter_probe_interval_ns": int(
            interval_series.max()
        ),
        "std_inter_probe_interval_ns": (
            interval_std_ns
        ),
        "inter_probe_interval_cv": (
            interval_std_ns
            / interval_mean_ns
        ),
    }

    return (
        scheduled_events,
        attacker_mapping,
        cross_step_metadata,
        schedule_metadata,
    )


# ============================================================
# Wall-clock scheduler
# ============================================================

def scheduled_event_sort_key(
    scheduled_event: dict,
) -> tuple[int, int, int]:
    """Sort events by release time."""

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
    Advance to a release timestamp while handling
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
    """Execute the combined timeline."""

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
    """Collect attacker-visible timings."""

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
                "No attacker metadata "
                f"for step {trace_step}."
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
    """Collect evaluator-only victim timing."""

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

    if (
        len(combined)
        != len(attacker_only)
    ):
        raise RuntimeError(
            "Attacker-only and "
            "victim-present request "
            "counts differ."
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
    """Measure final victim remote completion."""

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
    spacing_name: str,
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
    """Create one victim/spacing summary."""

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
        "victim_qasm": (
            victim_qasm
        ),
        "victim_tag": safe_tag(
            victim_qasm
        ),
        "knob": (
            "inter_probe_spacing"
        ),
        "spacing_name": (
            spacing_name
        ),
        "probe_name": (
            "probe_3_light_periodic"
        ),
        "rate_multiplier": 1.0,
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
    spacing_name: str,
    spacing_output_dir: Path,
) -> None:
    """Save timing traces."""

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
        "Probe spacing — "
        f"{spacing_name}: {victim_tag}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        spacing_output_dir
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
        f"{spacing_name}: {victim_tag}"
    )

    plt.tight_layout()

    plt.savefig(
        spacing_output_dir
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
    """Save a grouped spacing comparison."""

    pivot = (
        summary_dataframe.pivot(
            index="victim_tag",
            columns="spacing_name",
            values=metric,
        )
        .reindex(
            columns=SPACING_NAMES
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
        title="Spacing pattern",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


def create_spacing_schedule_dataframe(
) -> pd.DataFrame:
    """Save the exact schedules used."""

    rows: list[dict] = []

    for spacing_name in SPACING_NAMES:
        (
            release_times_ns,
            intervals_ns,
        ) = build_remote_release_times(
            spacing_name
        )

        for probe_id, release_time_ns in enumerate(
            release_times_ns
        ):
            preceding_interval_ns = (
                intervals_ns[
                    probe_id - 1
                ]
                if probe_id > 0
                else None
            )

            rows.append(
                {
                    "spacing_name": (
                        spacing_name
                    ),
                    "probe_id": (
                        probe_id
                    ),
                    "remote_probe_"
                    "release_time_ns": (
                        release_time_ns
                    ),
                    "preceding_interval_ns": (
                        preceding_interval_ns
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def save_spacing_pattern_plot(
    schedule_dataframe:
    pd.DataFrame,
) -> None:
    """Plot every spacing sequence."""

    plt.figure(
        figsize=(14, 7)
    )

    for spacing_name in SPACING_NAMES:
        subset = schedule_dataframe[
            (
                schedule_dataframe[
                    "spacing_name"
                ] == spacing_name
            )
            & (
                schedule_dataframe[
                    "probe_id"
                ] > 0
            )
        ]

        plt.plot(
            subset["probe_id"],
            subset[
                "preceding_interval_ns"
            ],
            marker="o",
            markersize=2.5,
            linewidth=1,
            label=spacing_name,
        )

    plt.xlabel(
        "Probe index"
    )

    plt.ylabel(
        "Preceding inter-probe "
        "interval (ns)"
    )

    plt.title(
        "Inter-Probe Spacing Patterns "
        "(48 Probes, Fixed Mean Rate)"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "inter_probe_spacing_patterns.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Experiment execution
# ============================================================

def run_one_configuration(
    victim_qasm: str,
    spacing_name: str,
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

    spacing_output_dir = (
        OUTPUT_DIR
        / spacing_name
    )

    spacing_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== Inter-probe spacing: "
        f"{spacing_name} | "
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
    ) = build_spacing_schedule(
        spacing_name,
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
        spacing_name=spacing_name,
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
        spacing_output_dir
        / (
            f"{victim_tag}_"
            "attacker_observations.csv"
        ),
        index=False,
    )

    victim_only_ground_truth.to_csv(
        spacing_output_dir
        / (
            f"{victim_tag}_"
            "victim_only_ground_truth.csv"
        ),
        index=False,
    )

    victim_on_ground_truth.to_csv(
        spacing_output_dir
        / (
            f"{victim_tag}_"
            "victim_on_ground_truth.csv"
        ),
        index=False,
    )

    summary_path = (
        spacing_output_dir
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
        spacing_name,
        spacing_output_dir,
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

    # Save the exact release schedules first.
    schedule_dataframe = (
        create_spacing_schedule_dataframe()
    )

    schedule_path = (
        OUTPUT_DIR
        / "inter_probe_spacing_schedules.csv"
    )

    schedule_dataframe.to_csv(
        schedule_path,
        index=False,
    )

    save_spacing_pattern_plot(
        schedule_dataframe
    )

    summaries: list[dict] = []

    for spacing_name in SPACING_NAMES:
        for victim_qasm in VICTIM_QASMS:
            summary = (
                run_one_configuration(
                    victim_qasm,
                    spacing_name,
                )
            )

            summaries.append(
                summary
            )

    summary_dataframe = pd.DataFrame(
        summaries
    )

    summary_dataframe[
        "spacing_name"
    ] = pd.Categorical(
        summary_dataframe[
            "spacing_name"
        ],
        categories=SPACING_NAMES,
        ordered=True,
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            [
                "spacing_name",
                "victim_tag",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = (
        OUTPUT_DIR
        / "inter_probe_spacing_summary.csv"
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
            "Inter-Probe Spacing: "
            "Average Victim-Induced Latency"
        ),
        filename=(
            "inter_probe_spacing_"
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
            "probes delayed"
        ),
        title=(
            "Inter-Probe Spacing: "
            "Contention Observation Rate"
        ),
        filename=(
            "inter_probe_spacing_"
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
            "Inter-Probe Spacing: "
            "Victim Slowdown"
        ),
        filename=(
            "inter_probe_spacing_"
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
            "Inter-Probe Spacing: "
            "Attacker Self-Contention"
        ),
        filename=(
            "inter_probe_spacing_"
            "baseline_self_wait.png"
        ),
    )

    print(
        "\n=== Combined inter-probe "
        "spacing summary ==="
    )

    display_columns = [
        "victim_tag",
        "spacing_name",
        "mean_inter_probe_interval_ns",
        "min_inter_probe_interval_ns",
        "max_inter_probe_interval_ns",
        "std_inter_probe_interval_ns",
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

    print(
        "Spacing schedules: "
        f"{schedule_path}"
    )


if __name__ == "__main__":
    main()
