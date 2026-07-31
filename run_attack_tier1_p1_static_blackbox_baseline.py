#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_blackbox_baseline.py

First Phase-1 black-box experiment.

Fixed:
- Five-module Architecture M
- Static-distributed victim
- P1 disjoint placement
- Probe 3
- R1-like uniform wall-clock probing
- Absolute D100 attacker budget

Important:
The victim QASM is parsed only to generate victim traffic and evaluator-only
ground truth. The attacker schedule does not use victim length, victim event
count, victim cross-operation count, or victim stage information.
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
# Configuration
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

OUTPUT_DIR = Path("blackbox_results") / "baseline"

LINK_LATENCY_NS = 10
HUB_MAX_CONCURRENT_TRANSFERS = 2
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80
VICTIM_EVENT_TICK_NS = 5

# The existing A5-like configuration inserts one attacker trace event
# after every 12 victim events. With a 5 ns victim-event tick, that
# corresponds approximately to one attacker event every 60 ns.
#
# Here, 60 ns is defined directly in wall-clock time and is completely
# independent of victim execution progress.
ATTACKER_EVENT_SPACING_NS = 60

# Probe 3 contains seven events per round:
#   h, h, x, x, z, z, cx
#
# Therefore, cross-module CX requests occur every:
#   7 * 60 ns = 420 ns
TOTAL_PROBE_ROUNDS = 100  # Absolute D100 budget

ATTACKER_START_TIME_NS = 0
VICTIM_START_OFFSET_NS = 0

# Determines which event is routed first when victim and attacker
# events have exactly the same timestamp.
TIE_BREAK_POLICY = "victim_first"

# Prevent attacker trace-step values from overlapping victim steps.
ATTACKER_STEP_BASE = 10_000_000


# ============================================================
# Mapping and architecture helpers
# ============================================================

def safe_tag(path_str: str) -> str:
    """Return a filename-safe identifier."""
    stem = Path(path_str).stem
    return "".join(
        character
        if character.isalnum() or character in "_-"
        else "_"
        for character in stem
    )


def build_subset_qubit_map(
    num_qubits: int,
    modules: list[str],
) -> dict[int, str]:
    """
    Map a circuit across a selected module subset using contiguous blocks.
    """
    if num_qubits < len(modules):
        raise ValueError(
            f"Need at least {len(modules)} qubits for modules {modules}; "
            f"received {num_qubits}."
        )

    block_size = (
        num_qubits + len(modules) - 1
    ) // len(modules)

    qubit_to_module = {
        qubit: modules[
            min(
                qubit // block_size,
                len(modules) - 1,
            )
        ]
        for qubit in range(num_qubits)
    }

    # Fallback for awkward circuit sizes that fail to populate all modules.
    if set(qubit_to_module.values()) != set(modules):
        qubit_to_module = {
            qubit: modules[qubit % len(modules)]
            for qubit in range(num_qubits)
        }

    return qubit_to_module


def attacker_qubit_map() -> dict[int, str]:
    """
    Probe-3 attacker allocation under P1.

    Attacker qubits 0 and 1 belong to module_3.
    Attacker qubits 2 and 3 belong to module_4.
    """
    return {
        0: "module_3",
        1: "module_3",
        2: "module_4",
        3: "module_4",
    }


def combine_mappings(
    victim_qubit_to_module: dict[int, str],
    victim_num_qubits: int,
    attacker_qubit_to_module: dict[int, str],
) -> dict[int, str]:
    """
    Combine victim and attacker qubit mappings into one global mapping.
    """
    combined_mapping = dict(victim_qubit_to_module)

    for attacker_qubit, module in attacker_qubit_to_module.items():
        global_qubit = attacker_qubit + victim_num_qubits
        combined_mapping[global_qubit] = module

    return combined_mapping


def make_architecture(
    victim_qubit_to_module: dict[int, str],
    victim_num_qubits: int,
    attacker_qubit_to_module: dict[int, str],
) -> FiveModuleLocalModularSuperconductingDQC:
    """
    Instantiate the existing five-module architecture.
    """
    combined_mapping = combine_mappings(
        victim_qubit_to_module,
        victim_num_qubits,
        attacker_qubit_to_module,
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

def extract_static_trace(
    qasm_file: str,
) -> tuple[
    list[dict],
    dict[int, str],
    int,
    int,
]:
    """
    Parse the victim QASM and generate simulator traffic.

    The returned trace is evaluator-side ground truth. It is used only
    to create the victim workload and evaluate the attack. It is never
    passed to the attacker scheduler.
    """
    qasm_path = Path(qasm_file)

    if not qasm_path.exists():
        raise FileNotFoundError(
            f"Missing QASM file: {qasm_path.resolve()}"
        )

    quantum_circuit = QuantumCircuit.from_qasm_file(
        str(qasm_path)
    )

    qubit_to_module = build_subset_qubit_map(
        quantum_circuit.num_qubits,
        VICTIM_MODULES,
    )

    trace: list[dict] = []
    cross_module_operations = 0

    for step_index, instruction in enumerate(
        quantum_circuit.data
    ):
        operation = instruction.operation

        qubits = [
            quantum_circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        clbits = [
            quantum_circuit.find_bit(clbit).index
            for clbit in instruction.clbits
        ]

        if not qubits:
            continue

        touched_modules = sorted(
            {
                qubit_to_module[qubit]
                for qubit in qubits
            }
        )

        is_cross_module = len(touched_modules) > 1

        if is_cross_module:
            cross_module_operations += 1

        trace.append(
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
        trace,
        qubit_to_module,
        quantum_circuit.num_qubits,
        cross_module_operations,
    )


def schedule_victim_trace(
    victim_trace: list[dict],
) -> list[dict]:
    """
    Assign evaluator-side timestamps to victim events.

    These timestamps are hidden from the attacker.
    """
    scheduled_events: list[dict] = []

    for event_index, trace_entry in enumerate(victim_trace):
        release_time_ns = (
            VICTIM_START_OFFSET_NS
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
# Black-box Probe-3 schedule
# ============================================================

def build_blackbox_probe3_schedule(
    victim_num_qubits: int,
) -> tuple[
    list[dict],
    dict[int, str],
    dict[int, dict],
]:
    """
    Build Probe 3 using attacker-controlled wall-clock timestamps.

    One Probe-3 round is:

        h(0)
        h(1)
        x(2)
        x(3)
        z(0)
        z(2)
        cx(0, 2)

    The final CX is the actual cross-module timing probe.

    No victim trace properties are accessed in this function.
    """
    attacker_qubit_to_module = attacker_qubit_map()

    operations = [
        ("h", [0]),
        ("h", [1]),
        ("x", [2]),
        ("x", [3]),
        ("z", [0]),
        ("z", [2]),
        ("cx", [0, 2]),
    ]

    scheduled_events: list[dict] = []

    # Maps each attacker cross-module trace step to its probe ID.
    cross_step_to_probe: dict[int, dict] = {}

    global_attacker_event_index = 0

    for probe_id in range(TOTAL_PROBE_ROUNDS):
        round_start_time_ns = (
            ATTACKER_START_TIME_NS
            + probe_id
            * len(operations)
            * ATTACKER_EVENT_SPACING_NS
        )

        for operation_name, local_qubits in operations:
            release_time_ns = (
                ATTACKER_START_TIME_NS
                + global_attacker_event_index
                * ATTACKER_EVENT_SPACING_NS
            )

            global_qubits = [
                qubit + victim_num_qubits
                for qubit in local_qubits
            ]

            touched_modules = sorted(
                {
                    attacker_qubit_to_module[qubit]
                    for qubit in local_qubits
                }
            )

            is_cross_module = len(touched_modules) > 1

            step = (
                ATTACKER_STEP_BASE
                + global_attacker_event_index
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
                    "sequence_index": (
                        global_attacker_event_index
                    ),
                    "entry": trace_entry,
                }
            )

            if is_cross_module:
                cross_step_to_probe[step] = {
                    "probe_id": probe_id,
                    "round_start_time_ns": (
                        round_start_time_ns
                    ),
                    "request_release_time_ns": (
                        release_time_ns
                    ),
                }

            global_attacker_event_index += 1

    return (
        scheduled_events,
        attacker_qubit_to_module,
        cross_step_to_probe,
    )


# ============================================================
# Independent wall-clock simulation
# ============================================================

def event_sort_key(
    scheduled_event: dict,
) -> tuple[int, int, int]:
    """
    Sort events by release time and deterministic tie-break policy.
    """
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
            "TIE_BREAK_POLICY must be "
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
    Advance the architecture to a target wall-clock time.

    This function advances through intermediate request-completion
    timestamps instead of jumping directly to the next event. That
    preserves correct hub queue admission and completion ordering.
    """
    current_time_ns = architecture.hub.current_time_ns

    if target_time_ns < current_time_ns:
        raise RuntimeError(
            f"Cannot move backwards from {current_time_ns} "
            f"to {target_time_ns}."
        )

    while architecture.hub.current_time_ns < target_time_ns:
        active_end_times = [
            request.end_time_ns
            for request in architecture.hub.active_requests
            if request.end_time_ns is not None
        ]

        if active_end_times:
            next_completion_ns = min(active_end_times)

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


def execute_scheduled_events(
    architecture,
    scheduled_events: list[dict],
) -> None:
    """
    Execute independently timestamped victim and attacker events.

    The attacker schedule has already been constructed before the
    simulator merges the two streams.
    """
    ordered_events = sorted(
        scheduled_events,
        key=event_sort_key,
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
# Attacker observation extraction
# ============================================================

def collect_attacker_observations(
    architecture,
    cross_step_to_probe: dict[int, dict],
    run_type: str,
) -> pd.DataFrame:
    """
    Extract only requests generated from attacker modules.
    """
    rows: list[dict] = []

    attacker_requests = [
        request
        for request in architecture.hub.completed_requests
        if request.source_module in ATTACKER_MODULES
    ]

    for request in attacker_requests:
        attacker_step = request.original_event.step

        probe_metadata = cross_step_to_probe.get(
            attacker_step
        )

        if probe_metadata is None:
            raise KeyError(
                "Missing probe mapping for attacker step "
                f"{attacker_step}."
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
                "source_module": (
                    request.source_module
                ),
                "target_modules": ",".join(
                    request.target_modules
                ),
                "request_id": request.request_id,
            }
        )

    observations = (
        pd.DataFrame(rows)
        .sort_values("probe_id")
        .reset_index(drop=True)
    )

    if len(observations) != TOTAL_PROBE_ROUNDS:
        raise RuntimeError(
            f"Expected {TOTAL_PROBE_ROUNDS} attacker "
            f"requests, but observed {len(observations)}."
        )

    return observations


# ============================================================
# Evaluator-only victim ground truth
# ============================================================

def collect_victim_ground_truth(
    architecture,
) -> pd.DataFrame:
    """
    Save the true victim remote-operation timeline.

    This output is evaluator-only and must not be used by the
    black-box attacker.
    """
    rows: list[dict] = []

    victim_requests = [
        request
        for request in architecture.hub.completed_requests
        if request.source_module in VICTIM_MODULES
    ]

    for event_id, request in enumerate(victim_requests):
        rows.append(
            {
                "victim_remote_event_id": event_id,
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
        return pd.DataFrame(columns=columns)

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
# Calibration subtraction
# ============================================================

def merge_baseline_and_victim(
    baseline_observations: pd.DataFrame,
    victim_on_observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each victim-on probe against the same probe in an
    attacker-only calibration run.
    """
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
# Summary metrics
# ============================================================

def build_summary(
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
    """
    Produce aggregate results while preserving the per-probe CSV.
    """
    return {
        "victim_qasm": victim_qasm,
        "workload_type": "static_distributed",
        "placement": "P1_disjoint",
        "threat_model": (
            "tier1_blackbox_wallclock"
        ),
        "probe": (
            "probe3_R1_uniform_blackbox"
        ),
        "duration_name": (
            f"D{TOTAL_PROBE_ROUNDS}"
        ),
        "total_probe_rounds": (
            TOTAL_PROBE_ROUNDS
        ),
        "attacker_event_spacing_ns": (
            ATTACKER_EVENT_SPACING_NS
        ),
        "effective_cross_probe_spacing_ns": (
            7 * ATTACKER_EVENT_SPACING_NS
        ),
        "attacker_start_time_ns": (
            ATTACKER_START_TIME_NS
        ),
        "victim_start_offset_ns": (
            VICTIM_START_OFFSET_NS
        ),
        "tie_break_policy": (
            TIE_BREAK_POLICY
        ),
        "victim_total_events": (
            len(victim_trace)
        ),
        "victim_cross_module_ops": (
            victim_cross_operations
        ),
        "victim_completed_remote_requests": (
            len(victim_ground_truth)
        ),
        "attacker_completed_requests": (
            len(victim_on_observations)
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
        "attacker_avg_waiting_time_ns": float(
            victim_on_observations[
                "waiting_time_ns"
            ].mean()
        ),
        "attacker_avg_turnaround_time_ns": float(
            victim_on_observations[
                "turnaround_time_ns"
            ].mean()
        ),
        "attacker_max_waiting_time_ns": float(
            victim_on_observations[
                "waiting_time_ns"
            ].max()
        ),
        "attacker_waited_fraction": float(
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
        "baseline_hub_makespan_ns": (
            baseline_architecture
            .hub.current_time_ns
        ),
        "victim_on_hub_makespan_ns": (
            victim_architecture
            .hub.current_time_ns
        ),
    }


# ============================================================
# Plotting
# ============================================================

def save_plots(
    combined_observations: pd.DataFrame,
    victim_tag: str,
) -> None:
    """
    Save the raw timing trace and baseline-subtracted timing trace.
    """
    request_times = combined_observations[
        "planned_request_release_time_ns"
    ]

    plt.figure(figsize=(13, 6))

    plt.plot(
        request_times,
        combined_observations[
            "victim_on_turnaround_time_ns"
        ],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Victim ON",
    )

    plt.plot(
        request_times,
        combined_observations[
            "baseline_turnaround_time_ns"
        ],
        linewidth=1,
        label="Attacker-only baseline",
    )

    plt.xlabel(
        "Attacker request release time (ns)"
    )
    plt.ylabel(
        "Observed turnaround time (ns)"
    )
    plt.title(
        f"Black-box Probe-3 Timing Trace: "
        f"{victim_tag}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{victim_tag}_blackbox_timing_trace.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(13, 5))

    plt.plot(
        request_times,
        combined_observations[
            "excess_turnaround_time_ns"
        ],
        marker="o",
        markersize=3,
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
        "Excess turnaround time (ns)"
    )
    plt.title(
        f"Victim-Induced Excess Probe Latency: "
        f"{victim_tag}"
    )
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"{victim_tag}_blackbox_excess_latency.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Run one victim
# ============================================================

def run_one_victim(
    victim_qasm: str,
) -> dict:
    """
    Run attacker-only calibration and victim-on black-box execution.
    """
    victim_tag = safe_tag(victim_qasm)

    print(
        f"\n=== Black-box baseline: "
        f"{victim_qasm} ==="
    )

    (
        victim_trace,
        victim_qubit_to_module,
        victim_num_qubits,
        victim_cross_operations,
    ) = extract_static_trace(
        victim_qasm
    )

    (
        attacker_schedule,
        attacker_qubit_to_module,
        cross_step_to_probe,
    ) = build_blackbox_probe3_schedule(
        victim_num_qubits
    )

    # --------------------------------------------------------
    # Attacker-only calibration run
    # --------------------------------------------------------

    baseline_architecture = make_architecture(
        victim_qubit_to_module,
        victim_num_qubits,
        attacker_qubit_to_module,
    )

    execute_scheduled_events(
        baseline_architecture,
        copy.deepcopy(attacker_schedule),
    )

    baseline_observations = (
        collect_attacker_observations(
            baseline_architecture,
            cross_step_to_probe,
            "attacker_only_baseline",
        )
    )

    # --------------------------------------------------------
    # Victim-on black-box run
    # --------------------------------------------------------

    victim_architecture = make_architecture(
        victim_qubit_to_module,
        victim_num_qubits,
        attacker_qubit_to_module,
    )

    victim_schedule = schedule_victim_trace(
        victim_trace
    )

    merged_schedule = (
        victim_schedule
        + copy.deepcopy(attacker_schedule)
    )

    execute_scheduled_events(
        victim_architecture,
        merged_schedule,
    )

    victim_on_observations = (
        collect_attacker_observations(
            victim_architecture,
            cross_step_to_probe,
            "victim_on",
        )
    )

    victim_ground_truth = (
        collect_victim_ground_truth(
            victim_architecture
        )
    )

    combined_observations = (
        merge_baseline_and_victim(
            baseline_observations,
            victim_on_observations,
        )
    )

    summary = build_summary(
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
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    attacker_output_path = (
        OUTPUT_DIR
        / (
            f"{victim_tag}_blackbox_"
            f"attacker_observations.csv"
        )
    )

    victim_ground_truth_path = (
        OUTPUT_DIR
        / (
            f"{victim_tag}_blackbox_"
            f"victim_ground_truth.csv"
        )
    )

    summary_output_path = (
        OUTPUT_DIR
        / f"{victim_tag}_blackbox_summary.json"
    )

    combined_observations.to_csv(
        attacker_output_path,
        index=False,
    )

    victim_ground_truth.to_csv(
        victim_ground_truth_path,
        index=False,
    )

    with summary_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=2,
        )

    save_plots(
        combined_observations,
        victim_tag,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Saved attacker observations: "
        f"{attacker_output_path}"
    )

    print(
        f"Saved victim ground truth: "
        f"{victim_ground_truth_path}"
    )

    print(
        f"Saved summary: "
        f"{summary_output_path}"
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

    for victim_qasm in VICTIM_QASMS:
        summary = run_one_victim(
            victim_qasm
        )

        summaries.append(
            summary
        )

    combined_summary = pd.DataFrame(
        summaries
    )

    combined_summary_path = (
        OUTPUT_DIR
        / "blackbox_baseline_summary.csv"
    )

    combined_summary.to_csv(
        combined_summary_path,
        index=False,
    )

    print(
        "\n=== Combined black-box "
        "baseline summary ==="
    )

    print(
        combined_summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved combined summary: "
        f"{combined_summary_path}"
    )


if __name__ == "__main__":
    main()
