#!/usr/bin/env python3
"""
run_sequential_modular_v2.py

Driver for:
- sequential modular trace extraction
- five-module superconducting architecture loading
- STAGE-AWARE sequential trace processing through the architecture

Purpose of v2:
- unlike the earlier sequential runner, this version actually releases work
  stage-by-stage
- later stages do not start until the current stage has been fully serviced
- optional inter-stage gap can be added to amplify timing sensitivity

This should create clearer timing differences relative to static distributed.

Assumes this file is in the same directory as:
    new_arch_fivenode_traceadded.py
"""

import json
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)

FIVE_MODULES = [f"module_{i}" for i in range(5)]


# ============================================================
# Trace extraction
# ============================================================

def build_static_qubit_to_module_map_five_modules(num_qubits: int):
    """
    Assign qubits to the fixed five-module architecture once, in contiguous blocks.
    """
    if num_qubits < 5:
        raise ValueError("Need at least 5 qubits to populate all five modules.")

    qubit_to_module = {}
    block_size = (num_qubits + 5 - 1) // 5  # ceil division

    for q in range(num_qubits):
        module_id = min(q // block_size, 4)
        qubit_to_module[q] = f"module_{module_id}"

    present = set(qubit_to_module.values())
    expected = set(FIVE_MODULES)
    if present != expected:
        qubit_to_module = {}
        for q in range(num_qubits):
            qubit_to_module[q] = f"module_{q % 5}"

    return qubit_to_module


def extract_sequential_modular_trace_from_qasm_five_modules(qasm_file: str):
    """
    Sequential modular execution trace for the fixed five-module architecture.

    Interpretation:
    - qubits are assigned statically to the five modules
    - execution is modeled as stage-by-stage across modules
    - at each step, only one module is considered the active execution module
    - if an operation touches qubits from another module, it is marked as an
      inter-stage transfer / dependency event

    Returns:
        trace: list of dictionaries
        qubit_to_module: dict mapping qubit -> module
    """
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    num_qubits = qc.num_qubits
    qubit_to_module = build_static_qubit_to_module_map_five_modules(num_qubits)

    trace = []
    current_active_module = None
    stage_id = -1

    for step_idx, instruction in enumerate(qc.data):
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        qubit_indices = [qc.find_bit(q).index for q in qargs]
        clbit_indices = [qc.find_bit(c).index for c in cargs]

        touched_modules = sorted({qubit_to_module[q] for q in qubit_indices})
        active_module = touched_modules[0] if touched_modules else None

        if active_module != current_active_module:
            stage_id += 1
            current_active_module = active_module

        is_cross_module = len(touched_modules) > 1
        transfer_event = is_cross_module

        trace_entry = {
            "step": step_idx,
            "stage": stage_id,
            "op_name": op.name,
            "qubits": qubit_indices,
            "clbits": clbit_indices,
            "params": [float(p) if hasattr(p, "__float__") else str(p) for p in op.params],
            "placement_style": "sequential_modular",
            "active_module": active_module,
            "modules_touched": touched_modules,
            "modules": touched_modules,
            "is_cross_module": is_cross_module,
            "cross_module": is_cross_module,
            "transfer_event": transfer_event,
        }
        trace.append(trace_entry)

    return trace, qubit_to_module


def print_static_mapping(qubit_to_module):
    print("=== Static Qubit -> Module Mapping (Five-Module Architecture) ===")
    for q, mod in qubit_to_module.items():
        print(f"q{q} -> {mod}")


def print_sequential_modular_trace(trace, max_lines: int = 30):
    print("\n=== Sequential Modular Execution Trace (v2 input) ===")
    for idx, item in enumerate(trace):
        if idx >= max_lines:
            print(f"... ({len(trace) - max_lines} more lines)")
            break

        print(
            f"step={item['step']:3d} | "
            f"stage={item['stage']:2d} | "
            f"op={item['op_name']:<12s} | "
            f"qubits={item['qubits']} | "
            f"active_module={item['active_module']} | "
            f"modules={item['modules_touched']} | "
            f"cross_module={item['is_cross_module']} | "
            f"transfer_event={item['transfer_event']}"
        )


# ============================================================
# Stage-aware sequential processing
# ============================================================

def process_trace_stagewise(
    arch,
    trace: list,
    inter_stage_gap_ns: int = 0,
):
    """
    Stage-aware sequential processing.

    Key behavior:
    - group trace entries by stage
    - process one stage at a time
    - after routing one stage, fully drain the hub before the next stage begins
    - optionally insert a fixed inter-stage gap

    This makes later stages depend on earlier-stage completion and should
    amplify timing effects compared to the earlier sequential runner.
    """
    normalized_events = []
    completed_stage_order = []

    # Group entries by stage
    stage_to_entries = {}
    for entry in trace:
        stage = entry.get("stage", None)
        if stage is None:
            raise ValueError("Sequential v2 requires every trace entry to have a 'stage' field.")
        stage_to_entries.setdefault(stage, []).append(entry)

    ordered_stages = sorted(stage_to_entries.keys())

    for stage in ordered_stages:
        stage_entries = stage_to_entries[stage]

        # Route all events in this stage
        for entry in stage_entries:
            event = normalize_trace_entry(entry, "sequential_modular")
            normalized_events.append(event)
            arch.route_trace_event(event)

            # Let the architecture advance slightly per event arrival
            arch.advance_architecture_time()

        # Force stage completion before next stage begins
        arch.drain_hub()
        completed_stage_order.append(stage)

        # Optional fixed inter-stage delay
        if inter_stage_gap_ns > 0:
            arch.advance_architecture_time(delta_ns=inter_stage_gap_ns)

    return normalized_events, completed_stage_order


def print_execution_summary_v2(arch, normalized_events: list, completed_stage_order: list):
    total_events = len(normalized_events)
    total_cross = sum(1 for e in normalized_events if e.is_cross_module)
    total_local = total_events - total_cross

    print("\n=== Sequential v2 Execution Summary ===")
    print("trace_type            : sequential_modular_v2")
    print("total_events          :", total_events)
    print("local_events          :", total_local)
    print("cross_module_events   :", total_cross)
    print("completed_stage_order :", completed_stage_order)

    print("\n=== Per-module Local Event Counts ===")
    for mod_id in arch.expected_modules:
        num_local = len(arch.compute_modules[mod_id].local_event_log)
        print(f"{mod_id}: {num_local}")

    print("\n=== Hub Summary ===")
    hub_info = arch.hub.describe()
    for k, v in hub_info.items():
        print(f"{k:22s}: {v}")

    if arch.hub.completed_requests:
        waits = [r.waiting_time_ns for r in arch.hub.completed_requests if r.waiting_time_ns is not None]
        turns = [r.turnaround_time_ns for r in arch.hub.completed_requests if r.turnaround_time_ns is not None]
        if waits:
            print("min_waiting_time_ns   :", min(waits))
            print("max_waiting_time_ns   :", max(waits))
        if turns:
            print("min_turnaround_ns     :", min(turns))
            print("max_turnaround_ns     :", max(turns))


def run_sequential_modular_trace_v2(arch, trace, inter_stage_gap_ns: int = 0):
    events, completed_stage_order = process_trace_stagewise(
        arch=arch,
        trace=trace,
        inter_stage_gap_ns=inter_stage_gap_ns,
    )
    print_execution_summary_v2(arch, events, completed_stage_order)
    return events, completed_stage_order


# ============================================================
# Stats export
# ============================================================

def collect_baseline_stats_v2(arch, trace, trace_type: str):
    total_events = len(trace)
    cross_module_events = sum(
        1 for t in trace
        if t.get("is_cross_module", t.get("cross_module", False))
    )
    local_events = total_events - cross_module_events

    per_module_local = {
        mod_id: len(arch.compute_modules[mod_id].local_event_log)
        for mod_id in arch.expected_modules
    }

    completed_reqs = arch.hub.completed_requests

    waiting_times = [
        r.waiting_time_ns for r in completed_reqs
        if getattr(r, "waiting_time_ns", None) is not None
    ]
    turnaround_times = [
        r.turnaround_time_ns for r in completed_reqs
        if getattr(r, "turnaround_time_ns", None) is not None
    ]

    avg_waiting_time_ns = (
        sum(waiting_times) / len(waiting_times) if waiting_times else 0
    )
    avg_turnaround_time_ns = (
        sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0
    )
    max_waiting_time_ns = max(waiting_times) if waiting_times else 0
    num_waited_requests = sum(1 for w in waiting_times if w > 0)

    stats = {
        "trace_type": trace_type,
        "total_events": total_events,
        "local_events": local_events,
        "cross_module_events": cross_module_events,
        "completed_hub_requests": len(arch.hub.completed_requests),
        "pending_hub_requests": len(arch.hub.pending_requests),
        "active_hub_requests": len(arch.hub.active_requests),
        "per_module_local_events": per_module_local,
        "avg_waiting_time_ns": avg_waiting_time_ns,
        "avg_turnaround_time_ns": avg_turnaround_time_ns,
        "max_waiting_time_ns": max_waiting_time_ns,
        "hub_makespan_ns": getattr(arch.hub, "current_time_ns", 0),
        "hub_current_time_ns": getattr(arch.hub, "current_time_ns", 0),
        "num_waited_requests": num_waited_requests,
    }

    # Sequential-only stage statistics
    stage_counts = {}
    stage_cross_counts = {}
    stage_wait_samples = {}

    for r in completed_reqs:
        stage = getattr(r.original_event, "stage", None)
        if stage is None:
            continue
        if stage not in stage_wait_samples:
            stage_wait_samples[stage] = []
        if getattr(r, "waiting_time_ns", None) is not None:
            stage_wait_samples[stage].append(r.waiting_time_ns)

    for t in trace:
        stage = t.get("stage", None)
        if stage is None:
            continue

        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        is_cross = t.get("is_cross_module", t.get("cross_module", False))
        if is_cross:
            stage_cross_counts[stage] = stage_cross_counts.get(stage, 0) + 1

    stage_avg_waiting_time_ns = {}
    for stage, vals in stage_wait_samples.items():
        stage_avg_waiting_time_ns[stage] = sum(vals) / len(vals) if vals else 0

    stats["stage_counts"] = stage_counts
    stats["stage_cross_counts"] = stage_cross_counts
    stats["stage_avg_waiting_time_ns"] = stage_avg_waiting_time_ns

    return stats


def save_stats_json(stats: dict, out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to: {out_file}")


# ============================================================
# Main
# ============================================================

def main():
    qasm_path = "square_root_n18.qasm"   # change if needed
    inter_stage_gap_ns = 20              # try 0, 20, 50, 100 for comparison

    # 1. Extract sequential modular trace
    sequential_trace, qubit_to_module = extract_sequential_modular_trace_from_qasm_five_modules(
        qasm_file=qasm_path
    )

    # 2. Build a fresh architecture instance
    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=qubit_to_module,
        link_latency_ns=10,
        hub_max_concurrent_transfers=2,
        hub_setup_latency_ns=20,
        hub_transfer_latency_ns=80,
        event_tick_ns=5,
    )

    # 3. Optional prints
    arch.describe()
    arch.print_qubit_mapping()
    arch.print_stick_diagram()
    print()
    print_static_mapping(qubit_to_module)
    print_sequential_modular_trace(sequential_trace, max_lines=25)

    print("\n=== Quick Trace Stats ===")
    print("Total ops:", len(sequential_trace))
    print("Unique ops:", sorted(set(t["op_name"] for t in sequential_trace)))
    print("Unique qubit tuples:", sorted(set(tuple(t["qubits"]) for t in sequential_trace)))
    print("Cross-module ops:", sum(1 for t in sequential_trace if t["is_cross_module"]))
    print("Transfer events:", sum(1 for t in sequential_trace if t["transfer_event"]))
    print("Num stages:", len(set(t["stage"] for t in sequential_trace)))
    print("Inter-stage gap (ns):", inter_stage_gap_ns)

    # 4. Run the stage-aware sequential trace through the architecture
    print("\n============= SEQUENTIAL MODULAR V2 ARCHITECTURE RUN =============")
    run_sequential_modular_trace_v2(
        arch=arch,
        trace=sequential_trace,
        inter_stage_gap_ns=inter_stage_gap_ns,
    )

    # 5. Save stats to a new JSON so you can compare v1 vs v2
    stats = collect_baseline_stats_v2(
        arch=arch,
        trace=sequential_trace,
        trace_type="sequential_modular_v2",
    )
    save_stats_json(stats, "sequential_modular_v2_stats.json")


if __name__ == "__main__":
    main()
