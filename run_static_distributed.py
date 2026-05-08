#!/usr/bin/env python3
"""
run_static_distributed.py

Driver for:
- static distributed trace extraction
- five-module superconducting architecture loading
- static distributed trace processing through the architecture

Assumes this file is in the same directory as:
    new_arch_fivenode_traceadded.py
"""

from qiskit import QuantumCircuit
import json
from pathlib import Path

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    run_static_distributed_trace,
)

FIVE_MODULES = [f"module_{i}" for i in range(5)]


def build_static_qubit_to_module_map_five_modules(num_qubits: int):
    """
    Static distributed placement for the fixed five-module architecture.

    Qubits are assigned once and remain fixed for the whole execution.
    The assignment is done in contiguous blocks across:
        module_0, module_1, module_2, module_3, module_4
    """
    if num_qubits < 5:
        raise ValueError("Need at least 5 qubits to populate all five modules.")

    qubit_to_module = {}
    block_size = (num_qubits + 5 - 1) // 5  # ceil division

    for q in range(num_qubits):
        module_id = min(q // block_size, 4)
        qubit_to_module[q] = f"module_{module_id}"

    # Ensure all five modules appear
    present = set(qubit_to_module.values())
    expected = set(FIVE_MODULES)
    if present != expected:
        qubit_to_module = {}
        for q in range(num_qubits):
            qubit_to_module[q] = f"module_{q % 5}"

    return qubit_to_module


def extract_static_distributed_trace_from_qasm_five_modules(qasm_file: str):
    """
    Extract a trace for static distributed execution under the fixed
    five-module architecture.

    Static distributed execution means:
    - qubits are assigned to modules once at the beginning
    - that placement never changes
    - any gate touching qubits from different modules is marked cross-module

    Returns:
        trace: list of dictionaries
        qubit_to_module: dict mapping qubit index -> module name
    """
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    num_qubits = qc.num_qubits

    qubit_to_module = build_static_qubit_to_module_map_five_modules(num_qubits)
    trace = []

    for step_idx, instruction in enumerate(qc.data):
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        qubit_indices = [qc.find_bit(q).index for q in qargs]
        clbit_indices = [qc.find_bit(c).index for c in cargs]

        touched_modules = sorted({qubit_to_module[q] for q in qubit_indices})
        is_cross_module = len(touched_modules) > 1

        trace_entry = {
            "step": step_idx,
            "op_name": op.name,
            "qubits": qubit_indices,
            "clbits": clbit_indices,
            "params": [float(p) if hasattr(p, "__float__") else str(p) for p in op.params],
            "placement_style": "static_distributed",
            "modules_touched": touched_modules,
            "modules": touched_modules,
            "is_cross_module": is_cross_module,
            "cross_module": is_cross_module,
            "communication_event": is_cross_module,
        }
        trace.append(trace_entry)

    return trace, qubit_to_module


def print_static_mapping(qubit_to_module):
    print("=== Static Qubit -> Module Mapping (Five-Module Architecture) ===")
    for q, mod in qubit_to_module.items():
        print(f"q{q} -> {mod}")


def print_static_distributed_trace(trace, max_lines: int = 30):
    print("\n=== Static Distributed Execution Trace ===")
    for idx, item in enumerate(trace):
        if idx >= max_lines:
            print(f"... ({len(trace) - max_lines} more lines)")
            break

        print(
            f"step={item['step']:3d} | "
            f"op={item['op_name']:<12s} | "
            f"qubits={item['qubits']} | "
            f"modules={item['modules_touched']} | "
            f"cross_module={item['is_cross_module']} | "
            f"clbits={item['clbits']} | "
            f"params={item['params']}"
        )


def collect_baseline_stats(arch, trace, trace_type: str):
    """
    Collect baseline statistics after a trace has been processed
    through the five-module architecture.
    """
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

        # New hub-performance metrics
        "avg_waiting_time_ns": avg_waiting_time_ns,
        "avg_turnaround_time_ns": avg_turnaround_time_ns,
        "max_waiting_time_ns": max_waiting_time_ns,
        "hub_makespan_ns": getattr(arch.hub, "current_time_ns", 0),
        "hub_current_time_ns": getattr(arch.hub, "current_time_ns", 0),
        "num_waited_requests": num_waited_requests,
    }

    if trace_type == "sequential_modular":
        stage_counts = {}
        stage_cross_counts = {}
        stage_wait_samples = {}

        # Map request -> stage using original event stage
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

def main():
    qasm_path = "square_root_n18.qasm"   # change if needed

    # 1. Extract static distributed trace
    static_trace, qubit_to_module = extract_static_distributed_trace_from_qasm_five_modules(
        qasm_file=qasm_path
    )

    # 2. Build a fresh architecture instance
    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=qubit_to_module,
        link_latency_ns=10,
    )

    # 3. Optional prints
    arch.describe()
    arch.print_qubit_mapping()
    arch.print_stick_diagram()
    print()
    print_static_mapping(qubit_to_module)
    print_static_distributed_trace(static_trace, max_lines=25)

    print("\n=== Quick Trace Stats ===")
    print("Total ops:", len(static_trace))
    print("Unique ops:", sorted(set(t["op_name"] for t in static_trace)))
    print("Unique qubit tuples:", sorted(set(tuple(t["qubits"]) for t in static_trace)))
    print("Cross-module ops:", sum(1 for t in static_trace if t["is_cross_module"]))

    # 4. Run the trace through the fixed architecture
    print("\n============= STATIC DISTRIBUTED ARCHITECTURE RUN =============")
    run_static_distributed_trace(arch, static_trace)

    stats = collect_baseline_stats(
        arch=arch,
        trace=static_trace,
        trace_type="static_distributed",
    )

    save_stats_json(stats, "static_distributed_stats.json")


if __name__ == "__main__":
    main()
