#!/usr/bin/env python3
"""
run_monolithic.py

Driver for:
- monolithic local trace extraction
- five-module superconducting architecture loading
- monolithic trace processing through the architecture
- stats export for plotting

Key fix:
For the monolithic run, ALL circuit qubits are mapped to module_0 in the
architecture instance, so the monolithic workload does not incorrectly
generate hub traffic.

Assumes this file is in the same directory as:
    new_arch_fivenode_traceadded.py
"""

import json
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    run_monolithic_trace,
)

FIVE_MODULES = [f"module_{i}" for i in range(5)]


def extract_monolithic_trace_from_qasm(qasm_file: str, target_module: str = "module_0"):
    """
    Extract a monolithic-local execution trace under the fixed five-module architecture.

    Monolithic local execution means:
    - the full circuit runs on exactly one compute module
    - no cross-module communication is introduced
    - every event is tagged to the chosen target_module

    Returns
    -------
    trace : list[dict]
    num_qubits : int
    """
    if target_module not in FIVE_MODULES:
        raise ValueError(
            f"target_module must be one of {FIVE_MODULES}, got {target_module}"
        )

    qc = QuantumCircuit.from_qasm_file(qasm_file)

    trace = []
    for step_idx, instruction in enumerate(qc.data):
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        qubit_indices = [qc.find_bit(q).index for q in qargs]
        clbit_indices = [qc.find_bit(c).index for c in cargs]

        trace_entry = {
            "step": step_idx,
            "op_name": op.name,
            "qubits": qubit_indices,
            "clbits": clbit_indices,
            "params": [float(p) if hasattr(p, "__float__") else str(p) for p in op.params],
            "placement_style": "monolithic_local",
            "module": target_module,
            "modules_touched": [target_module],
            "is_cross_module": False,
        }
        trace.append(trace_entry)

    return trace, qc.num_qubits


def build_monolithic_arch_mapping(num_qubits: int, target_module: str = "module_0"):
    """
    Build the architecture mapping for the monolithic run.

    Important:
    - ALL circuit qubits are mapped to target_module
    - other modules are still present in the architecture as unused modules
      via one placeholder qubit each

    This prevents the monolithic trace from being misinterpreted as
    cross-module by the architecture.
    """
    if target_module not in FIVE_MODULES:
        raise ValueError(
            f"target_module must be one of {FIVE_MODULES}, got {target_module}"
        )

    mapping = {}

    # Real circuit qubits: all on the monolithic target module
    for q in range(num_qubits):
        mapping[q] = target_module

    # Add one placeholder qubit index for each unused module so the
    # architecture validator still sees all five modules.
    next_q = num_qubits
    for mod in FIVE_MODULES:
        if mod == target_module:
            continue
        mapping[next_q] = mod
        next_q += 1

    return mapping


def print_monolithic_trace(trace, max_lines: int = 30):
    """
    Pretty-print the extracted monolithic trace.
    """
    print("=== Monolithic Local Execution Trace ===")
    for idx, item in enumerate(trace):
        if idx >= max_lines:
            print(f"... ({len(trace) - max_lines} more lines)")
            break

        print(
            f"step={item['step']:3d} | "
            f"op={item['op_name']:<12s} | "
            f"qubits={item['qubits']} | "
            f"clbits={item['clbits']} | "
            f"params={item['params']} | "
            f"module={item['module']}"
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
        "avg_waiting_time_ns": avg_waiting_time_ns,
        "avg_turnaround_time_ns": avg_turnaround_time_ns,
        "max_waiting_time_ns": max_waiting_time_ns,
        "hub_makespan_ns": getattr(arch.hub, "current_time_ns", 0),
        "hub_current_time_ns": getattr(arch.hub, "current_time_ns", 0),
        "num_waited_requests": num_waited_requests,
    }

    return stats


def save_stats_json(stats: dict, out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to: {out_file}")


def main():
    qasm_path = "square_root_n18.qasm"   # change if needed
    target_module = "module_0"

    # 1. Extract monolithic trace
    monolithic_trace, num_qubits = extract_monolithic_trace_from_qasm(
        qasm_file=qasm_path,
        target_module=target_module,
    )

    # 2. Build corrected monolithic mapping:
    #    all real circuit qubits belong to module_0
    qubit_to_module = build_monolithic_arch_mapping(
        num_qubits=num_qubits,
        target_module=target_module,
    )

    # 3. Build a fresh architecture instance
    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=qubit_to_module,
        link_latency_ns=10,
        hub_max_concurrent_transfers=2,
        hub_setup_latency_ns=20,
        hub_transfer_latency_ns=80,
        event_tick_ns=5,
    )

    # 4. Optional prints
    arch.describe()
    arch.print_qubit_mapping()
    arch.print_stick_diagram()
    print()
    print_monolithic_trace(monolithic_trace, max_lines=25)

    print("\n=== Quick Trace Stats ===")
    print("Total ops:", len(monolithic_trace))
    print("Unique ops:", sorted(set(t["op_name"] for t in monolithic_trace)))
    print("Unique qubit tuples:", sorted(set(tuple(t["qubits"]) for t in monolithic_trace)))

    # 5. Run the trace through the fixed architecture
    #    With the corrected mapping, hub usage should be zero.
    print("\n================ MONOLITHIC ARCHITECTURE RUN ================")
    run_monolithic_trace(arch, monolithic_trace)

    # 6. Save stats
    stats = collect_baseline_stats(
        arch=arch,
        trace=monolithic_trace,
        trace_type="monolithic",
    )

    save_stats_json(stats, "monolithic_stats.json")


if __name__ == "__main__":
    main()
