#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_probe3_duration_sweep_relative.py

Fixed:
- Probe 3
- R1_dense
- uniform spacing
- five-node Architecture M
- static distributed victim
- P1 disjoint placement
- Tier-1 attacker
- A5-like light schedule

Sweep:
- probe duration relative to victim cross-module op count:
  10%, 20%, 50%, 100%
"""

import json
import copy
import math
from pathlib import Path
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)

VICTIM_MODULES = ["module_0", "module_1", "module_2"]
ATTACKER_MODULES = ["module_3", "module_4"]

VICTIM_QASMS = [
    "square_root_n18.qasm",
    "qft_n18.qasm",
    "bv_n19.qasm",
    "dnn_n16.qasm",
    "sat_n11.qasm",
]

SCHEDULE_CFG = {
    "name": "A5_like_light",
    "period": 12,
    "burst_size": 1,
}

RELATIVE_DURATION_SWEEP = [
    {"name": "P10", "fraction": 0.10},
    {"name": "P20", "fraction": 0.20},
    {"name": "P50", "fraction": 0.50},
    {"name": "P100", "fraction": 1.00},
]


def safe_tag(path_str: str) -> str:
    stem = Path(path_str).stem
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)


def build_subset_qubit_map(num_qubits: int, module_subset):
    q2m = {}
    block_size = (num_qubits + len(module_subset) - 1) // len(module_subset)
    for q in range(num_qubits):
        module_idx = min(q // block_size, len(module_subset) - 1)
        q2m[q] = module_subset[module_idx]

    if set(q2m.values()) != set(module_subset) and num_qubits >= len(module_subset):
        q2m = {}
        for q in range(num_qubits):
            q2m[q] = module_subset[q % len(module_subset)]
    return q2m


def extract_static_trace(qasm_file: str, module_subset):
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    q2m = build_subset_qubit_map(qc.num_qubits, module_subset)

    trace = []
    cross_module_ops = 0
    for step_idx, instruction in enumerate(qc.data):
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        qubit_indices = [qc.find_bit(q).index for q in qargs]
        clbit_indices = [qc.find_bit(c).index for c in cargs]
        touched_modules = sorted({q2m[q] for q in qubit_indices})
        is_cross_module = len(touched_modules) > 1
        if is_cross_module:
            cross_module_ops += 1

        trace.append({
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
        })
    return trace, q2m, qc.num_qubits, cross_module_ops


def attacker_probe_qubit_map():
    return {0: "module_3", 1: "module_3", 2: "module_4", 3: "module_4"}


def trace_from_qiskit_circuit(qc: QuantumCircuit, q2m: dict):
    trace = []
    for step_idx, instruction in enumerate(qc.data):
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        qubit_indices = [qc.find_bit(q).index for q in qargs]
        clbit_indices = [qc.find_bit(c).index for c in cargs]
        touched_modules = sorted({q2m[q] for q in qubit_indices})
        is_cross_module = len(touched_modules) > 1

        trace.append({
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
        })
    return trace


def build_probe3_R1_uniform(total_rounds: int):
    qc = QuantumCircuit(4)
    for _ in range(total_rounds):
        qc.h(0)
        qc.h(1)
        qc.x(2)
        qc.x(3)
        qc.z(0)
        qc.z(2)
        qc.cx(0, 2)
    return qc


def generate_probe3_duration_trace(total_rounds: int):
    q2m = attacker_probe_qubit_map()
    qc = build_probe3_R1_uniform(total_rounds)
    return trace_from_qiskit_circuit(qc, q2m), q2m, qc.num_qubits


def offset_trace_qubits(trace, offset):
    out = []
    for entry in trace:
        e = copy.deepcopy(entry)
        e["qubits"] = [q + offset for q in entry["qubits"]]
        out.append(e)
    return out


def combine_victim_attacker_mappings(v_q2m, v_num_qubits, a_q2m):
    combined = {}
    for q, mod in v_q2m.items():
        combined[q] = mod
    for q, mod in a_q2m.items():
        combined[q + v_num_qubits] = mod
    return combined


def build_sparse_schedule(victim_trace, attacker_trace, period, burst_size):
    merged = []
    a_idx = 0
    for i, v in enumerate(victim_trace):
        merged.append(("victim", v))
        if (i + 1) % period == 0:
            for _ in range(burst_size):
                if a_idx < len(attacker_trace):
                    merged.append(("attacker", attacker_trace[a_idx]))
                    a_idx += 1
    while a_idx < len(attacker_trace):
        merged.append(("attacker", attacker_trace[a_idx]))
        a_idx += 1
    return merged


def attacker_request_metrics(completed_requests):
    waits = [r.waiting_time_ns for r in completed_requests if r.waiting_time_ns is not None]
    turns = [r.turnaround_time_ns for r in completed_requests if r.turnaround_time_ns is not None]
    return {
        "attacker_completed_requests": len(completed_requests),
        "attacker_avg_waiting_time_ns": (sum(waits) / len(waits)) if waits else 0,
        "attacker_avg_turnaround_time_ns": (sum(turns) / len(turns)) if turns else 0,
        "attacker_max_waiting_time_ns": max(waits) if waits else 0,
        "attacker_num_waited_requests": sum(1 for w in waits if w > 0),
    }


def attacker_job_metrics(completed_requests, first_release_ns, last_release_ns, total_events):
    if completed_requests and first_release_ns is not None:
        last_completion = max(r.end_time_ns for r in completed_requests if r.end_time_ns is not None)
        makespan = last_completion - first_release_ns
    elif first_release_ns is not None and last_release_ns is not None:
        makespan = last_release_ns - first_release_ns
    else:
        makespan = 0
    return {
        "attacker_job_makespan_ns": makespan,
        "attacker_total_events": total_events,
    }


def run_one_experiment(victim_qasm: str, duration_cfg: dict):
    victim_trace, victim_q2m, victim_num_qubits, victim_cross_ops = extract_static_trace(
        victim_qasm, VICTIM_MODULES
    )

    total_rounds = max(1, math.ceil(victim_cross_ops * duration_cfg["fraction"]))

    attacker_trace, attacker_q2m, _ = generate_probe3_duration_trace(total_rounds)
    attacker_trace = offset_trace_qubits(attacker_trace, victim_num_qubits)

    combined_q2m = combine_victim_attacker_mappings(victim_q2m, victim_num_qubits, attacker_q2m)

    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=combined_q2m,
        link_latency_ns=10,
        hub_max_concurrent_transfers=2,
        hub_setup_latency_ns=20,
        hub_transfer_latency_ns=80,
        event_tick_ns=5,
    )

    merged_stream = build_sparse_schedule(
        victim_trace=victim_trace,
        attacker_trace=attacker_trace,
        period=SCHEDULE_CFG["period"],
        burst_size=SCHEDULE_CFG["burst_size"],
    )

    attacker_first_release_ns = None
    attacker_last_release_ns = None
    attacker_event_count = 0

    for tenant, entry in merged_stream:
        if tenant == "attacker":
            if attacker_first_release_ns is None:
                attacker_first_release_ns = arch.hub.current_time_ns
            attacker_last_release_ns = arch.hub.current_time_ns
            attacker_event_count += 1

        event = normalize_trace_entry(entry, "static_distributed")
        arch.route_trace_event(event)
        arch.advance_architecture_time()

    arch.drain_hub()

    attacker_completed = [r for r in arch.hub.completed_requests if r.source_module in ATTACKER_MODULES]

    return {
        "victim_qasm": victim_qasm,
        "schedule_name": SCHEDULE_CFG["name"],
        "probe": "probe3_R1_uniform_duration_relative",
        "duration_name": duration_cfg["name"],
        "fraction_of_victim_cross_ops": duration_cfg["fraction"],
        "total_rounds": total_rounds,
        "victim_cross_module_ops": victim_cross_ops,
        "workload_type": "static_distributed",
        **attacker_request_metrics(attacker_completed),
        **attacker_job_metrics(attacker_completed, attacker_first_release_ns, attacker_last_release_ns, attacker_event_count),
        "hub_makespan_ns": arch.hub.current_time_ns,
    }


def plot_request_level(results, victim_tag):
    labels = [r["duration_name"] for r in results]
    avg_wait = [r["attacker_avg_waiting_time_ns"] for r in results]
    avg_turn = [r["attacker_avg_turnaround_time_ns"] for r in results]
    max_wait = [r["attacker_max_waiting_time_ns"] for r in results]

    x = list(range(len(labels)))
    w = 0.25

    plt.figure(figsize=(11, 5))
    plt.bar([i - w for i in x], avg_wait, width=w, label="Avg wait")
    plt.bar(x, avg_turn, width=w, label="Avg turnaround")
    plt.bar([i + w for i in x], max_wait, width=w, label="Max wait")
    plt.xticks(x, labels)
    plt.ylabel("Time (ns)")
    plt.title(f"Probe 3 R1 Uniform: Relative Duration Sweep ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"probe3_R1_uniform_reldur_{victim_tag}_request_level.png", dpi=300)
    plt.show()


def plot_job_makespan(results, victim_tag):
    labels = [r["duration_name"] for r in results]
    makespan = [r["attacker_job_makespan_ns"] for r in results]

    x = list(range(len(labels)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, makespan)
    plt.xticks(x, labels)
    plt.ylabel("Time (ns)")
    plt.title(f"Probe 3 R1 Uniform: Relative Duration Sweep Makespan ({victim_tag})")
    plt.tight_layout()
    plt.savefig(f"probe3_R1_uniform_reldur_{victim_tag}_job_makespan.png", dpi=300)
    plt.show()


def plot_job_counts(results, victim_tag):
    labels = [r["duration_name"] for r in results]
    completed = [r["attacker_completed_requests"] for r in results]
    waited = [r["attacker_num_waited_requests"] for r in results]

    x = list(range(len(labels)))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - w/2 for i in x], completed, width=w, label="Completed reqs")
    plt.bar([i + w/2 for i in x], waited, width=w, label="Waited reqs")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title(f"Probe 3 R1 Uniform: Relative Duration Sweep Counts ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"probe3_R1_uniform_reldur_{victim_tag}_job_counts.png", dpi=300)
    plt.show()


def main():
    for victim_qasm in VICTIM_QASMS:
        victim_tag = safe_tag(victim_qasm)
        results = []
        for duration_cfg in RELATIVE_DURATION_SWEEP:
            print(f"Running relative duration sweep on {victim_tag}: {duration_cfg['name']}")
            results.append(run_one_experiment(victim_qasm, duration_cfg))

        out_json = f"probe3_R1_uniform_reldur_{victim_tag}_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        plot_request_level(results, victim_tag)
        plot_job_makespan(results, victim_tag)
        plot_job_counts(results, victim_tag)


if __name__ == "__main__":
    main()
