#!/usr/bin/env python3
"""
run_attack_tier1_p2_static_r1_timescale_sweep.py

Static deployment + Tier-1 attacker + overlap placement (P2 first overlap case)
with the fixed best overlap rate:
    - Probe 3
    - R1_dense
    - uniform spacing

This script studies duration / timescale under overlap.

Placement:
- victim   on module_0, module_1, module_2
- attacker on module_2, module_3
- dummy anchor qubit on module_4 for architecture validation

Timescale groups:
- short  : P20
- medium : P50
- long   : P100

Victims:
- square_root_n18.qasm
- qft_n18.qasm
- bv_n19.qasm
- dnn_n16.qasm
- sat_n11.qasm

Outputs:
- one JSON per victim
- one request-level plot per victim
- one makespan plot per victim
- one counts plot per victim
"""

import json
import math
import copy
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)

# ============================================================
# Fixed config
# ============================================================

VICTIM_QASMS = [
    "square_root_n18.qasm",
    "qft_n18.qasm",
    "bv_n19.qasm",
    "dnn_n16.qasm",
    "sat_n11.qasm",
]

VICTIM_MODULES = ["module_0", "module_1", "module_2"]
ATTACKER_MODULES = ["module_2", "module_3"]

SCHEDULE_CFG = {
    "name": "A5_like_light",
    "period": 12,
    "burst_size": 1,
}

# Fixed best overlap rate
FIXED_RATE = {
    "name": "R1_dense",
    "density": 1.0,
}

TIMESCALE_SWEEP = [
    {"timescale": "short", "duration_name": "P20", "fraction": 0.20},
    {"timescale": "medium", "duration_name": "P50", "fraction": 0.50},
    {"timescale": "long", "duration_name": "P100", "fraction": 1.00},
]


# ============================================================
# Helpers
# ============================================================

def safe_tag(path_str: str) -> str:
    stem = Path(path_str).stem
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)


def build_subset_qubit_map(num_qubits: int, module_subset):
    if not module_subset:
        raise ValueError("module_subset cannot be empty")

    q2m = {}
    block_size = (num_qubits + len(module_subset) - 1) // len(module_subset)

    for q in range(num_qubits):
        module_idx = min(q // block_size, len(module_subset) - 1)
        q2m[q] = module_subset[module_idx]

    present = set(q2m.values())
    expected = set(module_subset)
    if present != expected and num_qubits >= len(module_subset):
        q2m = {}
        for q in range(num_qubits):
            q2m[q] = module_subset[q % len(module_subset)]

    return q2m


def extract_static_trace(qasm_file: str, module_subset):
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    q2m = build_subset_qubit_map(qc.num_qubits, module_subset)

    trace = []
    cross_module_ops = 0
    total_ops = 0

    for step_idx, instruction in enumerate(qc.data):
        op = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        qubit_indices = [qc.find_bit(q).index for q in qargs]
        clbit_indices = [qc.find_bit(c).index for c in cargs]

        if len(qubit_indices) == 0:
            continue

        total_ops += 1
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

    return trace, q2m, qc.num_qubits, cross_module_ops, total_ops


# ============================================================
# Attacker probe mapping (overlap case)
# ============================================================

def attacker_probe_qubit_map():
    return {
        0: "module_2",
        1: "module_2",
        2: "module_3",
        3: "module_3",
    }


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


def build_probe3_r1_dense(total_rounds: int):
    """
    Probe 3 + R1_dense + uniform spacing:
    cross-module CX in every round.
    """
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


def generate_probe3_r1_dense_trace(total_rounds: int):
    q2m = attacker_probe_qubit_map()
    qc = build_probe3_r1_dense(total_rounds)
    trace = trace_from_qiskit_circuit(qc, q2m)
    return trace, q2m, qc.num_qubits


# ============================================================
# Utilities
# ============================================================

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

    # dummy anchor so all five modules appear
    anchor_idx = v_num_qubits + len(a_q2m)
    combined[anchor_idx] = "module_4"

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
        "attacker_avg_waiting_time_ns": (sum(waits) / len(waits)) if waits else 0.0,
        "attacker_avg_turnaround_time_ns": (sum(turns) / len(turns)) if turns else 0.0,
        "attacker_max_waiting_time_ns": max(waits) if waits else 0.0,
        "attacker_num_waited_requests": sum(1 for w in waits if w > 0),
    }


def attacker_job_metrics(completed_requests, first_release_ns, last_release_ns, total_events):
    if completed_requests and first_release_ns is not None:
        last_completion = max(
            r.end_time_ns for r in completed_requests if r.end_time_ns is not None
        )
        makespan = last_completion - first_release_ns
    elif first_release_ns is not None and last_release_ns is not None:
        makespan = last_release_ns - first_release_ns
    else:
        makespan = 0.0

    return {
        "attacker_job_makespan_ns": makespan,
        "attacker_total_events": total_events,
    }


# ============================================================
# Core experiment
# ============================================================

def run_one_experiment(victim_qasm: str, timescale_cfg: dict):
    victim_trace, victim_q2m, victim_num_qubits, victim_cross_ops, victim_total_ops = extract_static_trace(
        victim_qasm, VICTIM_MODULES
    )

    total_rounds = max(1, math.ceil(victim_cross_ops * timescale_cfg["fraction"]))

    attacker_trace, attacker_q2m, _ = generate_probe3_r1_dense_trace(total_rounds)
    attacker_trace = offset_trace_qubits(attacker_trace, victim_num_qubits)

    combined_q2m = combine_victim_attacker_mappings(
        victim_q2m,
        victim_num_qubits,
        attacker_q2m,
    )

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

    attacker_completed = [
        r for r in arch.hub.completed_requests
        if r.source_module in ATTACKER_MODULES
    ]

    req = attacker_request_metrics(attacker_completed)
    job = attacker_job_metrics(
        attacker_completed,
        attacker_first_release_ns,
        attacker_last_release_ns,
        attacker_event_count,
    )

    waited_fraction = (
        req["attacker_num_waited_requests"] / req["attacker_completed_requests"]
        if req["attacker_completed_requests"] > 0 else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "schedule_name": SCHEDULE_CFG["name"],
        "probe": "probe3_static_overlap_p2_r1_timescale_sweep",
        "fixed_rate_name": FIXED_RATE["name"],
        "fixed_rate_density": FIXED_RATE["density"],
        "timescale": timescale_cfg["timescale"],
        "duration_name": timescale_cfg["duration_name"],
        "fraction_of_victim_cross_ops": timescale_cfg["fraction"],
        "total_rounds": total_rounds,
        "num_probe_rounds": total_rounds,
        "victim_total_ops": victim_total_ops,
        "victim_cross_module_ops": victim_cross_ops,
        "victim_cross_fraction": (victim_cross_ops / victim_total_ops) if victim_total_ops else 0.0,
        "workload_type": "static_overlap_p2",
        "placement_case": "P2_one_module_overlap",
        **req,
        **job,
        "attacker_waited_fraction": waited_fraction,
        "hub_makespan_ns": arch.hub.current_time_ns,
    }


# ============================================================
# Plotting
# ============================================================

def plot_request_level(results, victim_tag):
    labels = [r["timescale"] for r in results]
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
    plt.title(f"Overlap P2 R1 Timescale Sweep: Request-Level ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"overlap_p2_r1_timescale_{victim_tag}_request_level.png", dpi=300)
    plt.show()


def plot_job_makespan(results, victim_tag):
    labels = [r["timescale"] for r in results]
    makespan = [r["attacker_job_makespan_ns"] for r in results]

    x = list(range(len(labels)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, makespan)
    plt.xticks(x, labels)
    plt.ylabel("Time (ns)")
    plt.title(f"Overlap P2 R1 Timescale Sweep: Job Makespan ({victim_tag})")
    plt.tight_layout()
    plt.savefig(f"overlap_p2_r1_timescale_{victim_tag}_job_makespan.png", dpi=300)
    plt.show()


def plot_job_counts(results, victim_tag):
    labels = [r["timescale"] for r in results]
    completed = [r["attacker_completed_requests"] for r in results]
    waited = [r["attacker_num_waited_requests"] for r in results]

    x = list(range(len(labels)))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - w/2 for i in x], completed, width=w, label="Completed reqs")
    plt.bar([i + w/2 for i in x], waited, width=w, label="Waited reqs")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title(f"Overlap P2 R1 Timescale Sweep: Job Counts ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"overlap_p2_r1_timescale_{victim_tag}_job_counts.png", dpi=300)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    for victim_qasm in VICTIM_QASMS:
        victim_tag = safe_tag(victim_qasm)
        results = []

        for timescale_cfg in TIMESCALE_SWEEP:
            print(
                f"Running overlap P2 R1 timescale sweep on {victim_tag}: "
                f"{timescale_cfg['timescale']} ({timescale_cfg['duration_name']})"
            )
            result = run_one_experiment(victim_qasm, timescale_cfg)
            print(result)
            results.append(result)

        out_json = f"overlap_p2_r1_timescale_{victim_tag}_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        plot_request_level(results, victim_tag)
        plot_job_makespan(results, victim_tag)
        plot_job_counts(results, victim_tag)


if __name__ == "__main__":
    main()
