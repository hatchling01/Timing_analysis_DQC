#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_burst_sweep.py

Goal:
- keep the attacker sparse like A5
- add controlled burstiness
- sweep burst parameters to see which setting is most victim-sensitive

Fixed:
- five-node architecture
- static distributed victim
- P1 disjoint placement
- Tier-1 attacker

Outputs:
- one JSON per probe
- one request-level plot per probe
- one job-level makespan plot per probe
- filenames include victim circuit name
"""

import json
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)

VICTIM_MODULES = ["module_0", "module_1", "module_2"]
ATTACKER_MODULES = ["module_3", "module_4"]

ATTACKER_PROBES = [
    "probe_1_cx_chain",
    "probe_2_bursty_entangling",
    "probe_3_light_periodic",
]

# A5-like sparse burst settings:
# every `period` victim events, inject a burst of `burst_size` attacker events
BURST_SWEEP = [
    {"name": "B1_light", "period": 12, "burst_size": 1},
    {"name": "B2_mild", "period": 12, "burst_size": 2},
    {"name": "B3_medium", "period": 10, "burst_size": 2},
    {"name": "B4_strong", "period": 10, "burst_size": 3},
    {"name": "B5_sparse_heavy", "period": 16, "burst_size": 4},
]


def victim_tag_from_qasm(victim_qasm: str) -> str:
    """
    Convert something like 'qft_n18.qasm' -> 'qft_n18'
    and make it safe for filenames.
    """
    stem = Path(victim_qasm).stem
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)
    return safe


# ============================================================
# Victim trace extraction
# ============================================================

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

    return trace, q2m, qc.num_qubits


# ============================================================
# Attacker probes
# ============================================================

def attacker_probe_qubit_map():
    return {
        0: "module_3",
        1: "module_3",
        2: "module_4",
        3: "module_4",
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


def build_probe_1_cx_chain():
    qc = QuantumCircuit(4)
    for _ in range(20):
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.x(0)
        qc.x(2)
    return qc


def build_probe_2_bursty_entangling():
    qc = QuantumCircuit(4)
    for _ in range(8):
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(0, 3)
        qc.cx(1, 2)
        qc.h(0)
        qc.h(1)
        qc.x(2)
        qc.x(3)
    return qc


def build_probe_3_light_periodic():
    qc = QuantumCircuit(4)
    for _ in range(25):
        qc.h(0)
        qc.h(1)
        qc.x(2)
        qc.x(3)
        qc.cx(0, 2)
        qc.z(0)
        qc.z(2)
    return qc


def generate_attacker_probe_trace(probe_name: str):
    q2m = attacker_probe_qubit_map()

    if probe_name == "probe_1_cx_chain":
        qc = build_probe_1_cx_chain()
    elif probe_name == "probe_2_bursty_entangling":
        qc = build_probe_2_bursty_entangling()
    elif probe_name == "probe_3_light_periodic":
        qc = build_probe_3_light_periodic()
    else:
        raise ValueError(f"Unknown attacker probe: {probe_name}")

    return trace_from_qiskit_circuit(qc, q2m), q2m, qc.num_qubits


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
    return combined


# ============================================================
# Sparse-bursty schedule
# ============================================================

def build_sparse_bursty_schedule(victim_trace, attacker_trace, period, burst_size):
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


# ============================================================
# Metrics
# ============================================================

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
        last_completion = max(
            r.end_time_ns for r in completed_requests if r.end_time_ns is not None
        )
        makespan = last_completion - first_release_ns
    elif first_release_ns is not None and last_release_ns is not None:
        makespan = last_release_ns - first_release_ns
    else:
        makespan = 0

    return {
        "attacker_job_makespan_ns": makespan,
        "attacker_total_events": total_events,
    }


# ============================================================
# Experiment
# ============================================================

def run_one_experiment(victim_qasm: str, probe_name: str, burst_cfg: dict):
    victim_trace, victim_q2m, victim_num_qubits = extract_static_trace(
        victim_qasm, VICTIM_MODULES
    )
    attacker_trace, attacker_q2m, _ = generate_attacker_probe_trace(probe_name)

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

    merged_stream = build_sparse_bursty_schedule(
        victim_trace=victim_trace,
        attacker_trace=attacker_trace,
        period=burst_cfg["period"],
        burst_size=burst_cfg["burst_size"],
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

    req_metrics = attacker_request_metrics(attacker_completed)
    job_metrics = attacker_job_metrics(
        attacker_completed,
        attacker_first_release_ns,
        attacker_last_release_ns,
        attacker_event_count,
    )

    return {
        "victim_qasm": victim_qasm,
        "probe": probe_name,
        "burst_name": burst_cfg["name"],
        "period": burst_cfg["period"],
        "burst_size": burst_cfg["burst_size"],
        "workload_type": "static_distributed",
        **req_metrics,
        **job_metrics,
        "hub_makespan_ns": arch.hub.current_time_ns,
    }


# ============================================================
# Plotting
# ============================================================

def plot_request_level(results, probe_name, victim_tag):
    labels = [r["burst_name"] for r in results]
    avg_wait = [r["attacker_avg_waiting_time_ns"] for r in results]
    avg_turn = [r["attacker_avg_turnaround_time_ns"] for r in results]
    max_wait = [r["attacker_max_waiting_time_ns"] for r in results]

    x = list(range(len(labels)))
    w = 0.25

    plt.figure(figsize=(11, 5))
    plt.bar([i - w for i in x], avg_wait, width=w, label="Avg wait")
    plt.bar(x, avg_turn, width=w, label="Avg turnaround")
    plt.bar([i + w for i in x], max_wait, width=w, label="Max wait")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Time (ns)")
    plt.title(f"Sparse Bursty Sweep: Request-Level ({probe_name}, {victim_tag})")
    plt.legend()
    plt.tight_layout()
    out_png = f"burst_sweep_{victim_tag}_{probe_name}_request_level.png"
    plt.savefig(out_png, dpi=300)
    plt.show()


def plot_job_level(results, probe_name, victim_tag):
    labels = [r["burst_name"] for r in results]
    makespan = [r["attacker_job_makespan_ns"] for r in results]

    x = list(range(len(labels)))

    plt.figure(figsize=(10, 5))
    plt.bar(x, makespan)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Time (ns)")
    plt.title(f"Sparse Bursty Sweep: Job Makespan ({probe_name}, {victim_tag})")
    plt.tight_layout()
    out_png = f"burst_sweep_{victim_tag}_{probe_name}_job_makespan.png"
    plt.savefig(out_png, dpi=300)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    victim_qasm = "square_root_n18.qasm"
    victim_tag = victim_tag_from_qasm(victim_qasm)

    for probe_name in ATTACKER_PROBES:
        print(f"\n===== Probe: {probe_name} / Victim: {victim_tag} =====")
        results = []

        for burst_cfg in BURST_SWEEP:
            print(f"Running {probe_name} / {burst_cfg['name']}")
            result = run_one_experiment(victim_qasm, probe_name, burst_cfg)
            print(result)
            results.append(result)

        out_json = f"burst_sweep_{victim_tag}_{probe_name}_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {out_json}")

        plot_request_level(results, probe_name, victim_tag)
        plot_job_level(results, probe_name, victim_tag)


if __name__ == "__main__":
    main()
