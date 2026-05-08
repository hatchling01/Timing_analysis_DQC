#!/usr/bin/env python3
"""
qaoa_family_best_attack_overlap_p2.py

Best-attack family test for intra-family victim distinguishability
under static OVERLAP placement P2.

Placement:
- victim   on module_0, module_1, module_2
- attacker on module_2, module_3
- dummy anchor qubit on module_4 for architecture validation

Fixed best overlap attack:
- Probe 3
- R1_dense
- short timescale = P20

Victim family:
    qaoa_nativegates_ibm_qiskit_opt3_*.qasm

Outputs:
    - qaoa_family_best_attack_overlap_p2_results.json
    - qaoa_family_best_attack_overlap_p2_summary.csv
    - qaoa_family_best_attack_overlap_p2_fingerprints.csv
    - qaoa_family_best_attack_overlap_p2_pairwise_distance.csv
    - qaoa_family_best_attack_overlap_p2_request_metrics.png
    - qaoa_family_best_attack_overlap_p2_pairwise_distance.png
"""

import json
import math
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)

# ============================================================
# Fixed architecture / best overlap attack config
# ============================================================

QAOA_QASMS = [
    "qaoa_nativegates_ibm_qiskit_opt3_5.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_6.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_7.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_8.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_9.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_10.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_11.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_12.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_13.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_14.qasm",
    "qaoa_nativegates_ibm_qiskit_opt3_15.qasm",
]

VICTIM_MODULES = ["module_0", "module_1", "module_2"]
ATTACKER_MODULES = ["module_2", "module_3"]

SCHEDULE_CFG = {
    "name": "A5_like_light",
    "period": 12,
    "burst_size": 1,
}

FIXED_RATE = {
    "name": "R1_dense",
    "density": 1.0,
}

FIXED_TIMESCALE = {
    "timescale": "short",
    "duration_name": "P20",
    "fraction": 0.20,
}


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
# Attacker probe mapping and Probe 3
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
        "attacker_avg_waiting_time_ns": float(sum(waits) / len(waits)) if waits else 0.0,
        "attacker_avg_turnaround_time_ns": float(sum(turns) / len(turns)) if turns else 0.0,
        "attacker_max_waiting_time_ns": float(max(waits)) if waits else 0.0,
        "attacker_num_waited_requests": int(sum(1 for w in waits if w > 0)),
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
        "attacker_job_makespan_ns": float(makespan),
        "attacker_total_events": int(total_events),
    }


# ============================================================
# Core experiment
# ============================================================

def run_one_experiment(victim_qasm: str):
    victim_trace, victim_q2m, victim_num_qubits, victim_cross_ops, victim_total_ops = extract_static_trace(
        victim_qasm, VICTIM_MODULES
    )

    total_rounds = max(1, math.ceil(victim_cross_ops * FIXED_TIMESCALE["fraction"]))

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

    waited_frac = (
        req["attacker_num_waited_requests"] / req["attacker_completed_requests"]
        if req["attacker_completed_requests"] > 0 else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "schedule_name": SCHEDULE_CFG["name"],
        "probe": "Probe3",
        "fixed_rate_name": FIXED_RATE["name"],
        "fixed_rate_density": FIXED_RATE["density"],
        "timescale": FIXED_TIMESCALE["timescale"],
        "duration_name": FIXED_TIMESCALE["duration_name"],
        "fraction_of_victim_cross_ops": FIXED_TIMESCALE["fraction"],
        "total_rounds": total_rounds,
        "victim_total_ops": victim_total_ops,
        "victim_cross_module_ops": victim_cross_ops,
        "victim_cross_fraction": (victim_cross_ops / victim_total_ops) if victim_total_ops else 0.0,
        "workload_type": "static_overlap_p2",
        "placement_case": "P2_one_module_overlap",
        **req,
        **job,
        "attacker_waited_fraction": waited_frac,
        "hub_makespan_ns": arch.hub.current_time_ns,
    }


# ============================================================
# Fingerprint analysis
# ============================================================

def build_fingerprint_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in results_df.iterrows():
        rows.append({
            "victim_qasm": r["victim_qasm"],
            "victim_total_ops": r["victim_total_ops"],
            "victim_cross_module_ops": r["victim_cross_module_ops"],
            "victim_cross_fraction": r["victim_cross_fraction"],
            "avg_wait": r["attacker_avg_waiting_time_ns"],
            "avg_turn": r["attacker_avg_turnaround_time_ns"],
            "max_wait": r["attacker_max_waiting_time_ns"],
            "num_waited": r["attacker_num_waited_requests"],
            "waited_frac": r["attacker_waited_fraction"],
            "completed": r["attacker_completed_requests"],
            "makespan": r["attacker_job_makespan_ns"],
            "hub_makespan": r["hub_makespan_ns"],
        })
    return pd.DataFrame(rows).sort_values("victim_qasm").reset_index(drop=True)


def zscore_columns(df: pd.DataFrame, feature_cols):
    out = df.copy()
    for c in feature_cols:
        mu = out[c].mean()
        sigma = out[c].std(ddof=0)
        if sigma == 0:
            out[c] = 0.0
        else:
            out[c] = (out[c] - mu) / sigma
    return out


def pairwise_distance_matrix(df: pd.DataFrame, feature_cols):
    zdf = zscore_columns(df, feature_cols)
    X = zdf[feature_cols].to_numpy(dtype=float)
    names = zdf["victim_qasm"].tolist()

    n = len(names)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(X[i] - X[j])

    return pd.DataFrame(dist, index=names, columns=names)


# ============================================================
# Plotting
# ============================================================

def plot_request_metrics(results_df: pd.DataFrame):
    victims = results_df["victim_qasm"].tolist()
    vals = results_df["attacker_avg_waiting_time_ns"].tolist()

    x = np.arange(len(victims))
    plt.figure(figsize=(14, 6))
    plt.bar(x, vals)
    plt.xticks(x, [safe_tag(v) for v in victims], rotation=30, ha="right")
    plt.ylabel("Avg waiting time (ns)")
    plt.title("QAOA family fingerprints under best attack (overlap P2)")
    plt.tight_layout()
    plt.savefig("qaoa_family_best_attack_overlap_p2_request_metrics.png", dpi=300)
    plt.show()


def plot_pairwise_distance(dist_df: pd.DataFrame):
    plt.figure(figsize=(8, 7))
    mat = dist_df.to_numpy(dtype=float)
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Fingerprint distance")

    labels = [safe_tag(x) for x in dist_df.index.tolist()]
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("QAOA family pairwise fingerprint distance (overlap P2)")
    plt.tight_layout()
    plt.savefig("qaoa_family_best_attack_overlap_p2_pairwise_distance.png", dpi=300)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    results = []

    for victim_qasm in QAOA_QASMS:
        print(f"Running overlap P2 best attack on {victim_qasm}")
        result = run_one_experiment(victim_qasm)
        print(result)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(["victim_qasm"]).reset_index(drop=True)
    fingerprint_df = build_fingerprint_table(results_df)

    feature_cols = [c for c in fingerprint_df.columns if c not in {"victim_qasm"}]
    dist_df = pairwise_distance_matrix(fingerprint_df, feature_cols)

    with open("qaoa_family_best_attack_overlap_p2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    results_df.to_csv("qaoa_family_best_attack_overlap_p2_summary.csv", index=False)
    fingerprint_df.to_csv("qaoa_family_best_attack_overlap_p2_fingerprints.csv", index=False)
    dist_df.to_csv("qaoa_family_best_attack_overlap_p2_pairwise_distance.csv")

    print("\nSaved:")
    print("  qaoa_family_best_attack_overlap_p2_results.json")
    print("  qaoa_family_best_attack_overlap_p2_summary.csv")
    print("  qaoa_family_best_attack_overlap_p2_fingerprints.csv")
    print("  qaoa_family_best_attack_overlap_p2_pairwise_distance.csv")

    print("\n=== Fingerprint summary ===")
    print(fingerprint_df.to_string(index=False))

    print("\n=== Pairwise fingerprint distance ===")
    print(dist_df.round(3).to_string())

    plot_request_metrics(results_df)
    plot_pairwise_distance(dist_df)


if __name__ == "__main__":
    main()
