#!/usr/bin/env python3
"""
qaoa_family_best_attack_sequential_v2.py

Best-attack family test for intra-family victim distinguishability,
but now using TRUE sequential-v2 victim execution.

Victim family:
    qaoa_nativegates_ibm_qiskit_opt3_*.qasm

Attack configuration:
    - Probe 3
    - R1_dense
    - uniform spacing
    - relative-duration windows
    - sequential-v2 stage-aware victim execution
    - inter-stage gap

Outputs:
    - qaoa_family_best_attack_sequential_v2_results.json
    - qaoa_family_best_attack_sequential_v2_summary.csv
    - qaoa_family_best_attack_sequential_v2_fingerprints.csv
    - qaoa_family_best_attack_sequential_v2_pairwise_distance.csv
    - qaoa_family_best_attack_sequential_v2_request_metrics.png
    - qaoa_family_best_attack_sequential_v2_pairwise_distance.png
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
# Fixed architecture / best attack / sequential-v2 config
# ============================================================

FIVE_MODULES = [f"module_{i}" for i in range(5)]
ATTACKER_MODULES = ["module_3", "module_4"]

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

SCHEDULE_CFG = {
    "name": "A5_like_light",
    "period": 12,
    "burst_size": 1,
}

ATTACK_WINDOWS = [
    {"name": "P20", "fraction": 0.20},
    {"name": "P50", "fraction": 0.50},
    {"name": "P100", "fraction": 1.00},
]

INTER_STAGE_GAP_NS = 20


# ============================================================
# Helpers
# ============================================================

def safe_tag(path_str: str) -> str:
    stem = Path(path_str).stem
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)


def build_static_qubit_to_module_map_five_modules(num_qubits: int):
    """
    Match sequential-v2 baseline mapping.
    """
    if num_qubits < 5:
        raise ValueError("Need at least 5 qubits to populate all five modules.")

    qubit_to_module = {}
    block_size = (num_qubits + 5 - 1) // 5

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


# ============================================================
# Sequential-v2 victim trace extraction
# ============================================================

def extract_sequential_modular_trace_from_qasm_five_modules(qasm_file: str):
    """
    Match baseline sequential-v2 trace extraction.
    """
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    num_qubits = qc.num_qubits
    qubit_to_module = build_static_qubit_to_module_map_five_modules(num_qubits)

    trace = []
    current_active_module = None
    stage_id = -1
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
        touched_modules = sorted({qubit_to_module[q] for q in qubit_indices})
        active_module = touched_modules[0] if touched_modules else None

        if active_module != current_active_module:
            stage_id += 1
            current_active_module = active_module

        is_cross_module = len(touched_modules) > 1
        transfer_event = is_cross_module
        if is_cross_module:
            cross_module_ops += 1

        trace.append({
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
            "communication_event": is_cross_module,
        })

    return trace, qubit_to_module, qc.num_qubits, cross_module_ops, total_ops


# ============================================================
# Attacker probe: Probe 3 + R1 dense + uniform spacing
# ============================================================

def attacker_probe_qubit_map(victim_num_qubits: int):
    return {
        victim_num_qubits + 0: "module_3",
        victim_num_qubits + 1: "module_3",
        victim_num_qubits + 2: "module_4",
        victim_num_qubits + 3: "module_4",
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


def generate_probe3_trace(total_rounds: int, victim_num_qubits: int):
    qc = build_probe3_R1_uniform(total_rounds)
    local_q2m = {
        0: "module_3",
        1: "module_3",
        2: "module_4",
        3: "module_4",
    }
    trace = trace_from_qiskit_circuit(qc, local_q2m)

    out = []
    for entry in trace:
        e = copy.deepcopy(entry)
        e["qubits"] = [q + victim_num_qubits for q in entry["qubits"]]
        out.append(e)

    return out


# ============================================================
# Stage-aware sequential-v2 processing with attack injection
# ============================================================

def group_trace_by_stage(trace):
    stage_to_entries = {}
    for entry in trace:
        stage = entry.get("stage", None)
        if stage is None:
            raise ValueError("Sequential-v2 attack requires victim entries to have 'stage'.")
        stage_to_entries.setdefault(stage, []).append(entry)
    return stage_to_entries


def process_trace_stagewise_with_attack(
    arch,
    victim_trace,
    attacker_trace,
    inter_stage_gap_ns=0,
    period=12,
    burst_size=1,
):
    """
    True stage-aware sequential-v2 processing:
    - one victim stage at a time
    - drain hub after each stage
    - attacker injected lightly during victim stage execution
    """
    normalized_events = []
    completed_stage_order = []

    attacker_first_release_ns = None
    attacker_last_release_ns = None
    attacker_event_count = 0
    attacker_idx = 0
    victim_event_counter = 0

    stage_to_entries = group_trace_by_stage(victim_trace)
    ordered_stages = sorted(stage_to_entries.keys())

    for stage in ordered_stages:
        stage_entries = stage_to_entries[stage]

        for entry in stage_entries:
            victim_event_counter += 1

            event = normalize_trace_entry(entry, "sequential_modular")
            normalized_events.append(event)
            arch.route_trace_event(event)
            arch.advance_architecture_time()

            if victim_event_counter % period == 0:
                for _ in range(burst_size):
                    if attacker_idx < len(attacker_trace):
                        a_entry = attacker_trace[attacker_idx]
                        a_event = normalize_trace_entry(a_entry, "static_distributed")
                        normalized_events.append(a_event)

                        if attacker_first_release_ns is None:
                            attacker_first_release_ns = arch.hub.current_time_ns
                        attacker_last_release_ns = arch.hub.current_time_ns
                        attacker_event_count += 1

                        arch.route_trace_event(a_event)
                        arch.advance_architecture_time()
                        attacker_idx += 1

        arch.drain_hub()
        completed_stage_order.append(stage)

        if inter_stage_gap_ns > 0:
            arch.advance_architecture_time(delta_ns=inter_stage_gap_ns)

    while attacker_idx < len(attacker_trace):
        a_entry = attacker_trace[attacker_idx]
        a_event = normalize_trace_entry(a_entry, "static_distributed")
        normalized_events.append(a_event)

        if attacker_first_release_ns is None:
            attacker_first_release_ns = arch.hub.current_time_ns
        attacker_last_release_ns = arch.hub.current_time_ns
        attacker_event_count += 1

        arch.route_trace_event(a_event)
        arch.advance_architecture_time()
        attacker_idx += 1

    arch.drain_hub()

    return {
        "normalized_events": normalized_events,
        "completed_stage_order": completed_stage_order,
        "attacker_first_release_ns": attacker_first_release_ns,
        "attacker_last_release_ns": attacker_last_release_ns,
        "attacker_event_count": attacker_event_count,
    }


# ============================================================
# Metrics
# ============================================================

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
        last_completion = max(r.end_time_ns for r in completed_requests if r.end_time_ns is not None)
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

def run_one_experiment(victim_qasm: str, attack_window: dict):
    victim_trace, victim_q2m, victim_num_qubits, victim_cross_ops, victim_total_ops = (
        extract_sequential_modular_trace_from_qasm_five_modules(victim_qasm)
    )

    total_rounds = max(1, math.ceil(victim_cross_ops * attack_window["fraction"]))
    attacker_trace = generate_probe3_trace(total_rounds, victim_num_qubits)

    combined_q2m = {}
    combined_q2m.update(victim_q2m)
    combined_q2m.update(attacker_probe_qubit_map(victim_num_qubits))

    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=combined_q2m,
        link_latency_ns=10,
        hub_max_concurrent_transfers=2,
        hub_setup_latency_ns=20,
        hub_transfer_latency_ns=80,
        event_tick_ns=5,
    )

    run_info = process_trace_stagewise_with_attack(
        arch=arch,
        victim_trace=victim_trace,
        attacker_trace=attacker_trace,
        inter_stage_gap_ns=INTER_STAGE_GAP_NS,
        period=SCHEDULE_CFG["period"],
        burst_size=SCHEDULE_CFG["burst_size"],
    )

    attacker_completed = [
        r for r in arch.hub.completed_requests
        if r.source_module in ATTACKER_MODULES
    ]

    req = attacker_request_metrics(attacker_completed)
    job = attacker_job_metrics(
        attacker_completed,
        run_info["attacker_first_release_ns"],
        run_info["attacker_last_release_ns"],
        run_info["attacker_event_count"],
    )

    waited_frac = (
        req["attacker_num_waited_requests"] / req["attacker_completed_requests"]
        if req["attacker_completed_requests"] > 0 else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "schedule_name": SCHEDULE_CFG["name"],
        "probe": "probe3_R1_uniform_qaoa_family_sequential_v2",
        "attack_window": attack_window["name"],
        "fraction_of_victim_cross_ops": attack_window["fraction"],
        "total_rounds": total_rounds,
        "victim_total_ops": victim_total_ops,
        "victim_cross_module_ops": victim_cross_ops,
        "victim_cross_fraction": (victim_cross_ops / victim_total_ops) if victim_total_ops else 0.0,
        "workload_type": "sequential_modular_v2_attack",
        "inter_stage_gap_ns": INTER_STAGE_GAP_NS,
        "completed_stage_count": len(run_info["completed_stage_order"]),
        **req,
        **job,
        "attacker_waited_fraction": waited_frac,
        "hub_makespan_ns": arch.hub.current_time_ns,
    }


# ============================================================
# Fingerprint / distinguishability analysis
# ============================================================

def build_fingerprint_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for victim_qasm, g in results_df.groupby("victim_qasm"):
        row = {"victim_qasm": victim_qasm}
        g = g.sort_values("attack_window")

        static_fields = [
            "victim_total_ops",
            "victim_cross_module_ops",
            "victim_cross_fraction",
            "completed_stage_count",
        ]
        for f in static_fields:
            row[f] = g.iloc[0][f]

        for _, r in g.iterrows():
            w = r["attack_window"]
            row[f"{w}_avg_wait"] = r["attacker_avg_waiting_time_ns"]
            row[f"{w}_max_wait"] = r["attacker_max_waiting_time_ns"]
            row[f"{w}_waited_frac"] = r["attacker_waited_fraction"]
            row[f"{w}_makespan"] = r["attacker_job_makespan_ns"]
            row[f"{w}_completed"] = r["attacker_completed_requests"]

        rows.append(row)

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
    victims = sorted(results_df["victim_qasm"].unique())
    windows = [w["name"] for w in ATTACK_WINDOWS]

    plt.figure(figsize=(14, 6))
    width = 0.8 / len(windows)
    x = np.arange(len(victims))

    for k, w in enumerate(windows):
        vals = []
        for v in victims:
            sub = results_df[(results_df["victim_qasm"] == v) & (results_df["attack_window"] == w)]
            vals.append(float(sub["attacker_avg_waiting_time_ns"].iloc[0]))
        plt.bar(x + (k - (len(windows)-1)/2) * width, vals, width=width, label=f"{w} avg wait")

    plt.xticks(x, [safe_tag(v) for v in victims], rotation=30, ha="right")
    plt.ylabel("Avg waiting time (ns)")
    plt.title("QAOA family fingerprints under best attack (sequential-v2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qaoa_family_best_attack_sequential_v2_request_metrics.png", dpi=300)
    plt.show()


def plot_pairwise_distance(dist_df: pd.DataFrame):
    plt.figure(figsize=(8, 7))
    mat = dist_df.to_numpy(dtype=float)
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="Fingerprint distance")

    labels = [safe_tag(x) for x in dist_df.index.tolist()]
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("QAOA family pairwise fingerprint distance (sequential-v2)")
    plt.tight_layout()
    plt.savefig("qaoa_family_best_attack_sequential_v2_pairwise_distance.png", dpi=300)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    results = []

    for victim_qasm in QAOA_QASMS:
        for attack_window in ATTACK_WINDOWS:
            print(f"Running {victim_qasm} / {attack_window['name']}")
            result = run_one_experiment(victim_qasm, attack_window)
            print(result)
            results.append(result)

    results_df = pd.DataFrame(results).sort_values(["victim_qasm", "attack_window"]).reset_index(drop=True)
    fingerprint_df = build_fingerprint_table(results_df)

    feature_cols = [c for c in fingerprint_df.columns if c not in {
        "victim_qasm", "victim_total_ops", "victim_cross_module_ops",
        "victim_cross_fraction", "completed_stage_count"
    }]
    dist_df = pairwise_distance_matrix(fingerprint_df, feature_cols)

    with open("qaoa_family_best_attack_sequential_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    results_df.to_csv("qaoa_family_best_attack_sequential_v2_summary.csv", index=False)
    fingerprint_df.to_csv("qaoa_family_best_attack_sequential_v2_fingerprints.csv", index=False)
    dist_df.to_csv("qaoa_family_best_attack_sequential_v2_pairwise_distance.csv")

    print("\nSaved:")
    print("  qaoa_family_best_attack_sequential_v2_results.json")
    print("  qaoa_family_best_attack_sequential_v2_summary.csv")
    print("  qaoa_family_best_attack_sequential_v2_fingerprints.csv")
    print("  qaoa_family_best_attack_sequential_v2_pairwise_distance.csv")

    print("\n=== Fingerprint summary ===")
    print(fingerprint_df.to_string(index=False))

    print("\n=== Pairwise fingerprint distance ===")
    print(dist_df.round(3).to_string())

    plot_request_metrics(results_df)
    plot_pairwise_distance(dist_df)


if __name__ == "__main__":
    main()
