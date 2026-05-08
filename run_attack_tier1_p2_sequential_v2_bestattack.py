#!/usr/bin/env python3
"""
run_attack_tier1_p2_sequential_v2_bestattack.py

Sequential-v2 + overlap placement P2 + Tier-1 attacker.

Placement:
- victim   on module_0, module_1, module_2
- attacker on module_2, module_3
- dummy anchor qubit on module_4 for architecture validation

Fixed attack:
- Probe 3
- R1_dense
- short timescale = P20

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

FIVE_MODULES = [f"module_{i}" for i in range(5)]

# Overlap-P2 placement
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

INTER_STAGE_GAP_NS = 20


# ============================================================
# Helpers
# ============================================================

def safe_tag(path_str: str) -> str:
    stem = Path(path_str).stem
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)


def build_overlap_p2_victim_map(num_qubits: int):
    """
    Victim uses only module_0, module_1, module_2.
    """
    module_subset = VICTIM_MODULES
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


# ============================================================
# Sequential-v2 victim trace extraction
# ============================================================

def extract_sequential_modular_trace_overlap_p2(qasm_file: str):
    """
    Sequential-v2 trace extraction, but with the overlap-P2 victim placement:
    victim only occupies module_0, module_1, module_2.
    """
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    num_qubits = qc.num_qubits
    qubit_to_module = build_overlap_p2_victim_map(num_qubits)

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
# Attacker mapping and Probe 3
# ============================================================

def attacker_probe_qubit_map(victim_num_qubits: int):
    """
    Overlap-P2 attacker:
    q0,q1 -> module_2
    q2,q3 -> module_3
    Offset by victim qubit count.
    """
    return {
        victim_num_qubits + 0: "module_2",
        victim_num_qubits + 1: "module_2",
        victim_num_qubits + 2: "module_3",
        victim_num_qubits + 3: "module_3",
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


def generate_probe3_r1_dense_trace(total_rounds: int, victim_num_qubits: int):
    local_q2m = {
        0: "module_2",
        1: "module_2",
        2: "module_3",
        3: "module_3",
    }
    qc = build_probe3_r1_dense(total_rounds)
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

def run_one_experiment(victim_qasm: str):
    victim_trace, victim_q2m, victim_num_qubits, victim_cross_ops, victim_total_ops = (
        extract_sequential_modular_trace_overlap_p2(victim_qasm)
    )

    total_rounds = max(1, math.ceil(victim_cross_ops * FIXED_TIMESCALE["fraction"]))
    attacker_trace = generate_probe3_r1_dense_trace(total_rounds, victim_num_qubits)

    combined_q2m = {}
    combined_q2m.update(victim_q2m)
    combined_q2m.update(attacker_probe_qubit_map(victim_num_qubits))

    # dummy anchor so all five modules appear
    anchor_idx = victim_num_qubits + 4
    combined_q2m[anchor_idx] = "module_4"

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

    waited_fraction = (
        req["attacker_num_waited_requests"] / req["attacker_completed_requests"]
        if req["attacker_completed_requests"] > 0 else 0.0
    )

    return {
        "victim_qasm": victim_qasm,
        "schedule_name": SCHEDULE_CFG["name"],
        "probe": "probe3_overlap_p2_sequential_v2_bestattack",
        "fixed_rate_name": FIXED_RATE["name"],
        "fixed_rate_density": FIXED_RATE["density"],
        "timescale": FIXED_TIMESCALE["timescale"],
        "duration_name": FIXED_TIMESCALE["duration_name"],
        "fraction_of_victim_cross_ops": FIXED_TIMESCALE["fraction"],
        "total_rounds": total_rounds,
        "victim_total_ops": victim_total_ops,
        "victim_cross_module_ops": victim_cross_ops,
        "victim_cross_fraction": (victim_cross_ops / victim_total_ops) if victim_total_ops else 0.0,
        "workload_type": "sequential_v2_overlap_p2",
        "placement_case": "P2_one_module_overlap",
        "inter_stage_gap_ns": INTER_STAGE_GAP_NS,
        "completed_stage_count": len(run_info["completed_stage_order"]),
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
    plt.title(f"Sequential-v2 Overlap P2 Best Attack: Request-Level ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sequential_v2_overlap_p2_{victim_tag}_request_level.png", dpi=300)
    plt.show()


def plot_job_makespan(results, victim_tag):
    labels = [r["timescale"] for r in results]
    makespan = [r["attacker_job_makespan_ns"] for r in results]

    x = list(range(len(labels)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, makespan)
    plt.xticks(x, labels)
    plt.ylabel("Time (ns)")
    plt.title(f"Sequential-v2 Overlap P2 Best Attack: Job Makespan ({victim_tag})")
    plt.tight_layout()
    plt.savefig(f"sequential_v2_overlap_p2_{victim_tag}_job_makespan.png", dpi=300)
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
    plt.title(f"Sequential-v2 Overlap P2 Best Attack: Job Counts ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"sequential_v2_overlap_p2_{victim_tag}_job_counts.png", dpi=300)
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    for victim_qasm in VICTIM_QASMS:
        victim_tag = safe_tag(victim_qasm)
        results = []

        print(f"Running sequential-v2 overlap P2 best attack on {victim_tag}")
        result = run_one_experiment(victim_qasm)
        print(result)
        results.append(result)

        out_json = f"sequential_v2_overlap_p2_{victim_tag}_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        plot_request_level(results, victim_tag)
        plot_job_makespan(results, victim_tag)
        plot_job_counts(results, victim_tag)


if __name__ == "__main__":
    main()
