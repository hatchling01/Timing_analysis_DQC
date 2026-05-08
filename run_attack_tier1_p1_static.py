#!/usr/bin/env python3
"""
run_attack_tier1_p1_static.py

Tier 1 attacker on P1 placement for STATIC distributed execution only.

P1 placement:
- victim uses module_0, module_1, module_2
- attacker uses module_3, module_4
- no shared compute modules
- contention occurs only through the shared hub

Schedules:
A1 = victim-only baseline
A2 = always-on overlap
A3 = front-loaded
A4 = back-loaded
A5 = periodic probe
A6 = bursty synchronized
A7 = saturation

Outputs:
- tier1_p1_static_results.json
- tier1_p1_static_request_level.png
- tier1_p1_static_job_level.png
"""

import json
import copy
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from new_arch_fivenode_traceadded import (
    FiveModuleLocalModularSuperconductingDQC,
    normalize_trace_entry,
)

# ============================================================
# Fixed Tier-1 / P1 configuration
# ============================================================

VICTIM_MODULES = ["module_0", "module_1", "module_2"]
ATTACKER_MODULES = ["module_3", "module_4"]

SCHEDULES = [
    "A1_victim_only",
    "A2_always_on_overlap",
    "A3_front_loaded",
    "A4_back_loaded",
    "A5_periodic_probe",
    "A6_bursty_synchronized",
    "A7_saturation",
]


# ============================================================
# Static trace extraction
# ============================================================

def build_subset_qubit_map(num_qubits: int, module_subset):
    """
    Assign qubits across an arbitrary subset of modules using contiguous blocks.
    """
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
    """
    Extract static distributed trace over a chosen module subset.
    """
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


def offset_trace_qubits(trace, offset):
    """
    Shift qubit indices so victim and attacker can coexist in one architecture.
    """
    out = []
    for entry in trace:
        e = copy.deepcopy(entry)
        e["qubits"] = [q + offset for q in entry["qubits"]]
        out.append(e)
    return out


def clone_trace_with_new_steps(trace, num_repeats):
    """
    Repeat a trace and reassign step indices.
    """
    out = []
    step = 0
    for _ in range(num_repeats):
        for entry in trace:
            e = copy.deepcopy(entry)
            e["step"] = step
            out.append(e)
            step += 1
    return out


def combine_victim_attacker_mappings(v_q2m, v_num_qubits, a_q2m):
    """
    Victim qubits remain [0 .. v_num_qubits-1]
    Attacker qubits shift to [v_num_qubits .. ]
    """
    combined = {}

    for q, mod in v_q2m.items():
        combined[q] = mod

    for q, mod in a_q2m.items():
        combined[q + v_num_qubits] = mod

    return combined


# ============================================================
# Attack schedules
# ============================================================

def schedule_A1(victim_trace, attacker_trace):
    return [("victim", e) for e in victim_trace]


def schedule_A2(victim_trace, attacker_trace):
    merged = []
    i = j = 0
    while i < len(victim_trace) or j < len(attacker_trace):
        if i < len(victim_trace):
            merged.append(("victim", victim_trace[i]))
            i += 1
        if j < len(attacker_trace):
            merged.append(("attacker", attacker_trace[j]))
            j += 1
    return merged


def schedule_A3(victim_trace, attacker_trace):
    return [("attacker", e) for e in attacker_trace] + [("victim", e) for e in victim_trace]


def schedule_A4(victim_trace, attacker_trace):
    return [("victim", e) for e in victim_trace] + [("attacker", e) for e in attacker_trace]


def schedule_A5(victim_trace, attacker_trace, probe_period=8):
    merged = []
    a_idx = 0
    for i, v in enumerate(victim_trace):
        merged.append(("victim", v))
        if (i + 1) % probe_period == 0 and a_idx < len(attacker_trace):
            merged.append(("attacker", attacker_trace[a_idx]))
            a_idx += 1

    while a_idx < len(attacker_trace):
        merged.append(("attacker", attacker_trace[a_idx]))
        a_idx += 1

    return merged


def schedule_A6(victim_trace, attacker_trace, burst_size=3):
    merged = []
    a_idx = 0
    for v in victim_trace:
        merged.append(("victim", v))
        is_cross = v.get("is_cross_module", v.get("cross_module", False))
        if is_cross:
            for _ in range(burst_size):
                if a_idx < len(attacker_trace):
                    merged.append(("attacker", attacker_trace[a_idx]))
                    a_idx += 1

    while a_idx < len(attacker_trace):
        merged.append(("attacker", attacker_trace[a_idx]))
        a_idx += 1

    return merged


def schedule_A7(victim_trace, attacker_trace, saturation_ratio=3):
    merged = []
    a_idx = 0
    for v in victim_trace:
        merged.append(("victim", v))
        for _ in range(saturation_ratio):
            if a_idx < len(attacker_trace):
                merged.append(("attacker", attacker_trace[a_idx]))
                a_idx += 1

    while a_idx < len(attacker_trace):
        merged.append(("attacker", attacker_trace[a_idx]))
        a_idx += 1

    return merged


def build_schedule(schedule_name, victim_trace, attacker_trace):
    if schedule_name == "A1_victim_only":
        return schedule_A1(victim_trace, attacker_trace)
    if schedule_name == "A2_always_on_overlap":
        return schedule_A2(victim_trace, attacker_trace)
    if schedule_name == "A3_front_loaded":
        return schedule_A3(victim_trace, attacker_trace)
    if schedule_name == "A4_back_loaded":
        return schedule_A4(victim_trace, attacker_trace)
    if schedule_name == "A5_periodic_probe":
        return schedule_A5(victim_trace, attacker_trace, probe_period=8)
    if schedule_name == "A6_bursty_synchronized":
        return schedule_A6(victim_trace, attacker_trace, burst_size=3)
    if schedule_name == "A7_saturation":
        return schedule_A7(victim_trace, attacker_trace, saturation_ratio=3)
    raise ValueError(f"Unknown schedule: {schedule_name}")


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
# Experiment execution
# ============================================================

def run_one_experiment(victim_qasm: str, attacker_qasm: str, schedule_name: str):
    # Victim and attacker both use STATIC distributed execution here
    victim_trace, victim_q2m, victim_num_qubits = extract_static_trace(
        victim_qasm, VICTIM_MODULES
    )
    attacker_trace, attacker_q2m, attacker_num_qubits = extract_static_trace(
        attacker_qasm, ATTACKER_MODULES
    )

    # Make attacker stronger for more aggressive schedules
    if schedule_name in {"A6_bursty_synchronized", "A7_saturation"}:
        attacker_trace = clone_trace_with_new_steps(attacker_trace, num_repeats=3)
    elif schedule_name in {"A2_always_on_overlap", "A3_front_loaded", "A4_back_loaded"}:
        attacker_trace = clone_trace_with_new_steps(attacker_trace, num_repeats=2)

    # Shift attacker qubits into disjoint global index space
    attacker_trace = offset_trace_qubits(attacker_trace, victim_num_qubits)

    # Combined architecture mapping
    combined_q2m = combine_victim_attacker_mappings(
        victim_q2m,
        victim_num_qubits,
        attacker_q2m,
    )

    # Fresh architecture instance
    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=combined_q2m,
        link_latency_ns=10,
        hub_max_concurrent_transfers=2,
        hub_setup_latency_ns=20,
        hub_transfer_latency_ns=80,
        event_tick_ns=5,
    )

    # Build merged stream
    merged_stream = build_schedule(schedule_name, victim_trace, attacker_trace)

    attacker_first_release_ns = None
    attacker_last_release_ns = None
    attacker_event_count = 0

    # Run stream
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

    # Attacker sees only its own completed requests
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

    result = {
        "schedule": schedule_name,
        "workload_type": "static_distributed",
        **req_metrics,
        **job_metrics,
        "hub_makespan_ns": arch.hub.current_time_ns,
    }
    return result


# ============================================================
# Plotting
# ============================================================

def plot_request_level(results):
    labels = [r["schedule"] for r in results]
    avg_wait = [r["attacker_avg_waiting_time_ns"] for r in results]
    avg_turn = [r["attacker_avg_turnaround_time_ns"] for r in results]
    max_wait = [r["attacker_max_waiting_time_ns"] for r in results]

    x = list(range(len(labels)))
    w = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar([i - w for i in x], avg_wait, width=w, label="Avg wait")
    plt.bar(x, avg_turn, width=w, label="Avg turnaround")
    plt.bar([i + w for i in x], max_wait, width=w, label="Max wait")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Time (ns)")
    plt.title("Tier 1 / P1 / Static: Attacker Request-Level Observations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tier1_p1_static_request_level.png", dpi=300)
    plt.show()


def plot_job_level_time(results):
    labels = [r["schedule"] for r in results]
    makespan = [r["attacker_job_makespan_ns"] for r in results]

    x = list(range(len(labels)))

    plt.figure(figsize=(10, 5))
    plt.bar(x, makespan)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Time (ns)")
    plt.title("Tier 1 / P1 / Static: Attacker Job Makespan")
    plt.tight_layout()
    plt.savefig("tier1_p1_static_job_makespan.png", dpi=300)
    plt.show()


def plot_job_level_counts(results):
    labels = [r["schedule"] for r in results]
    completed = [r["attacker_completed_requests"] for r in results]
    waited = [r["attacker_num_waited_requests"] for r in results]

    x = list(range(len(labels)))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - w/2 for i in x], completed, width=w, label="Completed reqs")
    plt.bar([i + w/2 for i in x], waited, width=w, label="Waited reqs")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Tier 1 / P1 / Static: Attacker Job-Level Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tier1_p1_static_job_counts.png", dpi=300)
    plt.show()

# ============================================================
# Main
# ============================================================

def main():
    victim_qasm = "square_root_n18.qasm"
    attacker_qasm = "square_root_n18.qasm"

    results = []
    for schedule_name in SCHEDULES:
        print(f"\n=== Running static / {schedule_name} ===")
        result = run_one_experiment(
            victim_qasm=victim_qasm,
            attacker_qasm=attacker_qasm,
            schedule_name=schedule_name,
        )
        print(result)
        results.append(result)

    with open("tier1_p1_static_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: tier1_p1_static_results.json")

    plot_request_level(results)
    plot_job_level_time(results)
    plot_job_level_counts(results)


if __name__ == "__main__":
    main()
