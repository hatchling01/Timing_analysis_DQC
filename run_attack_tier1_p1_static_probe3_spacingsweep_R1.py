#!/usr/bin/env python3
"""
run_attack_tier1_p1_static_probe3_spacing_sweep_R1.py

Probe 3 spacing sweep at fixed dense probe rate (R1-like).

Fixed:
- five-node Architecture M
- static distributed victim
- P1 disjoint placement
- Tier-1 attacker
- A5-like light schedule
- probe rate fixed to dense

Sweep:
- spacing pattern of cross-module probe gates

Outputs:
- one JSON per victim
- one request-level plot per victim
- one job-makespan plot per victim
- one job-counts plot per victim
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

# Fixed dense rate: about 30 probe injections over 30 rounds
SPACING_SWEEP_R1 = [
    {"name": "S1_uniform_dense", "pattern": "uniform", "total_rounds": 30, "num_probes": 30},
    {"name": "S2_pairs_dense", "pattern": "pairs", "total_rounds": 30, "num_probes": 30},
    {"name": "S3_clustered_dense", "pattern": "clustered", "total_rounds": 30, "num_probes": 30},
    {"name": "S4_alt_dense", "pattern": "alternating", "total_rounds": 30, "num_probes": 30},
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


def pick_probe_rounds(total_rounds: int, num_probes: int, pattern: str):
    if num_probes > total_rounds:
        return list(range(total_rounds))

    if pattern == "uniform":
        if num_probes == total_rounds:
            return list(range(total_rounds))
        step = (total_rounds - 1) / max(1, num_probes - 1)
        return sorted(set(int(round(i * step)) for i in range(num_probes)))

    if pattern == "pairs":
        rounds = []
        base_gap = max(2, total_rounds // max(1, num_probes // 2))
        r = 0
        while len(rounds) < num_probes and r < total_rounds:
            rounds.append(r)
            if len(rounds) < num_probes and r + 1 < total_rounds:
                rounds.append(r + 1)
            r += base_gap
        return sorted(rounds[:num_probes])

    if pattern == "clustered":
        cluster_centers = [total_rounds // 4, total_rounds // 2, (3 * total_rounds) // 4]
        rounds = []
        idx = 0
        while len(rounds) < num_probes:
            c = cluster_centers[idx % len(cluster_centers)]
            for offset in [-2, -1, 0, 1, 2]:
                rr = c + offset
                if 0 <= rr < total_rounds and len(rounds) < num_probes:
                    rounds.append(rr)
            idx += 1
        return sorted(rounds[:num_probes])

    if pattern == "alternating":
        rounds = []
        r = 0
        short_gap = 1
        long_gap = max(3, total_rounds // max(2, num_probes // 2))
        toggle = True
        while len(rounds) < num_probes and r < total_rounds:
            rounds.append(r)
            r += short_gap if toggle else long_gap
            toggle = not toggle
        while len(rounds) < num_probes and rounds[-1] + 1 < total_rounds:
            rounds.append(rounds[-1] + 1)
        return sorted(rounds[:num_probes])

    raise ValueError(f"Unknown pattern: {pattern}")


def build_probe3_spacing_variant(spacing_cfg: dict):
    total_rounds = spacing_cfg["total_rounds"]
    probe_rounds = set(pick_probe_rounds(total_rounds, spacing_cfg["num_probes"], spacing_cfg["pattern"]))

    qc = QuantumCircuit(4)
    for r in range(total_rounds):
        qc.h(0)
        qc.h(1)
        qc.x(2)
        qc.x(3)
        qc.z(0)
        qc.z(2)
        if r in probe_rounds:
            qc.cx(0, 2)
    return qc


def generate_probe3_spacing_trace(spacing_cfg: dict):
    q2m = attacker_probe_qubit_map()
    qc = build_probe3_spacing_variant(spacing_cfg)
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


def run_one_experiment(victim_qasm: str, spacing_cfg: dict):
    victim_trace, victim_q2m, victim_num_qubits = extract_static_trace(victim_qasm, VICTIM_MODULES)
    attacker_trace, attacker_q2m, _ = generate_probe3_spacing_trace(spacing_cfg)
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
        "probe": "probe_3_spacing_sweep_R1",
        "spacing_name": spacing_cfg["name"],
        "pattern": spacing_cfg["pattern"],
        "total_rounds": spacing_cfg["total_rounds"],
        "num_probes": spacing_cfg["num_probes"],
        "workload_type": "static_distributed",
        **attacker_request_metrics(attacker_completed),
        **attacker_job_metrics(attacker_completed, attacker_first_release_ns, attacker_last_release_ns, attacker_event_count),
        "hub_makespan_ns": arch.hub.current_time_ns,
    }


def plot_request_level(results, victim_tag):
    labels = [r["spacing_name"] for r in results]
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
    plt.title(f"Probe 3 R1 Spacing Sweep: Request-Level ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"probe3_spacing_R1_{victim_tag}_request_level.png", dpi=300)
    plt.show()


def plot_job_makespan(results, victim_tag):
    labels = [r["spacing_name"] for r in results]
    makespan = [r["attacker_job_makespan_ns"] for r in results]

    x = list(range(len(labels)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, makespan)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Time (ns)")
    plt.title(f"Probe 3 R1 Spacing Sweep: Job Makespan ({victim_tag})")
    plt.tight_layout()
    plt.savefig(f"probe3_spacing_R1_{victim_tag}_job_makespan.png", dpi=300)
    plt.show()


def plot_job_counts(results, victim_tag):
    labels = [r["spacing_name"] for r in results]
    completed = [r["attacker_completed_requests"] for r in results]
    waited = [r["attacker_num_waited_requests"] for r in results]

    x = list(range(len(labels)))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - w/2 for i in x], completed, width=w, label="Completed reqs")
    plt.bar([i + w/2 for i in x], waited, width=w, label="Waited reqs")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title(f"Probe 3 R1 Spacing Sweep: Job Counts ({victim_tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"probe3_spacing_R1_{victim_tag}_job_counts.png", dpi=300)
    plt.show()


def main():
    for victim_qasm in VICTIM_QASMS:
        victim_tag = safe_tag(victim_qasm)
        results = []
        for spacing_cfg in SPACING_SWEEP_R1:
            print(f"Running R1 spacing sweep on {victim_tag}: {spacing_cfg['name']}")
            results.append(run_one_experiment(victim_qasm, spacing_cfg))

        out_json = f"probe3_spacing_R1_{victim_tag}_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        plot_request_level(results, victim_tag)
        plot_job_makespan(results, victim_tag)
        plot_job_counts(results, victim_tag)


if __name__ == "__main__":
    main()
