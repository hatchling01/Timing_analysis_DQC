#!/usr/bin/env python3
"""
saturation_time_check.py

Purpose:
- run only the five victim algorithms
- report simple timing-related structural counts under the fixed static deployment
- specifically:
    1) total local-only victim operations
    2) total cross-module victim operations
    3) an estimated number of local-only rounds
    4) an estimated number of probe rounds

Interpretation used here:
- "probe rounds" = victim operations that are cross-module under the fixed
  static distributed mapping across module_0, module_1, module_2
- "local-only rounds" = victim operations that stay within one module

This is a structural saturation-check helper, not a full timing simulator.
"""

from qiskit import QuantumCircuit

VICTIM_QASMS = [
    "square_root_n18.qasm",
    "qft_n18.qasm",
    "bv_n19.qasm",
    "dnn_n16.qasm",
    "sat_n11.qasm",
]

VICTIM_MODULES = ["module_0", "module_1", "module_2"]


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


def analyze_victim_qasm(qasm_file: str, module_subset):
    qc = QuantumCircuit.from_qasm_file(qasm_file)
    q2m = build_subset_qubit_map(qc.num_qubits, module_subset)

    total_ops = 0
    local_only_ops = 0
    cross_module_ops = 0

    local_only_steps = []
    cross_module_steps = []

    for step_idx, instruction in enumerate(qc.data):
        qargs = instruction.qubits
        qubit_indices = [qc.find_bit(q).index for q in qargs]

        # Ignore pure classical-only instructions if ever present
        if len(qubit_indices) == 0:
            continue

        total_ops += 1
        touched_modules = {q2m[q] for q in qubit_indices}
        is_cross_module = len(touched_modules) > 1

        if is_cross_module:
            cross_module_ops += 1
            cross_module_steps.append(step_idx)
        else:
            local_only_ops += 1
            local_only_steps.append(step_idx)

    # Here, "rounds" are approximated by operation counts in each class.
    # This is the cleanest first structural estimate.
    local_only_rounds = local_only_ops
    probe_rounds = cross_module_ops

    return {
        "qasm_file": qasm_file,
        "num_qubits": qc.num_qubits,
        "total_ops": total_ops,
        "local_only_ops": local_only_ops,
        "cross_module_ops": cross_module_ops,
        "local_only_rounds": local_only_rounds,
        "probe_rounds": probe_rounds,
        "cross_module_fraction": (cross_module_ops / total_ops) if total_ops else 0.0,
        "first_10_cross_module_steps": cross_module_steps[:10],
        "first_10_local_only_steps": local_only_steps[:10],
    }


def print_report(result):
    print("=" * 70)
    print(f"Victim QASM              : {result['qasm_file']}")
    print(f"Num qubits               : {result['num_qubits']}")
    print(f"Total victim ops         : {result['total_ops']}")
    print(f"Local-only ops           : {result['local_only_ops']}")
    print(f"Cross-module ops         : {result['cross_module_ops']}")
    print(f"Estimated local rounds   : {result['local_only_rounds']}")
    print(f"Estimated probe rounds   : {result['probe_rounds']}")
    print(f"Cross-module fraction    : {result['cross_module_fraction']:.4f}")
    print(f"First 10 cross steps     : {result['first_10_cross_module_steps']}")
    print(f"First 10 local-only steps: {result['first_10_local_only_steps']}")
    print("=" * 70)
    print()


def main():
    all_results = []

    for qasm_file in VICTIM_QASMS:
        result = analyze_victim_qasm(qasm_file, VICTIM_MODULES)
        all_results.append(result)
        print_report(result)

    print("Summary table:")
    print(
        f"{'QASM':<24} {'Ops':>8} {'Local':>8} {'Cross':>8} "
        f"{'LocalRnd':>10} {'ProbeRnd':>10} {'CrossFrac':>10}"
    )
    print("-" * 90)

    for r in all_results:
        print(
            f"{r['qasm_file']:<24} "
            f"{r['total_ops']:>8} "
            f"{r['local_only_ops']:>8} "
            f"{r['cross_module_ops']:>8} "
            f"{r['local_only_rounds']:>10} "
            f"{r['probe_rounds']:>10} "
            f"{r['cross_module_fraction']:>10.4f}"
        )


if __name__ == "__main__":
    main()
