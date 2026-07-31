#!/usr/bin/env python3
"""
phase1_01_job_module_allocation.py

Experiment 1.1 — Job-to-module allocation policies.

Research question
-----------------
Does timing leakage remain when victim and attacker jobs are dynamically
placed instead of being manually assigned to P1 or P2?

This script generalizes the original five-module experiment to 5, 8, 12,
and 16 modules while preserving the central hub semantics:

- every nonlocal operation becomes a hub request;
- the hub buffers requests;
- FCFS arbitration is used with deterministic tie-breaking;
- hub capacity limits globally concurrent transfers;
- each module has a finite communication-interface pool;
- each switch path is exclusive while a request is active;
- completion releases hub, path, and endpoint resources;
- arrival, waiting, service, completion, and turnaround are recorded.

The script uses the existing QASM victim workloads when available and falls
back to every *.qasm file matching the configured names.

Primary outputs
---------------
blackbox_window_results/phase1_01_job_module_allocation/
    job_module_allocation_trial_summary.csv
    job_module_allocation_allocations.csv
    job_module_allocation_attacker_observations.csv
    job_module_allocation_sharing_summary.csv
    job_module_allocation_policy_summary.csv
    job_module_allocation_knowledge_views.csv
    job_module_allocation_rejection_summary.csv
    allocation_policy_leakage.png
    overlap_condition_leakage.png
    policy_rejection_rate.png
    policy_fragmentation.png

Notes
-----
1. The attacker-only and victim-only controls retain the exact allocation
   produced for the combined experiment. This counterfactual design isolates
   victim-induced timing changes from allocation changes.
2. Background tenants are included in all three runs. Therefore, subtracting
   attacker-only from combined removes stable background contention.
3. Placement knowledge does not alter the fixed Probe-3 schedule in this
   experiment. Knowledge flags are emitted as separate attacker views without
   rerunning the physical simulation.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import itertools
import json
import math
import random
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit


# ============================================================================
# Experiment configuration
# ============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_01_job_module_allocation"
)

VICTIM_QASMS = [
    "square_root_n18.qasm",
    "qft_n18.qasm",
    "dnn_n16.qasm",
    "sat_n11.qasm",
    "bv_n19.qasm",
]

QASM_SEARCH_ROOTS = [
    Path("."),
    Path("/mnt/data"),
]

ALLOCATION_POLICIES = [
    "fixed_p1",
    "fixed_p2",
    "uniform_random",
    "load_balanced",
    "communication_minimizing",
    "communication_consolidating",
    "endpoint_separation",
    "first_fit_admission",
]

NUM_MODULE_OPTIONS = [5, 8, 12, 16]
TENANT_COUNT_OPTIONS = [2, 3, 4, 8]
VICTIM_MODULE_REQUEST_OPTIONS = [2, 3]
HUB_CAPACITY_OPTIONS = [1, 2]
INTERFACES_PER_MODULE_OPTIONS = [1, 2]

ATTACKER_KNOWS_OWN_PLACEMENT_OPTIONS = [False, True]
ATTACKER_KNOWS_VICTIM_PLACEMENT_OPTIONS = [False, True]

# A module can host this many tenants. Communication-interface capacity is
# modeled separately during remote-operation execution.
MODULE_TENANT_CAPACITY = 2

# Timing model retained from the five-node architecture.
LINK_LATENCY_NS = 10
HUB_SETUP_LATENCY_NS = 20
HUB_TRANSFER_LATENCY_NS = 80
REMOTE_SERVICE_TIME_NS = (
    LINK_LATENCY_NS
    + HUB_SETUP_LATENCY_NS
    + HUB_TRANSFER_LATENCY_NS
)

VICTIM_EVENT_TICK_NS = 5
LOCAL_GATE_DURATION_NS = 5

VICTIM_START_NS = 1_000
ATTACKER_START_NS = 1_000
OBSERVATION_DURATION_NS = 20_000
PROBE_PERIOD_NS = 420
PROBE_REMOTE_OFFSET_NS = 30

BACKGROUND_START_NS = 900
BACKGROUND_REMOTE_REQUESTS = 20
BACKGROUND_WINDOW_NS = 22_000

DEFAULT_MONTE_CARLO_TRIALS = 5
GLOBAL_SEED = 20260730

# The non-noisy simulator reports exact values. Any positive excess latency is
# therefore a detected timing event.
DETECTION_THRESHOLD_NS = 0.0

# Set with --save-request-level. Keeping it optional avoids very large CSVs for
# the full factorial sweep.
SAVE_REQUEST_LEVEL_DEFAULT = False


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class LogicalEvent:
    event_id: int
    release_offset_ns: int
    op_name: str
    qubits: tuple[int, ...]


@dataclass
class JobSpec:
    tenant_id: str
    role: str
    logical_qubits: int
    modules_requested: int
    partition_of_qubit: dict[int, int]
    partition_graph: dict[tuple[int, int], float]
    logical_events: list[LogicalEvent]
    start_time_ns: int

    @property
    def partitions(self) -> list[int]:
        return list(range(self.modules_requested))

    @property
    def remote_partitions(self) -> set[int]:
        endpoints: set[int] = set()
        for (left, right), weight in self.partition_graph.items():
            if weight > 0 and left != right:
                endpoints.add(left)
                endpoints.add(right)
        return endpoints


@dataclass
class Allocation:
    tenant_id: str
    role: str
    accepted: bool
    requested_modules: int
    partition_to_module: dict[int, int] = field(default_factory=dict)
    rejection_reason: str = ""

    @property
    def assigned_modules(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.partition_to_module.values())))


@dataclass(frozen=True)
class RemoteRequest:
    request_id: int
    tenant_id: str
    role: str
    release_time_ns: int
    source_module: int
    target_module: int
    switch_path: int
    logical_event_id: int

    @property
    def endpoints(self) -> tuple[int, int]:
        return (
            min(self.source_module, self.target_module),
            max(self.source_module, self.target_module),
        )


@dataclass
class CompletedRequest:
    request: RemoteRequest
    arrival_time_ns: int
    service_start_time_ns: int
    completion_time_ns: int

    @property
    def waiting_time_ns(self) -> int:
        return self.service_start_time_ns - self.arrival_time_ns

    @property
    def turnaround_time_ns(self) -> int:
        return self.completion_time_ns - self.request.release_time_ns


@dataclass
class ActiveRequest:
    completion_time_ns: int
    serial: int
    request: RemoteRequest
    service_start_time_ns: int

    def as_heap_item(self) -> tuple[int, int, "ActiveRequest"]:
        return (self.completion_time_ns, self.serial, self)


# ============================================================================
# QASM and communication-graph preparation
# ============================================================================

IGNORED_QASM_OPERATIONS = {"barrier", "measure", "reset", "delay"}


def safe_tag(value: str) -> str:
    stem = Path(value).stem
    return "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in stem
    )


def resolve_qasm(filename: str) -> Path:
    for root in QASM_SEARCH_ROOTS:
        direct = root / filename
        if direct.exists():
            return direct.resolve()

    stem = Path(filename).stem
    candidates: list[Path] = []
    for root in QASM_SEARCH_ROOTS:
        if root.exists():
            candidates.extend(root.glob(f"{stem}*.qasm"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find victim QASM {filename!r} in "
            f"{[str(root) for root in QASM_SEARCH_ROOTS]}."
        )

    return sorted(path.resolve() for path in candidates)[0]


def circuit_interaction_graph(
    circuit: QuantumCircuit,
) -> tuple[dict[tuple[int, int], float], list[LogicalEvent]]:
    """Return a weighted logical interaction graph and ordered events."""

    interaction_graph: dict[tuple[int, int], float] = defaultdict(float)
    logical_events: list[LogicalEvent] = []

    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in IGNORED_QASM_OPERATIONS:
            continue

        qubits = tuple(
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        )
        if not qubits:
            continue

        event_id = len(logical_events)
        logical_events.append(
            LogicalEvent(
                event_id=event_id,
                release_offset_ns=event_id * VICTIM_EVENT_TICK_NS,
                op_name=operation.name,
                qubits=qubits,
            )
        )

        if len(qubits) >= 2:
            for first, second in itertools.combinations(sorted(set(qubits)), 2):
                interaction_graph[(first, second)] += 1.0

    return dict(interaction_graph), logical_events


def weighted_degrees(
    num_qubits: int,
    graph: dict[tuple[int, int], float],
) -> dict[int, float]:
    degree = {qubit: 0.0 for qubit in range(num_qubits)}
    for (first, second), weight in graph.items():
        degree[first] += weight
        degree[second] += weight
    return degree


def greedy_logical_partition(
    num_qubits: int,
    graph: dict[tuple[int, int], float],
    num_partitions: int,
) -> dict[int, int]:
    """
    Greedy balanced graph partitioning.

    Qubits are visited by weighted degree. Each qubit is assigned to the
    partition minimizing newly cut interaction weight plus a balance penalty.
    """

    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive.")
    if num_partitions > num_qubits:
        raise ValueError(
            f"Cannot divide {num_qubits} qubits into {num_partitions} nonempty partitions."
        )

    degree = weighted_degrees(num_qubits, graph)
    ordering = sorted(range(num_qubits), key=lambda q: (-degree[q], q))
    partition_of: dict[int, int] = {}
    members: dict[int, list[int]] = {partition: [] for partition in range(num_partitions)}
    target_size = math.ceil(num_qubits / num_partitions)

    adjacency: dict[int, dict[int, float]] = defaultdict(dict)
    for (first, second), weight in graph.items():
        adjacency[first][second] = adjacency[first].get(second, 0.0) + weight
        adjacency[second][first] = adjacency[second].get(first, 0.0) + weight

    # Seed every partition so the requested module count is realized.
    for partition, qubit in enumerate(ordering[:num_partitions]):
        partition_of[qubit] = partition
        members[partition].append(qubit)

    for qubit in ordering[num_partitions:]:
        best_partition = None
        best_score = None

        for partition in range(num_partitions):
            cut_cost = 0.0
            affinity = 0.0
            for neighbor, weight in adjacency.get(qubit, {}).items():
                if neighbor not in partition_of:
                    continue
                if partition_of[neighbor] == partition:
                    affinity += weight
                else:
                    cut_cost += weight

            projected_size = len(members[partition]) + 1
            balance_penalty = max(0, projected_size - target_size) * 4.0
            score = cut_cost - affinity + balance_penalty

            candidate = (score, len(members[partition]), partition)
            if best_score is None or candidate < best_score:
                best_score = candidate
                best_partition = partition

        assert best_partition is not None
        partition_of[qubit] = best_partition
        members[best_partition].append(qubit)

    return partition_of


def partition_communication_graph(
    logical_graph: dict[tuple[int, int], float],
    partition_of_qubit: dict[int, int],
) -> dict[tuple[int, int], float]:
    partition_graph: dict[tuple[int, int], float] = defaultdict(float)

    for (first, second), weight in logical_graph.items():
        left = partition_of_qubit[first]
        right = partition_of_qubit[second]
        if left == right:
            continue
        edge = (min(left, right), max(left, right))
        partition_graph[edge] += weight

    return dict(partition_graph)


@lru_cache(maxsize=None)
def build_victim_job(
    qasm_path: Path,
    modules_requested: int,
) -> JobSpec:
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    logical_graph, events = circuit_interaction_graph(circuit)
    partition_of = greedy_logical_partition(
        circuit.num_qubits,
        logical_graph,
        modules_requested,
    )

    return JobSpec(
        tenant_id="victim",
        role="victim",
        logical_qubits=circuit.num_qubits,
        modules_requested=modules_requested,
        partition_of_qubit=partition_of,
        partition_graph=partition_communication_graph(logical_graph, partition_of),
        logical_events=events,
        start_time_ns=VICTIM_START_NS,
    )


def build_attacker_job() -> JobSpec:
    partition_of = {0: 0, 1: 0, 2: 1, 3: 1}
    partition_graph = {(0, 1): 1.0}

    remote_times = list(
        range(
            ATTACKER_START_NS + PROBE_REMOTE_OFFSET_NS,
            ATTACKER_START_NS + OBSERVATION_DURATION_NS,
            PROBE_PERIOD_NS,
        )
    )

    events = [
        LogicalEvent(
            event_id=probe_id,
            release_offset_ns=release_time - ATTACKER_START_NS,
            op_name="probe3_remote_cx",
            qubits=(0, 2),
        )
        for probe_id, release_time in enumerate(remote_times)
    ]

    return JobSpec(
        tenant_id="attacker",
        role="attacker",
        logical_qubits=4,
        modules_requested=2,
        partition_of_qubit=partition_of,
        partition_graph=partition_graph,
        logical_events=events,
        start_time_ns=ATTACKER_START_NS,
    )


def build_background_job(
    tenant_index: int,
    modules_requested: int,
    rng: random.Random,
) -> JobSpec:
    tenant_id = f"background_{tenant_index}"
    logical_qubits = max(2 * modules_requested, 6)

    # Construct a ring plus a few chords so every requested partition has a
    # communication role.
    logical_graph: dict[tuple[int, int], float] = defaultdict(float)
    for qubit in range(logical_qubits):
        edge = (min(qubit, (qubit + 1) % logical_qubits), max(qubit, (qubit + 1) % logical_qubits))
        if edge[0] != edge[1]:
            logical_graph[edge] += 1.0

    for _ in range(logical_qubits // 2):
        first, second = rng.sample(range(logical_qubits), 2)
        edge = (min(first, second), max(first, second))
        logical_graph[edge] += rng.uniform(0.5, 1.5)

    partition_of = greedy_logical_partition(
        logical_qubits,
        dict(logical_graph),
        modules_requested,
    )

    remote_pairs = list(partition_communication_graph(dict(logical_graph), partition_of))
    if not remote_pairs:
        remote_pairs = [(0, min(1, modules_requested - 1))]

    releases = sorted(
        rng.randint(BACKGROUND_START_NS, BACKGROUND_START_NS + BACKGROUND_WINDOW_NS)
        for _ in range(BACKGROUND_REMOTE_REQUESTS)
    )

    representative_qubit: dict[int, int] = {}
    for qubit, partition in partition_of.items():
        representative_qubit.setdefault(partition, qubit)

    events: list[LogicalEvent] = []
    for event_id, release_time_ns in enumerate(releases):
        left_partition, right_partition = remote_pairs[event_id % len(remote_pairs)]
        events.append(
            LogicalEvent(
                event_id=event_id,
                release_offset_ns=release_time_ns - BACKGROUND_START_NS,
                op_name="background_remote",
                qubits=(
                    representative_qubit[left_partition],
                    representative_qubit[right_partition],
                ),
            )
        )

    return JobSpec(
        tenant_id=tenant_id,
        role="background",
        logical_qubits=logical_qubits,
        modules_requested=modules_requested,
        partition_of_qubit=partition_of,
        partition_graph=partition_communication_graph(dict(logical_graph), partition_of),
        logical_events=events,
        start_time_ns=BACKGROUND_START_NS,
    )


# ============================================================================
# Allocation policies
# ============================================================================


def ring_distance(first: int, second: int, num_modules: int) -> int:
    direct = abs(first - second)
    return min(direct, num_modules - direct)


def candidate_subsets(
    num_modules: int,
    requested: int,
) -> Iterator[tuple[int, ...]]:
    return itertools.combinations(range(num_modules), requested)


def subset_feasible(
    subset: Sequence[int],
    module_loads: dict[int, int],
) -> bool:
    return all(module_loads[module] < MODULE_TENANT_CAPACITY for module in subset)


def partition_mapping_for_subset(
    job: JobSpec,
    subset: Sequence[int],
    num_modules: int,
    optimize_communication: bool,
) -> dict[int, int]:
    partitions = list(job.partitions)
    modules = tuple(subset)

    if not optimize_communication or len(partitions) <= 1:
        return dict(zip(partitions, modules))

    best_mapping: dict[int, int] | None = None
    best_cost: float | None = None

    for permutation in itertools.permutations(modules):
        mapping = dict(zip(partitions, permutation))
        cost = 0.0
        for (left, right), weight in job.partition_graph.items():
            cost += weight * ring_distance(mapping[left], mapping[right], num_modules)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_mapping = mapping

    assert best_mapping is not None
    return best_mapping


def remote_endpoint_modules(
    job: JobSpec,
    allocation: Allocation,
) -> set[int]:
    if not allocation.accepted:
        return set()
    return {
        allocation.partition_to_module[partition]
        for partition in job.remote_partitions
        if partition in allocation.partition_to_module
    }


def allocate_generic_job(
    job: JobSpec,
    policy: str,
    num_modules: int,
    module_loads: dict[int, int],
    active_modules: set[int],
    occupied_remote_endpoints: set[int],
    rng: random.Random,
) -> Allocation:
    requested = job.modules_requested

    feasible = [
        subset
        for subset in candidate_subsets(num_modules, requested)
        if subset_feasible(subset, module_loads)
    ]

    if policy == "endpoint_separation":
        feasible = [
            subset
            for subset in feasible
            if not (set(subset) & occupied_remote_endpoints)
        ]

    if not feasible:
        return Allocation(
            tenant_id=job.tenant_id,
            role=job.role,
            accepted=False,
            requested_modules=requested,
            rejection_reason="no_feasible_module_subset",
        )

    if policy == "uniform_random":
        chosen = rng.choice(feasible)
        mapping = partition_mapping_for_subset(
            job,
            chosen,
            num_modules,
            optimize_communication=False,
        )

    elif policy == "load_balanced":
        chosen = min(
            feasible,
            key=lambda subset: (
                sum(module_loads[module] for module in subset),
                max(module_loads[module] for module in subset),
                len(set(subset) & active_modules),
                tuple(subset),
            ),
        )
        mapping = partition_mapping_for_subset(job, chosen, num_modules, False)

    elif policy == "communication_minimizing":
        best: tuple[float, int, tuple[int, ...], dict[int, int]] | None = None
        for subset in feasible:
            mapping = partition_mapping_for_subset(job, subset, num_modules, True)
            communication_cost = sum(
                weight * ring_distance(mapping[left], mapping[right], num_modules)
                for (left, right), weight in job.partition_graph.items()
            )
            load_cost = sum(module_loads[module] for module in subset)
            candidate = (communication_cost, load_cost, tuple(subset), mapping)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        assert best is not None
        mapping = best[3]

    elif policy == "communication_consolidating":
        chosen = min(
            feasible,
            key=lambda subset: (
                len(set(subset) - active_modules),
                -sum(module_loads[module] for module in subset),
                tuple(subset),
            ),
        )
        mapping = partition_mapping_for_subset(job, chosen, num_modules, True)

    elif policy in {"endpoint_separation", "first_fit_admission"}:
        chosen = min(feasible)
        mapping = partition_mapping_for_subset(job, chosen, num_modules, False)

    else:
        raise ValueError(f"Unsupported generic policy: {policy}")

    return Allocation(
        tenant_id=job.tenant_id,
        role=job.role,
        accepted=True,
        requested_modules=requested,
        partition_to_module=mapping,
    )


def commit_allocation(
    job: JobSpec,
    allocation: Allocation,
    module_loads: dict[int, int],
    active_modules: set[int],
    occupied_remote_endpoints: set[int],
) -> None:
    if not allocation.accepted:
        return

    for module in allocation.assigned_modules:
        module_loads[module] += 1
        active_modules.add(module)

    occupied_remote_endpoints.update(remote_endpoint_modules(job, allocation))


def fixed_p1_or_p2_allocation(
    jobs: list[JobSpec],
    policy: str,
    num_modules: int,
    rng: random.Random,
) -> dict[str, Allocation]:
    """Generalize the original fixed P1/P2 layouts to N modules."""

    jobs_by_id = {job.tenant_id: job for job in jobs}
    victim = jobs_by_id["victim"]
    attacker = jobs_by_id["attacker"]

    module_loads = {module: 0 for module in range(num_modules)}
    active_modules: set[int] = set()
    occupied_endpoints: set[int] = set()
    allocations: dict[str, Allocation] = {}

    victim_subset = tuple(range(victim.modules_requested))
    if not subset_feasible(victim_subset, module_loads):
        allocations["victim"] = Allocation(
            "victim", "victim", False, victim.modules_requested,
            rejection_reason="fixed_victim_subset_unavailable",
        )
        return allocations

    victim_allocation = Allocation(
        tenant_id="victim",
        role="victim",
        accepted=True,
        requested_modules=victim.modules_requested,
        partition_to_module=partition_mapping_for_subset(
            victim,
            victim_subset,
            num_modules,
            optimize_communication=True,
        ),
    )
    allocations["victim"] = victim_allocation
    commit_allocation(victim, victim_allocation, module_loads, active_modules, occupied_endpoints)

    if policy == "fixed_p1":
        disjoint_candidates = [
            subset
            for subset in candidate_subsets(num_modules, attacker.modules_requested)
            if not (set(subset) & set(victim_subset))
            and subset_feasible(subset, module_loads)
        ]
        if not disjoint_candidates:
            attacker_allocation = Allocation(
                "attacker", "attacker", False, attacker.modules_requested,
                rejection_reason="fixed_p1_disjoint_subset_unavailable",
            )
        else:
            chosen = max(disjoint_candidates)
            attacker_allocation = Allocation(
                "attacker", "attacker", True, attacker.modules_requested,
                partition_to_module=partition_mapping_for_subset(
                    attacker, chosen, num_modules, False
                ),
            )

    else:
        shared_module = victim_subset[-1]
        other_candidates = [
            module
            for module in range(num_modules)
            if module != shared_module
            and module_loads[module] < MODULE_TENANT_CAPACITY
        ]
        if not other_candidates or module_loads[shared_module] >= MODULE_TENANT_CAPACITY:
            attacker_allocation = Allocation(
                "attacker", "attacker", False, attacker.modules_requested,
                rejection_reason="fixed_p2_overlap_subset_unavailable",
            )
        else:
            nonshared = next(
                (module for module in other_candidates if module not in victim_subset),
                other_candidates[0],
            )
            chosen = (shared_module, nonshared)
            attacker_allocation = Allocation(
                "attacker", "attacker", True, attacker.modules_requested,
                partition_to_module=partition_mapping_for_subset(
                    attacker, chosen, num_modules, False
                ),
            )

    allocations["attacker"] = attacker_allocation
    commit_allocation(attacker, attacker_allocation, module_loads, active_modules, occupied_endpoints)

    for job in jobs:
        if job.tenant_id in {"victim", "attacker"}:
            continue
        allocation = allocate_generic_job(
            job,
            "first_fit_admission",
            num_modules,
            module_loads,
            active_modules,
            occupied_endpoints,
            rng,
        )
        allocations[job.tenant_id] = allocation
        commit_allocation(job, allocation, module_loads, active_modules, occupied_endpoints)

    return allocations


def allocate_jobs(
    jobs: list[JobSpec],
    policy: str,
    num_modules: int,
    rng: random.Random,
) -> dict[str, Allocation]:
    if policy in {"fixed_p1", "fixed_p2"}:
        return fixed_p1_or_p2_allocation(jobs, policy, num_modules, rng)

    module_loads = {module: 0 for module in range(num_modules)}
    active_modules: set[int] = set()
    occupied_remote_endpoints: set[int] = set()
    allocations: dict[str, Allocation] = {}

    # Victim, attacker, then background jobs. This ordering is explicit so the
    # first-fit policy has a reproducible admission meaning.
    role_priority = {"victim": 0, "attacker": 1, "background": 2}
    ordered_jobs = sorted(jobs, key=lambda job: (role_priority[job.role], job.tenant_id))

    for job in ordered_jobs:
        allocation = allocate_generic_job(
            job,
            policy,
            num_modules,
            module_loads,
            active_modules,
            occupied_remote_endpoints,
            rng,
        )
        allocations[job.tenant_id] = allocation
        commit_allocation(
            job,
            allocation,
            module_loads,
            active_modules,
            occupied_remote_endpoints,
        )

    return allocations


# ============================================================================
# Resource-sharing characterization
# ============================================================================


def switch_path_for_pair(first: int, second: int, num_switch_paths: int) -> int:
    left, right = min(first, second), max(first, second)
    return (left * 31 + right * 17) % num_switch_paths


def job_physical_pairs(
    job: JobSpec,
    allocation: Allocation,
) -> set[tuple[int, int]]:
    if not allocation.accepted:
        return set()

    pairs: set[tuple[int, int]] = set()
    for (left_partition, right_partition), weight in job.partition_graph.items():
        if weight <= 0:
            continue
        left = allocation.partition_to_module[left_partition]
        right = allocation.partition_to_module[right_partition]
        if left == right:
            continue
        pairs.add((min(left, right), max(left, right)))
    return pairs


def sharing_characteristics(
    victim_job: JobSpec,
    attacker_job: JobSpec,
    victim_allocation: Allocation,
    attacker_allocation: Allocation,
    hub_capacity: int,
) -> dict[str, object]:
    if not victim_allocation.accepted or not attacker_allocation.accepted:
        return {
            "shared_hub_service": False,
            "shared_link": False,
            "shared_module": False,
            "shared_communication_pool": False,
            "shared_switch_path": False,
            "p1_like_disjoint": False,
            "p2_like_endpoint_overlap": False,
            "actual_sharing_condition": "rejected",
        }

    victim_modules = set(victim_allocation.assigned_modules)
    attacker_modules = set(attacker_allocation.assigned_modules)
    shared_modules = victim_modules & attacker_modules

    victim_endpoints = remote_endpoint_modules(victim_job, victim_allocation)
    attacker_endpoints = remote_endpoint_modules(attacker_job, attacker_allocation)
    shared_endpoint_modules = victim_endpoints & attacker_endpoints

    victim_pairs = job_physical_pairs(victim_job, victim_allocation)
    attacker_pairs = job_physical_pairs(attacker_job, attacker_allocation)

    num_switch_paths = max(1, hub_capacity)
    victim_paths = {
        switch_path_for_pair(left, right, num_switch_paths)
        for left, right in victim_pairs
    }
    attacker_paths = {
        switch_path_for_pair(left, right, num_switch_paths)
        for left, right in attacker_pairs
    }

    shared_hub = bool(victim_pairs and attacker_pairs)
    shared_link = bool(shared_endpoint_modules)
    shared_module = bool(shared_modules)
    shared_pool = bool(shared_endpoint_modules)
    shared_path = bool(victim_paths & attacker_paths)

    p2_like = shared_link or shared_pool
    p1_like = shared_hub and not shared_module and not shared_link and not shared_pool

    if p2_like:
        condition = "endpoint_overlap"
    elif shared_path:
        condition = "switch_path_only"
    elif shared_hub:
        condition = "hub_only"
    else:
        condition = "no_shared_remote_resource"

    return {
        "shared_hub_service": shared_hub,
        "shared_link": shared_link,
        "shared_module": shared_module,
        "shared_communication_pool": shared_pool,
        "shared_switch_path": shared_path,
        "p1_like_disjoint": p1_like,
        "p2_like_endpoint_overlap": p2_like,
        "actual_sharing_condition": condition,
        "shared_module_count": len(shared_modules),
        "shared_endpoint_module_count": len(shared_endpoint_modules),
        "shared_switch_path_count": len(victim_paths & attacker_paths),
        "victim_remote_pair_count": len(victim_pairs),
        "attacker_remote_pair_count": len(attacker_pairs),
    }


# ============================================================================
# Hub request generation and simulation
# ============================================================================


def build_remote_requests(
    jobs: Iterable[JobSpec],
    allocations: dict[str, Allocation],
    hub_capacity: int,
) -> list[RemoteRequest]:
    requests: list[RemoteRequest] = []
    request_id = 0
    num_switch_paths = max(1, hub_capacity)

    for job in jobs:
        allocation = allocations.get(job.tenant_id)
        if allocation is None or not allocation.accepted:
            continue

        for event in job.logical_events:
            touched_partitions = {
                job.partition_of_qubit[qubit]
                for qubit in event.qubits
                if qubit in job.partition_of_qubit
            }

            if len(touched_partitions) < 2:
                continue

            # The current workloads use two-qubit remote events. For an event
            # spanning more than two partitions, create one request per pair.
            for left_partition, right_partition in itertools.combinations(
                sorted(touched_partitions), 2
            ):
                source = allocation.partition_to_module[left_partition]
                target = allocation.partition_to_module[right_partition]
                if source == target:
                    continue

                requests.append(
                    RemoteRequest(
                        request_id=request_id,
                        tenant_id=job.tenant_id,
                        role=job.role,
                        release_time_ns=job.start_time_ns + event.release_offset_ns,
                        source_module=source,
                        target_module=target,
                        switch_path=switch_path_for_pair(
                            source, target, num_switch_paths
                        ),
                        logical_event_id=event.event_id,
                    )
                )
                request_id += 1

    role_priority = {"victim": 0, "background": 1, "attacker": 2}
    return sorted(
        requests,
        key=lambda request: (
            request.release_time_ns,
            role_priority[request.role],
            request.tenant_id,
            request.logical_event_id,
            request.request_id,
        ),
    )


class SharedHubSimulator:
    """FCFS shared-hub simulator with endpoint and switch-path locking."""

    def __init__(
        self,
        hub_capacity: int,
        interfaces_per_module: int,
    ) -> None:
        if hub_capacity <= 0:
            raise ValueError("hub_capacity must be positive.")
        if interfaces_per_module <= 0:
            raise ValueError("interfaces_per_module must be positive.")

        self.hub_capacity = hub_capacity
        self.interfaces_per_module = interfaces_per_module
        self.current_time_ns = 0
        self.waiting: deque[RemoteRequest] = deque()
        self.active_heap: list[tuple[int, int, ActiveRequest]] = []
        self.active_module_interfaces: dict[int, int] = defaultdict(int)
        self.active_switch_paths: set[int] = set()
        self.completed: list[CompletedRequest] = []
        self.serial = 0

    def _can_start(self, request: RemoteRequest) -> bool:
        if len(self.active_heap) >= self.hub_capacity:
            return False
        if request.switch_path in self.active_switch_paths:
            return False
        for module in request.endpoints:
            if self.active_module_interfaces[module] >= self.interfaces_per_module:
                return False
        return True

    def _start(self, request: RemoteRequest) -> None:
        start_time = max(self.current_time_ns, request.release_time_ns)
        completion_time = start_time + REMOTE_SERVICE_TIME_NS

        for module in request.endpoints:
            self.active_module_interfaces[module] += 1
        self.active_switch_paths.add(request.switch_path)

        active = ActiveRequest(
            completion_time_ns=completion_time,
            serial=self.serial,
            request=request,
            service_start_time_ns=start_time,
        )
        self.serial += 1
        heapq.heappush(self.active_heap, active.as_heap_item())

    def _complete_next(self) -> None:
        completion_time, _, active = heapq.heappop(self.active_heap)
        self.current_time_ns = completion_time

        for module in active.request.endpoints:
            self.active_module_interfaces[module] -= 1
        self.active_switch_paths.remove(active.request.switch_path)

        self.completed.append(
            CompletedRequest(
                request=active.request,
                arrival_time_ns=active.request.release_time_ns,
                service_start_time_ns=active.service_start_time_ns,
                completion_time_ns=completion_time,
            )
        )

    def _admit_waiting_fcfs(self) -> bool:
        """
        Admit the earliest feasible queued request.

        Requests that are blocked by endpoint/path resources do not block a
        later request that uses independent resources. Queue order is retained
        among equally feasible requests.
        """
        if not self.waiting or len(self.active_heap) >= self.hub_capacity:
            return False

        for index, request in enumerate(self.waiting):
            if self._can_start(request):
                del self.waiting[index]
                self._start(request)
                return True
        return False

    def run(self, requests: Sequence[RemoteRequest]) -> list[CompletedRequest]:
        release_index = 0
        requests = list(requests)

        while release_index < len(requests) or self.waiting or self.active_heap:
            next_release = (
                requests[release_index].release_time_ns
                if release_index < len(requests)
                else math.inf
            )
            next_completion = self.active_heap[0][0] if self.active_heap else math.inf

            if next_release <= next_completion:
                self.current_time_ns = max(self.current_time_ns, int(next_release))
                while (
                    release_index < len(requests)
                    and requests[release_index].release_time_ns == next_release
                ):
                    self.waiting.append(requests[release_index])
                    release_index += 1

                progress = True
                while progress:
                    progress = self._admit_waiting_fcfs()

                # If nothing can start, the next active completion frees a resource.
                if self.waiting and not self.active_heap:
                    raise RuntimeError(
                        "Hub deadlock: queued request cannot start despite no active request."
                    )

            else:
                self._complete_next()
                progress = True
                while progress:
                    progress = self._admit_waiting_fcfs()

        return self.completed


def completion_dataframe(completed: Sequence[CompletedRequest]) -> pd.DataFrame:
    rows = []
    for item in completed:
        request = item.request
        rows.append(
            {
                "request_id": request.request_id,
                "tenant_id": request.tenant_id,
                "role": request.role,
                "logical_event_id": request.logical_event_id,
                "release_time_ns": request.release_time_ns,
                "source_module": request.source_module,
                "target_module": request.target_module,
                "switch_path": request.switch_path,
                "arrival_time_ns": item.arrival_time_ns,
                "service_start_time_ns": item.service_start_time_ns,
                "completion_time_ns": item.completion_time_ns,
                "service_time_ns": REMOTE_SERVICE_TIME_NS,
                "waiting_time_ns": item.waiting_time_ns,
                "turnaround_time_ns": item.turnaround_time_ns,
            }
        )
    return pd.DataFrame(rows)


def run_scenario(
    jobs: list[JobSpec],
    allocations: dict[str, Allocation],
    included_roles: set[str],
    hub_capacity: int,
    interfaces_per_module: int,
) -> pd.DataFrame:
    included_jobs = [job for job in jobs if job.role in included_roles]
    requests = build_remote_requests(included_jobs, allocations, hub_capacity)
    simulator = SharedHubSimulator(hub_capacity, interfaces_per_module)
    completed = simulator.run(requests)
    return completion_dataframe(completed)


def compare_attacker_observations(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    attacker_baseline = attacker_only[attacker_only["role"] == "attacker"].copy()
    attacker_combined = combined[combined["role"] == "attacker"].copy()

    columns = [
        "logical_event_id",
        "release_time_ns",
        "source_module",
        "target_module",
        "switch_path",
        "waiting_time_ns",
        "turnaround_time_ns",
        "completion_time_ns",
    ]

    baseline = attacker_baseline[columns].rename(
        columns={
            "waiting_time_ns": "baseline_waiting_time_ns",
            "turnaround_time_ns": "baseline_turnaround_time_ns",
            "completion_time_ns": "baseline_completion_time_ns",
        }
    )
    victim_on = attacker_combined[columns].rename(
        columns={
            "waiting_time_ns": "combined_waiting_time_ns",
            "turnaround_time_ns": "combined_turnaround_time_ns",
            "completion_time_ns": "combined_completion_time_ns",
        }
    )

    merged = victim_on.merge(
        baseline,
        on=[
            "logical_event_id",
            "release_time_ns",
            "source_module",
            "target_module",
            "switch_path",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(attacker_baseline):
        raise RuntimeError(
            "Attacker request count differs between attacker-only and combined runs."
        )

    merged["excess_waiting_time_ns"] = (
        merged["combined_waiting_time_ns"]
        - merged["baseline_waiting_time_ns"]
    )
    merged["excess_turnaround_time_ns"] = (
        merged["combined_turnaround_time_ns"]
        - merged["baseline_turnaround_time_ns"]
    )
    merged["victim_contention_observed"] = (
        merged["excess_turnaround_time_ns"] > DETECTION_THRESHOLD_NS
    )

    return merged.sort_values("logical_event_id").reset_index(drop=True)


# ============================================================================
# Allocation utilization and fragmentation
# ============================================================================


def allocation_state_metrics(
    allocations: dict[str, Allocation],
    num_modules: int,
) -> dict[str, float]:
    module_loads = {module: 0 for module in range(num_modules)}
    for allocation in allocations.values():
        if not allocation.accepted:
            continue
        for module in allocation.assigned_modules:
            module_loads[module] += 1

    occupied_modules = {module for module, load in module_loads.items() if load > 0}
    free_modules = [module for module, load in module_loads.items() if load == 0]

    utilization = sum(module_loads.values()) / (
        num_modules * MODULE_TENANT_CAPACITY
    )

    if not free_modules:
        largest_free_block = 0
        free_segment_count = 0
        fragmentation = 0.0
    else:
        free_set = set(free_modules)
        visited: set[int] = set()
        blocks: list[int] = []

        for module in free_modules:
            if module in visited:
                continue
            block = 0
            current = module
            while current in free_set and current not in visited:
                visited.add(current)
                block += 1
                current = (current + 1) % num_modules
            blocks.append(block)

        # Merge first/last segments on the ring if they are actually adjacent.
        if len(blocks) > 1 and 0 in free_set and (num_modules - 1) in free_set:
            # Recompute robustly by cutting the ring at an occupied module.
            if occupied_modules:
                cut = next(iter(occupied_modules))
                ordered = [
                    (cut + offset) % num_modules
                    for offset in range(1, num_modules + 1)
                ]
                blocks = []
                run = 0
                for item in ordered:
                    if item in free_set:
                        run += 1
                    elif run:
                        blocks.append(run)
                        run = 0
                if run:
                    blocks.append(run)

        largest_free_block = max(blocks) if blocks else 0
        free_segment_count = len(blocks)
        fragmentation = (
            1.0 - largest_free_block / len(free_modules)
            if free_modules
            else 0.0
        )

    rejected_jobs = sum(not allocation.accepted for allocation in allocations.values())

    return {
        "allocation_utilization": float(utilization),
        "occupied_module_count": len(occupied_modules),
        "unused_module_count": len(free_modules),
        "largest_contiguous_unused_block": largest_free_block,
        "unused_module_segment_count": free_segment_count,
        "unused_module_fragmentation": float(fragmentation),
        "rejected_job_count": rejected_jobs,
        "job_rejection_rate": rejected_jobs / max(len(allocations), 1),
    }


# ============================================================================
# One Monte Carlo trial
# ============================================================================


def victim_completion_time(
    job: JobSpec,
    dataframe: pd.DataFrame,
) -> float:
    nominal_local_completion = (
        job.start_time_ns
        + (len(job.logical_events) - 1) * VICTIM_EVENT_TICK_NS
        + LOCAL_GATE_DURATION_NS
        if job.logical_events
        else job.start_time_ns
    )

    victim_remote = dataframe[dataframe["role"] == "victim"]
    if victim_remote.empty:
        return float(nominal_local_completion)

    return float(max(nominal_local_completion, victim_remote["completion_time_ns"].max()))


def run_trial(
    *,
    victim_qasm: Path,
    policy: str,
    num_modules: int,
    tenant_count: int,
    victim_modules_requested: int,
    hub_capacity: int,
    interfaces_per_module: int,
    trial_id: int,
    seed: int,
    save_request_level: bool,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame | None]:
    rng = random.Random(seed)

    victim_job = build_victim_job(victim_qasm, victim_modules_requested)
    attacker_job = build_attacker_job()

    background_jobs = [
        build_background_job(
            tenant_index=index,
            modules_requested=victim_modules_requested,
            rng=random.Random(seed + 10_000 + index),
        )
        for index in range(max(0, tenant_count - 2))
    ]

    jobs = [victim_job, attacker_job, *background_jobs]
    allocations = allocate_jobs(jobs, policy, num_modules, rng)

    victim_allocation = allocations.get("victim")
    attacker_allocation = allocations.get("attacker")
    assert victim_allocation is not None
    assert attacker_allocation is not None

    sharing = sharing_characteristics(
        victim_job,
        attacker_job,
        victim_allocation,
        attacker_allocation,
        hub_capacity,
    )
    allocation_metrics = allocation_state_metrics(allocations, num_modules)

    required_jobs_accepted = victim_allocation.accepted and attacker_allocation.accepted

    if not required_jobs_accepted:
        trial_summary: dict[str, object] = {
            "victim_qasm": victim_qasm.name,
            "victim_tag": safe_tag(victim_qasm.name),
            "allocation_policy": policy,
            "num_modules": num_modules,
            "simultaneous_tenants": tenant_count,
            "victim_modules_requested": victim_modules_requested,
            "attacker_modules_requested": attacker_job.modules_requested,
            "hub_capacity": hub_capacity,
            "interfaces_per_module": interfaces_per_module,
            "trial_id": trial_id,
            "seed": seed,
            "victim_accepted": victim_allocation.accepted,
            "attacker_accepted": attacker_allocation.accepted,
            "victim_assigned_modules": ",".join(map(str, victim_allocation.assigned_modules)),
            "attacker_assigned_modules": ",".join(map(str, attacker_allocation.assigned_modules)),
            **sharing,
            **allocation_metrics,
            "attacker_probe_count": 0,
            "baseline_avg_waiting_time_ns": np.nan,
            "avg_excess_turnaround_time_ns": 0.0,
            "max_excess_turnaround_time_ns": 0.0,
            "total_excess_turnaround_time_ns": 0.0,
            "contention_observed_fraction": 0.0,
            "attacker_detection": False,
            "victim_only_completion_ns": np.nan,
            "combined_victim_completion_ns": np.nan,
            "victim_slowdown_ns": np.nan,
            "victim_slowdown_ratio": np.nan,
        }
        return trial_summary, allocation_rows(jobs, allocations, trial_summary), None

    roles_background = {"background"}
    attacker_only = run_scenario(
        jobs,
        allocations,
        included_roles={"attacker", *roles_background},
        hub_capacity=hub_capacity,
        interfaces_per_module=interfaces_per_module,
    )
    victim_only = run_scenario(
        jobs,
        allocations,
        included_roles={"victim", *roles_background},
        hub_capacity=hub_capacity,
        interfaces_per_module=interfaces_per_module,
    )
    combined = run_scenario(
        jobs,
        allocations,
        included_roles={"victim", "attacker", *roles_background},
        hub_capacity=hub_capacity,
        interfaces_per_module=interfaces_per_module,
    )

    compared = compare_attacker_observations(attacker_only, combined)

    victim_only_completion = victim_completion_time(victim_job, victim_only)
    combined_victim_completion = victim_completion_time(victim_job, combined)
    victim_slowdown_ns = combined_victim_completion - victim_only_completion
    victim_slowdown_ratio = (
        (combined_victim_completion - victim_job.start_time_ns)
        / max(victim_only_completion - victim_job.start_time_ns, 1.0)
    )

    trial_summary = {
        "victim_qasm": victim_qasm.name,
        "victim_tag": safe_tag(victim_qasm.name),
        "allocation_policy": policy,
        "num_modules": num_modules,
        "simultaneous_tenants": tenant_count,
        "victim_modules_requested": victim_modules_requested,
        "attacker_modules_requested": attacker_job.modules_requested,
        "hub_capacity": hub_capacity,
        "interfaces_per_module": interfaces_per_module,
        "trial_id": trial_id,
        "seed": seed,
        "victim_accepted": True,
        "attacker_accepted": True,
        "victim_assigned_modules": ",".join(map(str, victim_allocation.assigned_modules)),
        "attacker_assigned_modules": ",".join(map(str, attacker_allocation.assigned_modules)),
        **sharing,
        **allocation_metrics,
        "attacker_probe_count": len(compared),
        "baseline_avg_waiting_time_ns": float(
            attacker_only.loc[attacker_only["role"] == "attacker", "waiting_time_ns"].mean()
        ),
        "baseline_max_waiting_time_ns": float(
            attacker_only.loc[attacker_only["role"] == "attacker", "waiting_time_ns"].max()
        ),
        "avg_excess_turnaround_time_ns": float(
            compared["excess_turnaround_time_ns"].mean()
        ),
        "max_excess_turnaround_time_ns": float(
            compared["excess_turnaround_time_ns"].max()
        ),
        "total_excess_turnaround_time_ns": float(
            compared["excess_turnaround_time_ns"].sum()
        ),
        "contention_observed_fraction": float(
            compared["victim_contention_observed"].mean()
        ),
        "attacker_detection": bool(
            compared["victim_contention_observed"].any()
        ),
        "victim_only_completion_ns": victim_only_completion,
        "combined_victim_completion_ns": combined_victim_completion,
        "victim_slowdown_ns": victim_slowdown_ns,
        "victim_slowdown_ratio": victim_slowdown_ratio,
    }

    request_frame: pd.DataFrame | None = None
    if save_request_level:
        request_frame = compared.copy()
        for column, value in trial_summary.items():
            if column not in request_frame.columns:
                request_frame.insert(0, column, value)

    return (
        trial_summary,
        allocation_rows(jobs, allocations, trial_summary),
        request_frame,
    )


def allocation_rows(
    jobs: list[JobSpec],
    allocations: dict[str, Allocation],
    trial_summary: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in jobs:
        allocation = allocations.get(job.tenant_id)
        if allocation is None:
            continue

        endpoint_modules = remote_endpoint_modules(job, allocation)
        rows.append(
            {
                "victim_qasm": trial_summary["victim_qasm"],
                "allocation_policy": trial_summary["allocation_policy"],
                "num_modules": trial_summary["num_modules"],
                "simultaneous_tenants": trial_summary["simultaneous_tenants"],
                "victim_modules_requested": trial_summary["victim_modules_requested"],
                "hub_capacity": trial_summary["hub_capacity"],
                "interfaces_per_module": trial_summary["interfaces_per_module"],
                "trial_id": trial_summary["trial_id"],
                "seed": trial_summary["seed"],
                "tenant_id": job.tenant_id,
                "role": job.role,
                "accepted": allocation.accepted,
                "requested_modules": allocation.requested_modules,
                "assigned_modules": ",".join(map(str, allocation.assigned_modules)),
                "remote_endpoint_modules": ",".join(map(str, sorted(endpoint_modules))),
                "partition_to_module_json": json.dumps(
                    allocation.partition_to_module,
                    sort_keys=True,
                ),
                "rejection_reason": allocation.rejection_reason,
            }
        )
    return rows


# ============================================================================
# Aggregation and plotting
# ============================================================================


def boolean_mean(series: pd.Series) -> float:
    return float(series.astype(float).mean()) if len(series) else float("nan")


def aggregate_outputs(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = trials[trials["victim_accepted"] & trials["attacker_accepted"]].copy()

    sharing_summary = (
        accepted.groupby(
            [
                "actual_sharing_condition",
                "hub_capacity",
                "interfaces_per_module",
            ],
            as_index=False,
        )
        .agg(
            trials=("trial_id", "count"),
            mean_excess_latency_ns=("avg_excess_turnaround_time_ns", "mean"),
            mean_total_excess_latency_ns=("total_excess_turnaround_time_ns", "mean"),
            mean_contention_fraction=("contention_observed_fraction", "mean"),
            attacker_detection_probability=("attacker_detection", boolean_mean),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
        )
    )

    policy_summary = (
        trials.groupby(
            [
                "allocation_policy",
                "num_modules",
                "simultaneous_tenants",
                "victim_modules_requested",
                "hub_capacity",
                "interfaces_per_module",
            ],
            as_index=False,
        )
        .agg(
            trials=("trial_id", "count"),
            probability_p1_like=("p1_like_disjoint", boolean_mean),
            probability_p2_like=("p2_like_endpoint_overlap", boolean_mean),
            unconditional_mean_excess_latency_ns=("avg_excess_turnaround_time_ns", "mean"),
            unconditional_detection_probability=("attacker_detection", boolean_mean),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
            mean_allocation_utilization=("allocation_utilization", "mean"),
            mean_job_rejection_rate=("job_rejection_rate", "mean"),
            mean_unused_module_fragmentation=("unused_module_fragmentation", "mean"),
        )
    )

    rejection_summary = (
        trials.groupby(
            [
                "allocation_policy",
                "num_modules",
                "simultaneous_tenants",
                "victim_modules_requested",
            ],
            as_index=False,
        )
        .agg(
            trials=("trial_id", "count"),
            victim_rejection_probability=(
                "victim_accepted",
                lambda series: float((~series.astype(bool)).mean()),
            ),
            attacker_rejection_probability=(
                "attacker_accepted",
                lambda series: float((~series.astype(bool)).mean()),
            ),
            mean_job_rejection_rate=("job_rejection_rate", "mean"),
        )
    )

    return sharing_summary, policy_summary, rejection_summary


def knowledge_views(trials: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for knows_own in ATTACKER_KNOWS_OWN_PLACEMENT_OPTIONS:
        for knows_victim in ATTACKER_KNOWS_VICTIM_PLACEMENT_OPTIONS:
            frame = trials.copy()
            frame["attacker_knows_own_placement"] = knows_own
            frame["attacker_knows_victim_placement"] = knows_victim
            frame["attacker_visible_own_modules"] = np.where(
                knows_own,
                np.where(
                    frame["attacker_accepted"],
                    frame["attacker_assigned_modules"],
                    "rejected",
                ),
                "unknown",
            )
            frame["attacker_visible_victim_modules"] = np.where(
                knows_victim,
                np.where(
                    frame["victim_accepted"],
                    frame["victim_assigned_modules"],
                    "rejected",
                ),
                "unknown",
            )
            # Raw timing does not change because Probe 3 is not adapted to
            # placement knowledge in Experiment 1.1.
            frame["knowledge_changes_probe_schedule"] = False
            frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def plot_policy_metric(
    policy_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    collapsed = (
        policy_summary.groupby("allocation_policy", as_index=False)[metric]
        .mean()
        .set_index("allocation_policy")
        .reindex(ALLOCATION_POLICIES)
    )
    axis = collapsed[metric].plot(kind="bar", figsize=(12, 6))
    axis.set_xlabel("Allocation policy")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def plot_sharing_metric(
    sharing_summary: pd.DataFrame,
) -> None:
    collapsed = (
        sharing_summary.groupby("actual_sharing_condition", as_index=False)[
            "mean_excess_latency_ns"
        ]
        .mean()
        .sort_values("mean_excess_latency_ns", ascending=False)
        .set_index("actual_sharing_condition")
    )
    axis = collapsed["mean_excess_latency_ns"].plot(kind="bar", figsize=(10, 6))
    axis.set_xlabel("Actual resource-sharing condition")
    axis.set_ylabel("Mean victim-induced latency (ns)")
    axis.set_title("Leakage Conditioned on Actual Resource Sharing")
    axis.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overlap_condition_leakage.png", dpi=300)
    plt.close()


# ============================================================================
# Main sweep
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_MONTE_CARLO_TRIALS,
        help="Monte Carlo trials per factorial configuration.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a smoke test: one workload, modules {5, 8}, tenants {2, 4}, "
            "one module-request level, one hub/interface level, and two trials."
        ),
    )
    parser.add_argument(
        "--save-request-level",
        action="store_true",
        default=SAVE_REQUEST_LEVEL_DEFAULT,
        help="Save request-by-request attacker observations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qasm_paths = [resolve_qasm(filename) for filename in VICTIM_QASMS]

    if args.quick:
        qasm_paths = qasm_paths[:1]
        num_module_options = [5, 8]
        tenant_count_options = [2, 4]
        victim_module_request_options = [2]
        hub_capacity_options = [1]
        interface_options = [1]
        trials_per_configuration = min(args.trials, 2)
    else:
        num_module_options = NUM_MODULE_OPTIONS
        tenant_count_options = TENANT_COUNT_OPTIONS
        victim_module_request_options = VICTIM_MODULE_REQUEST_OPTIONS
        hub_capacity_options = HUB_CAPACITY_OPTIONS
        interface_options = INTERFACES_PER_MODULE_OPTIONS
        trials_per_configuration = args.trials

    trial_rows: list[dict[str, object]] = []
    allocation_records: list[dict[str, object]] = []
    request_frames: list[pd.DataFrame] = []

    configuration_counter = 0

    for qasm_path in qasm_paths:
        for num_modules in num_module_options:
            for tenant_count in tenant_count_options:
                for victim_modules_requested in victim_module_request_options:
                    if victim_modules_requested > num_modules:
                        continue
                    for hub_capacity in hub_capacity_options:
                        for interfaces_per_module in interface_options:
                            for policy_index, policy in enumerate(ALLOCATION_POLICIES):
                                for trial_id in range(trials_per_configuration):
                                    seed = (
                                        args.seed
                                        + zlib.crc32(qasm_path.name.encode("utf-8")) % 100_000
                                        + num_modules * 1_000_000
                                        + tenant_count * 100_000
                                        + victim_modules_requested * 10_000
                                        + hub_capacity * 1_000
                                        + interfaces_per_module * 100
                                        + policy_index * 10
                                        + trial_id
                                    )

                                    configuration_counter += 1
                                    print(
                                        f"[{configuration_counter}] {qasm_path.name} | "
                                        f"M={num_modules} | tenants={tenant_count} | "
                                        f"req={victim_modules_requested} | hub={hub_capacity} | "
                                        f"ifaces={interfaces_per_module} | {policy} | "
                                        f"trial={trial_id}"
                                    )

                                    summary, allocations, request_frame = run_trial(
                                        victim_qasm=qasm_path,
                                        policy=policy,
                                        num_modules=num_modules,
                                        tenant_count=tenant_count,
                                        victim_modules_requested=victim_modules_requested,
                                        hub_capacity=hub_capacity,
                                        interfaces_per_module=interfaces_per_module,
                                        trial_id=trial_id,
                                        seed=seed,
                                        save_request_level=args.save_request_level,
                                    )

                                    trial_rows.append(summary)
                                    allocation_records.extend(allocations)
                                    if request_frame is not None:
                                        request_frames.append(request_frame)

    trials = pd.DataFrame(trial_rows)
    allocations = pd.DataFrame(allocation_records)

    trials.to_csv(
        OUTPUT_DIR / "job_module_allocation_trial_summary.csv",
        index=False,
    )
    allocations.to_csv(
        OUTPUT_DIR / "job_module_allocation_allocations.csv",
        index=False,
    )

    if request_frames:
        pd.concat(request_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "job_module_allocation_attacker_observations.csv",
            index=False,
        )

    sharing_summary, policy_summary, rejection_summary = aggregate_outputs(trials)
    sharing_summary.to_csv(
        OUTPUT_DIR / "job_module_allocation_sharing_summary.csv",
        index=False,
    )
    policy_summary.to_csv(
        OUTPUT_DIR / "job_module_allocation_policy_summary.csv",
        index=False,
    )
    rejection_summary.to_csv(
        OUTPUT_DIR / "job_module_allocation_rejection_summary.csv",
        index=False,
    )

    knowledge = knowledge_views(trials)
    knowledge.to_csv(
        OUTPUT_DIR / "job_module_allocation_knowledge_views.csv",
        index=False,
    )

    plot_policy_metric(
        policy_summary,
        "unconditional_mean_excess_latency_ns",
        "Unconditional mean excess latency (ns)",
        "Leakage Across Job-to-Module Allocation Policies",
        "allocation_policy_leakage.png",
    )
    plot_policy_metric(
        policy_summary,
        "mean_job_rejection_rate",
        "Mean rejected-job fraction",
        "Admission Failure Across Allocation Policies",
        "policy_rejection_rate.png",
    )
    plot_policy_metric(
        policy_summary,
        "mean_unused_module_fragmentation",
        "Unused-module fragmentation",
        "Fragmentation Across Allocation Policies",
        "policy_fragmentation.png",
    )
    plot_sharing_metric(sharing_summary)

    display_columns = [
        "allocation_policy",
        "num_modules",
        "simultaneous_tenants",
        "victim_modules_requested",
        "hub_capacity",
        "interfaces_per_module",
        "probability_p1_like",
        "probability_p2_like",
        "unconditional_mean_excess_latency_ns",
        "unconditional_detection_probability",
        "mean_victim_slowdown_ratio",
        "mean_job_rejection_rate",
        "mean_unused_module_fragmentation",
    ]

    print("\n=== Allocation-policy summary ===")
    print(policy_summary[display_columns].to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
