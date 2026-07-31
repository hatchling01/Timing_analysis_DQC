#!/usr/bin/env python3
"""
phase1_04_remote_operation_schedulers.py

Experiment 1.4 — Remote-operation scheduler policies.

Research question
-----------------
How does the remote-operation scheduler change the observable timing
fingerprint in a modular multi-tenant quantum computer?

Implemented scheduler policies
------------------------------
1. static_circuit_layer
2. first_come_first_served
3. dependency_ready
4. shortest_operation_first
5. longest_waiting_first
6. round_robin_tenants
7. weighted_fair_queueing
8. priority_scheduling
9. coherence_deadline
10. link_aware
11. endpoint_aware
12. randomized_arbitration

Independent variables
---------------------
- remote-operation queue depth;
- number of simultaneous tenants;
- hub capacity;
- physical-link capacity;
- communication-qubit availability per module;
- priority-weight profile;
- scheduler decision interval;
- lookahead window;
- preemption disabled/enabled;
- EPR prefetch disabled/enabled.

Experimental control
--------------------
Within every workload/configuration/trial tuple, every scheduler receives the
same released operation trace, dependency graph, deadlines, placement, and
background traffic. The trace-generation seed deliberately excludes scheduler
policy. Only arbitration and resource-selection decisions change.

The victim uses a P2-like three-module placement (modules 0, 1, 2), while the
attacker uses modules 2 and 3. Background tenants use deterministic module
pairs. This preserves endpoint and fabric exposure while allowing scheduler
policy to be isolated.

Integrated experimental design
------------------------------
A. core_policy_capacity_tenancy
   All policies × tenant counts × selected hub/link capacity pairs.

B. queue_communication_capacity
   All policies × queue depths × communication-qubit counts.

C. decision_interval_lookahead
   All policies × decision intervals × lookahead windows.

D. preemption_prefetch_priority
   All policies × preemption × EPR prefetch × priority profiles.

Every requested policy and independent variable is exercised without using a
prohibitively large full Cartesian product.

Controls per trial
------------------
- attacker + identical background tenants;
- victim + identical background tenants;
- victim + attacker + identical background tenants.

Outputs
-------
blackbox_window_results/phase1_04_remote_operation_schedulers/
    remote_scheduler_trial_summary.csv
    remote_scheduler_attacker_comparison.csv.gz
    remote_scheduler_request_log.csv.gz
    remote_scheduler_policy_summary.csv
    remote_scheduler_capacity_summary.csv
    remote_scheduler_queue_summary.csv
    remote_scheduler_tenancy_summary.csv
    remote_scheduler_communication_qubit_summary.csv
    remote_scheduler_priority_summary.csv
    remote_scheduler_decision_interval_summary.csv
    remote_scheduler_lookahead_summary.csv
    remote_scheduler_preemption_summary.csv
    remote_scheduler_prefetch_summary.csv
    remote_scheduler_fingerprint_stability.csv
    remote_scheduler_phase_visibility_summary.csv
    remote_scheduler_nonml_classification_metrics.csv
    remote_scheduler_nonml_classification_predictions.csv
    remote_scheduler_random_forest_metrics.csv
    remote_scheduler_random_forest_predictions.csv
    remote_scheduler_feature_importance.csv

    core_policy_capacity_tenancy/
    queue_communication_capacity/
    decision_interval_lookahead/
    preemption_prefetch_priority/

Execution
---------
Keep this file beside phase1_01_job_module_allocation.py and run:

    python phase1_04_remote_operation_schedulers.py

No terminal options are required. All run controls are defined below.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import json
import math
import random
import statistics
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase1_01_job_module_allocation as p1


# =============================================================================
# Output and integrated run controls
# =============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_04_remote_operation_schedulers"
)

TRIALS_PER_CONFIGURATION = 3
RUN_QUICK_VALIDATION = False
MAX_CONFIGURATIONS: int | None = None

# Detailed logs are compressed because the full experiment can contain
# millions of request rows.
SAVE_ATTACKER_COMPARISON = True
SAVE_REQUEST_LOG = True
SAVE_SCHEDULER_DECISION_LOG = False

GLOBAL_SEED = 20260730


# =============================================================================
# Fixed architecture, placement, and workload settings
# =============================================================================

NUM_MODULES = 5
PHYSICAL_LINKS = tuple(
    (module, (module + 1) % NUM_MODULES)
    for module in range(NUM_MODULES)
)

VICTIM_MODULES = (0, 1, 2)
ATTACKER_MODULES = (2, 3)
BACKGROUND_MODULE_PAIRS = (
    (2, 4),
    (1, 4),
    (0, 4),
    (2, 1),
    (3, 0),
    (2, 4),
)

VICTIM_MODULES_REQUESTED = 3
ATTACKER_MODULES_REQUESTED = 2
BACKGROUND_MODULES_REQUESTED = 2

VICTIM_QASMS = list(p1.VICTIM_QASMS)

# Remote-operation service model.
HUB_SETUP_AND_TRANSFER_NS = 100
PER_LINK_HOP_LATENCY_NS = 10
PREFETCHED_BASE_SERVICE_NS = 50
PREFETCHED_PER_HOP_NS = 5

# Dependency and coherence model.
ROLE_DEADLINE_SLACK_NS = {
    "victim": 1_000,
    "attacker": 1_500,
    "background": 1_200,
}

# EPR-prefetch model.
EPR_GENERATION_NS = 40
EPR_COHERENCE_TIME_NS = 1_000

# Attacker timing interpretation.
TIMING_CHANGE_THRESHOLD_NS = 0.0
PHASE_BOUNDARY_TOLERANCE_NS = p1.PROBE_PERIOD_NS
NUM_TEMPORAL_BINS = 8


# =============================================================================
# Scheduler policies and independent variables
# =============================================================================

SCHEDULER_POLICIES = [
    "static_circuit_layer",
    "first_come_first_served",
    "dependency_ready",
    "shortest_operation_first",
    "longest_waiting_first",
    "round_robin_tenants",
    "weighted_fair_queueing",
    "priority_scheduling",
    "coherence_deadline",
    "link_aware",
    "endpoint_aware",
    "randomized_arbitration",
]

QUEUE_DEPTH_OPTIONS = [8, 32, 128]
TENANT_COUNT_OPTIONS = [2, 3, 4, 8]
HUB_LINK_CAPACITY_OPTIONS = [
    (1, 1),
    (2, 1),
    (4, 1),
    (4, 2),
]
COMMUNICATION_QUBIT_OPTIONS = [1, 2, 4, 8]
DECISION_INTERVAL_OPTIONS_NS = [0, 20, 100]
LOOKAHEAD_OPTIONS_NS = [0, 500, 2_000]
PREEMPTION_OPTIONS = [False, True]
EPR_PREFETCH_OPTIONS = [False, True]
PRIORITY_PROFILES = [
    "equal",
    "victim_high",
    "attacker_high",
]

PRIORITY_WEIGHTS = {
    "equal": {
        "victim": 1.0,
        "attacker": 1.0,
        "background": 1.0,
    },
    "victim_high": {
        "victim": 4.0,
        "attacker": 1.0,
        "background": 1.0,
    },
    "attacker_high": {
        "victim": 1.0,
        "attacker": 4.0,
        "background": 1.0,
    },
}

# Canonical values used outside the corresponding sub-sweep.
DEFAULT_QUEUE_DEPTH = 32
DEFAULT_TENANT_COUNT = 4
DEFAULT_HUB_CAPACITY = 2
DEFAULT_LINK_CAPACITY = 1
DEFAULT_COMMUNICATION_QUBITS = 2
DEFAULT_PRIORITY_PROFILE = "equal"
DEFAULT_DECISION_INTERVAL_NS = 20
DEFAULT_LOOKAHEAD_NS = 500
DEFAULT_PREEMPTION = False
DEFAULT_EPR_PREFETCH = False


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class SchedulerRequest:
    request_id: int
    operation_uid: str
    tenant_id: str
    role: str
    logical_event_id: int
    circuit_order: int
    release_time_ns: int
    source_module: int
    target_module: int
    route_links: tuple[tuple[int, int], ...]
    base_service_time_ns: int
    dependencies: tuple[int, ...]
    layer: int
    deadline_ns: int
    priority_weight: float
    criticality: int = 0

    @property
    def endpoints(self) -> tuple[int, int]:
        return (
            min(self.source_module, self.target_module),
            max(self.source_module, self.target_module),
        )


@dataclass
class RuntimeRequest:
    request: SchedulerRequest
    status: str = "unreleased"
    admitted_time_ns: int | None = None
    first_selected_time_ns: int | None = None
    selection_count: int = 0
    service_start_time_ns: int | None = None
    completion_time_ns: int | None = None
    remaining_service_ns: int | None = None
    current_generation: int = 0
    preemption_count: int = 0
    used_prefetched_epr: bool = False
    rejection_reason: str = ""
    wfq_finish_tag: float = 0.0
    resource_acquisition_failures: int = 0

    @property
    def release_time_ns(self) -> int:
        return self.request.release_time_ns


@dataclass
class ActiveExecution:
    request_id: int
    generation: int
    start_time_ns: int
    completion_time_ns: int
    allocated_service_ns: int
    used_prefetched_epr: bool


@dataclass
class ActivePrefetch:
    request_id: int
    generation: int
    start_time_ns: int
    completion_time_ns: int


@dataclass
class EPRToken:
    request_id: int
    ready_time_ns: int
    expiry_time_ns: int
    source_module: int
    target_module: int
    used: bool = False


@dataclass(frozen=True)
class SchedulerConfiguration:
    subexperiment: str
    scheduler_policy: str
    queue_depth: int
    tenant_count: int
    hub_capacity: int
    link_capacity: int
    communication_qubits_per_module: int
    priority_profile: str
    decision_interval_ns: int
    lookahead_window_ns: int
    preemption_allowed: bool
    epr_prefetch_enabled: bool

    @property
    def configuration_id(self) -> str:
        payload = (
            self.subexperiment,
            self.scheduler_policy,
            self.queue_depth,
            self.tenant_count,
            self.hub_capacity,
            self.link_capacity,
            self.communication_qubits_per_module,
            self.priority_profile,
            self.decision_interval_ns,
            self.lookahead_window_ns,
            int(self.preemption_allowed),
            int(self.epr_prefetch_enabled),
        )
        text = "|".join(map(str, payload))
        return f"cfg_{zlib.crc32(text.encode('utf-8')):08x}"


# =============================================================================
# General helpers
# =============================================================================


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return (
        GLOBAL_SEED
        + zlib.crc32(text.encode("utf-8"))
    ) & 0x7FFFFFFF


def safe_tag(value: str) -> str:
    return p1.safe_tag(value)


def canonical_link(first: int, second: int) -> tuple[int, int]:
    return (min(first, second), max(first, second))


def clockwise_path(
    source: int,
    target: int,
    num_modules: int = NUM_MODULES,
) -> list[int]:
    nodes = [source]
    current = source
    while current != target:
        current = (current + 1) % num_modules
        nodes.append(current)
    return nodes


def counterclockwise_path(
    source: int,
    target: int,
    num_modules: int = NUM_MODULES,
) -> list[int]:
    nodes = [source]
    current = source
    while current != target:
        current = (current - 1) % num_modules
        nodes.append(current)
    return nodes


def route_links(
    source: int,
    target: int,
    num_modules: int = NUM_MODULES,
) -> tuple[tuple[int, int], ...]:
    """Return the deterministic shortest ring route."""

    clockwise = clockwise_path(source, target, num_modules)
    counterclockwise = counterclockwise_path(source, target, num_modules)

    if len(clockwise) <= len(counterclockwise):
        nodes = clockwise
    else:
        nodes = counterclockwise

    return tuple(
        canonical_link(nodes[index], nodes[index + 1])
        for index in range(len(nodes) - 1)
    )


def base_service_time_ns(route: Sequence[tuple[int, int]]) -> int:
    return (
        HUB_SETUP_AND_TRANSFER_NS
        + PER_LINK_HOP_LATENCY_NS * len(route)
    )


def prefetched_service_time_ns(route: Sequence[tuple[int, int]]) -> int:
    return (
        PREFETCHED_BASE_SERVICE_NS
        + PREFETCHED_PER_HOP_NS * len(route)
    )


def jain_fairness(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        return 1.0
    denominator = len(values_array) * float(np.square(values_array).sum())
    if denominator <= 0:
        return 1.0
    return float(np.square(values_array.sum()) / denominator)


def inversion_fraction(order_values: Sequence[int]) -> float:
    values = list(order_values)
    count = len(values)
    if count < 2:
        return 0.0

    inversions = 0
    total_pairs = count * (count - 1) // 2
    for first in range(count):
        for second in range(first + 1, count):
            if values[first] > values[second]:
                inversions += 1
    return inversions / total_pairs


def pearson_or_zero(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second) or len(first) == 0:
        return 0.0
    if np.std(first) == 0 or np.std(second) == 0:
        return 1.0 if np.array_equal(first, second) else 0.0
    value = np.corrcoef(first, second)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


# =============================================================================
# Fixed placements and released-operation trace generation
# =============================================================================


def fixed_allocations(
    victim_job: p1.JobSpec,
    attacker_job: p1.JobSpec,
    background_jobs: Sequence[p1.JobSpec],
) -> dict[str, p1.Allocation]:
    """Create deterministic P2-like placements used by every scheduler."""

    allocations: dict[str, p1.Allocation] = {}

    victim_mapping = {
        partition: VICTIM_MODULES[
            partition % len(VICTIM_MODULES)
        ]
        for partition in victim_job.partitions
    }
    allocations[victim_job.tenant_id] = p1.Allocation(
        tenant_id=victim_job.tenant_id,
        role=victim_job.role,
        accepted=True,
        requested_modules=victim_job.modules_requested,
        partition_to_module=victim_mapping,
    )

    attacker_mapping = {
        0: ATTACKER_MODULES[0],
        1: ATTACKER_MODULES[1],
    }
    allocations[attacker_job.tenant_id] = p1.Allocation(
        tenant_id=attacker_job.tenant_id,
        role=attacker_job.role,
        accepted=True,
        requested_modules=attacker_job.modules_requested,
        partition_to_module=attacker_mapping,
    )

    for index, job in enumerate(background_jobs):
        pair = BACKGROUND_MODULE_PAIRS[index % len(BACKGROUND_MODULE_PAIRS)]
        mapping = {
            partition: pair[partition % 2]
            for partition in job.partitions
        }
        allocations[job.tenant_id] = p1.Allocation(
            tenant_id=job.tenant_id,
            role=job.role,
            accepted=True,
            requested_modules=job.modules_requested,
            partition_to_module=mapping,
        )

    return allocations


def build_jobs_for_trial(
    qasm_path: Path,
    tenant_count: int,
    trace_seed: int,
) -> tuple[list[p1.JobSpec], dict[str, p1.Allocation]]:
    """Build a released trace shared by all scheduler policies."""

    victim_job = copy.deepcopy(
        p1.build_victim_job(
            qasm_path,
            VICTIM_MODULES_REQUESTED,
        )
    )
    attacker_job = copy.deepcopy(p1.build_attacker_job())

    background_jobs: list[p1.JobSpec] = []
    background_count = max(0, tenant_count - 2)
    for background_index in range(background_count):
        rng = random.Random(
            stable_seed(trace_seed, "background", background_index)
        )
        background_jobs.append(
            p1.build_background_job(
                background_index,
                BACKGROUND_MODULES_REQUESTED,
                rng,
            )
        )

    jobs = [victim_job, attacker_job, *background_jobs]
    allocations = fixed_allocations(
        victim_job,
        attacker_job,
        background_jobs,
    )
    return jobs, allocations


def touched_partitions_for_event(
    job: p1.JobSpec,
    event: p1.LogicalEvent,
) -> list[int]:
    return sorted(
        {
            job.partition_of_qubit[qubit]
            for qubit in event.qubits
            if qubit in job.partition_of_qubit
        }
    )


def build_scheduler_requests(
    jobs: Sequence[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    priority_profile: str,
) -> list[SchedulerRequest]:
    """
    Convert the common released trace into remote requests with dependencies.

    Dependencies are conservative: a remote operation depends on the previous
    remote operation touching either of its logical qubits. This captures the
    remote dependency frontier while keeping local gates represented by their
    original release times.
    """

    weights = PRIORITY_WEIGHTS[priority_profile]
    provisional: list[SchedulerRequest] = []
    request_id = 0

    for job in jobs:
        allocation = allocations[job.tenant_id]
        last_remote_by_qubit: dict[int, int] = {}
        layer_by_request: dict[int, int] = {}
        tenant_request_ids: list[int] = []

        for event in job.logical_events:
            touched_partitions = touched_partitions_for_event(job, event)
            if len(touched_partitions) < 2:
                continue

            pair_index = 0
            for left_partition, right_partition in itertools.combinations(
                touched_partitions,
                2,
            ):
                source = allocation.partition_to_module[left_partition]
                target = allocation.partition_to_module[right_partition]
                if source == target:
                    continue

                logical_qubits = tuple(sorted(set(event.qubits)))
                dependencies = sorted(
                    {
                        last_remote_by_qubit[qubit]
                        for qubit in logical_qubits
                        if qubit in last_remote_by_qubit
                    }
                )
                layer = (
                    0
                    if not dependencies
                    else 1 + max(layer_by_request[dependency] for dependency in dependencies)
                )

                route = route_links(source, target)
                release_time_ns = job.start_time_ns + event.release_offset_ns
                deadline_ns = (
                    release_time_ns
                    + ROLE_DEADLINE_SLACK_NS[job.role]
                )

                request = SchedulerRequest(
                    request_id=request_id,
                    operation_uid=(
                        f"{job.tenant_id}:"
                        f"{event.event_id}:"
                        f"{pair_index}"
                    ),
                    tenant_id=job.tenant_id,
                    role=job.role,
                    logical_event_id=event.event_id,
                    circuit_order=len(tenant_request_ids),
                    release_time_ns=release_time_ns,
                    source_module=source,
                    target_module=target,
                    route_links=route,
                    base_service_time_ns=base_service_time_ns(route),
                    dependencies=tuple(dependencies),
                    layer=layer,
                    deadline_ns=deadline_ns,
                    priority_weight=weights[job.role],
                    criticality=0,
                )
                provisional.append(request)
                tenant_request_ids.append(request_id)
                layer_by_request[request_id] = layer

                for qubit in logical_qubits:
                    last_remote_by_qubit[qubit] = request_id

                request_id += 1
                pair_index += 1

    # Compute a downstream criticality count for dependency-ready scheduling.
    successors: dict[int, list[int]] = defaultdict(list)
    for request in provisional:
        for dependency in request.dependencies:
            successors[dependency].append(request.request_id)

    criticality_cache: dict[int, int] = {}

    def criticality(request_id_value: int) -> int:
        if request_id_value in criticality_cache:
            return criticality_cache[request_id_value]
        value = len(successors.get(request_id_value, []))
        value += sum(criticality(child) for child in successors.get(request_id_value, []))
        criticality_cache[request_id_value] = value
        return value

    completed_requests = [
        SchedulerRequest(
            **{
                **request.__dict__,
                "criticality": criticality(request.request_id),
            }
        )
        for request in provisional
    ]

    role_order = {
        "victim": 0,
        "background": 1,
        "attacker": 2,
    }
    return sorted(
        completed_requests,
        key=lambda request: (
            request.release_time_ns,
            role_order[request.role],
            request.tenant_id,
            request.circuit_order,
            request.request_id,
        ),
    )


# =============================================================================
# Remote-operation scheduler simulator
# =============================================================================


class RemoteOperationSchedulerSimulator:
    """Discrete-event remote scheduler with hub/link/endpoint resources."""

    def __init__(
        self,
        requests: Sequence[SchedulerRequest],
        configuration: SchedulerConfiguration,
        seed: int,
    ) -> None:
        self.configuration = configuration
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.current_time_ns = 0
        self.last_decision_time_ns: int | None = None

        self.requests = list(requests)
        self.request_by_id = {
            request.request_id: request
            for request in self.requests
        }
        self.runtime = {
            request.request_id: RuntimeRequest(
                request=request,
                remaining_service_ns=request.base_service_time_ns,
            )
            for request in self.requests
        }

        self.release_order = sorted(
            self.requests,
            key=lambda request: (
                request.release_time_ns,
                request.request_id,
            ),
        )
        self.release_index = 0
        self.waiting: list[int] = []

        self.active_execution: dict[int, ActiveExecution] = {}
        self.active_prefetch: dict[int, ActivePrefetch] = {}
        self.event_heap: list[tuple[int, int, str, int, int]] = []
        self.event_serial = 0

        self.active_hub_slots = 0
        self.active_link_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.active_endpoint_counts: dict[int, int] = defaultdict(int)
        self.epr_reserved_endpoint_counts: dict[int, int] = defaultdict(int)

        self.epr_tokens: dict[int, EPRToken] = {}
        self.prefetch_scheduled: set[int] = set()

        self.completed_ids: set[int] = set()
        self.rejected_ids: set[int] = set()

        self.tenant_order = sorted(
            {request.tenant_id for request in self.requests}
        )
        self.round_robin_cursor = 0

        self.wfq_last_finish: dict[str, float] = defaultdict(float)
        self.wfq_virtual_time = 0.0

        self.scheduler_decisions = 0
        self.scheduler_idle_decisions = 0
        self.preemption_events = 0
        self.cancelled_prefetches = 0
        self.epr_prefetched = 0
        self.epr_used = 0
        self.epr_expired = 0
        self.epr_cancelled_or_orphaned = 0

        self.maximum_queue_occupancy = 0
        self.queue_area_ns = 0.0
        self.last_queue_area_update_ns = 0

        self.hub_busy_slot_ns = 0.0
        self.link_busy_capacity_ns = 0.0
        self.endpoint_busy_capacity_ns = 0.0
        self.epr_reserved_capacity_ns = 0.0
        self.last_resource_area_update_ns = 0

        self.decision_rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Time and resource accounting
    # -------------------------------------------------------------------------

    def _advance_accounting(self, new_time_ns: int) -> None:
        if new_time_ns < self.current_time_ns:
            raise RuntimeError("Scheduler time cannot move backward.")

        delta_ns = new_time_ns - self.current_time_ns
        if delta_ns <= 0:
            self.current_time_ns = new_time_ns
            return

        self.queue_area_ns += len(self.waiting) * delta_ns
        self.hub_busy_slot_ns += self.active_hub_slots * delta_ns
        self.link_busy_capacity_ns += sum(self.active_link_counts.values()) * delta_ns
        self.endpoint_busy_capacity_ns += sum(self.active_endpoint_counts.values()) * delta_ns
        self.epr_reserved_capacity_ns += sum(
            self.epr_reserved_endpoint_counts.values()
        ) * delta_ns

        self.current_time_ns = new_time_ns

    def _push_event(
        self,
        completion_time_ns: int,
        event_kind: str,
        request_id: int,
        generation: int,
    ) -> None:
        heapq.heappush(
            self.event_heap,
            (
                completion_time_ns,
                self.event_serial,
                event_kind,
                request_id,
                generation,
            ),
        )
        self.event_serial += 1

    # -------------------------------------------------------------------------
    # Queue admission and dependency state
    # -------------------------------------------------------------------------

    def _admit_releases(self) -> None:
        while (
            self.release_index < len(self.release_order)
            and self.release_order[self.release_index].release_time_ns
            <= self.current_time_ns
        ):
            request = self.release_order[self.release_index]
            runtime = self.runtime[request.request_id]
            self.release_index += 1

            if len(self.waiting) >= self.configuration.queue_depth:
                runtime.status = "rejected"
                runtime.rejection_reason = "queue_depth_exceeded"
                self.rejected_ids.add(request.request_id)
                continue

            runtime.status = "waiting"
            runtime.admitted_time_ns = self.current_time_ns

            service_for_wfq = request.base_service_time_ns
            start_tag = max(
                self.wfq_virtual_time,
                self.wfq_last_finish[request.tenant_id],
            )
            runtime.wfq_finish_tag = (
                start_tag
                + service_for_wfq / max(request.priority_weight, 1e-9)
            )
            self.wfq_last_finish[request.tenant_id] = runtime.wfq_finish_tag

            self.waiting.append(request.request_id)
            self.maximum_queue_occupancy = max(
                self.maximum_queue_occupancy,
                len(self.waiting),
            )

    def _propagate_dependency_failures(self) -> None:
        progress = True
        while progress:
            progress = False
            for request_id in list(self.waiting):
                request = self.request_by_id[request_id]
                if any(
                    dependency in self.rejected_ids
                    for dependency in request.dependencies
                ):
                    runtime = self.runtime[request_id]
                    runtime.status = "rejected"
                    runtime.rejection_reason = "dependency_failed"
                    self.waiting.remove(request_id)
                    self.rejected_ids.add(request_id)
                    progress = True

    def _dependencies_completed(self, request: SchedulerRequest) -> bool:
        return all(
            dependency in self.completed_ids
            for dependency in request.dependencies
        )

    def _current_static_layer(self, tenant_id: str) -> int | None:
        layers = []
        for runtime in self.runtime.values():
            request = runtime.request
            if request.tenant_id != tenant_id:
                continue
            if runtime.status not in {"completed", "rejected"}:
                layers.append(request.layer)
        return min(layers) if layers else None

    def _ready_request_ids(self) -> list[int]:
        ready = []
        for request_id in self.waiting:
            request = self.request_by_id[request_id]
            if not self._dependencies_completed(request):
                continue

            if self.configuration.scheduler_policy == "static_circuit_layer":
                current_layer = self._current_static_layer(request.tenant_id)
                if current_layer is None or request.layer != current_layer:
                    continue

            ready.append(request_id)
        return ready

    # -------------------------------------------------------------------------
    # Resource feasibility and acquisition
    # -------------------------------------------------------------------------

    def _valid_epr_token(self, request_id: int) -> EPRToken | None:
        token = self.epr_tokens.get(request_id)
        if token is None or token.used:
            return None
        if token.expiry_time_ns <= self.current_time_ns:
            return None
        return token

    def _expire_epr_tokens(self) -> None:
        for request_id, token in list(self.epr_tokens.items()):
            if token.used:
                continue
            if token.expiry_time_ns <= self.current_time_ns:
                for module in (token.source_module, token.target_module):
                    self.epr_reserved_endpoint_counts[module] -= 1
                token.used = True
                self.epr_expired += 1

    def _can_acquire_execution(self, request: SchedulerRequest) -> bool:
        if self.active_hub_slots >= self.configuration.hub_capacity:
            return False

        for link in request.route_links:
            if self.active_link_counts[link] >= self.configuration.link_capacity:
                return False

        token = self._valid_epr_token(request.request_id)
        if token is not None:
            # The token already reserves one communication qubit at each endpoint.
            return True

        for module in request.endpoints:
            occupied = (
                self.active_endpoint_counts[module]
                + self.epr_reserved_endpoint_counts[module]
            )
            if occupied >= self.configuration.communication_qubits_per_module:
                return False

        return True

    def _can_acquire_prefetch(self, request: SchedulerRequest) -> bool:
        if self.active_hub_slots >= self.configuration.hub_capacity:
            return False
        for link in request.route_links:
            if self.active_link_counts[link] >= self.configuration.link_capacity:
                return False
        for module in request.endpoints:
            occupied = (
                self.active_endpoint_counts[module]
                + self.epr_reserved_endpoint_counts[module]
            )
            if occupied >= self.configuration.communication_qubits_per_module:
                return False
        return True

    def _acquire_resources(
        self,
        request: SchedulerRequest,
        *,
        endpoint_already_reserved: bool,
    ) -> None:
        self.active_hub_slots += 1
        for link in request.route_links:
            self.active_link_counts[link] += 1
        if endpoint_already_reserved:
            for module in request.endpoints:
                self.epr_reserved_endpoint_counts[module] -= 1
                self.active_endpoint_counts[module] += 1
        else:
            for module in request.endpoints:
                self.active_endpoint_counts[module] += 1

    def _release_execution_resources(self, request: SchedulerRequest) -> None:
        self.active_hub_slots -= 1
        for link in request.route_links:
            self.active_link_counts[link] -= 1
        for module in request.endpoints:
            self.active_endpoint_counts[module] -= 1

    def _acquire_prefetch_resources(self, request: SchedulerRequest) -> None:
        self.active_hub_slots += 1
        for link in request.route_links:
            self.active_link_counts[link] += 1
        for module in request.endpoints:
            self.active_endpoint_counts[module] += 1

    def _release_prefetch_generation_resources(
        self,
        request: SchedulerRequest,
        *,
        convert_to_reservation: bool,
    ) -> None:
        self.active_hub_slots -= 1
        for link in request.route_links:
            self.active_link_counts[link] -= 1
        for module in request.endpoints:
            self.active_endpoint_counts[module] -= 1
            if convert_to_reservation:
                self.epr_reserved_endpoint_counts[module] += 1

    # -------------------------------------------------------------------------
    # Scheduler policy ranking
    # -------------------------------------------------------------------------

    def _lookahead_link_pressure(self, request: SchedulerRequest) -> int:
        horizon = self.current_time_ns + self.configuration.lookahead_window_ns
        pressure = 0
        for candidate in self.requests:
            runtime = self.runtime[candidate.request_id]
            if runtime.status in {"completed", "rejected"}:
                continue
            if candidate.release_time_ns > horizon:
                continue
            if set(candidate.route_links).intersection(request.route_links):
                pressure += 1
        pressure += sum(self.active_link_counts[link] for link in request.route_links)
        return pressure

    def _lookahead_endpoint_pressure(self, request: SchedulerRequest) -> int:
        horizon = self.current_time_ns + self.configuration.lookahead_window_ns
        endpoints = set(request.endpoints)
        pressure = 0
        for candidate in self.requests:
            runtime = self.runtime[candidate.request_id]
            if runtime.status in {"completed", "rejected"}:
                continue
            if candidate.release_time_ns > horizon:
                continue
            if endpoints.intersection(candidate.endpoints):
                pressure += 1
        pressure += sum(
            self.active_endpoint_counts[module]
            + self.epr_reserved_endpoint_counts[module]
            for module in endpoints
        )
        return pressure

    def _rank_key(self, request_id: int) -> tuple[Any, ...]:
        request = self.request_by_id[request_id]
        runtime = self.runtime[request_id]
        policy = self.configuration.scheduler_policy

        if policy == "static_circuit_layer":
            return (
                request.layer,
                request.release_time_ns,
                request.tenant_id,
                request.circuit_order,
            )
        if policy == "first_come_first_served":
            return (
                request.release_time_ns,
                request.request_id,
            )
        if policy == "dependency_ready":
            return (
                -request.criticality,
                request.layer,
                request.release_time_ns,
                request.request_id,
            )
        if policy == "shortest_operation_first":
            return (
                runtime.remaining_service_ns,
                request.release_time_ns,
                request.request_id,
            )
        if policy == "longest_waiting_first":
            admitted = runtime.admitted_time_ns or request.release_time_ns
            return (
                -(self.current_time_ns - admitted),
                request.release_time_ns,
                request.request_id,
            )
        if policy == "weighted_fair_queueing":
            return (
                runtime.wfq_finish_tag,
                request.release_time_ns,
                request.request_id,
            )
        if policy == "priority_scheduling":
            return (
                -request.priority_weight,
                request.release_time_ns,
                request.request_id,
            )
        if policy == "coherence_deadline":
            return (
                request.deadline_ns,
                request.release_time_ns,
                request.request_id,
            )
        if policy == "link_aware":
            return (
                self._lookahead_link_pressure(request),
                len(request.route_links),
                request.release_time_ns,
                request.request_id,
            )
        if policy == "endpoint_aware":
            return (
                self._lookahead_endpoint_pressure(request),
                request.release_time_ns,
                request.request_id,
            )
        if policy == "randomized_arbitration":
            # Request-specific deterministic arbitration value. The same seed
            # is reused across counterfactual scenarios, so attacker/background
            # requests receive coupled random priorities while victim requests
            # introduce only genuine additional competition.
            random_rank = stable_seed(
                self.seed,
                self.current_time_ns,
                request.operation_uid,
                "randomized_arbitration",
            ) / float(0x7FFFFFFF)
            return (
                random_rank,
                request.request_id,
            )

        # round_robin_tenants is selected separately.
        return (
            request.release_time_ns,
            request.request_id,
        )

    def _select_round_robin(self, candidates: Sequence[int]) -> int | None:
        if not candidates:
            return None
        candidates_by_tenant: dict[str, list[int]] = defaultdict(list)
        for request_id in candidates:
            candidates_by_tenant[self.request_by_id[request_id].tenant_id].append(request_id)
        for tenant_requests in candidates_by_tenant.values():
            tenant_requests.sort(key=self._rank_key)

        for offset in range(len(self.tenant_order)):
            index = (self.round_robin_cursor + offset) % len(self.tenant_order)
            tenant_id = self.tenant_order[index]
            if candidates_by_tenant.get(tenant_id):
                self.round_robin_cursor = (index + 1) % len(self.tenant_order)
                return candidates_by_tenant[tenant_id][0]
        return None

    def _select_candidate(self, candidates: Sequence[int]) -> int | None:
        if not candidates:
            return None
        if self.configuration.scheduler_policy == "round_robin_tenants":
            return self._select_round_robin(candidates)
        return min(candidates, key=self._rank_key)

    # -------------------------------------------------------------------------
    # Execution, completion, preemption, and prefetch
    # -------------------------------------------------------------------------

    def _start_execution(self, request_id: int) -> None:
        request = self.request_by_id[request_id]
        runtime = self.runtime[request_id]

        token = self._valid_epr_token(request_id)
        uses_token = token is not None
        self._acquire_resources(
            request,
            endpoint_already_reserved=uses_token,
        )

        if request_id in self.waiting:
            self.waiting.remove(request_id)

        if runtime.first_selected_time_ns is None:
            runtime.first_selected_time_ns = self.current_time_ns
        runtime.selection_count += 1
        runtime.status = "active"
        if runtime.service_start_time_ns is None:
            runtime.service_start_time_ns = self.current_time_ns

        if uses_token:
            assert token is not None
            token.used = True
            runtime.used_prefetched_epr = True
            self.epr_used += 1
            service_time = min(
                runtime.remaining_service_ns or request.base_service_time_ns,
                prefetched_service_time_ns(request.route_links),
            )
        else:
            service_time = runtime.remaining_service_ns or request.base_service_time_ns

        runtime.remaining_service_ns = int(service_time)
        runtime.current_generation += 1
        completion_time_ns = self.current_time_ns + int(service_time)
        active = ActiveExecution(
            request_id=request_id,
            generation=runtime.current_generation,
            start_time_ns=self.current_time_ns,
            completion_time_ns=completion_time_ns,
            allocated_service_ns=int(service_time),
            used_prefetched_epr=uses_token,
        )
        self.active_execution[request_id] = active
        self._push_event(
            completion_time_ns,
            "execution",
            request_id,
            runtime.current_generation,
        )

        if self.configuration.scheduler_policy == "weighted_fair_queueing":
            self.wfq_virtual_time = max(
                self.wfq_virtual_time,
                runtime.wfq_finish_tag,
            )

    def _complete_execution(self, active: ActiveExecution) -> None:
        runtime = self.runtime[active.request_id]
        if runtime.current_generation != active.generation:
            return
        if active.request_id not in self.active_execution:
            return

        request = runtime.request
        self._release_execution_resources(request)
        del self.active_execution[active.request_id]

        runtime.status = "completed"
        runtime.completion_time_ns = self.current_time_ns
        runtime.remaining_service_ns = 0
        self.completed_ids.add(active.request_id)

    def _start_prefetch(self, request_id: int) -> None:
        request = self.request_by_id[request_id]
        self._acquire_prefetch_resources(request)
        generation = 1
        completion = self.current_time_ns + EPR_GENERATION_NS
        active = ActivePrefetch(
            request_id=request_id,
            generation=generation,
            start_time_ns=self.current_time_ns,
            completion_time_ns=completion,
        )
        self.active_prefetch[request_id] = active
        self.prefetch_scheduled.add(request_id)
        self.epr_prefetched += 1
        self._push_event(
            completion,
            "prefetch",
            request_id,
            generation,
        )

    def _complete_prefetch(self, active: ActivePrefetch) -> None:
        if active.request_id not in self.active_prefetch:
            return
        request = self.request_by_id[active.request_id]
        self._release_prefetch_generation_resources(
            request,
            convert_to_reservation=True,
        )
        del self.active_prefetch[active.request_id]
        self.epr_tokens[active.request_id] = EPRToken(
            request_id=active.request_id,
            ready_time_ns=self.current_time_ns,
            expiry_time_ns=self.current_time_ns + EPR_COHERENCE_TIME_NS,
            source_module=request.source_module,
            target_module=request.target_module,
        )

    def _cancel_prefetch(self, request_id: int) -> None:
        active = self.active_prefetch.get(request_id)
        if active is None:
            return
        request = self.request_by_id[request_id]
        self._release_prefetch_generation_resources(
            request,
            convert_to_reservation=False,
        )
        del self.active_prefetch[request_id]
        self.cancelled_prefetches += 1
        self.epr_cancelled_or_orphaned += 1

    def _preempt_execution(self, request_id: int) -> None:
        active = self.active_execution.get(request_id)
        if active is None:
            return
        runtime = self.runtime[request_id]
        request = runtime.request

        remaining = max(1, active.completion_time_ns - self.current_time_ns)
        runtime.remaining_service_ns = remaining
        runtime.current_generation += 1
        runtime.preemption_count += 1
        runtime.status = "waiting"
        self.preemption_events += 1

        self._release_execution_resources(request)
        del self.active_execution[request_id]
        self.waiting.append(request_id)

    def _conflicts_with_request(
        self,
        active_request: SchedulerRequest,
        waiting_request: SchedulerRequest,
    ) -> bool:
        if set(active_request.route_links).intersection(waiting_request.route_links):
            return True
        if set(active_request.endpoints).intersection(waiting_request.endpoints):
            return True
        return self.active_hub_slots >= self.configuration.hub_capacity

    def _maybe_preempt(self, ready_ids: Sequence[int]) -> bool:
        if not self.configuration.preemption_allowed:
            return False
        if not ready_ids or not self.active_execution:
            return False

        best_waiting_id = self._select_candidate(ready_ids)
        if best_waiting_id is None:
            return False
        if self._can_acquire_execution(self.request_by_id[best_waiting_id]):
            return False

        waiting_request = self.request_by_id[best_waiting_id]
        conflicting_active = [
            request_id
            for request_id in self.active_execution
            if self._conflicts_with_request(
                self.request_by_id[request_id],
                waiting_request,
            )
        ]
        if not conflicting_active:
            return False

        worst_active_id = max(conflicting_active, key=self._rank_key)
        if self._rank_key(best_waiting_id) < self._rank_key(worst_active_id):
            self._preempt_execution(worst_active_id)
            return True
        return False

    def _prefetch_candidates(self) -> list[int]:
        if not self.configuration.epr_prefetch_enabled:
            return []
        horizon = self.current_time_ns + self.configuration.lookahead_window_ns
        candidates = []
        for request in self.requests:
            runtime = self.runtime[request.request_id]
            if runtime.status in {"completed", "rejected", "active"}:
                continue
            if request.request_id in self.prefetch_scheduled:
                continue
            if self._valid_epr_token(request.request_id) is not None:
                continue
            if request.release_time_ns > horizon:
                continue
            if any(dependency in self.rejected_ids for dependency in request.dependencies):
                continue
            candidates.append(request.request_id)
        return sorted(
            candidates,
            key=lambda request_id: (
                self.request_by_id[request_id].release_time_ns,
                self.request_by_id[request_id].deadline_ns,
                request_id,
            ),
        )

    def _fill_prefetch_capacity(self) -> None:
        if not self.configuration.epr_prefetch_enabled:
            return
        for request_id in self._prefetch_candidates():
            request = self.request_by_id[request_id]
            if self._can_acquire_prefetch(request):
                self._start_prefetch(request_id)
            if self.active_hub_slots >= self.configuration.hub_capacity:
                break

    # -------------------------------------------------------------------------
    # Decision loop
    # -------------------------------------------------------------------------

    def _strict_fcfs_blocking(self) -> bool:
        return self.configuration.scheduler_policy in {
            "first_come_first_served",
            "static_circuit_layer",
        }

    def _run_scheduler_decision(self) -> None:
        self.scheduler_decisions += 1
        started_count = 0
        selected_ids: list[int] = []

        self._propagate_dependency_failures()
        ready_ids = self._ready_request_ids()

        # Ready operations always take precedence over speculative prefetch.
        if ready_ids and self.active_prefetch:
            for request_id in list(self.active_prefetch):
                self._cancel_prefetch(request_id)
                ready_ids = self._ready_request_ids()
                if any(
                    self._can_acquire_execution(self.request_by_id[candidate])
                    for candidate in ready_ids
                ):
                    break

        self._maybe_preempt(ready_ids)

        considered = list(ready_ids)
        while considered:
            selected_id = self._select_candidate(considered)
            if selected_id is None:
                break
            selected_ids.append(selected_id)
            runtime = self.runtime[selected_id]
            if runtime.first_selected_time_ns is None:
                runtime.first_selected_time_ns = self.current_time_ns
            runtime.selection_count += 1

            request = self.request_by_id[selected_id]
            if self._can_acquire_execution(request):
                # _start_execution records the successful selection separately;
                # remove this provisional count to avoid double-counting.
                runtime.selection_count -= 1
                self._start_execution(selected_id)
                started_count += 1
                considered = [
                    candidate
                    for candidate in self._ready_request_ids()
                    if candidate not in selected_ids
                ]
            else:
                runtime.resource_acquisition_failures += 1
                if self._strict_fcfs_blocking():
                    break
                considered.remove(selected_id)

        self._fill_prefetch_capacity()

        if started_count == 0:
            self.scheduler_idle_decisions += 1

        if SAVE_SCHEDULER_DECISION_LOG:
            self.decision_rows.append(
                {
                    "decision_time_ns": self.current_time_ns,
                    "scheduler_policy": self.configuration.scheduler_policy,
                    "ready_count": len(ready_ids),
                    "selected_request_ids": ",".join(map(str, selected_ids)),
                    "started_count": started_count,
                    "active_execution_count": len(self.active_execution),
                    "active_prefetch_count": len(self.active_prefetch),
                    "queue_occupancy": len(self.waiting),
                }
            )

        self.last_decision_time_ns = self.current_time_ns

    # -------------------------------------------------------------------------
    # Event progression
    # -------------------------------------------------------------------------

    def _process_completion_events(self) -> None:
        while self.event_heap and self.event_heap[0][0] <= self.current_time_ns:
            _, _, event_kind, request_id, generation = heapq.heappop(self.event_heap)
            if event_kind == "execution":
                active = self.active_execution.get(request_id)
                if active is not None and active.generation == generation:
                    self._complete_execution(active)
            elif event_kind == "prefetch":
                active_prefetch = self.active_prefetch.get(request_id)
                if active_prefetch is not None and active_prefetch.generation == generation:
                    self._complete_prefetch(active_prefetch)

    def _next_release_time(self) -> int | float:
        if self.release_index >= len(self.release_order):
            return math.inf
        return self.release_order[self.release_index].release_time_ns

    def _next_completion_time(self) -> int | float:
        while self.event_heap:
            time_ns, _, event_kind, request_id, generation = self.event_heap[0]
            if event_kind == "execution":
                active = self.active_execution.get(request_id)
                if active is None or active.generation != generation:
                    heapq.heappop(self.event_heap)
                    continue
            elif event_kind == "prefetch":
                active_prefetch = self.active_prefetch.get(request_id)
                if active_prefetch is None or active_prefetch.generation != generation:
                    heapq.heappop(self.event_heap)
                    continue
            return time_ns
        return math.inf

    def _next_token_expiry(self) -> int | float:
        expiries = [
            token.expiry_time_ns
            for token in self.epr_tokens.values()
            if not token.used and token.expiry_time_ns > self.current_time_ns
        ]
        return min(expiries) if expiries else math.inf

    def _next_prefetch_opportunity(self) -> int | float:
        if not self.configuration.epr_prefetch_enabled:
            return math.inf
        if self.configuration.lookahead_window_ns <= 0:
            return math.inf
        opportunities = []
        for request in self.requests:
            runtime = self.runtime[request.request_id]
            if runtime.status in {"completed", "rejected", "active"}:
                continue
            if request.request_id in self.prefetch_scheduled:
                continue
            opportunity = max(
                0,
                request.release_time_ns - self.configuration.lookahead_window_ns,
            )
            if opportunity > self.current_time_ns:
                opportunities.append(opportunity)
        return min(opportunities) if opportunities else math.inf

    def _decision_is_due(self) -> bool:
        interval = self.configuration.decision_interval_ns
        if interval == 0:
            return True
        if self.current_time_ns % interval != 0:
            return False
        return self.last_decision_time_ns != self.current_time_ns

    def _next_decision_time(self) -> int | float:
        interval = self.configuration.decision_interval_ns
        has_reason = bool(self.waiting) or (
            self.configuration.epr_prefetch_enabled
            and self._next_prefetch_opportunity() <= self.current_time_ns
        )
        if not has_reason:
            return math.inf
        if interval == 0:
            if self.last_decision_time_ns == self.current_time_ns:
                return math.inf
            return self.current_time_ns
        quotient = math.ceil(self.current_time_ns / interval)
        candidate = quotient * interval
        if candidate == self.current_time_ns and self.last_decision_time_ns == candidate:
            candidate += interval
        return candidate

    def _finalize_orphaned_tokens(self) -> None:
        for token in self.epr_tokens.values():
            if token.used:
                continue
            for module in (token.source_module, token.target_module):
                self.epr_reserved_endpoint_counts[module] -= 1
            token.used = True
            self.epr_cancelled_or_orphaned += 1

    def run(self) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
        while True:
            self._expire_epr_tokens()
            self._admit_releases()
            self._process_completion_events()
            self._propagate_dependency_failures()

            if self._decision_is_due() and (
                self.waiting
                or self.configuration.epr_prefetch_enabled
            ):
                self._run_scheduler_decision()

            all_terminal = all(
                runtime.status in {"completed", "rejected"}
                for runtime in self.runtime.values()
            )
            if (
                all_terminal
                and not self.active_execution
                and not self.active_prefetch
            ):
                break

            next_times = [
                self._next_release_time(),
                self._next_completion_time(),
                self._next_token_expiry(),
                self._next_prefetch_opportunity(),
                self._next_decision_time(),
            ]
            finite_times = [
                int(value)
                for value in next_times
                if value != math.inf and value > self.current_time_ns
            ]

            if not finite_times:
                # Resolve dependency deadlocks or impossible resource requests.
                self._propagate_dependency_failures()
                ready = self._ready_request_ids()
                impossible = []
                for request_id in ready:
                    request = self.request_by_id[request_id]
                    if (
                        self.configuration.communication_qubits_per_module < 1
                        or self.configuration.hub_capacity < 1
                        or self.configuration.link_capacity < 1
                    ):
                        impossible.append(request_id)
                for request_id in impossible:
                    runtime = self.runtime[request_id]
                    runtime.status = "rejected"
                    runtime.rejection_reason = "resource_configuration_impossible"
                    self.waiting.remove(request_id)
                    self.rejected_ids.add(request_id)

                if not impossible:
                    unresolved = [
                        request_id
                        for request_id, runtime in self.runtime.items()
                        if runtime.status not in {"completed", "rejected"}
                    ]
                    for request_id in unresolved:
                        runtime = self.runtime[request_id]
                        runtime.status = "rejected"
                        runtime.rejection_reason = "scheduler_deadlock_or_unresolved_dependency"
                        if request_id in self.waiting:
                            self.waiting.remove(request_id)
                        self.rejected_ids.add(request_id)
                    break
                continue

            next_time = min(finite_times)
            self._advance_accounting(next_time)

        self._finalize_orphaned_tokens()
        request_log = self.request_log_dataframe()
        summary = self.simulator_summary(request_log)
        decision_log = pd.DataFrame(self.decision_rows)
        return request_log, summary, decision_log

    # -------------------------------------------------------------------------
    # Logs and summary
    # -------------------------------------------------------------------------

    def request_log_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for request in self.requests:
            runtime = self.runtime[request.request_id]
            completion = runtime.completion_time_ns
            service_start = runtime.service_start_time_ns
            admitted = runtime.admitted_time_ns

            rows.append(
                {
                    "request_id": request.request_id,
                    "operation_uid": request.operation_uid,
                    "tenant_id": request.tenant_id,
                    "role": request.role,
                    "logical_event_id": request.logical_event_id,
                    "circuit_order": request.circuit_order,
                    "release_time_ns": request.release_time_ns,
                    "admitted": admitted is not None,
                    "admission_time_ns": admitted,
                    "scheduler_first_selection_time_ns": runtime.first_selected_time_ns,
                    "scheduler_selection_count": runtime.selection_count,
                    "resource_acquisition_failure_count": runtime.resource_acquisition_failures,
                    "service_start_time_ns": service_start,
                    "completion_time_ns": completion,
                    "queue_delay_ns": (
                        service_start - request.release_time_ns
                        if service_start is not None
                        else np.nan
                    ),
                    "scheduler_delay_after_admission_ns": (
                        service_start - admitted
                        if service_start is not None and admitted is not None
                        else np.nan
                    ),
                    "turnaround_time_ns": (
                        completion - request.release_time_ns
                        if completion is not None
                        else np.nan
                    ),
                    "source_module": request.source_module,
                    "target_module": request.target_module,
                    "route_links": json.dumps(request.route_links),
                    "route_hops": len(request.route_links),
                    "base_service_time_ns": request.base_service_time_ns,
                    "dependency_count": len(request.dependencies),
                    "dependencies": json.dumps(request.dependencies),
                    "layer": request.layer,
                    "deadline_ns": request.deadline_ns,
                    "deadline_missed": (
                        completion > request.deadline_ns
                        if completion is not None
                        else True
                    ),
                    "priority_weight": request.priority_weight,
                    "criticality": request.criticality,
                    "preemption_count": runtime.preemption_count,
                    "used_prefetched_epr": runtime.used_prefetched_epr,
                    "completed": runtime.status == "completed",
                    "rejected": runtime.status == "rejected",
                    "rejection_reason": runtime.rejection_reason,
                }
            )
        return pd.DataFrame(rows)

    def simulator_summary(self, request_log: pd.DataFrame) -> dict[str, Any]:
        makespan_ns = max(
            self.current_time_ns,
            int(request_log["completion_time_ns"].max())
            if request_log["completion_time_ns"].notna().any()
            else 0,
        )
        duration_ns = max(1, makespan_ns)

        tenant_success_ratios = []
        tenant_mean_waits = []
        for _, tenant_frame in request_log.groupby("tenant_id"):
            tenant_success_ratios.append(float(tenant_frame["completed"].mean()))
            completed = tenant_frame[tenant_frame["completed"]]
            tenant_mean_waits.append(
                float(completed["queue_delay_ns"].mean())
                if not completed.empty
                else float(duration_ns)
            )

        completed_by_service = request_log[request_log["completed"]].sort_values(
            ["service_start_time_ns", "request_id"]
        )
        release_rank = {
            request_id: rank
            for rank, request_id in enumerate(
                request_log.sort_values(
                    ["release_time_ns", "request_id"]
                )["request_id"]
            )
        }
        service_release_ranks = [
            release_rank[request_id]
            for request_id in completed_by_service["request_id"]
        ]

        return {
            "request_count": int(len(request_log)),
            "completed_request_count": int(request_log["completed"].sum()),
            "rejected_request_count": int(request_log["rejected"].sum()),
            "queue_admission_rate": float(request_log["admitted"].mean()),
            "request_success_rate": float(request_log["completed"].mean()),
            "mean_remote_latency_ns": float(
                request_log.loc[request_log["completed"], "turnaround_time_ns"].mean()
            ) if request_log["completed"].any() else np.nan,
            "p95_remote_latency_ns": float(
                request_log.loc[request_log["completed"], "turnaround_time_ns"].quantile(0.95)
            ) if request_log["completed"].any() else np.nan,
            "p99_remote_latency_ns": float(
                request_log.loc[request_log["completed"], "turnaround_time_ns"].quantile(0.99)
            ) if request_log["completed"].any() else np.nan,
            "mean_queue_delay_ns": float(
                request_log.loc[request_log["completed"], "queue_delay_ns"].mean()
            ) if request_log["completed"].any() else np.nan,
            "p95_queue_delay_ns": float(
                request_log.loc[request_log["completed"], "queue_delay_ns"].quantile(0.95)
            ) if request_log["completed"].any() else np.nan,
            "deadline_miss_fraction": float(request_log["deadline_missed"].mean()),
            "reordering_rate": float(inversion_fraction(service_release_ranks)),
            "tenant_success_fairness": jain_fairness(tenant_success_ratios),
            "tenant_wait_fairness": jain_fairness(
                [1.0 / (1.0 + wait) for wait in tenant_mean_waits]
            ),
            "hub_utilization": float(
                self.hub_busy_slot_ns
                / (duration_ns * self.configuration.hub_capacity)
            ),
            "link_utilization": float(
                self.link_busy_capacity_ns
                / (
                    duration_ns
                    * self.configuration.link_capacity
                    * len(PHYSICAL_LINKS)
                )
            ),
            "communication_qubit_utilization": float(
                self.endpoint_busy_capacity_ns
                / (
                    duration_ns
                    * self.configuration.communication_qubits_per_module
                    * NUM_MODULES
                )
            ),
            "epr_reservation_utilization": float(
                self.epr_reserved_capacity_ns
                / (
                    duration_ns
                    * self.configuration.communication_qubits_per_module
                    * NUM_MODULES
                )
            ),
            "mean_queue_occupancy": float(self.queue_area_ns / duration_ns),
            "maximum_queue_occupancy": int(self.maximum_queue_occupancy),
            "scheduler_decision_count": int(self.scheduler_decisions),
            "scheduler_idle_decision_count": int(self.scheduler_idle_decisions),
            "preemption_event_count": int(self.preemption_events),
            "epr_prefetched_count": int(self.epr_prefetched),
            "epr_used_count": int(self.epr_used),
            "epr_expired_count": int(self.epr_expired),
            "epr_cancelled_or_orphaned_count": int(self.epr_cancelled_or_orphaned),
            "epr_expiration_fraction": float(
                self.epr_expired / self.epr_prefetched
                if self.epr_prefetched > 0
                else 0.0
            ),
            "makespan_ns": int(makespan_ns),
        }


# =============================================================================
# Scenario controls and attacker comparison
# =============================================================================


def scenario_requests(
    all_requests: Sequence[SchedulerRequest],
    included_roles: set[str],
) -> list[SchedulerRequest]:
    selected = [
        request
        for request in all_requests
        if request.role in included_roles
    ]
    selected_ids = {request.request_id for request in selected}

    # Remove dependencies belonging to omitted roles while preserving all
    # dependencies inside the selected counterfactual execution.
    return [
        SchedulerRequest(
            **{
                **request.__dict__,
                "dependencies": tuple(
                    dependency
                    for dependency in request.dependencies
                    if dependency in selected_ids
                ),
            }
        )
        for request in selected
    ]


def execute_scenario(
    all_requests: Sequence[SchedulerRequest],
    included_roles: set[str],
    configuration: SchedulerConfiguration,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    simulator = RemoteOperationSchedulerSimulator(
        scenario_requests(all_requests, included_roles),
        configuration,
        seed,
    )
    return simulator.run()


def compare_attacker_runs(
    attacker_only_log: pd.DataFrame,
    combined_log: pd.DataFrame,
) -> pd.DataFrame:
    baseline = attacker_only_log[attacker_only_log["role"] == "attacker"].copy()
    victim_on = combined_log[combined_log["role"] == "attacker"].copy()

    columns = [
        "operation_uid",
        "logical_event_id",
        "release_time_ns",
        "source_module",
        "target_module",
        "route_hops",
        "completed",
        "rejected",
        "rejection_reason",
        "queue_delay_ns",
        "turnaround_time_ns",
        "service_start_time_ns",
        "completion_time_ns",
        "preemption_count",
        "used_prefetched_epr",
    ]

    baseline = baseline[columns].rename(
        columns={
            column: f"baseline_{column}"
            for column in columns
            if column not in {
                "operation_uid",
                "logical_event_id",
                "release_time_ns",
                "source_module",
                "target_module",
                "route_hops",
            }
        }
    )
    victim_on = victim_on[columns].rename(
        columns={
            column: f"combined_{column}"
            for column in columns
            if column not in {
                "operation_uid",
                "logical_event_id",
                "release_time_ns",
                "source_module",
                "target_module",
                "route_hops",
            }
        }
    )

    merged = victim_on.merge(
        baseline,
        on=[
            "operation_uid",
            "logical_event_id",
            "release_time_ns",
            "source_module",
            "target_module",
            "route_hops",
        ],
        how="outer",
        validate="one_to_one",
    )

    merged["baseline_completed"] = merged["baseline_completed"].fillna(False)
    merged["combined_completed"] = merged["combined_completed"].fillna(False)
    merged["baseline_rejected"] = merged["baseline_rejected"].fillna(True)
    merged["combined_rejected"] = merged["combined_rejected"].fillna(True)

    both_completed = merged["baseline_completed"] & merged["combined_completed"]
    merged["signed_turnaround_change_ns"] = np.where(
        both_completed,
        merged["combined_turnaround_time_ns"]
        - merged["baseline_turnaround_time_ns"],
        np.nan,
    )
    merged["absolute_turnaround_change_ns"] = np.abs(
        merged["signed_turnaround_change_ns"]
    )
    merged["positive_delay_observed"] = (
        merged["signed_turnaround_change_ns"] > TIMING_CHANGE_THRESHOLD_NS
    )
    merged["negative_speedup_observed"] = (
        merged["signed_turnaround_change_ns"] < -TIMING_CHANGE_THRESHOLD_NS
    )
    merged["failure_transition_observed"] = (
        merged["baseline_completed"] & ~merged["combined_completed"]
    )
    merged["recovery_transition_observed"] = (
        ~merged["baseline_completed"] & merged["combined_completed"]
    )
    merged["any_timing_change_observed"] = (
        merged["absolute_turnaround_change_ns"] > TIMING_CHANGE_THRESHOLD_NS
    )
    merged["any_observable_change"] = (
        merged["any_timing_change_observed"]
        | merged["failure_transition_observed"]
        | merged["recovery_transition_observed"]
        | (
            merged["baseline_used_prefetched_epr"].fillna(False)
            != merged["combined_used_prefetched_epr"].fillna(False)
        )
        | (
            merged["baseline_preemption_count"].fillna(0)
            != merged["combined_preemption_count"].fillna(0)
        )
    )

    return merged.sort_values("logical_event_id").reset_index(drop=True)


# =============================================================================
# Trial metrics, phase visibility, and fingerprint features
# =============================================================================


def successful_role_completion_time(
    request_log: pd.DataFrame,
    role: str,
) -> tuple[bool, float]:
    role_frame = request_log[request_log["role"] == role]
    if role_frame.empty:
        return True, 0.0
    success = bool(role_frame["completed"].all())
    if not success:
        return False, np.nan
    return True, float(role_frame["completion_time_ns"].max())


def phase_boundary_visibility(
    attacker_comparison: pd.DataFrame,
    combined_request_log: pd.DataFrame,
) -> dict[str, float]:
    victim = combined_request_log[
        (combined_request_log["role"] == "victim")
        & combined_request_log["completed"]
    ].sort_values(["layer", "release_time_ns"])

    boundary_times = []
    for layer, frame in victim.groupby("layer"):
        if int(layer) == int(victim["layer"].min()):
            continue
        boundary_times.append(float(frame["release_time_ns"].min()))

    attacker = attacker_comparison.sort_values("release_time_ns")
    signed = attacker["signed_turnaround_change_ns"].fillna(0.0).to_numpy(dtype=float)
    release_times = attacker["release_time_ns"].to_numpy(dtype=float)

    transition_times = []
    if len(signed) > 0:
        observed = (
            attacker["any_observable_change"].to_numpy(dtype=bool)
        )
        for index in range(len(signed)):
            changed = observed[index]
            if index > 0:
                changed = changed or (
                    abs(signed[index] - signed[index - 1]) > 0
                )
            if changed:
                transition_times.append(float(release_times[index]))

    visible = 0
    for boundary in boundary_times:
        if any(
            abs(transition - boundary) <= PHASE_BOUNDARY_TOLERANCE_NS
            for transition in transition_times
        ):
            visible += 1

    return {
        "victim_phase_boundary_count": int(len(boundary_times)),
        "observed_timing_transition_count": int(len(transition_times)),
        "visible_phase_boundary_count": int(visible),
        "phase_boundary_visibility_fraction": float(
            visible / len(boundary_times)
            if boundary_times
            else 0.0
        ),
    }


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def safe_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    first = values[:-lag]
    second = values[lag:]
    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    value = np.corrcoef(first, second)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def attacker_fingerprint_features(
    attacker_comparison: pd.DataFrame,
) -> dict[str, float]:
    frame = attacker_comparison.sort_values("logical_event_id").reset_index(drop=True)
    signed = frame["signed_turnaround_change_ns"].fillna(0.0).to_numpy(dtype=float)
    absolute = frame["absolute_turnaround_change_ns"].fillna(0.0).to_numpy(dtype=float)
    positive = frame["positive_delay_observed"].to_numpy(dtype=bool)
    negative = frame["negative_speedup_observed"].to_numpy(dtype=bool)
    failure = frame["failure_transition_observed"].to_numpy(dtype=bool)
    observable = frame["any_observable_change"].to_numpy(dtype=bool)

    features: dict[str, float] = {
        "bb_probe_count": float(len(frame)),
        "bb_signed_mean_ns": float(signed.mean()) if len(signed) else 0.0,
        "bb_signed_std_ns": float(signed.std()) if len(signed) else 0.0,
        "bb_signed_median_ns": float(np.median(signed)) if len(signed) else 0.0,
        "bb_signed_min_ns": float(signed.min()) if len(signed) else 0.0,
        "bb_signed_max_ns": float(signed.max()) if len(signed) else 0.0,
        "bb_signed_sum_ns": float(signed.sum()),
        "bb_absolute_mean_ns": float(absolute.mean()) if len(absolute) else 0.0,
        "bb_absolute_max_ns": float(absolute.max()) if len(absolute) else 0.0,
        "bb_absolute_sum_ns": float(absolute.sum()),
        "bb_positive_delay_fraction": float(positive.mean()) if len(positive) else 0.0,
        "bb_negative_speedup_fraction": float(negative.mean()) if len(negative) else 0.0,
        "bb_failure_transition_fraction": float(failure.mean()) if len(failure) else 0.0,
        "bb_observable_change_fraction": float(observable.mean()) if len(observable) else 0.0,
        "bb_longest_observable_run": float(longest_true_run(observable)),
        "bb_transition_count": float(
            np.count_nonzero(observable[1:] != observable[:-1])
            if len(observable) > 1
            else 0
        ),
    }

    for percentile in [10, 25, 50, 75, 90, 95]:
        features[f"bb_absolute_p{percentile}_ns"] = float(
            np.percentile(absolute, percentile)
            if len(absolute)
            else 0.0
        )

    for lag in range(1, 6):
        features[f"bb_signed_autocorr_lag_{lag}"] = safe_autocorrelation(
            signed,
            lag,
        )

    bins = np.array_split(np.arange(len(frame)), NUM_TEMPORAL_BINS)
    for bin_index, indices in enumerate(bins):
        if len(indices) == 0:
            bin_signed = np.array([0.0])
            bin_observable = np.array([False])
        else:
            bin_signed = signed[indices]
            bin_observable = observable[indices]
        features[f"bb_bin_{bin_index:02d}_signed_mean_ns"] = float(bin_signed.mean())
        features[f"bb_bin_{bin_index:02d}_absolute_sum_ns"] = float(
            np.abs(bin_signed).sum()
        )
        features[f"bb_bin_{bin_index:02d}_observable_fraction"] = float(
            bin_observable.mean()
        )

    for probe_index, value in enumerate(signed):
        features[f"bb_probe_{probe_index:03d}_signed_change_ns"] = float(value)
    for probe_index, value in enumerate(failure):
        features[f"bb_probe_{probe_index:03d}_failure_transition"] = float(value)

    return features


def create_trial_summary(
    *,
    configuration: SchedulerConfiguration,
    qasm_name: str,
    trial_id: int,
    trace_seed: int,
    attacker_only_log: pd.DataFrame,
    attacker_only_summary: dict[str, Any],
    victim_only_log: pd.DataFrame,
    victim_only_summary: dict[str, Any],
    combined_log: pd.DataFrame,
    combined_summary: dict[str, Any],
    attacker_comparison: pd.DataFrame,
) -> dict[str, Any]:
    victim_only_success, victim_only_completion = successful_role_completion_time(
        victim_only_log,
        "victim",
    )
    combined_victim_success, combined_victim_completion = successful_role_completion_time(
        combined_log,
        "victim",
    )

    if victim_only_success and combined_victim_success and victim_only_completion > 0:
        victim_slowdown_ratio = combined_victim_completion / victim_only_completion
        victim_slowdown_ns = combined_victim_completion - victim_only_completion
    else:
        victim_slowdown_ratio = np.nan
        victim_slowdown_ns = np.nan

    both_completed = attacker_comparison[
        attacker_comparison["baseline_completed"]
        & attacker_comparison["combined_completed"]
    ]

    phase_metrics = phase_boundary_visibility(
        attacker_comparison,
        combined_log,
    )

    return {
        "configuration_id": configuration.configuration_id,
        "subexperiment": configuration.subexperiment,
        "scheduler_policy": configuration.scheduler_policy,
        "victim_qasm": qasm_name,
        "victim_tag": safe_tag(qasm_name),
        "trial_id": trial_id,
        "trace_seed": trace_seed,
        "queue_depth": configuration.queue_depth,
        "tenant_count": configuration.tenant_count,
        "hub_capacity": configuration.hub_capacity,
        "link_capacity": configuration.link_capacity,
        "communication_qubits_per_module": configuration.communication_qubits_per_module,
        "priority_profile": configuration.priority_profile,
        "decision_interval_ns": configuration.decision_interval_ns,
        "lookahead_window_ns": configuration.lookahead_window_ns,
        "preemption_allowed": configuration.preemption_allowed,
        "epr_prefetch_enabled": configuration.epr_prefetch_enabled,
        "attacker_probe_count": int(len(attacker_comparison)),
        "attacker_both_completed_count": int(len(both_completed)),
        "positive_delay_probe_fraction": float(
            attacker_comparison["positive_delay_observed"].mean()
        ),
        "negative_speedup_probe_fraction": float(
            attacker_comparison["negative_speedup_observed"].mean()
        ),
        "failure_transition_probe_fraction": float(
            attacker_comparison["failure_transition_observed"].mean()
        ),
        "recovery_transition_probe_fraction": float(
            attacker_comparison["recovery_transition_observed"].mean()
        ),
        "any_timing_change_probe_fraction": float(
            attacker_comparison["any_timing_change_observed"].mean()
        ),
        "any_observable_change_probe_fraction": float(
            attacker_comparison["any_observable_change"].mean()
        ),
        "mean_signed_attacker_change_ns": float(
            both_completed["signed_turnaround_change_ns"].mean()
        ) if not both_completed.empty else np.nan,
        "mean_absolute_attacker_change_ns": float(
            both_completed["absolute_turnaround_change_ns"].mean()
        ) if not both_completed.empty else np.nan,
        "max_positive_attacker_delay_ns": float(
            both_completed["signed_turnaround_change_ns"].max()
        ) if not both_completed.empty else np.nan,
        "max_attacker_speedup_ns": float(
            -both_completed["signed_turnaround_change_ns"].min()
        ) if not both_completed.empty else np.nan,
        "total_absolute_attacker_change_ns": float(
            both_completed["absolute_turnaround_change_ns"].sum()
        ) if not both_completed.empty else 0.0,
        "attacker_detection_probability": float(
            attacker_comparison["any_observable_change"].mean()
        ),
        "victim_only_success": victim_only_success,
        "combined_victim_success": combined_victim_success,
        "victim_slowdown_ns_successful_only": victim_slowdown_ns,
        "victim_slowdown_ratio_successful_only": victim_slowdown_ratio,
        **{
            f"attacker_only_{key}": value
            for key, value in attacker_only_summary.items()
        },
        **{
            f"victim_only_{key}": value
            for key, value in victim_only_summary.items()
        },
        **{
            f"combined_{key}": value
            for key, value in combined_summary.items()
        },
        **phase_metrics,
    }


# =============================================================================
# Experimental configuration generation
# =============================================================================


def generate_configurations() -> list[SchedulerConfiguration]:
    configurations: list[SchedulerConfiguration] = []

    for policy in SCHEDULER_POLICIES:
        for tenant_count in TENANT_COUNT_OPTIONS:
            for hub_capacity, link_capacity in HUB_LINK_CAPACITY_OPTIONS:
                configurations.append(
                    SchedulerConfiguration(
                        subexperiment="core_policy_capacity_tenancy",
                        scheduler_policy=policy,
                        queue_depth=DEFAULT_QUEUE_DEPTH,
                        tenant_count=tenant_count,
                        hub_capacity=hub_capacity,
                        link_capacity=link_capacity,
                        communication_qubits_per_module=DEFAULT_COMMUNICATION_QUBITS,
                        priority_profile=DEFAULT_PRIORITY_PROFILE,
                        decision_interval_ns=DEFAULT_DECISION_INTERVAL_NS,
                        lookahead_window_ns=DEFAULT_LOOKAHEAD_NS,
                        preemption_allowed=DEFAULT_PREEMPTION,
                        epr_prefetch_enabled=DEFAULT_EPR_PREFETCH,
                    )
                )

    for policy in SCHEDULER_POLICIES:
        for queue_depth in QUEUE_DEPTH_OPTIONS:
            for communication_qubits in COMMUNICATION_QUBIT_OPTIONS:
                configurations.append(
                    SchedulerConfiguration(
                        subexperiment="queue_communication_capacity",
                        scheduler_policy=policy,
                        queue_depth=queue_depth,
                        tenant_count=DEFAULT_TENANT_COUNT,
                        hub_capacity=DEFAULT_HUB_CAPACITY,
                        link_capacity=DEFAULT_LINK_CAPACITY,
                        communication_qubits_per_module=communication_qubits,
                        priority_profile=DEFAULT_PRIORITY_PROFILE,
                        decision_interval_ns=DEFAULT_DECISION_INTERVAL_NS,
                        lookahead_window_ns=DEFAULT_LOOKAHEAD_NS,
                        preemption_allowed=DEFAULT_PREEMPTION,
                        epr_prefetch_enabled=DEFAULT_EPR_PREFETCH,
                    )
                )

    for policy in SCHEDULER_POLICIES:
        for decision_interval_ns in DECISION_INTERVAL_OPTIONS_NS:
            for lookahead_window_ns in LOOKAHEAD_OPTIONS_NS:
                configurations.append(
                    SchedulerConfiguration(
                        subexperiment="decision_interval_lookahead",
                        scheduler_policy=policy,
                        queue_depth=DEFAULT_QUEUE_DEPTH,
                        tenant_count=DEFAULT_TENANT_COUNT,
                        hub_capacity=DEFAULT_HUB_CAPACITY,
                        link_capacity=DEFAULT_LINK_CAPACITY,
                        communication_qubits_per_module=DEFAULT_COMMUNICATION_QUBITS,
                        priority_profile=DEFAULT_PRIORITY_PROFILE,
                        decision_interval_ns=decision_interval_ns,
                        lookahead_window_ns=lookahead_window_ns,
                        preemption_allowed=DEFAULT_PREEMPTION,
                        epr_prefetch_enabled=DEFAULT_EPR_PREFETCH,
                    )
                )

    for policy in SCHEDULER_POLICIES:
        for preemption_allowed in PREEMPTION_OPTIONS:
            for epr_prefetch_enabled in EPR_PREFETCH_OPTIONS:
                for priority_profile in PRIORITY_PROFILES:
                    configurations.append(
                        SchedulerConfiguration(
                            subexperiment="preemption_prefetch_priority",
                            scheduler_policy=policy,
                            queue_depth=DEFAULT_QUEUE_DEPTH,
                            tenant_count=DEFAULT_TENANT_COUNT,
                            hub_capacity=DEFAULT_HUB_CAPACITY,
                            link_capacity=DEFAULT_LINK_CAPACITY,
                            communication_qubits_per_module=DEFAULT_COMMUNICATION_QUBITS,
                            priority_profile=priority_profile,
                            decision_interval_ns=DEFAULT_DECISION_INTERVAL_NS,
                            lookahead_window_ns=DEFAULT_LOOKAHEAD_NS,
                            preemption_allowed=preemption_allowed,
                            epr_prefetch_enabled=epr_prefetch_enabled,
                        )
                    )

    # De-duplicate exact configurations that may arise across complementary
    # sub-sweeps while preserving subexperiment labels.
    seen: set[tuple[Any, ...]] = set()
    unique: list[SchedulerConfiguration] = []
    for configuration in configurations:
        key = (
            configuration.subexperiment,
            configuration.scheduler_policy,
            configuration.queue_depth,
            configuration.tenant_count,
            configuration.hub_capacity,
            configuration.link_capacity,
            configuration.communication_qubits_per_module,
            configuration.priority_profile,
            configuration.decision_interval_ns,
            configuration.lookahead_window_ns,
            configuration.preemption_allowed,
            configuration.epr_prefetch_enabled,
        )
        if key not in seen:
            seen.add(key)
            unique.append(configuration)

    if RUN_QUICK_VALIDATION:
        quick_policies = [
            "first_come_first_served",
            "round_robin_tenants",
            "randomized_arbitration",
        ]
        unique = [
            configuration
            for configuration in unique
            if configuration.scheduler_policy in quick_policies
            and configuration.subexperiment == "core_policy_capacity_tenancy"
            and configuration.tenant_count == 4
            and configuration.hub_capacity == 2
            and configuration.link_capacity == 1
        ]

    if MAX_CONFIGURATIONS is not None:
        unique = unique[:MAX_CONFIGURATIONS]

    return unique


# =============================================================================
# One physical trial
# =============================================================================


def attach_trial_metadata(
    dataframe: pd.DataFrame,
    configuration: SchedulerConfiguration,
    qasm_name: str,
    trial_id: int,
    scenario_name: str,
) -> pd.DataFrame:
    frame = dataframe.copy()
    metadata = {
        "configuration_id": configuration.configuration_id,
        "subexperiment": configuration.subexperiment,
        "scheduler_policy": configuration.scheduler_policy,
        "victim_qasm": qasm_name,
        "victim_tag": safe_tag(qasm_name),
        "trial_id": trial_id,
        "scenario": scenario_name,
        "queue_depth": configuration.queue_depth,
        "tenant_count": configuration.tenant_count,
        "hub_capacity": configuration.hub_capacity,
        "link_capacity": configuration.link_capacity,
        "communication_qubits_per_module": configuration.communication_qubits_per_module,
        "priority_profile": configuration.priority_profile,
        "decision_interval_ns": configuration.decision_interval_ns,
        "lookahead_window_ns": configuration.lookahead_window_ns,
        "preemption_allowed": configuration.preemption_allowed,
        "epr_prefetch_enabled": configuration.epr_prefetch_enabled,
    }
    for column, value in reversed(list(metadata.items())):
        frame.insert(0, column, value)
    return frame


def run_one_trial(
    configuration: SchedulerConfiguration,
    qasm_path: Path,
    trial_id: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    list[pd.DataFrame],
    list[pd.DataFrame],
]:
    # The trace seed excludes scheduler policy, guaranteeing identical released
    # operations across policies for the same independent-variable setting.
    trace_seed = stable_seed(
        configuration.subexperiment,
        configuration.queue_depth,
        configuration.tenant_count,
        configuration.hub_capacity,
        configuration.link_capacity,
        configuration.communication_qubits_per_module,
        configuration.priority_profile,
        configuration.decision_interval_ns,
        configuration.lookahead_window_ns,
        configuration.preemption_allowed,
        configuration.epr_prefetch_enabled,
        qasm_path.name,
        trial_id,
    )

    jobs, allocations = build_jobs_for_trial(
        qasm_path,
        configuration.tenant_count,
        trace_seed,
    )
    all_requests = build_scheduler_requests(
        jobs,
        allocations,
        configuration.priority_profile,
    )

    # The same scheduler seed is used in all counterfactual controls. Randomized
    # arbitration itself is request-specific, so shared requests retain coupled
    # random priorities across attacker-only, victim-only, and combined runs.
    scheduler_seed = stable_seed(
        configuration.configuration_id,
        qasm_path.name,
        trial_id,
        "scheduler",
    )

    attacker_only_log, attacker_only_summary, attacker_decisions = execute_scenario(
        all_requests,
        {"attacker", "background"},
        configuration,
        scheduler_seed,
    )
    victim_only_log, victim_only_summary, victim_decisions = execute_scenario(
        all_requests,
        {"victim", "background"},
        configuration,
        scheduler_seed,
    )
    combined_log, combined_summary, combined_decisions = execute_scenario(
        all_requests,
        {"victim", "attacker", "background"},
        configuration,
        scheduler_seed,
    )

    attacker_comparison = compare_attacker_runs(
        attacker_only_log,
        combined_log,
    )

    trial_summary = create_trial_summary(
        configuration=configuration,
        qasm_name=qasm_path.name,
        trial_id=trial_id,
        trace_seed=trace_seed,
        attacker_only_log=attacker_only_log,
        attacker_only_summary=attacker_only_summary,
        victim_only_log=victim_only_log,
        victim_only_summary=victim_only_summary,
        combined_log=combined_log,
        combined_summary=combined_summary,
        attacker_comparison=attacker_comparison,
    )

    feature_row = {
        "configuration_id": configuration.configuration_id,
        "subexperiment": configuration.subexperiment,
        "scheduler_policy": configuration.scheduler_policy,
        "victim_qasm": qasm_path.name,
        "victim_tag": safe_tag(qasm_path.name),
        "trial_id": trial_id,
        "queue_depth": configuration.queue_depth,
        "tenant_count": configuration.tenant_count,
        "hub_capacity": configuration.hub_capacity,
        "link_capacity": configuration.link_capacity,
        "communication_qubits_per_module": configuration.communication_qubits_per_module,
        "priority_profile": configuration.priority_profile,
        "decision_interval_ns": configuration.decision_interval_ns,
        "lookahead_window_ns": configuration.lookahead_window_ns,
        "preemption_allowed": configuration.preemption_allowed,
        "epr_prefetch_enabled": configuration.epr_prefetch_enabled,
        **attacker_fingerprint_features(attacker_comparison),
    }

    attacker_comparison = attach_trial_metadata(
        attacker_comparison,
        configuration,
        qasm_path.name,
        trial_id,
        "attacker_comparison",
    )

    request_frames = [
        attach_trial_metadata(
            attacker_only_log,
            configuration,
            qasm_path.name,
            trial_id,
            "attacker_only",
        ),
        attach_trial_metadata(
            victim_only_log,
            configuration,
            qasm_path.name,
            trial_id,
            "victim_only",
        ),
        attach_trial_metadata(
            combined_log,
            configuration,
            qasm_path.name,
            trial_id,
            "combined",
        ),
    ]

    decision_frames = []
    if SAVE_SCHEDULER_DECISION_LOG:
        for scenario_name, frame in [
            ("attacker_only", attacker_decisions),
            ("victim_only", victim_decisions),
            ("combined", combined_decisions),
        ]:
            if not frame.empty:
                decision_frames.append(
                    attach_trial_metadata(
                        frame,
                        configuration,
                        qasm_path.name,
                        trial_id,
                        scenario_name,
                    )
                )

    return (
        trial_summary,
        feature_row,
        attacker_comparison,
        request_frames,
        decision_frames,
    )


# =============================================================================
# Aggregate summaries
# =============================================================================


def aggregate_summary(
    trial_dataframe: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    return (
        trial_dataframe.groupby(list(group_columns), dropna=False, as_index=False)
        .agg(
            trial_count=("trial_id", "count"),
            mean_remote_latency_ns=("combined_mean_remote_latency_ns", "mean"),
            p95_remote_latency_ns=("combined_p95_remote_latency_ns", "mean"),
            p99_remote_latency_ns=("combined_p99_remote_latency_ns", "mean"),
            mean_queue_delay_ns=("combined_mean_queue_delay_ns", "mean"),
            p95_queue_delay_ns=("combined_p95_queue_delay_ns", "mean"),
            mean_positive_delay_fraction=("positive_delay_probe_fraction", "mean"),
            mean_negative_speedup_fraction=("negative_speedup_probe_fraction", "mean"),
            mean_failure_transition_fraction=("failure_transition_probe_fraction", "mean"),
            mean_observable_change_fraction=("any_observable_change_probe_fraction", "mean"),
            mean_signed_attacker_change_ns=("mean_signed_attacker_change_ns", "mean"),
            mean_absolute_attacker_change_ns=("mean_absolute_attacker_change_ns", "mean"),
            mean_victim_slowdown_ratio_successful_only=(
                "victim_slowdown_ratio_successful_only",
                "mean",
            ),
            victim_success_probability=("combined_victim_success", "mean"),
            tenant_success_fairness=("combined_tenant_success_fairness", "mean"),
            tenant_wait_fairness=("combined_tenant_wait_fairness", "mean"),
            hub_utilization=("combined_hub_utilization", "mean"),
            link_utilization=("combined_link_utilization", "mean"),
            communication_qubit_utilization=(
                "combined_communication_qubit_utilization",
                "mean",
            ),
            scheduler_reordering_rate=("combined_reordering_rate", "mean"),
            deadline_miss_fraction=("combined_deadline_miss_fraction", "mean"),
            mean_phase_boundary_visibility=(
                "phase_boundary_visibility_fraction",
                "mean",
            ),
            mean_preemption_count=("combined_preemption_event_count", "mean"),
            mean_epr_expiration_fraction=(
                "combined_epr_expiration_fraction",
                "mean",
            ),
            mean_queue_occupancy=("combined_mean_queue_occupancy", "mean"),
            mean_maximum_queue_occupancy=(
                "combined_maximum_queue_occupancy",
                "mean",
            ),
            request_rejection_fraction=(
                "combined_rejected_request_count",
                lambda values: float(values.sum())
                / max(
                    1.0,
                    float(
                        trial_dataframe.loc[values.index, "combined_request_count"].sum()
                    ),
                ),
            ),
        )
    )


def build_all_summaries(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    common = ["scheduler_policy"]
    return {
        "policy": aggregate_summary(trials, common),
        "capacity": aggregate_summary(
            trials,
            ["scheduler_policy", "hub_capacity", "link_capacity"],
        ),
        "queue": aggregate_summary(
            trials,
            ["scheduler_policy", "queue_depth"],
        ),
        "tenancy": aggregate_summary(
            trials,
            ["scheduler_policy", "tenant_count"],
        ),
        "communication_qubit": aggregate_summary(
            trials,
            ["scheduler_policy", "communication_qubits_per_module"],
        ),
        "priority": aggregate_summary(
            trials,
            ["scheduler_policy", "priority_profile"],
        ),
        "decision_interval": aggregate_summary(
            trials,
            ["scheduler_policy", "decision_interval_ns"],
        ),
        "lookahead": aggregate_summary(
            trials,
            ["scheduler_policy", "lookahead_window_ns"],
        ),
        "preemption": aggregate_summary(
            trials,
            ["scheduler_policy", "preemption_allowed"],
        ),
        "prefetch": aggregate_summary(
            trials,
            ["scheduler_policy", "epr_prefetch_enabled"],
        ),
        "phase_visibility": aggregate_summary(
            trials,
            ["scheduler_policy", "victim_tag"],
        ),
    }


# =============================================================================
# Fingerprint stability
# =============================================================================


def fingerprint_stability(feature_dataframe: pd.DataFrame) -> pd.DataFrame:
    probe_columns = sorted(
        column
        for column in feature_dataframe.columns
        if column.startswith("bb_probe_")
        and column.endswith("_signed_change_ns")
    )

    rows: list[dict[str, Any]] = []
    grouping = [
        "subexperiment",
        "scheduler_policy",
        "victim_tag",
        "queue_depth",
        "tenant_count",
        "hub_capacity",
        "link_capacity",
        "communication_qubits_per_module",
        "priority_profile",
        "decision_interval_ns",
        "lookahead_window_ns",
        "preemption_allowed",
        "epr_prefetch_enabled",
    ]

    for keys, frame in feature_dataframe.groupby(grouping, dropna=False):
        traces = frame[probe_columns].to_numpy(dtype=float)
        correlations = []
        maes = []
        for first in range(len(traces)):
            for second in range(first + 1, len(traces)):
                correlations.append(pearson_or_zero(traces[first], traces[second]))
                maes.append(float(np.mean(np.abs(traces[first] - traces[second]))))

        row = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "trial_count": int(len(frame)),
                "mean_pairwise_trace_correlation": float(np.mean(correlations))
                if correlations
                else 1.0,
                "minimum_pairwise_trace_correlation": float(np.min(correlations))
                if correlations
                else 1.0,
                "mean_pairwise_trace_mae_ns": float(np.mean(maes))
                if maes
                else 0.0,
                "maximum_pairwise_trace_mae_ns": float(np.max(maes))
                if maes
                else 0.0,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Workload fingerprint classification
# =============================================================================


def canonical_classification_subset(features: pd.DataFrame) -> pd.DataFrame:
    return features[
        (features["subexperiment"] == "core_policy_capacity_tenancy")
        & (features["tenant_count"] == DEFAULT_TENANT_COUNT)
        & (features["hub_capacity"] == DEFAULT_HUB_CAPACITY)
        & (features["link_capacity"] == DEFAULT_LINK_CAPACITY)
        & (features["communication_qubits_per_module"] == DEFAULT_COMMUNICATION_QUBITS)
        & (features["queue_depth"] == DEFAULT_QUEUE_DEPTH)
        & (features["priority_profile"] == DEFAULT_PRIORITY_PROFILE)
        & (features["decision_interval_ns"] == DEFAULT_DECISION_INTERVAL_NS)
        & (features["lookahead_window_ns"] == DEFAULT_LOOKAHEAD_NS)
        & (~features["preemption_allowed"])
        & (~features["epr_prefetch_enabled"])
    ].copy()


def nonml_nearest_template_classification(
    feature_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = canonical_classification_subset(feature_dataframe)
    probe_columns = sorted(
        column
        for column in subset.columns
        if column.startswith("bb_probe_")
        and column.endswith("_signed_change_ns")
    )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for policy, policy_frame in subset.groupby("scheduler_policy"):
        trial_ids = sorted(policy_frame["trial_id"].unique())
        labels = sorted(policy_frame["victim_tag"].unique())

        for held_out_trial in trial_ids:
            training = policy_frame[policy_frame["trial_id"] != held_out_trial]
            testing = policy_frame[policy_frame["trial_id"] == held_out_trial]
            templates = {
                label: training[training["victim_tag"] == label][probe_columns]
                .mean(axis=0)
                .to_numpy(dtype=float)
                for label in labels
            }

            for _, row in testing.iterrows():
                trace = row[probe_columns].to_numpy(dtype=float)
                distances = {
                    label: float(np.mean(np.abs(trace - template)))
                    for label, template in templates.items()
                }
                predicted = min(distances, key=distances.get)
                ordered = sorted(distances.items(), key=lambda item: item[1])
                prediction_rows.append(
                    {
                        "scheduler_policy": policy,
                        "held_out_trial_id": int(held_out_trial),
                        "victim_tag": row["victim_tag"],
                        "predicted_victim_tag": predicted,
                        "correct": predicted == row["victim_tag"],
                        "best_template_mae_ns": ordered[0][1],
                        "template_margin_ns": ordered[1][1] - ordered[0][1]
                        if len(ordered) > 1
                        else np.nan,
                    }
                )

        predictions = pd.DataFrame(
            [row for row in prediction_rows if row["scheduler_policy"] == policy]
        )
        metric_rows.append(
            {
                "scheduler_policy": policy,
                "method": "nearest_mean_trace_template",
                "sample_count": int(len(predictions)),
                "class_count": int(len(labels)),
                "chance_accuracy": 1.0 / max(1, len(labels)),
                "accuracy": float(predictions["correct"].mean())
                if not predictions.empty
                else np.nan,
                "mean_template_margin_ns": float(
                    predictions["template_margin_ns"].mean()
                ) if not predictions.empty else np.nan,
            }
        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def random_forest_classification(
    feature_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = canonical_classification_subset(feature_dataframe)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
        )
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    except ImportError:
        return (
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "scheduler_policy": "all",
                        "method": "random_forest",
                        "error": "scikit-learn is not installed",
                    }
                ]
            ),
            pd.DataFrame(),
        )

    feature_columns = [
        column
        for column in subset.columns
        if column.startswith("bb_")
    ]

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []

    for policy, frame in subset.groupby("scheduler_policy"):
        frame = frame.reset_index(drop=True)
        if frame["trial_id"].nunique() < 2:
            continue

        matrix = (
            frame[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        labels = frame["victim_tag"].to_numpy()
        groups = frame["trial_id"].to_numpy()

        classifier = RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        )
        predictions = cross_val_predict(
            classifier,
            matrix,
            labels,
            groups=groups,
            cv=LeaveOneGroupOut(),
            n_jobs=-1,
        )

        prediction_frame = frame[
            ["scheduler_policy", "victim_tag", "trial_id"]
        ].copy()
        prediction_frame["predicted_victim_tag"] = predictions
        prediction_frame["correct"] = prediction_frame["victim_tag"] == predictions
        prediction_frames.append(prediction_frame)

        metric_rows.append(
            {
                "scheduler_policy": policy,
                "method": "random_forest",
                "sample_count": int(len(frame)),
                "class_count": int(frame["victim_tag"].nunique()),
                "chance_accuracy": 1.0 / frame["victim_tag"].nunique(),
                "accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predictions)
                ),
                "macro_f1": float(f1_score(labels, predictions, average="macro")),
            }
        )

        classifier.fit(matrix, labels)
        importance_frames.append(
            pd.DataFrame(
                {
                    "scheduler_policy": policy,
                    "feature": feature_columns,
                    "importance": classifier.feature_importances_,
                }
            )
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows)
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame()
    )
    return predictions, metrics, importance


# =============================================================================
# Plotting
# =============================================================================


def save_policy_plot(
    policy_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    data = policy_summary.set_index("scheduler_policy").reindex(SCHEDULER_POLICIES)
    axis = data[metric].plot(kind="bar", figsize=(14, 6))
    axis.set_xlabel("Remote-operation scheduler")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def save_classification_plot(
    nonml_metrics: pd.DataFrame,
    rf_metrics: pd.DataFrame,
) -> None:
    frames = []
    if not nonml_metrics.empty:
        frames.append(nonml_metrics[["scheduler_policy", "method", "accuracy"]])
    if not rf_metrics.empty and "accuracy" in rf_metrics.columns:
        frames.append(rf_metrics[["scheduler_policy", "method", "accuracy"]])
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot(
        index="scheduler_policy",
        columns="method",
        values="accuracy",
    ).reindex(SCHEDULER_POLICIES)

    axis = pivot.plot(kind="bar", figsize=(15, 6))
    axis.axhline(
        1.0 / max(1, len(VICTIM_QASMS)),
        linestyle="--",
        linewidth=1,
        label="Chance",
    )
    axis.set_xlabel("Remote-operation scheduler")
    axis.set_ylabel("Victim-workload classification accuracy")
    axis.set_title("Black-Box Fingerprint Classification Across Schedulers")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "remote_scheduler_classification_accuracy.png",
        dpi=300,
    )
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subfolder in [
        "core_policy_capacity_tenancy",
        "queue_communication_capacity",
        "decision_interval_lookahead",
        "preemption_prefetch_priority",
    ]:
        (OUTPUT_DIR / subfolder).mkdir(parents=True, exist_ok=True)

    configurations = generate_configurations()
    qasm_paths = [p1.resolve_qasm(filename) for filename in VICTIM_QASMS]

    print("Phase 1.4 — Remote-operation scheduler policies")
    print(f"Configurations: {len(configurations)}")
    print(f"Trials per configuration/workload: {TRIALS_PER_CONFIGURATION}")
    print(f"Victim workloads: {len(qasm_paths)}")
    print(
        "Total physical trial tuples: "
        f"{len(configurations) * len(qasm_paths) * TRIALS_PER_CONFIGURATION}"
    )

    trial_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    attacker_frames: list[pd.DataFrame] = []
    request_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []

    total = len(configurations) * len(qasm_paths) * TRIALS_PER_CONFIGURATION
    completed = 0

    for configuration in configurations:
        for qasm_path in qasm_paths:
            for trial_id in range(TRIALS_PER_CONFIGURATION):
                completed += 1
                print(
                    f"[{completed:05d}/{total:05d}] "
                    f"{configuration.subexperiment} | "
                    f"{configuration.scheduler_policy} | "
                    f"{qasm_path.name} | trial {trial_id}"
                )

                (
                    trial_summary,
                    feature_row,
                    attacker_comparison,
                    trial_request_frames,
                    trial_decision_frames,
                ) = run_one_trial(
                    configuration,
                    qasm_path,
                    trial_id,
                )

                trial_rows.append(trial_summary)
                feature_rows.append(feature_row)
                if SAVE_ATTACKER_COMPARISON:
                    attacker_frames.append(attacker_comparison)
                if SAVE_REQUEST_LOG:
                    request_frames.extend(trial_request_frames)
                if SAVE_SCHEDULER_DECISION_LOG:
                    decision_frames.extend(trial_decision_frames)

    trials = pd.DataFrame(trial_rows)
    features = pd.DataFrame(feature_rows)

    trials.to_csv(
        OUTPUT_DIR / "remote_scheduler_trial_summary.csv",
        index=False,
    )
    features.to_csv(
        OUTPUT_DIR / "remote_scheduler_fingerprint_features.csv",
        index=False,
    )

    if SAVE_ATTACKER_COMPARISON and attacker_frames:
        pd.concat(attacker_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "remote_scheduler_attacker_comparison.csv.gz",
            index=False,
            compression="gzip",
        )
    if SAVE_REQUEST_LOG and request_frames:
        pd.concat(request_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "remote_scheduler_request_log.csv.gz",
            index=False,
            compression="gzip",
        )
    if SAVE_SCHEDULER_DECISION_LOG and decision_frames:
        pd.concat(decision_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "remote_scheduler_decision_log.csv.gz",
            index=False,
            compression="gzip",
        )

    summaries = build_all_summaries(trials)
    output_names = {
        "policy": "remote_scheduler_policy_summary.csv",
        "capacity": "remote_scheduler_capacity_summary.csv",
        "queue": "remote_scheduler_queue_summary.csv",
        "tenancy": "remote_scheduler_tenancy_summary.csv",
        "communication_qubit": "remote_scheduler_communication_qubit_summary.csv",
        "priority": "remote_scheduler_priority_summary.csv",
        "decision_interval": "remote_scheduler_decision_interval_summary.csv",
        "lookahead": "remote_scheduler_lookahead_summary.csv",
        "preemption": "remote_scheduler_preemption_summary.csv",
        "prefetch": "remote_scheduler_prefetch_summary.csv",
        "phase_visibility": "remote_scheduler_phase_visibility_summary.csv",
    }
    for key, dataframe in summaries.items():
        dataframe.to_csv(OUTPUT_DIR / output_names[key], index=False)

    for subexperiment, frame in trials.groupby("subexperiment"):
        subdir = OUTPUT_DIR / subexperiment
        frame.to_csv(subdir / "trial_summary.csv", index=False)
        aggregate_summary(
            frame,
            [
                "scheduler_policy",
                "queue_depth",
                "tenant_count",
                "hub_capacity",
                "link_capacity",
                "communication_qubits_per_module",
                "priority_profile",
                "decision_interval_ns",
                "lookahead_window_ns",
                "preemption_allowed",
                "epr_prefetch_enabled",
            ],
        ).to_csv(subdir / "configuration_summary.csv", index=False)

    stability = fingerprint_stability(features)
    stability.to_csv(
        OUTPUT_DIR / "remote_scheduler_fingerprint_stability.csv",
        index=False,
    )

    nonml_predictions, nonml_metrics = nonml_nearest_template_classification(features)
    nonml_predictions.to_csv(
        OUTPUT_DIR / "remote_scheduler_nonml_classification_predictions.csv",
        index=False,
    )
    nonml_metrics.to_csv(
        OUTPUT_DIR / "remote_scheduler_nonml_classification_metrics.csv",
        index=False,
    )

    rf_predictions, rf_metrics, feature_importance = random_forest_classification(features)
    rf_predictions.to_csv(
        OUTPUT_DIR / "remote_scheduler_random_forest_predictions.csv",
        index=False,
    )
    rf_metrics.to_csv(
        OUTPUT_DIR / "remote_scheduler_random_forest_metrics.csv",
        index=False,
    )
    feature_importance.to_csv(
        OUTPUT_DIR / "remote_scheduler_feature_importance.csv",
        index=False,
    )

    policy_summary = summaries["policy"]
    save_policy_plot(
        policy_summary,
        "mean_absolute_attacker_change_ns",
        "Mean absolute attacker timing change (ns)",
        "Scheduler Policy: Black-Box Timing Leakage",
        "remote_scheduler_policy_leakage.png",
    )
    save_policy_plot(
        policy_summary,
        "mean_phase_boundary_visibility",
        "Phase-boundary visibility fraction",
        "Scheduler Policy: Victim Phase-Boundary Visibility",
        "remote_scheduler_phase_visibility.png",
    )
    save_policy_plot(
        policy_summary,
        "scheduler_reordering_rate",
        "Release-to-service reordering rate",
        "Scheduler Policy: Remote-Operation Reordering",
        "remote_scheduler_reordering_rate.png",
    )
    save_policy_plot(
        policy_summary,
        "tenant_wait_fairness",
        "Jain fairness of inverse tenant waiting time",
        "Scheduler Policy: Tenant Waiting-Time Fairness",
        "remote_scheduler_fairness.png",
    )
    save_classification_plot(nonml_metrics, rf_metrics)

    print("\n=== Policy summary ===")
    display_columns = [
        "scheduler_policy",
        "mean_remote_latency_ns",
        "p95_remote_latency_ns",
        "mean_absolute_attacker_change_ns",
        "mean_observable_change_fraction",
        "mean_phase_boundary_visibility",
        "scheduler_reordering_rate",
        "tenant_wait_fairness",
        "deadline_miss_fraction",
        "hub_utilization",
        "mean_epr_expiration_fraction",
    ]
    print(policy_summary[display_columns].to_string(index=False))

    print("\n=== Workload classification ===")
    if not nonml_metrics.empty:
        print(nonml_metrics.to_string(index=False))
    if not rf_metrics.empty:
        print(rf_metrics.to_string(index=False))

    print(f"\nSaved Phase 1.4 results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
