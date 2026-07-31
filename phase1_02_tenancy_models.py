#!/usr/bin/env python3
"""
phase1_02_tenancy_models.py

Experiment 1.2 — Tenancy and module-sharing models.

Research question
-----------------
Under what forms of multi-tenancy can endpoint occupancy and node-side timing
leakage actually arise?

This experiment reuses the victim logical partitions and Probe-3 attacker from
Phase 1.1, but holds placement in a P2-like form whenever the tenancy model
permits it:

    victim  -> modules 0, 1, 2
    attacker -> modules 2, 3

Module-exclusive tenancy cannot legally preserve that overlap, so the attacker
is relocated to modules 3 and 4. This is recorded as an isolation adjustment,
not silently treated as P2.

The simulator retains the Phase 1.1 hub actions:
- request arrival and buffering;
- FCFS arbitration with bypass for independent feasible requests;
- finite hub admission capacity;
- switch-path locking;
- resource acquisition;
- service progression and completion;
- resource release and queued-request re-admission;
- arrival, waiting, service, completion, and turnaround logging.

It additionally models node-side resources:
- module-wide endpoint locks;
- communication-qubit-specific locks;
- shared or tenant-dedicated communication-qubit assignments;
- local routing paths;
- reset/readout/feedforward pipelines;
- spatial-isolation guard-qubit overhead;
- alternating time slices on a shared module.

Two physical executions are run for every accepted configuration:
1. hub4: the requested four-slot hub;
2. node_only: an effectively unbounded hub with pair-specific switch paths.

The node_only result is the primary measurement of leakage with global hub
contention removed.

Required companion file
-----------------------
phase1_01_job_module_allocation.py

Primary outputs
---------------
blackbox_window_results/phase1_02_tenancy_models/
    tenancy_model_trial_summary.csv
    tenancy_model_attacker_observations.csv          [optional]
    tenancy_model_summary.csv
    tenancy_model_endpoint_wait_summary.csv
    tenancy_model_resource_blocking_summary.csv
    tenancy_model_resource_effect_summary.csv
    tenancy_model_admission_summary.csv
    tenancy_model_utilization_summary.csv
    tenancy_model_configuration_summary.csv

    tenancy_model_node_only_leakage.png
    tenancy_model_endpoint_wait.png
    tenancy_model_rejection_rate.png
    tenancy_model_isolation_overhead.png
    tenancy_model_resource_blocking.png

Execution
---------
Set the run controls in the "Integrated run controls" section below, then run:

    python phase1_02_tenancy_models.py

No command-line flags are required.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase1_01_job_module_allocation as p1


# =============================================================================
# Experiment configuration
# =============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_02_tenancy_models"
)

TENANCY_MODELS = [
    "module_exclusive",
    "spatial_qubit_partitioning",
    "shared_communication_interface",
    "dedicated_communication_interface",
    "time_sliced_module_sharing",
    "hybrid_shared_pipeline",
]

COMPUTE_QUBITS_PER_MODULE_OPTIONS = [8, 16, 32]
COMMUNICATION_QUBITS_PER_MODULE_OPTIONS = [1, 2, 4]
COMMUNICATION_QUBITS_PER_TENANT_OPTIONS = [1, 2]
SPATIAL_ISOLATION_OPTIONS = [0.0, 0.5, 1.0]
LOCAL_ROUTING_SHARED_OPTIONS = [False, True]
RESET_PIPELINE_SHARED_OPTIONS = [False, True]
ENDPOINT_LOCK_SCOPE_OPTIONS = [
    "module_wide",
    "communication_qubit_specific",
]

NUM_MODULES = 5
VICTIM_MODULES = (0, 1, 2)
P2_ATTACKER_MODULES = (2, 3)
EXCLUSIVE_ATTACKER_MODULES = (3, 4)

HUB_CAPACITY = 4
NODE_ONLY_HUB_CAPACITY = 10_000

REMOTE_SERVICE_TIME_NS = p1.REMOTE_SERVICE_TIME_NS
VICTIM_START_NS = p1.VICTIM_START_NS
ATTACKER_START_NS = p1.ATTACKER_START_NS

# Alternating victim/attacker ownership on a shared module.
TIME_SLICE_NS = 1_000
TIME_SLICE_EPOCH_NS = min(VICTIM_START_NS, ATTACKER_START_NS)

# Guard space reserved around spatial compute partitions.
ISOLATION_GUARD_FRACTION = 0.25

DETECTION_THRESHOLD_NS = 0.0
GLOBAL_SEED = 20260730

# =============================================================================
# Integrated run controls
# =============================================================================
# Change these values here; no terminal flags are needed.
#
# RUN_QUICK_VALIDATION = False runs the complete factorial experiment.
# RUN_QUICK_VALIDATION = True runs a small two-workload validation subset.
RUN_QUICK_VALIDATION = False

# Number of repeated trials for every workload/configuration pair.
TRIALS_PER_CONFIGURATION = 3

# Raw request-level output can contain millions of rows in the complete sweep.
# All aggregate and trial-level result files are saved regardless of this value.
SAVE_REQUEST_LEVEL_RESULTS = False

# Optional cap for debugging. Keep None for the complete configuration set.
MAX_CONFIGURATIONS = None

RESOURCE_CATEGORIES = [
    "time_slice",
    "hub",
    "switch_path",
    "endpoint_module",
    "communication_qubit",
    "local_routing",
    "reset_pipeline",
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class TenancyConfiguration:
    tenancy_model: str
    compute_qubits_per_module: int
    communication_qubits_per_module: int
    communication_qubits_per_tenant: int
    spatial_isolation_degree: float
    local_routing_paths_shared: bool
    reset_pipeline_shared: bool
    endpoint_lock_scope: str

    @property
    def configuration_id(self) -> str:
        isolation = str(self.spatial_isolation_degree).replace(".", "p")
        route = "route_shared" if self.local_routing_paths_shared else "route_dedicated"
        reset = "reset_shared" if self.reset_pipeline_shared else "reset_dedicated"
        return (
            f"{self.tenancy_model}__"
            f"cq{self.compute_qubits_per_module}__"
            f"mq{self.communication_qubits_per_module}__"
            f"mqt{self.communication_qubits_per_tenant}__"
            f"iso{isolation}__{route}__{reset}__"
            f"{self.endpoint_lock_scope}"
        )


@dataclass
class TenancyAllocation:
    accepted: bool
    victim_allocation: p1.Allocation
    attacker_allocation: p1.Allocation
    rejection_reason: str = ""
    placement_adjusted: bool = False
    shared_modules: tuple[int, ...] = field(default_factory=tuple)
    compute_regions: dict[tuple[str, int], int] = field(default_factory=dict)
    guard_qubits: dict[tuple[str, int], int] = field(default_factory=dict)
    communication_assignments: dict[tuple[str, int], tuple[int, ...]] = field(
        default_factory=dict
    )
    route_domains: dict[tuple[str, int], str] = field(default_factory=dict)
    reset_domains: dict[tuple[str, int], str] = field(default_factory=dict)
    effective_communication_mode: str = ""
    actual_communication_qubit_overlap: bool = False
    actual_local_route_overlap: bool = False
    actual_reset_pipeline_overlap: bool = False
    module_wide_endpoint_overlap: bool = False
    compute_qubits_allocated: int = 0
    compute_guard_qubits: int = 0
    communication_qubits_reserved_unique: int = 0
    active_module_count: int = 0


@dataclass(frozen=True)
class NodeRequest:
    request_id: int
    tenant_id: str
    role: str
    logical_event_id: int
    release_time_ns: int
    eligible_time_ns: int
    source_module: int
    target_module: int
    switch_path: tuple[int, int]
    resource_keys: tuple[str, ...]

    @property
    def endpoints(self) -> tuple[int, int]:
        return (
            min(self.source_module, self.target_module),
            max(self.source_module, self.target_module),
        )


@dataclass
class WaitingState:
    request: NodeRequest
    blocked_time_ns: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    blocker_checks: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )


@dataclass
class ActiveNodeRequest:
    request: NodeRequest
    start_time_ns: int
    completion_time_ns: int
    serial: int
    blocked_time_ns: dict[str, int]
    blocker_checks: dict[str, int]

    def heap_item(self) -> tuple[int, int, "ActiveNodeRequest"]:
        return (
            self.completion_time_ns,
            self.serial,
            self,
        )


@dataclass
class CompletedNodeRequest:
    request: NodeRequest
    arrival_time_ns: int
    service_start_time_ns: int
    completion_time_ns: int
    blocked_time_ns: dict[str, int]
    blocker_checks: dict[str, int]

    @property
    def waiting_time_ns(self) -> int:
        return self.service_start_time_ns - self.arrival_time_ns

    @property
    def turnaround_time_ns(self) -> int:
        return self.completion_time_ns - self.arrival_time_ns


# =============================================================================
# Basic helpers
# =============================================================================

def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    value = GLOBAL_SEED
    for character in text:
        value = (value * 131 + ord(character)) % (2**32 - 1)
    return value


def partition_sizes(job: p1.JobSpec) -> dict[int, int]:
    sizes: dict[int, int] = defaultdict(int)
    for partition in job.partition_of_qubit.values():
        sizes[partition] += 1
    return dict(sizes)


def role_priority(role: str) -> int:
    return {
        "victim": 0,
        "attacker": 1,
    }.get(role, 2)


def communication_mode_for_model(model: str) -> str:
    if model == "module_exclusive":
        return "dedicated"
    if model == "spatial_qubit_partitioning":
        return "independent_pool_selection"
    if model == "shared_communication_interface":
        return "shared"
    if model == "dedicated_communication_interface":
        return "dedicated"
    if model == "time_sliced_module_sharing":
        return "shared"
    if model == "hybrid_shared_pipeline":
        return "dedicated"
    raise ValueError(f"Unsupported tenancy model: {model}")


def physical_pair_path(first: int, second: int) -> tuple[int, int]:
    return (
        min(first, second),
        max(first, second),
    )


def request_touches_module(request: NodeRequest, module: int) -> bool:
    return module in request.endpoints


def next_owned_time(
    release_time_ns: int,
    role: str,
) -> int:
    """Return the earliest alternating slot owned by the role."""

    if release_time_ns < TIME_SLICE_EPOCH_NS:
        release_time_ns = TIME_SLICE_EPOCH_NS

    slot_index = (
        release_time_ns - TIME_SLICE_EPOCH_NS
    ) // TIME_SLICE_NS

    owner = "victim" if slot_index % 2 == 0 else "attacker"
    if owner == role:
        return release_time_ns

    return (
        TIME_SLICE_EPOCH_NS
        + (slot_index + 1) * TIME_SLICE_NS
    )


# =============================================================================
# P2-like placement and tenancy admission
# =============================================================================

def fixed_partition_mapping(
    job: p1.JobSpec,
    modules: Sequence[int],
) -> dict[int, int]:
    partitions = sorted(job.partitions)
    if len(partitions) != len(modules):
        raise ValueError(
            f"Job {job.tenant_id} has {len(partitions)} partitions but "
            f"received {len(modules)} modules."
        )
    return dict(zip(partitions, modules))


def build_base_allocations(
    victim_job: p1.JobSpec,
    attacker_job: p1.JobSpec,
    model: str,
) -> tuple[p1.Allocation, p1.Allocation, bool]:
    victim_allocation = p1.Allocation(
        tenant_id="victim",
        role="victim",
        accepted=True,
        requested_modules=victim_job.modules_requested,
        partition_to_module=fixed_partition_mapping(
            victim_job,
            VICTIM_MODULES,
        ),
    )

    if model == "module_exclusive":
        attacker_modules = EXCLUSIVE_ATTACKER_MODULES
        adjusted = True
    else:
        attacker_modules = P2_ATTACKER_MODULES
        adjusted = False

    attacker_allocation = p1.Allocation(
        tenant_id="attacker",
        role="attacker",
        accepted=True,
        requested_modules=attacker_job.modules_requested,
        partition_to_module=fixed_partition_mapping(
            attacker_job,
            attacker_modules,
        ),
    )

    return (
        victim_allocation,
        attacker_allocation,
        adjusted,
    )


def compute_region_requirements(
    job: p1.JobSpec,
    allocation: p1.Allocation,
) -> dict[tuple[str, int], int]:
    sizes = partition_sizes(job)
    requirements: dict[tuple[str, int], int] = defaultdict(int)
    for partition, module in allocation.partition_to_module.items():
        requirements[(job.tenant_id, module)] += sizes[partition]
    return dict(requirements)


def guard_requirement(
    region_size: int,
    isolation_degree: float,
) -> int:
    if isolation_degree <= 0:
        return 0
    return int(
        math.ceil(
            region_size
            * isolation_degree
            * ISOLATION_GUARD_FRACTION
        )
    )


def choose_subset(
    population: Sequence[int],
    count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if count > len(population):
        raise ValueError("Requested subset exceeds population.")
    return tuple(sorted(rng.sample(list(population), count)))


def assign_communication_qubits(
    *,
    config: TenancyConfiguration,
    modules_by_tenant: dict[str, set[int]],
    rng: random.Random,
) -> tuple[
    bool,
    str,
    dict[tuple[str, int], tuple[int, ...]],
    str,
]:
    """Assign shared, pooled, or dedicated communication qubits."""

    mode = communication_mode_for_model(
        config.tenancy_model
    )
    total = config.communication_qubits_per_module
    requested = config.communication_qubits_per_tenant

    if requested > total:
        return (
            False,
            "communication_qubits_per_tenant_exceeds_module_capacity",
            {},
            mode,
        )

    assignments: dict[tuple[str, int], tuple[int, ...]] = {}
    all_modules = sorted(
        set().union(*modules_by_tenant.values())
    )

    for module in all_modules:
        tenants = [
            tenant
            for tenant, modules in modules_by_tenant.items()
            if module in modules
        ]
        pool = tuple(range(total))

        if mode == "dedicated":
            if len(tenants) * requested > total:
                return (
                    False,
                    "insufficient_dedicated_communication_qubits",
                    {},
                    mode,
                )
            shuffled = list(pool)
            rng.shuffle(shuffled)
            cursor = 0
            for tenant in sorted(tenants):
                assignments[(tenant, module)] = tuple(
                    sorted(shuffled[cursor : cursor + requested])
                )
                cursor += requested

        elif mode == "shared":
            shared_subset = choose_subset(
                pool,
                requested,
                rng,
            )
            for tenant in tenants:
                assignments[(tenant, module)] = shared_subset

        elif mode == "independent_pool_selection":
            for tenant in sorted(tenants):
                assignments[(tenant, module)] = choose_subset(
                    pool,
                    requested,
                    rng,
                )

        else:
            raise ValueError(f"Unsupported communication mode: {mode}")

    return (
        True,
        "",
        assignments,
        mode,
    )


def build_route_domains(
    *,
    config: TenancyConfiguration,
    modules_by_tenant: dict[str, set[int]],
    rng: random.Random,
) -> dict[tuple[str, int], str]:
    domains: dict[tuple[str, int], str] = {}

    for module in sorted(
        set().union(*modules_by_tenant.values())
    ):
        tenants = sorted(
            tenant
            for tenant, modules in modules_by_tenant.items()
            if module in modules
        )

        share_route = False
        if (
            config.local_routing_paths_shared
            and len(tenants) > 1
        ):
            # Spatial isolation progressively separates local routing domains.
            share_probability = max(
                0.0,
                1.0 - config.spatial_isolation_degree,
            )
            share_route = rng.random() < share_probability

        for tenant in tenants:
            if share_route:
                domains[(tenant, module)] = f"route:m{module}:shared"
            else:
                domains[(tenant, module)] = (
                    f"route:m{module}:tenant:{tenant}"
                )

    return domains


def build_reset_domains(
    *,
    config: TenancyConfiguration,
    modules_by_tenant: dict[str, set[int]],
) -> dict[tuple[str, int], str]:
    domains: dict[tuple[str, int], str] = {}

    # The hybrid model explicitly retains shared reset/readout/feedforward.
    effective_shared = (
        config.reset_pipeline_shared
        or config.tenancy_model == "hybrid_shared_pipeline"
    )

    for module in sorted(
        set().union(*modules_by_tenant.values())
    ):
        tenants = sorted(
            tenant
            for tenant, modules in modules_by_tenant.items()
            if module in modules
        )
        for tenant in tenants:
            if effective_shared and len(tenants) > 1:
                domains[(tenant, module)] = (
                    f"reset:m{module}:shared"
                )
            else:
                domains[(tenant, module)] = (
                    f"reset:m{module}:tenant:{tenant}"
                )

    return domains


def evaluate_tenancy_allocation(
    *,
    victim_job: p1.JobSpec,
    attacker_job: p1.JobSpec,
    config: TenancyConfiguration,
    trial_id: int,
) -> TenancyAllocation:
    rng = random.Random(
        stable_seed(
            config.configuration_id,
            trial_id,
            victim_job.logical_qubits,
        )
    )

    (
        victim_allocation,
        attacker_allocation,
        placement_adjusted,
    ) = build_base_allocations(
        victim_job,
        attacker_job,
        config.tenancy_model,
    )

    victim_regions = compute_region_requirements(
        victim_job,
        victim_allocation,
    )
    attacker_regions = compute_region_requirements(
        attacker_job,
        attacker_allocation,
    )
    compute_regions = {
        **victim_regions,
        **attacker_regions,
    }

    modules_by_tenant = {
        "victim": set(victim_allocation.assigned_modules),
        "attacker": set(attacker_allocation.assigned_modules),
    }
    shared_modules = tuple(
        sorted(
            modules_by_tenant["victim"]
            & modules_by_tenant["attacker"]
        )
    )

    guard_qubits: dict[tuple[str, int], int] = {}
    for key, region_size in compute_regions.items():
        tenant, module = key
        if module in shared_modules:
            guard_qubits[key] = guard_requirement(
                region_size,
                config.spatial_isolation_degree,
            )
        else:
            guard_qubits[key] = 0

    # Compute admission. Time slicing reuses one physical region over time,
    # whereas spatial models reserve both regions concurrently.
    for module in range(NUM_MODULES):
        occupants = [
            key
            for key in compute_regions
            if key[1] == module
        ]
        if not occupants:
            continue

        if config.tenancy_model == "time_sliced_module_sharing" and len(occupants) > 1:
            required = max(
                compute_regions[key]
                + guard_qubits[key]
                for key in occupants
            )
        else:
            required = sum(
                compute_regions[key]
                + guard_qubits[key]
                for key in occupants
            )

        if required > config.compute_qubits_per_module:
            return TenancyAllocation(
                accepted=False,
                victim_allocation=victim_allocation,
                attacker_allocation=attacker_allocation,
                rejection_reason=(
                    f"compute_capacity_exceeded_module_{module}"
                ),
                placement_adjusted=placement_adjusted,
                shared_modules=shared_modules,
                compute_regions=compute_regions,
                guard_qubits=guard_qubits,
            )

    (
        comm_accepted,
        comm_reason,
        communication_assignments,
        communication_mode,
    ) = assign_communication_qubits(
        config=config,
        modules_by_tenant=modules_by_tenant,
        rng=rng,
    )

    if not comm_accepted:
        return TenancyAllocation(
            accepted=False,
            victim_allocation=victim_allocation,
            attacker_allocation=attacker_allocation,
            rejection_reason=comm_reason,
            placement_adjusted=placement_adjusted,
            shared_modules=shared_modules,
            compute_regions=compute_regions,
            guard_qubits=guard_qubits,
            effective_communication_mode=communication_mode,
        )

    route_domains = build_route_domains(
        config=config,
        modules_by_tenant=modules_by_tenant,
        rng=rng,
    )
    reset_domains = build_reset_domains(
        config=config,
        modules_by_tenant=modules_by_tenant,
    )

    comm_overlap = False
    route_overlap = False
    reset_overlap = False
    module_lock_overlap = False

    for module in shared_modules:
        victim_comm = set(
            communication_assignments.get(
                ("victim", module),
                (),
            )
        )
        attacker_comm = set(
            communication_assignments.get(
                ("attacker", module),
                (),
            )
        )
        comm_overlap = comm_overlap or bool(
            victim_comm & attacker_comm
        )

        route_overlap = route_overlap or (
            route_domains.get(("victim", module))
            == route_domains.get(("attacker", module))
        )

        reset_overlap = reset_overlap or (
            reset_domains.get(("victim", module))
            == reset_domains.get(("attacker", module))
        )

        module_lock_overlap = module_lock_overlap or (
            config.endpoint_lock_scope == "module_wide"
        )

    unique_comm_resources = {
        (module, qid)
        for (_, module), qids in communication_assignments.items()
        for qid in qids
    }

    return TenancyAllocation(
        accepted=True,
        victim_allocation=victim_allocation,
        attacker_allocation=attacker_allocation,
        rejection_reason="",
        placement_adjusted=placement_adjusted,
        shared_modules=shared_modules,
        compute_regions=compute_regions,
        guard_qubits=guard_qubits,
        communication_assignments=communication_assignments,
        route_domains=route_domains,
        reset_domains=reset_domains,
        effective_communication_mode=communication_mode,
        actual_communication_qubit_overlap=comm_overlap,
        actual_local_route_overlap=route_overlap,
        actual_reset_pipeline_overlap=reset_overlap,
        module_wide_endpoint_overlap=module_lock_overlap,
        compute_qubits_allocated=sum(compute_regions.values()),
        compute_guard_qubits=sum(guard_qubits.values()),
        communication_qubits_reserved_unique=len(unique_comm_resources),
        active_module_count=len(
            modules_by_tenant["victim"]
            | modules_by_tenant["attacker"]
        ),
    )


# =============================================================================
# Node-side request construction
# =============================================================================

def selected_comm_qubit(
    allocation: TenancyAllocation,
    tenant_id: str,
    module: int,
    logical_event_id: int,
) -> int:
    assigned = allocation.communication_assignments.get(
        (tenant_id, module)
    )
    if not assigned:
        raise KeyError(
            f"No communication-qubit assignment for {tenant_id} on module {module}."
        )
    return assigned[
        logical_event_id % len(assigned)
    ]


def resource_keys_for_request(
    *,
    tenant_id: str,
    source_module: int,
    target_module: int,
    logical_event_id: int,
    config: TenancyConfiguration,
    allocation: TenancyAllocation,
) -> tuple[str, ...]:
    keys: list[str] = []

    for module in (
        source_module,
        target_module,
    ):
        if config.endpoint_lock_scope == "module_wide":
            keys.append(
                f"endpoint_module:m{module}"
            )

        comm_qubit = selected_comm_qubit(
            allocation,
            tenant_id,
            module,
            logical_event_id,
        )
        keys.append(
            f"communication_qubit:m{module}:q{comm_qubit}"
        )

        keys.append(
            allocation.route_domains[
                (tenant_id, module)
            ].replace(
                "route:",
                "local_routing:",
                1,
            )
        )

        keys.append(
            allocation.reset_domains[
                (tenant_id, module)
            ].replace(
                "reset:",
                "reset_pipeline:",
                1,
            )
        )

    return tuple(sorted(set(keys)))


def build_node_requests(
    *,
    jobs: Iterable[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    tenancy_allocation: TenancyAllocation,
    config: TenancyConfiguration,
    combined_execution: bool,
) -> list[NodeRequest]:
    requests: list[NodeRequest] = []
    request_id = 0

    for job in jobs:
        allocation = allocations[job.tenant_id]
        for event in job.logical_events:
            touched_partitions = {
                job.partition_of_qubit[qubit]
                for qubit in event.qubits
                if qubit in job.partition_of_qubit
            }
            if len(touched_partitions) < 2:
                continue

            for left_partition, right_partition in itertools.combinations(
                sorted(touched_partitions),
                2,
            ):
                source = allocation.partition_to_module[left_partition]
                target = allocation.partition_to_module[right_partition]
                if source == target:
                    continue

                release_time = (
                    job.start_time_ns
                    + event.release_offset_ns
                )
                eligible_time = release_time

                if (
                    combined_execution
                    and config.tenancy_model
                    == "time_sliced_module_sharing"
                    and 2 in (source, target)
                ):
                    eligible_time = next_owned_time(
                        release_time,
                        job.role,
                    )

                requests.append(
                    NodeRequest(
                        request_id=request_id,
                        tenant_id=job.tenant_id,
                        role=job.role,
                        logical_event_id=event.event_id,
                        release_time_ns=release_time,
                        eligible_time_ns=eligible_time,
                        source_module=source,
                        target_module=target,
                        switch_path=physical_pair_path(
                            source,
                            target,
                        ),
                        resource_keys=resource_keys_for_request(
                            tenant_id=job.tenant_id,
                            source_module=source,
                            target_module=target,
                            logical_event_id=event.event_id,
                            config=config,
                            allocation=tenancy_allocation,
                        ),
                    )
                )
                request_id += 1

    return sorted(
        requests,
        key=lambda request: (
            request.release_time_ns,
            role_priority(request.role),
            request.tenant_id,
            request.logical_event_id,
            request.request_id,
        ),
    )


# =============================================================================
# Hub and node-resource simulator
# =============================================================================

def resource_category(resource_key: str) -> str:
    return resource_key.split(":", 1)[0]


class NodeAwareHubSimulator:
    """Hub simulator with explicit node-side locks and wait attribution."""

    def __init__(
        self,
        *,
        hub_capacity: int,
    ) -> None:
        if hub_capacity <= 0:
            raise ValueError("hub_capacity must be positive.")

        self.hub_capacity = hub_capacity
        self.current_time_ns = 0
        self.waiting: deque[WaitingState] = deque()
        self.active_heap: list[
            tuple[int, int, ActiveNodeRequest]
        ] = []
        self.active_switch_paths: set[tuple[int, int]] = set()
        self.active_resources: set[str] = set()
        self.completed: list[CompletedNodeRequest] = []
        self.serial = 0

    def blockers(
        self,
        waiting_state: WaitingState,
    ) -> set[str]:
        request = waiting_state.request
        blockers: set[str] = set()

        if self.current_time_ns < request.eligible_time_ns:
            blockers.add("time_slice")

        if len(self.active_heap) >= self.hub_capacity:
            blockers.add("hub")

        if request.switch_path in self.active_switch_paths:
            blockers.add("switch_path")

        for key in request.resource_keys:
            if key in self.active_resources:
                blockers.add(
                    resource_category(key)
                )

        return blockers

    def can_start(
        self,
        waiting_state: WaitingState,
    ) -> bool:
        return not self.blockers(
            waiting_state
        )

    def start(
        self,
        waiting_state: WaitingState,
    ) -> None:
        request = waiting_state.request
        start_time = max(
            self.current_time_ns,
            request.release_time_ns,
            request.eligible_time_ns,
        )
        completion_time = (
            start_time
            + REMOTE_SERVICE_TIME_NS
        )

        self.active_switch_paths.add(
            request.switch_path
        )
        self.active_resources.update(
            request.resource_keys
        )

        active = ActiveNodeRequest(
            request=request,
            start_time_ns=start_time,
            completion_time_ns=completion_time,
            serial=self.serial,
            blocked_time_ns=dict(
                waiting_state.blocked_time_ns
            ),
            blocker_checks=dict(
                waiting_state.blocker_checks
            ),
        )
        self.serial += 1
        heapq.heappush(
            self.active_heap,
            active.heap_item(),
        )

    def complete_all_at_current_time(self) -> None:
        while (
            self.active_heap
            and self.active_heap[0][0]
            <= self.current_time_ns
        ):
            _, _, active = heapq.heappop(
                self.active_heap
            )
            self.active_switch_paths.remove(
                active.request.switch_path
            )
            for key in active.request.resource_keys:
                self.active_resources.remove(key)

            self.completed.append(
                CompletedNodeRequest(
                    request=active.request,
                    arrival_time_ns=(
                        active.request.release_time_ns
                    ),
                    service_start_time_ns=(
                        active.start_time_ns
                    ),
                    completion_time_ns=(
                        active.completion_time_ns
                    ),
                    blocked_time_ns=(
                        active.blocked_time_ns
                    ),
                    blocker_checks=(
                        active.blocker_checks
                    ),
                )
            )

    def admit_waiting(self) -> bool:
        if not self.waiting:
            return False

        for index, waiting_state in enumerate(
            self.waiting
        ):
            blockers = self.blockers(
                waiting_state
            )
            for blocker in blockers:
                waiting_state.blocker_checks[
                    blocker
                ] += 1

            if not blockers:
                del self.waiting[index]
                self.start(waiting_state)
                return True

        return False

    def account_wait_interval(
        self,
        delta_ns: int,
    ) -> None:
        if delta_ns <= 0:
            return
        for waiting_state in self.waiting:
            blockers = self.blockers(
                waiting_state
            )
            for blocker in blockers:
                waiting_state.blocked_time_ns[
                    blocker
                ] += delta_ns

    def run(
        self,
        requests: Sequence[NodeRequest],
    ) -> list[CompletedNodeRequest]:
        requests = list(requests)
        release_index = 0

        if requests:
            self.current_time_ns = min(
                0,
                requests[0].release_time_ns,
            )

        while (
            release_index < len(requests)
            or self.waiting
            or self.active_heap
        ):
            while (
                release_index < len(requests)
                and requests[release_index].release_time_ns
                <= self.current_time_ns
            ):
                self.waiting.append(
                    WaitingState(
                        request=requests[
                            release_index
                        ]
                    )
                )
                release_index += 1

            self.complete_all_at_current_time()

            progress = True
            while progress:
                progress = self.admit_waiting()

            if (
                release_index >= len(requests)
                and not self.waiting
                and not self.active_heap
            ):
                break

            next_release = (
                requests[release_index].release_time_ns
                if release_index < len(requests)
                else math.inf
            )
            next_completion = (
                self.active_heap[0][0]
                if self.active_heap
                else math.inf
            )
            next_eligibility = min(
                (
                    state.request.eligible_time_ns
                    for state in self.waiting
                    if state.request.eligible_time_ns
                    > self.current_time_ns
                ),
                default=math.inf,
            )

            next_time = min(
                next_release,
                next_completion,
                next_eligibility,
            )

            if math.isinf(next_time):
                raise RuntimeError(
                    "Simulator deadlock: no future event can release a blocked request."
                )

            if next_time <= self.current_time_ns:
                next_time = self.current_time_ns + 1

            delta = int(next_time - self.current_time_ns)
            self.account_wait_interval(delta)
            self.current_time_ns = int(next_time)

        return self.completed


# =============================================================================
# Dataframe conversion and baseline subtraction
# =============================================================================

def completed_dataframe(
    completed: Sequence[CompletedNodeRequest],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in completed:
        request = item.request
        row: dict[str, Any] = {
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "role": request.role,
            "logical_event_id": request.logical_event_id,
            "release_time_ns": request.release_time_ns,
            "eligible_time_ns": request.eligible_time_ns,
            "source_module": request.source_module,
            "target_module": request.target_module,
            "switch_path": f"{request.switch_path[0]}-{request.switch_path[1]}",
            "arrival_time_ns": item.arrival_time_ns,
            "service_start_time_ns": item.service_start_time_ns,
            "completion_time_ns": item.completion_time_ns,
            "service_time_ns": REMOTE_SERVICE_TIME_NS,
            "waiting_time_ns": item.waiting_time_ns,
            "turnaround_time_ns": item.turnaround_time_ns,
        }
        for category in RESOURCE_CATEGORIES:
            row[
                f"blocked_{category}_ns"
            ] = int(
                item.blocked_time_ns.get(
                    category,
                    0,
                )
            )
            row[
                f"blocker_checks_{category}"
            ] = int(
                item.blocker_checks.get(
                    category,
                    0,
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


def run_node_scenario(
    *,
    jobs: list[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    tenancy_allocation: TenancyAllocation,
    config: TenancyConfiguration,
    included_roles: set[str],
    hub_capacity: int,
    combined_execution: bool,
) -> pd.DataFrame:
    included_jobs = [
        job
        for job in jobs
        if job.role in included_roles
    ]
    requests = build_node_requests(
        jobs=included_jobs,
        allocations=allocations,
        tenancy_allocation=tenancy_allocation,
        config=config,
        combined_execution=combined_execution,
    )
    simulator = NodeAwareHubSimulator(
        hub_capacity=hub_capacity
    )
    return completed_dataframe(
        simulator.run(requests)
    )


def compare_attacker(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    baseline = attacker_only[
        attacker_only["role"] == "attacker"
    ].copy()
    victim_on = combined[
        combined["role"] == "attacker"
    ].copy()

    key_columns = [
        "logical_event_id",
        "release_time_ns",
        "source_module",
        "target_module",
        "switch_path",
    ]

    baseline_columns = key_columns + [
        "waiting_time_ns",
        "turnaround_time_ns",
        "completion_time_ns",
    ] + [
        f"blocked_{category}_ns"
        for category in RESOURCE_CATEGORIES
    ]

    combined_columns = baseline_columns

    baseline = baseline[
        baseline_columns
    ].rename(
        columns={
            "waiting_time_ns": "baseline_waiting_time_ns",
            "turnaround_time_ns": "baseline_turnaround_time_ns",
            "completion_time_ns": "baseline_completion_time_ns",
            **{
                f"blocked_{category}_ns": (
                    f"baseline_blocked_{category}_ns"
                )
                for category in RESOURCE_CATEGORIES
            },
        }
    )

    victim_on = victim_on[
        combined_columns
    ].rename(
        columns={
            "waiting_time_ns": "combined_waiting_time_ns",
            "turnaround_time_ns": "combined_turnaround_time_ns",
            "completion_time_ns": "combined_completion_time_ns",
            **{
                f"blocked_{category}_ns": (
                    f"combined_blocked_{category}_ns"
                )
                for category in RESOURCE_CATEGORIES
            },
        }
    )

    merged = victim_on.merge(
        baseline,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(baseline):
        raise RuntimeError(
            "Attacker request counts differ between baseline and combined runs."
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
        merged["excess_turnaround_time_ns"]
        > DETECTION_THRESHOLD_NS
    )

    for category in RESOURCE_CATEGORIES:
        merged[
            f"excess_blocked_{category}_ns"
        ] = (
            merged[
                f"combined_blocked_{category}_ns"
            ]
            - merged[
                f"baseline_blocked_{category}_ns"
            ]
        )

    return merged.sort_values(
        "logical_event_id"
    ).reset_index(drop=True)


# =============================================================================
# Trial metrics
# =============================================================================

def victim_completion(
    dataframe: pd.DataFrame,
) -> float:
    victim = dataframe[
        dataframe["role"] == "victim"
    ]
    if victim.empty:
        return 0.0
    return float(
        victim["completion_time_ns"].max()
        - VICTIM_START_NS
    )


def sharing_condition(
    allocation: TenancyAllocation,
    config: TenancyConfiguration,
) -> str:
    if not allocation.accepted:
        return "rejected"
    if not allocation.shared_modules:
        return "module_disjoint"
    if config.tenancy_model == "time_sliced_module_sharing":
        return "time_sliced_shared_module"
    if allocation.module_wide_endpoint_overlap:
        return "module_wide_endpoint_lock"
    if allocation.actual_communication_qubit_overlap:
        return "communication_qubit_overlap"
    if allocation.actual_local_route_overlap:
        return "local_route_overlap"
    if allocation.actual_reset_pipeline_overlap:
        return "reset_pipeline_overlap"
    return "shared_module_fully_isolated"


def allocation_metrics(
    allocation: TenancyAllocation,
    config: TenancyConfiguration,
) -> dict[str, float]:
    total_compute_capacity = (
        NUM_MODULES
        * config.compute_qubits_per_module
    )
    total_comm_capacity = (
        NUM_MODULES
        * config.communication_qubits_per_module
    )
    active_compute_capacity = (
        allocation.active_module_count
        * config.compute_qubits_per_module
    )
    active_comm_capacity = (
        allocation.active_module_count
        * config.communication_qubits_per_module
    )

    return {
        "system_compute_utilization": (
            (
                allocation.compute_qubits_allocated
                + allocation.compute_guard_qubits
            )
            / total_compute_capacity
            if total_compute_capacity
            else 0.0
        ),
        "active_module_compute_utilization": (
            (
                allocation.compute_qubits_allocated
                + allocation.compute_guard_qubits
            )
            / active_compute_capacity
            if active_compute_capacity
            else 0.0
        ),
        "logical_compute_utilization": (
            allocation.compute_qubits_allocated
            / total_compute_capacity
            if total_compute_capacity
            else 0.0
        ),
        "communication_qubit_utilization": (
            allocation.communication_qubits_reserved_unique
            / total_comm_capacity
            if total_comm_capacity
            else 0.0
        ),
        "active_module_communication_utilization": (
            allocation.communication_qubits_reserved_unique
            / active_comm_capacity
            if active_comm_capacity
            else 0.0
        ),
        "isolation_overhead_qubits": float(
            allocation.compute_guard_qubits
        ),
        "isolation_overhead_fraction": (
            allocation.compute_guard_qubits
            / total_compute_capacity
            if total_compute_capacity
            else 0.0
        ),
        "active_module_fraction": (
            allocation.active_module_count
            / NUM_MODULES
        ),
        "placement_expansion_modules": float(
            max(
                0,
                allocation.active_module_count - 4,
            )
        ),
    }


def run_execution_mode(
    *,
    jobs: list[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    tenancy_allocation: TenancyAllocation,
    config: TenancyConfiguration,
    hub_capacity: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attacker_only = run_node_scenario(
        jobs=jobs,
        allocations=allocations,
        tenancy_allocation=tenancy_allocation,
        config=config,
        included_roles={"attacker"},
        hub_capacity=hub_capacity,
        combined_execution=False,
    )
    victim_only = run_node_scenario(
        jobs=jobs,
        allocations=allocations,
        tenancy_allocation=tenancy_allocation,
        config=config,
        included_roles={"victim"},
        hub_capacity=hub_capacity,
        combined_execution=False,
    )
    combined = run_node_scenario(
        jobs=jobs,
        allocations=allocations,
        tenancy_allocation=tenancy_allocation,
        config=config,
        included_roles={"victim", "attacker"},
        hub_capacity=hub_capacity,
        combined_execution=True,
    )
    compared = compare_attacker(
        attacker_only,
        combined,
    )
    return (
        attacker_only,
        victim_only,
        combined,
        compared,
    )


def compared_metrics(
    compared: pd.DataFrame,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        f"{prefix}_attacker_probe_count": float(
            len(compared)
        ),
        f"{prefix}_avg_excess_waiting_time_ns": float(
            compared["excess_waiting_time_ns"].mean()
        ),
        f"{prefix}_avg_excess_turnaround_time_ns": float(
            compared["excess_turnaround_time_ns"].mean()
        ),
        f"{prefix}_max_excess_turnaround_time_ns": float(
            compared["excess_turnaround_time_ns"].max()
        ),
        f"{prefix}_total_excess_turnaround_time_ns": float(
            compared["excess_turnaround_time_ns"].sum()
        ),
        f"{prefix}_delayed_probe_count": float(
            compared["victim_contention_observed"].sum()
        ),
        f"{prefix}_detection_probability": float(
            compared["victim_contention_observed"].mean()
        ),
    }

    endpoint_columns = [
        "excess_blocked_endpoint_module_ns",
        "excess_blocked_communication_qubit_ns",
    ]
    metrics[
        f"{prefix}_avg_endpoint_wait_time_ns"
    ] = float(
        compared[endpoint_columns].clip(lower=0).sum(axis=1).mean()
    )

    for category in RESOURCE_CATEGORIES:
        metrics[
            f"{prefix}_avg_excess_blocked_{category}_ns"
        ] = float(
            compared[
                f"excess_blocked_{category}_ns"
            ].clip(lower=0).mean()
        )

    return metrics


def rejected_trial_row(
    *,
    victim_name: str,
    victim_job: p1.JobSpec,
    config: TenancyConfiguration,
    trial_id: int,
    allocation: TenancyAllocation,
) -> dict[str, Any]:
    return {
        "victim_qasm": victim_name,
        "victim_tag": p1.safe_tag(victim_name),
        "victim_logical_qubits": victim_job.logical_qubits,
        "trial_id": trial_id,
        "configuration_id": config.configuration_id,
        **config.__dict__,
        "accepted": False,
        "rejection_reason": allocation.rejection_reason,
        "placement_adjusted": allocation.placement_adjusted,
        "shared_module_count": len(allocation.shared_modules),
        "actual_sharing_condition": "rejected",
        "actual_communication_qubit_overlap": False,
        "actual_local_route_overlap": False,
        "actual_reset_pipeline_overlap": False,
        "module_wide_endpoint_overlap": False,
        "effective_communication_mode": (
            allocation.effective_communication_mode
        ),
        "hub_capacity": HUB_CAPACITY,
        "node_only_hub_capacity": NODE_ONLY_HUB_CAPACITY,
        "unconditional_node_only_leakage_ns": 0.0,
        "unconditional_hub4_leakage_ns": 0.0,
    }


def run_trial(
    *,
    victim_path: Path,
    config: TenancyConfiguration,
    trial_id: int,
    save_request_level: bool,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    victim_job = copy.deepcopy(
        p1.build_victim_job(
            victim_path,
            modules_requested=3,
        )
    )
    attacker_job = p1.build_attacker_job()
    jobs = [victim_job, attacker_job]

    allocation = evaluate_tenancy_allocation(
        victim_job=victim_job,
        attacker_job=attacker_job,
        config=config,
        trial_id=trial_id,
    )

    if not allocation.accepted:
        return (
            rejected_trial_row(
                victim_name=victim_path.name,
                victim_job=victim_job,
                config=config,
                trial_id=trial_id,
                allocation=allocation,
            ),
            [],
        )

    allocations = {
        "victim": allocation.victim_allocation,
        "attacker": allocation.attacker_allocation,
    }

    (
        hub_attacker_only,
        hub_victim_only,
        hub_combined,
        hub_compared,
    ) = run_execution_mode(
        jobs=jobs,
        allocations=allocations,
        tenancy_allocation=allocation,
        config=config,
        hub_capacity=HUB_CAPACITY,
    )

    (
        node_attacker_only,
        node_victim_only,
        node_combined,
        node_compared,
    ) = run_execution_mode(
        jobs=jobs,
        allocations=allocations,
        tenancy_allocation=allocation,
        config=config,
        hub_capacity=NODE_ONLY_HUB_CAPACITY,
    )

    hub_victim_only_duration = victim_completion(
        hub_victim_only
    )
    hub_combined_duration = victim_completion(
        hub_combined
    )
    node_victim_only_duration = victim_completion(
        node_victim_only
    )
    node_combined_duration = victim_completion(
        node_combined
    )

    row: dict[str, Any] = {
        "victim_qasm": victim_path.name,
        "victim_tag": p1.safe_tag(
            victim_path.name
        ),
        "victim_logical_qubits": (
            victim_job.logical_qubits
        ),
        "victim_remote_partition_edges": len(
            victim_job.partition_graph
        ),
        "trial_id": trial_id,
        "configuration_id": config.configuration_id,
        **config.__dict__,
        "accepted": True,
        "rejection_reason": "",
        "placement_adjusted": (
            allocation.placement_adjusted
        ),
        "victim_assigned_modules": ",".join(
            map(
                str,
                allocation.victim_allocation.assigned_modules,
            )
        ),
        "attacker_assigned_modules": ",".join(
            map(
                str,
                allocation.attacker_allocation.assigned_modules,
            )
        ),
        "shared_module_count": len(
            allocation.shared_modules
        ),
        "shared_modules": ",".join(
            map(str, allocation.shared_modules)
        ),
        "actual_sharing_condition": sharing_condition(
            allocation,
            config,
        ),
        "effective_communication_mode": (
            allocation.effective_communication_mode
        ),
        "actual_communication_qubit_overlap": (
            allocation.actual_communication_qubit_overlap
        ),
        "actual_local_route_overlap": (
            allocation.actual_local_route_overlap
        ),
        "actual_reset_pipeline_overlap": (
            allocation.actual_reset_pipeline_overlap
        ),
        "module_wide_endpoint_overlap": (
            allocation.module_wide_endpoint_overlap
        ),
        "compute_qubits_allocated": (
            allocation.compute_qubits_allocated
        ),
        "compute_guard_qubits": (
            allocation.compute_guard_qubits
        ),
        "communication_qubits_reserved_unique": (
            allocation.communication_qubits_reserved_unique
        ),
        "active_module_count": (
            allocation.active_module_count
        ),
        "hub_capacity": HUB_CAPACITY,
        "node_only_hub_capacity": (
            NODE_ONLY_HUB_CAPACITY
        ),
        **allocation_metrics(
            allocation,
            config,
        ),
        **compared_metrics(
            hub_compared,
            "hub4",
        ),
        **compared_metrics(
            node_compared,
            "node_only",
        ),
        "hub4_victim_only_duration_ns": (
            hub_victim_only_duration
        ),
        "hub4_combined_victim_duration_ns": (
            hub_combined_duration
        ),
        "hub4_victim_slowdown_ns": (
            hub_combined_duration
            - hub_victim_only_duration
        ),
        "hub4_victim_slowdown_ratio": (
            hub_combined_duration
            / hub_victim_only_duration
            if hub_victim_only_duration > 0
            else 1.0
        ),
        "node_only_victim_only_duration_ns": (
            node_victim_only_duration
        ),
        "node_only_combined_victim_duration_ns": (
            node_combined_duration
        ),
        "node_only_victim_slowdown_ns": (
            node_combined_duration
            - node_victim_only_duration
        ),
        "node_only_victim_slowdown_ratio": (
            node_combined_duration
            / node_victim_only_duration
            if node_victim_only_duration > 0
            else 1.0
        ),
        "hub_only_leakage_component_ns": max(
            0.0,
            float(
                hub_compared[
                    "excess_turnaround_time_ns"
                ].mean()
                - node_compared[
                    "excess_turnaround_time_ns"
                ].mean()
            ),
        ),
        "unconditional_node_only_leakage_ns": float(
            node_compared[
                "excess_turnaround_time_ns"
            ].mean()
        ),
        "unconditional_hub4_leakage_ns": float(
            hub_compared[
                "excess_turnaround_time_ns"
            ].mean()
        ),
    }

    request_frames: list[pd.DataFrame] = []
    if save_request_level:
        for execution_name, dataframe in [
            ("hub4", hub_compared),
            ("node_only", node_compared),
        ]:
            frame = dataframe.copy()
            frame.insert(0, "execution_name", execution_name)
            frame.insert(0, "trial_id", trial_id)
            frame.insert(0, "configuration_id", config.configuration_id)
            frame.insert(0, "victim_qasm", victim_path.name)
            frame.insert(0, "tenancy_model", config.tenancy_model)
            request_frames.append(frame)

    return row, request_frames


# =============================================================================
# Aggregation
# =============================================================================

def boolean_mean(series: pd.Series) -> float:
    return float(series.astype(float).mean())


def aggregate_results(
    trials: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    accepted = trials[trials["accepted"]].copy()

    model_summary = (
        trials.groupby(
            "tenancy_model",
            as_index=False,
        )
        .agg(
            trial_count=("accepted", "count"),
            acceptance_probability=("accepted", boolean_mean),
            placement_adjustment_probability=(
                "placement_adjusted",
                boolean_mean,
            ),
            shared_module_probability=(
                "shared_module_count",
                lambda series: float((series > 0).mean()),
            ),
            communication_qubit_overlap_probability=(
                "actual_communication_qubit_overlap",
                boolean_mean,
            ),
            local_route_overlap_probability=(
                "actual_local_route_overlap",
                boolean_mean,
            ),
            reset_pipeline_overlap_probability=(
                "actual_reset_pipeline_overlap",
                boolean_mean,
            ),
            module_wide_lock_overlap_probability=(
                "module_wide_endpoint_overlap",
                boolean_mean,
            ),
            unconditional_mean_node_only_leakage_ns=(
                "unconditional_node_only_leakage_ns",
                "mean",
            ),
            unconditional_mean_hub4_leakage_ns=(
                "unconditional_hub4_leakage_ns",
                "mean",
            ),
        )
    )

    if not accepted.empty:
        accepted_summary = (
            accepted.groupby(
                "tenancy_model",
                as_index=False,
            )
            .agg(
                conditional_mean_node_only_leakage_ns=(
                    "node_only_avg_excess_turnaround_time_ns",
                    "mean",
                ),
                conditional_mean_hub4_leakage_ns=(
                    "hub4_avg_excess_turnaround_time_ns",
                    "mean",
                ),
                mean_node_only_endpoint_wait_ns=(
                    "node_only_avg_endpoint_wait_time_ns",
                    "mean",
                ),
                mean_node_only_detection_probability=(
                    "node_only_detection_probability",
                    "mean",
                ),
                mean_hub4_detection_probability=(
                    "hub4_detection_probability",
                    "mean",
                ),
                mean_victim_slowdown_ratio=(
                    "hub4_victim_slowdown_ratio",
                    "mean",
                ),
                mean_compute_utilization=(
                    "system_compute_utilization",
                    "mean",
                ),
                mean_communication_utilization=(
                    "communication_qubit_utilization",
                    "mean",
                ),
                mean_isolation_overhead_fraction=(
                    "isolation_overhead_fraction",
                    "mean",
                ),
                mean_hub_only_leakage_component_ns=(
                    "hub_only_leakage_component_ns",
                    "mean",
                ),
            )
        )
        model_summary = model_summary.merge(
            accepted_summary,
            on="tenancy_model",
            how="left",
        )

    endpoint_summary = (
        accepted.groupby(
            [
                "tenancy_model",
                "endpoint_lock_scope",
                "actual_sharing_condition",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            mean_node_only_endpoint_wait_ns=(
                "node_only_avg_endpoint_wait_time_ns",
                "mean",
            ),
            mean_node_only_leakage_ns=(
                "node_only_avg_excess_turnaround_time_ns",
                "mean",
            ),
            detection_probability=(
                "node_only_detection_probability",
                "mean",
            ),
        )
    )

    resource_rows: list[dict[str, Any]] = []
    for (
        model,
        condition,
    ), group in accepted.groupby(
        [
            "tenancy_model",
            "actual_sharing_condition",
        ]
    ):
        for category in RESOURCE_CATEGORIES:
            resource_rows.append(
                {
                    "tenancy_model": model,
                    "actual_sharing_condition": condition,
                    "resource_category": category,
                    "trial_count": len(group),
                    "mean_node_only_blocked_time_ns": float(
                        group[
                            f"node_only_avg_excess_blocked_{category}_ns"
                        ].mean()
                    ),
                    "mean_hub4_blocked_time_ns": float(
                        group[
                            f"hub4_avg_excess_blocked_{category}_ns"
                        ].mean()
                    ),
                }
            )
    resource_summary = pd.DataFrame(resource_rows)

    effect_rows: list[dict[str, Any]] = []
    effect_specs = [
        (
            "endpoint_lock_scope",
            "module_wide",
            "communication_qubit_specific",
            "module_wide_minus_specific",
        ),
        (
            "actual_local_route_overlap",
            True,
            False,
            "route_overlap_minus_no_overlap",
        ),
        (
            "actual_reset_pipeline_overlap",
            True,
            False,
            "reset_overlap_minus_no_overlap",
        ),
        (
            "actual_communication_qubit_overlap",
            True,
            False,
            "comm_overlap_minus_no_overlap",
        ),
    ]

    for model, model_group in accepted.groupby(
        "tenancy_model"
    ):
        for column, positive, negative, effect_name in effect_specs:
            positive_values = model_group[
                model_group[column] == positive
            ]["node_only_avg_excess_turnaround_time_ns"]
            negative_values = model_group[
                model_group[column] == negative
            ]["node_only_avg_excess_turnaround_time_ns"]

            effect_rows.append(
                {
                    "tenancy_model": model,
                    "effect_name": effect_name,
                    "positive_sample_count": len(positive_values),
                    "negative_sample_count": len(negative_values),
                    "positive_mean_node_only_leakage_ns": (
                        float(positive_values.mean())
                        if len(positive_values)
                        else np.nan
                    ),
                    "negative_mean_node_only_leakage_ns": (
                        float(negative_values.mean())
                        if len(negative_values)
                        else np.nan
                    ),
                    "estimated_main_effect_ns": (
                        float(
                            positive_values.mean()
                            - negative_values.mean()
                        )
                        if len(positive_values)
                        and len(negative_values)
                        else np.nan
                    ),
                }
            )
    effect_summary = pd.DataFrame(effect_rows)

    admission_summary = (
        trials.groupby(
            [
                "tenancy_model",
                "compute_qubits_per_module",
                "communication_qubits_per_module",
                "communication_qubits_per_tenant",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("accepted", "count"),
            acceptance_probability=("accepted", boolean_mean),
            rejection_rate=(
                "accepted",
                lambda series: float(1.0 - series.astype(float).mean()),
            ),
        )
    )

    utilization_summary = (
        accepted.groupby(
            [
                "tenancy_model",
                "compute_qubits_per_module",
                "communication_qubits_per_module",
                "spatial_isolation_degree",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            mean_system_compute_utilization=(
                "system_compute_utilization",
                "mean",
            ),
            mean_active_compute_utilization=(
                "active_module_compute_utilization",
                "mean",
            ),
            mean_communication_utilization=(
                "communication_qubit_utilization",
                "mean",
            ),
            mean_isolation_overhead_fraction=(
                "isolation_overhead_fraction",
                "mean",
            ),
            mean_active_module_fraction=(
                "active_module_fraction",
                "mean",
            ),
        )
    )

    configuration_summary = (
        trials.groupby(
            [
                "configuration_id",
                "tenancy_model",
                "compute_qubits_per_module",
                "communication_qubits_per_module",
                "communication_qubits_per_tenant",
                "spatial_isolation_degree",
                "local_routing_paths_shared",
                "reset_pipeline_shared",
                "endpoint_lock_scope",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("accepted", "count"),
            acceptance_probability=("accepted", boolean_mean),
            unconditional_node_only_leakage_ns=(
                "unconditional_node_only_leakage_ns",
                "mean",
            ),
            unconditional_hub4_leakage_ns=(
                "unconditional_hub4_leakage_ns",
                "mean",
            ),
        )
    )

    return {
        "model_summary": model_summary,
        "endpoint_summary": endpoint_summary,
        "resource_summary": resource_summary,
        "effect_summary": effect_summary,
        "admission_summary": admission_summary,
        "utilization_summary": utilization_summary,
        "configuration_summary": configuration_summary,
    }


# =============================================================================
# Plotting
# =============================================================================

def save_model_bar_plot(
    dataframe: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    ordered = dataframe.set_index(
        "tenancy_model"
    ).reindex(TENANCY_MODELS)

    axis = ordered[metric].plot(
        kind="bar",
        figsize=(12, 6),
    )
    axis.set_xlabel("Tenancy model")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )
    plt.close()


def save_resource_plot(
    resource_summary: pd.DataFrame,
) -> None:
    grouped = (
        resource_summary.groupby(
            [
                "tenancy_model",
                "resource_category",
            ],
            as_index=False,
        )["mean_node_only_blocked_time_ns"]
        .mean()
    )
    pivot = grouped.pivot(
        index="tenancy_model",
        columns="resource_category",
        values="mean_node_only_blocked_time_ns",
    ).reindex(TENANCY_MODELS)

    axis = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 7),
    )
    axis.set_xlabel("Tenancy model")
    axis.set_ylabel(
        "Mean attacker blocked time (ns; categories may overlap)"
    )
    axis.set_title(
        "Node-Side Blocking Sources with Global Hub Contention Removed"
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Blocking resource", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "tenancy_model_resource_blocking.png",
        dpi=300,
    )
    plt.close()


# =============================================================================
# Configuration generation and CLI
# =============================================================================

def configuration_product(
    quick: bool,
) -> list[TenancyConfiguration]:
    if quick:
        compute_options = [16]
        communication_options = [1, 4]
        per_tenant_options = [1]
        isolation_options = [0.0, 1.0]
    else:
        compute_options = COMPUTE_QUBITS_PER_MODULE_OPTIONS
        communication_options = COMMUNICATION_QUBITS_PER_MODULE_OPTIONS
        per_tenant_options = COMMUNICATION_QUBITS_PER_TENANT_OPTIONS
        isolation_options = SPATIAL_ISOLATION_OPTIONS

    return [
        TenancyConfiguration(
            tenancy_model=model,
            compute_qubits_per_module=compute,
            communication_qubits_per_module=communication,
            communication_qubits_per_tenant=per_tenant,
            spatial_isolation_degree=isolation,
            local_routing_paths_shared=route_shared,
            reset_pipeline_shared=reset_shared,
            endpoint_lock_scope=lock_scope,
        )
        for (
            model,
            compute,
            communication,
            per_tenant,
            isolation,
            route_shared,
            reset_shared,
            lock_scope,
        ) in itertools.product(
            TENANCY_MODELS,
            compute_options,
            communication_options,
            per_tenant_options,
            isolation_options,
            LOCAL_ROUTING_SHARED_OPTIONS,
            RESET_PIPELINE_SHARED_OPTIONS,
            ENDPOINT_LOCK_SCOPE_OPTIONS,
        )
    ]




# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if TRIALS_PER_CONFIGURATION <= 0:
        raise ValueError(
            "TRIALS_PER_CONFIGURATION must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    victim_paths = [
        p1.resolve_qasm(filename)
        for filename in p1.VICTIM_QASMS
    ]

    if RUN_QUICK_VALIDATION:
        victim_paths = victim_paths[:2]

    configurations = configuration_product(
        RUN_QUICK_VALIDATION
    )

    if MAX_CONFIGURATIONS is not None:
        configurations = configurations[
            : MAX_CONFIGURATIONS
        ]

    total_runs = (
        len(victim_paths)
        * len(configurations)
        * TRIALS_PER_CONFIGURATION
    )

    run_mode = (
        "quick validation"
        if RUN_QUICK_VALIDATION
        else "complete factorial"
    )

    print(
        "Phase 1.2 — Tenancy and module-sharing models"
    )
    print(f"Run mode:         {run_mode}")
    print(f"Victim workloads: {len(victim_paths)}")
    print(f"Configurations:   {len(configurations)}")
    print(
        "Trials each:      "
        f"{TRIALS_PER_CONFIGURATION}"
    )
    print(f"Total trials:     {total_runs}")
    print(
        "Request-level:    "
        f"{SAVE_REQUEST_LEVEL_RESULTS}"
    )
    print(f"Output directory: {OUTPUT_DIR}")

    trial_rows: list[dict[str, Any]] = []
    request_frames: list[pd.DataFrame] = []

    completed_count = 0
    for victim_path in victim_paths:
        for config in configurations:
            for trial_id in range(TRIALS_PER_CONFIGURATION):
                row, frames = run_trial(
                    victim_path=victim_path,
                    config=config,
                    trial_id=trial_id,
                    save_request_level=(
                        SAVE_REQUEST_LEVEL_RESULTS
                    ),
                )
                trial_rows.append(row)
                request_frames.extend(frames)
                completed_count += 1

                if (
                    completed_count % 100 == 0
                    or completed_count == total_runs
                ):
                    print(
                        f"Completed {completed_count}/{total_runs} trials"
                    )

    trials = pd.DataFrame(trial_rows)
    trial_path = (
        OUTPUT_DIR
        / "tenancy_model_trial_summary.csv"
    )
    trials.to_csv(
        trial_path,
        index=False,
    )

    if SAVE_REQUEST_LEVEL_RESULTS and request_frames:
        requests = pd.concat(
            request_frames,
            ignore_index=True,
        )
        requests.to_csv(
            OUTPUT_DIR
            / "tenancy_model_attacker_observations.csv",
            index=False,
        )

    outputs = aggregate_results(trials)

    output_files = {
        "model_summary": "tenancy_model_summary.csv",
        "endpoint_summary": "tenancy_model_endpoint_wait_summary.csv",
        "resource_summary": "tenancy_model_resource_blocking_summary.csv",
        "effect_summary": "tenancy_model_resource_effect_summary.csv",
        "admission_summary": "tenancy_model_admission_summary.csv",
        "utilization_summary": "tenancy_model_utilization_summary.csv",
        "configuration_summary": "tenancy_model_configuration_summary.csv",
    }

    for key, filename in output_files.items():
        outputs[key].to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    model_summary = outputs[
        "model_summary"
    ]

    save_model_bar_plot(
        model_summary,
        metric=(
            "conditional_mean_node_only_leakage_ns"
        ),
        ylabel=(
            "Mean attacker excess latency (ns)"
        ),
        title=(
            "Tenancy Models: Node-Side Leakage with Hub Contention Removed"
        ),
        filename=(
            "tenancy_model_node_only_leakage.png"
        ),
    )

    save_model_bar_plot(
        model_summary,
        metric=(
            "mean_node_only_endpoint_wait_ns"
        ),
        ylabel=(
            "Mean endpoint-related attacker wait (ns)"
        ),
        title=(
            "Endpoint Waiting by Tenancy Model"
        ),
        filename=(
            "tenancy_model_endpoint_wait.png"
        ),
    )

    rejection_plot = model_summary.copy()
    rejection_plot["rejection_rate"] = (
        1.0
        - rejection_plot["acceptance_probability"]
    )
    save_model_bar_plot(
        rejection_plot,
        metric="rejection_rate",
        ylabel="Job rejection rate",
        title="Tenancy-Model Admission Cost",
        filename="tenancy_model_rejection_rate.png",
    )

    save_model_bar_plot(
        model_summary,
        metric=(
            "mean_isolation_overhead_fraction"
        ),
        ylabel=(
            "Reserved guard-qubit fraction"
        ),
        title=(
            "Spatial-Isolation Overhead by Tenancy Model"
        ),
        filename=(
            "tenancy_model_isolation_overhead.png"
        ),
    )

    save_resource_plot(
        outputs[
            "resource_summary"
        ]
    )

    display_columns = [
        "tenancy_model",
        "acceptance_probability",
        "shared_module_probability",
        "communication_qubit_overlap_probability",
        "local_route_overlap_probability",
        "reset_pipeline_overlap_probability",
        "conditional_mean_node_only_leakage_ns",
        "mean_node_only_endpoint_wait_ns",
        "mean_node_only_detection_probability",
        "mean_victim_slowdown_ratio",
        "mean_isolation_overhead_fraction",
    ]

    print("\n=== Phase 1.2 model summary ===")
    print(
        model_summary[
            display_columns
        ].to_string(index=False)
    )

    print("\nSaved results:")
    print(trial_path)
    for filename in output_files.values():
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
