#!/usr/bin/env python3
"""
phase1_03_communication_qubit_allocation.py

Experiment 1.3 — Dynamic communication-qubit allocation.

Research question
-----------------
Does dynamically assigning communication qubits introduce, remove, or reshape
black-box timing leakage?

This experiment holds logical placement in a P2-like form so that the victim,
attacker, and optional background tenants all use module 2 as a communication
endpoint. The global hub is intentionally overprovisioned so measured delay is
caused primarily by communication-qubit allocation, reservation, reset, and
reuse rather than by hub-capacity contention.

Implemented allocation policies
-------------------------------
1. static_assignment
2. first_available
3. round_robin
4. random_allocation
5. per_tenant_reserved
6. shared_global_pool
7. priority_allocation
8. fair_share
9. coherence_aware
10. predicted_demand

Independent variables
---------------------
- communication qubits per module: 1, 2, 4, 8;
- tenants sharing the endpoint module: 2, 3, 4, 8;
- reservation duration: 0, 500, 2,000 ns;
- reset time: 0, 50, 200 ns;
- EPR prefetch disabled/enabled;
- local or globally coordinated endpoint allocation;
- allocation granularity: per operation, per layer, or per job.

Integrated experimental design
------------------------------
A complete Cartesian product would require hundreds of thousands of physical
executions. Instead, the script runs three complementary sub-sweeps:

A. core_policy_capacity_tenancy
   All policies × all communication-qubit counts × all tenant counts.

B. reservation_reset
   All policies × all reservation durations × all reset times.

C. epr_coordination_granularity
   All policies × EPR prefetch × local/global coordination × all granularities.

Every requested policy and independent variable is therefore exercised while
keeping the experiment executable. Set RUN_COMPLETE_FACTORIAL=True below only
if the full Cartesian product is explicitly needed.

Controls
--------
For every configuration and workload, the simulator runs:
- attacker + identical background tenants;
- victim + identical background tenants;
- victim + attacker + identical background tenants.

Attacker-only and combined traces are matched by logical probe ID. Background
traffic and deterministic request-specific random choices are coupled across
controls so baseline subtraction isolates victim-induced allocation effects.

Communication-qubit behavior
----------------------------
Each remote operation requires one communication qubit at each endpoint.
The manager records allocation request, partial/local grants, atomic/global
grants, occupancy, release, reset, reservation, reuse, redirection, failures,
allocation conflicts, and optional prefetched-EPR use or wastage.

Local coordination may hold one endpoint while waiting for the other. Global
coordination grants both endpoints atomically. A fixed acquisition order avoids
artificial deadlock in local mode.

Outputs
-------
blackbox_window_results/phase1_03_communication_qubit_allocation/
    communication_qubit_trial_summary.csv
    communication_qubit_attacker_request_log.csv
    communication_qubit_configuration_summary.csv
    communication_qubit_policy_summary.csv
    communication_qubit_capacity_summary.csv
    communication_qubit_tenancy_summary.csv
    communication_qubit_reservation_reset_summary.csv
    communication_qubit_granularity_summary.csv
    communication_qubit_coordination_summary.csv
    communication_qubit_epr_summary.csv
    communication_qubit_fairness_summary.csv
    communication_qubit_failure_summary.csv

    core_policy_capacity_tenancy/
    reservation_reset/
    epr_coordination_granularity/

Execution
---------
Keep this file beside phase1_01_job_module_allocation.py and run:

    python phase1_03_communication_qubit_allocation.py

No command-line options are required.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import json
import math
import random
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
    / "phase1_03_communication_qubit_allocation"
)

# Complete experiment defaults.
TRIALS_PER_CONFIGURATION = 3
RUN_COMPLETE_FACTORIAL = False
RUN_QUICK_VALIDATION = False

# Attacker-level rows are central to the experiment and are always saved.
SAVE_ATTACKER_REQUEST_LOG = True

# Saving every victim/background request can create a much larger file.
SAVE_ALL_REQUEST_LOGS = False

# Optional configuration cap for source-level debugging. Keep None normally.
MAX_CONFIGURATIONS: int | None = None

GLOBAL_SEED = 20260730


# =============================================================================
# Fixed architecture and workload settings
# =============================================================================

NUM_MODULES = 5
SHARED_ENDPOINT_MODULE = 2
VICTIM_MODULES = (0, 1, 2)
ATTACKER_MODULES = (2, 3)
BACKGROUND_PARTNER_MODULES = (4, 1, 0, 3)

# Large enough to suppress global hub-slot contention in the tested ranges.
HUB_CAPACITY = 64

BASE_REMOTE_SERVICE_NS = p1.REMOTE_SERVICE_TIME_NS
PREFETCHED_REMOTE_SERVICE_NS = 60

LAYER_WINDOW_NS = 250
REQUEST_COHERENCE_SLACK_NS = 1_000
PREDICTION_HORIZON_NS = 1_000
MAX_ALLOCATION_WAIT_NS = 10_000

EPR_PREFETCH_LATENCY_NS = 40
EPR_COHERENCE_TIME_NS = 1_000

DETECTION_THRESHOLD_NS = 0.0


# =============================================================================
# Independent variables and policies
# =============================================================================

ALLOCATION_POLICIES = [
    "static_assignment",
    "first_available",
    "round_robin",
    "random_allocation",
    "per_tenant_reserved",
    "shared_global_pool",
    "priority_allocation",
    "fair_share",
    "coherence_aware",
    "predicted_demand",
]

COMMUNICATION_QUBITS_PER_MODULE_OPTIONS = [1, 2, 4, 8]
TENANTS_SHARING_MODULE_OPTIONS = [2, 3, 4, 8]
RESERVATION_DURATION_OPTIONS_NS = [0, 500, 2_000]
RESET_TIME_OPTIONS_NS = [0, 50, 200]
EPR_PREFETCH_OPTIONS = [False, True]
COORDINATION_OPTIONS = ["local", "global"]
ALLOCATION_GRANULARITY_OPTIONS = [
    "per_operation",
    "per_layer",
    "per_job",
]

# Canonical values used when the corresponding factor is not being swept.
DEFAULT_COMMUNICATION_QUBITS_PER_MODULE = 2
DEFAULT_TENANTS_SHARING_MODULE = 4
DEFAULT_RESERVATION_DURATION_NS = 500
DEFAULT_RESET_TIME_NS = 50
DEFAULT_EPR_PREFETCH = False
DEFAULT_COORDINATION = "global"
DEFAULT_GRANULARITY = "per_operation"

ROLE_PRIORITY = {
    "victim": 0,
    "attacker": 1,
    "background": 2,
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfiguration:
    subexperiment: str
    allocation_policy: str
    communication_qubits_per_module: int
    tenants_sharing_module: int
    reservation_duration_ns: int
    reset_time_ns: int
    epr_prefetch_enabled: bool
    allocation_coordination: str
    allocation_granularity: str

    @property
    def configuration_id(self) -> str:
        epr = "epr" if self.epr_prefetch_enabled else "noepr"
        return (
            f"{self.subexperiment}__{self.allocation_policy}__"
            f"cq{self.communication_qubits_per_module}__"
            f"ten{self.tenants_sharing_module}__"
            f"res{self.reservation_duration_ns}__"
            f"reset{self.reset_time_ns}__{epr}__"
            f"{self.allocation_coordination}__"
            f"{self.allocation_granularity}"
        )


@dataclass(frozen=True)
class CommunicationRequest:
    request_id: int
    tenant_id: str
    role: str
    release_time_ns: int
    source_module: int
    target_module: int
    logical_event_id: int
    layer_id: int
    layer_end_ns: int
    job_end_ns: int
    coherence_deadline_ns: int

    @property
    def endpoints(self) -> tuple[int, int]:
        return (
            min(self.source_module, self.target_module),
            max(self.source_module, self.target_module),
        )


@dataclass
class CommunicationQubit:
    pool_id: str
    module_id: int | None
    qubit_id: int
    persistent_owner: str | None = None
    lease_owner: str | None = None
    lease_key: str | None = None
    lease_until_ns: int = 0
    held_by_request: int | None = None
    active_request: int | None = None
    reset_until_ns: int = 0
    last_tenant: str | None = None
    last_occupancy_end_ns: int = 0
    last_reset_completion_ns: int = 0
    service_busy_ns: int = 0
    reset_busy_ns: int = 0
    allocation_hold_busy_ns: int = 0
    reserved_capacity_ns: int = 0
    grant_count: int = 0
    epr_ready_time_ns: int | None = None
    epr_expiry_time_ns: int | None = None
    epr_prefetched_count: int = 0
    epr_used_count: int = 0
    epr_wasted_count: int = 0
    epr_stranded_count: int = 0

    @property
    def label(self) -> str:
        return f"{self.pool_id}:{self.qubit_id}"


@dataclass
class PendingRequest:
    request: CommunicationRequest
    source_qubit: CommunicationQubit | None = None
    target_qubit: CommunicationQubit | None = None
    source_grant_time_ns: int | None = None
    target_grant_time_ns: int | None = None
    conflict_checks: int = 0
    partial_hold_time_ns: int = 0
    redirected_endpoint_count: int = 0

    @property
    def allocation_grant_time_ns(self) -> int | None:
        if (
            self.source_grant_time_ns is None
            or self.target_grant_time_ns is None
        ):
            return None
        return max(
            self.source_grant_time_ns,
            self.target_grant_time_ns,
        )


@dataclass(order=True)
class ActiveService:
    completion_time_ns: int
    serial: int
    pending: PendingRequest = field(compare=False)
    service_start_time_ns: int = field(compare=False)
    service_time_ns: int = field(compare=False)
    source_reuse_delay_ns: int = field(compare=False)
    target_reuse_delay_ns: int = field(compare=False)
    used_prefetched_epr: bool = field(compare=False)
    reset_completion_time_ns: int = field(compare=False)


@dataclass
class CompletedAllocation:
    request: CommunicationRequest
    source_qubit_label: str
    target_qubit_label: str
    allocation_request_time_ns: int
    source_grant_time_ns: int
    target_grant_time_ns: int
    allocation_grant_time_ns: int
    occupancy_start_time_ns: int
    occupancy_end_time_ns: int
    reset_start_time_ns: int
    reset_completion_time_ns: int
    source_reuse_delay_ns: int
    target_reuse_delay_ns: int
    service_time_ns: int
    used_prefetched_epr: bool
    conflict_checks: int
    partial_hold_time_ns: int
    redirected_endpoint_count: int

    @property
    def communication_qubit_queue_delay_ns(self) -> int:
        return (
            self.allocation_grant_time_ns
            - self.allocation_request_time_ns
        )

    @property
    def service_queue_delay_ns(self) -> int:
        return (
            self.occupancy_start_time_ns
            - self.allocation_grant_time_ns
        )

    @property
    def waiting_time_ns(self) -> int:
        return (
            self.occupancy_start_time_ns
            - self.allocation_request_time_ns
        )

    @property
    def turnaround_time_ns(self) -> int:
        return (
            self.occupancy_end_time_ns
            - self.allocation_request_time_ns
        )


@dataclass
class FailedAllocation:
    request: CommunicationRequest
    failure_time_ns: int
    failure_reason: str
    conflict_checks: int
    partial_hold_time_ns: int
    redirected_endpoint_count: int


@dataclass
class ScenarioResult:
    request_dataframe: pd.DataFrame
    qubit_dataframe: pd.DataFrame
    tenant_dataframe: pd.DataFrame
    makespan_ns: int
    rejected_tenants: set[str]
    total_conflicts: int
    total_redirected_endpoints: int
    total_failed_requests: int
    total_prefetched_pairs: int
    total_used_prefetched_pairs: int
    total_wasted_prefetched_pairs: int
    total_stranded_prefetched_pairs: int
    communication_qubit_utilization: float
    reset_utilization: float
    allocation_hold_utilization: float
    reservation_utilization: float
    effective_communication_qubit_utilization: float
    jain_grant_fairness: float
    jain_inverse_wait_fairness: float


# =============================================================================
# Utility functions
# =============================================================================


def stable_text_seed(value: str) -> int:
    total = 0
    for index, character in enumerate(value):
        total += (index + 1) * ord(character)
    return total


def jain_index(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return 1.0
    if np.all(array == 0):
        return 1.0
    denominator = len(array) * float(np.square(array).sum())
    if denominator <= 0:
        return 1.0
    return float(np.square(array.sum()) / denominator)


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def safe_max(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.max())


def role_sort_key(role: str) -> int:
    return ROLE_PRIORITY.get(role, 99)


# =============================================================================
# Experimental design
# =============================================================================


def generate_screening_configurations() -> list[ExperimentConfiguration]:
    configurations: list[ExperimentConfiguration] = []

    for policy, communication_qubits, tenant_count in itertools.product(
        ALLOCATION_POLICIES,
        COMMUNICATION_QUBITS_PER_MODULE_OPTIONS,
        TENANTS_SHARING_MODULE_OPTIONS,
    ):
        configurations.append(
            ExperimentConfiguration(
                subexperiment="core_policy_capacity_tenancy",
                allocation_policy=policy,
                communication_qubits_per_module=communication_qubits,
                tenants_sharing_module=tenant_count,
                reservation_duration_ns=DEFAULT_RESERVATION_DURATION_NS,
                reset_time_ns=DEFAULT_RESET_TIME_NS,
                epr_prefetch_enabled=DEFAULT_EPR_PREFETCH,
                allocation_coordination=DEFAULT_COORDINATION,
                allocation_granularity=DEFAULT_GRANULARITY,
            )
        )

    for policy, reservation, reset in itertools.product(
        ALLOCATION_POLICIES,
        RESERVATION_DURATION_OPTIONS_NS,
        RESET_TIME_OPTIONS_NS,
    ):
        configurations.append(
            ExperimentConfiguration(
                subexperiment="reservation_reset",
                allocation_policy=policy,
                communication_qubits_per_module=(
                    DEFAULT_COMMUNICATION_QUBITS_PER_MODULE
                ),
                tenants_sharing_module=DEFAULT_TENANTS_SHARING_MODULE,
                reservation_duration_ns=reservation,
                reset_time_ns=reset,
                epr_prefetch_enabled=DEFAULT_EPR_PREFETCH,
                allocation_coordination=DEFAULT_COORDINATION,
                allocation_granularity=DEFAULT_GRANULARITY,
            )
        )

    for policy, epr, coordination, granularity in itertools.product(
        ALLOCATION_POLICIES,
        EPR_PREFETCH_OPTIONS,
        COORDINATION_OPTIONS,
        ALLOCATION_GRANULARITY_OPTIONS,
    ):
        configurations.append(
            ExperimentConfiguration(
                subexperiment="epr_coordination_granularity",
                allocation_policy=policy,
                communication_qubits_per_module=(
                    DEFAULT_COMMUNICATION_QUBITS_PER_MODULE
                ),
                tenants_sharing_module=DEFAULT_TENANTS_SHARING_MODULE,
                reservation_duration_ns=DEFAULT_RESERVATION_DURATION_NS,
                reset_time_ns=DEFAULT_RESET_TIME_NS,
                epr_prefetch_enabled=epr,
                allocation_coordination=coordination,
                allocation_granularity=granularity,
            )
        )

    unique = {
        configuration.configuration_id: configuration
        for configuration in configurations
    }
    return list(unique.values())


def generate_complete_factorial() -> list[ExperimentConfiguration]:
    configurations: list[ExperimentConfiguration] = []
    for values in itertools.product(
        ALLOCATION_POLICIES,
        COMMUNICATION_QUBITS_PER_MODULE_OPTIONS,
        TENANTS_SHARING_MODULE_OPTIONS,
        RESERVATION_DURATION_OPTIONS_NS,
        RESET_TIME_OPTIONS_NS,
        EPR_PREFETCH_OPTIONS,
        COORDINATION_OPTIONS,
        ALLOCATION_GRANULARITY_OPTIONS,
    ):
        (
            policy,
            communication_qubits,
            tenant_count,
            reservation,
            reset,
            epr,
            coordination,
            granularity,
        ) = values
        configurations.append(
            ExperimentConfiguration(
                subexperiment="complete_factorial",
                allocation_policy=policy,
                communication_qubits_per_module=communication_qubits,
                tenants_sharing_module=tenant_count,
                reservation_duration_ns=reservation,
                reset_time_ns=reset,
                epr_prefetch_enabled=epr,
                allocation_coordination=coordination,
                allocation_granularity=granularity,
            )
        )
    return configurations


def experiment_configurations() -> list[ExperimentConfiguration]:
    if RUN_QUICK_VALIDATION:
        policies = [
            "static_assignment",
            "first_available",
            "per_tenant_reserved",
            "shared_global_pool",
            "coherence_aware",
        ]
        return [
            ExperimentConfiguration(
                subexperiment="quick_validation",
                allocation_policy=policy,
                communication_qubits_per_module=2,
                tenants_sharing_module=4,
                reservation_duration_ns=500,
                reset_time_ns=50,
                epr_prefetch_enabled=True,
                allocation_coordination="global",
                allocation_granularity="per_operation",
            )
            for policy in policies
        ]

    configurations = (
        generate_complete_factorial()
        if RUN_COMPLETE_FACTORIAL
        else generate_screening_configurations()
    )
    configurations = sorted(
        configurations,
        key=lambda configuration: (
            configuration.subexperiment,
            configuration.allocation_policy,
            configuration.communication_qubits_per_module,
            configuration.tenants_sharing_module,
            configuration.reservation_duration_ns,
            configuration.reset_time_ns,
            configuration.epr_prefetch_enabled,
            configuration.allocation_coordination,
            configuration.allocation_granularity,
        ),
    )
    if MAX_CONFIGURATIONS is not None:
        configurations = configurations[:MAX_CONFIGURATIONS]
    return configurations


# =============================================================================
# Workload preparation and fixed P2-like placement
# =============================================================================


def build_jobs_and_allocations(
    victim_path: Path,
    tenants_sharing_module: int,
    trial_seed_value: int,
) -> tuple[list[p1.JobSpec], dict[str, p1.Allocation]]:
    if tenants_sharing_module < 2:
        raise ValueError("At least victim and attacker must share the module.")

    victim_job = copy.deepcopy(
        p1.build_victim_job(
            victim_path,
            modules_requested=3,
        )
    )
    attacker_job = p1.build_attacker_job()

    rng = random.Random(trial_seed_value)
    background_count = tenants_sharing_module - 2
    background_jobs = [
        p1.build_background_job(
            tenant_index=index,
            modules_requested=2,
            rng=rng,
        )
        for index in range(background_count)
    ]

    jobs = [victim_job, attacker_job, *background_jobs]

    victim_mapping = p1.partition_mapping_for_subset(
        victim_job,
        VICTIM_MODULES,
        NUM_MODULES,
        optimize_communication=True,
    )

    allocations: dict[str, p1.Allocation] = {
        "victim": p1.Allocation(
            tenant_id="victim",
            role="victim",
            accepted=True,
            requested_modules=3,
            partition_to_module=victim_mapping,
        ),
        "attacker": p1.Allocation(
            tenant_id="attacker",
            role="attacker",
            accepted=True,
            requested_modules=2,
            partition_to_module={0: 2, 1: 3},
        ),
    }

    for index, job in enumerate(background_jobs):
        partner_module = BACKGROUND_PARTNER_MODULES[
            index % len(BACKGROUND_PARTNER_MODULES)
        ]
        allocations[job.tenant_id] = p1.Allocation(
            tenant_id=job.tenant_id,
            role=job.role,
            accepted=True,
            requested_modules=2,
            partition_to_module={0: 2, 1: partner_module},
        )

    return jobs, allocations


def communication_requests(
    jobs: Sequence[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    included_roles: set[str],
) -> list[CommunicationRequest]:
    included_jobs = [
        job
        for job in jobs
        if job.role in included_roles
    ]
    raw_requests = p1.build_remote_requests(
        included_jobs,
        allocations,
        hub_capacity=HUB_CAPACITY,
    )

    job_by_tenant = {
        job.tenant_id: job
        for job in included_jobs
    }
    last_release_by_tenant: dict[str, int] = defaultdict(int)
    for request in raw_requests:
        last_release_by_tenant[request.tenant_id] = max(
            last_release_by_tenant[request.tenant_id],
            request.release_time_ns,
        )

    converted: list[CommunicationRequest] = []
    for request in raw_requests:
        job = job_by_tenant[request.tenant_id]
        relative_release = request.release_time_ns - job.start_time_ns
        layer_id = max(0, relative_release // LAYER_WINDOW_NS)
        layer_end_ns = (
            job.start_time_ns
            + (layer_id + 1) * LAYER_WINDOW_NS
        )
        job_end_ns = (
            last_release_by_tenant[request.tenant_id]
            + BASE_REMOTE_SERVICE_NS
        )
        converted.append(
            CommunicationRequest(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                role=request.role,
                release_time_ns=request.release_time_ns,
                source_module=request.source_module,
                target_module=request.target_module,
                logical_event_id=request.logical_event_id,
                layer_id=int(layer_id),
                layer_end_ns=int(layer_end_ns),
                job_end_ns=int(job_end_ns),
                coherence_deadline_ns=(
                    request.release_time_ns
                    + REQUEST_COHERENCE_SLACK_NS
                ),
            )
        )

    return sorted(
        converted,
        key=lambda request: (
            request.release_time_ns,
            role_sort_key(request.role),
            request.tenant_id,
            request.logical_event_id,
            request.request_id,
        ),
    )


# =============================================================================
# Communication-qubit manager and simulator
# =============================================================================


class CommunicationQubitSimulator:
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        requests: Sequence[CommunicationRequest],
        all_jobs: Sequence[p1.JobSpec],
        scenario_seed: int,
    ) -> None:
        self.configuration = configuration
        self.requests = list(requests)
        self.all_jobs = list(all_jobs)
        self.scenario_seed = scenario_seed
        self.current_time_ns = 0
        self.release_index = 0
        self.waiting: list[PendingRequest] = []
        self.active_heap: list[ActiveService] = []
        self.completed: list[CompletedAllocation] = []
        self.failed: list[FailedAllocation] = []
        self.serial = 0

        self.total_conflicts = 0
        self.total_redirected_endpoints = 0
        self.grants_by_tenant: dict[str, int] = defaultdict(int)
        self.arrivals_by_tenant: dict[str, int] = defaultdict(int)
        self.waiting_sum_by_tenant: dict[str, float] = defaultdict(float)
        self.completed_by_tenant: dict[str, int] = defaultdict(int)
        self.round_robin_pointer: dict[str, int] = defaultdict(int)

        self.job_by_tenant = {
            job.tenant_id: job
            for job in self.all_jobs
        }
        self.requests_by_tenant = defaultdict(list)
        for request in self.requests:
            self.requests_by_tenant[request.tenant_id].append(request)

        self.local_qubits: dict[int, list[CommunicationQubit]] = {
            module: [
                CommunicationQubit(
                    pool_id=f"module_{module}",
                    module_id=module,
                    qubit_id=qubit_id,
                )
                for qubit_id in range(
                    configuration.communication_qubits_per_module
                )
            ]
            for module in range(NUM_MODULES)
        }

        global_count = (
            NUM_MODULES
            * configuration.communication_qubits_per_module
        )
        self.global_qubits = [
            CommunicationQubit(
                pool_id="global_pool",
                module_id=None,
                qubit_id=qubit_id,
            )
            for qubit_id in range(global_count)
        ]

        self.static_mapping: dict[tuple[str, int], int] = {}
        self.reserved_mapping: dict[tuple[str, int], int] = {}
        self.rejected_tenants: set[str] = set()
        self._prepare_policy_plan()
        if self.configuration.epr_prefetch_enabled:
            for qubit in self.all_qubits():
                self._schedule_prefetch(qubit, 0)

    # ---------------------------------------------------------------------
    # Plan preparation
    # ---------------------------------------------------------------------

    def endpoint_modules_by_tenant(self) -> dict[str, set[int]]:
        endpoints: dict[str, set[int]] = defaultdict(set)
        for request in self.requests:
            endpoints[request.tenant_id].update(request.endpoints)
        return endpoints

    def _prepare_policy_plan(self) -> None:
        endpoints = self.endpoint_modules_by_tenant()

        for tenant_id, modules in endpoints.items():
            for module in modules:
                deterministic_value = (
                    stable_text_seed(tenant_id)
                    + 17 * module
                )
                self.static_mapping[(tenant_id, module)] = (
                    deterministic_value
                    % self.configuration.communication_qubits_per_module
                )

        if self.configuration.allocation_policy != "per_tenant_reserved":
            return

        tenants_by_module: dict[int, list[str]] = defaultdict(list)
        role_by_tenant = {
            job.tenant_id: job.role
            for job in self.all_jobs
        }
        for tenant_id, modules in endpoints.items():
            for module in modules:
                tenants_by_module[module].append(tenant_id)

        for module, tenants in tenants_by_module.items():
            ordered = sorted(
                tenants,
                key=lambda tenant_id: (
                    role_sort_key(role_by_tenant.get(tenant_id, "background")),
                    tenant_id,
                ),
            )
            qubits = self.local_qubits[module]
            for index, tenant_id in enumerate(ordered):
                if index >= len(qubits):
                    self.rejected_tenants.add(tenant_id)
                    continue
                qubit = qubits[index]
                qubit.persistent_owner = tenant_id
                self.reserved_mapping[(tenant_id, module)] = qubit.qubit_id

        # Reservation is job-wide: if any required endpoint cannot be reserved,
        # the tenant is rejected consistently for all of its remote requests.
        for tenant_id, modules in endpoints.items():
            if any(
                (tenant_id, module) not in self.reserved_mapping
                for module in modules
            ):
                self.rejected_tenants.add(tenant_id)

    # ---------------------------------------------------------------------
    # Communication-qubit state management
    # ---------------------------------------------------------------------

    def all_qubits(self) -> list[CommunicationQubit]:
        if self.configuration.allocation_policy == "shared_global_pool":
            return self.global_qubits
        return [
            qubit
            for module_qubits in self.local_qubits.values()
            for qubit in module_qubits
        ]

    def _expire_lease_if_needed(
        self,
        qubit: CommunicationQubit,
        current_time_ns: int,
    ) -> None:
        if (
            qubit.lease_owner is not None
            and current_time_ns >= qubit.lease_until_ns
            and qubit.active_request is None
            and qubit.held_by_request is None
        ):
            qubit.lease_owner = None
            qubit.lease_key = None

    def _account_expired_epr(
        self,
        qubit: CommunicationQubit,
        current_time_ns: int,
    ) -> None:
        if not self.configuration.epr_prefetch_enabled:
            return
        if (
            qubit.epr_expiry_time_ns is not None
            and current_time_ns > qubit.epr_expiry_time_ns
        ):
            qubit.epr_wasted_count += 1
            qubit.epr_ready_time_ns = None
            qubit.epr_expiry_time_ns = None

    def _schedule_prefetch(
        self,
        qubit: CommunicationQubit,
        reset_completion_ns: int,
    ) -> None:
        if not self.configuration.epr_prefetch_enabled:
            qubit.epr_ready_time_ns = None
            qubit.epr_expiry_time_ns = None
            return
        qubit.epr_ready_time_ns = (
            reset_completion_ns
            + EPR_PREFETCH_LATENCY_NS
        )
        qubit.epr_expiry_time_ns = (
            qubit.epr_ready_time_ns
            + EPR_COHERENCE_TIME_NS
        )
        qubit.epr_prefetched_count += 1

    def _valid_prefetched_epr(
        self,
        qubit: CommunicationQubit,
        current_time_ns: int,
    ) -> bool:
        self._account_expired_epr(qubit, current_time_ns)
        return bool(
            qubit.epr_ready_time_ns is not None
            and qubit.epr_expiry_time_ns is not None
            and qubit.epr_ready_time_ns <= current_time_ns
            <= qubit.epr_expiry_time_ns
        )

    def _qubit_available_for(
        self,
        qubit: CommunicationQubit,
        tenant_id: str,
        current_time_ns: int,
    ) -> bool:
        self._expire_lease_if_needed(qubit, current_time_ns)
        self._account_expired_epr(qubit, current_time_ns)
        if qubit.active_request is not None:
            return False
        if qubit.held_by_request is not None:
            return False
        if current_time_ns < qubit.reset_until_ns:
            return False
        if (
            qubit.persistent_owner is not None
            and qubit.persistent_owner != tenant_id
        ):
            return False
        if (
            qubit.lease_owner is not None
            and current_time_ns < qubit.lease_until_ns
            and qubit.lease_owner != tenant_id
        ):
            return False
        return True

    # ---------------------------------------------------------------------
    # Candidate selection
    # ---------------------------------------------------------------------

    def _candidate_qubits(
        self,
        request: CommunicationRequest,
        endpoint_module: int,
        current_time_ns: int,
        exclude: CommunicationQubit | None = None,
    ) -> list[CommunicationQubit]:
        policy = self.configuration.allocation_policy

        if policy == "shared_global_pool":
            candidates = self.global_qubits
        else:
            candidates = self.local_qubits[endpoint_module]

        if policy == "static_assignment":
            qubit_id = self.static_mapping[(request.tenant_id, endpoint_module)]
            candidates = [self.local_qubits[endpoint_module][qubit_id]]
        elif policy == "per_tenant_reserved":
            qubit_id = self.reserved_mapping.get(
                (request.tenant_id, endpoint_module)
            )
            if qubit_id is None:
                return []
            candidates = [self.local_qubits[endpoint_module][qubit_id]]

        return [
            qubit
            for qubit in candidates
            if qubit is not exclude
            and self._qubit_available_for(
                qubit,
                request.tenant_id,
                current_time_ns,
            )
        ]

    def _request_random_value(
        self,
        request: CommunicationRequest,
        endpoint_module: int,
    ) -> int:
        return (
            self.scenario_seed
            + stable_text_seed(request.tenant_id)
            + request.logical_event_id * 1_009
            + endpoint_module * 131
        )

    def _choose_qubit(
        self,
        request: CommunicationRequest,
        endpoint_module: int,
        candidates: list[CommunicationQubit],
        current_time_ns: int,
    ) -> CommunicationQubit:
        if not candidates:
            raise ValueError("Cannot choose from an empty candidate set.")

        policy = self.configuration.allocation_policy

        if policy == "round_robin":
            pool_key = (
                "global"
                if policy == "shared_global_pool"
                else f"module_{endpoint_module}"
            )
            ordered = sorted(candidates, key=lambda qubit: qubit.qubit_id)
            pointer = self.round_robin_pointer[pool_key] % len(ordered)
            selected = ordered[pointer]
            self.round_robin_pointer[pool_key] += 1
            return selected

        if policy == "random_allocation":
            rng = random.Random(
                self._request_random_value(request, endpoint_module)
            )
            return rng.choice(sorted(candidates, key=lambda qubit: qubit.qubit_id))

        if policy == "fair_share":
            return min(
                candidates,
                key=lambda qubit: (
                    qubit.grant_count,
                    qubit.last_tenant != request.tenant_id,
                    qubit.qubit_id,
                ),
            )

        if policy == "coherence_aware":
            return min(
                candidates,
                key=lambda qubit: (
                    0 if self._valid_prefetched_epr(qubit, current_time_ns) else 1,
                    (
                        qubit.epr_expiry_time_ns
                        if qubit.epr_expiry_time_ns is not None
                        else math.inf
                    ),
                    qubit.qubit_id,
                ),
            )

        if policy == "predicted_demand":
            return min(
                candidates,
                key=lambda qubit: (
                    qubit.last_tenant != request.tenant_id,
                    qubit.grant_count,
                    qubit.qubit_id,
                ),
            )

        # Static, reserved, first available, priority, and shared pool use the
        # first physically available candidate under deterministic ordering.
        return min(
            candidates,
            key=lambda qubit: (
                qubit.reset_until_ns,
                qubit.qubit_id,
            ),
        )

    # ---------------------------------------------------------------------
    # Waiting-queue ordering
    # ---------------------------------------------------------------------

    def _future_demand(
        self,
        request: CommunicationRequest,
        current_time_ns: int,
    ) -> int:
        horizon_end = current_time_ns + PREDICTION_HORIZON_NS
        return sum(
            1
            for future in self.requests_by_tenant[request.tenant_id]
            if current_time_ns <= future.release_time_ns <= horizon_end
            and bool(set(future.endpoints) & set(request.endpoints))
        )

    def _pending_sort_key(
        self,
        pending: PendingRequest,
    ) -> tuple[Any, ...]:
        request = pending.request
        policy = self.configuration.allocation_policy

        if policy == "priority_allocation":
            return (
                role_sort_key(request.role),
                request.release_time_ns,
                request.request_id,
            )

        if policy == "fair_share":
            arrivals = max(1, self.arrivals_by_tenant[request.tenant_id])
            served_fraction = (
                self.grants_by_tenant[request.tenant_id]
                / arrivals
            )
            return (
                served_fraction,
                request.release_time_ns,
                request.request_id,
            )

        if policy == "coherence_aware":
            return (
                request.coherence_deadline_ns,
                request.release_time_ns,
                request.request_id,
            )

        if policy == "predicted_demand":
            return (
                -self._future_demand(request, self.current_time_ns),
                request.release_time_ns,
                request.request_id,
            )

        return (
            request.release_time_ns,
            role_sort_key(request.role),
            request.tenant_id,
            request.request_id,
        )

    # ---------------------------------------------------------------------
    # Grant, service, lease, and completion
    # ---------------------------------------------------------------------

    def _grant_endpoint(
        self,
        pending: PendingRequest,
        endpoint_name: str,
        endpoint_module: int,
        exclude: CommunicationQubit | None = None,
    ) -> bool:
        request = pending.request
        candidates = self._candidate_qubits(
            request,
            endpoint_module,
            self.current_time_ns,
            exclude=exclude,
        )
        if not candidates:
            pending.conflict_checks += 1
            self.total_conflicts += 1
            return False

        selected = self._choose_qubit(
            request,
            endpoint_module,
            candidates,
            self.current_time_ns,
        )
        selected.held_by_request = request.request_id
        selected.grant_count += 1

        if endpoint_name == "source":
            pending.source_qubit = selected
            pending.source_grant_time_ns = self.current_time_ns
        else:
            pending.target_qubit = selected
            pending.target_grant_time_ns = self.current_time_ns

        if selected.module_id is None:
            pending.redirected_endpoint_count += 1
            self.total_redirected_endpoints += 1

        return True

    def _atomic_grant(self, pending: PendingRequest) -> bool:
        request = pending.request
        source_candidates = self._candidate_qubits(
            request,
            request.source_module,
            self.current_time_ns,
        )
        if not source_candidates:
            pending.conflict_checks += 1
            self.total_conflicts += 1
            return False

        source = self._choose_qubit(
            request,
            request.source_module,
            source_candidates,
            self.current_time_ns,
        )
        target_candidates = self._candidate_qubits(
            request,
            request.target_module,
            self.current_time_ns,
            exclude=source,
        )
        if not target_candidates:
            pending.conflict_checks += 1
            self.total_conflicts += 1
            return False

        target = self._choose_qubit(
            request,
            request.target_module,
            target_candidates,
            self.current_time_ns,
        )

        source.held_by_request = request.request_id
        target.held_by_request = request.request_id
        source.grant_count += 1
        target.grant_count += 1

        pending.source_qubit = source
        pending.target_qubit = target
        pending.source_grant_time_ns = self.current_time_ns
        pending.target_grant_time_ns = self.current_time_ns

        if source.module_id is None:
            pending.redirected_endpoint_count += 1
        if target.module_id is None:
            pending.redirected_endpoint_count += 1
        self.total_redirected_endpoints += pending.redirected_endpoint_count
        return True

    def _progress_local_grant(self, pending: PendingRequest) -> bool:
        request = pending.request
        # Ordered acquisition avoids artificial circular wait while preserving
        # local hold-and-wait behavior.
        endpoint_order = sorted(
            [
                (request.source_module, "source"),
                (request.target_module, "target"),
            ],
            key=lambda item: (item[0], item[1]),
        )

        progress = False
        for endpoint_module, endpoint_name in endpoint_order:
            if endpoint_name == "source" and pending.source_qubit is not None:
                continue
            if endpoint_name == "target" and pending.target_qubit is not None:
                continue

            exclude = (
                pending.target_qubit
                if endpoint_name == "source"
                else pending.source_qubit
            )
            granted = self._grant_endpoint(
                pending,
                endpoint_name,
                endpoint_module,
                exclude=exclude,
            )
            progress = progress or granted
            if not granted:
                break

        return progress

    def _consume_prefetched_pair(
        self,
        source: CommunicationQubit,
        target: CommunicationQubit,
    ) -> bool:
        source_valid = self._valid_prefetched_epr(
            source,
            self.current_time_ns,
        )
        target_valid = self._valid_prefetched_epr(
            target,
            self.current_time_ns,
        )

        if source_valid and target_valid:
            source.epr_used_count += 1
            target.epr_used_count += 1
            source.epr_ready_time_ns = None
            source.epr_expiry_time_ns = None
            target.epr_ready_time_ns = None
            target.epr_expiry_time_ns = None
            return True

        # An unmatched prefetched half cannot be used by this remote operation.
        for qubit, valid in ((source, source_valid), (target, target_valid)):
            if valid:
                qubit.epr_wasted_count += 1
                qubit.epr_ready_time_ns = None
                qubit.epr_expiry_time_ns = None
        return False

    def _lease_key_and_end(
        self,
        request: CommunicationRequest,
        completion_time_ns: int,
    ) -> tuple[str, int]:
        granularity = self.configuration.allocation_granularity
        reset_end = completion_time_ns + self.configuration.reset_time_ns

        if granularity == "per_job":
            return (
                f"job:{request.tenant_id}",
                max(
                    reset_end,
                    request.job_end_ns
                    + self.configuration.reservation_duration_ns,
                ),
            )

        if granularity == "per_layer":
            return (
                f"layer:{request.tenant_id}:{request.layer_id}",
                max(
                    reset_end,
                    request.layer_end_ns
                    + self.configuration.reservation_duration_ns,
                ),
            )

        return (
            f"operation:{request.tenant_id}:{request.request_id}",
            reset_end
            + self.configuration.reservation_duration_ns,
        )

    def _apply_lease(
        self,
        qubit: CommunicationQubit,
        request: CommunicationRequest,
        completion_time_ns: int,
    ) -> None:
        lease_key, lease_end = self._lease_key_and_end(
            request,
            completion_time_ns,
        )
        old_end = max(self.current_time_ns, qubit.lease_until_ns)
        if lease_end > old_end:
            qubit.reserved_capacity_ns += lease_end - old_end
        qubit.lease_owner = request.tenant_id
        qubit.lease_key = lease_key
        qubit.lease_until_ns = max(qubit.lease_until_ns, lease_end)

    def _start_service(self, pending: PendingRequest) -> bool:
        if len(self.active_heap) >= HUB_CAPACITY:
            pending.conflict_checks += 1
            self.total_conflicts += 1
            return False
        if pending.source_qubit is None or pending.target_qubit is None:
            return False

        source = pending.source_qubit
        target = pending.target_qubit
        request = pending.request

        used_prefetch = self._consume_prefetched_pair(source, target)
        service_time = (
            PREFETCHED_REMOTE_SERVICE_NS
            if used_prefetch
            else BASE_REMOTE_SERVICE_NS
        )
        completion_time = self.current_time_ns + service_time
        reset_completion = (
            completion_time
            + self.configuration.reset_time_ns
        )

        source_reuse_delay = max(
            0,
            self.current_time_ns
            - source.last_reset_completion_ns,
        )
        target_reuse_delay = max(
            0,
            self.current_time_ns
            - target.last_reset_completion_ns,
        )

        source.held_by_request = None
        target.held_by_request = None
        source.active_request = request.request_id
        target.active_request = request.request_id
        source.last_tenant = request.tenant_id
        target.last_tenant = request.tenant_id
        source.service_busy_ns += service_time
        target.service_busy_ns += service_time

        self._apply_lease(source, request, completion_time)
        self._apply_lease(target, request, completion_time)

        self.grants_by_tenant[request.tenant_id] += 1

        heapq.heappush(
            self.active_heap,
            ActiveService(
                completion_time_ns=completion_time,
                serial=self.serial,
                pending=pending,
                service_start_time_ns=self.current_time_ns,
                service_time_ns=service_time,
                source_reuse_delay_ns=source_reuse_delay,
                target_reuse_delay_ns=target_reuse_delay,
                used_prefetched_epr=used_prefetch,
                reset_completion_time_ns=reset_completion,
            ),
        )
        self.serial += 1
        return True

    def _complete_due_services(self) -> None:
        while (
            self.active_heap
            and self.active_heap[0].completion_time_ns <= self.current_time_ns
        ):
            active = heapq.heappop(self.active_heap)
            pending = active.pending
            request = pending.request
            source = pending.source_qubit
            target = pending.target_qubit
            if source is None or target is None:
                raise RuntimeError("Active service lost its communication qubits.")

            for qubit in (source, target):
                qubit.active_request = None
                qubit.last_occupancy_end_ns = active.completion_time_ns
                qubit.reset_until_ns = active.reset_completion_time_ns
                qubit.last_reset_completion_ns = active.reset_completion_time_ns
                qubit.reset_busy_ns += self.configuration.reset_time_ns
                self._schedule_prefetch(
                    qubit,
                    active.reset_completion_time_ns,
                )

            self.waiting_sum_by_tenant[request.tenant_id] += (
                active.service_start_time_ns
                - request.release_time_ns
            )
            self.completed_by_tenant[request.tenant_id] += 1

            self.completed.append(
                CompletedAllocation(
                    request=request,
                    source_qubit_label=source.label,
                    target_qubit_label=target.label,
                    allocation_request_time_ns=request.release_time_ns,
                    source_grant_time_ns=int(pending.source_grant_time_ns),
                    target_grant_time_ns=int(pending.target_grant_time_ns),
                    allocation_grant_time_ns=int(
                        pending.allocation_grant_time_ns
                    ),
                    occupancy_start_time_ns=active.service_start_time_ns,
                    occupancy_end_time_ns=active.completion_time_ns,
                    reset_start_time_ns=active.completion_time_ns,
                    reset_completion_time_ns=active.reset_completion_time_ns,
                    source_reuse_delay_ns=active.source_reuse_delay_ns,
                    target_reuse_delay_ns=active.target_reuse_delay_ns,
                    service_time_ns=active.service_time_ns,
                    used_prefetched_epr=active.used_prefetched_epr,
                    conflict_checks=pending.conflict_checks,
                    partial_hold_time_ns=pending.partial_hold_time_ns,
                    redirected_endpoint_count=pending.redirected_endpoint_count,
                )
            )

    def _release_partial_hold(self, pending: PendingRequest) -> None:
        for qubit in (pending.source_qubit, pending.target_qubit):
            if (
                qubit is not None
                and qubit.held_by_request == pending.request.request_id
            ):
                qubit.held_by_request = None

    def _fail_timed_out_requests(self) -> bool:
        progress = False
        retained: list[PendingRequest] = []
        for pending in self.waiting:
            request = pending.request
            if (
                self.current_time_ns
                >= request.release_time_ns + MAX_ALLOCATION_WAIT_NS
            ):
                self._release_partial_hold(pending)
                self.failed.append(
                    FailedAllocation(
                        request=request,
                        failure_time_ns=self.current_time_ns,
                        failure_reason="allocation_timeout",
                        conflict_checks=pending.conflict_checks,
                        partial_hold_time_ns=pending.partial_hold_time_ns,
                        redirected_endpoint_count=(
                            pending.redirected_endpoint_count
                        ),
                    )
                )
                progress = True
            else:
                retained.append(pending)
        self.waiting = retained
        return progress

    def _account_partial_hold_interval(self, delta_ns: int) -> None:
        if delta_ns <= 0:
            return
        for pending in self.waiting:
            one_granted = (
                (pending.source_qubit is None)
                != (pending.target_qubit is None)
            )
            if one_granted:
                pending.partial_hold_time_ns += delta_ns
                held_qubit = (
                    pending.source_qubit
                    if pending.source_qubit is not None
                    else pending.target_qubit
                )
                if held_qubit is not None:
                    held_qubit.allocation_hold_busy_ns += delta_ns

    def _admit_and_start_waiting(self) -> bool:
        progress = False
        ordered = sorted(self.waiting, key=self._pending_sort_key)

        for pending in ordered:
            if pending not in self.waiting:
                continue

            if pending.request.tenant_id in self.rejected_tenants:
                self._release_partial_hold(pending)
                self.failed.append(
                    FailedAllocation(
                        request=pending.request,
                        failure_time_ns=self.current_time_ns,
                        failure_reason="tenant_reservation_rejected",
                        conflict_checks=pending.conflict_checks,
                        partial_hold_time_ns=pending.partial_hold_time_ns,
                        redirected_endpoint_count=(
                            pending.redirected_endpoint_count
                        ),
                    )
                )
                self.waiting.remove(pending)
                progress = True
                continue

            if (
                pending.source_qubit is None
                or pending.target_qubit is None
            ):
                if self.configuration.allocation_coordination == "global":
                    if (
                        pending.source_qubit is None
                        and pending.target_qubit is None
                    ):
                        granted = self._atomic_grant(pending)
                        progress = progress or granted
                else:
                    granted = self._progress_local_grant(pending)
                    progress = progress or granted

            if (
                pending.source_qubit is not None
                and pending.target_qubit is not None
            ):
                if self._start_service(pending):
                    self.waiting.remove(pending)
                    progress = True

        return progress

    # ---------------------------------------------------------------------
    # Event loop
    # ---------------------------------------------------------------------

    def _next_state_change_time(self) -> int | None:
        candidates: list[int] = []

        if self.release_index < len(self.requests):
            candidates.append(
                self.requests[self.release_index].release_time_ns
            )

        if self.active_heap:
            candidates.append(self.active_heap[0].completion_time_ns)

        for qubit in self.all_qubits():
            if qubit.reset_until_ns > self.current_time_ns:
                candidates.append(qubit.reset_until_ns)
            if (
                qubit.lease_owner is not None
                and qubit.lease_until_ns > self.current_time_ns
            ):
                candidates.append(qubit.lease_until_ns)
            if (
                qubit.epr_ready_time_ns is not None
                and qubit.epr_ready_time_ns > self.current_time_ns
            ):
                candidates.append(qubit.epr_ready_time_ns)
            if (
                qubit.epr_expiry_time_ns is not None
                and qubit.epr_expiry_time_ns > self.current_time_ns
            ):
                candidates.append(qubit.epr_expiry_time_ns + 1)

        for pending in self.waiting:
            timeout = (
                pending.request.release_time_ns
                + MAX_ALLOCATION_WAIT_NS
            )
            if timeout > self.current_time_ns:
                candidates.append(timeout)

        future = [
            candidate
            for candidate in candidates
            if candidate > self.current_time_ns
        ]
        if not future:
            return None
        return min(future)

    def run(self) -> ScenarioResult:
        if self.requests:
            self.current_time_ns = min(
                0,
                self.requests[0].release_time_ns,
            )

        while (
            self.release_index < len(self.requests)
            or self.waiting
            or self.active_heap
        ):
            while (
                self.release_index < len(self.requests)
                and self.requests[self.release_index].release_time_ns
                <= self.current_time_ns
            ):
                request = self.requests[self.release_index]
                self.arrivals_by_tenant[request.tenant_id] += 1
                self.waiting.append(PendingRequest(request=request))
                self.release_index += 1

            self._complete_due_services()
            self._fail_timed_out_requests()

            progress = True
            while progress:
                progress = self._admit_and_start_waiting()
                self._complete_due_services()
                self._fail_timed_out_requests()

            if (
                self.release_index >= len(self.requests)
                and not self.waiting
                and not self.active_heap
            ):
                break

            next_time = self._next_state_change_time()
            if next_time is None:
                # Any remaining request is structurally impossible. Fail it
                # explicitly instead of silently deadlocking.
                for pending in list(self.waiting):
                    self._release_partial_hold(pending)
                    self.failed.append(
                        FailedAllocation(
                            request=pending.request,
                            failure_time_ns=self.current_time_ns,
                            failure_reason="no_future_allocation_event",
                            conflict_checks=pending.conflict_checks,
                            partial_hold_time_ns=pending.partial_hold_time_ns,
                            redirected_endpoint_count=(
                                pending.redirected_endpoint_count
                            ),
                        )
                    )
                self.waiting.clear()
                break

            delta_ns = next_time - self.current_time_ns
            self._account_partial_hold_interval(delta_ns)
            self.current_time_ns = next_time

        # Account prefetched pairs that remain unused at the observation end.
        for qubit in self.all_qubits():
            self._account_expired_epr(qubit, self.current_time_ns)
            if (
                qubit.epr_ready_time_ns is not None
                and qubit.epr_expiry_time_ns is not None
            ):
                qubit.epr_stranded_count += 1

        request_dataframe = self._request_dataframe()
        qubit_dataframe = self._qubit_dataframe()
        tenant_dataframe = self._tenant_dataframe(request_dataframe)

        resource_horizon = max(
            [self.current_time_ns]
            + [qubit.reset_until_ns for qubit in self.all_qubits()]
            + [qubit.lease_until_ns for qubit in self.all_qubits()]
        )
        makespan = max(
            resource_horizon,
            int(request_dataframe["completion_time_ns"].max())
            if (
                not request_dataframe.empty
                and request_dataframe["completion_time_ns"].notna().any()
            )
            else resource_horizon,
        )
        denominator = max(1, makespan * max(1, len(self.all_qubits())))

        total_service = sum(qubit.service_busy_ns for qubit in self.all_qubits())
        total_reset = sum(qubit.reset_busy_ns for qubit in self.all_qubits())
        total_hold = sum(
            qubit.allocation_hold_busy_ns
            for qubit in self.all_qubits()
        )
        total_reserved = sum(
            qubit.reserved_capacity_ns
            for qubit in self.all_qubits()
        )

        grants = tenant_dataframe["completed_request_count"].to_numpy(dtype=float)
        inverse_waits = 1.0 / (
            1.0
            + tenant_dataframe["mean_waiting_time_ns"].to_numpy(dtype=float)
        )

        return ScenarioResult(
            request_dataframe=request_dataframe,
            qubit_dataframe=qubit_dataframe,
            tenant_dataframe=tenant_dataframe,
            makespan_ns=int(makespan),
            rejected_tenants=set(self.rejected_tenants),
            total_conflicts=int(self.total_conflicts),
            total_redirected_endpoints=int(self.total_redirected_endpoints),
            total_failed_requests=len(self.failed),
            total_prefetched_pairs=sum(
                qubit.epr_prefetched_count
                for qubit in self.all_qubits()
            ),
            total_used_prefetched_pairs=sum(
                qubit.epr_used_count
                for qubit in self.all_qubits()
            ),
            total_wasted_prefetched_pairs=sum(
                qubit.epr_wasted_count
                for qubit in self.all_qubits()
            ),
            total_stranded_prefetched_pairs=sum(
                qubit.epr_stranded_count
                for qubit in self.all_qubits()
            ),
            communication_qubit_utilization=float(total_service / denominator),
            reset_utilization=float(total_reset / denominator),
            allocation_hold_utilization=float(total_hold / denominator),
            reservation_utilization=float(total_reserved / denominator),
            effective_communication_qubit_utilization=float(
                (total_service + total_reset + total_hold + total_reserved)
                / denominator
            ),
            jain_grant_fairness=jain_index(grants),
            jain_inverse_wait_fairness=jain_index(inverse_waits),
        )

    # ---------------------------------------------------------------------
    # Dataframe conversion
    # ---------------------------------------------------------------------

    def _request_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for item in self.completed:
            request = item.request
            rows.append(
                {
                    "status": "completed",
                    "failure_reason": "",
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "role": request.role,
                    "logical_event_id": request.logical_event_id,
                    "release_time_ns": request.release_time_ns,
                    "source_module": request.source_module,
                    "target_module": request.target_module,
                    "layer_id": request.layer_id,
                    "coherence_deadline_ns": request.coherence_deadline_ns,
                    "source_communication_qubit": item.source_qubit_label,
                    "target_communication_qubit": item.target_qubit_label,
                    "allocation_request_time_ns": item.allocation_request_time_ns,
                    "source_grant_time_ns": item.source_grant_time_ns,
                    "target_grant_time_ns": item.target_grant_time_ns,
                    "allocation_grant_time_ns": item.allocation_grant_time_ns,
                    "occupancy_start_time_ns": item.occupancy_start_time_ns,
                    "occupancy_end_time_ns": item.occupancy_end_time_ns,
                    "reset_start_time_ns": item.reset_start_time_ns,
                    "reset_completion_time_ns": item.reset_completion_time_ns,
                    "source_reuse_delay_ns": item.source_reuse_delay_ns,
                    "target_reuse_delay_ns": item.target_reuse_delay_ns,
                    "communication_qubit_queue_delay_ns": (
                        item.communication_qubit_queue_delay_ns
                    ),
                    "service_queue_delay_ns": item.service_queue_delay_ns,
                    "waiting_time_ns": item.waiting_time_ns,
                    "service_time_ns": item.service_time_ns,
                    "completion_time_ns": item.occupancy_end_time_ns,
                    "turnaround_time_ns": item.turnaround_time_ns,
                    "allocation_conflict_checks": item.conflict_checks,
                    "partial_hold_time_ns": item.partial_hold_time_ns,
                    "request_failed": False,
                    "request_redirected": item.redirected_endpoint_count > 0,
                    "redirected_endpoint_count": item.redirected_endpoint_count,
                    "used_prefetched_epr": item.used_prefetched_epr,
                }
            )

        for item in self.failed:
            request = item.request
            rows.append(
                {
                    "status": "failed",
                    "failure_reason": item.failure_reason,
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "role": request.role,
                    "logical_event_id": request.logical_event_id,
                    "release_time_ns": request.release_time_ns,
                    "source_module": request.source_module,
                    "target_module": request.target_module,
                    "layer_id": request.layer_id,
                    "coherence_deadline_ns": request.coherence_deadline_ns,
                    "source_communication_qubit": "",
                    "target_communication_qubit": "",
                    "allocation_request_time_ns": request.release_time_ns,
                    "source_grant_time_ns": np.nan,
                    "target_grant_time_ns": np.nan,
                    "allocation_grant_time_ns": np.nan,
                    "occupancy_start_time_ns": np.nan,
                    "occupancy_end_time_ns": np.nan,
                    "reset_start_time_ns": np.nan,
                    "reset_completion_time_ns": np.nan,
                    "source_reuse_delay_ns": np.nan,
                    "target_reuse_delay_ns": np.nan,
                    "communication_qubit_queue_delay_ns": np.nan,
                    "service_queue_delay_ns": np.nan,
                    "waiting_time_ns": np.nan,
                    "service_time_ns": np.nan,
                    "completion_time_ns": np.nan,
                    "turnaround_time_ns": np.nan,
                    "allocation_conflict_checks": item.conflict_checks,
                    "partial_hold_time_ns": item.partial_hold_time_ns,
                    "request_failed": True,
                    "request_redirected": item.redirected_endpoint_count > 0,
                    "redirected_endpoint_count": item.redirected_endpoint_count,
                    "used_prefetched_epr": False,
                }
            )

        columns = [
            "status",
            "failure_reason",
            "request_id",
            "tenant_id",
            "role",
            "logical_event_id",
            "release_time_ns",
            "source_module",
            "target_module",
            "layer_id",
            "coherence_deadline_ns",
            "source_communication_qubit",
            "target_communication_qubit",
            "allocation_request_time_ns",
            "source_grant_time_ns",
            "target_grant_time_ns",
            "allocation_grant_time_ns",
            "occupancy_start_time_ns",
            "occupancy_end_time_ns",
            "reset_start_time_ns",
            "reset_completion_time_ns",
            "source_reuse_delay_ns",
            "target_reuse_delay_ns",
            "communication_qubit_queue_delay_ns",
            "service_queue_delay_ns",
            "waiting_time_ns",
            "service_time_ns",
            "completion_time_ns",
            "turnaround_time_ns",
            "allocation_conflict_checks",
            "partial_hold_time_ns",
            "request_failed",
            "request_redirected",
            "redirected_endpoint_count",
            "used_prefetched_epr",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(
                [
                    "release_time_ns",
                    "role",
                    "tenant_id",
                    "logical_event_id",
                    "request_id",
                ]
            )
            .reset_index(drop=True)
        )

    def _qubit_dataframe(self) -> pd.DataFrame:
        rows = []
        for qubit in self.all_qubits():
            rows.append(
                {
                    "communication_qubit": qubit.label,
                    "module_id": qubit.module_id,
                    "persistent_owner": qubit.persistent_owner or "",
                    "service_busy_ns": qubit.service_busy_ns,
                    "reset_busy_ns": qubit.reset_busy_ns,
                    "allocation_hold_busy_ns": qubit.allocation_hold_busy_ns,
                    "reserved_capacity_ns": qubit.reserved_capacity_ns,
                    "grant_count": qubit.grant_count,
                    "epr_prefetched_count": qubit.epr_prefetched_count,
                    "epr_used_count": qubit.epr_used_count,
                    "epr_wasted_count": qubit.epr_wasted_count,
                    "epr_stranded_count": qubit.epr_stranded_count,
                }
            )
        return pd.DataFrame(rows)

    def _tenant_dataframe(self, requests: pd.DataFrame) -> pd.DataFrame:
        tenant_ids = sorted(
            {
                job.tenant_id
                for job in self.all_jobs
            }
            | set(self.rejected_tenants)
        )
        rows = []
        for tenant_id in tenant_ids:
            subset = requests[requests["tenant_id"] == tenant_id]
            completed = subset[subset["status"] == "completed"]
            rows.append(
                {
                    "tenant_id": tenant_id,
                    "role": (
                        self.job_by_tenant[tenant_id].role
                        if tenant_id in self.job_by_tenant
                        else "unknown"
                    ),
                    "tenant_rejected": tenant_id in self.rejected_tenants,
                    "request_count": len(subset),
                    "completed_request_count": len(completed),
                    "failed_request_count": int(
                        (subset["status"] == "failed").sum()
                    ),
                    "mean_waiting_time_ns": safe_mean(
                        completed["waiting_time_ns"]
                    ),
                    "mean_allocation_queue_delay_ns": safe_mean(
                        completed["communication_qubit_queue_delay_ns"]
                    ),
                }
            )
        return pd.DataFrame(rows)


# =============================================================================
# Scenario execution and control comparison
# =============================================================================


def run_scenario(
    configuration: ExperimentConfiguration,
    jobs: Sequence[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    included_roles: set[str],
    scenario_seed: int,
) -> ScenarioResult:
    requests = communication_requests(
        jobs,
        allocations,
        included_roles,
    )
    simulator = CommunicationQubitSimulator(
        configuration=configuration,
        requests=requests,
        all_jobs=[job for job in jobs if job.role in included_roles],
        scenario_seed=scenario_seed,
    )
    return simulator.run()


def compare_attacker_traces(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    baseline = attacker_only[attacker_only["role"] == "attacker"].copy()
    victim_on = combined[combined["role"] == "attacker"].copy()

    keys = [
        "logical_event_id",
        "release_time_ns",
        "source_module",
        "target_module",
    ]

    baseline = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in baseline.columns
            if column not in keys
        }
    )
    victim_on = victim_on.rename(
        columns={
            column: f"combined_{column}"
            for column in victim_on.columns
            if column not in keys
        }
    )

    merged = victim_on.merge(
        baseline,
        on=keys,
        how="outer",
        validate="one_to_one",
    )

    merged["baseline_completed"] = (
        merged["baseline_status"] == "completed"
    )
    merged["combined_completed"] = (
        merged["combined_status"] == "completed"
    )
    merged["combined_failed"] = ~merged["combined_completed"]

    both_completed = (
        merged["baseline_completed"]
        & merged["combined_completed"]
    )

    merged["excess_allocation_queue_delay_ns"] = np.where(
        both_completed,
        merged["combined_communication_qubit_queue_delay_ns"]
        - merged["baseline_communication_qubit_queue_delay_ns"],
        np.nan,
    )
    merged["excess_waiting_time_ns"] = np.where(
        both_completed,
        merged["combined_waiting_time_ns"]
        - merged["baseline_waiting_time_ns"],
        np.nan,
    )
    merged["excess_turnaround_time_ns"] = np.where(
        both_completed,
        merged["combined_turnaround_time_ns"]
        - merged["baseline_turnaround_time_ns"],
        np.nan,
    )

    # Failure of a probe that completed in the attacker-only calibration is a
    # strong observable event. Use the allocation timeout as an effective
    # lower-bound penalty for unconditional leakage summaries.
    merged["effective_excess_turnaround_time_ns"] = np.where(
        both_completed,
        merged["excess_turnaround_time_ns"],
        np.where(
            merged["baseline_completed"]
            & merged["combined_failed"],
            MAX_ALLOCATION_WAIT_NS,
            0.0,
        ),
    )

    merged["victim_contention_observed"] = (
        merged["baseline_completed"]
        & (
            merged["combined_failed"]
            | (
                merged["excess_turnaround_time_ns"].fillna(0)
                > DETECTION_THRESHOLD_NS
            )
        )
    )

    merged["allocation_changed"] = (
        (
            merged["baseline_source_communication_qubit"].fillna("")
            != merged["combined_source_communication_qubit"].fillna("")
        )
        | (
            merged["baseline_target_communication_qubit"].fillna("")
            != merged["combined_target_communication_qubit"].fillna("")
        )
    )

    return merged.sort_values("logical_event_id").reset_index(drop=True)


def victim_completion_time(
    result: ScenarioResult,
) -> float:
    victim = result.request_dataframe[
        (result.request_dataframe["role"] == "victim")
        & (result.request_dataframe["status"] == "completed")
    ]
    if victim.empty:
        return 0.0
    return float(
        victim["completion_time_ns"].max()
        - p1.VICTIM_START_NS
    )


# =============================================================================
# Trial metrics
# =============================================================================


def trial_row(
    victim_name: str,
    configuration: ExperimentConfiguration,
    trial_id: int,
    attacker_only: ScenarioResult,
    victim_only: ScenarioResult,
    combined: ScenarioResult,
    compared: pd.DataFrame,
) -> dict[str, Any]:
    eligible = compared[compared["baseline_completed"]]
    completed_pairs = eligible[eligible["combined_completed"]]

    victim_only_completion = victim_completion_time(victim_only)
    combined_victim_completion = victim_completion_time(combined)
    victim_failed_in_combined = bool(
        (
            (combined.request_dataframe["role"] == "victim")
            & (combined.request_dataframe["status"] == "failed")
        ).any()
    )

    if victim_only_completion > 0 and combined_victim_completion > 0:
        victim_slowdown_ratio = (
            combined_victim_completion
            / victim_only_completion
        )
    else:
        victim_slowdown_ratio = np.nan

    detection_probability = (
        float(eligible["victim_contention_observed"].mean())
        if not eligible.empty
        else 0.0
    )

    # A simple balanced victim-presence score with zero false positives under
    # exact differential calibration. It is reported as a detection baseline,
    # not as a learned classifier.
    balanced_detection_accuracy = 0.5 * (1.0 + detection_probability)

    combined_attacker = combined.request_dataframe[
        combined.request_dataframe["role"] == "attacker"
    ]
    victim_only_victim = victim_only.request_dataframe[
        victim_only.request_dataframe["role"] == "victim"
    ]

    return {
        "victim_qasm": victim_name,
        "victim_tag": p1.safe_tag(victim_name),
        "trial_id": trial_id,
        "configuration_id": configuration.configuration_id,
        **configuration.__dict__,
        "hub_capacity": HUB_CAPACITY,
        "attacker_probe_count": len(eligible),
        "attacker_baseline_rejected": (
            "attacker" in attacker_only.rejected_tenants
        ),
        "attacker_combined_rejected": (
            "attacker" in combined.rejected_tenants
        ),
        "victim_only_rejected": (
            "victim" in victim_only.rejected_tenants
        ),
        "victim_combined_rejected": (
            "victim" in combined.rejected_tenants
        ),
        "attacker_combined_failed_request_count": int(
            (
                combined_attacker["status"] == "failed"
            ).sum()
        ),
        "victim_combined_failed_request_count": int(
            (
                (combined.request_dataframe["role"] == "victim")
                & (combined.request_dataframe["status"] == "failed")
            ).sum()
        ),
        "baseline_avg_communication_qubit_queue_delay_ns": safe_mean(
            attacker_only.request_dataframe.loc[
                (attacker_only.request_dataframe["role"] == "attacker")
                & (attacker_only.request_dataframe["status"] == "completed"),
                "communication_qubit_queue_delay_ns",
            ]
        ),
        "combined_avg_communication_qubit_queue_delay_ns": safe_mean(
            combined_attacker.loc[
                combined_attacker["status"] == "completed",
                "communication_qubit_queue_delay_ns",
            ]
        ),
        "avg_excess_communication_qubit_queue_delay_ns": safe_mean(
            completed_pairs["excess_allocation_queue_delay_ns"]
        ),
        "avg_excess_turnaround_time_ns": safe_mean(
            completed_pairs["excess_turnaround_time_ns"]
        ),
        "max_excess_turnaround_time_ns": safe_max(
            completed_pairs["excess_turnaround_time_ns"]
        ),
        "total_effective_excess_turnaround_time_ns": float(
            eligible["effective_excess_turnaround_time_ns"].sum()
        ) if not eligible.empty else 0.0,
        "delayed_or_failed_probe_count": int(
            eligible["victim_contention_observed"].sum()
        ) if not eligible.empty else 0,
        "attacker_detection_probability": detection_probability,
        "balanced_detection_accuracy": balanced_detection_accuracy,
        "allocation_change_probability": float(
            eligible["allocation_changed"].mean()
        ) if not eligible.empty else 0.0,
        "allocation_conflict_count": combined.total_conflicts,
        "redirected_endpoint_count": combined.total_redirected_endpoints,
        "combined_failed_request_count": combined.total_failed_requests,
        "combined_request_failure_rate": (
            combined.total_failed_requests
            / max(1, len(combined.request_dataframe))
        ),
        "communication_qubit_utilization": (
            combined.communication_qubit_utilization
        ),
        "reset_utilization": combined.reset_utilization,
        "allocation_hold_utilization": combined.allocation_hold_utilization,
        "reservation_utilization": combined.reservation_utilization,
        "effective_communication_qubit_utilization": (
            combined.effective_communication_qubit_utilization
        ),
        "epr_prefetched_pair_count": combined.total_prefetched_pairs,
        "epr_used_pair_count": combined.total_used_prefetched_pairs,
        "epr_wasted_pair_count": combined.total_wasted_prefetched_pairs,
        "epr_stranded_pair_count": combined.total_stranded_prefetched_pairs,
        "epr_pair_wastage_fraction": (
            (
                combined.total_wasted_prefetched_pairs
                + combined.total_stranded_prefetched_pairs
            )
            / max(1, combined.total_prefetched_pairs)
        ),
        "jain_grant_fairness": combined.jain_grant_fairness,
        "jain_inverse_wait_fairness": (
            combined.jain_inverse_wait_fairness
        ),
        "victim_only_completion_time_ns": victim_only_completion,
        "combined_victim_completion_time_ns": combined_victim_completion,
        "victim_slowdown_ratio": victim_slowdown_ratio,
        "victim_failed_in_combined": victim_failed_in_combined,
        "mean_attacker_reuse_delay_ns": safe_mean(
            combined_attacker.loc[
                combined_attacker["status"] == "completed",
                ["source_reuse_delay_ns", "target_reuse_delay_ns"],
            ].stack()
        ) if not combined_attacker.empty else 0.0,
        "mean_victim_reuse_delay_ns": safe_mean(
            victim_only_victim.loc[
                victim_only_victim["status"] == "completed",
                ["source_reuse_delay_ns", "target_reuse_delay_ns"],
            ].stack()
        ) if not victim_only_victim.empty else 0.0,
    }


def annotate_request_log(
    dataframe: pd.DataFrame,
    victim_name: str,
    configuration: ExperimentConfiguration,
    trial_id: int,
    execution_mode: str,
) -> pd.DataFrame:
    annotated = dataframe.copy()
    annotated.insert(0, "execution_mode", execution_mode)
    annotated.insert(0, "trial_id", trial_id)
    annotated.insert(0, "configuration_id", configuration.configuration_id)
    annotated.insert(0, "subexperiment", configuration.subexperiment)
    annotated.insert(0, "victim_qasm", victim_name)
    for field_name, value in reversed(list(configuration.__dict__.items())):
        if field_name == "subexperiment":
            continue
        annotated.insert(5, field_name, value)
    return annotated


# =============================================================================
# One configuration/workload/trial
# =============================================================================


def run_trial(
    victim_path: Path,
    configuration: ExperimentConfiguration,
    trial_id: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[pd.DataFrame]]:
    seed = (
        GLOBAL_SEED
        + stable_text_seed(configuration.configuration_id)
        + stable_text_seed(victim_path.name) * 17
        + trial_id * 10_000
    )

    jobs, allocations = build_jobs_and_allocations(
        victim_path,
        configuration.tenants_sharing_module,
        seed,
    )

    # Deterministic request-specific random choices use the same scenario seed
    # in attacker-only and combined runs. They diverge only when victim demand
    # changes actual resource availability.
    attacker_only = run_scenario(
        configuration,
        jobs,
        allocations,
        included_roles={"attacker", "background"},
        scenario_seed=seed + 1,
    )
    victim_only = run_scenario(
        configuration,
        jobs,
        allocations,
        included_roles={"victim", "background"},
        scenario_seed=seed + 1,
    )
    combined = run_scenario(
        configuration,
        jobs,
        allocations,
        included_roles={"victim", "attacker", "background"},
        scenario_seed=seed + 1,
    )

    compared = compare_attacker_traces(
        attacker_only.request_dataframe,
        combined.request_dataframe,
    )

    summary = trial_row(
        victim_name=victim_path.name,
        configuration=configuration,
        trial_id=trial_id,
        attacker_only=attacker_only,
        victim_only=victim_only,
        combined=combined,
        compared=compared,
    )

    compared.insert(0, "trial_id", trial_id)
    compared.insert(0, "configuration_id", configuration.configuration_id)
    compared.insert(0, "subexperiment", configuration.subexperiment)
    compared.insert(0, "victim_qasm", victim_path.name)
    for field_name, value in reversed(list(configuration.__dict__.items())):
        if field_name == "subexperiment":
            continue
        compared.insert(4, field_name, value)

    logs: list[pd.DataFrame] = []
    if SAVE_ATTACKER_REQUEST_LOG:
        for execution_mode, result in (
            ("attacker_only", attacker_only),
            ("combined", combined),
        ):
            attacker_rows = result.request_dataframe[
                result.request_dataframe["role"] == "attacker"
            ]
            logs.append(
                annotate_request_log(
                    attacker_rows,
                    victim_path.name,
                    configuration,
                    trial_id,
                    execution_mode,
                )
            )

    if SAVE_ALL_REQUEST_LOGS:
        for execution_mode, result in (
            ("attacker_only_all", attacker_only),
            ("victim_only_all", victim_only),
            ("combined_all", combined),
        ):
            logs.append(
                annotate_request_log(
                    result.request_dataframe,
                    victim_path.name,
                    configuration,
                    trial_id,
                    execution_mode,
                )
            )

    return summary, compared, logs


# =============================================================================
# Aggregate summaries and plots
# =============================================================================


SUMMARY_METRICS = {
    "trial_count": ("trial_id", "count"),
    "mean_cq_queue_delay_ns": (
        "combined_avg_communication_qubit_queue_delay_ns",
        "mean",
    ),
    "mean_excess_cq_queue_delay_ns": (
        "avg_excess_communication_qubit_queue_delay_ns",
        "mean",
    ),
    "mean_excess_latency_ns": (
        "avg_excess_turnaround_time_ns",
        "mean",
    ),
    "mean_detection_probability": (
        "attacker_detection_probability",
        "mean",
    ),
    "mean_balanced_detection_accuracy": (
        "balanced_detection_accuracy",
        "mean",
    ),
    "mean_conflict_count": (
        "allocation_conflict_count",
        "mean",
    ),
    "mean_request_failure_rate": (
        "combined_request_failure_rate",
        "mean",
    ),
    "mean_cq_utilization": (
        "communication_qubit_utilization",
        "mean",
    ),
    "mean_reset_utilization": (
        "reset_utilization",
        "mean",
    ),
    "mean_allocation_hold_utilization": (
        "allocation_hold_utilization",
        "mean",
    ),
    "mean_reservation_utilization": (
        "reservation_utilization",
        "mean",
    ),
    "mean_effective_cq_utilization": (
        "effective_communication_qubit_utilization",
        "mean",
    ),
    "mean_epr_wastage_fraction": (
        "epr_pair_wastage_fraction",
        "mean",
    ),
    "mean_grant_fairness": (
        "jain_grant_fairness",
        "mean",
    ),
    "mean_wait_fairness": (
        "jain_inverse_wait_fairness",
        "mean",
    ),
    "mean_victim_slowdown_ratio": (
        "victim_slowdown_ratio",
        "mean",
    ),
    "mean_allocation_change_probability": (
        "allocation_change_probability",
        "mean",
    ),
    "mean_redirected_endpoints": (
        "redirected_endpoint_count",
        "mean",
    ),
}


def aggregate_by(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        dataframe.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(**SUMMARY_METRICS)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def save_metric_bar(
    dataframe: pd.DataFrame,
    category: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    plot_data = (
        dataframe.groupby(category, as_index=False)[metric]
        .mean()
        .sort_values(category)
    )
    axis = plot_data.plot(
        kind="bar",
        x=category,
        y=metric,
        legend=False,
        figsize=(13, 6),
    )
    axis.set_xlabel(category.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def save_capacity_plot(capacity_summary: pd.DataFrame) -> None:
    grouped = (
        capacity_summary.groupby(
            "communication_qubits_per_module",
            as_index=False,
        )
        .agg(
            mean_excess_latency_ns=("mean_excess_latency_ns", "mean"),
            mean_detection_probability=("mean_detection_probability", "mean"),
            mean_cq_utilization=("mean_cq_utilization", "mean"),
        )
        .sort_values("communication_qubits_per_module")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        grouped["communication_qubits_per_module"],
        grouped["mean_excess_latency_ns"],
        marker="o",
    )
    plt.xticks(COMMUNICATION_QUBITS_PER_MODULE_OPTIONS)
    plt.xlabel("Communication qubits per module")
    plt.ylabel("Mean attacker excess latency (ns)")
    plt.title("Effect of Communication-Qubit Overprovisioning")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "communication_qubit_overprovisioning_leakage.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        grouped["communication_qubits_per_module"],
        grouped["mean_detection_probability"],
        marker="o",
        label="Detection probability",
    )
    plt.plot(
        grouped["communication_qubits_per_module"],
        grouped["mean_cq_utilization"],
        marker="o",
        label="CQ utilization",
    )
    plt.xticks(COMMUNICATION_QUBITS_PER_MODULE_OPTIONS)
    plt.xlabel("Communication qubits per module")
    plt.ylabel("Fraction")
    plt.title("Leakage Coverage and Utilization vs CQ Capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "communication_qubit_capacity_detection_utilization.png",
        dpi=300,
    )
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configurations = experiment_configurations()
    victim_names = (
        p1.VICTIM_QASMS[:2]
        if RUN_QUICK_VALIDATION
        else p1.VICTIM_QASMS
    )
    victim_paths = [p1.resolve_qasm(name) for name in victim_names]
    trial_count = 1 if RUN_QUICK_VALIDATION else TRIALS_PER_CONFIGURATION

    print("Phase 1.3 — Dynamic communication-qubit allocation")
    print(f"Configurations: {len(configurations)}")
    print(f"Victim workloads: {len(victim_paths)}")
    print(f"Trials per configuration: {trial_count}")

    summary_rows: list[dict[str, Any]] = []
    compared_frames: list[pd.DataFrame] = []
    request_log_frames: list[pd.DataFrame] = []

    total = len(configurations) * len(victim_paths) * trial_count
    completed_trials = 0

    for configuration in configurations:
        subdirectory = OUTPUT_DIR / configuration.subexperiment
        subdirectory.mkdir(parents=True, exist_ok=True)

        for victim_path in victim_paths:
            for trial_id in range(trial_count):
                completed_trials += 1
                print(
                    f"[{completed_trials}/{total}] "
                    f"{configuration.subexperiment} | "
                    f"{configuration.allocation_policy} | "
                    f"{victim_path.name} | trial {trial_id}"
                )

                summary, compared, logs = run_trial(
                    victim_path,
                    configuration,
                    trial_id,
                )
                summary_rows.append(summary)
                compared_frames.append(compared)
                request_log_frames.extend(logs)

    trial_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            [
                "subexperiment",
                "allocation_policy",
                "communication_qubits_per_module",
                "tenants_sharing_module",
                "victim_tag",
                "trial_id",
            ]
        )
        .reset_index(drop=True)
    )
    trial_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_trial_summary.csv",
        index=False,
    )

    compared_dataframe = pd.concat(
        compared_frames,
        ignore_index=True,
    )
    compared_dataframe.to_csv(
        OUTPUT_DIR / "communication_qubit_attacker_comparison.csv",
        index=False,
    )

    if request_log_frames:
        request_logs = pd.concat(request_log_frames, ignore_index=True)
        request_logs.to_csv(
            OUTPUT_DIR / "communication_qubit_attacker_request_log.csv",
            index=False,
        )

    configuration_summary = aggregate_by(
        trial_summary,
        [
            "subexperiment",
            "configuration_id",
            "allocation_policy",
            "communication_qubits_per_module",
            "tenants_sharing_module",
            "reservation_duration_ns",
            "reset_time_ns",
            "epr_prefetch_enabled",
            "allocation_coordination",
            "allocation_granularity",
        ],
    )
    configuration_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_configuration_summary.csv",
        index=False,
    )

    policy_summary = aggregate_by(
        trial_summary,
        ["allocation_policy"],
    )
    policy_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_policy_summary.csv",
        index=False,
    )

    core = trial_summary[
        trial_summary["subexperiment"]
        == "core_policy_capacity_tenancy"
    ]
    capacity_summary = aggregate_by(
        core,
        [
            "allocation_policy",
            "communication_qubits_per_module",
        ],
    )
    capacity_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_capacity_summary.csv",
        index=False,
    )

    tenancy_summary = aggregate_by(
        core,
        [
            "allocation_policy",
            "tenants_sharing_module",
        ],
    )
    tenancy_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_tenancy_summary.csv",
        index=False,
    )

    timing = trial_summary[
        trial_summary["subexperiment"] == "reservation_reset"
    ]
    reservation_reset_summary = aggregate_by(
        timing,
        [
            "allocation_policy",
            "reservation_duration_ns",
            "reset_time_ns",
        ],
    )
    reservation_reset_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_reservation_reset_summary.csv",
        index=False,
    )

    architecture = trial_summary[
        trial_summary["subexperiment"]
        == "epr_coordination_granularity"
    ]
    granularity_summary = aggregate_by(
        architecture,
        [
            "allocation_policy",
            "allocation_granularity",
        ],
    )
    granularity_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_granularity_summary.csv",
        index=False,
    )

    coordination_summary = aggregate_by(
        architecture,
        [
            "allocation_policy",
            "allocation_coordination",
        ],
    )
    coordination_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_coordination_summary.csv",
        index=False,
    )

    epr_summary = aggregate_by(
        architecture,
        [
            "allocation_policy",
            "epr_prefetch_enabled",
        ],
    )
    epr_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_epr_summary.csv",
        index=False,
    )

    fairness_summary = aggregate_by(
        trial_summary,
        [
            "allocation_policy",
            "tenants_sharing_module",
        ],
    )
    fairness_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_fairness_summary.csv",
        index=False,
    )

    failure_summary = (
        trial_summary.groupby(
            [
                "allocation_policy",
                "communication_qubits_per_module",
                "tenants_sharing_module",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            attacker_baseline_rejection_rate=(
                "attacker_baseline_rejected",
                "mean",
            ),
            attacker_combined_rejection_rate=(
                "attacker_combined_rejected",
                "mean",
            ),
            victim_combined_rejection_rate=(
                "victim_combined_rejected",
                "mean",
            ),
            mean_request_failure_rate=(
                "combined_request_failure_rate",
                "mean",
            ),
            mean_conflict_count=(
                "allocation_conflict_count",
                "mean",
            ),
            mean_redirected_endpoints=(
                "redirected_endpoint_count",
                "mean",
            ),
        )
        .sort_values(
            [
                "allocation_policy",
                "communication_qubits_per_module",
                "tenants_sharing_module",
            ]
        )
    )
    failure_summary.to_csv(
        OUTPUT_DIR / "communication_qubit_failure_summary.csv",
        index=False,
    )

    # Save per-subexperiment trial and configuration outputs.
    for subexperiment in sorted(trial_summary["subexperiment"].unique()):
        subdirectory = OUTPUT_DIR / subexperiment
        sub_trials = trial_summary[
            trial_summary["subexperiment"] == subexperiment
        ]
        sub_configs = configuration_summary[
            configuration_summary["subexperiment"] == subexperiment
        ]
        sub_trials.to_csv(
            subdirectory / "trial_summary.csv",
            index=False,
        )
        sub_configs.to_csv(
            subdirectory / "configuration_summary.csv",
            index=False,
        )

    save_metric_bar(
        policy_summary,
        category="allocation_policy",
        metric="mean_excess_latency_ns",
        ylabel="Mean attacker excess latency (ns)",
        title="Communication-Qubit Allocation Policy Leakage",
        filename="communication_qubit_policy_leakage.png",
    )
    save_metric_bar(
        policy_summary,
        category="allocation_policy",
        metric="mean_cq_queue_delay_ns",
        ylabel="Mean communication-qubit queue delay (ns)",
        title="Communication-Qubit Queue Delay by Policy",
        filename="communication_qubit_policy_queue_delay.png",
    )
    save_metric_bar(
        policy_summary,
        category="allocation_policy",
        metric="mean_grant_fairness",
        ylabel="Jain grant-fairness index",
        title="Communication-Qubit Allocation Fairness",
        filename="communication_qubit_policy_fairness.png",
    )
    save_metric_bar(
        policy_summary,
        category="allocation_policy",
        metric="mean_victim_slowdown_ratio",
        ylabel="Victim completion-time ratio",
        title="Victim Slowdown by Communication-Qubit Policy",
        filename="communication_qubit_policy_victim_slowdown.png",
    )
    if not capacity_summary.empty:
        save_capacity_plot(capacity_summary)

    metadata = {
        "experiment": "Phase 1.3 — Dynamic communication-qubit allocation",
        "screening_design": not RUN_COMPLETE_FACTORIAL,
        "configuration_count": len(configurations),
        "victim_workload_count": len(victim_paths),
        "trials_per_configuration": trial_count,
        "allocation_policies": ALLOCATION_POLICIES,
        "communication_qubits_per_module_options": (
            COMMUNICATION_QUBITS_PER_MODULE_OPTIONS
        ),
        "tenants_sharing_module_options": TENANTS_SHARING_MODULE_OPTIONS,
        "reservation_duration_options_ns": RESERVATION_DURATION_OPTIONS_NS,
        "reset_time_options_ns": RESET_TIME_OPTIONS_NS,
        "epr_prefetch_options": EPR_PREFETCH_OPTIONS,
        "coordination_options": COORDINATION_OPTIONS,
        "allocation_granularity_options": ALLOCATION_GRANULARITY_OPTIONS,
        "hub_capacity": HUB_CAPACITY,
        "base_remote_service_ns": BASE_REMOTE_SERVICE_NS,
        "prefetched_remote_service_ns": PREFETCHED_REMOTE_SERVICE_NS,
        "max_allocation_wait_ns": MAX_ALLOCATION_WAIT_NS,
    }
    with (
        OUTPUT_DIR / "communication_qubit_experiment_metadata.json"
    ).open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    display_columns = [
        "allocation_policy",
        "trial_count",
        "mean_cq_queue_delay_ns",
        "mean_excess_latency_ns",
        "mean_detection_probability",
        "mean_cq_utilization",
        "mean_epr_wastage_fraction",
        "mean_grant_fairness",
        "mean_victim_slowdown_ratio",
        "mean_request_failure_rate",
    ]

    print("\n=== Phase 1.3 policy summary ===")
    print(policy_summary[display_columns].to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
