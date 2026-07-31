#!/usr/bin/env python3
"""
phase1_05_dynamic_rerouting_remapping.py

Experiment 1.5 — Mid-execution rerouting and resource remapping.

Research question
-----------------
Does changing the communication path or endpoint resource during execution
break the black-box timing attack, or does reconfiguration create a new
observable transition?

Implemented mechanisms
----------------------
1. static_path
2. per_job_path_selection
3. per_operation_path_selection
4. load_triggered_rerouting
5. failure_triggered_rerouting
6. communication_qubit_reassignment
7. hub_to_hub_migration
8. dynamic_victim_module_migration

Independent variables
---------------------
- number of available path alternatives: 1, 2, and 4;
- path-selection frequency: per job, per layer, and per operation;
- rerouting pressure threshold;
- path-switching cost;
- victim state-transfer cost;
- deterministic or randomized reconfiguration decisions.

Architecture abstraction
------------------------
The experiment uses eight modules, two independently capacity-limited hubs,
and a bidirectional module ring. A remote operation may use:

- hub 0;
- hub 1;
- the clockwise ring path;
- the counterclockwise ring path.

Each route reserves its hub/link resources and one communication qubit at each
endpoint. The attacker uses the selected Probe-3 release pattern but scans the
available path classes in a fixed round-robin order. This allows a black-box
attacker to test whether the victim moved between paths without observing the
victim's internal routing decisions.

Controls per trial
------------------
1. attacker + identical background tenants;
2. victim + identical background tenants;
3. victim + attacker + identical background tenants;
4. static-path victim + identical background tenants.

The fourth control measures additional victim latency caused by rerouting or
migration rather than by multi-tenant contention.

Integrated experimental design
------------------------------
A. core_mechanism_paths
   All mechanisms × path counts × deterministic/randomized decisions.

B. threshold_frequency
   Dynamic routing mechanisms × thresholds × selection frequencies.

C. switching_cost
   Rerouting/reassignment mechanisms × path-switch costs.

D. state_transfer_cost
   Dynamic victim-module migration × path counts × state-transfer costs ×
   deterministic/randomized decisions.

Outputs
-------
blackbox_window_results/phase1_05_dynamic_rerouting_remapping/
    dynamic_rerouting_trial_summary.csv
    dynamic_rerouting_attacker_comparison.csv.gz
    dynamic_rerouting_request_log.csv.gz
    dynamic_rerouting_event_log.csv
    dynamic_rerouting_mechanism_summary.csv
    dynamic_rerouting_path_count_summary.csv
    dynamic_rerouting_frequency_summary.csv
    dynamic_rerouting_threshold_summary.csv
    dynamic_rerouting_switching_cost_summary.csv
    dynamic_rerouting_state_transfer_summary.csv
    dynamic_rerouting_decision_mode_summary.csv
    dynamic_rerouting_path_localization.csv
    dynamic_rerouting_change_detection.csv
    dynamic_rerouting_fingerprint_stability.csv
    dynamic_rerouting_segment_summary.csv

Execution
---------
Keep this file beside phase1_01_job_module_allocation.py and run:

    python phase1_05_dynamic_rerouting_remapping.py

No terminal options are required. Run controls are defined below.
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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase1_01_job_module_allocation as p1


# =============================================================================
# Output and run controls
# =============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_05_dynamic_rerouting_remapping"
)

TRIALS_PER_CONFIGURATION = 3
RUN_QUICK_VALIDATION = False
MAX_CONFIGURATIONS: int | None = None

SAVE_ATTACKER_COMPARISON = True
SAVE_REQUEST_LOG = True
SAVE_EVENT_LOG = True

GLOBAL_SEED = 20260731


# =============================================================================
# Fixed architecture and workload settings
# =============================================================================

NUM_MODULES = 8
NUM_HUBS = 2

VICTIM_MODULES = (0, 1, 2)
ATTACKER_MODULES = (2, 3)
BACKGROUND_MODULE_PAIRS = (
    (4, 6),
    (1, 7),
    (2, 6),
    (0, 5),
    (3, 7),
    (1, 4),
)

VICTIM_MODULES_REQUESTED = 3
ATTACKER_MODULES_REQUESTED = 2
BACKGROUND_MODULES_REQUESTED = 2
DEFAULT_TENANT_COUNT = 4

VICTIM_QASMS = list(p1.VICTIM_QASMS)

# The selected black-box attacker configuration from the preceding experiments.
VICTIM_START_NS = 1_000
ATTACKER_START_NS = 1_000
OBSERVATION_DURATION_NS = 20_000
PROBE_PERIOD_NS = 420

# Fixed scheduler/resource controls. Queue capacity is intentionally large so
# this experiment isolates rerouting rather than request rejection.
QUEUE_DEPTH = 256
HUB_CAPACITY = 2
LINK_CAPACITY = 1
COMMUNICATION_QUBITS_PER_MODULE = 2

# Remote-operation timing.
HUB_SERVICE_BASE_NS = 80
RING_SERVICE_BASE_NS = 70
PER_RESOURCE_HOP_NS = 10

# Mid-execution reconfiguration controls.
RECONFIGURATION_TRIGGER_NS = 7_000
FAILURE_DURATION_NS = 3_000
TRANSIENT_HALF_WINDOW_NS = 2 * PROBE_PERIOD_NS
CHANGE_DETECTION_THRESHOLD_NS = 25.0

# Fingerprint interpretation.
TIMING_CHANGE_THRESHOLD_NS = 0.0
LOCALIZATION_MIN_PROBES = 2
MAX_SIMULATION_TIME_NS = 2_000_000


# =============================================================================
# Mechanisms and independent-variable values
# =============================================================================

MECHANISMS = [
    "static_path",
    "per_job_path_selection",
    "per_operation_path_selection",
    "load_triggered_rerouting",
    "failure_triggered_rerouting",
    "communication_qubit_reassignment",
    "hub_to_hub_migration",
    "dynamic_victim_module_migration",
]

ALTERNATE_PATH_OPTIONS = [1, 2, 4]
PATH_SELECTION_FREQUENCIES = [
    "per_job",
    "per_layer",
    "per_operation",
]
REROUTING_THRESHOLDS = [1, 2, 4]
PATH_SWITCHING_COSTS_NS = [0, 100, 500]
STATE_TRANSFER_COSTS_NS = [0, 500, 2_000]
DECISION_MODES = ["deterministic", "randomized"]

DEFAULT_ALTERNATE_PATHS = 4
DEFAULT_SELECTION_FREQUENCY = "per_operation"
DEFAULT_REROUTING_THRESHOLD = 2
DEFAULT_PATH_SWITCHING_COST_NS = 100
DEFAULT_STATE_TRANSFER_COST_NS = 500
DEFAULT_DECISION_MODE = "deterministic"


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class LogicalRemoteRequest:
    request_id: int
    operation_uid: str
    tenant_id: str
    role: str
    logical_event_id: int
    circuit_order: int
    release_time_ns: int
    source_partition: int
    target_partition: int
    layer: int


@dataclass(frozen=True)
class RouteOption:
    path_id: int
    path_name: str
    hub_id: int | None
    resource_ids: tuple[str, ...]
    hop_count: int
    base_service_time_ns: int


@dataclass
class RequestPlan:
    source_module: int
    target_module: int
    route: RouteOption
    source_communication_qubit: int
    target_communication_qubit: int
    original_path_id: int
    rerouted: bool
    redirect_reason: str
    path_pressure: float
    switching_overhead_ns: int
    state_transfer_overhead_ns: int
    communication_qubit_reassigned: bool


@dataclass
class RuntimeRequest:
    request: LogicalRemoteRequest
    status: str = "unreleased"
    admitted_time_ns: int | None = None
    service_start_time_ns: int | None = None
    completion_time_ns: int | None = None
    rejection_reason: str = ""
    plan: RequestPlan | None = None


@dataclass(order=True)
class ActiveRequest:
    completion_time_ns: int
    serial: int
    request_id: int = field(compare=False)
    plan: RequestPlan = field(compare=False)


@dataclass(frozen=True)
class ExperimentConfiguration:
    subexperiment: str
    mechanism: str
    alternate_path_count: int
    path_selection_frequency: str
    rerouting_threshold: int
    path_switching_cost_ns: int
    state_transfer_cost_ns: int
    decision_mode: str

    @property
    def configuration_id(self) -> str:
        payload = (
            self.subexperiment,
            self.mechanism,
            self.alternate_path_count,
            self.path_selection_frequency,
            self.rerouting_threshold,
            self.path_switching_cost_ns,
            self.state_transfer_cost_ns,
            self.decision_mode,
        )
        text = "|".join(map(str, payload))
        return f"cfg_{zlib.crc32(text.encode('utf-8')):08x}"


@dataclass
class SimulationResult:
    scenario: str
    request_log: pd.DataFrame
    event_log: pd.DataFrame
    completion_time_ns: int
    victim_completion_time_ns: float
    attacker_completion_time_ns: float
    resource_utilization: dict[str, float]
    total_switching_overhead_ns: int
    total_state_transfer_overhead_ns: int
    redirected_request_count: int
    communication_qubit_reassignment_count: int
    rejected_request_count: int


# =============================================================================
# General helpers
# =============================================================================


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return (
        GLOBAL_SEED
        + zlib.crc32(text.encode("utf-8"))
    ) & 0x7FFFFFFF



def canonical_ring_link(first: int, second: int) -> str:
    left, right = sorted((first, second))
    # The wraparound edge is represented canonically as 0--(N-1).
    return f"ring:{left}-{right}"



def clockwise_nodes(source: int, target: int) -> list[int]:
    nodes = [source]
    current = source
    while current != target:
        current = (current + 1) % NUM_MODULES
        nodes.append(current)
    return nodes



def counterclockwise_nodes(source: int, target: int) -> list[int]:
    nodes = [source]
    current = source
    while current != target:
        current = (current - 1) % NUM_MODULES
        nodes.append(current)
    return nodes



def path_links(nodes: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        canonical_ring_link(first, second)
        for first, second in zip(nodes[:-1], nodes[1:])
    )



def candidate_routes(
    source_module: int,
    target_module: int,
    alternate_path_count: int,
) -> list[RouteOption]:
    """Return up to four stable path classes for one endpoint pair."""

    if source_module == target_module:
        return []

    options: list[RouteOption] = []

    for hub_id in range(NUM_HUBS):
        resources = (
            f"hub:{hub_id}",
            f"access:m{source_module}:h{hub_id}",
            f"access:m{target_module}:h{hub_id}",
        )
        options.append(
            RouteOption(
                path_id=hub_id,
                path_name=f"hub_{hub_id}",
                hub_id=hub_id,
                resource_ids=resources,
                hop_count=2,
                base_service_time_ns=(
                    HUB_SERVICE_BASE_NS
                    + PER_RESOURCE_HOP_NS * 2
                ),
            )
        )

    clockwise = clockwise_nodes(source_module, target_module)
    counterclockwise = counterclockwise_nodes(source_module, target_module)

    clockwise_resources = path_links(clockwise)
    counterclockwise_resources = path_links(counterclockwise)

    options.append(
        RouteOption(
            path_id=2,
            path_name="ring_clockwise",
            hub_id=None,
            resource_ids=clockwise_resources,
            hop_count=len(clockwise_resources),
            base_service_time_ns=(
                RING_SERVICE_BASE_NS
                + PER_RESOURCE_HOP_NS * len(clockwise_resources)
            ),
        )
    )

    options.append(
        RouteOption(
            path_id=3,
            path_name="ring_counterclockwise",
            hub_id=None,
            resource_ids=counterclockwise_resources,
            hop_count=len(counterclockwise_resources),
            base_service_time_ns=(
                RING_SERVICE_BASE_NS
                + PER_RESOURCE_HOP_NS * len(counterclockwise_resources)
            ),
        )
    )

    return options[: max(1, min(alternate_path_count, len(options)))]



def dominant_value(values: Iterable[Any]) -> Any | None:
    values = list(values)
    if not values:
        return None
    counts = Counter(values)
    return sorted(
        counts.items(),
        key=lambda item: (-item[1], str(item[0])),
    )[0][0]



def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# =============================================================================
# Configuration generation
# =============================================================================


def build_configurations() -> list[ExperimentConfiguration]:
    configurations: list[ExperimentConfiguration] = []

    # A. All mechanisms, path counts, and decision modes.
    for mechanism, path_count, decision_mode in itertools.product(
        MECHANISMS,
        ALTERNATE_PATH_OPTIONS,
        DECISION_MODES,
    ):
        if mechanism in {"static_path", "per_job_path_selection"}:
            frequency = "per_job"
        elif mechanism == "dynamic_victim_module_migration":
            frequency = "per_job"
        else:
            frequency = DEFAULT_SELECTION_FREQUENCY

        configurations.append(
            ExperimentConfiguration(
                subexperiment="core_mechanism_paths",
                mechanism=mechanism,
                alternate_path_count=path_count,
                path_selection_frequency=frequency,
                rerouting_threshold=DEFAULT_REROUTING_THRESHOLD,
                path_switching_cost_ns=DEFAULT_PATH_SWITCHING_COST_NS,
                state_transfer_cost_ns=DEFAULT_STATE_TRANSFER_COST_NS,
                decision_mode=decision_mode,
            )
        )

    # B. Threshold and path-selection frequency.
    threshold_mechanisms = [
        "per_operation_path_selection",
        "load_triggered_rerouting",
        "failure_triggered_rerouting",
        "hub_to_hub_migration",
    ]
    for mechanism, threshold, frequency in itertools.product(
        threshold_mechanisms,
        REROUTING_THRESHOLDS,
        PATH_SELECTION_FREQUENCIES,
    ):
        configurations.append(
            ExperimentConfiguration(
                subexperiment="threshold_frequency",
                mechanism=mechanism,
                alternate_path_count=DEFAULT_ALTERNATE_PATHS,
                path_selection_frequency=frequency,
                rerouting_threshold=threshold,
                path_switching_cost_ns=DEFAULT_PATH_SWITCHING_COST_NS,
                state_transfer_cost_ns=DEFAULT_STATE_TRANSFER_COST_NS,
                decision_mode=DEFAULT_DECISION_MODE,
            )
        )

    # C. Path-switching cost.
    switching_mechanisms = [
        "per_operation_path_selection",
        "load_triggered_rerouting",
        "failure_triggered_rerouting",
        "communication_qubit_reassignment",
        "hub_to_hub_migration",
    ]
    for mechanism, switching_cost in itertools.product(
        switching_mechanisms,
        PATH_SWITCHING_COSTS_NS,
    ):
        configurations.append(
            ExperimentConfiguration(
                subexperiment="switching_cost",
                mechanism=mechanism,
                alternate_path_count=DEFAULT_ALTERNATE_PATHS,
                path_selection_frequency=DEFAULT_SELECTION_FREQUENCY,
                rerouting_threshold=DEFAULT_REROUTING_THRESHOLD,
                path_switching_cost_ns=switching_cost,
                state_transfer_cost_ns=DEFAULT_STATE_TRANSFER_COST_NS,
                decision_mode=DEFAULT_DECISION_MODE,
            )
        )

    # D. State-transfer cost for victim module migration.
    for path_count, state_cost, decision_mode in itertools.product(
        [2, 4],
        STATE_TRANSFER_COSTS_NS,
        DECISION_MODES,
    ):
        configurations.append(
            ExperimentConfiguration(
                subexperiment="state_transfer_cost",
                mechanism="dynamic_victim_module_migration",
                alternate_path_count=path_count,
                path_selection_frequency="per_job",
                rerouting_threshold=DEFAULT_REROUTING_THRESHOLD,
                path_switching_cost_ns=DEFAULT_PATH_SWITCHING_COST_NS,
                state_transfer_cost_ns=state_cost,
                decision_mode=decision_mode,
            )
        )

    # Remove exact duplicates while preserving order.
    unique: dict[str, ExperimentConfiguration] = {}
    for configuration in configurations:
        unique.setdefault(configuration.configuration_id, configuration)

    result = list(unique.values())

    if RUN_QUICK_VALIDATION:
        result = result[:8]
    if MAX_CONFIGURATIONS is not None:
        result = result[:MAX_CONFIGURATIONS]

    return result


# =============================================================================
# Job preparation and fixed placement
# =============================================================================


def fixed_allocations(
    victim_job: p1.JobSpec,
    attacker_job: p1.JobSpec,
    background_jobs: Sequence[p1.JobSpec],
) -> dict[str, p1.Allocation]:
    victim_mapping = {
        partition: VICTIM_MODULES[partition % len(VICTIM_MODULES)]
        for partition in victim_job.partitions
    }
    attacker_mapping = {
        partition: ATTACKER_MODULES[partition % len(ATTACKER_MODULES)]
        for partition in attacker_job.partitions
    }

    allocations = {
        victim_job.tenant_id: p1.Allocation(
            tenant_id=victim_job.tenant_id,
            role=victim_job.role,
            accepted=True,
            requested_modules=victim_job.modules_requested,
            partition_to_module=victim_mapping,
        ),
        attacker_job.tenant_id: p1.Allocation(
            tenant_id=attacker_job.tenant_id,
            role=attacker_job.role,
            accepted=True,
            requested_modules=attacker_job.modules_requested,
            partition_to_module=attacker_mapping,
        ),
    }

    for index, job in enumerate(background_jobs):
        pair = BACKGROUND_MODULE_PAIRS[index % len(BACKGROUND_MODULE_PAIRS)]
        mapping = {
            partition: pair[partition % len(pair)]
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
    trace_seed: int,
) -> tuple[list[p1.JobSpec], dict[str, p1.Allocation]]:
    victim_job = copy.deepcopy(
        p1.build_victim_job(qasm_path, VICTIM_MODULES_REQUESTED)
    )
    attacker_job = copy.deepcopy(p1.build_attacker_job())

    background_jobs: list[p1.JobSpec] = []
    for background_index in range(max(0, DEFAULT_TENANT_COUNT - 2)):
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



def touched_partitions(
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



def build_remote_requests(
    jobs: Sequence[p1.JobSpec],
) -> list[LogicalRemoteRequest]:
    requests: list[LogicalRemoteRequest] = []
    request_id = 0

    for job in jobs:
        circuit_order = 0
        for event in job.logical_events:
            partitions = touched_partitions(job, event)
            if len(partitions) < 2:
                continue

            for pair_index, (source_partition, target_partition) in enumerate(
                itertools.combinations(partitions, 2)
            ):
                requests.append(
                    LogicalRemoteRequest(
                        request_id=request_id,
                        operation_uid=(
                            f"{job.tenant_id}:"
                            f"{event.event_id}:"
                            f"{pair_index}"
                        ),
                        tenant_id=job.tenant_id,
                        role=job.role,
                        logical_event_id=event.event_id,
                        circuit_order=circuit_order,
                        release_time_ns=(
                            job.start_time_ns + event.release_offset_ns
                        ),
                        source_partition=source_partition,
                        target_partition=target_partition,
                        layer=(circuit_order // 8),
                    )
                )
                request_id += 1
                circuit_order += 1

    return requests


# =============================================================================
# Dynamic rerouting simulator
# =============================================================================


class DynamicReroutingSimulator:
    def __init__(
        self,
        requests: Sequence[LogicalRemoteRequest],
        allocations: dict[str, p1.Allocation],
        configuration: ExperimentConfiguration,
        scenario: str,
        trial_seed_value: int,
        *,
        force_static_victim: bool = False,
    ) -> None:
        self.requests = list(requests)
        self.request_by_id = {request.request_id: request for request in requests}
        self.allocations = copy.deepcopy(allocations)
        self.configuration = configuration
        self.scenario = scenario
        self.trial_seed_value = trial_seed_value
        self.force_static_victim = force_static_victim

        self.runtime = {
            request.request_id: RuntimeRequest(request=request)
            for request in self.requests
        }

        self.current_time_ns = 0
        self.pending: list[int] = []
        self.active_heap: list[ActiveRequest] = []
        self.active_serial = 0

        self.active_resource_counts: dict[str, int] = defaultdict(int)
        self.active_communication_qubits: dict[tuple[int, int], int] = defaultdict(int)
        self.resource_busy_time_ns: dict[str, float] = defaultdict(float)
        self.communication_qubit_busy_time_ns: dict[tuple[int, int], float] = defaultdict(float)

        self.job_path_cache: dict[str, int] = {}
        self.layer_path_cache: dict[tuple[str, int], int] = {}
        self.tenant_last_path_id: dict[str, int] = {}

        self.victim_migration_applied = False
        self.victim_blocked_until_ns = 0
        self.hub_migration_announced = False

        self.request_rows: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []

        self.total_switching_overhead_ns = 0
        self.total_state_transfer_overhead_ns = 0
        self.redirected_request_count = 0
        self.communication_qubit_reassignment_count = 0
        self.rejected_request_count = 0

        self.failed_resource = "hub:0"
        self.failure_start_ns = RECONFIGURATION_TRIGGER_NS
        self.failure_end_ns = RECONFIGURATION_TRIGGER_NS + FAILURE_DURATION_NS

    # ------------------------------------------------------------------
    # Resource model
    # ------------------------------------------------------------------

    def _resource_capacity(self, resource_id: str) -> int:
        if resource_id.startswith("hub:"):
            return HUB_CAPACITY
        return LINK_CAPACITY

    def _resource_is_failed(self, resource_id: str) -> bool:
        if self.configuration.mechanism != "failure_triggered_rerouting":
            return False
        return (
            resource_id == self.failed_resource
            and self.failure_start_ns <= self.current_time_ns < self.failure_end_ns
        )

    def _route_available(self, route: RouteOption) -> bool:
        for resource_id in route.resource_ids:
            if self._resource_is_failed(resource_id):
                return False
            if (
                self.active_resource_counts[resource_id]
                >= self._resource_capacity(resource_id)
            ):
                return False
        return True

    def _route_pressure(self, route: RouteOption) -> float:
        pressure = 0.0
        for resource_id in route.resource_ids:
            capacity = self._resource_capacity(resource_id)
            pressure += self.active_resource_counts[resource_id] / capacity
            if self._resource_is_failed(resource_id):
                pressure += 1_000.0
        return pressure

    def _default_communication_qubit(self, role: str) -> int:
        if role in {"victim", "attacker"}:
            return 0
        if COMMUNICATION_QUBITS_PER_MODULE <= 1:
            return 0
        return 1

    def _select_communication_qubit(
        self,
        module: int,
        role: str,
        allow_reassignment: bool,
    ) -> tuple[int | None, bool]:
        preferred = self._default_communication_qubit(role)
        preferred_key = (module, preferred)
        if self.active_communication_qubits[preferred_key] == 0:
            return preferred, False

        if allow_reassignment:
            for communication_qubit in range(COMMUNICATION_QUBITS_PER_MODULE):
                key = (module, communication_qubit)
                if self.active_communication_qubits[key] == 0:
                    return communication_qubit, communication_qubit != preferred

        return None, False

    def _can_acquire_plan(self, plan: RequestPlan) -> bool:
        if not self._route_available(plan.route):
            return False
        source_key = (plan.source_module, plan.source_communication_qubit)
        target_key = (plan.target_module, plan.target_communication_qubit)
        if self.active_communication_qubits[source_key] > 0:
            return False
        if self.active_communication_qubits[target_key] > 0:
            return False
        return True

    def _acquire_plan(self, plan: RequestPlan, service_time_ns: int) -> None:
        for resource_id in plan.route.resource_ids:
            self.active_resource_counts[resource_id] += 1
            self.resource_busy_time_ns[resource_id] += service_time_ns
        source_key = (plan.source_module, plan.source_communication_qubit)
        target_key = (plan.target_module, plan.target_communication_qubit)
        self.active_communication_qubits[source_key] += 1
        self.active_communication_qubits[target_key] += 1
        self.communication_qubit_busy_time_ns[source_key] += service_time_ns
        self.communication_qubit_busy_time_ns[target_key] += service_time_ns

    def _release_plan(self, plan: RequestPlan) -> None:
        for resource_id in plan.route.resource_ids:
            self.active_resource_counts[resource_id] -= 1
        source_key = (plan.source_module, plan.source_communication_qubit)
        target_key = (plan.target_module, plan.target_communication_qubit)
        self.active_communication_qubits[source_key] -= 1
        self.active_communication_qubits[target_key] -= 1

    # ------------------------------------------------------------------
    # Dynamic mapping and route selection
    # ------------------------------------------------------------------

    def _apply_mid_execution_events(self) -> None:
        mechanism = self.configuration.mechanism
        if self.force_static_victim:
            return

        if (
            mechanism == "dynamic_victim_module_migration"
            and not self.victim_migration_applied
            and self.current_time_ns >= RECONFIGURATION_TRIGGER_NS
        ):
            allocation = self.allocations.get("victim")
            if allocation is not None:
                old_mapping = dict(allocation.partition_to_module)
                # Move the partition assigned to the P2-shared module 2 to a
                # currently non-attacker module. Future requests use the new map.
                candidate_partitions = [
                    partition
                    for partition, module in allocation.partition_to_module.items()
                    if module == 2
                ]
                if not candidate_partitions:
                    candidate_partitions = [max(allocation.partition_to_module)]
                partition = candidate_partitions[0]
                destination_candidates = [4, 5, 6, 7]
                if self.configuration.decision_mode == "randomized":
                    rng = random.Random(
                        stable_seed(
                            self.trial_seed_value,
                            self.configuration.configuration_id,
                            "victim_module_migration",
                        )
                    )
                    destination = rng.choice(destination_candidates)
                else:
                    destination = destination_candidates[0]

                old_module = allocation.partition_to_module[partition]
                allocation.partition_to_module[partition] = destination
                self.victim_migration_applied = True
                self.victim_blocked_until_ns = (
                    self.current_time_ns
                    + self.configuration.state_transfer_cost_ns
                )
                self.total_state_transfer_overhead_ns += (
                    self.configuration.state_transfer_cost_ns
                )
                self.event_rows.append(
                    {
                        "scenario": self.scenario,
                        "time_ns": self.current_time_ns,
                        "tenant_id": "victim",
                        "role": "victim",
                        "event_type": "module_migration",
                        "old_value": json.dumps(old_mapping, sort_keys=True),
                        "new_value": json.dumps(
                            allocation.partition_to_module,
                            sort_keys=True,
                        ),
                        "old_module": old_module,
                        "new_module": destination,
                        "old_path_id": np.nan,
                        "new_path_id": np.nan,
                        "overhead_ns": self.configuration.state_transfer_cost_ns,
                    }
                )

        if (
            mechanism == "hub_to_hub_migration"
            and not self.hub_migration_announced
            and self.current_time_ns >= RECONFIGURATION_TRIGGER_NS
        ):
            self.hub_migration_announced = True
            new_hub = 1 if self.configuration.alternate_path_count >= 2 else 0
            self.event_rows.append(
                {
                    "scenario": self.scenario,
                    "time_ns": self.current_time_ns,
                    "tenant_id": "victim",
                    "role": "victim",
                    "event_type": "hub_migration",
                    "old_value": "hub_0",
                    "new_value": f"hub_{new_hub}",
                    "old_module": np.nan,
                    "new_module": np.nan,
                    "old_path_id": 0,
                    "new_path_id": new_hub,
                    "overhead_ns": self.configuration.path_switching_cost_ns,
                }
            )

        if (
            mechanism == "failure_triggered_rerouting"
            and self.current_time_ns == self.failure_start_ns
        ):
            self.event_rows.append(
                {
                    "scenario": self.scenario,
                    "time_ns": self.current_time_ns,
                    "tenant_id": "system",
                    "role": "system",
                    "event_type": "path_failure_start",
                    "old_value": self.failed_resource,
                    "new_value": "failed",
                    "old_module": np.nan,
                    "new_module": np.nan,
                    "old_path_id": 0,
                    "new_path_id": np.nan,
                    "overhead_ns": 0,
                }
            )

        if (
            mechanism == "failure_triggered_rerouting"
            and self.current_time_ns == self.failure_end_ns
        ):
            self.event_rows.append(
                {
                    "scenario": self.scenario,
                    "time_ns": self.current_time_ns,
                    "tenant_id": "system",
                    "role": "system",
                    "event_type": "path_failure_end",
                    "old_value": self.failed_resource,
                    "new_value": "recovered",
                    "old_module": np.nan,
                    "new_module": np.nan,
                    "old_path_id": np.nan,
                    "new_path_id": 0,
                    "overhead_ns": 0,
                }
            )

    def _actual_endpoints(
        self,
        request: LogicalRemoteRequest,
    ) -> tuple[int, int] | None:
        allocation = self.allocations[request.tenant_id]
        source_module = allocation.partition_to_module[request.source_partition]
        target_module = allocation.partition_to_module[request.target_partition]
        if source_module == target_module:
            return None
        return source_module, target_module

    def _random_choice(
        self,
        values: Sequence[int],
        request: LogicalRemoteRequest,
        label: str,
    ) -> int:
        rng = random.Random(
            stable_seed(
                self.trial_seed_value,
                self.configuration.configuration_id,
                request.operation_uid,
                label,
            )
        )
        return rng.choice(list(values))

    def _best_route_id(
        self,
        routes: Sequence[RouteOption],
        request: LogicalRemoteRequest,
        label: str,
    ) -> int:
        if not routes:
            raise RuntimeError("No routes available.")

        scored = sorted(
            (
                self._route_pressure(route),
                route.base_service_time_ns,
                route.path_id,
            )
            for route in routes
        )
        best_pressure = scored[0][0]
        best_ids = [
            path_id
            for pressure, _, path_id in scored
            if math.isclose(pressure, best_pressure)
        ]
        if self.configuration.decision_mode == "randomized" and len(best_ids) > 1:
            return self._random_choice(best_ids, request, label)
        return min(best_ids)

    def _frequency_cached_path(
        self,
        request: LogicalRemoteRequest,
        routes: Sequence[RouteOption],
    ) -> int:
        frequency = self.configuration.path_selection_frequency
        if frequency == "per_job":
            key: Any = request.tenant_id
            cache = self.job_path_cache
        elif frequency == "per_layer":
            key = (request.tenant_id, request.layer)
            cache = self.layer_path_cache
        else:
            return self._best_route_id(routes, request, "per_operation_path")

        if key not in cache:
            if self.configuration.decision_mode == "randomized":
                cache[key] = self._random_choice(
                    [route.path_id for route in routes],
                    request,
                    f"frequency:{frequency}",
                )
            else:
                cache[key] = min(
                    routes,
                    key=lambda route: (
                        route.base_service_time_ns,
                        route.path_id,
                    ),
                ).path_id
        return int(cache[key])

    def _select_victim_path_id(
        self,
        request: LogicalRemoteRequest,
        routes: Sequence[RouteOption],
    ) -> tuple[int, int, str]:
        """Return selected path, original path, and redirect reason."""

        available_ids = [route.path_id for route in routes]
        original_path_id = available_ids[0]
        mechanism = "static_path" if self.force_static_victim else self.configuration.mechanism

        if mechanism == "static_path":
            return original_path_id, original_path_id, ""

        if mechanism == "per_job_path_selection":
            selected = self._frequency_cached_path(request, routes)
            return selected, original_path_id, "per_job_selection" if selected != original_path_id else ""

        if mechanism == "per_operation_path_selection":
            selected = self._frequency_cached_path(request, routes)
            return selected, original_path_id, "per_operation_selection" if selected != original_path_id else ""

        if mechanism == "load_triggered_rerouting":
            original_route = next(route for route in routes if route.path_id == original_path_id)
            if self._route_pressure(original_route) >= self.configuration.rerouting_threshold:
                selected = self._best_route_id(routes, request, "load_triggered")
                return selected, original_path_id, "load_threshold" if selected != original_path_id else ""
            return original_path_id, original_path_id, ""

        if mechanism == "failure_triggered_rerouting":
            original_route = next(route for route in routes if route.path_id == original_path_id)
            if any(self._resource_is_failed(resource_id) for resource_id in original_route.resource_ids):
                viable = [
                    route
                    for route in routes
                    if not any(self._resource_is_failed(resource_id) for resource_id in route.resource_ids)
                ]
                if viable:
                    selected = self._best_route_id(viable, request, "failure_triggered")
                    return selected, original_path_id, "path_failure" if selected != original_path_id else ""
            return original_path_id, original_path_id, ""

        if mechanism == "hub_to_hub_migration":
            selected = 0
            if self.current_time_ns >= RECONFIGURATION_TRIGGER_NS and 1 in available_ids:
                selected = 1
            return selected, original_path_id, "hub_migration" if selected != original_path_id else ""

        # Communication-qubit reassignment and module migration preserve the
        # default route unless the frequency setting explicitly requests a path.
        if mechanism in {
            "communication_qubit_reassignment",
            "dynamic_victim_module_migration",
        }:
            return original_path_id, original_path_id, ""

        return original_path_id, original_path_id, ""

    def _build_plan(self, request: LogicalRemoteRequest) -> RequestPlan | None:
        endpoints = self._actual_endpoints(request)
        if endpoints is None:
            return None
        source_module, target_module = endpoints
        routes = candidate_routes(
            source_module,
            target_module,
            self.configuration.alternate_path_count,
        )
        if not routes:
            return None

        if request.role == "attacker":
            # Fixed path-scanning probe. The same probe ID uses the same path in
            # attacker-only and combined runs.
            route_index = request.circuit_order % len(routes)
            selected_path_id = routes[route_index].path_id
            original_path_id = selected_path_id
            redirect_reason = "attacker_path_scan"
        elif request.role == "background":
            selected_path_id = routes[0].path_id
            original_path_id = selected_path_id
            redirect_reason = ""
        else:
            selected_path_id, original_path_id, redirect_reason = (
                self._select_victim_path_id(request, routes)
            )

        route_by_id = {route.path_id: route for route in routes}
        if selected_path_id not in route_by_id:
            selected_path_id = routes[0].path_id
        route = route_by_id[selected_path_id]

        allow_reassignment = (
            request.role == "victim"
            and not self.force_static_victim
            and self.configuration.mechanism == "communication_qubit_reassignment"
        )
        source_q, source_reassigned = self._select_communication_qubit(
            source_module,
            request.role,
            allow_reassignment,
        )
        target_q, target_reassigned = self._select_communication_qubit(
            target_module,
            request.role,
            allow_reassignment,
        )
        if source_q is None or target_q is None:
            return None

        previous_path = self.tenant_last_path_id.get(request.tenant_id)
        path_changed = previous_path is not None and previous_path != route.path_id
        switching_overhead_ns = 0
        if request.role == "victim" and path_changed and not self.force_static_victim:
            switching_overhead_ns = self.configuration.path_switching_cost_ns

        state_transfer_overhead_ns = 0
        if (
            request.role == "victim"
            and self.configuration.mechanism == "dynamic_victim_module_migration"
            and not self.force_static_victim
            and self.current_time_ns < self.victim_blocked_until_ns
        ):
            return None

        reassigned = source_reassigned or target_reassigned
        if reassigned and not redirect_reason:
            redirect_reason = "communication_qubit_reassignment"

        return RequestPlan(
            source_module=source_module,
            target_module=target_module,
            route=route,
            source_communication_qubit=source_q,
            target_communication_qubit=target_q,
            original_path_id=original_path_id,
            rerouted=(route.path_id != original_path_id) or reassigned,
            redirect_reason=redirect_reason,
            path_pressure=self._route_pressure(route),
            switching_overhead_ns=switching_overhead_ns,
            state_transfer_overhead_ns=state_transfer_overhead_ns,
            communication_qubit_reassigned=reassigned,
        )

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _admit_released_requests(self) -> None:
        for request in self.requests:
            runtime = self.runtime[request.request_id]
            if runtime.status != "unreleased":
                continue
            if request.release_time_ns > self.current_time_ns:
                continue

            if len(self.pending) >= QUEUE_DEPTH:
                runtime.status = "rejected"
                runtime.rejection_reason = "queue_full"
                self.rejected_request_count += 1
                continue

            runtime.status = "pending"
            runtime.admitted_time_ns = self.current_time_ns
            self.pending.append(request.request_id)

    def _start_request(self, request_id: int, plan: RequestPlan) -> None:
        runtime = self.runtime[request_id]
        request = runtime.request

        service_time_ns = (
            plan.route.base_service_time_ns
            + plan.switching_overhead_ns
            + plan.state_transfer_overhead_ns
        )

        runtime.status = "active"
        runtime.service_start_time_ns = self.current_time_ns
        runtime.plan = plan
        self.pending.remove(request_id)

        self._acquire_plan(plan, service_time_ns)
        self.tenant_last_path_id[request.tenant_id] = plan.route.path_id

        self.active_serial += 1
        completion_time_ns = self.current_time_ns + service_time_ns
        heapq.heappush(
            self.active_heap,
            ActiveRequest(
                completion_time_ns=completion_time_ns,
                serial=self.active_serial,
                request_id=request_id,
                plan=plan,
            ),
        )

        self.total_switching_overhead_ns += plan.switching_overhead_ns
        if plan.rerouted:
            self.redirected_request_count += 1
        if plan.communication_qubit_reassigned:
            self.communication_qubit_reassignment_count += 1

        if request.role == "victim":
            previous_path = self.tenant_last_path_id.get(request.tenant_id)
            if plan.rerouted or plan.switching_overhead_ns > 0:
                self.event_rows.append(
                    {
                        "scenario": self.scenario,
                        "time_ns": self.current_time_ns,
                        "tenant_id": request.tenant_id,
                        "role": request.role,
                        "event_type": (
                            "communication_qubit_reassignment"
                            if plan.communication_qubit_reassigned
                            else "path_switch"
                        ),
                        "old_value": str(plan.original_path_id),
                        "new_value": str(plan.route.path_id),
                        "old_module": np.nan,
                        "new_module": np.nan,
                        "old_path_id": plan.original_path_id,
                        "new_path_id": plan.route.path_id,
                        "overhead_ns": plan.switching_overhead_ns,
                    }
                )

    def _schedule_pending(self) -> None:
        made_progress = True
        while made_progress:
            made_progress = False
            ordered = sorted(
                self.pending,
                key=lambda request_id: (
                    self.runtime[request_id].request.release_time_ns,
                    self.runtime[request_id].request.request_id,
                ),
            )
            for request_id in ordered:
                runtime = self.runtime[request_id]
                request = runtime.request
                if request.role == "victim" and self.current_time_ns < self.victim_blocked_until_ns:
                    continue
                plan = self._build_plan(request)
                if plan is None:
                    continue
                if self._can_acquire_plan(plan):
                    self._start_request(request_id, plan)
                    made_progress = True
                    break

    def _complete_requests(self) -> None:
        while self.active_heap and self.active_heap[0].completion_time_ns <= self.current_time_ns:
            active = heapq.heappop(self.active_heap)
            runtime = self.runtime[active.request_id]
            request = runtime.request
            runtime.status = "completed"
            runtime.completion_time_ns = active.completion_time_ns
            self._release_plan(active.plan)

            self.request_rows.append(
                {
                    "scenario": self.scenario,
                    "configuration_id": self.configuration.configuration_id,
                    "subexperiment": self.configuration.subexperiment,
                    "mechanism": self.configuration.mechanism,
                    "operation_uid": request.operation_uid,
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "role": request.role,
                    "logical_event_id": request.logical_event_id,
                    "circuit_order": request.circuit_order,
                    "layer": request.layer,
                    "release_time_ns": request.release_time_ns,
                    "admitted_time_ns": runtime.admitted_time_ns,
                    "service_start_time_ns": runtime.service_start_time_ns,
                    "completion_time_ns": runtime.completion_time_ns,
                    "queue_delay_ns": (
                        runtime.service_start_time_ns - request.release_time_ns
                    ),
                    "service_time_ns": (
                        runtime.completion_time_ns - runtime.service_start_time_ns
                    ),
                    "turnaround_time_ns": (
                        runtime.completion_time_ns - request.release_time_ns
                    ),
                    "source_module": active.plan.source_module,
                    "target_module": active.plan.target_module,
                    "path_id": active.plan.route.path_id,
                    "path_name": active.plan.route.path_name,
                    "hub_id": active.plan.route.hub_id,
                    "route_resources": "|".join(active.plan.route.resource_ids),
                    "source_communication_qubit": active.plan.source_communication_qubit,
                    "target_communication_qubit": active.plan.target_communication_qubit,
                    "original_path_id": active.plan.original_path_id,
                    "rerouted": active.plan.rerouted,
                    "redirect_reason": active.plan.redirect_reason,
                    "path_pressure_at_start": active.plan.path_pressure,
                    "switching_overhead_ns": active.plan.switching_overhead_ns,
                    "state_transfer_overhead_ns": active.plan.state_transfer_overhead_ns,
                    "communication_qubit_reassigned": (
                        active.plan.communication_qubit_reassigned
                    ),
                    "segment": segment_for_time(request.release_time_ns),
                }
            )

    def _next_event_time(self) -> int | None:
        candidates: list[int] = []

        unreleased_times = [
            request.release_time_ns
            for request in self.requests
            if self.runtime[request.request_id].status == "unreleased"
        ]
        if unreleased_times:
            candidates.append(min(unreleased_times))

        if self.active_heap:
            candidates.append(self.active_heap[0].completion_time_ns)

        if (
            self.configuration.mechanism
            in {"dynamic_victim_module_migration", "hub_to_hub_migration"}
            and self.current_time_ns < RECONFIGURATION_TRIGGER_NS
        ):
            candidates.append(RECONFIGURATION_TRIGGER_NS)

        if self.configuration.mechanism == "failure_triggered_rerouting":
            if self.current_time_ns < self.failure_start_ns:
                candidates.append(self.failure_start_ns)
            elif self.current_time_ns < self.failure_end_ns:
                candidates.append(self.failure_end_ns)

        if self.victim_blocked_until_ns > self.current_time_ns:
            candidates.append(self.victim_blocked_until_ns)

        future = [candidate for candidate in candidates if candidate > self.current_time_ns]
        if future:
            return min(future)

        if self.pending:
            # A pending request may become feasible after a same-time completion
            # already processed. If no future event exists, advance one ns to
            # avoid a stalled equality edge.
            return self.current_time_ns + 1

        return None

    def run(self) -> SimulationResult:
        while True:
            if self.current_time_ns > MAX_SIMULATION_TIME_NS:
                raise RuntimeError(
                    f"Simulation exceeded {MAX_SIMULATION_TIME_NS} ns for "
                    f"{self.configuration.configuration_id} ({self.scenario})."
                )

            self._complete_requests()
            self._apply_mid_execution_events()
            self._admit_released_requests()
            self._schedule_pending()

            unfinished = any(
                runtime.status in {"unreleased", "pending", "active"}
                for runtime in self.runtime.values()
            )
            if not unfinished:
                break

            next_time = self._next_event_time()
            if next_time is None:
                raise RuntimeError("Simulation deadlocked with unfinished requests.")
            self.current_time_ns = next_time

        request_log = pd.DataFrame(self.request_rows)
        event_log = pd.DataFrame(self.event_rows)

        if request_log.empty:
            victim_completion = 0.0
            attacker_completion = 0.0
        else:
            victim_rows = request_log[request_log["role"] == "victim"]
            attacker_rows = request_log[request_log["role"] == "attacker"]
            victim_completion = (
                float(victim_rows["completion_time_ns"].max())
                if not victim_rows.empty
                else 0.0
            )
            attacker_completion = (
                float(attacker_rows["completion_time_ns"].max())
                if not attacker_rows.empty
                else 0.0
            )

        makespan = max(self.current_time_ns, 1)
        resource_utilization: dict[str, float] = {}
        for resource_id, busy_time in self.resource_busy_time_ns.items():
            resource_utilization[resource_id] = safe_ratio(
                busy_time,
                self._resource_capacity(resource_id) * makespan,
            )
        for key, busy_time in self.communication_qubit_busy_time_ns.items():
            resource_utilization[
                f"communication_qubit:m{key[0]}:q{key[1]}"
            ] = safe_ratio(busy_time, makespan)

        return SimulationResult(
            scenario=self.scenario,
            request_log=request_log,
            event_log=event_log,
            completion_time_ns=self.current_time_ns,
            victim_completion_time_ns=victim_completion,
            attacker_completion_time_ns=attacker_completion,
            resource_utilization=resource_utilization,
            total_switching_overhead_ns=self.total_switching_overhead_ns,
            total_state_transfer_overhead_ns=self.total_state_transfer_overhead_ns,
            redirected_request_count=self.redirected_request_count,
            communication_qubit_reassignment_count=(
                self.communication_qubit_reassignment_count
            ),
            rejected_request_count=self.rejected_request_count,
        )


# =============================================================================
# Counterfactual comparison and attacker interpretation
# =============================================================================


def segment_for_time_at(time_ns: int, trigger_time_ns: int) -> str:
    if time_ns < trigger_time_ns - TRANSIENT_HALF_WINDOW_NS:
        return "before"
    if time_ns <= trigger_time_ns + TRANSIENT_HALF_WINDOW_NS:
        return "transient"
    return "after"


def segment_for_time(time_ns: int) -> str:
    return segment_for_time_at(time_ns, RECONFIGURATION_TRIGGER_NS)



def attacker_comparison(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    baseline = attacker_only[attacker_only["role"] == "attacker"].copy()
    victim_on = combined[combined["role"] == "attacker"].copy()

    baseline = baseline.rename(
        columns={
            "turnaround_time_ns": "baseline_turnaround_time_ns",
            "queue_delay_ns": "baseline_queue_delay_ns",
            "completion_time_ns": "baseline_completion_time_ns",
            "path_id": "probe_path_id",
            "path_name": "probe_path_name",
        }
    )
    victim_on = victim_on.rename(
        columns={
            "turnaround_time_ns": "combined_turnaround_time_ns",
            "queue_delay_ns": "combined_queue_delay_ns",
            "completion_time_ns": "combined_completion_time_ns",
        }
    )

    baseline_columns = [
        "operation_uid",
        "request_id",
        "circuit_order",
        "release_time_ns",
        "segment",
        "baseline_turnaround_time_ns",
        "baseline_queue_delay_ns",
        "baseline_completion_time_ns",
        "probe_path_id",
        "probe_path_name",
    ]
    combined_columns = [
        "operation_uid",
        "combined_turnaround_time_ns",
        "combined_queue_delay_ns",
        "combined_completion_time_ns",
    ]

    compared = baseline[baseline_columns].merge(
        victim_on[combined_columns],
        on="operation_uid",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    compared["baseline_failed"] = compared["_merge"] == "right_only"
    compared["combined_failed"] = compared["_merge"] == "left_only"
    compared["failure_transition"] = (
        compared["baseline_failed"] != compared["combined_failed"]
    )

    compared["excess_turnaround_time_ns"] = (
        compared["combined_turnaround_time_ns"]
        - compared["baseline_turnaround_time_ns"]
    ).fillna(0.0)
    compared["absolute_timing_change_ns"] = compared[
        "excess_turnaround_time_ns"
    ].abs()
    compared["positive_delay_observed"] = (
        compared["excess_turnaround_time_ns"] > TIMING_CHANGE_THRESHOLD_NS
    )
    compared["negative_speedup_observed"] = (
        compared["excess_turnaround_time_ns"] < -TIMING_CHANGE_THRESHOLD_NS
    )
    compared["any_observable_change"] = (
        compared["absolute_timing_change_ns"] > TIMING_CHANGE_THRESHOLD_NS
    ) | compared["failure_transition"]

    return compared.sort_values("circuit_order").reset_index(drop=True)



def segment_metrics(compared: pd.DataFrame, segment: str) -> dict[str, float]:
    subset = compared[compared["segment"] == segment]
    if subset.empty:
        return {
            f"{segment}_probe_count": 0,
            f"{segment}_mean_signed_excess_ns": 0.0,
            f"{segment}_mean_absolute_change_ns": 0.0,
            f"{segment}_max_absolute_change_ns": 0.0,
            f"{segment}_positive_delay_fraction": 0.0,
            f"{segment}_negative_speedup_fraction": 0.0,
            f"{segment}_observable_change_fraction": 0.0,
        }
    return {
        f"{segment}_probe_count": int(len(subset)),
        f"{segment}_mean_signed_excess_ns": float(
            subset["excess_turnaround_time_ns"].mean()
        ),
        f"{segment}_mean_absolute_change_ns": float(
            subset["absolute_timing_change_ns"].mean()
        ),
        f"{segment}_max_absolute_change_ns": float(
            subset["absolute_timing_change_ns"].max()
        ),
        f"{segment}_positive_delay_fraction": float(
            subset["positive_delay_observed"].mean()
        ),
        f"{segment}_negative_speedup_fraction": float(
            subset["negative_speedup_observed"].mean()
        ),
        f"{segment}_observable_change_fraction": float(
            subset["any_observable_change"].mean()
        ),
    }



def actual_victim_path(
    combined_log: pd.DataFrame,
    segment: str,
) -> int | None:
    subset = combined_log[
        (combined_log["role"] == "victim")
        & (combined_log["segment"] == segment)
    ]
    if subset.empty:
        return None
    value = dominant_value(subset["path_id"].astype(int).tolist())
    return int(value) if value is not None else None



def predicted_victim_path(
    compared: pd.DataFrame,
    segment: str,
) -> tuple[int | None, float]:
    subset = compared[compared["segment"] == segment]
    if subset.empty:
        return None, 0.0

    grouped = (
        subset.groupby("probe_path_id", as_index=False)
        .agg(
            probe_count=("operation_uid", "count"),
            mean_absolute_change_ns=("absolute_timing_change_ns", "mean"),
        )
    )
    grouped = grouped[grouped["probe_count"] >= LOCALIZATION_MIN_PROBES]
    if grouped.empty:
        return None, 0.0
    best = grouped.sort_values(
        ["mean_absolute_change_ns", "probe_path_id"],
        ascending=[False, True],
    ).iloc[0]
    return int(best["probe_path_id"]), float(best["mean_absolute_change_ns"])



def detect_reconfiguration(
    compared: pd.DataFrame,
    trigger_time_ns: int,
) -> tuple[bool, float]:
    ordered = compared.sort_values("release_time_ns")
    before = ordered[
        ordered["release_time_ns"] < trigger_time_ns
    ].tail(4)
    after = ordered[
        ordered["release_time_ns"] >= trigger_time_ns
    ].head(4)
    if before.empty or after.empty:
        return False, 0.0
    shift = abs(
        float(after["excess_turnaround_time_ns"].mean())
        - float(before["excess_turnaround_time_ns"].mean())
    )
    return shift >= CHANGE_DETECTION_THRESHOLD_NS, shift



def victim_route_changed(combined_log: pd.DataFrame) -> bool:
    before = actual_victim_path(combined_log, "before")
    after = actual_victim_path(combined_log, "after")
    return before is not None and after is not None and before != after



def ground_truth_reconfiguration(
    configuration: ExperimentConfiguration,
    combined_events: pd.DataFrame,
) -> bool:
    if configuration.mechanism in {
        "static_path",
        "per_job_path_selection",
    }:
        return False
    if combined_events.empty:
        return False
    meaningful = combined_events[
        combined_events["event_type"].isin(
            [
                "path_switch",
                "communication_qubit_reassignment",
                "module_migration",
                "hub_migration",
                "path_failure_start",
            ]
        )
    ]
    return not meaningful.empty


# =============================================================================
# One trial
# =============================================================================


def run_one_trial(
    configuration: ExperimentConfiguration,
    qasm_filename: str,
    trial_id: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    qasm_path = p1.resolve_qasm(qasm_filename)
    victim_tag = p1.safe_tag(qasm_filename)
    trace_seed = stable_seed(victim_tag, trial_id, "released_trace")
    jobs, allocations = build_jobs_for_trial(qasm_path, trace_seed)
    all_requests = build_remote_requests(jobs)

    scenario_roles = {
        "attacker_only": {"attacker", "background"},
        "victim_only": {"victim", "background"},
        "combined": {"victim", "attacker", "background"},
        "static_reference": {"victim", "background"},
    }

    scenario_results: dict[str, SimulationResult] = {}
    for scenario, roles in scenario_roles.items():
        requests = [request for request in all_requests if request.role in roles]
        simulator = DynamicReroutingSimulator(
            requests=requests,
            allocations=allocations,
            configuration=configuration,
            scenario=scenario,
            trial_seed_value=stable_seed(
                configuration.configuration_id,
                victim_tag,
                trial_id,
            ),
            force_static_victim=(scenario == "static_reference"),
        )
        scenario_results[scenario] = simulator.run()

    compared = attacker_comparison(
        scenario_results["attacker_only"].request_log,
        scenario_results["combined"].request_log,
    )

    combined_log = scenario_results["combined"].request_log.copy()
    combined_events = scenario_results["combined"].event_log

    meaningful_event_types = {
        "path_switch",
        "communication_qubit_reassignment",
        "module_migration",
        "hub_migration",
        "path_failure_start",
    }
    if not combined_events.empty:
        meaningful_events = combined_events[
            combined_events["event_type"].isin(meaningful_event_types)
        ]
    else:
        meaningful_events = pd.DataFrame()
    effective_trigger_ns = (
        int(meaningful_events["time_ns"].min())
        if not meaningful_events.empty
        else RECONFIGURATION_TRIGGER_NS
    )

    compared["segment"] = compared["release_time_ns"].apply(
        lambda value: segment_for_time_at(int(value), effective_trigger_ns)
    )
    if not combined_log.empty:
        combined_log["segment"] = combined_log["release_time_ns"].apply(
            lambda value: segment_for_time_at(int(value), effective_trigger_ns)
        )

    actual_before = actual_victim_path(combined_log, "before")
    actual_after = actual_victim_path(combined_log, "after")
    predicted_before, before_score = predicted_victim_path(compared, "before")
    predicted_after, after_score = predicted_victim_path(compared, "after")

    ground_truth_path_change = victim_route_changed(combined_log)
    predicted_path_change = (
        predicted_before is not None
        and predicted_after is not None
        and predicted_before != predicted_after
    )

    reconfiguration_present = ground_truth_reconfiguration(
        configuration,
        combined_events,
    )
    change_detected, change_score_ns = detect_reconfiguration(
        compared,
        effective_trigger_ns,
    )

    victim_only_completion = scenario_results["victim_only"].victim_completion_time_ns
    combined_victim_completion = scenario_results["combined"].victim_completion_time_ns
    static_reference_completion = scenario_results[
        "static_reference"
    ].victim_completion_time_ns

    victim_slowdown_ratio = safe_ratio(
        combined_victim_completion - VICTIM_START_NS,
        victim_only_completion - VICTIM_START_NS,
    )
    rerouting_additional_victim_latency_ns = (
        victim_only_completion - static_reference_completion
    )

    victim_rows = combined_log[combined_log["role"] == "victim"]
    redirected_victim_count = int(victim_rows["rerouted"].sum()) if not victim_rows.empty else 0
    redirected_fraction = safe_ratio(redirected_victim_count, len(victim_rows))

    segment_values: dict[str, float] = {}
    for segment in ["before", "transient", "after"]:
        segment_values.update(segment_metrics(compared, segment))

    steady_mean = np.mean(
        [
            segment_values["before_mean_absolute_change_ns"],
            segment_values["after_mean_absolute_change_ns"],
        ]
    )
    transient_amplification = safe_ratio(
        segment_values["transient_mean_absolute_change_ns"],
        steady_mean,
    )

    mean_hub_utilization = np.mean(
        [
            value
            for key, value in scenario_results["combined"].resource_utilization.items()
            if key.startswith("hub:")
        ]
        or [0.0]
    )
    mean_link_utilization = np.mean(
        [
            value
            for key, value in scenario_results["combined"].resource_utilization.items()
            if key.startswith("ring:") or key.startswith("access:")
        ]
        or [0.0]
    )
    mean_communication_qubit_utilization = np.mean(
        [
            value
            for key, value in scenario_results["combined"].resource_utilization.items()
            if key.startswith("communication_qubit:")
        ]
        or [0.0]
    )

    trial_summary = {
        "configuration_id": configuration.configuration_id,
        "subexperiment": configuration.subexperiment,
        "mechanism": configuration.mechanism,
        "alternate_path_count": configuration.alternate_path_count,
        "path_selection_frequency": configuration.path_selection_frequency,
        "rerouting_threshold": configuration.rerouting_threshold,
        "path_switching_cost_ns": configuration.path_switching_cost_ns,
        "state_transfer_cost_ns": configuration.state_transfer_cost_ns,
        "decision_mode": configuration.decision_mode,
        "victim_qasm": qasm_filename,
        "victim_tag": victim_tag,
        "trial_id": trial_id,
        "attacker_probe_count": int(len(compared)),
        "effective_reconfiguration_time_ns": effective_trigger_ns,
        "mean_signed_excess_latency_ns": float(
            compared["excess_turnaround_time_ns"].mean()
        ),
        "mean_absolute_timing_change_ns": float(
            compared["absolute_timing_change_ns"].mean()
        ),
        "max_absolute_timing_change_ns": float(
            compared["absolute_timing_change_ns"].max()
        ),
        "positive_delay_fraction": float(
            compared["positive_delay_observed"].mean()
        ),
        "negative_speedup_fraction": float(
            compared["negative_speedup_observed"].mean()
        ),
        "observable_change_fraction": float(
            compared["any_observable_change"].mean()
        ),
        "failure_transition_fraction": float(
            compared["failure_transition"].mean()
        ),
        **segment_values,
        "transient_amplification": float(transient_amplification),
        "ground_truth_reconfiguration": reconfiguration_present,
        "change_detected": change_detected,
        "change_detection_correct": change_detected == reconfiguration_present,
        "change_score_ns": change_score_ns,
        "effective_reconfiguration_time_ns": effective_trigger_ns,
        "ground_truth_path_change": ground_truth_path_change,
        "predicted_path_change": predicted_path_change,
        "path_change_detection_correct": (
            predicted_path_change == ground_truth_path_change
        ),
        "actual_path_before": actual_before,
        "actual_path_after": actual_after,
        "predicted_path_before": predicted_before,
        "predicted_path_after": predicted_after,
        "path_localization_before_correct": (
            actual_before is not None and predicted_before == actual_before
        ),
        "path_localization_after_correct": (
            actual_after is not None and predicted_after == actual_after
        ),
        "path_localization_before_score_ns": before_score,
        "path_localization_after_score_ns": after_score,
        "victim_only_completion_time_ns": victim_only_completion,
        "combined_victim_completion_time_ns": combined_victim_completion,
        "static_reference_victim_completion_time_ns": static_reference_completion,
        "victim_slowdown_ratio": victim_slowdown_ratio,
        "rerouting_additional_victim_latency_ns": (
            rerouting_additional_victim_latency_ns
        ),
        "combined_total_switching_overhead_ns": (
            scenario_results["combined"].total_switching_overhead_ns
        ),
        "combined_total_state_transfer_overhead_ns": (
            scenario_results["combined"].total_state_transfer_overhead_ns
        ),
        "redirected_victim_request_count": redirected_victim_count,
        "redirected_victim_fraction": redirected_fraction,
        "communication_qubit_reassignment_count": (
            scenario_results["combined"].communication_qubit_reassignment_count
        ),
        "combined_rejected_request_count": (
            scenario_results["combined"].rejected_request_count
        ),
        "mean_hub_utilization": float(mean_hub_utilization),
        "mean_link_utilization": float(mean_link_utilization),
        "mean_communication_qubit_utilization": float(
            mean_communication_qubit_utilization
        ),
    }

    compared.insert(0, "trial_id", trial_id)
    compared.insert(0, "victim_tag", victim_tag)
    compared.insert(0, "victim_qasm", qasm_filename)
    compared.insert(0, "mechanism", configuration.mechanism)
    compared.insert(0, "subexperiment", configuration.subexperiment)
    compared.insert(0, "configuration_id", configuration.configuration_id)

    request_frames = []
    event_frames = []
    for scenario, result in scenario_results.items():
        frame = result.request_log.copy()
        if not frame.empty:
            # The simulator's request log already contains fields such as
            # configuration_id, subexperiment, mechanism, and scenario. Assign
            # metadata instead of using DataFrame.insert(), so existing columns
            # are validated/overwritten rather than inserted twice.
            request_metadata = {
                "configuration_id": configuration.configuration_id,
                "decision_mode": configuration.decision_mode,
                "victim_qasm": qasm_filename,
                "victim_tag": victim_tag,
                "trial_id": trial_id,
            }
            for column, value in request_metadata.items():
                frame[column] = value

            request_front_columns = list(request_metadata)
            frame = frame[
                request_front_columns
                + [
                    column
                    for column in frame.columns
                    if column not in request_front_columns
                ]
            ]
            request_frames.append(frame)

        event_frame = result.event_log.copy()
        if not event_frame.empty:
            # Event rows are also created with configuration metadata inside the
            # simulator, so use assignment here for the same reason.
            event_metadata = {
                "configuration_id": configuration.configuration_id,
                "mechanism": configuration.mechanism,
                "victim_qasm": qasm_filename,
                "victim_tag": victim_tag,
                "trial_id": trial_id,
            }
            for column, value in event_metadata.items():
                event_frame[column] = value

            event_front_columns = list(event_metadata)
            event_frame = event_frame[
                event_front_columns
                + [
                    column
                    for column in event_frame.columns
                    if column not in event_front_columns
                ]
            ]
            event_frames.append(event_frame)

    request_log = (
        pd.concat(request_frames, ignore_index=True)
        if request_frames
        else pd.DataFrame()
    )
    event_log = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )

    localization_row = {
        "configuration_id": configuration.configuration_id,
        "subexperiment": configuration.subexperiment,
        "mechanism": configuration.mechanism,
        "victim_qasm": qasm_filename,
        "victim_tag": victim_tag,
        "trial_id": trial_id,
        "actual_path_before": actual_before,
        "actual_path_after": actual_after,
        "predicted_path_before": predicted_before,
        "predicted_path_after": predicted_after,
        "before_correct": actual_before is not None and predicted_before == actual_before,
        "after_correct": actual_after is not None and predicted_after == actual_after,
        "ground_truth_path_change": ground_truth_path_change,
        "predicted_path_change": predicted_path_change,
        "path_change_detection_correct": predicted_path_change == ground_truth_path_change,
    }

    change_row = {
        "configuration_id": configuration.configuration_id,
        "subexperiment": configuration.subexperiment,
        "mechanism": configuration.mechanism,
        "victim_qasm": qasm_filename,
        "victim_tag": victim_tag,
        "trial_id": trial_id,
        "ground_truth_reconfiguration": reconfiguration_present,
        "change_detected": change_detected,
        "change_detection_correct": change_detected == reconfiguration_present,
        "change_score_ns": change_score_ns,
        "transient_mean_absolute_change_ns": segment_values[
            "transient_mean_absolute_change_ns"
        ],
        "steady_mean_absolute_change_ns": float(steady_mean),
        "transient_amplification": float(transient_amplification),
    }

    return (
        trial_summary,
        compared,
        request_log,
        event_log,
        localization_row,
        change_row,
    )


# =============================================================================
# Aggregate summaries
# =============================================================================


def aggregate_by(
    trials: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    return (
        trials.groupby(list(columns), as_index=False, dropna=False)
        .agg(
            trial_count=("trial_id", "count"),
            mean_absolute_timing_change_ns=(
                "mean_absolute_timing_change_ns",
                "mean",
            ),
            std_absolute_timing_change_ns=(
                "mean_absolute_timing_change_ns",
                "std",
            ),
            mean_positive_delay_fraction=("positive_delay_fraction", "mean"),
            mean_negative_speedup_fraction=("negative_speedup_fraction", "mean"),
            mean_observable_change_fraction=("observable_change_fraction", "mean"),
            mean_before_leakage_ns=("before_mean_absolute_change_ns", "mean"),
            mean_transient_leakage_ns=("transient_mean_absolute_change_ns", "mean"),
            mean_after_leakage_ns=("after_mean_absolute_change_ns", "mean"),
            mean_transient_amplification=("transient_amplification", "mean"),
            reconfiguration_detection_accuracy=("change_detection_correct", "mean"),
            path_change_detection_accuracy=("path_change_detection_correct", "mean"),
            path_localization_before_accuracy=(
                "path_localization_before_correct",
                "mean",
            ),
            path_localization_after_accuracy=(
                "path_localization_after_correct",
                "mean",
            ),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
            mean_rerouting_additional_victim_latency_ns=(
                "rerouting_additional_victim_latency_ns",
                "mean",
            ),
            mean_switching_overhead_ns=(
                "combined_total_switching_overhead_ns",
                "mean",
            ),
            mean_state_transfer_overhead_ns=(
                "combined_total_state_transfer_overhead_ns",
                "mean",
            ),
            mean_redirected_victim_fraction=("redirected_victim_fraction", "mean"),
            mean_hub_utilization=("mean_hub_utilization", "mean"),
            mean_link_utilization=("mean_link_utilization", "mean"),
            mean_communication_qubit_utilization=(
                "mean_communication_qubit_utilization",
                "mean",
            ),
        )
    )



def fingerprint_stability(
    attacker_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping_columns = [
        "configuration_id",
        "subexperiment",
        "mechanism",
        "victim_qasm",
        "victim_tag",
    ]

    for keys, group in attacker_comparisons.groupby(grouping_columns, dropna=False):
        traces: dict[int, np.ndarray] = {}
        for trial_id, trial_group in group.groupby("trial_id"):
            ordered = trial_group.sort_values("circuit_order")
            traces[int(trial_id)] = ordered[
                "excess_turnaround_time_ns"
            ].to_numpy(dtype=float)

        correlations: list[float] = []
        mean_absolute_distances: list[float] = []
        for first_id, second_id in itertools.combinations(sorted(traces), 2):
            first = traces[first_id]
            second = traces[second_id]
            length = min(len(first), len(second))
            if length == 0:
                continue
            first = first[:length]
            second = second[:length]
            mean_absolute_distances.append(float(np.mean(np.abs(first - second))))
            if np.std(first) == 0 or np.std(second) == 0:
                correlations.append(1.0 if np.array_equal(first, second) else 0.0)
            else:
                correlation = float(np.corrcoef(first, second)[0, 1])
                correlations.append(correlation if np.isfinite(correlation) else 0.0)

        row = dict(zip(grouping_columns, keys))
        row.update(
            {
                "trial_count": len(traces),
                "mean_pairwise_correlation": (
                    float(np.mean(correlations)) if correlations else 1.0
                ),
                "mean_pairwise_mae_ns": (
                    float(np.mean(mean_absolute_distances))
                    if mean_absolute_distances
                    else 0.0
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)



def save_summary_plots(
    mechanism_summary: pd.DataFrame,
) -> None:
    ordered = mechanism_summary.set_index("mechanism").reindex(MECHANISMS)

    axis = ordered["mean_absolute_timing_change_ns"].plot(
        kind="bar",
        figsize=(13, 6),
    )
    axis.set_xlabel("Rerouting/remapping mechanism")
    axis.set_ylabel("Mean absolute attacker timing change (ns)")
    axis.set_title("Phase 1.5: Leakage Under Dynamic Rerouting")
    axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "dynamic_rerouting_mechanism_leakage.png",
        dpi=300,
    )
    plt.close()

    axis = ordered[
        [
            "mean_before_leakage_ns",
            "mean_transient_leakage_ns",
            "mean_after_leakage_ns",
        ]
    ].plot(
        kind="bar",
        figsize=(14, 6),
    )
    axis.set_xlabel("Rerouting/remapping mechanism")
    axis.set_ylabel("Mean absolute timing change (ns)")
    axis.set_title("Leakage Before, During, and After Reconfiguration")
    axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "dynamic_rerouting_before_transient_after.png",
        dpi=300,
    )
    plt.close()

    axis = ordered[
        [
            "reconfiguration_detection_accuracy",
            "path_change_detection_accuracy",
            "path_localization_after_accuracy",
        ]
    ].plot(
        kind="bar",
        figsize=(14, 6),
    )
    axis.set_xlabel("Rerouting/remapping mechanism")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.05)
    axis.set_title("Attacker Detection and Path Localization")
    axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "dynamic_rerouting_detection_localization.png",
        dpi=300,
    )
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configurations = build_configurations()
    trial_rows: list[dict[str, Any]] = []
    attacker_frames: list[pd.DataFrame] = []
    request_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    localization_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []

    total_trials = (
        len(configurations)
        * len(VICTIM_QASMS)
        * TRIALS_PER_CONFIGURATION
    )
    completed_trials = 0

    print("Phase 1.5 — Dynamic rerouting and remapping")
    print(f"Configurations: {len(configurations)}")
    print(f"Trial tuples:   {total_trials}")

    for configuration in configurations:
        for victim_qasm in VICTIM_QASMS:
            for trial_id in range(TRIALS_PER_CONFIGURATION):
                completed_trials += 1
                print(
                    f"[{completed_trials:04d}/{total_trials:04d}] "
                    f"{configuration.subexperiment} | "
                    f"{configuration.mechanism} | "
                    f"paths={configuration.alternate_path_count} | "
                    f"{victim_qasm} | trial={trial_id}"
                )

                (
                    trial_summary,
                    attacker_frame,
                    request_frame,
                    event_frame,
                    localization_row,
                    change_row,
                ) = run_one_trial(
                    configuration,
                    victim_qasm,
                    trial_id,
                )

                trial_rows.append(trial_summary)
                attacker_frames.append(attacker_frame)
                if not request_frame.empty:
                    request_frames.append(request_frame)
                if not event_frame.empty:
                    event_frames.append(event_frame)
                localization_rows.append(localization_row)
                change_rows.append(change_row)

    trials = pd.DataFrame(trial_rows)
    attackers = pd.concat(attacker_frames, ignore_index=True)
    requests = (
        pd.concat(request_frames, ignore_index=True)
        if request_frames
        else pd.DataFrame()
    )
    events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )
    localization = pd.DataFrame(localization_rows)
    change_detection = pd.DataFrame(change_rows)

    trials.to_csv(
        OUTPUT_DIR / "dynamic_rerouting_trial_summary.csv",
        index=False,
    )
    localization.to_csv(
        OUTPUT_DIR / "dynamic_rerouting_path_localization.csv",
        index=False,
    )
    change_detection.to_csv(
        OUTPUT_DIR / "dynamic_rerouting_change_detection.csv",
        index=False,
    )

    if SAVE_ATTACKER_COMPARISON:
        attackers.to_csv(
            OUTPUT_DIR / "dynamic_rerouting_attacker_comparison.csv.gz",
            index=False,
            compression="gzip",
        )
    if SAVE_REQUEST_LOG and not requests.empty:
        requests.to_csv(
            OUTPUT_DIR / "dynamic_rerouting_request_log.csv.gz",
            index=False,
            compression="gzip",
        )
    if SAVE_EVENT_LOG and not events.empty:
        events.to_csv(
            OUTPUT_DIR / "dynamic_rerouting_event_log.csv",
            index=False,
        )

    summaries = {
        "dynamic_rerouting_mechanism_summary.csv": aggregate_by(
            trials,
            ["mechanism"],
        ),
        "dynamic_rerouting_path_count_summary.csv": aggregate_by(
            trials,
            ["alternate_path_count"],
        ),
        "dynamic_rerouting_frequency_summary.csv": aggregate_by(
            trials,
            ["path_selection_frequency"],
        ),
        "dynamic_rerouting_threshold_summary.csv": aggregate_by(
            trials,
            ["rerouting_threshold"],
        ),
        "dynamic_rerouting_switching_cost_summary.csv": aggregate_by(
            trials,
            ["path_switching_cost_ns"],
        ),
        "dynamic_rerouting_state_transfer_summary.csv": aggregate_by(
            trials,
            ["state_transfer_cost_ns"],
        ),
        "dynamic_rerouting_decision_mode_summary.csv": aggregate_by(
            trials,
            ["decision_mode"],
        ),
        "dynamic_rerouting_segment_summary.csv": aggregate_by(
            trials,
            ["mechanism", "subexperiment"],
        ),
    }

    for filename, dataframe in summaries.items():
        dataframe.to_csv(OUTPUT_DIR / filename, index=False)

    stability = fingerprint_stability(attackers)
    stability.to_csv(
        OUTPUT_DIR / "dynamic_rerouting_fingerprint_stability.csv",
        index=False,
    )

    save_summary_plots(summaries["dynamic_rerouting_mechanism_summary.csv"])

    print("\n=== Mechanism summary ===")
    display_columns = [
        "mechanism",
        "trial_count",
        "mean_absolute_timing_change_ns",
        "mean_before_leakage_ns",
        "mean_transient_leakage_ns",
        "mean_after_leakage_ns",
        "reconfiguration_detection_accuracy",
        "path_change_detection_accuracy",
        "path_localization_after_accuracy",
        "mean_rerouting_additional_victim_latency_ns",
        "mean_redirected_victim_fraction",
    ]
    print(
        summaries["dynamic_rerouting_mechanism_summary.csv"][
            display_columns
        ].to_string(index=False)
    )
    print(f"\nSaved all Phase 1.5 results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
