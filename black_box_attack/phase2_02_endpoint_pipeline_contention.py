#!/usr/bin/env python3
"""
Phase 2.2 — Endpoint Pipeline Contention
========================================

This experiment replaces the Phase 1 ``module_busy``-style endpoint abstraction
with an explicit superconducting-module endpoint pipeline.  It uses the staged
causal model validated in Phase 2.1 and varies one endpoint component at a time.

The black-box attacker observes only its own request release/completion timing.
Internal stage names, physical resource identities, and blocking labels are
written to evaluator-only outputs so that the observed timing can be explained
without granting those labels to the attacker.

Run
---
    python phase2_02_endpoint_pipeline_contention.py

Optional arguments
------------------
    --trials 12
    --seed 2202
    --no-plots
    --output-dir PATH

Default output directory
------------------------
blackbox_window_results/
└── phase2/
    └── phase2_02_endpoint_pipeline_contention/
        ├── raw/
        ├── plots/
        ├── phase2_02_configuration_table.csv
        ├── phase2_02_scenario_summary.csv
        ├── phase2_02_workload_summary.csv
        ├── phase2_02_delay_decomposition_summary.csv
        ├── phase2_02_blocking_attribution_summary.csv
        ├── phase2_02_resource_utilization_summary.csv
        ├── phase2_02_endpoint_reuse_summary.csv
        ├── phase2_02_blackbox_trace_summary.csv
        ├── phase2_02_validation_assertions.csv
        ├── phase2_02_validation_summary.csv
        └── phase2_02_run_manifest.json

Primary architectural controls
------------------------------
1. Separate modules and separate endpoint pipelines.
2. Same module, but fully independent endpoint pipelines.
3. Shared local route/coupler only.
4. Shared communication qubit/interface only.
5. Shared interconnect-facing port only.
6. Shared reset/recovery engine only.
7. Shared route + communication qubit + port, but separate reset engines.
8. Shared full endpoint pipeline.

The intended progression is:

    module co-location alone -> no leakage
    one shared endpoint component -> component-specific leakage
    duplicated endpoint pipeline -> leakage removed
    duplicated endpoint with shared reset -> weaker late-stage leakage
    full endpoint sharing -> strongest multi-stage endpoint channel
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Experiment defaults
# =============================================================================

DEFAULT_OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase2"
    / "phase2_02_endpoint_pipeline_contention"
)

DEFAULT_TRIALS = 12
DEFAULT_SEED = 2202
OBSERVATION_WINDOW_NS = 20_000.0
ATTACKER_FIRST_RELEASE_NS = 30.0
ATTACKER_PERIOD_NS = 420.0
AFFECTED_THRESHOLD_NS = 1e-9
FLOAT_TOLERANCE_NS = 1e-9
FAIL_ON_VALIDATION_ERROR = True

# A direct remote-CX-like endpoint pipeline.  The exact values are architectural
# parameters, not claims about a specific vendor device.
STAGE_DURATIONS_NS = {
    "local_route_acquisition": 10.0,
    "communication_qubit_load": 20.0,
    "interconnect_port_acquisition": 10.0,
    "switch_path_acquisition": 15.0,
    "intermodule_transfer": 80.0,
    "receiver_side_gate": 30.0,
    "communication_reset": 40.0,
}

ENDPOINT_COMPONENTS = (
    "local_route",
    "communication_qubit",
    "interconnect_port",
    "reset_engine",
)

NETWORK_COMPONENTS = (
    "switch_path",
    "intermodule_link",
    "receiver_engine",
)

STAGE_TO_COMPONENT = {
    "local_route_acquisition": "local_route",
    "communication_qubit_load": "communication_qubit",
    "interconnect_port_acquisition": "interconnect_port",
    "switch_path_acquisition": "switch_path",
    "intermodule_transfer": "intermodule_link",
    "receiver_side_gate": "receiver_engine",
    "communication_reset": "reset_engine",
}


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class StageDefinition:
    stage_name: str
    duration_ns: float
    acquire_held: tuple[str, ...] = ()
    acquire_scoped: tuple[str, ...] = ()
    require_held: tuple[str, ...] = ()
    release_held: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class EndpointScenario:
    scenario_name: str
    same_module: bool
    shared_components: tuple[str, ...]
    description: str
    expected_mechanism: str


@dataclass(frozen=True)
class VictimWorkload:
    workload_name: str
    description: str
    release_pattern: str


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    tenant: str
    ready_ns: float
    request_index: int
    workload_name: str
    trial_id: int


@dataclass
class StageRecord:
    run_kind: str
    scenario_name: str
    workload_name: str
    trial_id: int
    request_id: str
    tenant: str
    request_index: int
    stage_index: int
    stage_name: str
    causal_component: str
    stage_ready_ns: float
    stage_start_ns: float
    stage_end_ns: float
    wait_ns: float
    service_ns: float
    physical_acquired_resources: str
    physical_required_resources: str
    physical_released_resources: str
    blocking_physical_resources: str
    blocking_components: str
    blocking_owner_tenants: str
    cross_tenant_blocked: bool
    self_blocked: bool


@dataclass
class RequestRecord:
    run_kind: str
    scenario_name: str
    workload_name: str
    trial_id: int
    request_id: str
    tenant: str
    request_index: int
    ready_ns: float
    completion_ns: float
    turnaround_ns: float
    total_wait_ns: float
    total_service_ns: float
    completed: bool


@dataclass
class ResourceInterval:
    run_kind: str
    scenario_name: str
    workload_name: str
    trial_id: int
    physical_resource: str
    logical_component: str
    shared_between_tenants: bool
    tenant: str
    request_id: str
    interval_kind: str
    acquired_stage: str
    released_stage: str
    start_ns: float
    end_ns: float
    occupancy_ns: float


@dataclass
class ValidationAssertion:
    validation_group: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


@dataclass
class ActiveStage:
    end_ns: float
    sequence: int
    request_id: str
    stage_index: int
    scoped_resources: tuple[str, ...]

    def heap_key(self) -> tuple[float, int, str, int]:
        return self.end_ns, self.sequence, self.request_id, self.stage_index


@dataclass
class RequestRuntime:
    spec: RequestSpec
    stage_index: int = 0
    stage_ready_ns: float = 0.0
    active: bool = False
    completed: bool = False
    completion_ns: float = math.nan
    held_resources: dict[str, tuple[str, float, str]] = field(default_factory=dict)
    stage_records: list[StageRecord] = field(default_factory=list)
    blocking_observations: set[tuple[str, str, str]] = field(default_factory=set)


@dataclass
class ResourceState:
    physical_resource: str
    logical_component: str
    shared_between_tenants: bool
    owner_request_id: Optional[str] = None
    owner_tenant: Optional[str] = None
    acquired_ns: Optional[float] = None
    acquired_stage: Optional[str] = None
    interval_kind: Optional[str] = None

    @property
    def free(self) -> bool:
        return self.owner_request_id is None


# =============================================================================
# Pipeline and scenario definitions
# =============================================================================


def build_endpoint_pipeline() -> tuple[StageDefinition, ...]:
    """Explicit source-endpoint pipeline for one remote CX-like primitive."""

    return (
        StageDefinition(
            stage_name="local_route_acquisition",
            duration_ns=STAGE_DURATIONS_NS["local_route_acquisition"],
            acquire_held=("local_route",),
            description=(
                "Acquire the local coupling/routing resource that moves the "
                "source state toward the communication interface."
            ),
        ),
        StageDefinition(
            stage_name="communication_qubit_load",
            duration_ns=STAGE_DURATIONS_NS["communication_qubit_load"],
            acquire_held=("communication_qubit",),
            require_held=("local_route",),
            release_held=("local_route",),
            description=(
                "Load or couple the operation into the source communication "
                "qubit/interface, then release the local route."
            ),
        ),
        StageDefinition(
            stage_name="interconnect_port_acquisition",
            duration_ns=STAGE_DURATIONS_NS["interconnect_port_acquisition"],
            acquire_held=("interconnect_port",),
            require_held=("communication_qubit",),
            description="Acquire the interconnect-facing endpoint port.",
        ),
        StageDefinition(
            stage_name="switch_path_acquisition",
            duration_ns=STAGE_DURATIONS_NS["switch_path_acquisition"],
            acquire_held=("switch_path",),
            require_held=("communication_qubit", "interconnect_port"),
            description="Acquire the remote-network switch path.",
        ),
        StageDefinition(
            stage_name="intermodule_transfer",
            duration_ns=STAGE_DURATIONS_NS["intermodule_transfer"],
            acquire_scoped=("intermodule_link",),
            require_held=(
                "communication_qubit",
                "interconnect_port",
                "switch_path",
            ),
            release_held=("interconnect_port", "switch_path"),
            description=(
                "Execute the inter-module primitive, then release the port and "
                "switch path while retaining the communication qubit."
            ),
        ),
        StageDefinition(
            stage_name="receiver_side_gate",
            duration_ns=STAGE_DURATIONS_NS["receiver_side_gate"],
            acquire_scoped=("receiver_engine",),
            require_held=("communication_qubit",),
            description="Execute the receiver-side local operation.",
        ),
        StageDefinition(
            stage_name="communication_reset",
            duration_ns=STAGE_DURATIONS_NS["communication_reset"],
            acquire_scoped=("reset_engine",),
            require_held=("communication_qubit",),
            release_held=("communication_qubit",),
            description=(
                "Reset/reinitialize the communication qubit before releasing "
                "it for reuse."
            ),
        ),
    )


def build_scenarios() -> tuple[EndpointScenario, ...]:
    return (
        EndpointScenario(
            scenario_name="separate_modules_control",
            same_module=False,
            shared_components=(),
            description=(
                "Victim and attacker occupy different modules and use fully "
                "independent endpoint pipelines."
            ),
            expected_mechanism="no_cross_tenant_endpoint_contention",
        ),
        EndpointScenario(
            scenario_name="shared_module_fully_independent",
            same_module=True,
            shared_components=(),
            description=(
                "Victim and attacker share a physical module but use duplicated "
                "local routes, communication qubits, ports, and reset engines."
            ),
            expected_mechanism="module_colocation_without_pipeline_sharing",
        ),
        EndpointScenario(
            scenario_name="shared_local_route_only",
            same_module=True,
            shared_components=("local_route",),
            description=(
                "Compute qubits and the remaining endpoint pipeline are separate, "
                "but both tenants use one local route/coupler."
            ),
            expected_mechanism="local_route_wait",
        ),
        EndpointScenario(
            scenario_name="shared_communication_qubit_only",
            same_module=True,
            shared_components=("communication_qubit",),
            description=(
                "Tenants have separate compute routing, ports, and reset engines "
                "but use one communication qubit/interface."
            ),
            expected_mechanism="communication_qubit_reuse_wait",
        ),
        EndpointScenario(
            scenario_name="shared_interconnect_port_only",
            same_module=True,
            shared_components=("interconnect_port",),
            description=(
                "Tenants have separate local routes and communication qubits but "
                "use one interconnect-facing port."
            ),
            expected_mechanism="interconnect_port_wait",
        ),
        EndpointScenario(
            scenario_name="shared_reset_engine_only",
            same_module=True,
            shared_components=("reset_engine",),
            description=(
                "The active endpoint paths are duplicated, but both communication "
                "qubits use one reset/recovery engine."
            ),
            expected_mechanism="late_reset_recovery_wait",
        ),
        EndpointScenario(
            scenario_name="shared_endpoint_without_reset",
            same_module=True,
            shared_components=(
                "local_route",
                "communication_qubit",
                "interconnect_port",
            ),
            description=(
                "The active endpoint path is shared, but each tenant has a "
                "dedicated reset/recovery engine."
            ),
            expected_mechanism="active_endpoint_contention",
        ),
        EndpointScenario(
            scenario_name="shared_full_endpoint_pipeline",
            same_module=True,
            shared_components=ENDPOINT_COMPONENTS,
            description=(
                "Local route, communication qubit, interconnect port, and reset "
                "engine are all shared."
            ),
            expected_mechanism="full_endpoint_pipeline_contention",
        ),
    )


def build_workloads() -> tuple[VictimWorkload, ...]:
    return (
        VictimWorkload(
            workload_name="periodic_sparse",
            description="One remote operation every 900 ns.",
            release_pattern="period=900",
        ),
        VictimWorkload(
            workload_name="periodic_dense",
            description="One remote operation every 460 ns.",
            release_pattern="period=460",
        ),
        VictimWorkload(
            workload_name="layered_bursty",
            description=(
                "Four-operation communication layers separated by long local "
                "compute intervals."
            ),
            release_pattern="burst_period=2400; offsets=0,120,240,360",
        ),
    )


# =============================================================================
# Request generation and physical resource mapping
# =============================================================================


def generate_attacker_specs(
    *,
    trial_id: int,
    workload_name: str,
) -> list[RequestSpec]:
    releases = np.arange(
        ATTACKER_FIRST_RELEASE_NS,
        OBSERVATION_WINDOW_NS,
        ATTACKER_PERIOD_NS,
        dtype=float,
    )
    return [
        RequestSpec(
            request_id=f"A_t{trial_id:03d}_p{index:03d}",
            tenant="attacker",
            ready_ns=float(release_ns),
            request_index=index,
            workload_name=workload_name,
            trial_id=trial_id,
        )
        for index, release_ns in enumerate(releases)
    ]


def generate_victim_specs(
    *,
    workload: VictimWorkload,
    trial_id: int,
    phase_ns: float,
) -> list[RequestSpec]:
    releases: list[float] = []

    if workload.workload_name == "periodic_sparse":
        releases = list(
            np.arange(phase_ns, OBSERVATION_WINDOW_NS, 900.0, dtype=float)
        )
    elif workload.workload_name == "periodic_dense":
        releases = list(
            np.arange(phase_ns, OBSERVATION_WINDOW_NS, 460.0, dtype=float)
        )
    elif workload.workload_name == "layered_bursty":
        burst_start = phase_ns
        while burst_start < OBSERVATION_WINDOW_NS:
            for offset_ns in (0.0, 120.0, 240.0, 360.0):
                release_ns = burst_start + offset_ns
                if release_ns < OBSERVATION_WINDOW_NS:
                    releases.append(release_ns)
            burst_start += 2400.0
    else:
        raise ValueError(f"unknown workload: {workload.workload_name}")

    return [
        RequestSpec(
            request_id=f"V_t{trial_id:03d}_r{index:03d}",
            tenant="victim",
            ready_ns=float(release_ns),
            request_index=index,
            workload_name=workload.workload_name,
            trial_id=trial_id,
        )
        for index, release_ns in enumerate(sorted(releases))
    ]


class ResourceResolver:
    """Map logical pipeline components to physical evaluator-side resources."""

    def __init__(self, scenario: EndpointScenario) -> None:
        self.scenario = scenario

    def resolve(self, tenant: str, logical_component: str) -> str:
        if logical_component in ENDPOINT_COMPONENTS:
            if logical_component in self.scenario.shared_components:
                return f"shared_module::{logical_component}"
            return f"{tenant}::endpoint::{logical_component}"

        # Network and receiver-side resources are intentionally duplicated in
        # Phase 2.2 so that measured cross-tenant leakage is endpoint-local.
        if logical_component in NETWORK_COMPONENTS:
            return f"{tenant}::network::{logical_component}"

        raise KeyError(f"unknown logical component: {logical_component}")

    def is_shared(self, logical_component: str) -> bool:
        return logical_component in self.scenario.shared_components

    def module_label(self, tenant: str) -> str:
        if self.scenario.same_module:
            return "module_shared"
        return f"module_{tenant}"


# =============================================================================
# Event-driven endpoint simulator
# =============================================================================


class EndpointPipelineSimulator:
    """FCFS, work-conserving, stage-level resource simulator."""

    def __init__(
        self,
        *,
        scenario: EndpointScenario,
        pipeline: tuple[StageDefinition, ...],
        run_kind: str,
        workload_name: str,
        trial_id: int,
    ) -> None:
        self.scenario = scenario
        self.pipeline = pipeline
        self.run_kind = run_kind
        self.workload_name = workload_name
        self.trial_id = trial_id
        self.resolver = ResourceResolver(scenario)

        self.resources: dict[str, ResourceState] = {}
        self.resource_intervals: list[ResourceInterval] = []
        self._active_heap: list[tuple[float, int, str, int, tuple[str, ...]]] = []
        self._event_sequence = 0

    def _resource_state(
        self,
        tenant: str,
        logical_component: str,
    ) -> ResourceState:
        physical = self.resolver.resolve(tenant, logical_component)
        if physical not in self.resources:
            self.resources[physical] = ResourceState(
                physical_resource=physical,
                logical_component=logical_component,
                shared_between_tenants=self.resolver.is_shared(logical_component),
            )
        return self.resources[physical]

    def _acquire(
        self,
        *,
        runtime: RequestRuntime,
        logical_component: str,
        stage_name: str,
        now_ns: float,
        interval_kind: str,
    ) -> str:
        state = self._resource_state(runtime.spec.tenant, logical_component)
        if not state.free:
            raise RuntimeError(
                f"attempted to acquire occupied resource {state.physical_resource}"
            )
        state.owner_request_id = runtime.spec.request_id
        state.owner_tenant = runtime.spec.tenant
        state.acquired_ns = now_ns
        state.acquired_stage = stage_name
        state.interval_kind = interval_kind
        return state.physical_resource

    def _release(
        self,
        *,
        runtime: RequestRuntime,
        logical_component: str,
        released_stage: str,
        now_ns: float,
    ) -> str:
        state = self._resource_state(runtime.spec.tenant, logical_component)
        if state.owner_request_id != runtime.spec.request_id:
            raise RuntimeError(
                f"{runtime.spec.request_id} cannot release "
                f"{state.physical_resource}, owned by {state.owner_request_id}"
            )
        assert state.acquired_ns is not None
        assert state.acquired_stage is not None
        assert state.interval_kind is not None
        self.resource_intervals.append(
            ResourceInterval(
                run_kind=self.run_kind,
                scenario_name=self.scenario.scenario_name,
                workload_name=self.workload_name,
                trial_id=self.trial_id,
                physical_resource=state.physical_resource,
                logical_component=state.logical_component,
                shared_between_tenants=state.shared_between_tenants,
                tenant=runtime.spec.tenant,
                request_id=runtime.spec.request_id,
                interval_kind=state.interval_kind,
                acquired_stage=state.acquired_stage,
                released_stage=released_stage,
                start_ns=float(state.acquired_ns),
                end_ns=float(now_ns),
                occupancy_ns=float(now_ns - state.acquired_ns),
            )
        )
        physical = state.physical_resource
        state.owner_request_id = None
        state.owner_tenant = None
        state.acquired_ns = None
        state.acquired_stage = None
        state.interval_kind = None
        return physical

    def _stage_new_components(self, stage: StageDefinition) -> tuple[str, ...]:
        return tuple(stage.acquire_held) + tuple(stage.acquire_scoped)

    def _stage_startable(
        self,
        runtime: RequestRuntime,
        stage: StageDefinition,
    ) -> bool:
        for logical_component in stage.require_held:
            if logical_component not in runtime.held_resources:
                raise RuntimeError(
                    f"{runtime.spec.request_id} entered {stage.stage_name} "
                    f"without held {logical_component}"
                )

        return all(
            self._resource_state(runtime.spec.tenant, component).free
            for component in self._stage_new_components(stage)
        )

    def _record_current_blockers(
        self,
        runtime: RequestRuntime,
        stage: StageDefinition,
    ) -> None:
        """Record evaluator-side owners of resources blocking this stage."""

        for logical_component in self._stage_new_components(stage):
            state = self._resource_state(runtime.spec.tenant, logical_component)
            if not state.free:
                runtime.blocking_observations.add(
                    (
                        state.physical_resource,
                        logical_component,
                        str(state.owner_tenant),
                    )
                )

    def _start_stage(
        self,
        *,
        runtime: RequestRuntime,
        now_ns: float,
    ) -> None:
        stage_index = runtime.stage_index
        stage = self.pipeline[stage_index]

        acquired_physical: list[str] = []
        required_physical: list[str] = []

        for logical_component in stage.require_held:
            physical, _, _ = runtime.held_resources[logical_component]
            required_physical.append(physical)

        for logical_component in stage.acquire_held:
            physical = self._acquire(
                runtime=runtime,
                logical_component=logical_component,
                stage_name=stage.stage_name,
                now_ns=now_ns,
                interval_kind="held_across_stages",
            )
            runtime.held_resources[logical_component] = (
                physical,
                now_ns,
                stage.stage_name,
            )
            acquired_physical.append(physical)

        scoped_physical: list[str] = []
        for logical_component in stage.acquire_scoped:
            physical = self._acquire(
                runtime=runtime,
                logical_component=logical_component,
                stage_name=stage.stage_name,
                now_ns=now_ns,
                interval_kind="stage_scoped",
            )
            scoped_physical.append(physical)
            acquired_physical.append(physical)

        stage_end_ns = now_ns + stage.duration_ns
        runtime.stage_records.append(
            StageRecord(
                run_kind=self.run_kind,
                scenario_name=self.scenario.scenario_name,
                workload_name=self.workload_name,
                trial_id=self.trial_id,
                request_id=runtime.spec.request_id,
                tenant=runtime.spec.tenant,
                request_index=runtime.spec.request_index,
                stage_index=stage_index,
                stage_name=stage.stage_name,
                causal_component=STAGE_TO_COMPONENT[stage.stage_name],
                stage_ready_ns=float(runtime.stage_ready_ns),
                stage_start_ns=float(now_ns),
                stage_end_ns=float(stage_end_ns),
                wait_ns=float(now_ns - runtime.stage_ready_ns),
                service_ns=float(stage.duration_ns),
                physical_acquired_resources=";".join(acquired_physical),
                physical_required_resources=";".join(required_physical),
                physical_released_resources=";".join(
                    self.resolver.resolve(runtime.spec.tenant, component)
                    for component in stage.release_held
                ),
                blocking_physical_resources=";".join(
                    sorted(item[0] for item in runtime.blocking_observations)
                ),
                blocking_components=";".join(
                    sorted({item[1] for item in runtime.blocking_observations})
                ),
                blocking_owner_tenants=";".join(
                    sorted({item[2] for item in runtime.blocking_observations})
                ),
                cross_tenant_blocked=any(
                    item[2] != runtime.spec.tenant
                    for item in runtime.blocking_observations
                ),
                self_blocked=any(
                    item[2] == runtime.spec.tenant
                    for item in runtime.blocking_observations
                ),
            )
        )
        runtime.blocking_observations.clear()

        runtime.active = True
        self._event_sequence += 1
        heapq.heappush(
            self._active_heap,
            (
                stage_end_ns,
                self._event_sequence,
                runtime.spec.request_id,
                stage_index,
                tuple(scoped_physical),
            ),
        )

    def _complete_stage(
        self,
        *,
        runtime: RequestRuntime,
        stage_index: int,
        now_ns: float,
        scoped_physical: tuple[str, ...],
    ) -> None:
        if runtime.stage_index != stage_index or not runtime.active:
            raise RuntimeError("stage completion does not match runtime state")

        stage = self.pipeline[stage_index]

        # Scoped resources are released at stage completion.
        for logical_component in stage.acquire_scoped:
            physical = self._release(
                runtime=runtime,
                logical_component=logical_component,
                released_stage=stage.stage_name,
                now_ns=now_ns,
            )
            if physical not in scoped_physical:
                raise RuntimeError("scoped resource completion mismatch")

        # Held resources are released only at explicit pipeline boundaries.
        for logical_component in stage.release_held:
            self._release(
                runtime=runtime,
                logical_component=logical_component,
                released_stage=stage.stage_name,
                now_ns=now_ns,
            )
            runtime.held_resources.pop(logical_component, None)

        runtime.active = False
        runtime.stage_index += 1
        runtime.stage_ready_ns = float(now_ns)

        if runtime.stage_index == len(self.pipeline):
            if runtime.held_resources:
                raise RuntimeError(
                    f"request completed with held resources: "
                    f"{runtime.held_resources}"
                )
            runtime.completed = True
            runtime.completion_ns = float(now_ns)

    @staticmethod
    def _candidate_sort_key(runtime: RequestRuntime) -> tuple[float, int, int, str]:
        # Victim wins exact ties.  This is deterministic and conservative for
        # measuring whether victim occupancy can delay an attacker probe.
        tenant_order = 0 if runtime.spec.tenant == "victim" else 1
        return (
            runtime.stage_ready_ns,
            tenant_order,
            runtime.spec.request_index,
            runtime.spec.request_id,
        )

    def run(
        self,
        request_specs: Iterable[RequestSpec],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        specs = list(request_specs)
        runtimes = {
            spec.request_id: RequestRuntime(
                spec=spec,
                stage_ready_ns=float(spec.ready_ns),
            )
            for spec in specs
        }

        now_ns = min((spec.ready_ns for spec in specs), default=0.0)
        completed_count = 0

        while completed_count < len(runtimes):
            # Complete every stage ending at the current timestamp.
            while self._active_heap and self._active_heap[0][0] <= now_ns + FLOAT_TOLERANCE_NS:
                (
                    end_ns,
                    _,
                    request_id,
                    stage_index,
                    scoped_resources,
                ) = heapq.heappop(self._active_heap)
                runtime = runtimes[request_id]
                was_completed = runtime.completed
                self._complete_stage(
                    runtime=runtime,
                    stage_index=stage_index,
                    now_ns=float(end_ns),
                    scoped_physical=scoped_resources,
                )
                if runtime.completed and not was_completed:
                    completed_count += 1

            # Start every causally ready stage that can acquire its resources.
            made_progress = True
            while made_progress:
                made_progress = False
                candidates = sorted(
                    (
                        runtime
                        for runtime in runtimes.values()
                        if not runtime.active
                        and not runtime.completed
                        and runtime.stage_ready_ns <= now_ns + FLOAT_TOLERANCE_NS
                    ),
                    key=self._candidate_sort_key,
                )
                for runtime in candidates:
                    stage = self.pipeline[runtime.stage_index]
                    if self._stage_startable(runtime, stage):
                        self._start_stage(runtime=runtime, now_ns=now_ns)
                        made_progress = True
                    else:
                        self._record_current_blockers(runtime, stage)

            if completed_count == len(runtimes):
                break

            next_times: list[float] = []
            if self._active_heap:
                next_times.append(float(self._active_heap[0][0]))

            future_ready = [
                runtime.stage_ready_ns
                for runtime in runtimes.values()
                if not runtime.active
                and not runtime.completed
                and runtime.stage_ready_ns > now_ns + FLOAT_TOLERANCE_NS
            ]
            if future_ready:
                next_times.append(float(min(future_ready)))

            if not next_times:
                blocked = [
                    runtime.spec.request_id
                    for runtime in runtimes.values()
                    if not runtime.completed
                ]
                raise RuntimeError(f"simulation deadlock: {blocked}")

            next_now = min(next_times)
            if next_now <= now_ns + FLOAT_TOLERANCE_NS:
                raise RuntimeError("event loop failed to advance")
            now_ns = next_now

        request_rows: list[dict[str, object]] = []
        stage_rows: list[dict[str, object]] = []

        nominal_service_ns = float(sum(stage.duration_ns for stage in self.pipeline))
        for runtime in runtimes.values():
            total_wait_ns = float(sum(record.wait_ns for record in runtime.stage_records))
            request_rows.append(
                asdict(
                    RequestRecord(
                        run_kind=self.run_kind,
                        scenario_name=self.scenario.scenario_name,
                        workload_name=self.workload_name,
                        trial_id=self.trial_id,
                        request_id=runtime.spec.request_id,
                        tenant=runtime.spec.tenant,
                        request_index=runtime.spec.request_index,
                        ready_ns=float(runtime.spec.ready_ns),
                        completion_ns=float(runtime.completion_ns),
                        turnaround_ns=float(runtime.completion_ns - runtime.spec.ready_ns),
                        total_wait_ns=total_wait_ns,
                        total_service_ns=nominal_service_ns,
                        completed=runtime.completed,
                    )
                )
            )
            stage_rows.extend(asdict(record) for record in runtime.stage_records)

        interval_rows = [asdict(interval) for interval in self.resource_intervals]
        return (
            pd.DataFrame(request_rows),
            pd.DataFrame(stage_rows),
            pd.DataFrame(interval_rows),
        )


# =============================================================================
# Analysis helpers
# =============================================================================


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return math.nan
    return float(numerator / denominator)


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def run_one_controlled_trial(
    *,
    scenario: EndpointScenario,
    workload: VictimWorkload,
    trial_id: int,
    victim_phase_ns: float,
    pipeline: tuple[StageDefinition, ...],
) -> dict[str, pd.DataFrame]:
    attacker_specs = generate_attacker_specs(
        trial_id=trial_id,
        workload_name=workload.workload_name,
    )
    victim_specs = generate_victim_specs(
        workload=workload,
        trial_id=trial_id,
        phase_ns=victim_phase_ns,
    )

    outputs: dict[str, pd.DataFrame] = {}
    for run_kind, specs in (
        ("attacker_only", attacker_specs),
        ("victim_only", victim_specs),
        ("combined", victim_specs + attacker_specs),
    ):
        simulator = EndpointPipelineSimulator(
            scenario=scenario,
            pipeline=pipeline,
            run_kind=run_kind,
            workload_name=workload.workload_name,
            trial_id=trial_id,
        )
        request_df, stage_df, interval_df = simulator.run(specs)
        outputs[f"{run_kind}_requests"] = request_df
        outputs[f"{run_kind}_stages"] = stage_df
        outputs[f"{run_kind}_intervals"] = interval_df

    return outputs


def create_attacker_comparison(
    *,
    attacker_only_requests: pd.DataFrame,
    attacker_only_stages: pd.DataFrame,
    combined_requests: pd.DataFrame,
    combined_stages: pd.DataFrame,
    scenario: EndpointScenario,
    workload: VictimWorkload,
    trial_id: int,
    victim_phase_ns: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = attacker_only_requests.query("tenant == 'attacker'").copy()
    combined = combined_requests.query("tenant == 'attacker'").copy()

    request_merge = base.merge(
        combined,
        on=["request_index", "tenant"],
        suffixes=("_baseline", "_combined"),
        validate="one_to_one",
    )

    comparison = pd.DataFrame(
        {
            "scenario_name": scenario.scenario_name,
            "workload_name": workload.workload_name,
            "trial_id": trial_id,
            "victim_phase_ns": victim_phase_ns,
            "probe_index": request_merge["request_index"].astype(int),
            "attacker_release_ns": request_merge["ready_ns_baseline"].astype(float),
            "baseline_completion_ns": request_merge["completion_ns_baseline"].astype(float),
            "combined_completion_ns": request_merge["completion_ns_combined"].astype(float),
            "baseline_turnaround_ns": request_merge["turnaround_ns_baseline"].astype(float),
            "combined_turnaround_ns": request_merge["turnaround_ns_combined"].astype(float),
            "baseline_total_wait_ns": request_merge["total_wait_ns_baseline"].astype(float),
            "combined_total_wait_ns": request_merge["total_wait_ns_combined"].astype(float),
        }
    )
    comparison["excess_turnaround_ns"] = (
        comparison["combined_turnaround_ns"]
        - comparison["baseline_turnaround_ns"]
    )
    comparison["affected"] = (
        comparison["excess_turnaround_ns"] > AFFECTED_THRESHOLD_NS
    )

    stage_keys = ["request_index", "stage_name", "causal_component"]
    base_stage = (
        attacker_only_stages.query("tenant == 'attacker'")[stage_keys + ["wait_ns"]]
        .rename(columns={"wait_ns": "baseline_stage_wait_ns"})
    )
    combined_stage = (
        combined_stages.query("tenant == 'attacker'")[
            stage_keys
            + [
                "wait_ns",
                "blocking_physical_resources",
                "blocking_components",
                "blocking_owner_tenants",
                "cross_tenant_blocked",
                "self_blocked",
            ]
        ]
        .rename(
            columns={
                "wait_ns": "combined_stage_wait_ns",
                "blocking_physical_resources": "combined_blocking_physical_resources",
                "blocking_components": "combined_blocking_components",
                "blocking_owner_tenants": "combined_blocking_owner_tenants",
                "cross_tenant_blocked": "combined_cross_tenant_blocked",
                "self_blocked": "combined_self_blocked",
            }
        )
    )
    stage_comparison = base_stage.merge(
        combined_stage,
        on=stage_keys,
        validate="one_to_one",
    )
    stage_comparison.insert(0, "scenario_name", scenario.scenario_name)
    stage_comparison.insert(1, "workload_name", workload.workload_name)
    stage_comparison.insert(2, "trial_id", trial_id)
    stage_comparison.insert(3, "victim_phase_ns", victim_phase_ns)
    stage_comparison["added_stage_wait_ns"] = (
        stage_comparison["combined_stage_wait_ns"]
        - stage_comparison["baseline_stage_wait_ns"]
    )
    stage_comparison["stage_affected"] = (
        stage_comparison["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS
    )

    added_wait_totals = (
        stage_comparison.groupby("request_index", as_index=False)["added_stage_wait_ns"]
        .sum()
        .rename(columns={"request_index": "probe_index"})
    )
    comparison = comparison.merge(
        added_wait_totals,
        on="probe_index",
        validate="one_to_one",
    )
    comparison["causal_accounting_error_ns"] = (
        comparison["excess_turnaround_ns"]
        - comparison["added_stage_wait_ns"]
    )

    # Evaluator-side immediate attribution.  These columns are intentionally
    # absent from the black-box attacker-observation output.
    positive_stage = stage_comparison.loc[
        (stage_comparison["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS)
        & stage_comparison["combined_cross_tenant_blocked"]
    ].copy()
    if positive_stage.empty:
        positive_stage = stage_comparison.loc[
            stage_comparison["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS
        ].copy()
    if not positive_stage.empty:
        dominant = (
            positive_stage.sort_values(
                ["request_index", "added_stage_wait_ns"],
                ascending=[True, False],
            )
            .drop_duplicates("request_index")
            [["request_index", "stage_name", "causal_component", "added_stage_wait_ns"]]
            .rename(
                columns={
                    "request_index": "probe_index",
                    "stage_name": "dominant_blocking_stage",
                    "causal_component": "dominant_blocking_component",
                    "added_stage_wait_ns": "dominant_added_wait_ns",
                }
            )
        )
        comparison = comparison.merge(
            dominant,
            on="probe_index",
            how="left",
            validate="one_to_one",
        )
    else:
        comparison["dominant_blocking_stage"] = "none"
        comparison["dominant_blocking_component"] = "none"
        comparison["dominant_added_wait_ns"] = 0.0

    comparison["dominant_blocking_stage"] = comparison[
        "dominant_blocking_stage"
    ].fillna("none")
    comparison["dominant_blocking_component"] = comparison[
        "dominant_blocking_component"
    ].fillna("none")
    comparison["dominant_added_wait_ns"] = comparison[
        "dominant_added_wait_ns"
    ].fillna(0.0)

    return comparison, stage_comparison


def create_trial_summary(
    *,
    comparison: pd.DataFrame,
    victim_only_requests: pd.DataFrame,
    combined_requests: pd.DataFrame,
    scenario: EndpointScenario,
    workload: VictimWorkload,
    trial_id: int,
    victim_phase_ns: float,
) -> dict[str, object]:
    victim_only = victim_only_requests.query("tenant == 'victim'").copy()
    victim_combined = combined_requests.query("tenant == 'victim'").copy()
    victim_merge = victim_only.merge(
        victim_combined,
        on=["request_index", "tenant"],
        suffixes=("_baseline", "_combined"),
        validate="one_to_one",
    )

    baseline_mean = float(victim_merge["turnaround_ns_baseline"].mean())
    combined_mean = float(victim_merge["turnaround_ns_combined"].mean())

    baseline_first = float(victim_merge["ready_ns_baseline"].min())
    baseline_last = float(victim_merge["completion_ns_baseline"].max())
    combined_first = float(victim_merge["ready_ns_combined"].min())
    combined_last = float(victim_merge["completion_ns_combined"].max())

    baseline_makespan = baseline_last - baseline_first
    combined_makespan = combined_last - combined_first

    return {
        "scenario_name": scenario.scenario_name,
        "same_module": scenario.same_module,
        "shared_components": ";".join(scenario.shared_components) or "none",
        "shared_component_count": len(scenario.shared_components),
        "expected_mechanism": scenario.expected_mechanism,
        "workload_name": workload.workload_name,
        "trial_id": trial_id,
        "victim_phase_ns": victim_phase_ns,
        "probe_count": int(len(comparison)),
        "affected_probe_count": int(comparison["affected"].sum()),
        "affected_probe_fraction": float(comparison["affected"].mean()),
        "mean_excess_turnaround_ns": float(comparison["excess_turnaround_ns"].mean()),
        "median_excess_turnaround_ns": float(comparison["excess_turnaround_ns"].median()),
        "p95_excess_turnaround_ns": percentile(comparison["excess_turnaround_ns"], 95),
        "max_excess_turnaround_ns": float(comparison["excess_turnaround_ns"].max()),
        "total_excess_turnaround_ns": float(comparison["excess_turnaround_ns"].sum()),
        "attacker_baseline_mean_wait_ns": float(comparison["baseline_total_wait_ns"].mean()),
        "attacker_combined_mean_wait_ns": float(comparison["combined_total_wait_ns"].mean()),
        "victim_request_count": int(len(victim_merge)),
        "victim_baseline_mean_latency_ns": baseline_mean,
        "victim_combined_mean_latency_ns": combined_mean,
        "victim_mean_latency_slowdown": safe_divide(combined_mean, baseline_mean),
        "victim_baseline_makespan_ns": baseline_makespan,
        "victim_combined_makespan_ns": combined_makespan,
        "victim_makespan_slowdown": safe_divide(combined_makespan, baseline_makespan),
        "max_causal_accounting_error_ns": float(
            comparison["causal_accounting_error_ns"].abs().max()
        ),
    }


def summarize_resource_utilization(
    intervals: pd.DataFrame,
    *,
    scenario: EndpointScenario,
    workload: VictimWorkload,
    trial_id: int,
) -> pd.DataFrame:
    combined = intervals.query("run_kind == 'combined'").copy()
    if combined.empty:
        return pd.DataFrame()

    horizon_start = 0.0
    horizon_end = float(combined["end_ns"].max())
    horizon = max(horizon_end - horizon_start, FLOAT_TOLERANCE_NS)

    grouped = (
        combined.groupby(
            [
                "physical_resource",
                "logical_component",
                "shared_between_tenants",
            ],
            as_index=False,
        )
        .agg(
            busy_time_ns=("occupancy_ns", "sum"),
            interval_count=("request_id", "count"),
            distinct_tenants=("tenant", "nunique"),
            attacker_busy_time_ns=(
                "occupancy_ns",
                lambda values: float(
                    values[
                        combined.loc[values.index, "tenant"].eq("attacker")
                    ].sum()
                ),
            ),
            victim_busy_time_ns=(
                "occupancy_ns",
                lambda values: float(
                    values[
                        combined.loc[values.index, "tenant"].eq("victim")
                    ].sum()
                ),
            ),
        )
    )
    grouped.insert(0, "scenario_name", scenario.scenario_name)
    grouped.insert(1, "workload_name", workload.workload_name)
    grouped.insert(2, "trial_id", trial_id)
    grouped["observation_horizon_ns"] = horizon
    grouped["utilization"] = grouped["busy_time_ns"] / horizon
    return grouped


def create_reuse_rows(
    intervals: pd.DataFrame,
    *,
    scenario: EndpointScenario,
    workload: VictimWorkload,
    trial_id: int,
) -> pd.DataFrame:
    combined = intervals.query("run_kind == 'combined'").copy()
    combined = combined.loc[
        combined["logical_component"].isin(ENDPOINT_COMPONENTS)
    ]
    rows: list[dict[str, object]] = []

    for physical_resource, group in combined.groupby("physical_resource"):
        ordered = group.sort_values(["start_ns", "end_ns", "request_id"])
        records = list(ordered.to_dict("records"))
        for reuse_index in range(1, len(records)):
            previous = records[reuse_index - 1]
            current = records[reuse_index]
            rows.append(
                {
                    "scenario_name": scenario.scenario_name,
                    "workload_name": workload.workload_name,
                    "trial_id": trial_id,
                    "physical_resource": physical_resource,
                    "logical_component": current["logical_component"],
                    "shared_between_tenants": current[
                        "shared_between_tenants"
                    ],
                    "reuse_index": reuse_index,
                    "previous_tenant": previous["tenant"],
                    "current_tenant": current["tenant"],
                    "cross_tenant_reuse": previous["tenant"] != current["tenant"],
                    "previous_request_id": previous["request_id"],
                    "current_request_id": current["request_id"],
                    "previous_acquire_ns": previous["start_ns"],
                    "previous_release_ns": previous["end_ns"],
                    "current_acquire_ns": current["start_ns"],
                    "acquisition_to_acquisition_ns": (
                        current["start_ns"] - previous["start_ns"]
                    ),
                    "release_to_reacquire_gap_ns": (
                        current["start_ns"] - previous["end_ns"]
                    ),
                    "previous_hold_ns": previous["occupancy_ns"],
                }
            )

    return pd.DataFrame(rows)


def build_blackbox_observations(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return only timing fields available to the black-box attacker."""

    columns = [
        "scenario_name",       # evaluator grouping only
        "workload_name",       # evaluator grouping only
        "trial_id",            # evaluator grouping only
        "probe_index",
        "attacker_release_ns",
        "baseline_completion_ns",
        "combined_completion_ns",
        "baseline_turnaround_ns",
        "combined_turnaround_ns",
        "excess_turnaround_ns",
        "affected",
    ]
    return comparison[columns].copy()


# =============================================================================
# Validation
# =============================================================================


def add_assertion(
    assertions: list[ValidationAssertion],
    *,
    group: str,
    name: str,
    passed: bool,
    expected: object,
    observed: object,
    details: str = "",
) -> None:
    assertions.append(
        ValidationAssertion(
            validation_group=group,
            assertion_name=name,
            passed=bool(passed),
            expected=str(expected),
            observed=str(observed),
            details=details,
        )
    )


def validate_results(
    *,
    trial_summary: pd.DataFrame,
    attacker_comparison: pd.DataFrame,
    stage_comparison: pd.DataFrame,
    request_records: pd.DataFrame,
    resource_intervals: pd.DataFrame,
    blackbox_observations: pd.DataFrame,
) -> pd.DataFrame:
    assertions: list[ValidationAssertion] = []

    # Baseline probe schedule must remain low-contention.
    baseline_max_wait = float(
        request_records.query(
            "run_kind == 'attacker_only' and tenant == 'attacker'"
        )["total_wait_ns"].max()
    )
    add_assertion(
        assertions,
        group="attacker_baseline",
        name="low_self_contention",
        passed=baseline_max_wait <= FLOAT_TOLERANCE_NS,
        expected="maximum attacker-only wait = 0 ns",
        observed=f"{baseline_max_wait:.12g} ns",
    )

    # Every external excess must equal the sum of evaluator-attributed stage waits.
    max_accounting_error = float(
        attacker_comparison["causal_accounting_error_ns"].abs().max()
    )
    add_assertion(
        assertions,
        group="causal_accounting",
        name="external_delay_equals_added_stage_wait",
        passed=max_accounting_error <= FLOAT_TOLERANCE_NS,
        expected="maximum absolute accounting error <= 1e-9 ns",
        observed=f"{max_accounting_error:.12g} ns",
    )

    # Module co-location without shared endpoint components must not leak.
    for scenario_name in (
        "separate_modules_control",
        "shared_module_fully_independent",
    ):
        subset = trial_summary.loc[
            trial_summary["scenario_name"].eq(scenario_name)
        ]
        maximum = float(subset["max_excess_turnaround_ns"].max())
        add_assertion(
            assertions,
            group="isolation_controls",
            name=f"{scenario_name}_zero_leakage",
            passed=maximum <= FLOAT_TOLERANCE_NS,
            expected="maximum excess turnaround = 0 ns",
            observed=f"{maximum:.12g} ns",
        )

    # One-resource-at-a-time cases must attribute added waits only to that resource.
    expected_component = {
        "shared_local_route_only": "local_route",
        "shared_communication_qubit_only": "communication_qubit",
        "shared_interconnect_port_only": "interconnect_port",
        "shared_reset_engine_only": "reset_engine",
    }
    for scenario_name, component in expected_component.items():
        positive = stage_comparison.loc[
            stage_comparison["scenario_name"].eq(scenario_name)
            & (stage_comparison["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS)
            & stage_comparison["combined_cross_tenant_blocked"]
        ]
        observed_components = sorted(positive["causal_component"].unique().tolist())
        add_assertion(
            assertions,
            group="single_resource_attribution",
            name=f"{scenario_name}_only_{component}",
            passed=bool(observed_components) and observed_components == [component],
            expected=f"positive added wait only at {component}",
            observed=observed_components,
        )

    # The shared active endpoint without shared reset must not attribute direct
    # wait to the reset engine.
    no_reset_positive = stage_comparison.loc[
        stage_comparison["scenario_name"].eq("shared_endpoint_without_reset")
        & (stage_comparison["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS)
        & stage_comparison["combined_cross_tenant_blocked"]
    ]
    no_reset_components = sorted(
        no_reset_positive["causal_component"].unique().tolist()
    )
    add_assertion(
        assertions,
        group="combined_endpoint_controls",
        name="shared_endpoint_without_reset_has_no_reset_wait",
        passed="reset_engine" not in no_reset_components,
        expected="reset_engine absent from positive added-wait components",
        observed=no_reset_components,
    )

    # Network resources are duplicated and therefore must never directly block
    # the attacker in this endpoint-local experiment.
    network_positive = stage_comparison.loc[
        stage_comparison["causal_component"].isin(NETWORK_COMPONENTS)
        & (stage_comparison["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS)
        & stage_comparison["combined_cross_tenant_blocked"]
    ]
    add_assertion(
        assertions,
        group="endpoint_locality",
        name="no_network_resource_leakage",
        passed=network_positive.empty,
        expected="zero positive added wait on network-only resources",
        observed=f"{len(network_positive)} positive rows",
    )

    # No resource can have overlapping half-open intervals.
    overlap_count = 0
    for _, group in resource_intervals.groupby(
        ["run_kind", "scenario_name", "workload_name", "trial_id", "physical_resource"]
    ):
        ordered = group.sort_values(["start_ns", "end_ns"])
        previous_end = -math.inf
        for row in ordered.itertuples(index=False):
            if row.start_ns < previous_end - FLOAT_TOLERANCE_NS:
                overlap_count += 1
            previous_end = max(previous_end, row.end_ns)
    add_assertion(
        assertions,
        group="resource_calendar",
        name="no_illegal_resource_overlap",
        passed=overlap_count == 0,
        expected=0,
        observed=overlap_count,
    )

    # Every request must complete and preserve fixed service time.
    incomplete_count = int((~request_records["completed"]).sum())
    expected_service = float(sum(STAGE_DURATIONS_NS.values()))
    service_error = float(
        (request_records["total_service_ns"] - expected_service).abs().max()
    )
    add_assertion(
        assertions,
        group="request_completion",
        name="all_requests_complete",
        passed=incomplete_count == 0,
        expected=0,
        observed=incomplete_count,
    )
    add_assertion(
        assertions,
        group="request_completion",
        name="fixed_pipeline_service_time",
        passed=service_error <= FLOAT_TOLERANCE_NS,
        expected=f"{expected_service} ns",
        observed=f"max error {service_error:.12g} ns",
    )

    # Reset must be the final stage and communication qubits must remain held
    # until reset completion.
    comm_intervals = resource_intervals.loc[
        resource_intervals["logical_component"].eq("communication_qubit")
    ]
    invalid_comm_release = int(
        (~comm_intervals["released_stage"].eq("communication_reset")).sum()
    )
    add_assertion(
        assertions,
        group="resource_lifetimes",
        name="communication_qubit_released_after_reset",
        passed=invalid_comm_release == 0,
        expected="all communication-qubit intervals released at communication_reset",
        observed=f"{invalid_comm_release} invalid intervals",
    )

    # Interconnect ports and switch paths must be released at transfer completion,
    # earlier than communication-qubit release for the same request.
    lifetime_pivot = resource_intervals.pivot_table(
        index=[
            "run_kind",
            "scenario_name",
            "workload_name",
            "trial_id",
            "request_id",
            "tenant",
        ],
        columns="logical_component",
        values="end_ns",
        aggfunc="max",
    )
    comparable = lifetime_pivot.dropna(
        subset=["communication_qubit", "interconnect_port", "switch_path"]
    )
    bad_release_order = int(
        (
            (comparable["interconnect_port"] > comparable["communication_qubit"] + FLOAT_TOLERANCE_NS)
            | (comparable["switch_path"] > comparable["communication_qubit"] + FLOAT_TOLERANCE_NS)
        ).sum()
    )
    add_assertion(
        assertions,
        group="resource_lifetimes",
        name="port_and_switch_release_before_comm_qubit",
        passed=bad_release_order == 0,
        expected=0,
        observed=bad_release_order,
    )

    # Black-box observation table must not expose internal mechanism labels.
    forbidden_tokens = (
        "resource",
        "component",
        "stage",
        "shared_components",
        "blocking",
        "utilization",
        "reuse",
    )
    forbidden_columns = [
        column
        for column in blackbox_observations.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    add_assertion(
        assertions,
        group="threat_model",
        name="blackbox_output_excludes_internal_labels",
        passed=not forbidden_columns,
        expected="no internal stage/resource columns",
        observed=forbidden_columns,
    )

    # The reset-only channel should be weaker on average than the full endpoint.
    scenario_means = trial_summary.groupby("scenario_name")[
        "mean_excess_turnaround_ns"
    ].mean()
    reset_mean = float(scenario_means.get("shared_reset_engine_only", math.nan))
    full_mean = float(scenario_means.get("shared_full_endpoint_pipeline", math.nan))
    add_assertion(
        assertions,
        group="expected_progression",
        name="reset_only_weaker_than_full_endpoint",
        passed=np.isfinite(reset_mean) and np.isfinite(full_mean) and reset_mean < full_mean,
        expected="mean reset-only excess < mean full-pipeline excess",
        observed=f"reset={reset_mean:.6g}, full={full_mean:.6g}",
        details=(
            "This is an architectural progression check rather than a causal "
            "correctness invariant."
        ),
    )

    return pd.DataFrame(asdict(assertion) for assertion in assertions)


# =============================================================================
# Summaries and plots
# =============================================================================


def aggregate_outputs(
    *,
    trial_summary: pd.DataFrame,
    stage_comparison: pd.DataFrame,
    attacker_comparison: pd.DataFrame,
    utilization: pd.DataFrame,
    reuse: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scenario_summary = (
        trial_summary.groupby(
            [
                "scenario_name",
                "same_module",
                "shared_components",
                "shared_component_count",
                "expected_mechanism",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
            std_affected_probe_fraction=("affected_probe_fraction", "std"),
            mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
            median_excess_turnaround_ns=("median_excess_turnaround_ns", "mean"),
            mean_p95_excess_turnaround_ns=("p95_excess_turnaround_ns", "mean"),
            max_excess_turnaround_ns=("max_excess_turnaround_ns", "max"),
            mean_total_excess_turnaround_ns=("total_excess_turnaround_ns", "mean"),
            attacker_baseline_mean_wait_ns=("attacker_baseline_mean_wait_ns", "mean"),
            mean_victim_latency_slowdown=("victim_mean_latency_slowdown", "mean"),
            max_victim_latency_slowdown=("victim_mean_latency_slowdown", "max"),
            mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
            max_causal_accounting_error_ns=("max_causal_accounting_error_ns", "max"),
        )
    )

    workload_summary = (
        trial_summary.groupby(
            ["scenario_name", "workload_name"],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
            mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
            mean_p95_excess_turnaround_ns=("p95_excess_turnaround_ns", "mean"),
            max_excess_turnaround_ns=("max_excess_turnaround_ns", "max"),
            mean_victim_latency_slowdown=("victim_mean_latency_slowdown", "mean"),
            mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
        )
    )

    delay_decomposition = (
        stage_comparison.groupby(
            ["scenario_name", "workload_name", "stage_name", "causal_component"],
            as_index=False,
        )
        .agg(
            probe_stage_count=("request_index", "count"),
            affected_probe_stage_count=("stage_affected", "sum"),
            affected_probe_stage_fraction=("stage_affected", "mean"),
            mean_added_stage_wait_ns=("added_stage_wait_ns", "mean"),
            mean_positive_added_stage_wait_ns=(
                "added_stage_wait_ns",
                lambda values: float(values[values > AFFECTED_THRESHOLD_NS].mean())
                if (values > AFFECTED_THRESHOLD_NS).any()
                else 0.0,
            ),
            max_added_stage_wait_ns=("added_stage_wait_ns", "max"),
            cross_tenant_blocked_count=("combined_cross_tenant_blocked", "sum"),
            self_blocked_count=("combined_self_blocked", "sum"),
        )
    )

    positive = attacker_comparison.loc[attacker_comparison["affected"]].copy()
    if positive.empty:
        blocking_attribution = pd.DataFrame(
            columns=[
                "scenario_name",
                "workload_name",
                "dominant_blocking_component",
                "affected_probe_count",
                "fraction_of_affected_probes",
                "mean_dominant_added_wait_ns",
                "max_dominant_added_wait_ns",
            ]
        )
    else:
        blocking_attribution = (
            positive.groupby(
                [
                    "scenario_name",
                    "workload_name",
                    "dominant_blocking_component",
                ],
                as_index=False,
            )
            .agg(
                affected_probe_count=("probe_index", "count"),
                mean_dominant_added_wait_ns=("dominant_added_wait_ns", "mean"),
                max_dominant_added_wait_ns=("dominant_added_wait_ns", "max"),
            )
        )
        totals = (
            positive.groupby(["scenario_name", "workload_name"], as_index=False)
            .size()
            .rename(columns={"size": "total_affected_probes"})
        )
        blocking_attribution = blocking_attribution.merge(
            totals,
            on=["scenario_name", "workload_name"],
            validate="many_to_one",
        )
        blocking_attribution["fraction_of_affected_probes"] = (
            blocking_attribution["affected_probe_count"]
            / blocking_attribution["total_affected_probes"]
        )

    if utilization.empty:
        utilization_summary = utilization.copy()
    else:
        utilization_summary = (
            utilization.groupby(
                [
                    "scenario_name",
                    "logical_component",
                    "shared_between_tenants",
                ],
                as_index=False,
            )
            .agg(
                trial_resource_count=("physical_resource", "count"),
                mean_utilization=("utilization", "mean"),
                max_utilization=("utilization", "max"),
                mean_busy_time_ns=("busy_time_ns", "mean"),
                mean_attacker_busy_time_ns=("attacker_busy_time_ns", "mean"),
                mean_victim_busy_time_ns=("victim_busy_time_ns", "mean"),
                mean_interval_count=("interval_count", "mean"),
            )
        )

    if reuse.empty:
        reuse_summary = reuse.copy()
    else:
        reuse_summary = (
            reuse.groupby(
                [
                    "scenario_name",
                    "logical_component",
                    "shared_between_tenants",
                    "cross_tenant_reuse",
                ],
                as_index=False,
            )
            .agg(
                reuse_event_count=("reuse_index", "count"),
                mean_acquisition_to_acquisition_ns=(
                    "acquisition_to_acquisition_ns",
                    "mean",
                ),
                median_acquisition_to_acquisition_ns=(
                    "acquisition_to_acquisition_ns",
                    "median",
                ),
                mean_release_to_reacquire_gap_ns=(
                    "release_to_reacquire_gap_ns",
                    "mean",
                ),
                minimum_release_to_reacquire_gap_ns=(
                    "release_to_reacquire_gap_ns",
                    "min",
                ),
                mean_previous_hold_ns=("previous_hold_ns", "mean"),
            )
        )

    blackbox_trace_summary = (
        attacker_comparison.groupby(
            ["scenario_name", "workload_name", "trial_id"],
            as_index=False,
        )
        .agg(
            probe_count=("probe_index", "count"),
            affected_probe_count=("affected", "sum"),
            affected_probe_fraction=("affected", "mean"),
            mean_excess_turnaround_ns=("excess_turnaround_ns", "mean"),
            median_excess_turnaround_ns=("excess_turnaround_ns", "median"),
            p95_excess_turnaround_ns=(
                "excess_turnaround_ns",
                lambda values: float(np.percentile(values, 95)),
            ),
            max_excess_turnaround_ns=("excess_turnaround_ns", "max"),
            total_excess_turnaround_ns=("excess_turnaround_ns", "sum"),
        )
    )

    return {
        "scenario_summary": scenario_summary,
        "workload_summary": workload_summary,
        "delay_decomposition_summary": delay_decomposition,
        "blocking_attribution_summary": blocking_attribution,
        "resource_utilization_summary": utilization_summary,
        "endpoint_reuse_summary": reuse_summary,
        "blackbox_trace_summary": blackbox_trace_summary,
    }


def save_plots(
    *,
    scenario_summary: pd.DataFrame,
    delay_decomposition: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = scenario_summary.sort_values("mean_excess_turnaround_ns")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(ordered["scenario_name"], ordered["mean_excess_turnaround_ns"])
    ax.set_xlabel("Mean attacker excess turnaround (ns)")
    ax.set_ylabel("Endpoint-sharing scenario")
    ax.set_title("Phase 2.2: Timing leakage by endpoint component")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_02_mean_excess_by_scenario.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(ordered["scenario_name"], ordered["mean_affected_probe_fraction"])
    ax.set_xlabel("Fraction of attacker probes affected")
    ax.set_ylabel("Endpoint-sharing scenario")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Phase 2.2: Probe exposure by endpoint component")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_02_affected_fraction_by_scenario.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(ordered["scenario_name"], ordered["mean_victim_latency_slowdown"])
    ax.axvline(1.0, linewidth=1.0)
    ax.set_xlabel("Mean victim latency slowdown")
    ax.set_ylabel("Endpoint-sharing scenario")
    ax.set_title("Phase 2.2: Victim disruption")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_02_victim_slowdown_by_scenario.png", dpi=200)
    plt.close(fig)

    decomposition = (
        delay_decomposition.groupby(
            ["scenario_name", "causal_component"],
            as_index=False,
        )["mean_added_stage_wait_ns"]
        .mean()
        .pivot(
            index="scenario_name",
            columns="causal_component",
            values="mean_added_stage_wait_ns",
        )
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    decomposition.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Endpoint-sharing scenario")
    ax.set_ylabel("Mean added stage wait (ns per probe)")
    ax.set_title("Phase 2.2: Evaluator-side delay decomposition")
    ax.legend(title="Causal component", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_02_wait_decomposition.png", dpi=200)
    plt.close(fig)


# =============================================================================
# Main experiment
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.2 endpoint-pipeline contention experiment"
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def configuration_table(
    scenarios: tuple[EndpointScenario, ...],
    workloads: tuple[VictimWorkload, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for workload in workloads:
            rows.append(
                {
                    "scenario_name": scenario.scenario_name,
                    "same_module": scenario.same_module,
                    "shared_components": ";".join(scenario.shared_components) or "none",
                    "shared_local_route": "local_route" in scenario.shared_components,
                    "shared_communication_qubit": (
                        "communication_qubit" in scenario.shared_components
                    ),
                    "shared_interconnect_port": (
                        "interconnect_port" in scenario.shared_components
                    ),
                    "shared_reset_engine": (
                        "reset_engine" in scenario.shared_components
                    ),
                    "expected_mechanism": scenario.expected_mechanism,
                    "scenario_description": scenario.description,
                    "workload_name": workload.workload_name,
                    "workload_description": workload.description,
                    "release_pattern": workload.release_pattern,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    output_dir: Path = args.output_dir
    raw_dir = output_dir / "raw"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pipeline = build_endpoint_pipeline()
    scenarios = build_scenarios()
    workloads = build_workloads()

    all_requests: list[pd.DataFrame] = []
    all_stages: list[pd.DataFrame] = []
    all_intervals: list[pd.DataFrame] = []
    all_comparisons: list[pd.DataFrame] = []
    all_stage_comparisons: list[pd.DataFrame] = []
    all_trial_summaries: list[dict[str, object]] = []
    all_utilization: list[pd.DataFrame] = []
    all_reuse: list[pd.DataFrame] = []
    all_blackbox: list[pd.DataFrame] = []
    phase_rows: list[dict[str, object]] = []

    total_configurations = len(scenarios) * len(workloads) * args.trials
    completed_configurations = 0

    print("Phase 2.2 — Endpoint Pipeline Contention")
    print("=" * 58)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Victim workloads: {len(workloads)}")
    print(f"Trials per scenario/workload: {args.trials}")
    print(f"Controlled trial tuples: {total_configurations}")
    print(f"Nominal remote-operation latency: {sum(STAGE_DURATIONS_NS.values()):.1f} ns")

    # Use the same phase schedule across scenarios so architectural controls are
    # paired rather than confounded by different victim timing.
    phase_schedule: dict[tuple[str, int], float] = {}
    for workload in workloads:
        for trial_id in range(args.trials):
            phase_schedule[(workload.workload_name, trial_id)] = float(
                rng.uniform(0.0, ATTACKER_PERIOD_NS)
            )

    control_scenario = EndpointScenario(
        scenario_name="paired_control_reference",
        same_module=False,
        shared_components=(),
        description=(
            "Reusable attacker-only and victim-only control with no "
            "cross-tenant resource sharing."
        ),
        expected_mechanism="no_cross_tenant_endpoint_contention",
    )

    for workload in workloads:
        for trial_id in range(args.trials):
            victim_phase_ns = phase_schedule[(workload.workload_name, trial_id)]
            phase_rows.append(
                {
                    "workload_name": workload.workload_name,
                    "trial_id": trial_id,
                    "victim_phase_ns": victim_phase_ns,
                }
            )

            attacker_specs = generate_attacker_specs(
                trial_id=trial_id,
                workload_name=workload.workload_name,
            )
            victim_specs = generate_victim_specs(
                workload=workload,
                trial_id=trial_id,
                phase_ns=victim_phase_ns,
            )

            # Attacker-only and victim-only controls are invariant across endpoint
            # sharing scenarios because only one tenant is present.  Run each
            # control once per workload/trial and reuse it for all paired cases.
            control_outputs: dict[str, pd.DataFrame] = {}
            for run_kind, specs in (
                ("attacker_only", attacker_specs),
                ("victim_only", victim_specs),
            ):
                control_simulator = EndpointPipelineSimulator(
                    scenario=control_scenario,
                    pipeline=pipeline,
                    run_kind=run_kind,
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                )
                control_requests, control_stages, control_intervals = (
                    control_simulator.run(specs)
                )
                control_outputs[f"{run_kind}_requests"] = control_requests
                control_outputs[f"{run_kind}_stages"] = control_stages
                control_outputs[f"{run_kind}_intervals"] = control_intervals
                all_requests.append(control_requests)
                all_stages.append(control_stages)
                all_intervals.append(control_intervals)

            for scenario in scenarios:
                combined_simulator = EndpointPipelineSimulator(
                    scenario=scenario,
                    pipeline=pipeline,
                    run_kind="combined",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                )
                combined_requests, combined_stages, combined_intervals = (
                    combined_simulator.run(victim_specs + attacker_specs)
                )

                comparison, stage_comparison = create_attacker_comparison(
                    attacker_only_requests=control_outputs["attacker_only_requests"],
                    attacker_only_stages=control_outputs["attacker_only_stages"],
                    combined_requests=combined_requests,
                    combined_stages=combined_stages,
                    scenario=scenario,
                    workload=workload,
                    trial_id=trial_id,
                    victim_phase_ns=victim_phase_ns,
                )

                trial_summary = create_trial_summary(
                    comparison=comparison,
                    victim_only_requests=control_outputs["victim_only_requests"],
                    combined_requests=combined_requests,
                    scenario=scenario,
                    workload=workload,
                    trial_id=trial_id,
                    victim_phase_ns=victim_phase_ns,
                )

                utilization = summarize_resource_utilization(
                    combined_intervals,
                    scenario=scenario,
                    workload=workload,
                    trial_id=trial_id,
                )
                reuse = create_reuse_rows(
                    combined_intervals,
                    scenario=scenario,
                    workload=workload,
                    trial_id=trial_id,
                )
                blackbox = build_blackbox_observations(comparison)

                all_requests.append(combined_requests)
                all_stages.append(combined_stages)
                all_intervals.append(combined_intervals)
                all_comparisons.append(comparison)
                all_stage_comparisons.append(stage_comparison)
                all_trial_summaries.append(trial_summary)
                if not utilization.empty:
                    all_utilization.append(utilization)
                if not reuse.empty:
                    all_reuse.append(reuse)
                all_blackbox.append(blackbox)

                completed_configurations += 1
                if completed_configurations % max(1, total_configurations // 10) == 0:
                    print(
                        f"  completed {completed_configurations}/{total_configurations} "
                        "trial tuples"
                    )

    request_records = pd.concat(all_requests, ignore_index=True)
    stage_records = pd.concat(all_stages, ignore_index=True)
    resource_intervals = pd.concat(all_intervals, ignore_index=True)
    attacker_comparison = pd.concat(all_comparisons, ignore_index=True)
    stage_comparison = pd.concat(all_stage_comparisons, ignore_index=True)
    trial_summary = pd.DataFrame(all_trial_summaries)
    utilization = (
        pd.concat(all_utilization, ignore_index=True)
        if all_utilization
        else pd.DataFrame()
    )
    reuse = (
        pd.concat(all_reuse, ignore_index=True)
        if all_reuse
        else pd.DataFrame()
    )
    blackbox_observations = pd.concat(all_blackbox, ignore_index=True)

    summaries = aggregate_outputs(
        trial_summary=trial_summary,
        stage_comparison=stage_comparison,
        attacker_comparison=attacker_comparison,
        utilization=utilization,
        reuse=reuse,
    )

    validations = validate_results(
        trial_summary=trial_summary,
        attacker_comparison=attacker_comparison,
        stage_comparison=stage_comparison,
        request_records=request_records,
        resource_intervals=resource_intervals,
        blackbox_observations=blackbox_observations,
    )
    validation_summary = (
        validations.groupby("validation_group", as_index=False)
        .agg(
            assertion_count=("passed", "count"),
            passed_count=("passed", "sum"),
        )
    )
    validation_summary["failed_count"] = (
        validation_summary["assertion_count"]
        - validation_summary["passed_count"]
    )
    validation_summary["pass_rate"] = (
        validation_summary["passed_count"]
        / validation_summary["assertion_count"]
    )

    # Configuration and compact summaries remain uncompressed for inspection.
    configuration_table(scenarios, workloads).to_csv(
        output_dir / "phase2_02_configuration_table.csv",
        index=False,
    )
    pd.DataFrame(phase_rows).to_csv(
        output_dir / "phase2_02_trial_phase_schedule.csv",
        index=False,
    )
    trial_summary.to_csv(
        output_dir / "phase2_02_trial_summary.csv",
        index=False,
    )
    summaries["scenario_summary"].to_csv(
        output_dir / "phase2_02_scenario_summary.csv",
        index=False,
    )
    summaries["workload_summary"].to_csv(
        output_dir / "phase2_02_workload_summary.csv",
        index=False,
    )
    summaries["delay_decomposition_summary"].to_csv(
        output_dir / "phase2_02_delay_decomposition_summary.csv",
        index=False,
    )
    summaries["blocking_attribution_summary"].to_csv(
        output_dir / "phase2_02_blocking_attribution_summary.csv",
        index=False,
    )
    summaries["resource_utilization_summary"].to_csv(
        output_dir / "phase2_02_resource_utilization_summary.csv",
        index=False,
    )
    summaries["endpoint_reuse_summary"].to_csv(
        output_dir / "phase2_02_endpoint_reuse_summary.csv",
        index=False,
    )
    summaries["blackbox_trace_summary"].to_csv(
        output_dir / "phase2_02_blackbox_trace_summary.csv",
        index=False,
    )
    validations.to_csv(
        output_dir / "phase2_02_validation_assertions.csv",
        index=False,
    )
    validation_summary.to_csv(
        output_dir / "phase2_02_validation_summary.csv",
        index=False,
    )

    # Large evaluator/raw outputs are compressed.
    request_records.to_csv(
        raw_dir / "phase2_02_request_records.csv.gz",
        index=False,
        compression="gzip",
    )
    stage_records.to_csv(
        raw_dir / "phase2_02_stage_timeline.csv.gz",
        index=False,
        compression="gzip",
    )
    resource_intervals.to_csv(
        raw_dir / "phase2_02_resource_intervals.csv.gz",
        index=False,
        compression="gzip",
    )
    attacker_comparison.to_csv(
        raw_dir / "phase2_02_attacker_evaluator_comparison.csv.gz",
        index=False,
        compression="gzip",
    )
    stage_comparison.to_csv(
        raw_dir / "phase2_02_stage_wait_comparison.csv.gz",
        index=False,
        compression="gzip",
    )
    blackbox_observations.to_csv(
        raw_dir / "phase2_02_blackbox_attacker_observations.csv.gz",
        index=False,
        compression="gzip",
    )
    if not utilization.empty:
        utilization.to_csv(
            raw_dir / "phase2_02_resource_utilization_trials.csv.gz",
            index=False,
            compression="gzip",
        )
    if not reuse.empty:
        reuse.to_csv(
            raw_dir / "phase2_02_endpoint_reuse_events.csv.gz",
            index=False,
            compression="gzip",
        )

    if not args.no_plots:
        save_plots(
            scenario_summary=summaries["scenario_summary"],
            delay_decomposition=summaries["delay_decomposition_summary"],
            output_dir=plot_dir,
        )

    manifest = {
        "experiment": "Phase 2.2 — Endpoint Pipeline Contention",
        "output_directory": str(output_dir),
        "seed": args.seed,
        "trials_per_scenario_workload": args.trials,
        "scenario_count": len(scenarios),
        "workload_count": len(workloads),
        "controlled_trial_tuple_count": total_configurations,
        "attacker_period_ns": ATTACKER_PERIOD_NS,
        "observation_window_ns": OBSERVATION_WINDOW_NS,
        "nominal_pipeline_latency_ns": float(sum(STAGE_DURATIONS_NS.values())),
        "stage_durations_ns": STAGE_DURATIONS_NS,
        "request_record_count": int(len(request_records)),
        "stage_record_count": int(len(stage_records)),
        "resource_interval_count": int(len(resource_intervals)),
        "attacker_probe_comparison_count": int(len(attacker_comparison)),
        "validation_assertion_count": int(len(validations)),
        "passed_assertions": int(validations["passed"].sum()),
        "failed_assertions": int((~validations["passed"]).sum()),
        "all_validations_passed": bool(validations["passed"].all()),
        "blackbox_attacker_fields": list(blackbox_observations.columns),
        "evaluator_only_outputs": [
            "phase2_02_stage_timeline.csv.gz",
            "phase2_02_resource_intervals.csv.gz",
            "phase2_02_attacker_evaluator_comparison.csv.gz",
            "phase2_02_stage_wait_comparison.csv.gz",
        ],
    }
    with (output_dir / "phase2_02_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nValidation summary:")
    print(validation_summary.to_string(index=False))

    display_columns = [
        "scenario_name",
        "mean_affected_probe_fraction",
        "mean_excess_turnaround_ns",
        "mean_victim_latency_slowdown",
    ]
    print("\nEndpoint-component summary:")
    print(
        summaries["scenario_summary"][display_columns]
        .sort_values("mean_excess_turnaround_ns")
        .to_string(index=False)
    )

    print(f"\nResults saved to: {output_dir}")

    if not validations["passed"].all():
        failed = validations.loc[~validations["passed"]]
        print("\nFailed validations:", file=sys.stderr)
        print(failed.to_string(index=False), file=sys.stderr)
        if FAIL_ON_VALIDATION_ERROR:
            raise SystemExit(1)

    print("\nAll Phase 2.2 endpoint-pipeline validations passed.")


if __name__ == "__main__":
    main()
