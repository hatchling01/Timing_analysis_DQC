#!/usr/bin/env python3
"""
Phase 2.3 — Link/Switch Pipeline Contention
===========================================

This experiment moves outward from the module endpoint validated in Phase 2.1
and decomposed in Phase 2.2.  Every attacker and victim request uses a fully
independent endpoint pipeline.  Cross-tenant contention can occur only in an
explicit interconnect component:

    controller admission lane
    source module-to-switch link interface
    switch configuration/arbitration lane
    switch path
    shared transmission segment
    destination link interface

The experiment shares one component at a time, then evaluates combined data-path
and full-pipeline sharing.  Capacity has a concrete physical interpretation:

    shared_pool, capacity=1  -> one shared lane/path/channel
    shared_pool, capacity=2  -> two simultaneous shared lanes/paths/channels
    partitioned, capacity=2  -> one fixed lane per tenant

The black-box attacker output contains only its own release and completion timing.
Stage labels, resource identities, sharing modes, and blocker ownership are kept
in evaluator-only files.

Run
---
    python phase2_03_link_switch_pipeline_contention.py

Optional arguments
------------------
    --trials 10
    --seed 2303
    --no-plots
    --output-dir PATH

Default output directory
------------------------
blackbox_window_results/
└── phase2/
    └── phase2_03_link_switch_pipeline_contention/
        ├── raw/
        ├── plots/
        ├── phase2_03_configuration_table.csv
        ├── phase2_03_trial_phase_schedule.csv
        ├── phase2_03_trial_summary.csv
        ├── phase2_03_scenario_summary.csv
        ├── phase2_03_workload_summary.csv
        ├── phase2_03_delay_decomposition_summary.csv
        ├── phase2_03_blocking_attribution_summary.csv
        ├── phase2_03_resource_utilization_summary.csv
        ├── phase2_03_resource_capacity_summary.csv
        ├── phase2_03_interconnect_reuse_summary.csv
        ├── phase2_03_workload_fingerprint_metrics.csv
        ├── phase2_03_workload_fingerprint_predictions.csv
        ├── phase2_03_blackbox_trace_summary.csv
        ├── phase2_03_trace_key.csv
        ├── phase2_03_validation_assertions.csv
        ├── phase2_03_validation_summary.csv
        └── phase2_03_run_manifest.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Defaults and pipeline parameters
# =============================================================================

DEFAULT_OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase2"
    / "phase2_03_link_switch_pipeline_contention"
)
DEFAULT_TRIALS = 10
DEFAULT_SEED = 2303
OBSERVATION_WINDOW_NS = 20_000.0
ATTACKER_FIRST_RELEASE_NS = 30.0
ATTACKER_PERIOD_NS = 420.0
AFFECTED_THRESHOLD_NS = 1e-9
FLOAT_TOLERANCE_NS = 1e-9
FAIL_ON_VALIDATION_ERROR = True
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 1, "mtime": 1}

STAGE_DURATIONS_NS = {
    "local_route_acquisition": 10.0,
    "communication_qubit_load": 20.0,
    "endpoint_port_acquisition": 10.0,
    "controller_admission": 8.0,
    "source_link_interface_acquisition": 12.0,
    "switch_configuration_arbitration": 12.0,
    "switch_path_acquisition": 15.0,
    "intermodule_transmission": 80.0,
    "destination_link_delivery": 12.0,
    "receiver_side_gate": 30.0,
    "communication_reset": 40.0,
}

ENDPOINT_COMPONENTS = (
    "local_route",
    "communication_qubit",
    "endpoint_port",
    "receiver_engine",
    "reset_engine",
)

INTERCONNECT_COMPONENTS = (
    "controller_admission",
    "source_link_interface",
    "switch_arbiter",
    "switch_path",
    "transmission_segment",
    "destination_link_interface",
)

DATA_PATH_COMPONENTS = (
    "source_link_interface",
    "switch_path",
    "transmission_segment",
    "destination_link_interface",
)

STAGE_TO_COMPONENT = {
    "local_route_acquisition": "local_route",
    "communication_qubit_load": "communication_qubit",
    "endpoint_port_acquisition": "endpoint_port",
    "controller_admission": "controller_admission",
    "source_link_interface_acquisition": "source_link_interface",
    "switch_configuration_arbitration": "switch_arbiter",
    "switch_path_acquisition": "switch_path",
    "intermodule_transmission": "transmission_segment",
    "destination_link_delivery": "destination_link_interface",
    "receiver_side_gate": "receiver_engine",
    "communication_reset": "reset_engine",
}

BLACKBOX_ALLOWED_COLUMNS = {
    "trace_id",
    "probe_index",
    "release_ns",
    "attacker_only_completion_ns",
    "combined_completion_ns",
    "attacker_only_turnaround_ns",
    "combined_turnaround_ns",
    "excess_turnaround_ns",
    "affected",
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
class InterconnectScenario:
    scenario_name: str
    shared_components: tuple[str, ...]
    sharing_mode: str  # dedicated, shared_pool, partitioned
    component_capacity: int
    description: str
    expected_mechanism: str
    scenario_family: str


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
    sharing_mode: str
    component_capacity: int
    cross_tenant_capable: bool
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
    sharing_mode: str
    component_capacity: int
    cross_tenant_capable: bool
    owner_request_id: Optional[str] = None
    owner_tenant: Optional[str] = None
    acquired_ns: Optional[float] = None
    acquired_stage: Optional[str] = None
    interval_kind: Optional[str] = None

    @property
    def free(self) -> bool:
        return self.owner_request_id is None


# =============================================================================
# Pipeline, scenarios, and workloads
# =============================================================================


def build_pipeline() -> tuple[StageDefinition, ...]:
    """Remote-CX-like pipeline with independent endpoint and network stages."""

    return (
        StageDefinition(
            "local_route_acquisition",
            STAGE_DURATIONS_NS["local_route_acquisition"],
            acquire_held=("local_route",),
            description="Acquire the tenant-dedicated local endpoint route.",
        ),
        StageDefinition(
            "communication_qubit_load",
            STAGE_DURATIONS_NS["communication_qubit_load"],
            acquire_held=("communication_qubit",),
            require_held=("local_route",),
            release_held=("local_route",),
            description="Load the dedicated communication interface.",
        ),
        StageDefinition(
            "endpoint_port_acquisition",
            STAGE_DURATIONS_NS["endpoint_port_acquisition"],
            acquire_held=("endpoint_port",),
            require_held=("communication_qubit",),
            description="Acquire the dedicated endpoint-facing port.",
        ),
        StageDefinition(
            "controller_admission",
            STAGE_DURATIONS_NS["controller_admission"],
            acquire_scoped=("controller_admission",),
            require_held=("communication_qubit", "endpoint_port"),
            description="Obtain a controller-side admission/service lane.",
        ),
        StageDefinition(
            "source_link_interface_acquisition",
            STAGE_DURATIONS_NS["source_link_interface_acquisition"],
            acquire_held=("source_link_interface",),
            require_held=("communication_qubit", "endpoint_port"),
            description="Acquire a source module-to-switch link channel.",
        ),
        StageDefinition(
            "switch_configuration_arbitration",
            STAGE_DURATIONS_NS["switch_configuration_arbitration"],
            acquire_scoped=("switch_arbiter",),
            require_held=(
                "communication_qubit",
                "endpoint_port",
                "source_link_interface",
            ),
            description="Configure and arbitrate the switch fabric.",
        ),
        StageDefinition(
            "switch_path_acquisition",
            STAGE_DURATIONS_NS["switch_path_acquisition"],
            acquire_held=("switch_path",),
            require_held=(
                "communication_qubit",
                "endpoint_port",
                "source_link_interface",
            ),
            description="Acquire a concrete path through the switch fabric.",
        ),
        StageDefinition(
            "intermodule_transmission",
            STAGE_DURATIONS_NS["intermodule_transmission"],
            acquire_scoped=("transmission_segment",),
            require_held=(
                "communication_qubit",
                "endpoint_port",
                "source_link_interface",
                "switch_path",
            ),
            release_held=(
                "endpoint_port",
                "source_link_interface",
                "switch_path",
            ),
            description=(
                "Execute the inter-module transfer, then release the endpoint "
                "port, source link interface, and switch path."
            ),
        ),
        StageDefinition(
            "destination_link_delivery",
            STAGE_DURATIONS_NS["destination_link_delivery"],
            acquire_scoped=("destination_link_interface",),
            require_held=("communication_qubit",),
            description="Deliver through the destination-side link interface.",
        ),
        StageDefinition(
            "receiver_side_gate",
            STAGE_DURATIONS_NS["receiver_side_gate"],
            acquire_scoped=("receiver_engine",),
            require_held=("communication_qubit",),
            description="Execute the tenant-dedicated receiver-side operation.",
        ),
        StageDefinition(
            "communication_reset",
            STAGE_DURATIONS_NS["communication_reset"],
            acquire_scoped=("reset_engine",),
            require_held=("communication_qubit",),
            release_held=("communication_qubit",),
            description="Reset the dedicated communication qubit before reuse.",
        ),
    )


def _single_component_scenarios(component: str) -> list[InterconnectScenario]:
    pretty = component.replace("_", " ")
    return [
        InterconnectScenario(
            scenario_name=f"shared_{component}_capacity1",
            shared_components=(component,),
            sharing_mode="shared_pool",
            component_capacity=1,
            description=f"One cross-tenant shared {pretty} lane.",
            expected_mechanism=f"{component}_serialization",
            scenario_family=f"single_{component}",
        ),
        InterconnectScenario(
            scenario_name=f"shared_{component}_capacity2",
            shared_components=(component,),
            sharing_mode="shared_pool",
            component_capacity=2,
            description=f"Two simultaneous cross-tenant {pretty} lanes.",
            expected_mechanism=f"{component}_two_lane_pool",
            scenario_family=f"single_{component}",
        ),
        InterconnectScenario(
            scenario_name=f"partitioned_{component}_two_lanes",
            shared_components=(component,),
            sharing_mode="partitioned",
            component_capacity=2,
            description=f"Two {pretty} lanes, statically partitioned by tenant.",
            expected_mechanism=f"{component}_partitioned_isolation",
            scenario_family=f"single_{component}",
        ),
    ]


def build_scenarios() -> tuple[InterconnectScenario, ...]:
    scenarios: list[InterconnectScenario] = [
        InterconnectScenario(
            scenario_name="fully_isolated_interconnect_control",
            shared_components=(),
            sharing_mode="dedicated",
            component_capacity=1,
            description=(
                "Attacker and victim have independent endpoints, controller "
                "lanes, link interfaces, switch paths, and transmission paths."
            ),
            expected_mechanism="no_cross_tenant_interconnect_contention",
            scenario_family="isolated_control",
        )
    ]

    for component in INTERCONNECT_COMPONENTS:
        scenarios.extend(_single_component_scenarios(component))

    for family_name, components in (
        ("shared_data_path", DATA_PATH_COMPONENTS),
        ("shared_full_pipeline", INTERCONNECT_COMPONENTS),
    ):
        scenarios.extend(
            [
                InterconnectScenario(
                    scenario_name=f"{family_name}_capacity1",
                    shared_components=tuple(components),
                    sharing_mode="shared_pool",
                    component_capacity=1,
                    description=f"One shared lane for every component in {family_name}.",
                    expected_mechanism=f"{family_name}_serialization",
                    scenario_family=family_name,
                ),
                InterconnectScenario(
                    scenario_name=f"{family_name}_capacity2",
                    shared_components=tuple(components),
                    sharing_mode="shared_pool",
                    component_capacity=2,
                    description=f"Two pooled lanes for every component in {family_name}.",
                    expected_mechanism=f"{family_name}_two_lane_pool",
                    scenario_family=family_name,
                ),
                InterconnectScenario(
                    scenario_name=f"{family_name}_partitioned_two_lanes",
                    shared_components=tuple(components),
                    sharing_mode="partitioned",
                    component_capacity=2,
                    description=(
                        f"Two lanes for every component in {family_name}, with "
                        "one fixed lane per tenant."
                    ),
                    expected_mechanism=f"{family_name}_partitioned_isolation",
                    scenario_family=family_name,
                ),
            ]
        )

    return tuple(scenarios)


def build_workloads() -> tuple[VictimWorkload, ...]:
    return (
        VictimWorkload(
            "periodic_sparse",
            "One victim remote operation every 900 ns.",
            "period=900",
        ),
        VictimWorkload(
            "periodic_dense",
            "One victim remote operation every 460 ns.",
            "period=460",
        ),
        VictimWorkload(
            "layered_bursty",
            "Four-operation communication layers separated by local compute.",
            "burst_period=2400; offsets=0,120,240,360",
        ),
    )


# =============================================================================
# Request generation
# =============================================================================


def generate_attacker_specs(*, trial_id: int, workload_name: str) -> list[RequestSpec]:
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
    *, workload: VictimWorkload, trial_id: int, phase_ns: float
) -> list[RequestSpec]:
    releases: list[float] = []
    if workload.workload_name == "periodic_sparse":
        releases = list(np.arange(phase_ns, OBSERVATION_WINDOW_NS, 900.0))
    elif workload.workload_name == "periodic_dense":
        releases = list(np.arange(phase_ns, OBSERVATION_WINDOW_NS, 460.0))
    elif workload.workload_name == "layered_bursty":
        burst_start = phase_ns
        while burst_start < OBSERVATION_WINDOW_NS:
            for offset_ns in (0.0, 120.0, 240.0, 360.0):
                release_ns = burst_start + offset_ns
                if release_ns < OBSERVATION_WINDOW_NS:
                    releases.append(release_ns)
            burst_start += 2400.0
    else:
        raise ValueError(f"Unknown workload: {workload.workload_name}")

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


# =============================================================================
# Physical resource resolution
# =============================================================================


class ResourceResolver:
    """Resolve a logical component to one or more concrete physical lanes."""

    def __init__(self, scenario: InterconnectScenario) -> None:
        self.scenario = scenario

    def candidates(self, tenant: str, component: str) -> tuple[str, ...]:
        # Phase 2.3 starts with fully independent endpoint pipelines.
        if component in ENDPOINT_COMPONENTS:
            return (f"{tenant}::endpoint::{component}::lane0",)

        if component not in INTERCONNECT_COMPONENTS:
            raise KeyError(f"Unknown component: {component}")

        if component not in self.scenario.shared_components:
            return (f"{tenant}::interconnect::{component}::lane0",)

        if self.scenario.sharing_mode == "partitioned":
            return (f"partitioned::{component}::{tenant}::lane0",)

        if self.scenario.sharing_mode == "shared_pool":
            return tuple(
                f"shared::{component}::lane{lane}"
                for lane in range(self.scenario.component_capacity)
            )

        raise ValueError(
            f"Invalid sharing mode {self.scenario.sharing_mode} for shared component"
        )

    def physical_metadata(
        self, tenant: str, component: str
    ) -> tuple[str, int, bool]:
        if component in ENDPOINT_COMPONENTS:
            return "dedicated", 1, False
        if component not in self.scenario.shared_components:
            return "dedicated", 1, False
        if self.scenario.sharing_mode == "partitioned":
            return "partitioned", self.scenario.component_capacity, False
        return "shared_pool", self.scenario.component_capacity, True


# =============================================================================
# Event-driven staged simulator
# =============================================================================


class LinkSwitchPipelineSimulator:
    """FCFS, work-conserving simulator with concrete resource pools."""

    def __init__(
        self,
        *,
        scenario: InterconnectScenario,
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

    def _state_for_physical(self, tenant: str, component: str, physical: str) -> ResourceState:
        if physical not in self.resources:
            mode, capacity, cross_tenant = self.resolver.physical_metadata(
                tenant, component
            )
            self.resources[physical] = ResourceState(
                physical_resource=physical,
                logical_component=component,
                sharing_mode=mode,
                component_capacity=capacity,
                cross_tenant_capable=cross_tenant,
            )
        return self.resources[physical]

    def _candidate_states(self, tenant: str, component: str) -> list[ResourceState]:
        return [
            self._state_for_physical(tenant, component, physical)
            for physical in self.resolver.candidates(tenant, component)
        ]

    def _free_state(self, tenant: str, component: str) -> Optional[ResourceState]:
        free = [state for state in self._candidate_states(tenant, component) if state.free]
        if not free:
            return None
        return sorted(free, key=lambda state: state.physical_resource)[0]

    def _acquire(
        self,
        *,
        runtime: RequestRuntime,
        component: str,
        stage_name: str,
        now_ns: float,
        interval_kind: str,
    ) -> str:
        state = self._free_state(runtime.spec.tenant, component)
        if state is None:
            raise RuntimeError(f"No free lane for {component}")
        state.owner_request_id = runtime.spec.request_id
        state.owner_tenant = runtime.spec.tenant
        state.acquired_ns = now_ns
        state.acquired_stage = stage_name
        state.interval_kind = interval_kind
        return state.physical_resource

    def _release_physical(
        self,
        *,
        runtime: RequestRuntime,
        component: str,
        physical: str,
        released_stage: str,
        now_ns: float,
    ) -> str:
        state = self._state_for_physical(runtime.spec.tenant, component, physical)
        if state.owner_request_id != runtime.spec.request_id:
            raise RuntimeError(
                f"{runtime.spec.request_id} cannot release {physical}; "
                f"owner is {state.owner_request_id}"
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
                physical_resource=physical,
                logical_component=component,
                sharing_mode=state.sharing_mode,
                component_capacity=state.component_capacity,
                cross_tenant_capable=state.cross_tenant_capable,
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
        state.owner_request_id = None
        state.owner_tenant = None
        state.acquired_ns = None
        state.acquired_stage = None
        state.interval_kind = None
        return physical

    @staticmethod
    def _new_components(stage: StageDefinition) -> tuple[str, ...]:
        return tuple(stage.acquire_held) + tuple(stage.acquire_scoped)

    def _stage_startable(self, runtime: RequestRuntime, stage: StageDefinition) -> bool:
        for component in stage.require_held:
            if component not in runtime.held_resources:
                raise RuntimeError(
                    f"{runtime.spec.request_id} entered {stage.stage_name} "
                    f"without held {component}"
                )
        return all(
            self._free_state(runtime.spec.tenant, component) is not None
            for component in self._new_components(stage)
        )

    def _record_blockers(self, runtime: RequestRuntime, stage: StageDefinition) -> None:
        for component in self._new_components(stage):
            states = self._candidate_states(runtime.spec.tenant, component)
            if any(state.free for state in states):
                continue
            for state in states:
                runtime.blocking_observations.add(
                    (
                        state.physical_resource,
                        component,
                        str(state.owner_tenant),
                    )
                )

    def _start_stage(self, runtime: RequestRuntime, now_ns: float) -> None:
        stage_index = runtime.stage_index
        stage = self.pipeline[stage_index]
        acquired: list[str] = []
        required: list[str] = []

        for component in stage.require_held:
            required.append(runtime.held_resources[component][0])

        for component in stage.acquire_held:
            physical = self._acquire(
                runtime=runtime,
                component=component,
                stage_name=stage.stage_name,
                now_ns=now_ns,
                interval_kind="held_across_stages",
            )
            runtime.held_resources[component] = (physical, now_ns, stage.stage_name)
            acquired.append(physical)

        scoped: list[tuple[str, str]] = []
        for component in stage.acquire_scoped:
            physical = self._acquire(
                runtime=runtime,
                component=component,
                stage_name=stage.stage_name,
                now_ns=now_ns,
                interval_kind="stage_scoped",
            )
            scoped.append((component, physical))
            acquired.append(physical)

        blockers = sorted(runtime.blocking_observations)
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
                physical_acquired_resources=";".join(acquired),
                physical_required_resources=";".join(required),
                physical_released_resources=";".join(
                    runtime.held_resources[component][0]
                    for component in stage.release_held
                ),
                blocking_physical_resources=";".join(item[0] for item in blockers),
                blocking_components=";".join(sorted({item[1] for item in blockers})),
                blocking_owner_tenants=";".join(sorted({item[2] for item in blockers})),
                cross_tenant_blocked=any(
                    item[2] not in ("None", runtime.spec.tenant) for item in blockers
                ),
                self_blocked=any(item[2] == runtime.spec.tenant for item in blockers),
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
                tuple(physical for _, physical in scoped),
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
        stage = self.pipeline[stage_index]
        if runtime.stage_index != stage_index or not runtime.active:
            raise RuntimeError("Stage completion mismatch")

        for component, physical in zip(stage.acquire_scoped, scoped_physical):
            self._release_physical(
                runtime=runtime,
                component=component,
                physical=physical,
                released_stage=stage.stage_name,
                now_ns=now_ns,
            )

        for component in stage.release_held:
            physical = runtime.held_resources[component][0]
            self._release_physical(
                runtime=runtime,
                component=component,
                physical=physical,
                released_stage=stage.stage_name,
                now_ns=now_ns,
            )
            runtime.held_resources.pop(component, None)

        runtime.active = False
        runtime.stage_index += 1
        runtime.stage_ready_ns = float(now_ns)
        if runtime.stage_index == len(self.pipeline):
            if runtime.held_resources:
                raise RuntimeError(
                    f"Request completed while holding {runtime.held_resources}"
                )
            runtime.completed = True
            runtime.completion_ns = float(now_ns)

    @staticmethod
    def _candidate_sort_key(runtime: RequestRuntime) -> tuple[float, int, int, str]:
        tenant_order = 0 if runtime.spec.tenant == "victim" else 1
        return (
            runtime.stage_ready_ns,
            tenant_order,
            runtime.spec.request_index,
            runtime.spec.request_id,
        )

    def run(
        self, request_specs: Iterable[RequestSpec]
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
            while self._active_heap and self._active_heap[0][0] <= now_ns + FLOAT_TOLERANCE_NS:
                end_ns, _, request_id, stage_index, scoped = heapq.heappop(
                    self._active_heap
                )
                runtime = runtimes[request_id]
                was_completed = runtime.completed
                self._complete_stage(
                    runtime=runtime,
                    stage_index=stage_index,
                    now_ns=float(end_ns),
                    scoped_physical=scoped,
                )
                if runtime.completed and not was_completed:
                    completed_count += 1

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
                        self._start_stage(runtime, now_ns)
                        made_progress = True
                    else:
                        self._record_blockers(runtime, stage)

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
                raise RuntimeError(f"Simulation deadlock: {blocked}")
            next_now = min(next_times)
            if next_now <= now_ns + FLOAT_TOLERANCE_NS:
                raise RuntimeError("Event loop failed to advance")
            now_ns = next_now

        nominal_service_ns = float(sum(stage.duration_ns for stage in self.pipeline))
        request_rows: list[dict[str, object]] = []
        stage_rows: list[dict[str, object]] = []
        for runtime in runtimes.values():
            total_wait = float(sum(record.wait_ns for record in runtime.stage_records))
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
                        total_wait_ns=total_wait,
                        total_service_ns=nominal_service_ns,
                        completed=runtime.completed,
                    )
                )
            )
            stage_rows.extend(asdict(record) for record in runtime.stage_records)

        return (
            pd.DataFrame(request_rows),
            pd.DataFrame(stage_rows),
            pd.DataFrame(asdict(interval) for interval in self.resource_intervals),
        )


# =============================================================================
# Trial execution and paired comparisons
# =============================================================================


def run_simulation(
    *,
    scenario: InterconnectScenario,
    pipeline: tuple[StageDefinition, ...],
    run_kind: str,
    workload_name: str,
    trial_id: int,
    specs: list[RequestSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return LinkSwitchPipelineSimulator(
        scenario=scenario,
        pipeline=pipeline,
        run_kind=run_kind,
        workload_name=workload_name,
        trial_id=trial_id,
    ).run(specs)


def compare_attacker(
    *,
    baseline_requests: pd.DataFrame,
    baseline_stages: pd.DataFrame,
    combined_requests: pd.DataFrame,
    combined_stages: pd.DataFrame,
    trace_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Baseline controls are physically identical across Phase 2.3 scenarios and
    # are simulated once per workload/trial.  Drop their scenario label before
    # pairing them with each concrete combined configuration.
    base = baseline_requests[baseline_requests["tenant"] == "attacker"].copy()
    base = base.drop(columns=["scenario_name", "run_kind"])
    combo = combined_requests[combined_requests["tenant"] == "attacker"].copy()
    merged = combo.merge(
        base,
        on=["workload_name", "trial_id", "request_index"],
        suffixes=("_combined", "_attacker_only"),
        validate="one_to_one",
    )
    merged = merged.rename(columns={"request_index": "probe_index"})
    merged["trace_id"] = trace_id
    merged["excess_turnaround_ns"] = (
        merged["turnaround_ns_combined"] - merged["turnaround_ns_attacker_only"]
    )
    merged["affected"] = merged["excess_turnaround_ns"] > AFFECTED_THRESHOLD_NS

    base_stage = baseline_stages[baseline_stages["tenant"] == "attacker"].copy()
    base_stage = base_stage.drop(columns=["scenario_name", "run_kind"])
    combo_stage = combined_stages[combined_stages["tenant"] == "attacker"].copy()
    stage_cmp = combo_stage.merge(
        base_stage,
        on=[
            "workload_name",
            "trial_id",
            "request_index",
            "stage_index",
            "stage_name",
            "causal_component",
        ],
        suffixes=("_combined", "_attacker_only"),
        validate="one_to_one",
    )
    stage_cmp = stage_cmp.rename(columns={"request_index": "probe_index"})
    stage_cmp["trace_id"] = trace_id
    stage_cmp["added_stage_wait_ns"] = (
        stage_cmp["wait_ns_combined"] - stage_cmp["wait_ns_attacker_only"]
    )
    stage_cmp["positive_added_stage_wait"] = (
        stage_cmp["added_stage_wait_ns"] > AFFECTED_THRESHOLD_NS
    )
    return merged, stage_cmp

def compare_victim(
    *, victim_only: pd.DataFrame, combined: pd.DataFrame
) -> pd.DataFrame:
    left = victim_only[victim_only["tenant"] == "victim"].copy()
    left = left.drop(columns=["scenario_name", "run_kind"])
    right = combined[combined["tenant"] == "victim"].copy()
    merged = right.merge(
        left,
        on=["workload_name", "trial_id", "request_index"],
        suffixes=("_combined", "_victim_only"),
        validate="one_to_one",
    )
    merged["added_victim_turnaround_ns"] = (
        merged["turnaround_ns_combined"] - merged["turnaround_ns_victim_only"]
    )
    return merged

def trial_summary(
    *,
    scenario: InterconnectScenario,
    workload: VictimWorkload,
    trial_id: int,
    phase_ns: float,
    attacker_cmp: pd.DataFrame,
    stage_cmp: pd.DataFrame,
    victim_cmp: pd.DataFrame,
    victim_only_requests: pd.DataFrame,
    combined_requests: pd.DataFrame,
) -> dict[str, object]:
    positive = attacker_cmp[attacker_cmp["affected"]]
    cross_stage = stage_cmp[
        stage_cmp["positive_added_stage_wait"]
        & stage_cmp["cross_tenant_blocked_combined"]
    ]
    self_stage = stage_cmp[
        stage_cmp["positive_added_stage_wait"] & stage_cmp["self_blocked_combined"]
    ]

    victim_only_rows = victim_only_requests[victim_only_requests["tenant"] == "victim"]
    combined_victim = combined_requests[combined_requests["tenant"] == "victim"]
    base_mean = float(victim_only_rows["turnaround_ns"].mean())
    combo_mean = float(combined_victim["turnaround_ns"].mean())
    base_makespan = float(
        victim_only_rows["completion_ns"].max() - victim_only_rows["ready_ns"].min()
    )
    combo_makespan = float(
        combined_victim["completion_ns"].max() - combined_victim["ready_ns"].min()
    )

    root_components = sorted(cross_stage["causal_component"].unique().tolist())
    return {
        "scenario_name": scenario.scenario_name,
        "scenario_family": scenario.scenario_family,
        "sharing_mode": scenario.sharing_mode,
        "component_capacity": scenario.component_capacity,
        "shared_components": ";".join(scenario.shared_components) or "none",
        "workload_name": workload.workload_name,
        "trial_id": trial_id,
        "victim_phase_ns": phase_ns,
        "probe_count": int(len(attacker_cmp)),
        "affected_probe_count": int(attacker_cmp["affected"].sum()),
        "affected_probe_fraction": float(attacker_cmp["affected"].mean()),
        "mean_excess_turnaround_ns": float(attacker_cmp["excess_turnaround_ns"].mean()),
        "median_excess_turnaround_ns": float(attacker_cmp["excess_turnaround_ns"].median()),
        "p95_excess_turnaround_ns": float(np.percentile(attacker_cmp["excess_turnaround_ns"], 95)),
        "max_excess_turnaround_ns": float(attacker_cmp["excess_turnaround_ns"].max()),
        "total_excess_turnaround_ns": float(attacker_cmp["excess_turnaround_ns"].sum()),
        "cross_tenant_added_wait_ns": float(cross_stage["added_stage_wait_ns"].sum()),
        "self_cascade_added_wait_ns": float(self_stage["added_stage_wait_ns"].sum()),
        "root_blocking_components": ";".join(root_components) or "none",
        "victim_mean_latency_slowdown": combo_mean / base_mean if base_mean > 0 else math.nan,
        "victim_makespan_slowdown": combo_makespan / base_makespan if base_makespan > 0 else math.nan,
        "mean_added_victim_turnaround_ns": float(victim_cmp["added_victim_turnaround_ns"].mean()),
        "max_added_victim_turnaround_ns": float(victim_cmp["added_victim_turnaround_ns"].max()),
        "affected_probe_mean_delay_ns": float(positive["excess_turnaround_ns"].mean()) if not positive.empty else 0.0,
    }


# =============================================================================
# Resource analysis
# =============================================================================


def union_busy_time(intervals: pd.DataFrame) -> float:
    if intervals.empty:
        return 0.0
    ordered = intervals.sort_values(["start_ns", "end_ns"])
    total = 0.0
    start = float(ordered.iloc[0]["start_ns"])
    end = float(ordered.iloc[0]["end_ns"])
    for _, row in ordered.iloc[1:].iterrows():
        row_start = float(row["start_ns"])
        row_end = float(row["end_ns"])
        if row_start <= end + FLOAT_TOLERANCE_NS:
            end = max(end, row_end)
        else:
            total += end - start
            start, end = row_start, row_end
    return total + end - start


def resource_utilization_rows(intervals: pd.DataFrame) -> pd.DataFrame:
    combo = intervals[intervals["run_kind"] == "combined"].copy()
    if combo.empty:
        return pd.DataFrame()
    combo["attacker_occupancy_ns"] = np.where(
        combo["tenant"] == "attacker", combo["occupancy_ns"], 0.0
    )
    combo["victim_occupancy_ns"] = np.where(
        combo["tenant"] == "victim", combo["occupancy_ns"], 0.0
    )
    group_cols = [
        "scenario_name",
        "workload_name",
        "trial_id",
        "physical_resource",
        "logical_component",
        "sharing_mode",
        "component_capacity",
        "cross_tenant_capable",
    ]
    result = (
        combo.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            busy_time_ns=("occupancy_ns", "sum"),
            attacker_busy_time_ns=("attacker_occupancy_ns", "sum"),
            victim_busy_time_ns=("victim_occupancy_ns", "sum"),
            interval_count=("request_id", "count"),
            maximum_end_ns=("end_ns", "max"),
        )
    )
    result["analysis_horizon_ns"] = np.maximum(
        result["maximum_end_ns"].to_numpy(dtype=float), OBSERVATION_WINDOW_NS
    )
    result["utilization"] = (
        result["busy_time_ns"] / result["analysis_horizon_ns"]
    )
    return result.drop(columns=["maximum_end_ns"])

def resource_capacity_rows(intervals: pd.DataFrame) -> pd.DataFrame:
    combo = intervals[
        (intervals["run_kind"] == "combined")
        & (intervals["logical_component"].isin(INTERCONNECT_COMPONENTS))
    ].copy()
    if combo.empty:
        return pd.DataFrame()

    group_cols = [
        "scenario_name",
        "workload_name",
        "trial_id",
        "logical_component",
        "sharing_mode",
        "component_capacity",
    ]
    starts = combo[group_cols + ["start_ns"]].rename(
        columns={"start_ns": "event_time_ns"}
    )
    starts["delta"] = 1
    ends = combo[group_cols + ["end_ns"]].rename(
        columns={"end_ns": "event_time_ns"}
    )
    ends["delta"] = -1
    events = pd.concat([starts, ends], ignore_index=True)
    # Release (-1) before acquisition (+1) at an identical timestamp.
    events = events.sort_values(group_cols + ["event_time_ns", "delta"])
    events["concurrent"] = events.groupby(group_cols, sort=False)["delta"].cumsum()
    concurrency = (
        events.groupby(group_cols, as_index=False)["concurrent"]
        .max()
        .rename(columns={"concurrent": "maximum_concurrent_occupancy"})
    )

    aggregate = (
        combo.groupby(group_cols, as_index=False)
        .agg(
            aggregate_occupancy_ns=("occupancy_ns", "sum"),
            maximum_end_ns=("end_ns", "max"),
            physical_lane_count=("physical_resource", "nunique"),
        )
    )
    result = concurrency.merge(aggregate, on=group_cols, validate="one_to_one")
    result["capacity_limit"] = np.where(
        result["sharing_mode"] == "shared_pool",
        result["component_capacity"],
        result["physical_lane_count"],
    ).astype(int)
    result["configured_component_capacity"] = result["component_capacity"]
    result["capacity_respected"] = (
        result["maximum_concurrent_occupancy"] <= result["capacity_limit"]
    )
    result["analysis_horizon_ns"] = np.maximum(
        result["maximum_end_ns"].to_numpy(dtype=float), OBSERVATION_WINDOW_NS
    )
    result["aggregate_lane_utilization"] = (
        result["aggregate_occupancy_ns"]
        / (result["capacity_limit"] * result["analysis_horizon_ns"])
    )
    return result.drop(columns=["maximum_end_ns", "physical_lane_count"])

def reuse_rows(intervals: pd.DataFrame) -> pd.DataFrame:
    combo = intervals[
        (intervals["run_kind"] == "combined")
        & (intervals["logical_component"].isin(INTERCONNECT_COMPONENTS))
    ].copy()
    if combo.empty:
        return pd.DataFrame()

    group_cols = [
        "scenario_name",
        "workload_name",
        "trial_id",
        "physical_resource",
    ]
    ordered = combo.sort_values(group_cols + ["start_ns", "end_ns", "request_id"]).copy()
    grouped = ordered.groupby(group_cols, sort=False)
    ordered["previous_tenant"] = grouped["tenant"].shift(1)
    ordered["previous_start_ns"] = grouped["start_ns"].shift(1)
    ordered["previous_end_ns"] = grouped["end_ns"].shift(1)
    ordered["previous_hold_ns"] = grouped["occupancy_ns"].shift(1)
    ordered["reuse_index"] = grouped.cumcount()
    ordered = ordered[ordered["reuse_index"] > 0].copy()
    ordered["current_tenant"] = ordered["tenant"]
    ordered["current_start_ns"] = ordered["start_ns"]
    ordered["cross_tenant_reuse"] = (
        ordered["previous_tenant"] != ordered["current_tenant"]
    )
    ordered["acquisition_to_acquisition_ns"] = (
        ordered["current_start_ns"] - ordered["previous_start_ns"]
    )
    ordered["release_to_reacquire_gap_ns"] = (
        ordered["current_start_ns"] - ordered["previous_end_ns"]
    )
    keep = [
        "scenario_name",
        "workload_name",
        "trial_id",
        "physical_resource",
        "logical_component",
        "sharing_mode",
        "component_capacity",
        "reuse_index",
        "previous_tenant",
        "current_tenant",
        "cross_tenant_reuse",
        "previous_start_ns",
        "previous_end_ns",
        "current_start_ns",
        "acquisition_to_acquisition_ns",
        "release_to_reacquire_gap_ns",
        "previous_hold_ns",
    ]
    return ordered[keep].reset_index(drop=True)


# =============================================================================
# Fingerprinting and aggregation
# =============================================================================


def macro_f1_score(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else math.nan


def workload_fingerprint(
    attacker_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for scenario_name, scenario_group in attacker_comparison.groupby("scenario_name"):
        vectors: dict[tuple[str, int], np.ndarray] = {}
        for (workload, trial_id), group in scenario_group.groupby(["workload_name", "trial_id"]):
            vector = (
                group.sort_values("probe_index")["excess_turnaround_ns"]
                .to_numpy(dtype=float)
            )
            vectors[(str(workload), int(trial_id))] = vector

        trial_ids = sorted({trial_id for _, trial_id in vectors})
        for test_trial in trial_ids:
            train_keys = [key for key in vectors if key[1] != test_trial]
            test_keys = [key for key in vectors if key[1] == test_trial]
            if not train_keys or not test_keys:
                continue
            training_values = np.concatenate([vectors[key] for key in train_keys])
            scale = float(np.std(training_values))
            if scale <= FLOAT_TOLERANCE_NS:
                scale = 1.0
            centroids: dict[str, np.ndarray] = {}
            for workload in sorted({key[0] for key in train_keys}):
                centroids[workload] = np.mean(
                    [vectors[key] for key in train_keys if key[0] == workload], axis=0
                )
            for true_workload, trial_id in test_keys:
                vector = vectors[(true_workload, trial_id)]
                distances = {
                    workload: float(np.linalg.norm((vector - centroid) / scale))
                    for workload, centroid in centroids.items()
                }
                predicted = min(distances, key=distances.get)
                predictions.append(
                    {
                        "scenario_name": scenario_name,
                        "trial_id": trial_id,
                        "true_workload": true_workload,
                        "predicted_workload": predicted,
                        "correct": predicted == true_workload,
                        "nearest_distance": distances[predicted],
                        "distance_margin": (
                            sorted(distances.values())[1] - sorted(distances.values())[0]
                            if len(distances) > 1
                            else math.nan
                        ),
                    }
                )

    pred_df = pd.DataFrame(predictions)
    if pred_df.empty:
        return pd.DataFrame(), pred_df
    for scenario_name, group in pred_df.groupby("scenario_name"):
        true = group["true_workload"].tolist()
        pred = group["predicted_workload"].tolist()
        trace_rows.append(
            {
                "scenario_name": scenario_name,
                "sample_count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "macro_f1": macro_f1_score(true, pred),
                "chance_accuracy": 1.0 / len(build_workloads()),
                "mean_distance_margin": float(group["distance_margin"].mean()),
            }
        )
    return pd.DataFrame(trace_rows), pred_df


def aggregate_outputs(
    *,
    trial_df: pd.DataFrame,
    attacker_cmp: pd.DataFrame,
    stage_cmp: pd.DataFrame,
    utilization: pd.DataFrame,
    capacity: pd.DataFrame,
    reuse: pd.DataFrame,
    fingerprint_metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    scenario_summary = (
        trial_df.groupby(
            [
                "scenario_name",
                "scenario_family",
                "sharing_mode",
                "component_capacity",
                "shared_components",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
            mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
            median_trial_excess_turnaround_ns=("mean_excess_turnaround_ns", "median"),
            mean_p95_excess_turnaround_ns=("p95_excess_turnaround_ns", "mean"),
            maximum_excess_turnaround_ns=("max_excess_turnaround_ns", "max"),
            mean_total_excess_turnaround_ns=("total_excess_turnaround_ns", "mean"),
            mean_victim_latency_slowdown=("victim_mean_latency_slowdown", "mean"),
            mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
            mean_cross_tenant_added_wait_ns=("cross_tenant_added_wait_ns", "mean"),
            mean_self_cascade_added_wait_ns=("self_cascade_added_wait_ns", "mean"),
        )
    )
    if not fingerprint_metrics.empty:
        scenario_summary = scenario_summary.merge(
            fingerprint_metrics[["scenario_name", "accuracy", "macro_f1"]].rename(
                columns={
                    "accuracy": "workload_fingerprint_accuracy",
                    "macro_f1": "workload_fingerprint_macro_f1",
                }
            ),
            on="scenario_name",
            how="left",
            validate="one_to_one",
        )

    workload_summary = (
        trial_df.groupby(
            [
                "scenario_name",
                "sharing_mode",
                "component_capacity",
                "workload_name",
            ],
            as_index=False,
        )
        .agg(
            trial_count=("trial_id", "count"),
            mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
            mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
            mean_p95_excess_turnaround_ns=("p95_excess_turnaround_ns", "mean"),
            maximum_excess_turnaround_ns=("max_excess_turnaround_ns", "max"),
            mean_victim_latency_slowdown=("victim_mean_latency_slowdown", "mean"),
            mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
        )
    )

    positive_stage = stage_cmp[stage_cmp["positive_added_stage_wait"]].copy()
    if positive_stage.empty:
        decomposition = pd.DataFrame()
        blocking = pd.DataFrame()
    else:
        decomposition = (
            stage_cmp.groupby(
                [
                    "scenario_name",
                    "workload_name",
                    "causal_component",
                    "stage_name",
                ],
                as_index=False,
            )
            .agg(
                probe_stage_count=("probe_index", "count"),
                affected_stage_count=("positive_added_stage_wait", "sum"),
                affected_stage_fraction=("positive_added_stage_wait", "mean"),
                mean_added_stage_wait_ns=("added_stage_wait_ns", "mean"),
                mean_positive_added_stage_wait_ns=(
                    "added_stage_wait_ns",
                    lambda values: float(values[values > AFFECTED_THRESHOLD_NS].mean())
                    if (values > AFFECTED_THRESHOLD_NS).any()
                    else 0.0,
                ),
                maximum_added_stage_wait_ns=("added_stage_wait_ns", "max"),
                cross_tenant_block_count=("cross_tenant_blocked_combined", "sum"),
                self_block_count=("self_blocked_combined", "sum"),
            )
        )
        root = positive_stage[positive_stage["cross_tenant_blocked_combined"]].copy()
        if root.empty:
            blocking = pd.DataFrame()
        else:
            blocking = (
                root.groupby(
                    [
                        "scenario_name",
                        "workload_name",
                        "causal_component",
                        "blocking_components_combined",
                    ],
                    as_index=False,
                )
                .agg(
                    affected_probe_stage_count=("probe_index", "count"),
                    mean_added_wait_ns=("added_stage_wait_ns", "mean"),
                    total_added_wait_ns=("added_stage_wait_ns", "sum"),
                    maximum_added_wait_ns=("added_stage_wait_ns", "max"),
                )
            )

    if utilization.empty:
        utilization_summary = utilization.copy()
    else:
        utilization_summary = (
            utilization.groupby(
                [
                    "scenario_name",
                    "logical_component",
                    "sharing_mode",
                    "component_capacity",
                    "cross_tenant_capable",
                ],
                as_index=False,
            )
            .agg(
                physical_lane_trial_count=("physical_resource", "count"),
                mean_lane_utilization=("utilization", "mean"),
                max_lane_utilization=("utilization", "max"),
                mean_busy_time_ns=("busy_time_ns", "mean"),
                mean_interval_count=("interval_count", "mean"),
            )
        )

    if capacity.empty:
        capacity_summary = capacity.copy()
    else:
        capacity_summary = (
            capacity.groupby(
                [
                    "scenario_name",
                    "logical_component",
                    "sharing_mode",
                    "component_capacity",
                ],
                as_index=False,
            )
            .agg(
                trial_count=("trial_id", "count"),
                maximum_observed_concurrency=("maximum_concurrent_occupancy", "max"),
                all_capacity_checks_passed=("capacity_respected", "all"),
                mean_aggregate_lane_utilization=("aggregate_lane_utilization", "mean"),
                max_aggregate_lane_utilization=("aggregate_lane_utilization", "max"),
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
                    "sharing_mode",
                    "component_capacity",
                    "cross_tenant_reuse",
                ],
                as_index=False,
            )
            .agg(
                reuse_event_count=("reuse_index", "count"),
                mean_acquisition_to_acquisition_ns=("acquisition_to_acquisition_ns", "mean"),
                median_acquisition_to_acquisition_ns=("acquisition_to_acquisition_ns", "median"),
                mean_release_to_reacquire_gap_ns=("release_to_reacquire_gap_ns", "mean"),
                minimum_release_to_reacquire_gap_ns=("release_to_reacquire_gap_ns", "min"),
                mean_previous_hold_ns=("previous_hold_ns", "mean"),
            )
        )

    blackbox_trace_summary = (
        attacker_cmp.groupby(["trace_id"], as_index=False)
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
        "delay_decomposition_summary": decomposition,
        "blocking_attribution_summary": blocking,
        "resource_utilization_summary": utilization_summary,
        "resource_capacity_summary": capacity_summary,
        "interconnect_reuse_summary": reuse_summary,
        "blackbox_trace_summary": blackbox_trace_summary,
    }


# =============================================================================
# Validation
# =============================================================================


def add_assertion(
    assertions: list[ValidationAssertion],
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
    requests: pd.DataFrame,
    stages: pd.DataFrame,
    intervals: pd.DataFrame,
    attacker_cmp: pd.DataFrame,
    stage_cmp: pd.DataFrame,
    capacity_df: pd.DataFrame,
    blackbox: pd.DataFrame,
    scenarios: tuple[InterconnectScenario, ...],
) -> pd.DataFrame:
    assertions: list[ValidationAssertion] = []

    add_assertion(
        assertions,
        "completion",
        "all_requests_completed",
        bool(requests["completed"].all()),
        True,
        bool(requests["completed"].all()),
    )

    attacker_only = requests[
        (requests["run_kind"] == "attacker_only")
        & (requests["tenant"] == "attacker")
    ]
    max_self_wait = float(attacker_only["total_wait_ns"].max())
    add_assertion(
        assertions,
        "baseline",
        "selected_probe_has_zero_attacker_only_wait",
        max_self_wait <= FLOAT_TOLERANCE_NS,
        "0 ns",
        max_self_wait,
    )

    isolated = attacker_cmp[
        attacker_cmp["scenario_name"] == "fully_isolated_interconnect_control"
    ]
    max_isolated = float(isolated["excess_turnaround_ns"].abs().max())
    add_assertion(
        assertions,
        "negative_controls",
        "fully_isolated_interconnect_has_zero_leakage",
        max_isolated <= FLOAT_TOLERANCE_NS,
        "0 ns",
        max_isolated,
    )

    partitioned_names = {
        scenario.scenario_name
        for scenario in scenarios
        if scenario.sharing_mode == "partitioned"
    }
    partitioned = attacker_cmp[attacker_cmp["scenario_name"].isin(partitioned_names)]
    max_partitioned = (
        float(partitioned["excess_turnaround_ns"].abs().max())
        if not partitioned.empty
        else 0.0
    )
    add_assertion(
        assertions,
        "negative_controls",
        "tenant_partitioned_lanes_remove_cross_tenant_leakage",
        max_partitioned <= FLOAT_TOLERANCE_NS,
        "0 ns",
        max_partitioned,
    )

    endpoint_cross = stages[
        stages["causal_component"].isin(ENDPOINT_COMPONENTS)
        & stages["cross_tenant_blocked"]
    ]
    add_assertion(
        assertions,
        "endpoint_isolation",
        "no_cross_tenant_endpoint_blocking",
        endpoint_cross.empty,
        0,
        len(endpoint_cross),
    )

    # External request delay must equal summed stage-wait increase.
    sums = (
        stage_cmp.groupby(
            ["trace_id", "probe_index"], as_index=False
        )["added_stage_wait_ns"].sum()
        .rename(columns={"added_stage_wait_ns": "summed_added_stage_wait_ns"})
    )
    accounting = attacker_cmp.merge(
        sums,
        on=["trace_id", "probe_index"],
        validate="one_to_one",
    )
    max_accounting_error = float(
        (
            accounting["excess_turnaround_ns"]
            - accounting["summed_added_stage_wait_ns"]
        ).abs().max()
    )
    add_assertion(
        assertions,
        "causal_accounting",
        "external_delay_equals_sum_of_added_stage_wait",
        max_accounting_error <= 1e-7,
        "<=1e-7 ns",
        max_accounting_error,
    )

    # Single-component capacity-one cases must attribute cross-tenant root waits
    # only to the intentionally shared component.
    single_cap1 = [
        scenario
        for scenario in scenarios
        if scenario.scenario_family.startswith("single_")
        and scenario.sharing_mode == "shared_pool"
        and scenario.component_capacity == 1
    ]
    for scenario in single_cap1:
        target = scenario.shared_components[0]
        rows = stage_cmp[
            (stage_cmp["scenario_name"] == scenario.scenario_name)
            & stage_cmp["positive_added_stage_wait"]
            & stage_cmp["cross_tenant_blocked_combined"]
        ]
        observed = set(rows["causal_component"].unique().tolist())
        add_assertion(
            assertions,
            "single_resource_attribution",
            f"{scenario.scenario_name}_root_blocking_is_target_only",
            observed.issubset({target}),
            {target},
            observed,
        )
        add_assertion(
            assertions,
            "single_resource_attribution",
            f"{scenario.scenario_name}_target_blocking_observed",
            target in observed,
            target,
            observed,
        )

    add_assertion(
        assertions,
        "capacity",
        "all_interconnect_capacity_limits_respected",
        bool(capacity_df["capacity_respected"].all()),
        True,
        bool(capacity_df["capacity_respected"].all()),
    )

    # No physical lane may have overlapping intervals.
    overlap_group_cols = [
        "run_kind",
        "scenario_name",
        "workload_name",
        "trial_id",
        "physical_resource",
    ]
    ordered_intervals = intervals.sort_values(
        overlap_group_cols + ["start_ns", "end_ns"]
    ).copy()
    group_ids = ordered_intervals.groupby(overlap_group_cols, sort=False).ngroup()
    cumulative_end = ordered_intervals.groupby(group_ids)["end_ns"].cummax()
    previous_end = cumulative_end.groupby(group_ids).shift(1)
    overlap_count = int(
        (
            ordered_intervals["start_ns"]
            < previous_end.fillna(-np.inf) - FLOAT_TOLERANCE_NS
        ).sum()
    )
    add_assertion(
        assertions,
        "resource_calendar",
        "no_illegal_physical_lane_overlap",
        overlap_count == 0,
        0,
        overlap_count,
    )

    release_checks = {
        "endpoint_port": "intermodule_transmission",
        "source_link_interface": "intermodule_transmission",
        "switch_path": "intermodule_transmission",
        "communication_qubit": "communication_reset",
    }
    for component, expected_stage in release_checks.items():
        rows = intervals[intervals["logical_component"] == component]
        invalid = rows[rows["released_stage"] != expected_stage]
        add_assertion(
            assertions,
            "resource_lifetimes",
            f"{component}_released_at_expected_stage",
            invalid.empty,
            expected_stage,
            sorted(rows["released_stage"].unique().tolist()),
        )

    add_assertion(
        assertions,
        "blackbox_boundary",
        "blackbox_file_contains_only_external_timing_fields",
        set(blackbox.columns) == BLACKBOX_ALLOWED_COLUMNS,
        sorted(BLACKBOX_ALLOWED_COLUMNS),
        sorted(blackbox.columns),
    )

    return pd.DataFrame(asdict(assertion) for assertion in assertions)


# =============================================================================
# Configuration and plots
# =============================================================================


def configuration_table(
    scenarios: tuple[InterconnectScenario, ...],
    workloads: tuple[VictimWorkload, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for workload in workloads:
            row: dict[str, object] = {
                "scenario_name": scenario.scenario_name,
                "scenario_family": scenario.scenario_family,
                "sharing_mode": scenario.sharing_mode,
                "component_capacity": scenario.component_capacity,
                "shared_components": ";".join(scenario.shared_components) or "none",
                "expected_mechanism": scenario.expected_mechanism,
                "scenario_description": scenario.description,
                "workload_name": workload.workload_name,
                "workload_description": workload.description,
                "release_pattern": workload.release_pattern,
            }
            for component in INTERCONNECT_COMPONENTS:
                row[f"shared_{component}"] = component in scenario.shared_components
            rows.append(row)
    return pd.DataFrame(rows)


def save_plots(
    *,
    scenario_summary: pd.DataFrame,
    workload_summary: pd.DataFrame,
    fingerprint_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = scenario_summary.sort_values("mean_excess_turnaround_ns")

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(ordered["scenario_name"], ordered["mean_excess_turnaround_ns"])
    ax.set_xlabel("Mean attacker excess turnaround (ns)")
    ax.set_ylabel("Interconnect configuration")
    ax.set_title("Phase 2.3: Link/switch leakage map")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_03_mean_excess_by_scenario.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(ordered["scenario_name"], ordered["mean_affected_probe_fraction"])
    ax.set_xlabel("Fraction of attacker probes affected")
    ax.set_ylabel("Interconnect configuration")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Phase 2.3: Probe exposure by interconnect component")
    fig.tight_layout()
    fig.savefig(output_dir / "phase2_03_affected_fraction_by_scenario.png", dpi=200)
    plt.close(fig)

    single = scenario_summary[
        scenario_summary["scenario_family"].str.startswith("single_")
        & scenario_summary["sharing_mode"].isin(["shared_pool", "partitioned"])
    ].copy()
    if not single.empty:
        pivot = single.pivot_table(
            index="scenario_family",
            columns=["sharing_mode", "component_capacity"],
            values="mean_excess_turnaround_ns",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(12, 7))
        image = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_xticks(
            range(len(pivot.columns)),
            labels=[f"{mode}/cap{capacity}" for mode, capacity in pivot.columns],
            rotation=35,
            ha="right",
        )
        ax.set_title("Phase 2.3: Concrete capacity and partitioning controls")
        ax.set_xlabel("Resource organization")
        ax.set_ylabel("Shared interconnect component")
        fig.colorbar(image, ax=ax, label="Mean excess turnaround (ns)")
        fig.tight_layout()
        fig.savefig(output_dir / "phase2_03_capacity_leakage_map.png", dpi=200)
        plt.close(fig)

    if not fingerprint_metrics.empty:
        ordered_fp = fingerprint_metrics.sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.barh(ordered_fp["scenario_name"], ordered_fp["accuracy"])
        ax.axvline(1.0 / 3.0, linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Leave-one-trial-out workload classification accuracy")
        ax.set_ylabel("Interconnect configuration")
        ax.set_title("Phase 2.3: Victim communication-structure visibility")
        fig.tight_layout()
        fig.savefig(output_dir / "phase2_03_workload_fingerprint_accuracy.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.3 link/switch pipeline contention experiment"
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 1:
        raise ValueError("--trials must be at least 2 for workload fingerprinting")

    output_dir: Path = args.output_dir
    raw_dir = output_dir / "raw"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pipeline = build_pipeline()
    scenarios = build_scenarios()
    workloads = build_workloads()

    print("Phase 2.3 — Link/Switch Pipeline Contention")
    print("=" * 58)
    print(f"Interconnect scenarios: {len(scenarios)}")
    print(f"Victim workloads: {len(workloads)}")
    print(f"Trials per scenario/workload: {args.trials}")
    print(f"Controlled trial tuples: {len(scenarios) * len(workloads) * args.trials}")
    print(f"Nominal staged remote-operation latency: {sum(STAGE_DURATIONS_NS.values()):.1f} ns")

    phase_schedule: dict[tuple[str, int], float] = {}
    phase_rows: list[dict[str, object]] = []
    for workload in workloads:
        period = 900.0 if workload.workload_name == "periodic_sparse" else 460.0
        if workload.workload_name == "layered_bursty":
            period = 2400.0
        for trial_id in range(args.trials):
            phase = float(rng.uniform(0.0, period))
            phase_schedule[(workload.workload_name, trial_id)] = phase
            phase_rows.append(
                {
                    "workload_name": workload.workload_name,
                    "trial_id": trial_id,
                    "victim_phase_ns": phase,
                }
            )

    request_frames: list[pd.DataFrame] = []
    stage_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    attacker_frames: list[pd.DataFrame] = []
    stage_comparison_frames: list[pd.DataFrame] = []
    victim_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, object]] = []
    trace_key_rows: list[dict[str, object]] = []

    # Attacker-only and victim-only controls are independent of which network
    # component is shared: every tenant has its own endpoint pipeline, and the
    # single-tenant communication qubit serializes its own requests before any
    # cross-tenant interconnect organization can matter.  Simulate these paired
    # controls once per workload/trial and reuse them for all scenarios.
    baseline_cache: dict[tuple[str, int], dict[str, object]] = {}
    baseline_scenario = scenarios[0]
    print("Precomputing paired attacker-only and victim-only controls...")
    for workload in workloads:
        for trial_id in range(args.trials):
            phase = phase_schedule[(workload.workload_name, trial_id)]
            attacker_specs = generate_attacker_specs(
                trial_id=trial_id, workload_name=workload.workload_name
            )
            victim_specs = generate_victim_specs(
                workload=workload, trial_id=trial_id, phase_ns=phase
            )
            a_req, a_stage, a_int = run_simulation(
                scenario=baseline_scenario,
                pipeline=pipeline,
                run_kind="attacker_only",
                workload_name=workload.workload_name,
                trial_id=trial_id,
                specs=attacker_specs,
            )
            v_req, v_stage, v_int = run_simulation(
                scenario=baseline_scenario,
                pipeline=pipeline,
                run_kind="victim_only",
                workload_name=workload.workload_name,
                trial_id=trial_id,
                specs=victim_specs,
            )
            baseline_cache[(workload.workload_name, trial_id)] = {
                "attacker_specs": attacker_specs,
                "victim_specs": victim_specs,
                "a_req": a_req,
                "a_stage": a_stage,
                "a_int": a_int,
                "v_req": v_req,
                "v_stage": v_stage,
                "v_int": v_int,
            }
            request_frames.extend([a_req, v_req])
            stage_frames.extend([a_stage, v_stage])
            interval_frames.extend([a_int, v_int])

    completed = 0
    total = len(scenarios) * len(workloads) * args.trials
    for scenario_index, scenario in enumerate(scenarios):
        for workload in workloads:
            for trial_id in range(args.trials):
                phase = phase_schedule[(workload.workload_name, trial_id)]
                cached = baseline_cache[(workload.workload_name, trial_id)]
                attacker_specs = cached["attacker_specs"]
                victim_specs = cached["victim_specs"]
                a_req = cached["a_req"]
                a_stage = cached["a_stage"]
                v_req = cached["v_req"]
                trace_id = (
                    f"trace_s{scenario_index:02d}_w{workload.workload_name}_t{trial_id:03d}"
                )

                c_req, c_stage, c_int = run_simulation(
                    scenario=scenario,
                    pipeline=pipeline,
                    run_kind="combined",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                    specs=attacker_specs + victim_specs,
                )

                attacker_cmp, stage_cmp = compare_attacker(
                    baseline_requests=a_req,
                    baseline_stages=a_stage,
                    combined_requests=c_req,
                    combined_stages=c_stage,
                    trace_id=trace_id,
                )
                victim_cmp = compare_victim(victim_only=v_req, combined=c_req)

                request_frames.append(c_req)
                stage_frames.append(c_stage)
                interval_frames.append(c_int)
                attacker_frames.append(attacker_cmp)
                stage_comparison_frames.append(stage_cmp)
                victim_frames.append(victim_cmp)
                trial_rows.append(
                    trial_summary(
                        scenario=scenario,
                        workload=workload,
                        trial_id=trial_id,
                        phase_ns=phase,
                        attacker_cmp=attacker_cmp,
                        stage_cmp=stage_cmp,
                        victim_cmp=victim_cmp,
                        victim_only_requests=v_req,
                        combined_requests=c_req,
                    )
                )
                trace_key_rows.append(
                    {
                        "trace_id": trace_id,
                        "scenario_name": scenario.scenario_name,
                        "scenario_family": scenario.scenario_family,
                        "sharing_mode": scenario.sharing_mode,
                        "component_capacity": scenario.component_capacity,
                        "shared_components": ";".join(scenario.shared_components) or "none",
                        "workload_name": workload.workload_name,
                        "trial_id": trial_id,
                        "victim_phase_ns": phase,
                    }
                )
                completed += 1
                if completed % max(1, total // 10) == 0 or completed == total:
                    print(f"Completed {completed}/{total} controlled trial tuples")

    print("Assembling evaluator data frames...")
    requests = pd.concat(request_frames, ignore_index=True)
    stages = pd.concat(stage_frames, ignore_index=True)
    intervals = pd.concat(interval_frames, ignore_index=True)
    attacker_cmp = pd.concat(attacker_frames, ignore_index=True)
    stage_cmp = pd.concat(stage_comparison_frames, ignore_index=True)
    victim_cmp = pd.concat(victim_frames, ignore_index=True)
    trial_df = pd.DataFrame(trial_rows)
    trace_key = pd.DataFrame(trace_key_rows)

    print("Computing utilization, capacity, reuse, and fingerprint summaries...")
    utilization = resource_utilization_rows(intervals)
    capacity_df = resource_capacity_rows(intervals)
    reuse = reuse_rows(intervals)
    fingerprint_metrics, fingerprint_predictions = workload_fingerprint(attacker_cmp)

    aggregates = aggregate_outputs(
        trial_df=trial_df,
        attacker_cmp=attacker_cmp,
        stage_cmp=stage_cmp,
        utilization=utilization,
        capacity=capacity_df,
        reuse=reuse,
        fingerprint_metrics=fingerprint_metrics,
    )

    blackbox = pd.DataFrame(
        {
            "trace_id": attacker_cmp["trace_id"],
            "probe_index": attacker_cmp["probe_index"],
            "release_ns": attacker_cmp["ready_ns_attacker_only"],
            "attacker_only_completion_ns": attacker_cmp["completion_ns_attacker_only"],
            "combined_completion_ns": attacker_cmp["completion_ns_combined"],
            "attacker_only_turnaround_ns": attacker_cmp["turnaround_ns_attacker_only"],
            "combined_turnaround_ns": attacker_cmp["turnaround_ns_combined"],
            "excess_turnaround_ns": attacker_cmp["excess_turnaround_ns"],
            "affected": attacker_cmp["affected"],
        }
    )

    print("Running causal and resource-calendar validations...")
    assertions = validate_results(
        requests=requests,
        stages=stages,
        intervals=intervals,
        attacker_cmp=attacker_cmp,
        stage_cmp=stage_cmp,
        capacity_df=capacity_df,
        blackbox=blackbox,
        scenarios=scenarios,
    )
    validation_summary = (
        assertions.groupby("validation_group", as_index=False)
        .agg(
            assertion_count=("assertion_name", "count"),
            passed_count=("passed", "sum"),
        )
    )
    validation_summary["failed_count"] = (
        validation_summary["assertion_count"] - validation_summary["passed_count"]
    )
    validation_summary["pass_rate"] = (
        validation_summary["passed_count"] / validation_summary["assertion_count"]
    )

    print("Saving raw evaluator logs and summary outputs...")
    # Save evaluator-level raw data.
    requests.to_csv(raw_dir / "phase2_03_request_log.csv.gz", index=False, compression=GZIP_COMPRESSION)
    stages.to_csv(raw_dir / "phase2_03_stage_log.csv.gz", index=False, compression=GZIP_COMPRESSION)
    intervals.to_csv(raw_dir / "phase2_03_resource_intervals.csv.gz", index=False, compression=GZIP_COMPRESSION)
    attacker_cmp.to_csv(raw_dir / "phase2_03_attacker_comparison.csv.gz", index=False, compression=GZIP_COMPRESSION)
    stage_cmp.to_csv(raw_dir / "phase2_03_stage_comparison.csv.gz", index=False, compression=GZIP_COMPRESSION)
    victim_cmp.to_csv(raw_dir / "phase2_03_victim_comparison.csv.gz", index=False, compression=GZIP_COMPRESSION)
    blackbox.to_csv(raw_dir / "phase2_03_blackbox_observations.csv.gz", index=False, compression=GZIP_COMPRESSION)

    configuration_table(scenarios, workloads).to_csv(
        output_dir / "phase2_03_configuration_table.csv", index=False
    )
    pd.DataFrame(phase_rows).to_csv(
        output_dir / "phase2_03_trial_phase_schedule.csv", index=False
    )
    trial_df.to_csv(output_dir / "phase2_03_trial_summary.csv", index=False)
    trace_key.to_csv(output_dir / "phase2_03_trace_key.csv", index=False)
    fingerprint_metrics.to_csv(
        output_dir / "phase2_03_workload_fingerprint_metrics.csv", index=False
    )
    fingerprint_predictions.to_csv(
        output_dir / "phase2_03_workload_fingerprint_predictions.csv", index=False
    )
    utilization.to_csv(
        output_dir / "phase2_03_resource_utilization_trial.csv", index=False
    )
    capacity_df.to_csv(
        output_dir / "phase2_03_resource_capacity_trial.csv", index=False
    )
    reuse.to_csv(output_dir / "phase2_03_interconnect_reuse_events.csv", index=False)

    output_names = {
        "scenario_summary": "phase2_03_scenario_summary.csv",
        "workload_summary": "phase2_03_workload_summary.csv",
        "delay_decomposition_summary": "phase2_03_delay_decomposition_summary.csv",
        "blocking_attribution_summary": "phase2_03_blocking_attribution_summary.csv",
        "resource_utilization_summary": "phase2_03_resource_utilization_summary.csv",
        "resource_capacity_summary": "phase2_03_resource_capacity_summary.csv",
        "interconnect_reuse_summary": "phase2_03_interconnect_reuse_summary.csv",
        "blackbox_trace_summary": "phase2_03_blackbox_trace_summary.csv",
    }
    for key, filename in output_names.items():
        aggregates[key].to_csv(output_dir / filename, index=False)

    assertions.to_csv(output_dir / "phase2_03_validation_assertions.csv", index=False)
    validation_summary.to_csv(
        output_dir / "phase2_03_validation_summary.csv", index=False
    )

    manifest = {
        "experiment": "Phase 2.3 — Link/Switch Pipeline Contention",
        "output_directory": str(output_dir),
        "seed": args.seed,
        "trials_per_scenario_workload": args.trials,
        "scenario_count": len(scenarios),
        "workload_count": len(workloads),
        "controlled_trial_tuple_count": total,
        "nominal_remote_operation_latency_ns": float(sum(STAGE_DURATIONS_NS.values())),
        "attacker_period_ns": ATTACKER_PERIOD_NS,
        "observation_window_ns": OBSERVATION_WINDOW_NS,
        "interconnect_components": list(INTERCONNECT_COMPONENTS),
        "capacity_interpretation": {
            "shared_pool_capacity_1": "one simultaneous physical lane/path/channel",
            "shared_pool_capacity_2": "two simultaneous pooled physical lanes/paths/channels",
            "partitioned_capacity_2": "two lanes with one statically assigned per tenant",
        },
        "request_record_count": int(len(requests)),
        "stage_record_count": int(len(stages)),
        "resource_interval_count": int(len(intervals)),
        "blackbox_probe_record_count": int(len(blackbox)),
        "validation_assertion_count": int(len(assertions)),
        "passed_assertions": int(assertions["passed"].sum()),
        "failed_assertions": int((~assertions["passed"]).sum()),
        "all_validations_passed": bool(assertions["passed"].all()),
    }
    with (output_dir / "phase2_03_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.no_plots:
        save_plots(
            scenario_summary=aggregates["scenario_summary"],
            workload_summary=aggregates["workload_summary"],
            fingerprint_metrics=fingerprint_metrics,
            output_dir=plot_dir,
        )

    print("\nValidation summary:")
    print(validation_summary.to_string(index=False))
    print("\nHighest-leakage configurations:")
    print(
        aggregates["scenario_summary"]
        .sort_values("mean_excess_turnaround_ns", ascending=False)
        .head(10)[
            [
                "scenario_name",
                "sharing_mode",
                "component_capacity",
                "mean_affected_probe_fraction",
                "mean_excess_turnaround_ns",
                "mean_victim_latency_slowdown",
                "workload_fingerprint_accuracy",
            ]
        ]
        .to_string(index=False)
    )
    print(f"\nResults saved to: {output_dir}")

    if FAIL_ON_VALIDATION_ERROR and not bool(assertions["passed"].all()):
        failed = assertions[~assertions["passed"]]
        print("\nFailed validations:")
        print(failed.to_string(index=False))
        raise SystemExit(1)

    print("\nAll Phase 2.3 link/switch validations passed.")


if __name__ == "__main__":
    main()
