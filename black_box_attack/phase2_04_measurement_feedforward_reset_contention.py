#!/usr/bin/env python3
"""
Phase 2.4 — Measurement, Feedforward, and Reset Contention
===========================================================

Purpose
-------
Identify secondary remote-operation timing bottlenecks after the endpoint and
interconnect have been decomposed in Phases 2.1–2.3.

This experiment explicitly models:
  * a concrete shared inter-module link channel,
  * measurement/readout resources,
  * classical feedforward resources,
  * conditional-control resources,
  * post-completion reset/recovery resources,
  * communication-qubit reuse after reset.

The main measurement-based remote primitive completes *before* reset finishes.
Reset is therefore not on the current operation's critical path, but the
communication qubit remains occupied until reset/recovery completes.  A reset
collision can consequently delay a later request without changing the current
request's completion time.  This creates a reuse fingerprint distinct from
direct queueing at measurement/feedforward stages.

Black-box boundary
------------------
The attacker-visible output contains only its own request release/completion and
turnaround timing.  Stage names, physical resource labels, sharing state,
blocking owner, victim workload labels, and root-cause labels are evaluator-only.

Output directory
----------------
blackbox_window_results/phase2/phase2_04_measurement_feedforward_reset_contention/

Usage
-----
    python phase2_04_measurement_feedforward_reset_contention.py
    python phase2_04_measurement_feedforward_reset_contention.py --trials 4

The default sweep uses 10 randomized victim phases per workload/configuration.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Global configuration
# =============================================================================

DEFAULT_TRIALS = 10
DEFAULT_SEED = 2404
OBSERVATION_WINDOW_NS = 20_000.0
ATTACKER_FIRST_RELEASE_NS = 30.0
ATTACKER_PERIOD_NS = 420.0
AFFECTED_THRESHOLD_NS = 1e-9
FLOAT_TOLERANCE_NS = 1e-9
FAIL_ON_VALIDATION_ERROR = True
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 1, "mtime": 1}

OUTPUT_DIR = Path(
    "blackbox_window_results/phase2/"
    "phase2_04_measurement_feedforward_reset_contention"
)

# Main protocol: measurement + feedforward remote primitive.
STAGE_DURATIONS_NS = {
    "communication_qubit_acquisition": 20.0,
    "endpoint_port_acquisition": 10.0,
    "link_transmission": 80.0,
    "receiver_side_operation": 25.0,
    "measurement_readout": 70.0,
    "classical_feedforward": 40.0,
    "conditional_control": 20.0,
    "reset_recovery": 120.0,
}

ENDPOINT_COMPONENTS = (
    "communication_qubit",
    "endpoint_port",
    "receiver_engine",
)

BACKEND_COMPONENTS = (
    "readout_engine",
    "feedforward_engine",
    "conditional_control_engine",
    "reset_engine",
)

STAGE_TO_COMPONENT = {
    "communication_qubit_acquisition": "communication_qubit",
    "endpoint_port_acquisition": "endpoint_port",
    "link_transmission": "link_channel",
    "receiver_side_operation": "receiver_engine",
    "measurement_readout": "readout_engine",
    "classical_feedforward": "feedforward_engine",
    "conditional_control": "conditional_control_engine",
    "reset_recovery": "reset_engine",
}

CRITICAL_STAGES = (
    "communication_qubit_acquisition",
    "endpoint_port_acquisition",
    "link_transmission",
    "receiver_side_operation",
    "measurement_readout",
    "classical_feedforward",
    "conditional_control",
)
POSTCOMPLETION_STAGES = ("reset_recovery",)

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
    external_completion_here: bool = False
    critical_path: bool = True
    description: str = ""


@dataclass(frozen=True)
class BackendScenario:
    scenario_name: str
    link_capacity: int
    shared_backend_components: tuple[str, ...]
    backend_capacity: int
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
    stage_phase: str
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
    external_completion_ns: float
    cleanup_completion_ns: float
    external_turnaround_ns: float
    cleanup_turnaround_ns: float
    critical_wait_ns: float
    postcompletion_wait_ns: float
    completed_externally: bool
    cleanup_completed: bool


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
    external_completed: bool = False
    cleanup_completed: bool = False
    external_completion_ns: float = math.nan
    cleanup_completion_ns: float = math.nan
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
# Protocol definitions
# =============================================================================


def build_measurement_feedforward_pipeline() -> tuple[StageDefinition, ...]:
    """Measurement-based remote primitive with post-completion reset."""
    return (
        StageDefinition(
            "communication_qubit_acquisition",
            STAGE_DURATIONS_NS["communication_qubit_acquisition"],
            acquire_held=("communication_qubit",),
            description="Acquire the tenant-dedicated communication qubit.",
        ),
        StageDefinition(
            "endpoint_port_acquisition",
            STAGE_DURATIONS_NS["endpoint_port_acquisition"],
            acquire_held=("endpoint_port",),
            require_held=("communication_qubit",),
            description="Acquire the tenant-dedicated interconnect-facing port.",
        ),
        StageDefinition(
            "link_transmission",
            STAGE_DURATIONS_NS["link_transmission"],
            acquire_scoped=("link_channel",),
            require_held=("communication_qubit", "endpoint_port"),
            release_held=("endpoint_port",),
            description="Execute the inter-module transfer on a concrete link lane.",
        ),
        StageDefinition(
            "receiver_side_operation",
            STAGE_DURATIONS_NS["receiver_side_operation"],
            acquire_scoped=("receiver_engine",),
            require_held=("communication_qubit",),
            description="Execute the tenant-dedicated receiver-side operation.",
        ),
        StageDefinition(
            "measurement_readout",
            STAGE_DURATIONS_NS["measurement_readout"],
            acquire_scoped=("readout_engine",),
            require_held=("communication_qubit",),
            description="Measure/read out the protocol result.",
        ),
        StageDefinition(
            "classical_feedforward",
            STAGE_DURATIONS_NS["classical_feedforward"],
            acquire_scoped=("feedforward_engine",),
            require_held=("communication_qubit",),
            description="Process and deliver the classical protocol result.",
        ),
        StageDefinition(
            "conditional_control",
            STAGE_DURATIONS_NS["conditional_control"],
            acquire_scoped=("conditional_control_engine",),
            require_held=("communication_qubit",),
            external_completion_here=True,
            description=(
                "Apply the conditional receiver-side action. The remote operation "
                "is externally complete at the end of this stage."
            ),
        ),
        StageDefinition(
            "reset_recovery",
            STAGE_DURATIONS_NS["reset_recovery"],
            acquire_scoped=("reset_engine",),
            require_held=("communication_qubit",),
            release_held=("communication_qubit",),
            critical_path=False,
            description=(
                "Post-completion reset/recovery. It does not extend the current "
                "operation's completion time, but the communication qubit remains "
                "reserved until this stage ends."
            ),
        ),
    )


def build_coherent_control_pipeline() -> tuple[StageDefinition, ...]:
    """Negative protocol control: no measurement or feedforward stages."""
    return (
        StageDefinition(
            "communication_qubit_acquisition",
            STAGE_DURATIONS_NS["communication_qubit_acquisition"],
            acquire_held=("communication_qubit",),
        ),
        StageDefinition(
            "endpoint_port_acquisition",
            STAGE_DURATIONS_NS["endpoint_port_acquisition"],
            acquire_held=("endpoint_port",),
            require_held=("communication_qubit",),
        ),
        StageDefinition(
            "link_transmission",
            STAGE_DURATIONS_NS["link_transmission"],
            acquire_scoped=("link_channel",),
            require_held=("communication_qubit", "endpoint_port"),
            release_held=("endpoint_port",),
        ),
        StageDefinition(
            "receiver_side_operation",
            STAGE_DURATIONS_NS["receiver_side_operation"],
            acquire_scoped=("receiver_engine",),
            require_held=("communication_qubit",),
            external_completion_here=True,
        ),
        StageDefinition(
            "reset_recovery",
            STAGE_DURATIONS_NS["reset_recovery"],
            acquire_scoped=("reset_engine",),
            require_held=("communication_qubit",),
            release_held=("communication_qubit",),
            critical_path=False,
        ),
    )


# =============================================================================
# Scenarios and workloads
# =============================================================================


def _scenario(
    name: str,
    *,
    link_capacity: int,
    shared: tuple[str, ...] = (),
    backend_capacity: int = 1,
    description: str,
    mechanism: str,
    family: str,
) -> BackendScenario:
    return BackendScenario(
        scenario_name=name,
        link_capacity=link_capacity,
        shared_backend_components=shared,
        backend_capacity=backend_capacity,
        description=description,
        expected_mechanism=mechanism,
        scenario_family=family,
    )


def build_scenarios() -> tuple[BackendScenario, ...]:
    scenarios: list[BackendScenario] = []
    for link_capacity in (1, 2):
        suffix = f"link{link_capacity}"
        scenarios.extend(
            [
                _scenario(
                    f"isolated_backend_{suffix}",
                    link_capacity=link_capacity,
                    description="All measurement/feedforward/reset resources are tenant-dedicated.",
                    mechanism="link_only" if link_capacity == 1 else "fully_isolated",
                    family="isolated_backend",
                ),
                _scenario(
                    f"shared_readout_capacity1_{suffix}",
                    link_capacity=link_capacity,
                    shared=("readout_engine",),
                    backend_capacity=1,
                    description="One cross-tenant readout/measurement lane.",
                    mechanism="readout_queueing",
                    family="readout",
                ),
                _scenario(
                    f"shared_readout_capacity2_{suffix}",
                    link_capacity=link_capacity,
                    shared=("readout_engine",),
                    backend_capacity=2,
                    description="Two pooled readout/measurement lanes.",
                    mechanism="readout_two_lane_pool",
                    family="readout",
                ),
                _scenario(
                    f"shared_feedforward_capacity1_{suffix}",
                    link_capacity=link_capacity,
                    shared=("feedforward_engine",),
                    backend_capacity=1,
                    description="One cross-tenant classical feedforward lane.",
                    mechanism="feedforward_queueing",
                    family="feedforward",
                ),
                _scenario(
                    f"shared_feedforward_capacity2_{suffix}",
                    link_capacity=link_capacity,
                    shared=("feedforward_engine",),
                    backend_capacity=2,
                    description="Two pooled classical feedforward lanes.",
                    mechanism="feedforward_two_lane_pool",
                    family="feedforward",
                ),
                _scenario(
                    f"shared_conditional_control_capacity1_{suffix}",
                    link_capacity=link_capacity,
                    shared=("conditional_control_engine",),
                    backend_capacity=1,
                    description="One cross-tenant conditional-control lane.",
                    mechanism="conditional_control_queueing",
                    family="conditional_control",
                ),
                _scenario(
                    f"shared_reset_capacity1_{suffix}",
                    link_capacity=link_capacity,
                    shared=("reset_engine",),
                    backend_capacity=1,
                    description="One shared post-completion reset/recovery engine.",
                    mechanism="reset_reuse",
                    family="reset",
                ),
                _scenario(
                    f"shared_reset_capacity2_{suffix}",
                    link_capacity=link_capacity,
                    shared=("reset_engine",),
                    backend_capacity=2,
                    description="Two pooled post-completion reset/recovery engines.",
                    mechanism="reset_two_lane_pool",
                    family="reset",
                ),
                _scenario(
                    f"shared_measurement_feedback_stack_capacity1_{suffix}",
                    link_capacity=link_capacity,
                    shared=(
                        "readout_engine",
                        "feedforward_engine",
                        "conditional_control_engine",
                    ),
                    backend_capacity=1,
                    description="Shared measurement, feedforward, and conditional-control stack.",
                    mechanism="measurement_feedback_stack",
                    family="measurement_feedback_stack",
                ),
                _scenario(
                    f"shared_full_backend_capacity1_{suffix}",
                    link_capacity=link_capacity,
                    shared=BACKEND_COMPONENTS,
                    backend_capacity=1,
                    description="All back-end measurement/feedforward/reset resources shared.",
                    mechanism="full_backend",
                    family="full_backend",
                ),
                _scenario(
                    f"shared_full_backend_capacity2_{suffix}",
                    link_capacity=link_capacity,
                    shared=BACKEND_COMPONENTS,
                    backend_capacity=2,
                    description="Two pooled lanes for every back-end resource.",
                    mechanism="full_backend_two_lane_pool",
                    family="full_backend",
                ),
            ]
        )
    return tuple(scenarios)


def build_workloads() -> tuple[VictimWorkload, ...]:
    return (
        VictimWorkload(
            "sparse_periodic",
            "Low-rate remote operations with broad spacing.",
            "periodic_sparse",
        ),
        VictimWorkload(
            "dense_periodic",
            "Higher-rate periodic communication near the attacker's probe rate.",
            "periodic_dense",
        ),
        VictimWorkload(
            "synchronization_bursty",
            "Layered groups of remote operations around synchronization phases.",
            "synchronization_bursty",
        ),
    )


def attacker_specs(trial_id: int, workload_name: str) -> list[RequestSpec]:
    releases = np.arange(
        ATTACKER_FIRST_RELEASE_NS,
        OBSERVATION_WINDOW_NS,
        ATTACKER_PERIOD_NS,
        dtype=float,
    )
    return [
        RequestSpec(
            request_id=f"attacker::{trial_id}::{index}",
            tenant="attacker",
            ready_ns=float(release),
            request_index=index,
            workload_name=workload_name,
            trial_id=trial_id,
        )
        for index, release in enumerate(releases)
    ]


def victim_release_times(
    workload: VictimWorkload,
    *,
    phase_ns: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if workload.release_pattern == "periodic_sparse":
        period = 900.0
        releases = np.arange(phase_ns, OBSERVATION_WINDOW_NS, period)
    elif workload.release_pattern == "periodic_dense":
        period = 470.0
        releases = np.arange(phase_ns, OBSERVATION_WINDOW_NS, period)
    elif workload.release_pattern == "synchronization_bursty":
        # Deliberately clustered release phases. Endpoint serialization may spread
        # individual operations, but the shared measurement/feedforward resources
        # still see repeated synchronization epochs.
        burst_period = 1_650.0
        burst_offsets = np.array([0.0, 75.0, 150.0])
        rows: list[float] = []
        base = phase_ns
        while base < OBSERVATION_WINDOW_NS:
            rows.extend(float(base + offset) for offset in burst_offsets)
            base += burst_period
        releases = np.array(rows, dtype=float)
    else:
        raise ValueError(workload.release_pattern)

    releases = releases[(releases >= 0.0) & (releases < OBSERVATION_WINDOW_NS)]
    # Tiny deterministic per-trial perturbation prevents accidental perfect phase
    # locking while preserving workload structure.
    if len(releases):
        jitter = rng.uniform(-3.0, 3.0, size=len(releases))
        releases = np.maximum(0.0, releases + jitter)
    return np.sort(releases)


def victim_specs(
    workload: VictimWorkload,
    trial_id: int,
    phase_ns: float,
    rng: np.random.Generator,
) -> list[RequestSpec]:
    releases = victim_release_times(workload, phase_ns=phase_ns, rng=rng)
    return [
        RequestSpec(
            request_id=f"victim::{trial_id}::{index}",
            tenant="victim",
            ready_ns=float(release),
            request_index=index,
            workload_name=workload.workload_name,
            trial_id=trial_id,
        )
        for index, release in enumerate(releases)
    ]


# =============================================================================
# Resource resolution and event-driven simulation
# =============================================================================


class ResourceResolver:
    def __init__(self, scenario: BackendScenario) -> None:
        self.scenario = scenario

    def candidates(self, tenant: str, component: str) -> tuple[str, ...]:
        if component in ENDPOINT_COMPONENTS:
            return (f"{tenant}::endpoint::{component}::lane0",)
        if component == "link_channel":
            return tuple(
                f"shared::link_channel::lane{lane}"
                for lane in range(self.scenario.link_capacity)
            )
        if component in BACKEND_COMPONENTS:
            if component not in self.scenario.shared_backend_components:
                return (f"{tenant}::backend::{component}::lane0",)
            return tuple(
                f"shared::{component}::lane{lane}"
                for lane in range(self.scenario.backend_capacity)
            )
        raise KeyError(component)

    def metadata(self, tenant: str, component: str) -> tuple[str, int, bool]:
        if component in ENDPOINT_COMPONENTS:
            return "dedicated", 1, False
        if component == "link_channel":
            return "shared_pool", self.scenario.link_capacity, True
        if component in BACKEND_COMPONENTS:
            if component in self.scenario.shared_backend_components:
                return "shared_pool", self.scenario.backend_capacity, True
            return "dedicated", 1, False
        raise KeyError(component)


class BackendPipelineSimulator:
    """FCFS, work-conserving staged simulator with post-completion cleanup."""

    def __init__(
        self,
        *,
        scenario: BackendScenario,
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
            mode, capacity, cross_tenant = self.resolver.metadata(tenant, component)
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
        free = [s for s in self._candidate_states(tenant, component) if s.free]
        if not free:
            return None
        return sorted(free, key=lambda s: s.physical_resource)[0]

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
                f"{runtime.spec.request_id} cannot release {physical}; owner={state.owner_request_id}"
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
                    f"{runtime.spec.request_id} entered {stage.stage_name} without held {component}"
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
                    (state.physical_resource, component, str(state.owner_tenant))
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
        end_ns = now_ns + stage.duration_ns
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
                stage_phase="critical_path" if stage.critical_path else "postcompletion_cleanup",
                stage_ready_ns=float(runtime.stage_ready_ns),
                stage_start_ns=float(now_ns),
                stage_end_ns=float(end_ns),
                wait_ns=float(now_ns - runtime.stage_ready_ns),
                service_ns=float(stage.duration_ns),
                physical_acquired_resources=";".join(acquired),
                physical_required_resources=";".join(required),
                physical_released_resources=";".join(
                    runtime.held_resources[c][0] for c in stage.release_held
                ),
                blocking_physical_resources=";".join(x[0] for x in blockers),
                blocking_components=";".join(sorted({x[1] for x in blockers})),
                blocking_owner_tenants=";".join(sorted({x[2] for x in blockers})),
                cross_tenant_blocked=any(
                    x[2] not in ("None", runtime.spec.tenant) for x in blockers
                ),
                self_blocked=any(x[2] == runtime.spec.tenant for x in blockers),
            )
        )
        runtime.blocking_observations.clear()
        runtime.active = True
        self._event_sequence += 1
        heapq.heappush(
            self._active_heap,
            (
                end_ns,
                self._event_sequence,
                runtime.spec.request_id,
                stage_index,
                tuple(p for _, p in scoped),
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

        if stage.external_completion_here:
            runtime.external_completed = True
            runtime.external_completion_ns = float(now_ns)

        runtime.active = False
        runtime.stage_index += 1
        runtime.stage_ready_ns = float(now_ns)

        if runtime.stage_index == len(self.pipeline):
            if runtime.held_resources:
                raise RuntimeError(
                    f"Cleanup finished while holding resources: {runtime.held_resources}"
                )
            if not runtime.external_completed:
                # Defensive fallback for a protocol whose last stage is the
                # externally visible completion point.
                runtime.external_completed = True
                runtime.external_completion_ns = float(now_ns)
            runtime.cleanup_completed = True
            runtime.cleanup_completion_ns = float(now_ns)

    @staticmethod
    def _candidate_sort_key(runtime: RequestRuntime) -> tuple[float, int, int, str]:
        # Victim wins exact ties. This deterministic rule is evaluator-side and
        # keeps contention reproducible.
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
            spec.request_id: RequestRuntime(spec=spec, stage_ready_ns=float(spec.ready_ns))
            for spec in specs
        }
        now_ns = min((s.ready_ns for s in specs), default=0.0)
        cleanup_count = 0

        while cleanup_count < len(runtimes):
            while self._active_heap and self._active_heap[0][0] <= now_ns + FLOAT_TOLERANCE_NS:
                end_ns, _, request_id, stage_index, scoped = heapq.heappop(self._active_heap)
                runtime = runtimes[request_id]
                was_cleanup = runtime.cleanup_completed
                self._complete_stage(
                    runtime=runtime,
                    stage_index=stage_index,
                    now_ns=float(end_ns),
                    scoped_physical=scoped,
                )
                if runtime.cleanup_completed and not was_cleanup:
                    cleanup_count += 1

            made_progress = True
            while made_progress:
                made_progress = False
                candidates = sorted(
                    (
                        rt
                        for rt in runtimes.values()
                        if not rt.active
                        and not rt.cleanup_completed
                        and rt.stage_ready_ns <= now_ns + FLOAT_TOLERANCE_NS
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

            if cleanup_count == len(runtimes):
                break

            next_times: list[float] = []
            if self._active_heap:
                next_times.append(float(self._active_heap[0][0]))
            future_ready = [
                rt.stage_ready_ns
                for rt in runtimes.values()
                if not rt.active
                and not rt.cleanup_completed
                and rt.stage_ready_ns > now_ns + FLOAT_TOLERANCE_NS
            ]
            if future_ready:
                next_times.append(float(min(future_ready)))
            if not next_times:
                blocked = [rt.spec.request_id for rt in runtimes.values() if not rt.cleanup_completed]
                raise RuntimeError(f"Simulation deadlock: {blocked}")
            next_now = min(next_times)
            if next_now <= now_ns + FLOAT_TOLERANCE_NS:
                raise RuntimeError("Event loop failed to advance")
            now_ns = next_now

        request_rows: list[dict[str, object]] = []
        stage_rows: list[dict[str, object]] = []
        for runtime in runtimes.values():
            critical_wait = sum(
                r.wait_ns for r in runtime.stage_records if r.stage_phase == "critical_path"
            )
            post_wait = sum(
                r.wait_ns
                for r in runtime.stage_records
                if r.stage_phase == "postcompletion_cleanup"
            )
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
                        external_completion_ns=float(runtime.external_completion_ns),
                        cleanup_completion_ns=float(runtime.cleanup_completion_ns),
                        external_turnaround_ns=float(runtime.external_completion_ns - runtime.spec.ready_ns),
                        cleanup_turnaround_ns=float(runtime.cleanup_completion_ns - runtime.spec.ready_ns),
                        critical_wait_ns=float(critical_wait),
                        postcompletion_wait_ns=float(post_wait),
                        completed_externally=runtime.external_completed,
                        cleanup_completed=runtime.cleanup_completed,
                    )
                )
            )
            stage_rows.extend(asdict(r) for r in runtime.stage_records)

        return (
            pd.DataFrame(request_rows),
            pd.DataFrame(stage_rows),
            pd.DataFrame(asdict(x) for x in self.resource_intervals),
        )


# =============================================================================
# Paired-run helpers
# =============================================================================


def simulate(
    *,
    scenario: BackendScenario,
    pipeline: tuple[StageDefinition, ...],
    run_kind: str,
    workload_name: str,
    trial_id: int,
    specs: list[RequestSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return BackendPipelineSimulator(
        scenario=scenario,
        pipeline=pipeline,
        run_kind=run_kind,
        workload_name=workload_name,
        trial_id=trial_id,
    ).run(specs)


def paired_attacker_trace(
    attacker_only_requests: pd.DataFrame,
    combined_requests: pd.DataFrame,
) -> pd.DataFrame:
    a = attacker_only_requests[attacker_only_requests["tenant"] == "attacker"].copy()
    c = combined_requests[combined_requests["tenant"] == "attacker"].copy()
    keep_a = [
        "request_index",
        "ready_ns",
        "external_completion_ns",
        "external_turnaround_ns",
    ]
    keep_c = [
        "request_index",
        "external_completion_ns",
        "external_turnaround_ns",
    ]
    out = a[keep_a].merge(c[keep_c], on="request_index", suffixes=("_attacker_only", "_combined"))
    out = out.rename(
        columns={
            "ready_ns": "release_ns",
            "external_completion_ns_attacker_only": "attacker_only_completion_ns",
            "external_completion_ns_combined": "combined_completion_ns",
            "external_turnaround_ns_attacker_only": "attacker_only_turnaround_ns",
            "external_turnaround_ns_combined": "combined_turnaround_ns",
        }
    )
    out["excess_turnaround_ns"] = (
        out["combined_turnaround_ns"] - out["attacker_only_turnaround_ns"]
    )
    out["affected"] = out["excess_turnaround_ns"] > AFFECTED_THRESHOLD_NS
    out = out.rename(columns={"request_index": "probe_index"})
    return out


def attacker_stage_table(stage_df: pd.DataFrame) -> pd.DataFrame:
    return stage_df[stage_df["tenant"] == "attacker"].copy()


def per_probe_causal_features(
    trace: pd.DataFrame,
    attacker_combined_stages: pd.DataFrame,
) -> pd.DataFrame:
    stages = attacker_combined_stages.copy()
    stages["cross_tenant_wait_ns"] = np.where(
        stages["cross_tenant_blocked"], stages["wait_ns"], 0.0
    )
    stages["self_wait_ns"] = np.where(stages["self_blocked"], stages["wait_ns"], 0.0)

    agg = stages.groupby("request_index", as_index=False).agg(
        critical_path_wait_ns=(
            "wait_ns",
            lambda x: float(x[stages.loc[x.index, "stage_phase"].eq("critical_path")].sum()),
        ),
        postcompletion_wait_ns=(
            "wait_ns",
            lambda x: float(x[stages.loc[x.index, "stage_phase"].eq("postcompletion_cleanup")].sum()),
        ),
        direct_cross_tenant_critical_wait_ns=(
            "cross_tenant_wait_ns",
            lambda x: float(x[stages.loc[x.index, "stage_phase"].eq("critical_path")].sum()),
        ),
        cross_tenant_cleanup_wait_ns=(
            "cross_tenant_wait_ns",
            lambda x: float(x[stages.loc[x.index, "stage_phase"].eq("postcompletion_cleanup")].sum()),
        ),
        self_reuse_wait_ns=(
            "self_wait_ns",
            lambda x: float(
                x[stages.loc[x.index, "stage_name"].eq("communication_qubit_acquisition")].sum()
            ),
        ),
    )
    out = trace.merge(agg, left_on="probe_index", right_on="request_index", how="left")
    out = out.drop(columns=["request_index"], errors="ignore")
    for col in (
        "critical_path_wait_ns",
        "postcompletion_wait_ns",
        "direct_cross_tenant_critical_wait_ns",
        "cross_tenant_cleanup_wait_ns",
        "self_reuse_wait_ns",
    ):
        out[col] = out[col].fillna(0.0)

    # Propagate the previous request's cross-tenant cleanup or critical blocking
    # to current same-tenant communication-qubit reuse waiting.  This captures
    # the causal distinction between direct queueing and delayed reuse.
    prior_cleanup = out["cross_tenant_cleanup_wait_ns"].shift(1, fill_value=0.0)
    prior_direct = out["direct_cross_tenant_critical_wait_ns"].shift(1, fill_value=0.0)
    out["reset_reuse_cascade_ns"] = np.where(
        (out["self_reuse_wait_ns"] > 0.0) & (prior_cleanup > 0.0),
        out["self_reuse_wait_ns"],
        0.0,
    )
    out["other_reuse_cascade_ns"] = np.where(
        (out["self_reuse_wait_ns"] > 0.0)
        & (prior_cleanup <= 0.0)
        & (prior_direct > 0.0),
        out["self_reuse_wait_ns"],
        0.0,
    )
    out["direct_critical_affected"] = out["direct_cross_tenant_critical_wait_ns"] > 0.0
    out["reset_reuse_affected"] = out["reset_reuse_cascade_ns"] > 0.0
    out["other_reuse_affected"] = out["other_reuse_cascade_ns"] > 0.0
    return out


def longest_true_run(values: Iterable[bool]) -> int:
    best = cur = 0
    for value in values:
        if bool(value):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def trace_features(trace: pd.DataFrame) -> dict[str, float]:
    x = trace.sort_values("probe_index")["excess_turnaround_ns"].to_numpy(float)
    affected = x > AFFECTED_THRESHOLD_NS
    positive = x[affected]
    n = len(x)
    thirds = np.array_split(x, 3)
    if n > 1 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
        autocorr1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        autocorr1 = 0.0
    indices = np.flatnonzero(affected)
    if len(indices) >= 2:
        gaps = np.diff(indices).astype(float)
        gap_cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 0.0
    else:
        gap_cv = 0.0
    return {
        "mean_excess_ns": float(np.mean(x)) if n else 0.0,
        "std_excess_ns": float(np.std(x)) if n else 0.0,
        "p95_excess_ns": float(np.percentile(x, 95)) if n else 0.0,
        "max_excess_ns": float(np.max(x)) if n else 0.0,
        "affected_fraction": float(np.mean(affected)) if n else 0.0,
        "positive_mean_ns": float(np.mean(positive)) if len(positive) else 0.0,
        "longest_affected_run": float(longest_true_run(affected)),
        "affected_gap_cv": gap_cv,
        "autocorr_lag1": autocorr1,
        "segment1_mean_ns": float(np.mean(thirds[0])) if len(thirds[0]) else 0.0,
        "segment2_mean_ns": float(np.mean(thirds[1])) if len(thirds[1]) else 0.0,
        "segment3_mean_ns": float(np.mean(thirds[2])) if len(thirds[2]) else 0.0,
    }


# =============================================================================
# Aggregation
# =============================================================================


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) else math.nan


def summarize_trial(
    *,
    scenario: BackendScenario,
    workload: VictimWorkload,
    trial_id: int,
    phase_ns: float,
    trace: pd.DataFrame,
    causal: pd.DataFrame,
    victim_only_req: pd.DataFrame,
    combined_req: pd.DataFrame,
) -> dict[str, object]:
    victim_only = victim_only_req[victim_only_req["tenant"] == "victim"].sort_values("request_index")
    victim_combined = combined_req[combined_req["tenant"] == "victim"].sort_values("request_index")
    joined = victim_only[["request_index", "external_turnaround_ns", "external_completion_ns"]].merge(
        victim_combined[["request_index", "external_turnaround_ns", "external_completion_ns"]],
        on="request_index",
        suffixes=("_victim_only", "_combined"),
    )
    vo_mean = float(joined["external_turnaround_ns_victim_only"].mean()) if len(joined) else math.nan
    vc_mean = float(joined["external_turnaround_ns_combined"].mean()) if len(joined) else math.nan
    if len(joined):
        vo_makespan = float(joined["external_completion_ns_victim_only"].max() - victim_only["ready_ns"].min())
        vc_makespan = float(joined["external_completion_ns_combined"].max() - victim_combined["ready_ns"].min())
    else:
        vo_makespan = vc_makespan = math.nan

    feat = trace_features(trace)
    return {
        "scenario_name": scenario.scenario_name,
        "scenario_family": scenario.scenario_family,
        "link_capacity": scenario.link_capacity,
        "shared_backend_components": ";".join(scenario.shared_backend_components),
        "backend_capacity": scenario.backend_capacity,
        "workload_name": workload.workload_name,
        "trial_id": trial_id,
        "victim_phase_ns": phase_ns,
        "probe_count": int(len(trace)),
        "affected_probe_count": int(trace["affected"].sum()),
        "affected_probe_fraction": float(trace["affected"].mean()),
        "mean_excess_turnaround_ns": float(trace["excess_turnaround_ns"].mean()),
        "p95_excess_turnaround_ns": float(trace["excess_turnaround_ns"].quantile(0.95)),
        "max_excess_turnaround_ns": float(trace["excess_turnaround_ns"].max()),
        "direct_critical_affected_fraction": float(causal["direct_critical_affected"].mean()),
        "reset_reuse_affected_fraction": float(causal["reset_reuse_affected"].mean()),
        "other_reuse_affected_fraction": float(causal["other_reuse_affected"].mean()),
        "mean_direct_cross_tenant_critical_wait_ns": float(causal["direct_cross_tenant_critical_wait_ns"].mean()),
        "mean_reset_reuse_cascade_ns": float(causal["reset_reuse_cascade_ns"].mean()),
        "mean_postcompletion_wait_ns": float(causal["postcompletion_wait_ns"].mean()),
        "longest_affected_run": int(feat["longest_affected_run"]),
        "affected_gap_cv": feat["affected_gap_cv"],
        "autocorr_lag1": feat["autocorr_lag1"],
        "victim_mean_request_slowdown": safe_ratio(vc_mean, vo_mean),
        "victim_makespan_slowdown": safe_ratio(vc_makespan, vo_makespan),
    }


def scenario_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return trials.groupby(
        [
            "scenario_name",
            "scenario_family",
            "link_capacity",
            "shared_backend_components",
            "backend_capacity",
        ],
        as_index=False,
    ).agg(
        trial_count=("trial_id", "count"),
        mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
        mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
        mean_p95_excess_turnaround_ns=("p95_excess_turnaround_ns", "mean"),
        max_excess_turnaround_ns=("max_excess_turnaround_ns", "max"),
        mean_direct_critical_affected_fraction=("direct_critical_affected_fraction", "mean"),
        mean_reset_reuse_affected_fraction=("reset_reuse_affected_fraction", "mean"),
        mean_direct_cross_tenant_critical_wait_ns=("mean_direct_cross_tenant_critical_wait_ns", "mean"),
        mean_reset_reuse_cascade_ns=("mean_reset_reuse_cascade_ns", "mean"),
        mean_postcompletion_wait_ns=("mean_postcompletion_wait_ns", "mean"),
        mean_victim_request_slowdown=("victim_mean_request_slowdown", "mean"),
        mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
        mean_longest_affected_run=("longest_affected_run", "mean"),
        mean_affected_gap_cv=("affected_gap_cv", "mean"),
    )


def workload_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return trials.groupby(
        ["scenario_name", "link_capacity", "workload_name"], as_index=False
    ).agg(
        trial_count=("trial_id", "count"),
        mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
        mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
        mean_direct_critical_affected_fraction=("direct_critical_affected_fraction", "mean"),
        mean_reset_reuse_affected_fraction=("reset_reuse_affected_fraction", "mean"),
        mean_victim_request_slowdown=("victim_mean_request_slowdown", "mean"),
        mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
        mean_longest_affected_run=("longest_affected_run", "mean"),
    )


def delay_decomposition(stage_df: pd.DataFrame) -> pd.DataFrame:
    a = stage_df[(stage_df["run_kind"] == "combined") & (stage_df["tenant"] == "attacker")].copy()
    a["cross_tenant_wait_ns"] = np.where(a["cross_tenant_blocked"], a["wait_ns"], 0.0)
    a["self_wait_ns"] = np.where(a["self_blocked"], a["wait_ns"], 0.0)
    return a.groupby(
        ["scenario_name", "workload_name", "stage_phase", "stage_name", "causal_component"],
        as_index=False,
    ).agg(
        stage_observations=("request_id", "count"),
        mean_wait_ns=("wait_ns", "mean"),
        max_wait_ns=("wait_ns", "max"),
        total_wait_ns=("wait_ns", "sum"),
        cross_tenant_block_fraction=("cross_tenant_blocked", "mean"),
        self_block_fraction=("self_blocked", "mean"),
        total_cross_tenant_wait_ns=("cross_tenant_wait_ns", "sum"),
        total_self_wait_ns=("self_wait_ns", "sum"),
    )


def blocking_attribution(stage_df: pd.DataFrame) -> pd.DataFrame:
    a = stage_df[
        (stage_df["run_kind"] == "combined")
        & (stage_df["tenant"] == "attacker")
        & (stage_df["wait_ns"] > AFFECTED_THRESHOLD_NS)
    ].copy()
    if a.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "workload_name",
                "stage_phase",
                "stage_name",
                "causal_component",
                "blocking_components",
                "cross_tenant_blocked",
                "self_blocked",
                "event_count",
                "total_wait_ns",
                "mean_wait_ns",
            ]
        )
    return a.groupby(
        [
            "scenario_name",
            "workload_name",
            "stage_phase",
            "stage_name",
            "causal_component",
            "blocking_components",
            "cross_tenant_blocked",
            "self_blocked",
        ],
        as_index=False,
    ).agg(
        event_count=("request_id", "count"),
        total_wait_ns=("wait_ns", "sum"),
        mean_wait_ns=("wait_ns", "mean"),
    )


def resource_utilization(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    x = intervals[intervals["run_kind"] == "combined"].copy()
    span = max(OBSERVATION_WINDOW_NS, float(x["end_ns"].max()))
    grouped = x.groupby(
        [
            "scenario_name",
            "workload_name",
            "logical_component",
            "physical_resource",
            "sharing_mode",
            "component_capacity",
            "cross_tenant_capable",
        ],
        as_index=False,
    ).agg(
        occupancy_ns=("occupancy_ns", "sum"),
        interval_count=("request_id", "count"),
        tenant_count=("tenant", "nunique"),
    )
    grouped["utilization_fraction"] = grouped["occupancy_ns"] / span
    return grouped


def endpoint_reuse_summary(stage_df: pd.DataFrame, request_df: pd.DataFrame) -> pd.DataFrame:
    stages = stage_df[(stage_df["run_kind"] == "combined") & (stage_df["tenant"] == "attacker")].copy()
    comm = stages[stages["stage_name"] == "communication_qubit_acquisition"].copy()
    reset = stages[stages["stage_name"] == "reset_recovery"].copy()
    comm = comm[["scenario_name", "workload_name", "trial_id", "request_index", "wait_ns", "self_blocked"]].rename(
        columns={"wait_ns": "communication_qubit_reuse_wait_ns"}
    )
    reset = reset[[
        "scenario_name",
        "workload_name",
        "trial_id",
        "request_index",
        "wait_ns",
        "cross_tenant_blocked",
    ]].rename(columns={"wait_ns": "reset_wait_ns", "cross_tenant_blocked": "reset_cross_tenant_blocked"})
    merged = comm.merge(reset, on=["scenario_name", "workload_name", "trial_id", "request_index"], how="outer")
    merged = merged.sort_values(["scenario_name", "workload_name", "trial_id", "request_index"])
    merged["prior_reset_wait_ns"] = merged.groupby(
        ["scenario_name", "workload_name", "trial_id"]
    )["reset_wait_ns"].shift(1, fill_value=0.0)
    merged["reset_caused_reuse"] = (
        (merged["communication_qubit_reuse_wait_ns"].fillna(0.0) > 0.0)
        & (merged["prior_reset_wait_ns"].fillna(0.0) > 0.0)
    )
    return merged.groupby(["scenario_name", "workload_name"], as_index=False).agg(
        probe_count=("request_index", "count"),
        reuse_wait_fraction=("communication_qubit_reuse_wait_ns", lambda x: float((x.fillna(0.0) > 0.0).mean())),
        mean_reuse_wait_ns=("communication_qubit_reuse_wait_ns", "mean"),
        max_reuse_wait_ns=("communication_qubit_reuse_wait_ns", "max"),
        reset_cross_tenant_block_fraction=("reset_cross_tenant_blocked", "mean"),
        mean_reset_wait_ns=("reset_wait_ns", "mean"),
        reset_caused_reuse_fraction=("reset_caused_reuse", "mean"),
    )


def temporal_fingerprint_summary(causal_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in causal_df.groupby(["scenario_name", "workload_name"]):
        group = group.sort_values(["trial_id", "probe_index"])
        runs = []
        gap_cvs = []
        for _, t in group.groupby("trial_id"):
            affected = t["affected"].to_numpy(bool)
            runs.append(longest_true_run(affected))
            idx = np.flatnonzero(affected)
            if len(idx) >= 2:
                gaps = np.diff(idx).astype(float)
                gap_cvs.append(float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) else 0.0)
        rows.append(
            {
                "scenario_name": keys[0],
                "workload_name": keys[1],
                "probe_count": int(len(group)),
                "affected_fraction": float(group["affected"].mean()),
                "direct_critical_fraction": float(group["direct_critical_affected"].mean()),
                "reset_reuse_fraction": float(group["reset_reuse_affected"].mean()),
                "mean_excess_ns": float(group["excess_turnaround_ns"].mean()),
                "mean_direct_critical_wait_ns": float(group["direct_cross_tenant_critical_wait_ns"].mean()),
                "mean_reset_reuse_ns": float(group["reset_reuse_cascade_ns"].mean()),
                "mean_longest_affected_run": float(np.mean(runs)) if runs else 0.0,
                "mean_affected_gap_cv": float(np.mean(gap_cvs)) if gap_cvs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_trace_feature_table(
    blackbox: pd.DataFrame,
    trace_key: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trace_id, group in blackbox.groupby("trace_id"):
        key = trace_key[trace_key["trace_id"] == trace_id].iloc[0].to_dict()
        rows.append({**key, **trace_features(group)})
    return pd.DataFrame(rows)


def leave_one_trial_out_nearest_centroid(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "mean_excess_ns",
        "std_excess_ns",
        "p95_excess_ns",
        "max_excess_ns",
        "affected_fraction",
        "positive_mean_ns",
        "longest_affected_run",
        "affected_gap_cv",
        "autocorr_lag1",
        "segment1_mean_ns",
        "segment2_mean_ns",
        "segment3_mean_ns",
    ]
    predictions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    labels = sorted(features["workload_name"].unique())

    for scenario_name, sf in features.groupby("scenario_name"):
        scenario_preds: list[dict[str, object]] = []
        for trial_id in sorted(sf["trial_id"].unique()):
            train = sf[sf["trial_id"] != trial_id]
            test = sf[sf["trial_id"] == trial_id]
            if train.empty or test.empty:
                continue
            mu = train[feature_cols].mean()
            sigma = train[feature_cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
            centroids = {
                label: ((train[train["workload_name"] == label][feature_cols] - mu) / sigma).mean().to_numpy(float)
                for label in labels
            }
            for _, row in test.iterrows():
                vector = ((row[feature_cols] - mu) / sigma).to_numpy(float)
                distances = {label: float(np.linalg.norm(vector - centroid)) for label, centroid in centroids.items()}
                pred = min(distances, key=distances.get)
                item = {
                    "scenario_name": scenario_name,
                    "trial_id": int(trial_id),
                    "trace_id": row["trace_id"],
                    "true_workload": row["workload_name"],
                    "predicted_workload": pred,
                    "correct": pred == row["workload_name"],
                }
                scenario_preds.append(item)
                predictions.append(item)

        if scenario_preds:
            pdf = pd.DataFrame(scenario_preds)
            recalls = []
            f1s = []
            for label in labels:
                tp = int(((pdf["true_workload"] == label) & (pdf["predicted_workload"] == label)).sum())
                fn = int(((pdf["true_workload"] == label) & (pdf["predicted_workload"] != label)).sum())
                fp = int(((pdf["true_workload"] != label) & (pdf["predicted_workload"] == label)).sum())
                recall = tp / (tp + fn) if tp + fn else 0.0
                precision = tp / (tp + fp) if tp + fp else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                recalls.append(recall)
                f1s.append(f1)
            metrics.append(
                {
                    "scenario_name": scenario_name,
                    "sample_count": int(len(pdf)),
                    "class_count": len(labels),
                    "chance_accuracy": 1.0 / len(labels),
                    "accuracy": float(pdf["correct"].mean()),
                    "balanced_accuracy": float(np.mean(recalls)),
                    "macro_f1": float(np.mean(f1s)),
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


# =============================================================================
# Validation
# =============================================================================


def assertion(group: str, name: str, passed: bool, expected: str, observed: object, details: str = "") -> ValidationAssertion:
    return ValidationAssertion(group, name, bool(passed), expected, str(observed), details)


def capacity_overlap_violations(intervals: pd.DataFrame) -> int:
    """Count physical-lane overlaps. Each physical_resource is capacity one."""
    if intervals.empty:
        return 0
    violations = 0
    for _, group in intervals.groupby(["run_kind", "scenario_name", "workload_name", "trial_id", "physical_resource"]):
        g = group.sort_values(["start_ns", "end_ns"])
        prior_end = -math.inf
        for row in g.itertuples():
            if row.start_ns < prior_end - FLOAT_TOLERANCE_NS:
                violations += 1
            prior_end = max(prior_end, row.end_ns)
    return violations


def validate(
    *,
    scenarios: tuple[BackendScenario, ...],
    request_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    interval_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    blackbox_df: pd.DataFrame,
    protocol_control_df: pd.DataFrame,
) -> list[ValidationAssertion]:
    out: list[ValidationAssertion] = []

    out.append(assertion(
        "completion",
        "all_requests_externally_complete",
        bool(request_df["completed_externally"].all()),
        "all True",
        request_df["completed_externally"].value_counts(dropna=False).to_dict(),
    ))
    out.append(assertion(
        "completion",
        "all_cleanup_stages_complete",
        bool(request_df["cleanup_completed"].all()),
        "all True",
        request_df["cleanup_completed"].value_counts(dropna=False).to_dict(),
    ))

    attacker_only = stage_df[(stage_df["run_kind"] == "attacker_only") & (stage_df["tenant"] == "attacker")]
    max_baseline_wait = float(attacker_only["wait_ns"].max()) if len(attacker_only) else 0.0
    out.append(assertion(
        "baseline",
        "attacker_only_has_zero_self_contention",
        max_baseline_wait <= FLOAT_TOLERANCE_NS,
        "max wait = 0 ns",
        max_baseline_wait,
    ))

    iso = trial_df[trial_df["scenario_name"] == "isolated_backend_link2"]
    out.append(assertion(
        "negative_control",
        "overprovisioned_link_isolated_backend_has_zero_leakage",
        bool((iso["affected_probe_fraction"] <= FLOAT_TOLERANCE_NS).all()),
        "affected fraction = 0",
        float(iso["affected_probe_fraction"].max()) if len(iso) else "missing",
    ))

    link1 = trial_df[trial_df["scenario_name"] == "isolated_backend_link1"]
    out.append(assertion(
        "link_control",
        "underprovisioned_link_creates_leakage",
        bool((link1["affected_probe_fraction"] > 0.0).any()),
        "some affected probes",
        float(link1["affected_probe_fraction"].mean()) if len(link1) else "missing",
    ))

    # Overprovisioned-link one-resource controls.
    for component, scenario_name in [
        ("readout_engine", "shared_readout_capacity1_link2"),
        ("feedforward_engine", "shared_feedforward_capacity1_link2"),
        ("conditional_control_engine", "shared_conditional_control_capacity1_link2"),
    ]:
        rows = stage_df[
            (stage_df["scenario_name"] == scenario_name)
            & (stage_df["run_kind"] == "combined")
            & (stage_df["tenant"] == "attacker")
            & (stage_df["cross_tenant_blocked"])
            & (stage_df["wait_ns"] > 0.0)
        ]
        observed_components = set(rows["blocking_components"].dropna().astype(str))
        out.append(assertion(
            "single_resource_attribution",
            f"{component}_is_only_cross_tenant_backend_blocker_with_link2",
            bool(len(rows) > 0 and all(component in x for x in observed_components)),
            component,
            sorted(observed_components),
        ))

    reset_name = "shared_reset_capacity1_link2"
    reset_rows = stage_df[
        (stage_df["scenario_name"] == reset_name)
        & (stage_df["run_kind"] == "combined")
        & (stage_df["tenant"] == "attacker")
        & (stage_df["stage_name"] == "reset_recovery")
    ]
    reset_cross_wait = float(reset_rows.loc[reset_rows["cross_tenant_blocked"], "wait_ns"].sum())
    reset_causal = causal_df[causal_df["scenario_name"] == reset_name]
    reuse_total = float(reset_causal["reset_reuse_cascade_ns"].sum()) if len(reset_causal) else 0.0
    out.append(assertion(
        "reset_reuse",
        "shared_reset_has_cross_tenant_cleanup_contention",
        reset_cross_wait > 0.0,
        "> 0 ns",
        reset_cross_wait,
    ))
    out.append(assertion(
        "reset_reuse",
        "shared_reset_delays_later_probe_reuse",
        reuse_total > 0.0,
        "> 0 ns",
        reuse_total,
    ))

    # Reset is post-completion: current operation completion precedes reset end.
    rr = stage_df[(stage_df["stage_name"] == "reset_recovery") & (stage_df["run_kind"] == "combined")]
    req_lookup = request_df[request_df["run_kind"] == "combined"][
        ["scenario_name", "workload_name", "trial_id", "request_id", "external_completion_ns"]
    ]
    rrj = rr.merge(req_lookup, on=["scenario_name", "workload_name", "trial_id", "request_id"], how="left")
    post_ok = bool((rrj["external_completion_ns"] <= rrj["stage_start_ns"] + FLOAT_TOLERANCE_NS).all())
    out.append(assertion(
        "critical_path",
        "reset_begins_after_external_operation_completion",
        post_ok,
        "external completion <= reset start",
        f"violations={(rrj['external_completion_ns'] > rrj['stage_start_ns'] + FLOAT_TOLERANCE_NS).sum()}",
    ))

    critical_service = sum(STAGE_DURATIONS_NS[s] for s in CRITICAL_STAGES)
    baseline_req = request_df[(request_df["run_kind"] == "attacker_only") & (request_df["tenant"] == "attacker")]
    ext_min = float((baseline_req["external_turnaround_ns"] - critical_service).abs().max()) if len(baseline_req) else 0.0
    out.append(assertion(
        "critical_path",
        "baseline_external_latency_excludes_reset_service",
        ext_min <= FLOAT_TOLERANCE_NS,
        f"external turnaround = {critical_service} ns",
        ext_min,
    ))

    cleanup_extra = request_df[(request_df["run_kind"] == "attacker_only") & (request_df["tenant"] == "attacker")]
    expected_cleanup_extra = STAGE_DURATIONS_NS["reset_recovery"]
    cleanup_diff = (cleanup_extra["cleanup_completion_ns"] - cleanup_extra["external_completion_ns"] - expected_cleanup_extra).abs()
    out.append(assertion(
        "critical_path",
        "baseline_cleanup_extends_past_completion_by_reset_duration",
        bool((cleanup_diff <= FLOAT_TOLERANCE_NS).all()),
        f"{expected_cleanup_extra} ns",
        float(cleanup_diff.max()) if len(cleanup_diff) else 0.0,
    ))

    # Pooled capacity two should remove single-resource cross-tenant backend contention
    # with two tenants when the link is also capacity two.
    for family, scenario_name in [
        ("readout", "shared_readout_capacity2_link2"),
        ("feedforward", "shared_feedforward_capacity2_link2"),
        ("reset", "shared_reset_capacity2_link2"),
        ("full_backend", "shared_full_backend_capacity2_link2"),
    ]:
        rows = trial_df[trial_df["scenario_name"] == scenario_name]
        out.append(assertion(
            "capacity",
            f"{family}_capacity2_with_link2_removes_leakage",
            bool((rows["affected_probe_fraction"] <= FLOAT_TOLERANCE_NS).all()),
            "affected fraction = 0",
            float(rows["affected_probe_fraction"].max()) if len(rows) else "missing",
        ))

    # Endpoints are always tenant-dedicated: no cross-tenant endpoint blocker.
    endpoint_cross = stage_df[
        (stage_df["run_kind"] == "combined")
        & (stage_df["tenant"] == "attacker")
        & (stage_df["cross_tenant_blocked"])
        & (stage_df["blocking_components"].astype(str).str.contains("communication_qubit|endpoint_port|receiver_engine", regex=True))
    ]
    out.append(assertion(
        "isolation",
        "no_cross_tenant_endpoint_blocking",
        endpoint_cross.empty,
        "0 rows",
        len(endpoint_cross),
    ))

    violations = capacity_overlap_violations(interval_df)
    out.append(assertion(
        "resource_calendar",
        "no_illegal_physical_lane_overlap",
        violations == 0,
        "0",
        violations,
    ))

    # External completion delay must equal current request critical-path wait when
    # attacker-only wait is zero. Postcompletion reset wait must not be added to
    # that same request's completion time.
    merged = causal_df.copy()
    err = (merged["excess_turnaround_ns"] - merged["critical_path_wait_ns"]).abs()
    out.append(assertion(
        "causal_accounting",
        "external_delay_equals_critical_path_wait_not_cleanup_wait",
        bool((err <= 1e-7).all()),
        "max absolute error <= 1e-7 ns",
        float(err.max()) if len(err) else 0.0,
    ))

    out.append(assertion(
        "blackbox_boundary",
        "attacker_visible_columns_are_external_only",
        set(blackbox_df.columns).issubset(BLACKBOX_ALLOWED_COLUMNS),
        sorted(BLACKBOX_ALLOWED_COLUMNS),
        sorted(blackbox_df.columns),
    ))

    # Protocol negative control: shared readout/feedforward must not matter for a
    # coherent protocol that never requests those resources.
    if not protocol_control_df.empty:
        p = protocol_control_df[protocol_control_df["control_name"].isin(["coherent_shared_readout", "coherent_shared_feedforward"])]
        out.append(assertion(
            "protocol_dependence",
            "coherent_protocol_ignores_absent_readout_feedforward_resources",
            bool((p["affected_probe_fraction"] <= FLOAT_TOLERANCE_NS).all()),
            "0 affected fraction",
            p[["control_name", "affected_probe_fraction"]].to_dict("records"),
        ))

    return out


# =============================================================================
# Protocol-negative-control mini experiment
# =============================================================================


def run_protocol_controls(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 777)
    pipeline = build_coherent_control_pipeline()
    workload = build_workloads()[0]
    phase = 170.0
    controls = [
        _scenario(
            "coherent_isolated",
            link_capacity=2,
            description="Coherent protocol, all backend resources isolated.",
            mechanism="negative_control",
            family="protocol_control",
        ),
        _scenario(
            "coherent_shared_readout",
            link_capacity=2,
            shared=("readout_engine",),
            backend_capacity=1,
            description="Coherent protocol with a nominally shared readout engine it never uses.",
            mechanism="absent_stage_control",
            family="protocol_control",
        ),
        _scenario(
            "coherent_shared_feedforward",
            link_capacity=2,
            shared=("feedforward_engine",),
            backend_capacity=1,
            description="Coherent protocol with a nominally shared feedforward engine it never uses.",
            mechanism="absent_stage_control",
            family="protocol_control",
        ),
    ]
    rows = []
    for sc in controls:
        a_specs = attacker_specs(0, workload.workload_name)
        v_specs = victim_specs(workload, 0, phase, rng)
        a_req, _, _ = simulate(
            scenario=sc,
            pipeline=pipeline,
            run_kind="attacker_only",
            workload_name=workload.workload_name,
            trial_id=0,
            specs=a_specs,
        )
        c_req, _, _ = simulate(
            scenario=sc,
            pipeline=pipeline,
            run_kind="combined",
            workload_name=workload.workload_name,
            trial_id=0,
            specs=a_specs + v_specs,
        )
        trace = paired_attacker_trace(a_req, c_req)
        rows.append(
            {
                "control_name": sc.scenario_name,
                "probe_count": len(trace),
                "affected_probe_fraction": float(trace["affected"].mean()),
                "mean_excess_turnaround_ns": float(trace["excess_turnaround_ns"].mean()),
                "max_excess_turnaround_ns": float(trace["excess_turnaround_ns"].max()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Main experiment
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 2:
        raise ValueError("Use at least two trials for workload-fingerprint evaluation.")

    outdir: Path = args.output_dir
    rawdir = outdir / "raw"
    plotdir = outdir / "plots"
    rawdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pipeline = build_measurement_feedforward_pipeline()
    scenarios = build_scenarios()
    workloads = build_workloads()

    configuration = pd.DataFrame(
        [
            {
                **asdict(sc),
                "shared_backend_components": ";".join(sc.shared_backend_components),
            }
            for sc in scenarios
        ]
    )
    configuration.to_csv(outdir / "phase2_04_configuration_table.csv", index=False)

    protocol_def = pd.DataFrame(
        [
            {
                "stage_index": i,
                "stage_name": s.stage_name,
                "causal_component": STAGE_TO_COMPONENT[s.stage_name],
                "duration_ns": s.duration_ns,
                "critical_path": s.critical_path,
                "external_completion_here": s.external_completion_here,
                "acquire_held": ";".join(s.acquire_held),
                "acquire_scoped": ";".join(s.acquire_scoped),
                "require_held": ";".join(s.require_held),
                "release_held": ";".join(s.release_held),
                "description": s.description,
            }
            for i, s in enumerate(pipeline)
        ]
    )
    protocol_def.to_csv(outdir / "phase2_04_protocol_definition.csv", index=False)

    # One randomized phase per workload/trial; every architecture scenario sees
    # the same release schedule for paired comparability.
    phase_rows: list[dict[str, object]] = []
    phases: dict[tuple[str, int], float] = {}
    for workload in workloads:
        for trial_id in range(args.trials):
            phase = float(rng.uniform(80.0, 700.0))
            phases[(workload.workload_name, trial_id)] = phase
            phase_rows.append(
                {
                    "workload_name": workload.workload_name,
                    "trial_id": trial_id,
                    "victim_phase_ns": phase,
                }
            )
    pd.DataFrame(phase_rows).to_csv(outdir / "phase2_04_trial_phase_schedule.csv", index=False)

    request_frames: list[pd.DataFrame] = []
    stage_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, object]] = []
    blackbox_frames: list[pd.DataFrame] = []
    causal_frames: list[pd.DataFrame] = []
    trace_key_rows: list[dict[str, object]] = []

    # Attacker-only and victim-only controls are identical across architecture
    # scenarios because only one tenant is present and the selected attacker
    # period has zero self-contention.  Cache them once per workload/trial and
    # relabel scenario_name when pairing with each combined run.
    control_scenario = next(sc for sc in scenarios if sc.scenario_name == "isolated_backend_link2")
    control_cache: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[RequestSpec], list[RequestSpec]]] = {}
    for workload_index, workload in enumerate(workloads):
        for trial_id in range(args.trials):
            phase = phases[(workload.workload_name, trial_id)]
            local_rng = np.random.default_rng(args.seed + 10_000 * trial_id + 100 * workload_index)
            a_specs = attacker_specs(trial_id, workload.workload_name)
            v_specs = victim_specs(workload, trial_id, phase, local_rng)
            a_req, a_stage, a_int = simulate(
                scenario=control_scenario,
                pipeline=pipeline,
                run_kind="attacker_only",
                workload_name=workload.workload_name,
                trial_id=trial_id,
                specs=a_specs,
            )
            v_req, v_stage, v_int = simulate(
                scenario=control_scenario,
                pipeline=pipeline,
                run_kind="victim_only",
                workload_name=workload.workload_name,
                trial_id=trial_id,
                specs=v_specs,
            )
            control_cache[(workload.workload_name, trial_id)] = (
                a_req, a_stage, a_int, v_req, v_stage, v_int, a_specs, v_specs
            )

    total_tuples = len(scenarios) * len(workloads) * args.trials
    done = 0
    for scenario in scenarios:
        for workload in workloads:
            for trial_id in range(args.trials):
                phase = phases[(workload.workload_name, trial_id)]
                (a_req0, a_stage0, a_int0, v_req0, v_stage0, v_int0, a_specs, v_specs) = control_cache[(workload.workload_name, trial_id)]
                # Clone and relabel cached controls so all output tables retain
                # the exact scenario identity used by the paired combined run.
                a_req, a_stage, a_int = (x.copy() for x in (a_req0, a_stage0, a_int0))
                v_req, v_stage, v_int = (x.copy() for x in (v_req0, v_stage0, v_int0))
                for frame in (a_req, a_stage, a_int, v_req, v_stage, v_int):
                    if not frame.empty and "scenario_name" in frame.columns:
                        frame["scenario_name"] = scenario.scenario_name

                c_req, c_stage, c_int = simulate(
                    scenario=scenario,
                    pipeline=pipeline,
                    run_kind="combined",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                    specs=a_specs + v_specs,
                )

                request_frames.extend([a_req, v_req, c_req])
                stage_frames.extend([a_stage, v_stage, c_stage])
                interval_frames.extend([a_int, v_int, c_int])

                trace = paired_attacker_trace(a_req, c_req)
                trace_id = f"{scenario.scenario_name}::{workload.workload_name}::trial{trial_id}"
                bb = trace[[
                    "probe_index",
                    "release_ns",
                    "attacker_only_completion_ns",
                    "combined_completion_ns",
                    "attacker_only_turnaround_ns",
                    "combined_turnaround_ns",
                    "excess_turnaround_ns",
                    "affected",
                ]].copy()
                bb.insert(0, "trace_id", trace_id)
                blackbox_frames.append(bb)

                causal = per_probe_causal_features(trace, attacker_stage_table(c_stage))
                causal.insert(0, "trial_id", trial_id)
                causal.insert(0, "workload_name", workload.workload_name)
                causal.insert(0, "scenario_name", scenario.scenario_name)
                causal.insert(0, "trace_id", trace_id)
                causal_frames.append(causal)

                trace_key_rows.append(
                    {
                        "trace_id": trace_id,
                        "scenario_name": scenario.scenario_name,
                        "scenario_family": scenario.scenario_family,
                        "link_capacity": scenario.link_capacity,
                        "shared_backend_components": ";".join(scenario.shared_backend_components),
                        "backend_capacity": scenario.backend_capacity,
                        "workload_name": workload.workload_name,
                        "trial_id": trial_id,
                        "victim_phase_ns": phase,
                    }
                )

                trial_rows.append(
                    summarize_trial(
                        scenario=scenario,
                        workload=workload,
                        trial_id=trial_id,
                        phase_ns=phase,
                        trace=trace,
                        causal=causal,
                        victim_only_req=v_req,
                        combined_req=c_req,
                    )
                )

                done += 1
                if done % max(1, total_tuples // 10) == 0 or done == total_tuples:
                    print(f"  completed {done}/{total_tuples} scenario-workload-trial tuples")

    request_df = pd.concat(request_frames, ignore_index=True)
    stage_df = pd.concat(stage_frames, ignore_index=True)
    interval_df = pd.concat(interval_frames, ignore_index=True)
    trial_df = pd.DataFrame(trial_rows)
    blackbox_df = pd.concat(blackbox_frames, ignore_index=True)
    causal_df = pd.concat(causal_frames, ignore_index=True)
    trace_key_df = pd.DataFrame(trace_key_rows)

    # Raw evaluator files.
    request_df.to_csv(rawdir / "phase2_04_request_records.csv.gz", index=False, compression=GZIP_COMPRESSION)
    stage_df.to_csv(rawdir / "phase2_04_stage_records.csv.gz", index=False, compression=GZIP_COMPRESSION)
    interval_df.to_csv(rawdir / "phase2_04_resource_intervals.csv.gz", index=False, compression=GZIP_COMPRESSION)
    causal_df.to_csv(rawdir / "phase2_04_causal_probe_records.csv.gz", index=False, compression=GZIP_COMPRESSION)

    # Attacker-visible timing and evaluator key are deliberately separated.
    blackbox_df.to_csv(outdir / "phase2_04_blackbox_trace_summary.csv", index=False)
    trace_key_df.to_csv(outdir / "phase2_04_trace_key.csv", index=False)
    trial_df.to_csv(outdir / "phase2_04_trial_summary.csv", index=False)

    sc_summary = scenario_summary(trial_df)
    wl_summary = workload_summary(trial_df)
    decomp = delay_decomposition(stage_df)
    block = blocking_attribution(stage_df)
    util = resource_utilization(interval_df)
    reuse = endpoint_reuse_summary(stage_df, request_df)
    temporal = temporal_fingerprint_summary(causal_df)

    sc_summary.to_csv(outdir / "phase2_04_scenario_summary.csv", index=False)
    wl_summary.to_csv(outdir / "phase2_04_workload_summary.csv", index=False)
    decomp.to_csv(outdir / "phase2_04_delay_decomposition_summary.csv", index=False)
    block.to_csv(outdir / "phase2_04_blocking_attribution_summary.csv", index=False)
    util.to_csv(outdir / "phase2_04_resource_utilization_summary.csv", index=False)
    reuse.to_csv(outdir / "phase2_04_endpoint_reuse_summary.csv", index=False)
    temporal.to_csv(outdir / "phase2_04_temporal_fingerprint_summary.csv", index=False)

    # Timing-only workload structure test.
    features = build_trace_feature_table(blackbox_df, trace_key_df)
    features.to_csv(outdir / "phase2_04_trace_features.csv", index=False)
    fingerprint_metrics, fingerprint_predictions = leave_one_trial_out_nearest_centroid(features)
    fingerprint_metrics.to_csv(outdir / "phase2_04_workload_fingerprint_metrics.csv", index=False)
    fingerprint_predictions.to_csv(outdir / "phase2_04_workload_fingerprint_predictions.csv", index=False)

    protocol_control_df = run_protocol_controls(args.seed)
    protocol_control_df.to_csv(outdir / "phase2_04_protocol_control_summary.csv", index=False)

    validations = validate(
        scenarios=scenarios,
        request_df=request_df,
        stage_df=stage_df,
        interval_df=interval_df,
        trial_df=trial_df,
        causal_df=causal_df,
        blackbox_df=blackbox_df,
        protocol_control_df=protocol_control_df,
    )
    validation_df = pd.DataFrame(asdict(v) for v in validations)
    validation_df.to_csv(outdir / "phase2_04_validation_assertions.csv", index=False)
    validation_summary = validation_df.groupby("validation_group", as_index=False).agg(
        assertion_count=("assertion_name", "count"),
        passed_count=("passed", "sum"),
    )
    validation_summary["failed_count"] = (
        validation_summary["assertion_count"] - validation_summary["passed_count"]
    )
    validation_summary["pass_rate"] = (
        validation_summary["passed_count"] / validation_summary["assertion_count"]
    )
    validation_summary.to_csv(outdir / "phase2_04_validation_summary.csv", index=False)

    manifest = {
        "experiment": "Phase 2.4 — Measurement, Feedforward, and Reset Contention",
        "output_directory": str(outdir),
        "trial_count_per_workload_configuration": args.trials,
        "scenario_count": len(scenarios),
        "workload_count": len(workloads),
        "scenario_workload_trial_tuples": int(total_tuples),
        "probe_period_ns": ATTACKER_PERIOD_NS,
        "observation_window_ns": OBSERVATION_WINDOW_NS,
        "critical_path_nominal_latency_ns": float(sum(STAGE_DURATIONS_NS[s] for s in CRITICAL_STAGES)),
        "postcompletion_reset_service_ns": STAGE_DURATIONS_NS["reset_recovery"],
        "cleanup_nominal_latency_ns": float(sum(STAGE_DURATIONS_NS.values())),
        "link_capacities": [1, 2],
        "backend_components": list(BACKEND_COMPONENTS),
        "validation_assertion_count": int(len(validation_df)),
        "passed_assertions": int(validation_df["passed"].sum()),
        "failed_assertions": int((~validation_df["passed"]).sum()),
        "all_validations_passed": bool(validation_df["passed"].all()),
        "blackbox_columns": list(blackbox_df.columns),
    }
    with open(outdir / "phase2_04_run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nPhase 2.4 — Measurement, Feedforward, and Reset Contention")
    print("=" * 68)
    print(validation_summary.to_string(index=False))
    print("\nNominal external completion latency:", manifest["critical_path_nominal_latency_ns"], "ns")
    print("Post-completion reset service:", STAGE_DURATIONS_NS["reset_recovery"], "ns")
    print("Nominal cleanup completion latency:", manifest["cleanup_nominal_latency_ns"], "ns")
    print("\nResults saved to:", outdir)
    if manifest["all_validations_passed"]:
        print("\nAll Phase 2.4 causal/resource validations passed.")
    else:
        failed = validation_df[~validation_df["passed"]]
        print("\nFAILED VALIDATIONS:")
        print(failed[["validation_group", "assertion_name", "expected", "observed"]].to_string(index=False))
        if FAIL_ON_VALIDATION_ERROR:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
