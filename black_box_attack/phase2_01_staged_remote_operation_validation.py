#!/usr/bin/env python3
"""
Phase 2.1 — Staged Remote-Operation Validation
===============================================

This experiment replaces the Phase 1 monolithic remote-operation delay with an
explicit protocol-dependent stage pipeline. It does not perform an attack.
Instead, it validates causal timing, stage-specific waiting, and resource
lifetimes before later Phase 2 leakage experiments use the staged model.

Run
---
    python phase2_01_staged_remote_operation_validation.py

Output directory
----------------
blackbox_window_results/
└── phase2/
    └── phase2_01_staged_remote_operation_validation/
        ├── baseline_no_contention/
        ├── single_stage_contention/
        ├── resource_lifetimes/
        ├── plots/
        ├── phase2_01_protocol_definitions.csv
        ├── phase2_01_stage_timeline.csv
        ├── phase2_01_resource_intervals.csv
        ├── phase2_01_assertion_results.csv
        ├── phase2_01_validation_summary.csv
        ├── phase2_01_contention_isolation_summary.csv
        ├── phase2_01_resource_lifetime_summary.csv
        └── phase2_01_run_manifest.json

Validation goals
----------------
1. With no contention, request latency equals the sum of relevant stage times.
2. Artificial contention at one resource increases only the corresponding
   stage's waiting time.
3. Communication endpoints/qubits remain unavailable through reset and are
   released only after reset completion.
4. Switch paths may be released before endpoints. A request that needs only the
   switch may proceed while an endpoint remains reserved.
5. Every stage and resource interval has a causally consistent timeline.

The protocol definitions intentionally differ. Not every protocol is forced to
use EPR generation, Bell-state measurement, or classical coordination.
"""

from __future__ import annotations

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
# Integrated experiment settings
# =============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase2"
    / "phase2_01_staged_remote_operation_validation"
)

BASELINE_DIR = OUTPUT_DIR / "baseline_no_contention"
CONTENTION_DIR = OUTPUT_DIR / "single_stage_contention"
LIFETIME_DIR = OUTPUT_DIR / "resource_lifetimes"
PLOT_DIR = OUTPUT_DIR / "plots"

CONTROLLED_CONTENTION_NS = 200
FLOAT_TOLERANCE_NS = 1e-9

# Stage and resource logging are intentionally enabled for Phase 2.1.
SAVE_DETAILED_STAGE_TIMELINE = True
SAVE_DETAILED_RESOURCE_INTERVALS = True
FAIL_PROCESS_ON_ASSERTION_ERROR = True


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class StageDefinition:
    """One stage in a protocol-specific remote-operation pipeline."""

    stage_name: str
    duration_ns: float

    # Resources acquired at stage start and held until a later stage explicitly
    # releases them.
    acquire_held: tuple[str, ...] = ()

    # Resources acquired only for this stage's duration.
    acquire_scoped: tuple[str, ...] = ()

    # Previously held resources required to be present during this stage.
    require_held: tuple[str, ...] = ()

    # Held resources released at this stage's completion.
    release_held: tuple[str, ...] = ()

    # One resource used for the controlled single-stage contention test.
    contention_resource: Optional[str] = None

    description: str = ""


@dataclass(frozen=True)
class ProtocolDefinition:
    """Protocol-dependent remote-operation stage sequence."""

    protocol_name: str
    operation_kind: str
    stages: tuple[StageDefinition, ...]
    description: str

    @property
    def nominal_latency_ns(self) -> float:
        return float(sum(stage.duration_ns for stage in self.stages))


@dataclass
class ResourceInterval:
    resource_name: str
    unit_id: int
    request_id: str
    interval_kind: str
    start_ns: float
    end_ns: float
    acquired_stage: str
    released_stage: str
    scenario_name: str
    protocol_name: str


@dataclass
class StageRecord:
    scenario_name: str
    protocol_name: str
    request_id: str
    request_ready_ns: float
    stage_index: int
    stage_name: str
    stage_ready_ns: float
    stage_start_ns: float
    stage_end_ns: float
    wait_ns: float
    service_ns: float
    acquired_held_resources: str
    acquired_scoped_resources: str
    required_held_resources: str
    released_held_resources: str


@dataclass
class RequestResult:
    scenario_name: str
    protocol_name: str
    request_id: str
    request_ready_ns: float
    completion_ns: float
    total_latency_ns: float
    total_wait_ns: float
    total_service_ns: float
    stage_records: list[StageRecord] = field(default_factory=list)


@dataclass
class AssertionResult:
    validation_group: str
    scenario_name: str
    protocol_name: str
    request_id: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


# =============================================================================
# Resource calendar
# =============================================================================


class ResourcePool:
    """Finite-capacity resource with explicit non-overlapping intervals."""

    def __init__(self, resource_name: str, capacity: int = 1) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive for {resource_name}")

        self.resource_name = resource_name
        self.capacity = int(capacity)
        self._intervals: dict[int, list[ResourceInterval]] = {
            unit_id: [] for unit_id in range(self.capacity)
        }

    @property
    def intervals(self) -> list[ResourceInterval]:
        return [
            interval
            for unit_intervals in self._intervals.values()
            for interval in unit_intervals
        ]

    @staticmethod
    def _overlaps(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> bool:
        # Half-open intervals [start, end): release at t makes the resource
        # immediately available to another request at t.
        return start_a < end_b and start_b < end_a

    def _unit_available(
        self,
        unit_id: int,
        start_ns: float,
        end_ns: float,
    ) -> bool:
        return all(
            not self._overlaps(
                start_ns,
                end_ns,
                interval.start_ns,
                interval.end_ns,
            )
            for interval in self._intervals[unit_id]
        )

    def earliest_slot(
        self,
        ready_ns: float,
        duration_ns: float,
    ) -> tuple[float, int]:
        """Find the earliest unit and start time for a scoped interval."""

        if duration_ns < 0:
            raise ValueError("duration_ns cannot be negative")

        # Zero-duration acquisition/release stages still need a point at which
        # a unit is free. Use a tiny epsilon for calendar search only.
        search_duration = max(float(duration_ns), 1e-9)

        candidates: list[tuple[float, int]] = []

        for unit_id in range(self.capacity):
            candidate = float(ready_ns)
            intervals = sorted(
                self._intervals[unit_id],
                key=lambda interval: (interval.start_ns, interval.end_ns),
            )

            for interval in intervals:
                if candidate + search_duration <= interval.start_ns:
                    break
                if candidate < interval.end_ns:
                    candidate = interval.end_ns

            candidates.append((candidate, unit_id))

        return min(candidates, key=lambda item: (item[0], item[1]))

    def reserve(
        self,
        *,
        unit_id: int,
        start_ns: float,
        end_ns: float,
        request_id: str,
        interval_kind: str,
        acquired_stage: str,
        released_stage: str,
        scenario_name: str,
        protocol_name: str,
    ) -> ResourceInterval:
        if end_ns < start_ns:
            raise ValueError(
                f"invalid interval for {self.resource_name}: "
                f"{start_ns} -> {end_ns}"
            )

        if end_ns > start_ns and not self._unit_available(
            unit_id,
            start_ns,
            end_ns,
        ):
            raise RuntimeError(
                f"resource overlap on {self.resource_name}[{unit_id}] "
                f"for {request_id}: {start_ns} -> {end_ns}"
            )

        interval = ResourceInterval(
            resource_name=self.resource_name,
            unit_id=int(unit_id),
            request_id=request_id,
            interval_kind=interval_kind,
            start_ns=float(start_ns),
            end_ns=float(end_ns),
            acquired_stage=acquired_stage,
            released_stage=released_stage,
            scenario_name=scenario_name,
            protocol_name=protocol_name,
        )

        self._intervals[unit_id].append(interval)
        self._intervals[unit_id].sort(
            key=lambda item: (item.start_ns, item.end_ns, item.request_id)
        )

        return interval

    def add_blocker(
        self,
        *,
        start_ns: float,
        end_ns: float,
        scenario_name: str,
        protocol_name: str,
        blocker_id: str,
        unit_id: int = 0,
    ) -> None:
        self.reserve(
            unit_id=unit_id,
            start_ns=start_ns,
            end_ns=end_ns,
            request_id=blocker_id,
            interval_kind="artificial_blocker",
            acquired_stage="external_contention_start",
            released_stage="external_contention_end",
            scenario_name=scenario_name,
            protocol_name=protocol_name,
        )


class ResourceSystem:
    """Collection of resources used by the staged protocols."""

    DEFAULT_CAPACITIES = {
        "source_endpoint": 1,
        "target_endpoint": 1,
        "source_comm_qubit": 1,
        "target_comm_qubit": 1,
        "switch_path": 1,
        "intermodule_link": 1,
        "receiver_engine": 1,
        "measurement_engine": 1,
        "classical_channel": 1,
        "reset_engine": 1,
    }

    def __init__(self, capacities: Optional[dict[str, int]] = None) -> None:
        merged = dict(self.DEFAULT_CAPACITIES)
        if capacities:
            merged.update(capacities)

        self.pools: dict[str, ResourcePool] = {
            name: ResourcePool(name, capacity)
            for name, capacity in merged.items()
        }

    def pool(self, resource_name: str) -> ResourcePool:
        try:
            return self.pools[resource_name]
        except KeyError as error:
            raise KeyError(f"unknown resource: {resource_name}") from error

    @property
    def all_intervals(self) -> list[ResourceInterval]:
        return [
            interval
            for pool in self.pools.values()
            for interval in pool.intervals
        ]


# =============================================================================
# Protocol definitions
# =============================================================================


def build_protocols() -> dict[str, ProtocolDefinition]:
    """Return explicit, protocol-dependent remote-operation pipelines."""

    direct_remote_cx = ProtocolDefinition(
        protocol_name="direct_remote_cx",
        operation_kind="remote_cx",
        description=(
            "Direct remote-CX abstraction with endpoint acquisition, path "
            "setup, inter-module primitive, receiver-side gate, and reset. "
            "It does not require measurement or classical feedforward."
        ),
        stages=(
            StageDefinition(
                stage_name="endpoint_acquisition",
                duration_ns=20,
                acquire_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                contention_resource="source_endpoint",
                description="Reserve source/target interfaces and communication qubits.",
            ),
            StageDefinition(
                stage_name="route_acquisition",
                duration_ns=15,
                acquire_held=("switch_path",),
                require_held=("source_endpoint", "target_endpoint"),
                contention_resource="switch_path",
                description="Acquire the selected switch path.",
            ),
            StageDefinition(
                stage_name="intermodule_transfer",
                duration_ns=80,
                acquire_scoped=("intermodule_link",),
                require_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                    "switch_path",
                ),
                release_held=("switch_path",),
                contention_resource="intermodule_link",
                description="Execute the nonlocal transfer primitive and release the path.",
            ),
            StageDefinition(
                stage_name="receiver_side_gate",
                duration_ns=30,
                acquire_scoped=("receiver_engine",),
                require_held=("target_endpoint", "target_comm_qubit"),
                contention_resource="receiver_engine",
                description="Apply the receiver-side local component of remote CX.",
            ),
            StageDefinition(
                stage_name="communication_reset",
                duration_ns=40,
                acquire_scoped=("reset_engine",),
                require_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                release_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                contention_resource="reset_engine",
                description="Reset communication qubits, then release endpoints.",
            ),
        ),
    )

    direct_state_transfer = ProtocolDefinition(
        protocol_name="direct_state_transfer",
        operation_kind="state_transfer",
        description=(
            "Coherent state-transfer pipeline with no BSM or classical "
            "feedforward requirement."
        ),
        stages=(
            StageDefinition(
                stage_name="endpoint_acquisition",
                duration_ns=20,
                acquire_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                contention_resource="source_comm_qubit",
            ),
            StageDefinition(
                stage_name="route_acquisition",
                duration_ns=15,
                acquire_held=("switch_path",),
                require_held=("source_endpoint", "target_endpoint"),
                contention_resource="switch_path",
            ),
            StageDefinition(
                stage_name="coherent_transfer",
                duration_ns=120,
                acquire_scoped=("intermodule_link",),
                require_held=(
                    "source_comm_qubit",
                    "target_comm_qubit",
                    "switch_path",
                ),
                release_held=("switch_path",),
                contention_resource="intermodule_link",
            ),
            StageDefinition(
                stage_name="receiver_latch",
                duration_ns=25,
                acquire_scoped=("receiver_engine",),
                require_held=("target_endpoint", "target_comm_qubit"),
                contention_resource="receiver_engine",
            ),
            StageDefinition(
                stage_name="communication_reset",
                duration_ns=50,
                acquire_scoped=("reset_engine",),
                require_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                release_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                contention_resource="reset_engine",
            ),
        ),
    )

    teleportation_state_transfer = ProtocolDefinition(
        protocol_name="teleportation_state_transfer",
        operation_kind="state_transfer",
        description=(
            "Teleportation-style state transfer. This protocol specifically "
            "includes source measurement and classical coordination; the "
            "other protocol definitions do not."
        ),
        stages=(
            StageDefinition(
                stage_name="endpoint_acquisition",
                duration_ns=20,
                acquire_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                contention_resource="target_comm_qubit",
            ),
            StageDefinition(
                stage_name="route_acquisition",
                duration_ns=15,
                acquire_held=("switch_path",),
                require_held=("source_endpoint", "target_endpoint"),
                contention_resource="switch_path",
            ),
            StageDefinition(
                stage_name="entanglement_transfer_primitive",
                duration_ns=100,
                acquire_scoped=("intermodule_link",),
                require_held=(
                    "source_comm_qubit",
                    "target_comm_qubit",
                    "switch_path",
                ),
                release_held=("switch_path",),
                contention_resource="intermodule_link",
            ),
            StageDefinition(
                stage_name="source_measurement",
                duration_ns=35,
                acquire_scoped=("measurement_engine",),
                require_held=("source_endpoint", "source_comm_qubit"),
                contention_resource="measurement_engine",
            ),
            StageDefinition(
                stage_name="classical_coordination",
                duration_ns=25,
                acquire_scoped=("classical_channel",),
                require_held=("source_endpoint", "target_endpoint"),
                contention_resource="classical_channel",
            ),
            StageDefinition(
                stage_name="receiver_correction",
                duration_ns=20,
                acquire_scoped=("receiver_engine",),
                require_held=("target_endpoint", "target_comm_qubit"),
                contention_resource="receiver_engine",
            ),
            StageDefinition(
                stage_name="communication_reset",
                duration_ns=50,
                acquire_scoped=("reset_engine",),
                require_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                release_held=(
                    "source_endpoint",
                    "target_endpoint",
                    "source_comm_qubit",
                    "target_comm_qubit",
                ),
                contention_resource="reset_engine",
            ),
        ),
    )

    return {
        protocol.protocol_name: protocol
        for protocol in (
            direct_remote_cx,
            direct_state_transfer,
            teleportation_state_transfer,
        )
    }


# =============================================================================
# Request execution
# =============================================================================


class StagedRemoteOperationSimulator:
    """Execute protocol stages and attribute wait to the causal stage."""

    def __init__(self, resources: ResourceSystem) -> None:
        self.resources = resources

    def _find_common_scoped_start(
        self,
        resource_names: Iterable[str],
        ready_ns: float,
        duration_ns: float,
    ) -> tuple[float, dict[str, int]]:
        """Find an atomic common start for all newly acquired resources."""

        names = list(resource_names)
        if not names:
            return float(ready_ns), {}

        candidate = float(ready_ns)

        for _ in range(10_000):
            selections: dict[str, int] = {}
            next_candidate = candidate

            for resource_name in names:
                start_ns, unit_id = self.resources.pool(resource_name).earliest_slot(
                    candidate,
                    duration_ns,
                )
                selections[resource_name] = unit_id
                next_candidate = max(next_candidate, start_ns)

            if math.isclose(
                next_candidate,
                candidate,
                abs_tol=FLOAT_TOLERANCE_NS,
            ):
                # Recheck every selected unit at the common candidate.
                if all(
                    self.resources.pool(resource_name)._unit_available(
                        selections[resource_name],
                        candidate,
                        candidate + max(duration_ns, 1e-9),
                    )
                    for resource_name in names
                ):
                    return candidate, selections

            candidate = next_candidate

        raise RuntimeError("failed to find a common resource-acquisition slot")

    def execute_request(
        self,
        *,
        protocol: ProtocolDefinition,
        request_id: str,
        ready_ns: float,
        scenario_name: str,
    ) -> RequestResult:
        current_ready_ns = float(ready_ns)
        held: dict[str, tuple[int, float, str]] = {}
        stage_records: list[StageRecord] = []

        for stage_index, stage in enumerate(protocol.stages):
            missing_required = [
                name for name in stage.require_held if name not in held
            ]
            if missing_required:
                raise RuntimeError(
                    f"{request_id} entered {stage.stage_name} without held "
                    f"resources: {missing_required}"
                )

            newly_acquired = [
                name for name in stage.acquire_held if name not in held
            ]
            resources_to_acquire = list(stage.acquire_scoped) + newly_acquired

            stage_start_ns, selections = self._find_common_scoped_start(
                resources_to_acquire,
                current_ready_ns,
                stage.duration_ns,
            )
            stage_end_ns = stage_start_ns + stage.duration_ns
            wait_ns = stage_start_ns - current_ready_ns

            # Record held acquisition start. The complete interval is committed
            # only when the protocol explicitly releases the resource.
            for resource_name in newly_acquired:
                held[resource_name] = (
                    selections[resource_name],
                    stage_start_ns,
                    stage.stage_name,
                )

            # Scoped resources are occupied exactly for this stage.
            for resource_name in stage.acquire_scoped:
                self.resources.pool(resource_name).reserve(
                    unit_id=selections[resource_name],
                    start_ns=stage_start_ns,
                    end_ns=stage_end_ns,
                    request_id=request_id,
                    interval_kind="stage_scoped",
                    acquired_stage=stage.stage_name,
                    released_stage=stage.stage_name,
                    scenario_name=scenario_name,
                    protocol_name=protocol.protocol_name,
                )

            # Commit held-resource lifetimes at explicit release.
            for resource_name in stage.release_held:
                if resource_name not in held:
                    raise RuntimeError(
                        f"{request_id} attempted to release unheld "
                        f"resource {resource_name} in {stage.stage_name}"
                    )

                unit_id, acquired_ns, acquired_stage = held.pop(resource_name)
                self.resources.pool(resource_name).reserve(
                    unit_id=unit_id,
                    start_ns=acquired_ns,
                    end_ns=stage_end_ns,
                    request_id=request_id,
                    interval_kind="held_across_stages",
                    acquired_stage=acquired_stage,
                    released_stage=stage.stage_name,
                    scenario_name=scenario_name,
                    protocol_name=protocol.protocol_name,
                )

            stage_records.append(
                StageRecord(
                    scenario_name=scenario_name,
                    protocol_name=protocol.protocol_name,
                    request_id=request_id,
                    request_ready_ns=float(ready_ns),
                    stage_index=stage_index,
                    stage_name=stage.stage_name,
                    stage_ready_ns=current_ready_ns,
                    stage_start_ns=stage_start_ns,
                    stage_end_ns=stage_end_ns,
                    wait_ns=wait_ns,
                    service_ns=stage.duration_ns,
                    acquired_held_resources=",".join(newly_acquired),
                    acquired_scoped_resources=",".join(stage.acquire_scoped),
                    required_held_resources=",".join(stage.require_held),
                    released_held_resources=",".join(stage.release_held),
                )
            )

            current_ready_ns = stage_end_ns

        if held:
            raise RuntimeError(
                f"{request_id} completed with unreleased resources: "
                f"{sorted(held)}"
            )

        total_latency_ns = current_ready_ns - ready_ns
        total_wait_ns = sum(record.wait_ns for record in stage_records)
        total_service_ns = sum(record.service_ns for record in stage_records)

        return RequestResult(
            scenario_name=scenario_name,
            protocol_name=protocol.protocol_name,
            request_id=request_id,
            request_ready_ns=float(ready_ns),
            completion_ns=current_ready_ns,
            total_latency_ns=total_latency_ns,
            total_wait_ns=total_wait_ns,
            total_service_ns=total_service_ns,
            stage_records=stage_records,
        )

    def execute_switch_only_request(
        self,
        *,
        request_id: str,
        ready_ns: float,
        duration_ns: float,
        scenario_name: str,
    ) -> RequestResult:
        """Diagnostic request that needs only the switch path."""

        stage_start_ns, selections = self._find_common_scoped_start(
            ["switch_path"],
            ready_ns,
            duration_ns,
        )
        stage_end_ns = stage_start_ns + duration_ns

        self.resources.pool("switch_path").reserve(
            unit_id=selections["switch_path"],
            start_ns=stage_start_ns,
            end_ns=stage_end_ns,
            request_id=request_id,
            interval_kind="stage_scoped",
            acquired_stage="switch_only_operation",
            released_stage="switch_only_operation",
            scenario_name=scenario_name,
            protocol_name="switch_only_diagnostic",
        )

        record = StageRecord(
            scenario_name=scenario_name,
            protocol_name="switch_only_diagnostic",
            request_id=request_id,
            request_ready_ns=ready_ns,
            stage_index=0,
            stage_name="switch_only_operation",
            stage_ready_ns=ready_ns,
            stage_start_ns=stage_start_ns,
            stage_end_ns=stage_end_ns,
            wait_ns=stage_start_ns - ready_ns,
            service_ns=duration_ns,
            acquired_held_resources="",
            acquired_scoped_resources="switch_path",
            required_held_resources="",
            released_held_resources="",
        )

        return RequestResult(
            scenario_name=scenario_name,
            protocol_name="switch_only_diagnostic",
            request_id=request_id,
            request_ready_ns=ready_ns,
            completion_ns=stage_end_ns,
            total_latency_ns=stage_end_ns - ready_ns,
            total_wait_ns=stage_start_ns - ready_ns,
            total_service_ns=duration_ns,
            stage_records=[record],
        )


# =============================================================================
# Assertion helpers
# =============================================================================


def add_assertion(
    assertions: list[AssertionResult],
    *,
    validation_group: str,
    scenario_name: str,
    protocol_name: str,
    request_id: str,
    assertion_name: str,
    passed: bool,
    expected: object,
    observed: object,
    details: str = "",
) -> None:
    assertions.append(
        AssertionResult(
            validation_group=validation_group,
            scenario_name=scenario_name,
            protocol_name=protocol_name,
            request_id=request_id,
            assertion_name=assertion_name,
            passed=bool(passed),
            expected=str(expected),
            observed=str(observed),
            details=details,
        )
    )


def close_enough(first: float, second: float) -> bool:
    return math.isclose(first, second, abs_tol=FLOAT_TOLERANCE_NS)


def validate_stage_continuity(
    result: RequestResult,
    assertions: list[AssertionResult],
    validation_group: str,
) -> None:
    previous_end = result.request_ready_ns
    for record in result.stage_records:
        passed = close_enough(record.stage_ready_ns, previous_end)
        add_assertion(
            assertions,
            validation_group=validation_group,
            scenario_name=result.scenario_name,
            protocol_name=result.protocol_name,
            request_id=result.request_id,
            assertion_name=f"stage_continuity::{record.stage_name}",
            passed=passed,
            expected=previous_end,
            observed=record.stage_ready_ns,
            details="Each stage must become ready at the previous stage completion.",
        )
        previous_end = record.stage_end_ns


# =============================================================================
# Validation scenarios
# =============================================================================


def run_baseline_validation(
    protocols: dict[str, ProtocolDefinition],
    assertions: list[AssertionResult],
) -> tuple[list[RequestResult], list[ResourceInterval], list[dict]]:
    results: list[RequestResult] = []
    intervals: list[ResourceInterval] = []
    summaries: list[dict] = []

    for protocol in protocols.values():
        scenario = f"baseline_no_contention::{protocol.protocol_name}"
        resources = ResourceSystem()
        simulator = StagedRemoteOperationSimulator(resources)
        result = simulator.execute_request(
            protocol=protocol,
            request_id=f"baseline_{protocol.protocol_name}",
            ready_ns=0,
            scenario_name=scenario,
        )

        results.append(result)
        intervals.extend(resources.all_intervals)

        add_assertion(
            assertions,
            validation_group="baseline_no_contention",
            scenario_name=scenario,
            protocol_name=protocol.protocol_name,
            request_id=result.request_id,
            assertion_name="latency_equals_sum_of_stage_times",
            passed=close_enough(result.total_latency_ns, protocol.nominal_latency_ns),
            expected=protocol.nominal_latency_ns,
            observed=result.total_latency_ns,
        )
        add_assertion(
            assertions,
            validation_group="baseline_no_contention",
            scenario_name=scenario,
            protocol_name=protocol.protocol_name,
            request_id=result.request_id,
            assertion_name="zero_total_wait",
            passed=close_enough(result.total_wait_ns, 0),
            expected=0,
            observed=result.total_wait_ns,
        )
        add_assertion(
            assertions,
            validation_group="baseline_no_contention",
            scenario_name=scenario,
            protocol_name=protocol.protocol_name,
            request_id=result.request_id,
            assertion_name="every_stage_has_zero_wait",
            passed=all(close_enough(record.wait_ns, 0) for record in result.stage_records),
            expected="all stage waits = 0",
            observed={record.stage_name: record.wait_ns for record in result.stage_records},
        )
        validate_stage_continuity(result, assertions, "baseline_no_contention")

        summaries.append(
            {
                "validation_group": "baseline_no_contention",
                "scenario_name": scenario,
                "protocol_name": protocol.protocol_name,
                "request_id": result.request_id,
                "nominal_stage_sum_ns": protocol.nominal_latency_ns,
                "observed_latency_ns": result.total_latency_ns,
                "total_wait_ns": result.total_wait_ns,
                "latency_error_ns": result.total_latency_ns - protocol.nominal_latency_ns,
            }
        )

    return results, intervals, summaries


def run_single_stage_contention_validation(
    protocols: dict[str, ProtocolDefinition],
    assertions: list[AssertionResult],
) -> tuple[list[RequestResult], list[ResourceInterval], list[dict]]:
    results: list[RequestResult] = []
    intervals: list[ResourceInterval] = []
    summaries: list[dict] = []

    for protocol in protocols.values():
        # Obtain nominal stage-ready times once.
        baseline_resources = ResourceSystem()
        baseline_result = StagedRemoteOperationSimulator(
            baseline_resources
        ).execute_request(
            protocol=protocol,
            request_id=f"contention_reference_{protocol.protocol_name}",
            ready_ns=0,
            scenario_name=f"contention_reference::{protocol.protocol_name}",
        )
        baseline_by_stage = {
            record.stage_name: record for record in baseline_result.stage_records
        }

        for target_stage in protocol.stages:
            if target_stage.contention_resource is None:
                continue

            scenario = (
                f"single_stage_contention::{protocol.protocol_name}::"
                f"{target_stage.stage_name}"
            )
            resources = ResourceSystem()

            nominal_ready = baseline_by_stage[target_stage.stage_name].stage_ready_ns
            blocker_end = nominal_ready + CONTROLLED_CONTENTION_NS
            resources.pool(target_stage.contention_resource).add_blocker(
                start_ns=0,
                end_ns=blocker_end,
                scenario_name=scenario,
                protocol_name=protocol.protocol_name,
                blocker_id=(
                    f"blocker_{protocol.protocol_name}_{target_stage.stage_name}"
                ),
            )

            result = StagedRemoteOperationSimulator(resources).execute_request(
                protocol=protocol,
                request_id=f"request_{protocol.protocol_name}_{target_stage.stage_name}",
                ready_ns=0,
                scenario_name=scenario,
            )
            results.append(result)
            intervals.extend(resources.all_intervals)

            observed_by_stage = {
                record.stage_name: record.wait_ns for record in result.stage_records
            }
            target_wait = observed_by_stage[target_stage.stage_name]
            non_target_waits = {
                name: wait
                for name, wait in observed_by_stage.items()
                if name != target_stage.stage_name
            }

            add_assertion(
                assertions,
                validation_group="single_stage_contention",
                scenario_name=scenario,
                protocol_name=protocol.protocol_name,
                request_id=result.request_id,
                assertion_name="target_stage_wait_matches_injected_contention",
                passed=close_enough(target_wait, CONTROLLED_CONTENTION_NS),
                expected=CONTROLLED_CONTENTION_NS,
                observed=target_wait,
                details=f"Blocked resource: {target_stage.contention_resource}",
            )
            add_assertion(
                assertions,
                validation_group="single_stage_contention",
                scenario_name=scenario,
                protocol_name=protocol.protocol_name,
                request_id=result.request_id,
                assertion_name="non_target_stages_do_not_accumulate_wait",
                passed=all(close_enough(wait, 0) for wait in non_target_waits.values()),
                expected="all non-target stage waits = 0",
                observed=non_target_waits,
                details=(
                    "Downstream stage start times may shift causally, but their "
                    "own waiting time must remain zero."
                ),
            )
            add_assertion(
                assertions,
                validation_group="single_stage_contention",
                scenario_name=scenario,
                protocol_name=protocol.protocol_name,
                request_id=result.request_id,
                assertion_name="total_latency_increases_only_by_injected_wait",
                passed=close_enough(
                    result.total_latency_ns,
                    protocol.nominal_latency_ns + CONTROLLED_CONTENTION_NS,
                ),
                expected=protocol.nominal_latency_ns + CONTROLLED_CONTENTION_NS,
                observed=result.total_latency_ns,
            )
            validate_stage_continuity(result, assertions, "single_stage_contention")

            summaries.append(
                {
                    "validation_group": "single_stage_contention",
                    "scenario_name": scenario,
                    "protocol_name": protocol.protocol_name,
                    "target_stage": target_stage.stage_name,
                    "blocked_resource": target_stage.contention_resource,
                    "injected_contention_ns": CONTROLLED_CONTENTION_NS,
                    "target_stage_wait_ns": target_wait,
                    "sum_non_target_wait_ns": sum(non_target_waits.values()),
                    "observed_total_wait_ns": result.total_wait_ns,
                    "observed_latency_ns": result.total_latency_ns,
                    "expected_latency_ns": (
                        protocol.nominal_latency_ns + CONTROLLED_CONTENTION_NS
                    ),
                }
            )

    return results, intervals, summaries


def _find_request_interval(
    intervals: list[ResourceInterval],
    *,
    request_id: str,
    resource_name: str,
) -> ResourceInterval:
    matches = [
        interval
        for interval in intervals
        if interval.request_id == request_id
        and interval.resource_name == resource_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one interval for {request_id}/{resource_name}, "
            f"found {len(matches)}"
        )
    return matches[0]


def run_resource_lifetime_validation(
    protocols: dict[str, ProtocolDefinition],
    assertions: list[AssertionResult],
) -> tuple[list[RequestResult], list[ResourceInterval], list[dict]]:
    protocol = protocols["direct_remote_cx"]
    results: list[RequestResult] = []
    intervals: list[ResourceInterval] = []
    summaries: list[dict] = []

    # -------------------------------------------------------------------------
    # A. Endpoint and communication-qubit lifetimes extend through reset.
    # -------------------------------------------------------------------------
    scenario = "resource_lifetime::endpoint_held_through_reset"
    resources = ResourceSystem()
    simulator = StagedRemoteOperationSimulator(resources)

    first = simulator.execute_request(
        protocol=protocol,
        request_id="lifetime_request_A",
        ready_ns=0,
        scenario_name=scenario,
    )

    first_intervals = list(resources.all_intervals)
    source_endpoint_interval = _find_request_interval(
        first_intervals,
        request_id=first.request_id,
        resource_name="source_endpoint",
    )
    source_comm_interval = _find_request_interval(
        first_intervals,
        request_id=first.request_id,
        resource_name="source_comm_qubit",
    )
    switch_interval = _find_request_interval(
        first_intervals,
        request_id=first.request_id,
        resource_name="switch_path",
    )

    # Request B becomes ready after the switch has been released but before the
    # endpoint/communication qubit has completed reset.
    second_ready_ns = switch_interval.end_ns + 1
    second = simulator.execute_request(
        protocol=protocol,
        request_id="lifetime_request_B",
        ready_ns=second_ready_ns,
        scenario_name=scenario,
    )

    results.extend([first, second])
    intervals.extend(resources.all_intervals)

    second_endpoint_stage = second.stage_records[0]

    add_assertion(
        assertions,
        validation_group="resource_lifetimes",
        scenario_name=scenario,
        protocol_name=protocol.protocol_name,
        request_id=second.request_id,
        assertion_name="second_request_cannot_acquire_endpoint_before_reset_release",
        passed=(
            second_endpoint_stage.stage_start_ns
            >= source_endpoint_interval.end_ns - FLOAT_TOLERANCE_NS
        ),
        expected=f">= {source_endpoint_interval.end_ns}",
        observed=second_endpoint_stage.stage_start_ns,
    )
    add_assertion(
        assertions,
        validation_group="resource_lifetimes",
        scenario_name=scenario,
        protocol_name=protocol.protocol_name,
        request_id=first.request_id,
        assertion_name="endpoint_and_comm_qubit_release_together_after_reset",
        passed=close_enough(source_endpoint_interval.end_ns, source_comm_interval.end_ns),
        expected=source_endpoint_interval.end_ns,
        observed=source_comm_interval.end_ns,
    )
    add_assertion(
        assertions,
        validation_group="resource_lifetimes",
        scenario_name=scenario,
        protocol_name=protocol.protocol_name,
        request_id=first.request_id,
        assertion_name="switch_releases_before_endpoint",
        passed=switch_interval.end_ns < source_endpoint_interval.end_ns,
        expected=f"switch end < {source_endpoint_interval.end_ns}",
        observed=switch_interval.end_ns,
    )

    summaries.append(
        {
            "validation_group": "resource_lifetimes",
            "scenario_name": scenario,
            "protocol_name": protocol.protocol_name,
            "request_A_endpoint_reserved_ns": source_endpoint_interval.start_ns,
            "request_A_switch_released_ns": switch_interval.end_ns,
            "request_A_endpoint_released_ns": source_endpoint_interval.end_ns,
            "request_B_ready_ns": second_ready_ns,
            "request_B_endpoint_grant_ns": second_endpoint_stage.stage_start_ns,
            "request_B_endpoint_wait_ns": second_endpoint_stage.wait_ns,
        }
    )

    # -------------------------------------------------------------------------
    # B. A switch-only request proceeds while endpoint reset is still active.
    # -------------------------------------------------------------------------
    switch_scenario = "resource_lifetime::switch_released_independently"
    switch_resources = ResourceSystem()
    switch_simulator = StagedRemoteOperationSimulator(switch_resources)

    owner = switch_simulator.execute_request(
        protocol=protocol,
        request_id="switch_owner_request",
        ready_ns=0,
        scenario_name=switch_scenario,
    )
    owner_intervals = list(switch_resources.all_intervals)
    owner_switch = _find_request_interval(
        owner_intervals,
        request_id=owner.request_id,
        resource_name="switch_path",
    )
    owner_endpoint = _find_request_interval(
        owner_intervals,
        request_id=owner.request_id,
        resource_name="source_endpoint",
    )

    diagnostic_ready_ns = owner_switch.end_ns
    diagnostic = switch_simulator.execute_switch_only_request(
        request_id="switch_only_request",
        ready_ns=diagnostic_ready_ns,
        duration_ns=10,
        scenario_name=switch_scenario,
    )

    results.extend([owner, diagnostic])
    intervals.extend(switch_resources.all_intervals)

    diagnostic_stage = diagnostic.stage_records[0]
    add_assertion(
        assertions,
        validation_group="resource_lifetimes",
        scenario_name=switch_scenario,
        protocol_name="switch_only_diagnostic",
        request_id=diagnostic.request_id,
        assertion_name="switch_only_request_starts_at_switch_release",
        passed=close_enough(diagnostic_stage.stage_start_ns, owner_switch.end_ns),
        expected=owner_switch.end_ns,
        observed=diagnostic_stage.stage_start_ns,
    )
    add_assertion(
        assertions,
        validation_group="resource_lifetimes",
        scenario_name=switch_scenario,
        protocol_name="switch_only_diagnostic",
        request_id=diagnostic.request_id,
        assertion_name="switch_only_request_proceeds_while_endpoint_still_held",
        passed=diagnostic_stage.stage_end_ns <= owner_endpoint.end_ns,
        expected=f"completion <= endpoint release {owner_endpoint.end_ns}",
        observed=diagnostic_stage.stage_end_ns,
    )
    add_assertion(
        assertions,
        validation_group="resource_lifetimes",
        scenario_name=switch_scenario,
        protocol_name="switch_only_diagnostic",
        request_id=diagnostic.request_id,
        assertion_name="switch_only_request_has_zero_wait",
        passed=close_enough(diagnostic.total_wait_ns, 0),
        expected=0,
        observed=diagnostic.total_wait_ns,
    )

    summaries.append(
        {
            "validation_group": "resource_lifetimes",
            "scenario_name": switch_scenario,
            "protocol_name": protocol.protocol_name,
            "owner_switch_release_ns": owner_switch.end_ns,
            "owner_endpoint_release_ns": owner_endpoint.end_ns,
            "switch_only_ready_ns": diagnostic_ready_ns,
            "switch_only_start_ns": diagnostic_stage.stage_start_ns,
            "switch_only_completion_ns": diagnostic_stage.stage_end_ns,
            "switch_only_wait_ns": diagnostic.total_wait_ns,
        }
    )

    return results, intervals, summaries


# =============================================================================
# Cross-cutting resource assertions
# =============================================================================


def validate_resource_non_overlap(
    intervals: list[ResourceInterval],
    assertions: list[AssertionResult],
) -> None:
    grouped: dict[tuple[str, int, str], list[ResourceInterval]] = {}
    for interval in intervals:
        key = (interval.resource_name, interval.unit_id, interval.scenario_name)
        grouped.setdefault(key, []).append(interval)

    for (resource_name, unit_id, scenario_name), group in grouped.items():
        ordered = sorted(group, key=lambda item: (item.start_ns, item.end_ns))
        overlaps: list[str] = []
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end_ns > current.start_ns + FLOAT_TOLERANCE_NS:
                overlaps.append(
                    f"{previous.request_id}[{previous.start_ns},{previous.end_ns}) "
                    f"overlaps {current.request_id}[{current.start_ns},{current.end_ns})"
                )

        add_assertion(
            assertions,
            validation_group="resource_calendar",
            scenario_name=scenario_name,
            protocol_name="all",
            request_id="all",
            assertion_name=f"no_overlap::{resource_name}[{unit_id}]",
            passed=not overlaps,
            expected="no overlapping intervals on one resource unit",
            observed="; ".join(overlaps) if overlaps else "no overlaps",
        )


# =============================================================================
# Output generation
# =============================================================================


def protocol_definition_dataframe(
    protocols: dict[str, ProtocolDefinition],
) -> pd.DataFrame:
    rows: list[dict] = []
    for protocol in protocols.values():
        for stage_index, stage in enumerate(protocol.stages):
            rows.append(
                {
                    "protocol_name": protocol.protocol_name,
                    "operation_kind": protocol.operation_kind,
                    "protocol_description": protocol.description,
                    "nominal_protocol_latency_ns": protocol.nominal_latency_ns,
                    "stage_index": stage_index,
                    "stage_name": stage.stage_name,
                    "stage_duration_ns": stage.duration_ns,
                    "acquire_held": ",".join(stage.acquire_held),
                    "acquire_scoped": ",".join(stage.acquire_scoped),
                    "require_held": ",".join(stage.require_held),
                    "release_held": ",".join(stage.release_held),
                    "contention_resource": stage.contention_resource or "",
                    "stage_description": stage.description,
                }
            )
    return pd.DataFrame(rows)


def stage_dataframe(results: list[RequestResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [asdict(record) for result in results for record in result.stage_records]
    )


def interval_dataframe(intervals: list[ResourceInterval]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(interval) for interval in intervals])
    if not frame.empty:
        frame["occupancy_ns"] = frame["end_ns"] - frame["start_ns"]
    return frame


def assertion_dataframe(assertions: list[AssertionResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(assertion) for assertion in assertions])


def save_baseline_plot(protocols: dict[str, ProtocolDefinition]) -> None:
    protocol_names = list(protocols)
    stage_names = sorted(
        {stage.stage_name for protocol in protocols.values() for stage in protocol.stages}
    )

    bottom = np.zeros(len(protocol_names), dtype=float)
    plt.figure(figsize=(12, 6))

    for stage_name in stage_names:
        values = np.array(
            [
                next(
                    (
                        stage.duration_ns
                        for stage in protocols[protocol_name].stages
                        if stage.stage_name == stage_name
                    ),
                    0,
                )
                for protocol_name in protocol_names
            ],
            dtype=float,
        )
        if np.any(values > 0):
            plt.bar(protocol_names, values, bottom=bottom, label=stage_name)
            bottom += values

    plt.ylabel("Nominal stage time (ns)")
    plt.xlabel("Remote-operation protocol")
    plt.title("Phase 2.1 Protocol-Dependent Remote-Operation Timelines")
    plt.xticks(rotation=20, ha="right")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "protocol_stage_latency_breakdown.png", dpi=300)
    plt.close()


def save_contention_heatmap(contention_summary: pd.DataFrame) -> None:
    if contention_summary.empty:
        return

    pivot = contention_summary.pivot_table(
        index="target_stage",
        columns="protocol_name",
        values="target_stage_wait_ns",
        aggfunc="mean",
        fill_value=0,
    )

    plt.figure(figsize=(10, max(5, 0.45 * len(pivot))))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(image, label="Attributed stage wait (ns)")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Protocol")
    plt.ylabel("Artificially contended stage")
    plt.title("Single-Stage Contention Attribution")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "single_stage_contention_attribution.png", dpi=300)
    plt.close()


def save_timeline_plot(
    stage_frame: pd.DataFrame,
    protocol_name: str,
    scenario_pattern: str,
    filename: str,
) -> None:
    subset = stage_frame[
        (stage_frame["protocol_name"] == protocol_name)
        & stage_frame["scenario_name"].str.contains(scenario_pattern, regex=False)
    ].copy()
    if subset.empty:
        return

    request_ids = list(dict.fromkeys(subset["request_id"].tolist()))
    y_lookup = {request_id: index for index, request_id in enumerate(request_ids)}

    plt.figure(figsize=(13, max(4, 1.1 * len(request_ids))))
    for _, row in subset.iterrows():
        y_value = y_lookup[row["request_id"]]
        plt.barh(
            y_value,
            row["service_ns"],
            left=row["stage_start_ns"],
            height=0.45,
        )
        if row["wait_ns"] > 0:
            plt.barh(
                y_value,
                row["wait_ns"],
                left=row["stage_ready_ns"],
                height=0.18,
            )
        plt.text(
            row["stage_start_ns"] + row["service_ns"] / 2,
            y_value,
            row["stage_name"],
            ha="center",
            va="center",
            fontsize=7,
        )

    plt.yticks(range(len(request_ids)), request_ids)
    plt.xlabel("Time (ns)")
    plt.ylabel("Request")
    plt.title(f"Staged Timeline: {protocol_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    for directory in (
        OUTPUT_DIR,
        BASELINE_DIR,
        CONTENTION_DIR,
        LIFETIME_DIR,
        PLOT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    protocols = build_protocols()
    assertions: list[AssertionResult] = []

    baseline_results, baseline_intervals, baseline_summary = (
        run_baseline_validation(protocols, assertions)
    )
    contention_results, contention_intervals, contention_summary = (
        run_single_stage_contention_validation(protocols, assertions)
    )
    lifetime_results, lifetime_intervals, lifetime_summary = (
        run_resource_lifetime_validation(protocols, assertions)
    )

    all_results = baseline_results + contention_results + lifetime_results
    all_intervals = baseline_intervals + contention_intervals + lifetime_intervals

    validate_resource_non_overlap(all_intervals, assertions)

    protocol_frame = protocol_definition_dataframe(protocols)
    stage_frame = stage_dataframe(all_results)
    interval_frame = interval_dataframe(all_intervals)
    assertion_frame = assertion_dataframe(assertions)
    baseline_frame = pd.DataFrame(baseline_summary)
    contention_frame = pd.DataFrame(contention_summary)
    lifetime_frame = pd.DataFrame(lifetime_summary)

    validation_summary = (
        assertion_frame.groupby("validation_group", as_index=False)
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

    protocol_frame.to_csv(
        OUTPUT_DIR / "phase2_01_protocol_definitions.csv", index=False
    )
    assertion_frame.to_csv(
        OUTPUT_DIR / "phase2_01_assertion_results.csv", index=False
    )
    validation_summary.to_csv(
        OUTPUT_DIR / "phase2_01_validation_summary.csv", index=False
    )
    baseline_frame.to_csv(
        BASELINE_DIR / "baseline_validation_summary.csv", index=False
    )
    contention_frame.to_csv(
        CONTENTION_DIR / "contention_isolation_summary.csv", index=False
    )
    lifetime_frame.to_csv(
        LIFETIME_DIR / "resource_lifetime_summary.csv", index=False
    )

    # Duplicate the three primary summaries at the experiment root for easier
    # artifact collection.
    baseline_frame.to_csv(
        OUTPUT_DIR / "phase2_01_baseline_validation_summary.csv", index=False
    )
    contention_frame.to_csv(
        OUTPUT_DIR / "phase2_01_contention_isolation_summary.csv", index=False
    )
    lifetime_frame.to_csv(
        OUTPUT_DIR / "phase2_01_resource_lifetime_summary.csv", index=False
    )

    if SAVE_DETAILED_STAGE_TIMELINE:
        stage_frame.to_csv(
            OUTPUT_DIR / "phase2_01_stage_timeline.csv", index=False
        )
    if SAVE_DETAILED_RESOURCE_INTERVALS:
        interval_frame.to_csv(
            OUTPUT_DIR / "phase2_01_resource_intervals.csv", index=False
        )

    save_baseline_plot(protocols)
    save_contention_heatmap(contention_frame)
    save_timeline_plot(
        stage_frame,
        protocol_name="direct_remote_cx",
        scenario_pattern="resource_lifetime::endpoint_held_through_reset",
        filename="endpoint_lifetime_timeline.png",
    )

    failed_assertions = assertion_frame[~assertion_frame["passed"]]
    manifest = {
        "experiment": "Phase 2.1 — Staged Remote-Operation Validation",
        "output_directory": str(OUTPUT_DIR),
        "protocol_count": len(protocols),
        "protocols": list(protocols),
        "baseline_scenario_count": len(baseline_frame),
        "single_stage_contention_scenario_count": len(contention_frame),
        "resource_lifetime_scenario_count": len(lifetime_frame),
        "stage_record_count": len(stage_frame),
        "resource_interval_count": len(interval_frame),
        "assertion_count": len(assertion_frame),
        "passed_assertions": int(assertion_frame["passed"].sum()),
        "failed_assertions": int((~assertion_frame["passed"]).sum()),
        "all_validations_passed": failed_assertions.empty,
        "controlled_contention_ns": CONTROLLED_CONTENTION_NS,
    }
    with (OUTPUT_DIR / "phase2_01_run_manifest.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(manifest, output_file, indent=2)

    print("\nPhase 2.1 — Staged Remote-Operation Validation")
    print("=" * 58)
    print(validation_summary.to_string(index=False))
    print("\nProtocol baseline latencies:")
    print(
        baseline_frame[
            [
                "protocol_name",
                "nominal_stage_sum_ns",
                "observed_latency_ns",
                "total_wait_ns",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults saved to: {OUTPUT_DIR}")

    if not failed_assertions.empty:
        print("\nFAILED VALIDATIONS:")
        print(
            failed_assertions[
                [
                    "validation_group",
                    "scenario_name",
                    "assertion_name",
                    "expected",
                    "observed",
                ]
            ].to_string(index=False)
        )
        if FAIL_PROCESS_ON_ASSERTION_ERROR:
            sys.exit(1)

    print("\nAll Phase 2.1 causal-timeline validations passed.")


if __name__ == "__main__":
    main()
