#!/usr/bin/env python3
"""
Phase 2.7 — Remote Protocol Comparison
======================================

Purpose
-------
Generalize the Phase-2 leakage mechanisms across two fundamentally different
realizations of the same logical nonlocal operation.  The victim and attacker
issue the same logical remote-CX requests on exactly the same schedules; only
the underlying physical protocol changes.

Primary protocols
-----------------
1. direct_coherent_remote_cx
   A synchronous coherent/microwave-linked realization:

       endpoint (held through cleanup)
           -> switch/path setup
           -> synchronous quantum-link transfer
           -> receiver-side local gate
           -> EXTERNAL COMPLETION
           -> reset/recovery
           -> endpoint reusable

2. entanglement_assisted_remote_cx
   A prefetched entanglement-assisted realization:

       asynchronous EPR generation/refill
           -> finite EPR pool
           -> request consumes stored EPR pair
           -> short endpoint/local-interface use
           -> measurement/readout
           -> classical feedforward
           -> local correction
           -> EXTERNAL COMPLETION
           -> reset/recovery

   EPR generation uses the same conceptual quantum-link resource, but it is
   decoupled from the logical operation.  Consequently victim communication can
   migrate from synchronous link occupancy into persistent EPR depletion/refill
   state.

Normalization
-------------
Both protocols implement the same logical operation, ``logical_remote_cx``.
For a request whose prerequisite communication resource is immediately ready,
both protocols have the same nominal 150 ns critical latency before external
completion and the same 120 ns post-completion cleanup duration.  The release
schedule and logical request count are identical across protocols.

Security question
-----------------
Do Phase-2 conclusions generalize when the remote primitive changes, and if so,
does the leakage mechanism migrate to the resources actually used by the new
protocol?

The experiment therefore compares:
* everything isolated,
* each protocol-used resource shared one at a time,
* representative protocol-specific stacks,
* all protocol-used resources shared.

The output includes a protocol x resource table recording resource use,
occupancy timing, mutual-exclusion semantics, and measured single-resource
leakage.

Black-box boundary
------------------
The attacker-visible export contains only its own protocol name, opaque trace
identifier, release/completion/success, and paired timing/failure outcomes.
Victim workload labels, sharing configuration, resource waits, EPR-pool state,
blocking-owner identity, and evaluator attribution are excluded.

Default output directory
------------------------
blackbox_window_results/phase2/phase2_07_remote_protocol_comparison/

Run
---
Full default run:

    python phase2_07_remote_protocol_comparison.py

Smoke test:

    python phase2_07_remote_protocol_comparison.py --trials 1 \
        --observation-window-ns 5000

Notes
-----
* All service durations are controlled architecture parameters, not vendor
  measurements.
* The entanglement-assisted protocol intentionally uses prefetched EPR state,
  because Phase 2.5 established that this decoupling creates the persistent
  resource-state mechanism we want to test for protocol migration.
* The comparison does not require identical timing fingerprints.  Migration of
  the dominant resource/fingerprint is itself the principal result.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Global settings
# =============================================================================

DEFAULT_TRIALS = 10
DEFAULT_SEED = 2707
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
ATTACKER_FIRST_RELEASE_NS = 30.0
ATTACKER_PERIOD_NS = 420.0
AFFECTED_THRESHOLD_NS = 1e-9
FLOAT_TOLERANCE_NS = 1e-9
FAIL_ON_VALIDATION_ERROR = True
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 1, "mtime": 1}

DEFAULT_OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase2"
    / "phase2_07_remote_protocol_comparison"
)

LOGICAL_OPERATION = "logical_remote_cx"

# Shared-resource vocabulary.  A protocol need not use every resource.
RESOURCE_NAMES = (
    "endpoint",
    "switch_path",
    "quantum_link",
    "epr_pool",
    "epr_generator",
    "readout",
    "feedforward",
    "reset",
)

DIRECT_PROTOCOL = "direct_coherent_remote_cx"
ENTANGLED_PROTOCOL = "entanglement_assisted_remote_cx"

# Carefully normalized service times.
DIRECT_ENDPOINT_PREP_NS = 30.0
DIRECT_SWITCH_SETUP_NS = 15.0
DIRECT_LINK_TRANSFER_NS = 80.0
DIRECT_RECEIVER_GATE_NS = 25.0

ENT_ENDPOINT_ACCESS_NS = 20.0
ENT_READOUT_NS = 70.0
ENT_FEEDFORWARD_NS = 40.0
ENT_CORRECTION_NS = 20.0

RESET_NS = 120.0

DIRECT_CRITICAL_NS = (
    DIRECT_ENDPOINT_PREP_NS
    + DIRECT_SWITCH_SETUP_NS
    + DIRECT_LINK_TRANSFER_NS
    + DIRECT_RECEIVER_GATE_NS
)
ENT_CRITICAL_AFTER_EPR_NS = (
    ENT_ENDPOINT_ACCESS_NS
    + ENT_READOUT_NS
    + ENT_FEEDFORWARD_NS
    + ENT_CORRECTION_NS
)

# EPR-prefetch subsystem used only by the entanglement-assisted realization.
EPR_POOL_CAPACITY = 2
EPR_PREFETCH_TARGET = 2
EPR_GENERATOR_SETUP_NS = 40.0
EPR_LINK_GENERATION_NS = 200.0
EPR_GENERATION_LATENCY_NS = EPR_GENERATOR_SETUP_NS + EPR_LINK_GENERATION_NS
EPR_PAIR_LIFETIME_NS = 2_000.0

BLACKBOX_ALLOWED_COLUMNS = {
    "trace_id",
    "protocol_name",
    "probe_index",
    "release_ns",
    "attacker_only_success",
    "combined_success",
    "attacker_only_completion_ns",
    "combined_completion_ns",
    "attacker_only_turnaround_ns",
    "combined_turnaround_ns",
    "excess_turnaround_ns",
    "affected",
    "speedup",
    "failure_transition",
}


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class ProtocolDefinition:
    protocol_name: str
    protocol_family: str
    logical_operation: str
    description: str
    nominal_critical_latency_ns: float
    postcompletion_cleanup_ns: float
    uses_epr: bool
    used_resources: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolScenario:
    scenario_id: str
    protocol_name: str
    scenario_class: str
    shared_resources: tuple[str, ...]
    description: str

    def shared(self, resource_name: str) -> bool:
        return resource_name in self.shared_resources


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
class RequestState:
    spec: RequestSpec
    protocol_name: str
    success: bool = True
    failure_reason: str = ""
    external_completion_ns: float = math.nan
    cleanup_completion_ns: float = math.nan
    epr_acquired_ns: float = math.nan
    epr_wait_ns: float = 0.0
    epr_miss: bool = False
    epr_pool_level_at_arrival: int = 0
    epr_persistent_victim_state: bool = False
    epr_direct_victim_generation_overlap: bool = False
    epr_last_consumption_tenant_at_arrival: str = ""
    epr_last_consumption_ns_at_arrival: float = math.nan
    resource_wait_ns: dict[str, float] = field(default_factory=lambda: defaultdict(float))


@dataclass
class QueueWaiter:
    actor_id: str
    tenant: str
    actor_kind: str
    stage_tag: str
    arrival_ns: float
    blocking_tenants_at_arrival: tuple[str, ...]


@dataclass
class ActiveResourceUse:
    actor_id: str
    tenant: str
    actor_kind: str
    stage_tag: str
    start_ns: float


@dataclass
class EPRPair:
    pair_id: str
    pool_key: str
    generated_ns: float
    expires_ns: float
    trigger_tenant: str


@dataclass
class GenerationJob:
    job_id: str
    pool_key: str
    trigger_tenant: str
    trigger_kind: str
    created_ns: float
    generator_wait_ns: float = 0.0
    link_wait_ns: float = 0.0
    start_ns: float = math.nan
    end_ns: float = math.nan


@dataclass
class ValidationAssertion:
    validation_group: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


# =============================================================================
# Protocols, scenarios, and workloads
# =============================================================================


def build_protocols() -> dict[str, ProtocolDefinition]:
    direct = ProtocolDefinition(
        protocol_name=DIRECT_PROTOCOL,
        protocol_family="direct_coherent",
        logical_operation=LOGICAL_OPERATION,
        description=(
            "Direct coherent remote CX with synchronous endpoint, switch/path, "
            "and quantum-link occupancy. Endpoint is held through post-completion reset."
        ),
        nominal_critical_latency_ns=DIRECT_CRITICAL_NS,
        postcompletion_cleanup_ns=RESET_NS,
        uses_epr=False,
        used_resources=("endpoint", "switch_path", "quantum_link", "reset"),
    )
    entangled = ProtocolDefinition(
        protocol_name=ENTANGLED_PROTOCOL,
        protocol_family="entanglement_assisted",
        logical_operation=LOGICAL_OPERATION,
        description=(
            "Prefetched entanglement-assisted remote CX. EPR generation/refill uses "
            "the quantum link asynchronously; the logical operation consumes stored "
            "EPR state and then uses endpoint access, readout, feedforward, correction, "
            "and post-completion reset."
        ),
        nominal_critical_latency_ns=ENT_CRITICAL_AFTER_EPR_NS,
        postcompletion_cleanup_ns=RESET_NS,
        uses_epr=True,
        used_resources=(
            "endpoint",
            "quantum_link",
            "epr_pool",
            "epr_generator",
            "readout",
            "feedforward",
            "reset",
        ),
    )
    return {direct.protocol_name: direct, entangled.protocol_name: entangled}


def _scenario(
    protocol_name: str,
    suffix: str,
    scenario_class: str,
    shared_resources: Iterable[str],
    description: str,
) -> ProtocolScenario:
    shared = tuple(sorted(set(shared_resources)))
    unknown = sorted(set(shared) - set(RESOURCE_NAMES))
    if unknown:
        raise ValueError(f"Unknown resource(s): {unknown}")
    return ProtocolScenario(
        scenario_id=f"{protocol_name}__{suffix}",
        protocol_name=protocol_name,
        scenario_class=scenario_class,
        shared_resources=shared,
        description=description,
    )


def build_scenarios(protocols: dict[str, ProtocolDefinition]) -> tuple[ProtocolScenario, ...]:
    rows: list[ProtocolScenario] = []
    for protocol in protocols.values():
        rows.append(
            _scenario(
                protocol.protocol_name,
                "isolated",
                "isolated_control",
                (),
                "All resources used by this protocol are tenant-private.",
            )
        )
        for resource in protocol.used_resources:
            rows.append(
                _scenario(
                    protocol.protocol_name,
                    f"share_{resource}",
                    "single_resource",
                    (resource,),
                    f"Only {resource} is cross-tenant shared.",
                )
            )

        if protocol.protocol_name == DIRECT_PROTOCOL:
            rows.extend(
                [
                    _scenario(
                        protocol.protocol_name,
                        "share_interconnect_stack",
                        "protocol_stack",
                        ("switch_path", "quantum_link"),
                        "Direct protocol switch/path and synchronous quantum link are shared.",
                    ),
                    _scenario(
                        protocol.protocol_name,
                        "all_used_shared",
                        "all_used_shared",
                        protocol.used_resources,
                        "Every resource used by the direct coherent protocol is shared.",
                    ),
                ]
            )
        else:
            rows.extend(
                [
                    _scenario(
                        protocol.protocol_name,
                        "share_epr_management_stack",
                        "protocol_stack",
                        ("epr_pool", "epr_generator", "quantum_link"),
                        "EPR inventory, refill generator, and refill quantum link are shared.",
                    ),
                    _scenario(
                        protocol.protocol_name,
                        "share_measurement_stack",
                        "protocol_stack",
                        ("readout", "feedforward"),
                        "Measurement/readout and classical feedforward are shared.",
                    ),
                    _scenario(
                        protocol.protocol_name,
                        "all_used_shared",
                        "all_used_shared",
                        protocol.used_resources,
                        "Every resource used by the entanglement-assisted protocol is shared.",
                    ),
                ]
            )
    return tuple(rows)


def build_workloads() -> tuple[VictimWorkload, ...]:
    return (
        VictimWorkload(
            "sparse_periodic",
            "Low-rate logical remote-CX operations with broad spacing.",
            "periodic_sparse",
        ),
        VictimWorkload(
            "dense_periodic",
            "Higher-rate periodic logical remote-CX operations near the attacker probe rate.",
            "periodic_dense",
        ),
        VictimWorkload(
            "synchronization_bursty",
            "Three-operation logical remote-CX bursts around synchronization epochs.",
            "synchronization_bursty",
        ),
    )


# =============================================================================
# Deterministic logical schedules
# =============================================================================


def attacker_specs(trial_id: int, workload_name: str, observation_window_ns: float) -> list[RequestSpec]:
    releases = np.arange(
        ATTACKER_FIRST_RELEASE_NS,
        observation_window_ns,
        ATTACKER_PERIOD_NS,
        dtype=float,
    )
    return [
        RequestSpec(
            request_id=f"attacker::{trial_id}::{idx}",
            tenant="attacker",
            ready_ns=float(t),
            request_index=idx,
            workload_name=workload_name,
            trial_id=trial_id,
        )
        for idx, t in enumerate(releases)
    ]


def victim_release_times(
    workload: VictimWorkload,
    *,
    phase_ns: float,
    rng: np.random.Generator,
    observation_window_ns: float,
) -> np.ndarray:
    if workload.release_pattern == "periodic_sparse":
        releases = np.arange(phase_ns, observation_window_ns, 900.0)
    elif workload.release_pattern == "periodic_dense":
        releases = np.arange(phase_ns, observation_window_ns, 470.0)
    elif workload.release_pattern == "synchronization_bursty":
        values: list[float] = []
        base = phase_ns
        offsets = (0.0, 75.0, 150.0)
        while base < observation_window_ns:
            values.extend(base + x for x in offsets)
            base += 1_650.0
        releases = np.asarray(values, dtype=float)
    else:
        raise ValueError(workload.release_pattern)

    releases = releases[(releases >= 0.0) & (releases < observation_window_ns)]
    if len(releases):
        releases = np.maximum(0.0, releases + rng.uniform(-3.0, 3.0, len(releases)))
    return np.sort(releases)


def deterministic_trial_phase(seed: int, workload_name: str, trial_id: int) -> float:
    token = f"{seed}|{workload_name}|{trial_id}|phase".encode()
    digest = hashlib.sha256(token).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64 - 1)
    return 80.0 + 320.0 * value


def deterministic_schedule_seed(seed: int, workload_name: str, trial_id: int) -> int:
    token = f"{seed}|{workload_name}|{trial_id}|schedule".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


def victim_specs(
    workload: VictimWorkload,
    trial_id: int,
    phase_ns: float,
    seed: int,
    observation_window_ns: float,
) -> list[RequestSpec]:
    rng = np.random.default_rng(seed)
    releases = victim_release_times(
        workload,
        phase_ns=phase_ns,
        rng=rng,
        observation_window_ns=observation_window_ns,
    )
    return [
        RequestSpec(
            request_id=f"victim::{trial_id}::{idx}",
            tenant="victim",
            ready_ns=float(t),
            request_index=idx,
            workload_name=workload.workload_name,
            trial_id=trial_id,
        )
        for idx, t in enumerate(releases)
    ]


# =============================================================================
# Generic finite-capacity resource
# =============================================================================


class QueueResource:
    def __init__(self, name: str, key: str, capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.name = name
        self.key = key
        self.capacity = int(capacity)
        self.active: dict[str, ActiveResourceUse] = {}
        self.queue: Deque[QueueWaiter] = deque()

    def request(
        self,
        sim: "ProtocolSimulator",
        *,
        actor_id: str,
        tenant: str,
        actor_kind: str,
        stage_tag: str,
        now: float,
    ) -> None:
        blockers = tuple(sorted({a.tenant for a in self.active.values()}))
        waiter = QueueWaiter(
            actor_id=actor_id,
            tenant=tenant,
            actor_kind=actor_kind,
            stage_tag=stage_tag,
            arrival_ns=float(now),
            blocking_tenants_at_arrival=blockers,
        )
        if len(self.active) < self.capacity:
            self._grant(sim, waiter, now)
        else:
            self.queue.append(waiter)

    def _grant(self, sim: "ProtocolSimulator", waiter: QueueWaiter, now: float) -> None:
        active = ActiveResourceUse(
            actor_id=waiter.actor_id,
            tenant=waiter.tenant,
            actor_kind=waiter.actor_kind,
            stage_tag=waiter.stage_tag,
            start_ns=float(now),
        )
        self.active[waiter.actor_id] = active
        wait_ns = max(0.0, float(now) - waiter.arrival_ns)
        cross_tenant = wait_ns if any(t != waiter.tenant for t in waiter.blocking_tenants_at_arrival) else 0.0
        sim.resource_wait_rows.append(
            {
                "run_kind": sim.run_kind,
                "protocol_name": sim.protocol.protocol_name,
                "scenario_id": sim.scenario.scenario_id,
                "workload_name": sim.workload_name,
                "trial_id": sim.trial_id,
                "actor_id": waiter.actor_id,
                "actor_kind": waiter.actor_kind,
                "tenant": waiter.tenant,
                "resource_name": self.name,
                "resource_key": self.key,
                "stage_tag": waiter.stage_tag,
                "arrival_ns": waiter.arrival_ns,
                "service_start_ns": float(now),
                "wait_ns": wait_ns,
                "cross_tenant_wait_ns": cross_tenant,
                "blocking_tenants_at_arrival": "+".join(waiter.blocking_tenants_at_arrival),
            }
        )
        sim.schedule(
            now,
            "resource_granted",
            {
                "actor_id": waiter.actor_id,
                "actor_kind": waiter.actor_kind,
                "stage_tag": waiter.stage_tag,
                "resource_name": self.name,
                "resource_key": self.key,
                "wait_ns": wait_ns,
                "cross_tenant_wait_ns": cross_tenant,
            },
        )

    def release(self, sim: "ProtocolSimulator", actor_id: str, now: float) -> None:
        active = self.active.pop(actor_id, None)
        if active is None:
            raise RuntimeError(f"{self.name}/{self.key}: release of inactive actor {actor_id}")
        sim.resource_interval_rows.append(
            {
                "run_kind": sim.run_kind,
                "protocol_name": sim.protocol.protocol_name,
                "scenario_id": sim.scenario.scenario_id,
                "workload_name": sim.workload_name,
                "trial_id": sim.trial_id,
                "actor_id": actor_id,
                "actor_kind": active.actor_kind,
                "tenant": active.tenant,
                "resource_name": self.name,
                "resource_key": self.key,
                "stage_tag": active.stage_tag,
                "start_ns": active.start_ns,
                "end_ns": float(now),
                "duration_ns": max(0.0, float(now) - active.start_ns),
            }
        )
        while self.queue and len(self.active) < self.capacity:
            waiter = self.queue.popleft()
            self._grant(sim, waiter, now)


# =============================================================================
# EPR prefetch subsystem
# =============================================================================


class EPRSubsystem:
    def __init__(self, sim: "ProtocolSimulator") -> None:
        self.sim = sim
        self.available: dict[str, Deque[EPRPair]] = defaultdict(deque)
        self.waiters: dict[str, Deque[str]] = defaultdict(deque)
        self.inflight_by_pool: dict[str, int] = defaultdict(int)
        self.jobs: dict[str, GenerationJob] = {}
        self.job_counter = 0
        self.pair_counter = 0
        self.last_consumption_tenant: dict[str, str] = defaultdict(str)
        self.last_consumption_ns: dict[str, float] = defaultdict(lambda: math.nan)

    def pool_key(self, tenant: str) -> str:
        if self.sim.scenario.shared("epr_pool"):
            return "shared::epr_pool"
        return f"{tenant}::epr_pool"

    def generator_key(self, tenant: str) -> str:
        if self.sim.scenario.shared("epr_generator"):
            return "shared::epr_generator"
        return f"{tenant}::epr_generator"

    def link_key(self, tenant: str) -> str:
        if self.sim.scenario.shared("quantum_link"):
            return "shared::quantum_link"
        return f"{tenant}::quantum_link"

    def initialize(self) -> None:
        # Warm prefetched pools remove startup latency from the protocol comparison.
        # Shared-pool scenarios get one common inventory; private-pool scenarios get
        # one inventory per tenant.
        keys = ["shared::epr_pool"] if self.sim.scenario.shared("epr_pool") else [
            "attacker::epr_pool",
            "victim::epr_pool",
        ]
        for pool_key in keys:
            for _ in range(EPR_PREFETCH_TARGET):
                self.pair_counter += 1
                self.available[pool_key].append(
                    EPRPair(
                        pair_id=f"warm_pair::{self.pair_counter}",
                        pool_key=pool_key,
                        generated_ns=0.0,
                        expires_ns=EPR_PAIR_LIFETIME_NS,
                        trigger_tenant="warm_prefetch",
                    )
                )
                self.sim.epr_state_rows.append(
                    {
                        "run_kind": self.sim.run_kind,
                        "protocol_name": self.sim.protocol.protocol_name,
                        "scenario_id": self.sim.scenario.scenario_id,
                        "workload_name": self.sim.workload_name,
                        "trial_id": self.sim.trial_id,
                        "time_ns": 0.0,
                        "event": "warm_prefetch",
                        "pool_key": pool_key,
                        "tenant": "system",
                        "pair_id": f"warm_pair::{self.pair_counter}",
                        "pool_level_after": len(self.available[pool_key]),
                    }
                )

    def _cleanup_expired(self, pool_key: str, now: float, tenant_hint: str) -> None:
        kept: Deque[EPRPair] = deque()
        expired_count = 0
        while self.available[pool_key]:
            pair = self.available[pool_key].popleft()
            if pair.expires_ns <= now:
                expired_count += 1
                self.sim.epr_state_rows.append(
                    {
                        "run_kind": self.sim.run_kind,
                        "protocol_name": self.sim.protocol.protocol_name,
                        "scenario_id": self.sim.scenario.scenario_id,
                        "workload_name": self.sim.workload_name,
                        "trial_id": self.sim.trial_id,
                        "time_ns": float(now),
                        "event": "pair_expired",
                        "pool_key": pool_key,
                        "tenant": tenant_hint,
                        "pair_id": pair.pair_id,
                        "pool_level_after": math.nan,
                    }
                )
            else:
                kept.append(pair)
        self.available[pool_key] = kept
        if expired_count:
            self.ensure_refill(pool_key, now, tenant_hint, "expiration_refill")

    def _active_victim_generation(self) -> bool:
        for job in self.jobs.values():
            if job.trigger_tenant != "victim" or math.isfinite(job.end_ns):
                continue
            return True
        return False

    def acquire(self, request_id: str, tenant: str, now: float) -> None:
        pool_key = self.pool_key(tenant)
        self._cleanup_expired(pool_key, now, tenant)
        state = self.sim.requests[request_id]
        state.epr_pool_level_at_arrival = len(self.available[pool_key])
        state.epr_last_consumption_tenant_at_arrival = self.last_consumption_tenant[pool_key]
        state.epr_last_consumption_ns_at_arrival = self.last_consumption_ns[pool_key]
        state.epr_direct_victim_generation_overlap = bool(
            tenant == "attacker" and self._active_victim_generation()
        )
        state.epr_persistent_victim_state = bool(
            tenant == "attacker"
            and self.sim.scenario.shared("epr_pool")
            and len(self.available[pool_key]) == 0
            and self.last_consumption_tenant[pool_key] == "victim"
            and math.isfinite(self.last_consumption_ns[pool_key])
            and self.last_consumption_ns[pool_key] < now
        )

        if self.available[pool_key]:
            pair = self.available[pool_key].popleft()
            self._consume_pair(pair, request_id, tenant, now, acquisition_source="prefetch_hit")
            self.ensure_refill(pool_key, now, tenant, "consumption_refill")
            return

        state.epr_miss = True
        self.waiters[pool_key].append(request_id)
        self.sim.epr_state_rows.append(
            {
                "run_kind": self.sim.run_kind,
                "protocol_name": self.sim.protocol.protocol_name,
                "scenario_id": self.sim.scenario.scenario_id,
                "workload_name": self.sim.workload_name,
                "trial_id": self.sim.trial_id,
                "time_ns": float(now),
                "event": "pool_miss",
                "pool_key": pool_key,
                "tenant": tenant,
                "pair_id": "",
                "pool_level_after": 0,
            }
        )
        self.ensure_refill(pool_key, now, tenant, "miss_refill")

    def _consume_pair(
        self,
        pair: EPRPair,
        request_id: str,
        tenant: str,
        now: float,
        acquisition_source: str,
    ) -> None:
        state = self.sim.requests[request_id]
        state.epr_acquired_ns = float(now)
        state.epr_wait_ns = max(0.0, float(now) - state.spec.ready_ns)
        self.last_consumption_tenant[pair.pool_key] = tenant
        self.last_consumption_ns[pair.pool_key] = float(now)
        self.sim.epr_state_rows.append(
            {
                "run_kind": self.sim.run_kind,
                "protocol_name": self.sim.protocol.protocol_name,
                "scenario_id": self.sim.scenario.scenario_id,
                "workload_name": self.sim.workload_name,
                "trial_id": self.sim.trial_id,
                "time_ns": float(now),
                "event": "pair_consumed",
                "pool_key": pair.pool_key,
                "tenant": tenant,
                "pair_id": pair.pair_id,
                "pool_level_after": len(self.available[pair.pool_key]),
                "request_id": request_id,
                "acquisition_source": acquisition_source,
            }
        )
        self.sim.schedule(now, "epr_ready", {"request_id": request_id})

    def ensure_refill(self, pool_key: str, now: float, trigger_tenant: str, trigger_kind: str) -> None:
        desired = EPR_PREFETCH_TARGET
        current = len(self.available[pool_key]) + self.inflight_by_pool[pool_key]
        while current < desired:
            self.job_counter += 1
            job_id = f"eprgen::{self.sim.run_kind}::{self.sim.trial_id}::{self.job_counter}"
            job = GenerationJob(
                job_id=job_id,
                pool_key=pool_key,
                trigger_tenant=trigger_tenant,
                trigger_kind=trigger_kind,
                created_ns=float(now),
            )
            self.jobs[job_id] = job
            self.inflight_by_pool[pool_key] += 1
            self._request_generator(job, now)
            current += 1

    def _request_generator(self, job: GenerationJob, now: float) -> None:
        resource = self.sim.get_resource(
            "epr_generator",
            job.trigger_tenant,
            explicit_key=self.generator_key(job.trigger_tenant),
        )
        resource.request(
            self.sim,
            actor_id=job.job_id,
            tenant=job.trigger_tenant,
            actor_kind="epr_generation",
            stage_tag="epr_generator_setup",
            now=now,
        )

    def on_generator_granted(self, job_id: str, resource_key: str, now: float, wait_ns: float) -> None:
        job = self.jobs[job_id]
        job.generator_wait_ns += wait_ns
        if not math.isfinite(job.start_ns):
            job.start_ns = float(now)
        self.sim.schedule(
            now + EPR_GENERATOR_SETUP_NS,
            "epr_generator_done",
            {"job_id": job_id, "resource_key": resource_key},
        )

    def on_generator_done(self, job_id: str, resource_key: str, now: float) -> None:
        self.sim.resources[("epr_generator", resource_key)].release(self.sim, job_id, now)
        job = self.jobs[job_id]
        link = self.sim.get_resource(
            "quantum_link",
            job.trigger_tenant,
            explicit_key=self.link_key(job.trigger_tenant),
        )
        link.request(
            self.sim,
            actor_id=job.job_id,
            tenant=job.trigger_tenant,
            actor_kind="epr_generation",
            stage_tag="epr_link_generation",
            now=now,
        )

    def on_link_granted(self, job_id: str, resource_key: str, now: float, wait_ns: float) -> None:
        job = self.jobs[job_id]
        job.link_wait_ns += wait_ns
        self.sim.schedule(
            now + EPR_LINK_GENERATION_NS,
            "epr_link_done",
            {"job_id": job_id, "resource_key": resource_key},
        )

    def on_link_done(self, job_id: str, resource_key: str, now: float) -> None:
        self.sim.resources[("quantum_link", resource_key)].release(self.sim, job_id, now)
        job = self.jobs[job_id]
        job.end_ns = float(now)
        self.inflight_by_pool[job.pool_key] = max(0, self.inflight_by_pool[job.pool_key] - 1)
        self.pair_counter += 1
        pair = EPRPair(
            pair_id=f"pair::{self.sim.run_kind}::{self.sim.trial_id}::{self.pair_counter}",
            pool_key=job.pool_key,
            generated_ns=float(now),
            expires_ns=float(now) + EPR_PAIR_LIFETIME_NS,
            trigger_tenant=job.trigger_tenant,
        )
        self.sim.generation_rows.append(
            {
                "run_kind": self.sim.run_kind,
                "protocol_name": self.sim.protocol.protocol_name,
                "scenario_id": self.sim.scenario.scenario_id,
                "workload_name": self.sim.workload_name,
                "trial_id": self.sim.trial_id,
                "job_id": job.job_id,
                "pool_key": job.pool_key,
                "trigger_tenant": job.trigger_tenant,
                "trigger_kind": job.trigger_kind,
                "created_ns": job.created_ns,
                "generation_start_ns": job.start_ns,
                "generation_end_ns": job.end_ns,
                "generator_wait_ns": job.generator_wait_ns,
                "link_wait_ns": job.link_wait_ns,
                "total_generation_latency_ns": job.end_ns - job.created_ns,
            }
        )

        if self.waiters[job.pool_key]:
            request_id = self.waiters[job.pool_key].popleft()
            tenant = self.sim.requests[request_id].spec.tenant
            self._consume_pair(pair, request_id, tenant, now, acquisition_source="refill_wait")
        else:
            self.available[job.pool_key].append(pair)
            self.sim.epr_state_rows.append(
                {
                    "run_kind": self.sim.run_kind,
                    "protocol_name": self.sim.protocol.protocol_name,
                    "scenario_id": self.sim.scenario.scenario_id,
                    "workload_name": self.sim.workload_name,
                    "trial_id": self.sim.trial_id,
                    "time_ns": float(now),
                    "event": "pair_stored_after_refill",
                    "pool_key": job.pool_key,
                    "tenant": job.trigger_tenant,
                    "pair_id": pair.pair_id,
                    "pool_level_after": len(self.available[job.pool_key]),
                }
            )
        self.ensure_refill(job.pool_key, now, job.trigger_tenant, "maintain_target")


# =============================================================================
# Protocol simulator
# =============================================================================


class ProtocolSimulator:
    def __init__(
        self,
        *,
        protocol: ProtocolDefinition,
        scenario: ProtocolScenario,
        workload_name: str,
        trial_id: int,
        run_kind: str,
    ) -> None:
        self.protocol = protocol
        self.scenario = scenario
        self.workload_name = workload_name
        self.trial_id = trial_id
        self.run_kind = run_kind
        self.events: list[tuple[float, int, str, dict[str, Any]]] = []
        self.sequence = 0
        self.requests: dict[str, RequestState] = {}
        self.resources: dict[tuple[str, str], QueueResource] = {}
        self.resource_wait_rows: list[dict[str, Any]] = []
        self.resource_interval_rows: list[dict[str, Any]] = []
        self.stage_rows: list[dict[str, Any]] = []
        self.epr_state_rows: list[dict[str, Any]] = []
        self.generation_rows: list[dict[str, Any]] = []
        self.epr = EPRSubsystem(self) if protocol.uses_epr else None

    def schedule(self, time_ns: float, event_type: str, payload: dict[str, Any]) -> None:
        self.sequence += 1
        heapq.heappush(self.events, (float(time_ns), self.sequence, event_type, payload))

    def resource_key(self, resource_name: str, tenant: str) -> str:
        if self.scenario.shared(resource_name):
            return f"shared::{resource_name}"
        return f"{tenant}::{resource_name}"

    def get_resource(
        self,
        resource_name: str,
        tenant: str,
        *,
        explicit_key: Optional[str] = None,
    ) -> QueueResource:
        key = explicit_key or self.resource_key(resource_name, tenant)
        lookup = (resource_name, key)
        if lookup not in self.resources:
            self.resources[lookup] = QueueResource(resource_name, key, capacity=1)
        return self.resources[lookup]

    def request_resource(
        self,
        request_id: str,
        resource_name: str,
        stage_tag: str,
        now: float,
    ) -> None:
        state = self.requests[request_id]
        self.get_resource(resource_name, state.spec.tenant).request(
            self,
            actor_id=request_id,
            tenant=state.spec.tenant,
            actor_kind="remote_request",
            stage_tag=stage_tag,
            now=now,
        )

    def run(self, specs: Iterable[RequestSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for spec in specs:
            self.requests[spec.request_id] = RequestState(spec=spec, protocol_name=self.protocol.protocol_name)
            self.schedule(spec.ready_ns, "request_release", {"request_id": spec.request_id})
        if self.epr is not None:
            self.epr.initialize()

        while self.events:
            now, _, event_type, payload = heapq.heappop(self.events)
            self._handle_event(now, event_type, payload)

        request_rows: list[dict[str, Any]] = []
        for state in self.requests.values():
            turnaround = state.external_completion_ns - state.spec.ready_ns
            row = {
                "run_kind": self.run_kind,
                "protocol_name": self.protocol.protocol_name,
                "scenario_id": self.scenario.scenario_id,
                "workload_name": self.workload_name,
                "trial_id": self.trial_id,
                "request_id": state.spec.request_id,
                "request_index": state.spec.request_index,
                "tenant": state.spec.tenant,
                "release_ns": state.spec.ready_ns,
                "success": state.success,
                "failure_reason": state.failure_reason,
                "external_completion_ns": state.external_completion_ns,
                "cleanup_completion_ns": state.cleanup_completion_ns,
                "turnaround_ns": turnaround,
                "epr_acquired_ns": state.epr_acquired_ns,
                "epr_wait_ns": state.epr_wait_ns,
                "epr_miss": state.epr_miss,
                "epr_pool_level_at_arrival": state.epr_pool_level_at_arrival,
                "epr_persistent_victim_state": state.epr_persistent_victim_state,
                "epr_direct_victim_generation_overlap": state.epr_direct_victim_generation_overlap,
                "epr_last_consumption_tenant_at_arrival": state.epr_last_consumption_tenant_at_arrival,
                "epr_last_consumption_ns_at_arrival": state.epr_last_consumption_ns_at_arrival,
            }
            for resource in RESOURCE_NAMES:
                row[f"wait_{resource}_ns"] = float(state.resource_wait_ns.get(resource, 0.0))
            request_rows.append(row)

        return (
            pd.DataFrame(request_rows),
            pd.DataFrame(self.resource_wait_rows),
            pd.DataFrame(self.resource_interval_rows),
            pd.DataFrame(self.stage_rows),
            pd.DataFrame(self.epr_state_rows),
            pd.DataFrame(self.generation_rows),
        )

    def _log_stage(self, request_id: str, stage_name: str, start_ns: float, end_ns: float) -> None:
        state = self.requests[request_id]
        self.stage_rows.append(
            {
                "run_kind": self.run_kind,
                "protocol_name": self.protocol.protocol_name,
                "scenario_id": self.scenario.scenario_id,
                "workload_name": self.workload_name,
                "trial_id": self.trial_id,
                "request_id": request_id,
                "tenant": state.spec.tenant,
                "stage_name": stage_name,
                "start_ns": float(start_ns),
                "end_ns": float(end_ns),
                "duration_ns": max(0.0, float(end_ns) - float(start_ns)),
            }
        )

    def _record_request_wait(self, request_id: str, resource_name: str, wait_ns: float) -> None:
        if request_id in self.requests:
            self.requests[request_id].resource_wait_ns[resource_name] += max(0.0, wait_ns)

    def _handle_event(self, now: float, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "request_release":
            request_id = payload["request_id"]
            if self.protocol.protocol_name == DIRECT_PROTOCOL:
                self.request_resource(request_id, "endpoint", "direct_endpoint_hold", now)
            else:
                assert self.epr is not None
                self.epr.acquire(request_id, self.requests[request_id].spec.tenant, now)
            return

        if event_type == "epr_ready":
            request_id = payload["request_id"]
            # Private local operation slot is always tenant-isolated.  It is held
            # through reset so shared reset can create a reuse cascade without
            # pretending that the intermodule endpoint remains occupied.
            self.get_resource("local_entangled_slot", self.requests[request_id].spec.tenant).request(
                self,
                actor_id=request_id,
                tenant=self.requests[request_id].spec.tenant,
                actor_kind="remote_request",
                stage_tag="entangled_local_slot_hold",
                now=now,
            )
            return

        if event_type == "resource_granted":
            actor_id = payload["actor_id"]
            actor_kind = payload["actor_kind"]
            stage_tag = payload["stage_tag"]
            resource_name = payload["resource_name"]
            resource_key = payload["resource_key"]
            wait_ns = float(payload["wait_ns"])

            if actor_kind == "epr_generation":
                assert self.epr is not None
                if stage_tag == "epr_generator_setup":
                    self.epr.on_generator_granted(actor_id, resource_key, now, wait_ns)
                elif stage_tag == "epr_link_generation":
                    self.epr.on_link_granted(actor_id, resource_key, now, wait_ns)
                else:
                    raise RuntimeError(stage_tag)
                return

            self._record_request_wait(actor_id, resource_name, wait_ns)

            if self.protocol.protocol_name == DIRECT_PROTOCOL:
                self._handle_direct_grant(now, actor_id, stage_tag, resource_key)
            else:
                self._handle_entangled_grant(now, actor_id, stage_tag, resource_key)
            return

        if event_type == "epr_generator_done":
            assert self.epr is not None
            self.epr.on_generator_done(payload["job_id"], payload["resource_key"], now)
            return

        if event_type == "epr_link_done":
            assert self.epr is not None
            self.epr.on_link_done(payload["job_id"], payload["resource_key"], now)
            return

        if self.protocol.protocol_name == DIRECT_PROTOCOL:
            self._handle_direct_event(now, event_type, payload)
        else:
            self._handle_entangled_event(now, event_type, payload)

    # ------------------------------------------------------------------
    # Direct coherent protocol
    # ------------------------------------------------------------------

    def _handle_direct_grant(self, now: float, request_id: str, stage_tag: str, resource_key: str) -> None:
        if stage_tag == "direct_endpoint_hold":
            self.schedule(now + DIRECT_ENDPOINT_PREP_NS, "direct_endpoint_prepared", {"request_id": request_id})
        elif stage_tag == "direct_switch_hold":
            self.schedule(
                now + DIRECT_SWITCH_SETUP_NS,
                "direct_switch_prepared",
                {"request_id": request_id},
            )
        elif stage_tag == "direct_link_transfer":
            self.schedule(
                now + DIRECT_LINK_TRANSFER_NS,
                "direct_link_done",
                {"request_id": request_id, "resource_key": resource_key},
            )
        elif stage_tag == "direct_reset":
            self.schedule(
                now + RESET_NS,
                "direct_reset_done",
                {"request_id": request_id, "resource_key": resource_key},
            )
        else:
            raise RuntimeError(stage_tag)

    def _handle_direct_event(self, now: float, event_type: str, payload: dict[str, Any]) -> None:
        request_id = payload["request_id"]
        if event_type == "direct_endpoint_prepared":
            self._log_stage(request_id, "endpoint_prepare", now - DIRECT_ENDPOINT_PREP_NS, now)
            self.request_resource(request_id, "switch_path", "direct_switch_hold", now)
        elif event_type == "direct_switch_prepared":
            self._log_stage(request_id, "switch_setup", now - DIRECT_SWITCH_SETUP_NS, now)
            self.request_resource(request_id, "quantum_link", "direct_link_transfer", now)
        elif event_type == "direct_link_done":
            link_key = payload["resource_key"]
            self._log_stage(request_id, "synchronous_quantum_link_transfer", now - DIRECT_LINK_TRANSFER_NS, now)
            self.resources[("quantum_link", link_key)].release(self, request_id, now)
            switch_key = self.resource_key("switch_path", self.requests[request_id].spec.tenant)
            self.resources[("switch_path", switch_key)].release(self, request_id, now)
            self.schedule(now + DIRECT_RECEIVER_GATE_NS, "direct_external_complete", {"request_id": request_id})
        elif event_type == "direct_external_complete":
            self._log_stage(request_id, "receiver_side_gate", now - DIRECT_RECEIVER_GATE_NS, now)
            self.requests[request_id].external_completion_ns = float(now)
            self.request_resource(request_id, "reset", "direct_reset", now)
        elif event_type == "direct_reset_done":
            reset_key = payload["resource_key"]
            self._log_stage(request_id, "postcompletion_reset", now - RESET_NS, now)
            self.resources[("reset", reset_key)].release(self, request_id, now)
            endpoint_key = self.resource_key("endpoint", self.requests[request_id].spec.tenant)
            self.resources[("endpoint", endpoint_key)].release(self, request_id, now)
            self.requests[request_id].cleanup_completion_ns = float(now)
        else:
            raise RuntimeError(event_type)

    # ------------------------------------------------------------------
    # Entanglement-assisted protocol
    # ------------------------------------------------------------------

    def _handle_entangled_grant(self, now: float, request_id: str, stage_tag: str, resource_key: str) -> None:
        if stage_tag == "entangled_local_slot_hold":
            # Slot remains held through reset; endpoint itself is only a short
            # scoped access and therefore does not inherit direct-protocol lifetime.
            self.request_resource(request_id, "endpoint", "entangled_endpoint_access", now)
        elif stage_tag == "entangled_endpoint_access":
            self.schedule(
                now + ENT_ENDPOINT_ACCESS_NS,
                "entangled_endpoint_done",
                {"request_id": request_id, "resource_key": resource_key},
            )
        elif stage_tag == "entangled_readout":
            self.schedule(
                now + ENT_READOUT_NS,
                "entangled_readout_done",
                {"request_id": request_id, "resource_key": resource_key},
            )
        elif stage_tag == "entangled_feedforward":
            self.schedule(
                now + ENT_FEEDFORWARD_NS,
                "entangled_feedforward_done",
                {"request_id": request_id, "resource_key": resource_key},
            )
        elif stage_tag == "entangled_reset":
            self.schedule(
                now + RESET_NS,
                "entangled_reset_done",
                {"request_id": request_id, "resource_key": resource_key},
            )
        else:
            raise RuntimeError(stage_tag)

    def _handle_entangled_event(self, now: float, event_type: str, payload: dict[str, Any]) -> None:
        request_id = payload["request_id"]
        if event_type == "entangled_endpoint_done":
            endpoint_key = payload["resource_key"]
            self._log_stage(request_id, "stored_epr_endpoint_access", now - ENT_ENDPOINT_ACCESS_NS, now)
            self.resources[("endpoint", endpoint_key)].release(self, request_id, now)
            self.request_resource(request_id, "readout", "entangled_readout", now)
        elif event_type == "entangled_readout_done":
            readout_key = payload["resource_key"]
            self._log_stage(request_id, "bell_measurement_readout", now - ENT_READOUT_NS, now)
            self.resources[("readout", readout_key)].release(self, request_id, now)
            self.request_resource(request_id, "feedforward", "entangled_feedforward", now)
        elif event_type == "entangled_feedforward_done":
            ff_key = payload["resource_key"]
            self._log_stage(request_id, "classical_feedforward", now - ENT_FEEDFORWARD_NS, now)
            self.resources[("feedforward", ff_key)].release(self, request_id, now)
            self.schedule(now + ENT_CORRECTION_NS, "entangled_external_complete", {"request_id": request_id})
        elif event_type == "entangled_external_complete":
            self._log_stage(request_id, "receiver_correction", now - ENT_CORRECTION_NS, now)
            self.requests[request_id].external_completion_ns = float(now)
            self.request_resource(request_id, "reset", "entangled_reset", now)
        elif event_type == "entangled_reset_done":
            reset_key = payload["resource_key"]
            self._log_stage(request_id, "postcompletion_reset", now - RESET_NS, now)
            self.resources[("reset", reset_key)].release(self, request_id, now)
            slot_key = f"{self.requests[request_id].spec.tenant}::local_entangled_slot"
            self.resources[("local_entangled_slot", slot_key)].release(self, request_id, now)
            self.requests[request_id].cleanup_completion_ns = float(now)
        else:
            raise RuntimeError(event_type)


# =============================================================================
# Trial execution and pairing
# =============================================================================


def run_one(
    protocol: ProtocolDefinition,
    scenario: ProtocolScenario,
    workload_name: str,
    trial_id: int,
    run_kind: str,
    specs: list[RequestSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = ProtocolSimulator(
        protocol=protocol,
        scenario=scenario,
        workload_name=workload_name,
        trial_id=trial_id,
        run_kind=run_kind,
    )
    return sim.run(specs)


def pair_attacker_trace(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    protocol_name: str,
    scenario_id: str,
    workload_name: str,
    trial_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = attacker_only[attacker_only["tenant"] == "attacker"].copy()
    c = combined[combined["tenant"] == "attacker"].copy()
    merged = a.merge(
        c,
        on="request_index",
        suffixes=("_attacker_only", "_combined"),
        validate="one_to_one",
    )
    excess = (
        merged["turnaround_ns_combined"].to_numpy(dtype=float)
        - merged["turnaround_ns_attacker_only"].to_numpy(dtype=float)
    )
    trace_id = f"trace::{protocol_name}::{scenario_id}::{workload_name}::{trial_id}"
    blackbox = pd.DataFrame(
        {
            "trace_id": trace_id,
            "protocol_name": protocol_name,
            "scenario_id": scenario_id,
            "workload_name": workload_name,
            "trial_id": trial_id,
            "probe_index": merged["request_index"].astype(int),
            "release_ns": merged["release_ns_attacker_only"].astype(float),
            "attacker_only_success": merged["success_attacker_only"].astype(bool),
            "combined_success": merged["success_combined"].astype(bool),
            "attacker_only_completion_ns": merged["external_completion_ns_attacker_only"].astype(float),
            "combined_completion_ns": merged["external_completion_ns_combined"].astype(float),
            "attacker_only_turnaround_ns": merged["turnaround_ns_attacker_only"].astype(float),
            "combined_turnaround_ns": merged["turnaround_ns_combined"].astype(float),
            "excess_turnaround_ns": excess,
            "affected": np.abs(excess) > AFFECTED_THRESHOLD_NS,
            "speedup": excess < -AFFECTED_THRESHOLD_NS,
            "failure_transition": (
                merged["success_attacker_only"].astype(bool)
                != merged["success_combined"].astype(bool)
            ),
        }
    )

    evaluator = blackbox.copy()
    evaluator["epr_miss"] = merged["epr_miss_combined"].astype(bool).to_numpy()
    evaluator["epr_persistent_victim_state"] = merged[
        "epr_persistent_victim_state_combined"
    ].astype(bool).to_numpy()
    evaluator["epr_direct_victim_generation_overlap"] = merged[
        "epr_direct_victim_generation_overlap_combined"
    ].astype(bool).to_numpy()
    evaluator["epr_pool_level_at_arrival"] = merged[
        "epr_pool_level_at_arrival_combined"
    ].astype(float).to_numpy()
    for resource in RESOURCE_NAMES:
        evaluator[f"wait_{resource}_ns"] = merged[f"wait_{resource}_ns_combined"].astype(float).to_numpy()
    return blackbox, evaluator


def victim_slowdown_metrics(victim_only: pd.DataFrame, combined: pd.DataFrame) -> dict[str, float]:
    v = victim_only[victim_only["tenant"] == "victim"].copy()
    c = combined[combined["tenant"] == "victim"].copy()
    if v.empty or c.empty:
        return {
            "victim_mean_request_slowdown": 1.0,
            "victim_makespan_slowdown": 1.0,
            "victim_mean_added_turnaround_ns": 0.0,
        }
    m = v[["request_index", "turnaround_ns"]].merge(
        c[["request_index", "turnaround_ns"]],
        on="request_index",
        suffixes=("_victim_only", "_combined"),
        validate="one_to_one",
    )
    base = m["turnaround_ns_victim_only"].to_numpy(dtype=float)
    combined_t = m["turnaround_ns_combined"].to_numpy(dtype=float)
    ratios = np.divide(combined_t, base, out=np.ones(len(m)), where=base > 0)
    base_start = float(v["release_ns"].min())
    base_makespan = float(v["external_completion_ns"].max() - base_start)
    combined_makespan = float(c["external_completion_ns"].max() - base_start)
    return {
        "victim_mean_request_slowdown": float(np.mean(ratios)),
        "victim_makespan_slowdown": combined_makespan / base_makespan if base_makespan > 0 else 1.0,
        "victim_mean_added_turnaround_ns": float(np.mean(combined_t - base)),
    }


# =============================================================================
# Black-box trace features and classification
# =============================================================================


def longest_true_run(values: Iterable[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def true_run_count(values: Iterable[bool]) -> int:
    count = 0
    previous = False
    for value in values:
        current = bool(value)
        if current and not previous:
            count += 1
        previous = current
    return count


def trace_feature_row(trace: pd.DataFrame) -> dict[str, Any]:
    trace = trace.sort_values("probe_index")
    excess = trace["excess_turnaround_ns"].to_numpy(dtype=float)
    positive = np.maximum(excess, 0.0)
    affected = trace["affected"].to_numpy(dtype=bool)
    finite = excess[np.isfinite(excess)]
    if len(finite) >= 2 and np.std(finite[:-1]) > 0 and np.std(finite[1:]) > 0:
        lag1 = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
    else:
        lag1 = 0.0
    return {
        "protocol_name": str(trace["protocol_name"].iloc[0]),
        "scenario_id": str(trace["scenario_id"].iloc[0]),
        "workload_name": str(trace["workload_name"].iloc[0]),
        "trial_id": int(trace["trial_id"].iloc[0]),
        "probe_count": int(len(trace)),
        "affected_probe_fraction": float(np.mean(affected)) if len(trace) else 0.0,
        "speedup_probe_fraction": float(np.mean(trace["speedup"])) if len(trace) else 0.0,
        "failure_transition_fraction": float(np.mean(trace["failure_transition"])) if len(trace) else 0.0,
        "mean_excess_turnaround_ns": float(np.mean(excess)) if len(excess) else 0.0,
        "mean_absolute_timing_change_ns": float(np.mean(np.abs(excess))) if len(excess) else 0.0,
        "cumulative_positive_excess_ns": float(np.sum(positive)),
        "maximum_positive_excess_ns": float(np.max(positive)) if len(positive) else 0.0,
        "p95_positive_excess_ns": float(np.percentile(positive, 95)) if len(positive) else 0.0,
        "std_excess_turnaround_ns": float(np.std(excess)) if len(excess) else 0.0,
        "longest_affected_run": int(longest_true_run(affected)),
        "affected_run_count": int(true_run_count(affected)),
        "lag1_excess_autocorrelation": lag1,
    }


CLASSIFIER_FEATURES = [
    "affected_probe_fraction",
    "speedup_probe_fraction",
    "failure_transition_fraction",
    "mean_absolute_timing_change_ns",
    "cumulative_positive_excess_ns",
    "maximum_positive_excess_ns",
    "p95_positive_excess_ns",
    "std_excess_turnaround_ns",
    "longest_affected_run",
    "affected_run_count",
    "lag1_excess_autocorrelation",
]


def nearest_centroid_fingerprinting(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for (protocol_name, scenario_id), group in features.groupby(
        ["protocol_name", "scenario_id"], sort=True
    ):
        labels = sorted(group["workload_name"].unique())
        correct = 0
        total = 0
        for test_trial in sorted(group["trial_id"].unique()):
            train = group[group["trial_id"] != test_trial]
            test = group[group["trial_id"] == test_trial]
            if train.empty or test.empty:
                continue
            x_train = train[CLASSIFIER_FEATURES].astype(float).to_numpy()
            mean = np.mean(x_train, axis=0)
            std = np.std(x_train, axis=0)
            std[std < 1e-12] = 1.0
            centroids: dict[str, np.ndarray] = {}
            for label in labels:
                rows = train[train["workload_name"] == label][CLASSIFIER_FEATURES].astype(float).to_numpy()
                if len(rows):
                    centroids[label] = np.mean((rows - mean) / std, axis=0)
            for row in test.itertuples(index=False):
                x = np.array([float(getattr(row, f)) for f in CLASSIFIER_FEATURES])
                x = (x - mean) / std
                distances = {label: float(np.linalg.norm(x - centroid)) for label, centroid in centroids.items()}
                prediction = min(distances.items(), key=lambda kv: (kv[1], kv[0]))[0]
                actual = row.workload_name
                ok = prediction == actual
                correct += int(ok)
                total += 1
                pred_rows.append(
                    {
                        "protocol_name": protocol_name,
                        "scenario_id": scenario_id,
                        "trial_id": int(row.trial_id),
                        "actual_workload": actual,
                        "predicted_workload": prediction,
                        "correct": ok,
                        "distance": distances[prediction],
                    }
                )
        metric_rows.append(
            {
                "protocol_name": protocol_name,
                "scenario_id": scenario_id,
                "classifier": "leave_one_trial_out_nearest_centroid_blackbox_only",
                "sample_count": total,
                "workload_count": len(labels),
                "chance_accuracy": 1.0 / len(labels) if labels else math.nan,
                "accuracy": correct / total if total else math.nan,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


# =============================================================================
# Summaries
# =============================================================================


def scenario_summary(
    trial_summary: pd.DataFrame,
    classifier_metrics: pd.DataFrame,
    scenarios_df: pd.DataFrame,
) -> pd.DataFrame:
    agg = (
        trial_summary.groupby(["protocol_name", "scenario_id"], as_index=False)
        .agg(
            affected_probe_fraction=("affected_probe_fraction", "mean"),
            speedup_probe_fraction=("speedup_probe_fraction", "mean"),
            failure_transition_fraction=("failure_transition_fraction", "mean"),
            mean_excess_turnaround_ns=("mean_excess_turnaround_ns", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            cumulative_positive_excess_ns=("cumulative_positive_excess_ns", "mean"),
            maximum_positive_excess_ns=("maximum_positive_excess_ns", "mean"),
            p95_positive_excess_ns=("p95_positive_excess_ns", "mean"),
            longest_affected_run=("longest_affected_run", "mean"),
            lag1_excess_autocorrelation=("lag1_excess_autocorrelation", "mean"),
            victim_mean_request_slowdown=("victim_mean_request_slowdown", "mean"),
            victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
            epr_miss_fraction=("epr_miss_fraction", "mean"),
            persistent_epr_state_fraction=("persistent_epr_state_fraction", "mean"),
            direct_victim_generation_overlap_fraction=("direct_victim_generation_overlap_fraction", "mean"),
        )
    )
    metrics = classifier_metrics[["protocol_name", "scenario_id", "accuracy", "chance_accuracy"]].rename(
        columns={"accuracy": "classification_accuracy"}
    )
    out = scenarios_df.merge(agg, on=["protocol_name", "scenario_id"], how="left")
    out = out.merge(metrics, on=["protocol_name", "scenario_id"], how="left")
    return out


def build_protocol_resource_table(
    protocols: dict[str, ProtocolDefinition],
    scenario_metrics: pd.DataFrame,
) -> pd.DataFrame:
    semantic: dict[tuple[str, str], dict[str, Any]] = {
        (DIRECT_PROTOCOL, "endpoint"): {
            "used": True,
            "occupancy_phase": "synchronous; acquired at request start and held through reset",
            "nominal_hold_ns": DIRECT_CRITICAL_NS + RESET_NS,
            "role": "communication endpoint / communication-qubit lifetime",
            "state_type": "held critical-path + post-completion",
            "external_feature": "large direct delay and reuse cascades when shared",
        },
        (DIRECT_PROTOCOL, "switch_path"): {
            "used": True,
            "occupancy_phase": "synchronous; setup plus link transfer",
            "nominal_hold_ns": DIRECT_SWITCH_SETUP_NS + DIRECT_LINK_TRANSFER_NS,
            "role": "route/path reservation",
            "state_type": "critical-path",
            "external_feature": "path-queue delay",
        },
        (DIRECT_PROTOCOL, "quantum_link"): {
            "used": True,
            "occupancy_phase": "synchronous during every logical remote CX",
            "nominal_hold_ns": DIRECT_LINK_TRANSFER_NS,
            "role": "coherent/microwave intermodule transfer",
            "state_type": "critical-path",
            "external_feature": "direct transfer latency",
        },
        (DIRECT_PROTOCOL, "reset"): {
            "used": True,
            "occupancy_phase": "after external completion",
            "nominal_hold_ns": RESET_NS,
            "role": "communication resource recovery",
            "state_type": "post-completion persistent reuse state",
            "external_feature": "later-probe delay",
        },
        (ENTANGLED_PROTOCOL, "endpoint"): {
            "used": True,
            "occupancy_phase": "short synchronous local access after EPR acquisition",
            "nominal_hold_ns": ENT_ENDPOINT_ACCESS_NS,
            "role": "consume/access locally stored entangled resource",
            "state_type": "critical-path scoped",
            "external_feature": "short endpoint-access delay",
        },
        (ENTANGLED_PROTOCOL, "quantum_link"): {
            "used": True,
            "occupancy_phase": "asynchronous during EPR refill, decoupled from logical request",
            "nominal_hold_ns": EPR_LINK_GENERATION_NS,
            "role": "entanglement distribution/refill",
            "state_type": "background resource-management path",
            "external_feature": "refill delay / depletion persistence",
        },
        (ENTANGLED_PROTOCOL, "epr_pool"): {
            "used": True,
            "occupancy_phase": "persistent storage between generation and consumption",
            "nominal_hold_ns": EPR_PAIR_LIFETIME_NS,
            "role": "finite prefetched EPR inventory",
            "state_type": "persistent inventory state",
            "external_feature": "pool miss, refill wait, delayed depletion fingerprint",
        },
        (ENTANGLED_PROTOCOL, "epr_generator"): {
            "used": True,
            "occupancy_phase": "asynchronous refill before/after logical requests",
            "nominal_hold_ns": EPR_GENERATOR_SETUP_NS,
            "role": "EPR generation setup/control",
            "state_type": "background resource-management path",
            "external_feature": "refill queueing visible after pool miss",
        },
        (ENTANGLED_PROTOCOL, "readout"): {
            "used": True,
            "occupancy_phase": "synchronous after EPR consumption",
            "nominal_hold_ns": ENT_READOUT_NS,
            "role": "Bell/ancilla measurement readout",
            "state_type": "critical-path",
            "external_feature": "measurement queue delay",
        },
        (ENTANGLED_PROTOCOL, "feedforward"): {
            "used": True,
            "occupancy_phase": "synchronous after measurement",
            "nominal_hold_ns": ENT_FEEDFORWARD_NS,
            "role": "classical coordination / conditional decision",
            "state_type": "critical-path",
            "external_feature": "feedforward queue delay",
        },
        (ENTANGLED_PROTOCOL, "reset"): {
            "used": True,
            "occupancy_phase": "after external completion",
            "nominal_hold_ns": RESET_NS,
            "role": "measurement/local communication ancilla recovery",
            "state_type": "post-completion persistent reuse state",
            "external_feature": "later-probe local-slot delay",
        },
    }

    rows: list[dict[str, Any]] = []
    for protocol in protocols.values():
        for resource in RESOURCE_NAMES:
            item = semantic.get((protocol.protocol_name, resource))
            used = item is not None
            row = {
                "protocol_name": protocol.protocol_name,
                "protocol_family": protocol.protocol_family,
                "logical_operation": protocol.logical_operation,
                "resource_name": resource,
                "used_by_protocol": used,
                "occupancy_phase": item["occupancy_phase"] if used else "not used",
                "nominal_hold_ns": item["nominal_hold_ns"] if used else 0.0,
                "role": item["role"] if used else "not on this protocol path",
                "state_type": item["state_type"] if used else "none",
                "possible_external_feature": item["external_feature"] if used else "none",
            }
            single_id = f"{protocol.protocol_name}__share_{resource}"
            match = scenario_metrics[scenario_metrics["scenario_id"] == single_id]
            if not match.empty:
                r = match.iloc[0]
                row.update(
                    {
                        "single_share_affected_probe_fraction": float(r["affected_probe_fraction"]),
                        "single_share_mean_abs_timing_change_ns": float(r["mean_absolute_timing_change_ns"]),
                        "single_share_maximum_positive_excess_ns": float(r["maximum_positive_excess_ns"]),
                        "single_share_longest_affected_run": float(r["longest_affected_run"]),
                        "single_share_classification_accuracy": float(r["classification_accuracy"]),
                    }
                )
            else:
                row.update(
                    {
                        "single_share_affected_probe_fraction": math.nan,
                        "single_share_mean_abs_timing_change_ns": math.nan,
                        "single_share_maximum_positive_excess_ns": math.nan,
                        "single_share_longest_affected_run": math.nan,
                        "single_share_classification_accuracy": math.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def protocol_comparison_summary(
    protocols: dict[str, ProtocolDefinition],
    scenario_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol in protocols.values():
        p = scenario_metrics[scenario_metrics["protocol_name"] == protocol.protocol_name]
        singles = p[p["scenario_class"] == "single_resource"]
        dominant = singles.sort_values(
            ["mean_absolute_timing_change_ns", "affected_probe_fraction"], ascending=False
        ).head(1)
        isolated = p[p["scenario_class"] == "isolated_control"].head(1)
        all_shared = p[p["scenario_class"] == "all_used_shared"].head(1)
        row: dict[str, Any] = {
            "protocol_name": protocol.protocol_name,
            "protocol_family": protocol.protocol_family,
            "logical_operation": protocol.logical_operation,
            "nominal_critical_latency_ns": protocol.nominal_critical_latency_ns,
            "postcompletion_cleanup_ns": protocol.postcompletion_cleanup_ns,
            "uses_epr": protocol.uses_epr,
            "used_resource_count": len(protocol.used_resources),
            "used_resources": "+".join(protocol.used_resources),
        }
        if not isolated.empty:
            r = isolated.iloc[0]
            row.update(
                {
                    "isolated_affected_probe_fraction": float(r["affected_probe_fraction"]),
                    "isolated_mean_abs_timing_change_ns": float(r["mean_absolute_timing_change_ns"]),
                    "isolated_classification_accuracy": float(r["classification_accuracy"]),
                }
            )
        if not all_shared.empty:
            r = all_shared.iloc[0]
            row.update(
                {
                    "all_shared_affected_probe_fraction": float(r["affected_probe_fraction"]),
                    "all_shared_mean_abs_timing_change_ns": float(r["mean_absolute_timing_change_ns"]),
                    "all_shared_longest_affected_run": float(r["longest_affected_run"]),
                    "all_shared_victim_request_slowdown": float(r["victim_mean_request_slowdown"]),
                    "all_shared_victim_makespan_slowdown": float(r["victim_makespan_slowdown"]),
                    "all_shared_classification_accuracy": float(r["classification_accuracy"]),
                    "all_shared_persistent_epr_state_fraction": float(r["persistent_epr_state_fraction"]),
                }
            )
        if not dominant.empty:
            r = dominant.iloc[0]
            shared_resource = str(r["shared_resources"])
            row.update(
                {
                    "dominant_single_resource": shared_resource,
                    "dominant_single_affected_probe_fraction": float(r["affected_probe_fraction"]),
                    "dominant_single_mean_abs_timing_change_ns": float(r["mean_absolute_timing_change_ns"]),
                    "dominant_single_longest_affected_run": float(r["longest_affected_run"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def protocol_migration_summary(resource_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol_name, group in resource_table.groupby("protocol_name", sort=True):
        used = group[group["used_by_protocol"]].dropna(subset=["single_share_mean_abs_timing_change_ns"])
        if used.empty:
            continue
        ranked = used.sort_values(
            ["single_share_mean_abs_timing_change_ns", "single_share_affected_probe_fraction"],
            ascending=False,
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "protocol_name": protocol_name,
                    "rank_by_mean_abs_timing": rank,
                    "resource_name": row["resource_name"],
                    "resource_role": row["role"],
                    "state_type": row["state_type"],
                    "affected_probe_fraction": row["single_share_affected_probe_fraction"],
                    "mean_abs_timing_change_ns": row["single_share_mean_abs_timing_change_ns"],
                    "longest_affected_run": row["single_share_longest_affected_run"],
                    "classification_accuracy": row["single_share_classification_accuracy"],
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Validation
# =============================================================================


def _assertion(group: str, name: str, passed: bool, expected: Any, observed: Any, details: str = "") -> dict[str, Any]:
    return asdict(
        ValidationAssertion(
            validation_group=group,
            assertion_name=name,
            passed=bool(passed),
            expected=str(expected),
            observed=str(observed),
            details=details,
        )
    )


def validate_no_capacity_overlap(intervals: pd.DataFrame) -> bool:
    if intervals.empty:
        return True
    # Resource calendars are independent across protocol/scenario/workload/trial
    # and attacker-only/victim-only/combined executions.  Validate overlap only
    # within one actual simulator run and one physical resource key.
    group_cols = [
        "run_kind",
        "protocol_name",
        "scenario_id",
        "workload_name",
        "trial_id",
        "resource_name",
        "resource_key",
    ]
    for _, group in intervals.groupby(group_cols):
        g = group.sort_values(["start_ns", "end_ns"])
        ends = g["end_ns"].to_numpy(dtype=float)
        starts = g["start_ns"].to_numpy(dtype=float)
        if len(g) > 1 and np.any(starts[1:] < ends[:-1] - FLOAT_TOLERANCE_NS):
            return False
    return True


def build_validations(
    protocols: dict[str, ProtocolDefinition],
    scenarios_df: pd.DataFrame,
    trial_summary: pd.DataFrame,
    blackbox: pd.DataFrame,
    attacker_visible: pd.DataFrame,
    intervals: pd.DataFrame,
    resource_waits: pd.DataFrame,
    epr_events: pd.DataFrame,
    generation_events: pd.DataFrame,
    phase_schedule: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append(_assertion(
        "protocol_definition",
        "two_primary_protocols_present",
        set(protocols) == {DIRECT_PROTOCOL, ENTANGLED_PROTOCOL},
        {DIRECT_PROTOCOL, ENTANGLED_PROTOCOL},
        set(protocols),
    ))
    logical_ops = {p.logical_operation for p in protocols.values()}
    rows.append(_assertion(
        "normalization",
        "same_logical_operation",
        logical_ops == {LOGICAL_OPERATION},
        {LOGICAL_OPERATION},
        logical_ops,
    ))
    criticals = [p.nominal_critical_latency_ns for p in protocols.values()]
    rows.append(_assertion(
        "normalization",
        "equal_nominal_critical_latency_after_prerequisite",
        max(criticals) - min(criticals) <= FLOAT_TOLERANCE_NS,
        "all protocols equal",
        criticals,
        "Direct starts with synchronous transfer resources; entanglement-assisted starts after EPR availability.",
    ))
    cleanups = [p.postcompletion_cleanup_ns for p in protocols.values()]
    rows.append(_assertion(
        "normalization",
        "equal_postcompletion_cleanup",
        max(cleanups) - min(cleanups) <= FLOAT_TOLERANCE_NS,
        "all protocols equal",
        cleanups,
    ))

    # The same workload/trial phase schedule is global and reused by both protocols.
    duplicate_phase_counts = phase_schedule.groupby(["workload_name", "trial_id"])["phase_ns"].nunique()
    rows.append(_assertion(
        "normalization",
        "one_shared_logical_release_schedule_per_workload_trial",
        bool((duplicate_phase_counts == 1).all()),
        "exactly one phase/schedule definition per workload+trial",
        f"max unique phases={int(duplicate_phase_counts.max()) if len(duplicate_phase_counts) else 0}",
    ))

    # Every protocol's isolated control should be differentially zero.
    iso = trial_summary[trial_summary["scenario_class"] == "isolated_control"]
    for protocol_name in protocols:
        g = iso[iso["protocol_name"] == protocol_name]
        affected = float(g["affected_probe_fraction"].mean()) if not g.empty else math.nan
        change = float(g["mean_absolute_timing_change_ns"].mean()) if not g.empty else math.nan
        rows.append(_assertion(
            "negative_control",
            f"{protocol_name}_isolated_zero_leakage",
            (not g.empty) and affected <= 1e-12 and change <= 1e-9,
            "affected=0 and mean_abs_timing=0",
            f"affected={affected}, mean_abs={change}",
        ))

    # All-used-shared should produce some externally visible difference for each protocol.
    all_shared = trial_summary[trial_summary["scenario_class"] == "all_used_shared"]
    for protocol_name in protocols:
        g = all_shared[all_shared["protocol_name"] == protocol_name]
        affected = float(g["affected_probe_fraction"].mean()) if not g.empty else 0.0
        rows.append(_assertion(
            "positive_control",
            f"{protocol_name}_all_used_shared_nonzero_channel",
            affected > 0.0,
            "> 0 affected fraction",
            affected,
        ))

    # Direct protocol must not create EPR lifecycle/generation events.
    direct_epr = epr_events[epr_events["protocol_name"] == DIRECT_PROTOCOL] if not epr_events.empty and "protocol_name" in epr_events else pd.DataFrame()
    direct_gen = generation_events[generation_events["protocol_name"] == DIRECT_PROTOCOL] if not generation_events.empty and "protocol_name" in generation_events else pd.DataFrame()
    rows.append(_assertion(
        "protocol_specificity",
        "direct_protocol_has_no_epr_state_or_generation",
        direct_epr.empty and direct_gen.empty,
        "zero EPR events",
        f"state={len(direct_epr)}, generation={len(direct_gen)}",
    ))

    ent_epr = epr_events[epr_events["protocol_name"] == ENTANGLED_PROTOCOL] if not epr_events.empty and "protocol_name" in epr_events else pd.DataFrame()
    ent_gen = generation_events[generation_events["protocol_name"] == ENTANGLED_PROTOCOL] if not generation_events.empty and "protocol_name" in generation_events else pd.DataFrame()
    rows.append(_assertion(
        "protocol_specificity",
        "entanglement_assisted_protocol_uses_epr_state_and_generation",
        (not ent_epr.empty) and (not ent_gen.empty),
        "nonzero EPR state and generation events",
        f"state={len(ent_epr)}, generation={len(ent_gen)}",
    ))

    # Direct protocol should never request readout/feedforward; entangled should never request switch_path.
    if resource_waits.empty:
        direct_wrong = ent_wrong = 0
    else:
        direct_wrong = int(resource_waits[
            (resource_waits["protocol_name"] == DIRECT_PROTOCOL)
            & (resource_waits["resource_name"].isin(["readout", "feedforward", "epr_generator"]))
        ].shape[0])
        ent_wrong = int(resource_waits[
            (resource_waits["protocol_name"] == ENTANGLED_PROTOCOL)
            & (resource_waits["resource_name"] == "switch_path")
        ].shape[0])
    rows.append(_assertion(
        "protocol_specificity",
        "protocols_only_request_their_defined_resources",
        direct_wrong == 0 and ent_wrong == 0,
        "zero invalid resource requests",
        f"direct_wrong={direct_wrong}, entangled_wrong={ent_wrong}",
    ))

    rows.append(_assertion(
        "resource_calendar",
        "no_capacity_one_resource_overlap",
        validate_no_capacity_overlap(intervals),
        "no overlap on any capacity-1 resource key",
        "checked all resource intervals",
    ))

    # Attacker-visible export must be a strict subset of allowed columns.
    unexpected = sorted(set(attacker_visible.columns) - BLACKBOX_ALLOWED_COLUMNS)
    rows.append(_assertion(
        "blackbox_boundary",
        "attacker_visible_columns_are_blackbox_only",
        len(unexpected) == 0,
        sorted(BLACKBOX_ALLOWED_COLUMNS),
        sorted(attacker_visible.columns),
        f"unexpected={unexpected}",
    ))

    # Master blackbox trace must contain the paired external timing columns.
    required_blackbox = {
        "protocol_name", "probe_index", "release_ns",
        "attacker_only_completion_ns", "combined_completion_ns",
        "excess_turnaround_ns", "affected",
    }
    rows.append(_assertion(
        "blackbox_boundary",
        "master_blackbox_contains_required_external_observables",
        required_blackbox.issubset(set(blackbox.columns)),
        sorted(required_blackbox),
        sorted(set(blackbox.columns) & required_blackbox),
    ))

    # Scenarios may share only resources actually used by that protocol.
    okay = True
    bad: list[str] = []
    for row in scenarios_df.itertuples(index=False):
        used = set(protocols[row.protocol_name].used_resources)
        shared = set(str(row.shared_resources).split("+")) if row.shared_resources not in ("", "none") else set()
        if not shared.issubset(used):
            okay = False
            bad.append(row.scenario_id)
    rows.append(_assertion(
        "scenario_definition",
        "sharing_toggles_are_protocol_relevant",
        okay,
        "every shared resource is used by its protocol",
        bad if bad else "all valid",
    ))

    return pd.DataFrame(rows)


# =============================================================================
# Experiment driver
# =============================================================================


def run_experiment(
    *,
    output_dir: Path,
    trials: int,
    seed: int,
    observation_window_ns: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    protocols = build_protocols()
    scenarios = build_scenarios(protocols)
    workloads = build_workloads()

    scenario_rows = []
    for s in scenarios:
        protocol = protocols[s.protocol_name]
        scenario_rows.append(
            {
                **asdict(s),
                "shared_resources": "+".join(s.shared_resources) if s.shared_resources else "none",
                "shared_resource_count": len(s.shared_resources),
                "protocol_family": protocol.protocol_family,
                "logical_operation": protocol.logical_operation,
                "nominal_critical_latency_ns": protocol.nominal_critical_latency_ns,
                "postcompletion_cleanup_ns": protocol.postcompletion_cleanup_ns,
            }
        )
    scenarios_df = pd.DataFrame(scenario_rows)

    phase_rows: list[dict[str, Any]] = []
    logical_schedules: dict[tuple[str, int], tuple[list[RequestSpec], list[RequestSpec]]] = {}
    for workload in workloads:
        for trial_id in range(trials):
            phase = deterministic_trial_phase(seed, workload.workload_name, trial_id)
            schedule_seed = deterministic_schedule_seed(seed, workload.workload_name, trial_id)
            a_specs = attacker_specs(trial_id, workload.workload_name, observation_window_ns)
            v_specs = victim_specs(
                workload,
                trial_id,
                phase,
                schedule_seed,
                observation_window_ns,
            )
            logical_schedules[(workload.workload_name, trial_id)] = (a_specs, v_specs)
            phase_rows.append(
                {
                    "workload_name": workload.workload_name,
                    "trial_id": trial_id,
                    "phase_ns": phase,
                    "schedule_seed": schedule_seed,
                    "attacker_logical_request_count": len(a_specs),
                    "victim_logical_request_count": len(v_specs),
                    "attacker_release_hash": hashlib.sha256(
                        "|".join(f"{x.ready_ns:.9f}" for x in a_specs).encode()
                    ).hexdigest(),
                    "victim_release_hash": hashlib.sha256(
                        "|".join(f"{x.ready_ns:.9f}" for x in v_specs).encode()
                    ).hexdigest(),
                }
            )
    phase_schedule = pd.DataFrame(phase_rows)

    blackbox_rows: list[pd.DataFrame] = []
    evaluator_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    wait_rows: list[pd.DataFrame] = []
    interval_rows: list[pd.DataFrame] = []
    stage_rows: list[pd.DataFrame] = []
    epr_rows: list[pd.DataFrame] = []
    generation_rows: list[pd.DataFrame] = []

    total = len(scenarios) * len(workloads) * trials
    completed = 0

    for scenario in scenarios:
        protocol = protocols[scenario.protocol_name]
        for workload in workloads:
            for trial_id in range(trials):
                a_specs, v_specs = logical_schedules[(workload.workload_name, trial_id)]

                attacker_only, aw, ai, ast, ae, ag = run_one(
                    protocol, scenario, workload.workload_name, trial_id, "attacker_only", list(a_specs)
                )
                victim_only, vw, vi, vst, ve, vg = run_one(
                    protocol, scenario, workload.workload_name, trial_id, "victim_only", list(v_specs)
                )
                combined, cw, ci, cst, ce, cg = run_one(
                    protocol,
                    scenario,
                    workload.workload_name,
                    trial_id,
                    "combined",
                    sorted(list(a_specs) + list(v_specs), key=lambda x: (x.ready_ns, x.tenant, x.request_index)),
                )

                trace, evaluator = pair_attacker_trace(
                    attacker_only,
                    combined,
                    protocol_name=protocol.protocol_name,
                    scenario_id=scenario.scenario_id,
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                )
                blackbox_rows.append(trace)
                evaluator_rows.append(evaluator)

                features = trace_feature_row(trace)
                combined_att = combined[combined["tenant"] == "attacker"]
                features.update(
                    {
                        "epr_miss_fraction": float(combined_att["epr_miss"].mean()) if len(combined_att) else 0.0,
                        "persistent_epr_state_fraction": float(combined_att["epr_persistent_victim_state"].mean()) if len(combined_att) else 0.0,
                        "direct_victim_generation_overlap_fraction": float(combined_att["epr_direct_victim_generation_overlap"].mean()) if len(combined_att) else 0.0,
                    }
                )
                feature_rows.append(features)
                slowdown = victim_slowdown_metrics(victim_only, combined)
                trial_rows.append(
                    {
                        **features,
                        **slowdown,
                        "scenario_class": scenario.scenario_class,
                        "shared_resources": "+".join(scenario.shared_resources) if scenario.shared_resources else "none",
                    }
                )

                for df in (aw, vw, cw):
                    if not df.empty:
                        wait_rows.append(df)
                for df in (ai, vi, ci):
                    if not df.empty:
                        interval_rows.append(df)
                for df in (ast, vst, cst):
                    if not df.empty:
                        stage_rows.append(df)
                for df in (ae, ve, ce):
                    if not df.empty:
                        epr_rows.append(df)
                for df in (ag, vg, cg):
                    if not df.empty:
                        generation_rows.append(df)

                completed += 1
                if completed % max(1, total // 10) == 0 or completed == total:
                    print(f"Progress: {completed}/{total} trial tuples")

    blackbox = pd.concat(blackbox_rows, ignore_index=True) if blackbox_rows else pd.DataFrame()
    evaluator = pd.concat(evaluator_rows, ignore_index=True) if evaluator_rows else pd.DataFrame()
    features_df = pd.DataFrame(feature_rows)
    trial_summary = pd.DataFrame(trial_rows)
    waits = pd.concat(wait_rows, ignore_index=True) if wait_rows else pd.DataFrame()
    intervals = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
    stages = pd.concat(stage_rows, ignore_index=True) if stage_rows else pd.DataFrame()
    epr_events = pd.concat(epr_rows, ignore_index=True) if epr_rows else pd.DataFrame()
    generation_events = pd.concat(generation_rows, ignore_index=True) if generation_rows else pd.DataFrame()

    classifier_metrics, classifier_predictions = nearest_centroid_fingerprinting(features_df)
    scenario_metrics = scenario_summary(trial_summary, classifier_metrics, scenarios_df)
    resource_table = build_protocol_resource_table(protocols, scenario_metrics)
    comparison_summary = protocol_comparison_summary(protocols, scenario_metrics)
    migration_summary = protocol_migration_summary(resource_table)

    # Internal resource wait summary: attacker cross-tenant waits only.
    resource_wait_summary = pd.DataFrame()
    if not waits.empty:
        w = waits.copy()
        resource_wait_summary = (
            w.groupby(["protocol_name", "scenario_id", "resource_name", "actor_kind"], as_index=False)
            .agg(
                acquisition_count=("actor_id", "size"),
                total_wait_ns=("wait_ns", "sum"),
                total_cross_tenant_wait_ns=("cross_tenant_wait_ns", "sum"),
                mean_wait_ns=("wait_ns", "mean"),
                maximum_wait_ns=("wait_ns", "max"),
            )
        )

    protocol_definition_df = pd.DataFrame([asdict(p) for p in protocols.values()])
    protocol_definition_df["used_resources"] = protocol_definition_df["used_resources"].apply(lambda x: "+".join(x))

    # Strict black-box export: protocol is known to the attacker, but scenario,
    # workload, and trial evaluator labels are not.
    attacker_cols = [
        "trace_id", "protocol_name", "probe_index", "release_ns",
        "attacker_only_success", "combined_success",
        "attacker_only_completion_ns", "combined_completion_ns",
        "attacker_only_turnaround_ns", "combined_turnaround_ns",
        "excess_turnaround_ns", "affected", "speedup", "failure_transition",
    ]
    attacker_visible = blackbox[attacker_cols].copy()

    validations = build_validations(
        protocols,
        scenarios_df,
        trial_summary,
        blackbox,
        attacker_visible,
        intervals,
        waits,
        epr_events,
        generation_events,
        phase_schedule,
    )
    all_passed = bool(validations["passed"].all()) if not validations.empty else False
    validation_summary = pd.DataFrame(
        [
            {
                "validation_assertion_count": int(len(validations)),
                "passed_assertions": int(validations["passed"].sum()),
                "failed_assertions": int((~validations["passed"]).sum()),
                "all_validations_passed": all_passed,
            }
        ]
    )

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------
    protocol_definition_df.to_csv(output_dir / "phase2_07_protocol_definitions.csv", index=False)
    scenarios_df.to_csv(output_dir / "phase2_07_configuration_table.csv", index=False)
    phase_schedule.to_csv(output_dir / "phase2_07_trial_phase_schedule.csv", index=False)
    trial_summary.to_csv(output_dir / "phase2_07_trial_summary.csv", index=False)
    features_df.to_csv(output_dir / "phase2_07_trace_features.csv", index=False)
    scenario_metrics.to_csv(output_dir / "phase2_07_scenario_summary.csv", index=False)
    resource_table.to_csv(output_dir / "phase2_07_protocol_resource_table.csv", index=False)
    comparison_summary.to_csv(output_dir / "phase2_07_protocol_comparison_summary.csv", index=False)
    migration_summary.to_csv(output_dir / "phase2_07_protocol_migration_summary.csv", index=False)
    classifier_metrics.to_csv(output_dir / "phase2_07_workload_fingerprint_metrics.csv", index=False)
    classifier_predictions.to_csv(output_dir / "phase2_07_workload_fingerprint_predictions.csv", index=False)
    blackbox.to_csv(output_dir / "phase2_07_blackbox_trace_summary.csv", index=False)
    attacker_visible.to_csv(output_dir / "phase2_07_attacker_visible_trace.csv", index=False)
    evaluator.to_csv(output_dir / "phase2_07_evaluator_trace_attribution.csv.gz", index=False, compression=GZIP_COMPRESSION)

    if not resource_wait_summary.empty:
        resource_wait_summary.to_csv(output_dir / "phase2_07_resource_wait_summary.csv", index=False)
    if not waits.empty:
        waits.to_csv(output_dir / "phase2_07_resource_wait_events.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not intervals.empty:
        intervals.to_csv(output_dir / "phase2_07_resource_intervals.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not stages.empty:
        stages.to_csv(output_dir / "phase2_07_stage_records.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not epr_events.empty:
        epr_events.to_csv(output_dir / "phase2_07_epr_state_events.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not generation_events.empty:
        generation_events.to_csv(output_dir / "phase2_07_epr_generation_events.csv.gz", index=False, compression=GZIP_COMPRESSION)

    validations.to_csv(output_dir / "phase2_07_validation_assertions.csv", index=False)
    validation_summary.to_csv(output_dir / "phase2_07_validation_summary.csv", index=False)

    manifest = {
        "experiment": "Phase 2.7 — Remote Protocol Comparison",
        "output_directory": str(output_dir),
        "logical_operation": LOGICAL_OPERATION,
        "protocol_count": len(protocols),
        "protocols": list(protocols),
        "scenario_count": len(scenarios),
        "workload_count": len(workloads),
        "trial_count_per_workload_configuration": trials,
        "scenario_workload_trial_tuples": len(scenarios) * len(workloads) * trials,
        "observation_window_ns": observation_window_ns,
        "probe_period_ns": ATTACKER_PERIOD_NS,
        "normalized_nominal_critical_latency_ns": DIRECT_CRITICAL_NS,
        "postcompletion_cleanup_ns": RESET_NS,
        "epr_pool_capacity": EPR_POOL_CAPACITY,
        "epr_prefetch_target": EPR_PREFETCH_TARGET,
        "epr_generation_latency_ns": EPR_GENERATION_LATENCY_NS,
        "epr_pair_lifetime_ns": EPR_PAIR_LIFETIME_NS,
        "validation_assertion_count": int(len(validations)),
        "passed_assertions": int(validations["passed"].sum()),
        "failed_assertions": int((~validations["passed"]).sum()),
        "all_validations_passed": all_passed,
        "blackbox_columns": sorted(BLACKBOX_ALLOWED_COLUMNS),
    }
    (output_dir / "phase2_07_run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nPhase 2.7 complete")
    print(f"Output directory: {output_dir}")
    print(f"Protocols: {len(protocols)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Trial tuples: {len(trial_summary)}")
    print(f"Validation: {int(validations['passed'].sum())}/{len(validations)} passed")
    if not all_passed:
        failed = validations[~validations["passed"]]
        print("Failed validations:")
        print(failed[["validation_group", "assertion_name", "expected", "observed"]].to_string(index=False))
        if FAIL_ON_VALIDATION_ERROR:
            raise RuntimeError("Phase 2.7 validation failed")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.7 remote protocol comparison")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--observation-window-ns", type=float, default=DEFAULT_OBSERVATION_WINDOW_NS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.observation_window_ns <= ATTACKER_PERIOD_NS:
        raise ValueError("Observation window is too short")
    run_experiment(
        output_dir=args.output_dir,
        trials=args.trials,
        seed=args.seed,
        observation_window_ns=args.observation_window_ns,
    )


if __name__ == "__main__":
    main()
