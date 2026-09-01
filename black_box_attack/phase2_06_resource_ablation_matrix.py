#!/usr/bin/env python3
"""
Phase 2.6 — Resource Ablation Matrix
====================================

Purpose
-------
Tie together the endpoint, interconnect, backend-control, reset/recovery, and
EPR-management channels established in Phases 2.2–2.5 using one controlled
remote-operation model.  Each candidate resource can be tenant-private or
cross-tenant shared while every other architectural parameter is held fixed.

The primary experiment is the complete 2^7 factorial matrix over:

    endpoint
    switch_path
    link
    readout
    feedforward
    reset
    epr_pool

This makes it possible to answer causal systems questions that cannot be
answered by isolated sweeps alone:

* Sufficiency: does sharing resource R alone create a black-box channel?
* Necessity: how much leakage disappears when R alone is privatized from the
  all-shared configuration?
* Dominance/masking: does R1+R2 leak no more than the stronger single resource?
* Synergy: do individually weak resources interact to create a larger channel?
* Composition: which combinations preserve temporal/failure fingerprints?

Remote-operation model
----------------------
The model intentionally combines mechanisms already validated separately:

    EPR pair acquisition / prefetched pool
        -> endpoint/communication-qubit acquisition (held through cleanup)
        -> switch/path acquisition
        -> link transmission
        -> receiver-side operation
        -> measurement/readout
        -> classical feedforward
        -> conditional control
        -> EXTERNAL COMPLETION
        -> reset/recovery
        -> endpoint released for reuse

The endpoint token is held through post-completion reset.  Consequently a
shared reset engine can delay endpoint reuse and affect a later attacker probe,
matching the stateful reuse mechanism established in Phase 2.4.  The EPR pool
is prefetched and asynchronously refilled.  Therefore a victim can consume a
pair and leave a persistent depletion/refill state that is visible to a later
attacker probe, matching Phase 2.5.

Black-box boundary
------------------
The attacker-visible trace contains only its own release, success/completion,
turnaround, and paired timing changes.  Internal resource waits, blocker
identity, EPR-pool state, workload labels, and sharing configuration are
analysis/evaluator-only.

Default output directory
------------------------
blackbox_window_results/phase2/phase2_06_resource_ablation_matrix/

Run
---
Full factorial (recommended paper experiment):

    python phase2_06_resource_ablation_matrix.py

Quick curated matrix:

    python phase2_06_resource_ablation_matrix.py --matrix core

Smoke test:

    python phase2_06_resource_ablation_matrix.py --matrix core --trials 1 \\
        --observation-window-ns 5000

Notes
-----
* Numerical service durations are controlled architecture parameters, not
  vendor measurements.
* Primary runs are deterministic apart from the controlled victim phase/jitter.
  The same workload schedule is reused across every ablation configuration.
* The full factorial is required for marginal-effect and pairwise-interaction
  calculations.  Core mode still produces trace/configuration summaries but
  skips factorial-only analyses where the required counterfactual is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Global configuration
# =============================================================================

DEFAULT_TRIALS = 10
DEFAULT_SEED = 2606
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
    / "phase2_06_resource_ablation_matrix"
)

RESOURCE_NAMES = (
    "endpoint",
    "switch_path",
    "link",
    "readout",
    "feedforward",
    "reset",
    "epr_pool",
)

# Controlled service durations inherited from the staged Phase-2 models.
STAGE_DURATIONS_NS = {
    "endpoint_prepare": 30.0,          # communication qubit + endpoint port
    "switch_path": 15.0,
    "link": 80.0,
    "receiver_private": 25.0,
    "readout": 70.0,
    "feedforward": 40.0,
    "conditional_private": 20.0,
    "reset": 120.0,
}

CRITICAL_LATENCY_AFTER_EPR_NS = float(
    STAGE_DURATIONS_NS["endpoint_prepare"]
    + STAGE_DURATIONS_NS["switch_path"]
    + STAGE_DURATIONS_NS["link"]
    + STAGE_DURATIONS_NS["receiver_private"]
    + STAGE_DURATIONS_NS["readout"]
    + STAGE_DURATIONS_NS["feedforward"]
    + STAGE_DURATIONS_NS["conditional_private"]
)
CLEANUP_LATENCY_AFTER_EPR_NS = CRITICAL_LATENCY_AFTER_EPR_NS + STAGE_DURATIONS_NS["reset"]

# EPR subsystem: finite prefetched pool with asynchronous refill.
EPR_POOL_CAPACITY = 2
EPR_PREFETCH_TARGET = 2
EPR_GENERATION_LATENCY_NS = 240.0
EPR_PAIR_LIFETIME_NS = 1_200.0
EPR_GENERATION_CAPACITY = 1

BLACKBOX_ALLOWED_COLUMNS = {
    "trace_id",
    "configuration_id",
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
class AblationConfiguration:
    configuration_id: str
    shared_endpoint: bool
    shared_switch_path: bool
    shared_link: bool
    shared_readout: bool
    shared_feedforward: bool
    shared_reset: bool
    shared_epr_pool: bool
    shared_count: int
    shared_resources: str
    configuration_class: str

    def shared(self, resource_name: str) -> bool:
        return bool(getattr(self, f"shared_{resource_name}"))


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
    epr_acquired_ns: float = math.nan
    epr_wait_ns: float = 0.0
    epr_pool_key: str = ""
    epr_pool_level_at_arrival: int = 0
    epr_miss: bool = False
    epr_direct_cross_tenant_generation_at_arrival: bool = False
    epr_last_consumption_tenant_at_arrival: str = ""
    epr_last_consumption_ns_at_arrival: float = math.nan
    epr_persistent_victim_state: bool = False
    endpoint_acquired_ns: float = math.nan
    external_completion_ns: float = math.nan
    cleanup_completion_ns: float = math.nan
    success: bool = True
    failure_reason: str = ""
    resource_wait_ns: dict[str, float] = field(default_factory=lambda: defaultdict(float))


@dataclass
class WaitingAcquisition:
    request_id: str
    tenant: str
    arrival_ns: float
    resource_name: str
    resource_key: str


@dataclass
class ActiveAcquisition:
    request_id: str
    tenant: str
    start_ns: float
    resource_name: str
    resource_key: str


@dataclass
class PairRecord:
    pair_id: str
    pool_key: str
    generated_ns: float
    expires_ns: float
    trigger_tenant: str


@dataclass
class GenerationJob:
    job_id: str
    pool_key: str
    generator_key: str
    trigger_tenant: str
    start_ns: float
    end_ns: float


@dataclass
class ValidationAssertion:
    validation_group: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


# =============================================================================
# Configuration and workload definitions
# =============================================================================


def config_from_bits(bits: Iterable[int]) -> AblationConfiguration:
    bit_tuple = tuple(int(x) for x in bits)
    if len(bit_tuple) != len(RESOURCE_NAMES):
        raise ValueError("Incorrect number of ablation bits")
    flags = dict(zip(RESOURCE_NAMES, bit_tuple))
    shared = [r for r in RESOURCE_NAMES if flags[r]]
    bit_string = "".join(str(x) for x in bit_tuple)
    if not shared:
        cls = "everything_isolated"
    elif len(shared) == 1:
        cls = "single_resource"
    elif len(shared) == len(RESOURCE_NAMES):
        cls = "everything_shared"
    elif len(shared) == 2:
        cls = "resource_pair"
    else:
        cls = "multi_resource"
    return AblationConfiguration(
        configuration_id=f"cfg_{bit_string}",
        shared_endpoint=bool(flags["endpoint"]),
        shared_switch_path=bool(flags["switch_path"]),
        shared_link=bool(flags["link"]),
        shared_readout=bool(flags["readout"]),
        shared_feedforward=bool(flags["feedforward"]),
        shared_reset=bool(flags["reset"]),
        shared_epr_pool=bool(flags["epr_pool"]),
        shared_count=len(shared),
        shared_resources="+".join(shared) if shared else "none",
        configuration_class=cls,
    )


def build_full_configurations() -> tuple[AblationConfiguration, ...]:
    return tuple(config_from_bits(bits) for bits in itertools.product((0, 1), repeat=len(RESOURCE_NAMES)))


def build_core_configurations() -> tuple[AblationConfiguration, ...]:
    desired: set[tuple[str, ...]] = {
        (),
        # Singles
        *( (r,) for r in RESOURCE_NAMES ),
        # Interconnect and endpoint pairs
        ("endpoint", "switch_path"),
        ("endpoint", "link"),
        ("endpoint", "reset"),
        ("endpoint", "epr_pool"),
        ("switch_path", "link"),
        ("link", "reset"),
        ("link", "epr_pool"),
        # Backend interactions
        ("readout", "feedforward"),
        ("readout", "reset"),
        ("feedforward", "reset"),
        ("reset", "epr_pool"),
        ("readout", "epr_pool"),
        # Representative larger compositions
        ("endpoint", "switch_path", "link"),
        ("endpoint", "link", "epr_pool"),
        ("readout", "feedforward", "reset"),
        ("readout", "feedforward", "reset", "epr_pool"),
        tuple(RESOURCE_NAMES),
    }
    out: list[AblationConfiguration] = []
    for shared_tuple in desired:
        shared = set(shared_tuple)
        bits = tuple(int(r in shared) for r in RESOURCE_NAMES)
        out.append(config_from_bits(bits))
    return tuple(sorted(out, key=lambda c: (c.shared_count, c.configuration_id)))


def build_workloads() -> tuple[VictimWorkload, ...]:
    return (
        VictimWorkload(
            "sparse_periodic",
            "Low-rate remote operations with broad spacing.",
            "periodic_sparse",
        ),
        VictimWorkload(
            "dense_periodic",
            "Higher-rate periodic communication near the attacker probe rate.",
            "periodic_dense",
        ),
        VictimWorkload(
            "synchronization_bursty",
            "Three-operation communication bursts around synchronization epochs.",
            "synchronization_bursty",
        ),
    )


# =============================================================================
# Request schedules
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
    observation_window_ns: float,
) -> np.ndarray:
    if workload.release_pattern == "periodic_sparse":
        releases = np.arange(phase_ns, observation_window_ns, 900.0)
    elif workload.release_pattern == "periodic_dense":
        releases = np.arange(phase_ns, observation_window_ns, 470.0)
    elif workload.release_pattern == "synchronization_bursty":
        rows: list[float] = []
        base = phase_ns
        offsets = np.array([0.0, 75.0, 150.0])
        while base < observation_window_ns:
            rows.extend(float(base + x) for x in offsets)
            base += 1_650.0
        releases = np.array(rows, dtype=float)
    else:
        raise ValueError(workload.release_pattern)

    releases = releases[(releases >= 0.0) & (releases < observation_window_ns)]
    if len(releases):
        releases = np.maximum(0.0, releases + rng.uniform(-3.0, 3.0, len(releases)))
    return np.sort(releases)


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
            request_id=f"victim::{trial_id}::{index}",
            tenant="victim",
            ready_ns=float(release),
            request_index=index,
            workload_name=workload.workload_name,
            trial_id=trial_id,
        )
        for index, release in enumerate(releases)
    ]


def deterministic_trial_phase(seed: int, workload_name: str, trial_id: int) -> float:
    token = f"{seed}|{workload_name}|{trial_id}|phase".encode()
    digest = hashlib.sha256(token).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64 - 1)
    return 80.0 + 320.0 * value


def deterministic_schedule_seed(seed: int, workload_name: str, trial_id: int) -> int:
    token = f"{seed}|{workload_name}|{trial_id}|schedule".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


# =============================================================================
# Queueing resource
# =============================================================================


class QueueResource:
    def __init__(self, name: str, key: str, capacity: int = 1) -> None:
        self.name = name
        self.key = key
        self.capacity = int(capacity)
        self.active: dict[str, ActiveAcquisition] = {}
        self.queue: Deque[WaitingAcquisition] = deque()

    def request(self, sim: "RemoteOperationSimulator", request_id: str, tenant: str, now: float) -> None:
        waiter = WaitingAcquisition(
            request_id=request_id,
            tenant=tenant,
            arrival_ns=now,
            resource_name=self.name,
            resource_key=self.key,
        )
        if len(self.active) < self.capacity:
            self._grant(sim, waiter, now)
        else:
            self.queue.append(waiter)

    def _grant(self, sim: "RemoteOperationSimulator", waiter: WaitingAcquisition, now: float) -> None:
        active = ActiveAcquisition(
            request_id=waiter.request_id,
            tenant=waiter.tenant,
            start_ns=now,
            resource_name=self.name,
            resource_key=self.key,
        )
        self.active[waiter.request_id] = active
        wait_ns = max(0.0, now - waiter.arrival_ns)
        sim.requests[waiter.request_id].resource_wait_ns[self.name] += wait_ns
        sim.resource_wait_rows.append(
            {
                "run_kind": sim.run_kind,
                "workload_name": sim.workload_name,
                "trial_id": sim.trial_id,
                "configuration_id": sim.configuration.configuration_id,
                "request_id": waiter.request_id,
                "tenant": waiter.tenant,
                "resource_name": self.name,
                "resource_key": self.key,
                "arrival_ns": waiter.arrival_ns,
                "service_start_ns": now,
                "wait_ns": wait_ns,
            }
        )
        sim.schedule(now, "resource_granted", {"request_id": waiter.request_id, "resource_name": self.name})

    def release(self, sim: "RemoteOperationSimulator", request_id: str, now: float) -> None:
        active = self.active.pop(request_id, None)
        if active is None:
            raise RuntimeError(f"Release of inactive request {request_id} from {self.name}/{self.key}")
        sim.resource_interval_rows.append(
            {
                "run_kind": sim.run_kind,
                "workload_name": sim.workload_name,
                "trial_id": sim.trial_id,
                "configuration_id": sim.configuration.configuration_id,
                "request_id": request_id,
                "tenant": active.tenant,
                "resource_name": self.name,
                "resource_key": self.key,
                "start_ns": active.start_ns,
                "end_ns": now,
                "duration_ns": max(0.0, now - active.start_ns),
            }
        )
        while self.queue and len(self.active) < self.capacity:
            waiter = self.queue.popleft()
            self._grant(sim, waiter, now)


# =============================================================================
# EPR subsystem
# =============================================================================


class EPRSubsystem:
    def __init__(self, sim: "RemoteOperationSimulator") -> None:
        self.sim = sim
        self.available_pairs: dict[str, list[PairRecord]] = defaultdict(list)
        self.waiters: dict[str, Deque[str]] = defaultdict(deque)
        self.active_generations: dict[str, dict[str, GenerationJob]] = defaultdict(dict)
        self.inflight_by_pool: dict[str, int] = defaultdict(int)
        self.generation_seq = 0
        self.pair_seq = 0
        self.last_consumption_tenant: dict[str, str] = defaultdict(str)
        self.last_consumption_ns: dict[str, float] = defaultdict(lambda: math.nan)
        self.last_refill_trigger_tenant: dict[str, str] = defaultdict(str)

        tenants = sorted({s.tenant for s in sim.specs})
        pool_keys = {self.pool_key(t) for t in tenants}
        # Seed each pool at target at t=0. This represents warm prefetched state.
        for pool_key in sorted(pool_keys):
            for _ in range(EPR_PREFETCH_TARGET):
                self.pair_seq += 1
                self.available_pairs[pool_key].append(
                    PairRecord(
                        pair_id=f"warm::{pool_key}::{self.pair_seq}",
                        pool_key=pool_key,
                        generated_ns=0.0,
                        expires_ns=EPR_PAIR_LIFETIME_NS,
                        trigger_tenant="warm_prefetch",
                    )
                )

    def pool_key(self, tenant: str) -> str:
        return "shared_epr_pool" if self.sim.configuration.shared_epr_pool else f"epr_pool::{tenant}"

    def generator_key(self, tenant: str) -> str:
        return "shared_epr_generator" if self.sim.configuration.shared_epr_pool else f"epr_generator::{tenant}"

    def _expire(self, pool_key: str, now: float) -> None:
        kept: list[PairRecord] = []
        for pair in self.available_pairs[pool_key]:
            if pair.expires_ns <= now + FLOAT_TOLERANCE_NS:
                self.sim.epr_event_rows.append(
                    {
                        "run_kind": self.sim.run_kind,
                        "workload_name": self.sim.workload_name,
                        "trial_id": self.sim.trial_id,
                        "configuration_id": self.sim.configuration.configuration_id,
                        "time_ns": now,
                        "event_type": "pair_expired",
                        "tenant": "",
                        "pool_key": pool_key,
                        "pair_id": pair.pair_id,
                        "trigger_tenant": pair.trigger_tenant,
                        "pool_level_after": math.nan,
                    }
                )
            else:
                kept.append(pair)
        kept.sort(key=lambda p: (p.expires_ns, p.generated_ns, p.pair_id))
        self.available_pairs[pool_key] = kept

    def request_pair(self, request_id: str, now: float) -> None:
        req = self.sim.requests[request_id]
        tenant = req.spec.tenant
        pool_key = self.pool_key(tenant)
        gen_key = self.generator_key(tenant)
        self._expire(pool_key, now)
        req.epr_pool_key = pool_key
        req.epr_pool_level_at_arrival = len(self.available_pairs[pool_key])
        req.epr_last_consumption_tenant_at_arrival = self.last_consumption_tenant[pool_key]
        req.epr_last_consumption_ns_at_arrival = self.last_consumption_ns[pool_key]
        active_trigger_tenants = {
            job.trigger_tenant for job in self.active_generations[gen_key].values()
        }
        req.epr_direct_cross_tenant_generation_at_arrival = any(
            x not in ("", tenant, "warm_prefetch", "background") for x in active_trigger_tenants
        )

        if self.available_pairs[pool_key]:
            pair = self.available_pairs[pool_key].pop(0)
            self._consume_pair(request_id, pair, now, source="prefetched_hit")
            self._ensure_refill(pool_key, tenant, now)
            return

        req.epr_miss = True
        self.waiters[pool_key].append(request_id)
        self.sim.epr_event_rows.append(
            {
                "run_kind": self.sim.run_kind,
                "workload_name": self.sim.workload_name,
                "trial_id": self.sim.trial_id,
                "configuration_id": self.sim.configuration.configuration_id,
                "time_ns": now,
                "event_type": "pool_miss",
                "tenant": tenant,
                "pool_key": pool_key,
                "pair_id": "",
                "trigger_tenant": tenant,
                "pool_level_after": 0,
            }
        )
        self._ensure_refill(pool_key, tenant, now)

    def _consume_pair(self, request_id: str, pair: PairRecord, now: float, source: str) -> None:
        req = self.sim.requests[request_id]
        tenant = req.spec.tenant
        pool_key = pair.pool_key
        req.epr_acquired_ns = now
        req.epr_wait_ns = max(0.0, now - req.spec.ready_ns)

        previous_tenant = self.last_consumption_tenant[pool_key]
        previous_ns = self.last_consumption_ns[pool_key]
        if (
            tenant == "attacker"
            and self.sim.configuration.shared_epr_pool
            and previous_tenant == "victim"
            and math.isfinite(previous_ns)
            and previous_ns < now - FLOAT_TOLERANCE_NS
            and req.epr_miss
        ):
            req.epr_persistent_victim_state = True

        self.last_consumption_tenant[pool_key] = tenant
        self.last_consumption_ns[pool_key] = now
        self.last_refill_trigger_tenant[pool_key] = tenant
        self.sim.epr_event_rows.append(
            {
                "run_kind": self.sim.run_kind,
                "workload_name": self.sim.workload_name,
                "trial_id": self.sim.trial_id,
                "configuration_id": self.sim.configuration.configuration_id,
                "time_ns": now,
                "event_type": f"pair_consumed::{source}",
                "tenant": tenant,
                "pool_key": pool_key,
                "pair_id": pair.pair_id,
                "trigger_tenant": pair.trigger_tenant,
                "pool_level_after": len(self.available_pairs[pool_key]),
            }
        )
        self.sim.schedule(now, "epr_ready", {"request_id": request_id})

    def _needed_pairs(self, pool_key: str) -> int:
        stored = len(self.available_pairs[pool_key])
        waiting = len(self.waiters[pool_key])
        inflight = self.inflight_by_pool[pool_key]
        return max(0, waiting + EPR_PREFETCH_TARGET - stored - inflight)

    def _ensure_refill(self, pool_key: str, trigger_tenant: str, now: float) -> None:
        if trigger_tenant:
            self.last_refill_trigger_tenant[pool_key] = trigger_tenant
        # Private pools have a tenant encoded in their key. Shared pool generation
        # uses the same shared generator for all tenants.
        if self.sim.configuration.shared_epr_pool:
            gen_key = "shared_epr_generator"
        else:
            tenant = pool_key.split("::", 1)[1]
            gen_key = f"epr_generator::{tenant}"

        while (
            self._needed_pairs(pool_key) > 0
            and len(self.active_generations[gen_key]) < EPR_GENERATION_CAPACITY
        ):
            self.generation_seq += 1
            job_id = f"gen::{self.generation_seq}"
            trigger = self.last_refill_trigger_tenant[pool_key] or trigger_tenant or "background"
            job = GenerationJob(
                job_id=job_id,
                pool_key=pool_key,
                generator_key=gen_key,
                trigger_tenant=trigger,
                start_ns=now,
                end_ns=now + EPR_GENERATION_LATENCY_NS,
            )
            self.active_generations[gen_key][job_id] = job
            self.inflight_by_pool[pool_key] += 1
            self.sim.generation_rows.append(
                {
                    "run_kind": self.sim.run_kind,
                    "workload_name": self.sim.workload_name,
                    "trial_id": self.sim.trial_id,
                    "configuration_id": self.sim.configuration.configuration_id,
                    "job_id": job_id,
                    "pool_key": pool_key,
                    "generator_key": gen_key,
                    "trigger_tenant": trigger,
                    "start_ns": now,
                    "end_ns": job.end_ns,
                }
            )
            self.sim.schedule(job.end_ns, "generation_complete", {"job": job})

    def generation_complete(self, job: GenerationJob, now: float) -> None:
        active = self.active_generations[job.generator_key]
        active.pop(job.job_id, None)
        self.inflight_by_pool[job.pool_key] = max(0, self.inflight_by_pool[job.pool_key] - 1)
        self._expire(job.pool_key, now)
        self.pair_seq += 1
        pair = PairRecord(
            pair_id=f"pair::{self.pair_seq}",
            pool_key=job.pool_key,
            generated_ns=now,
            expires_ns=now + EPR_PAIR_LIFETIME_NS,
            trigger_tenant=job.trigger_tenant,
        )

        if self.waiters[job.pool_key]:
            request_id = self.waiters[job.pool_key].popleft()
            self._consume_pair(request_id, pair, now, source="generated_after_miss")
        elif len(self.available_pairs[job.pool_key]) < EPR_POOL_CAPACITY:
            self.available_pairs[job.pool_key].append(pair)
            self.available_pairs[job.pool_key].sort(key=lambda p: (p.expires_ns, p.pair_id))
            self.sim.epr_event_rows.append(
                {
                    "run_kind": self.sim.run_kind,
                    "workload_name": self.sim.workload_name,
                    "trial_id": self.sim.trial_id,
                    "configuration_id": self.sim.configuration.configuration_id,
                    "time_ns": now,
                    "event_type": "pair_stored_after_refill",
                    "tenant": "",
                    "pool_key": job.pool_key,
                    "pair_id": pair.pair_id,
                    "trigger_tenant": job.trigger_tenant,
                    "pool_level_after": len(self.available_pairs[job.pool_key]),
                }
            )

        self._ensure_refill(job.pool_key, job.trigger_tenant, now)


# =============================================================================
# Remote-operation simulator
# =============================================================================


class RemoteOperationSimulator:
    def __init__(
        self,
        *,
        configuration: AblationConfiguration,
        specs: Iterable[RequestSpec],
        run_kind: str,
        workload_name: str,
        trial_id: int,
    ) -> None:
        self.configuration = configuration
        self.specs = sorted(list(specs), key=lambda s: (s.ready_ns, s.tenant, s.request_index))
        self.run_kind = run_kind
        self.workload_name = workload_name
        self.trial_id = trial_id
        self.requests: dict[str, RequestState] = {s.request_id: RequestState(spec=s) for s in self.specs}

        self.events: list[tuple[float, int, str, dict[str, Any]]] = []
        self.event_seq = 0
        self.resources: dict[tuple[str, str], QueueResource] = {}
        self.resource_wait_rows: list[dict[str, Any]] = []
        self.resource_interval_rows: list[dict[str, Any]] = []
        self.epr_event_rows: list[dict[str, Any]] = []
        self.generation_rows: list[dict[str, Any]] = []
        self.epr = EPRSubsystem(self)

    def schedule(self, time_ns: float, kind: str, payload: dict[str, Any]) -> None:
        self.event_seq += 1
        heapq.heappush(self.events, (float(time_ns), self.event_seq, kind, payload))

    def resource_key(self, resource_name: str, tenant: str) -> str:
        return "shared" if self.configuration.shared(resource_name) else tenant

    def resource(self, resource_name: str, tenant: str) -> QueueResource:
        key = self.resource_key(resource_name, tenant)
        token = (resource_name, key)
        if token not in self.resources:
            self.resources[token] = QueueResource(resource_name, key, capacity=1)
        return self.resources[token]

    def acquire(self, request_id: str, resource_name: str, now: float) -> None:
        req = self.requests[request_id]
        self.resource(resource_name, req.spec.tenant).request(
            self, request_id, req.spec.tenant, now
        )

    def release(self, request_id: str, resource_name: str, now: float) -> None:
        req = self.requests[request_id]
        self.resource(resource_name, req.spec.tenant).release(self, request_id, now)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for spec in self.specs:
            self.schedule(spec.ready_ns, "request_arrival", {"request_id": spec.request_id})

        while self.events:
            now, _, kind, payload = heapq.heappop(self.events)
            if kind == "request_arrival":
                self.epr.request_pair(payload["request_id"], now)
            elif kind == "generation_complete":
                self.epr.generation_complete(payload["job"], now)
            elif kind == "epr_ready":
                self.acquire(payload["request_id"], "endpoint", now)
            elif kind == "resource_granted":
                self._handle_resource_granted(payload["request_id"], payload["resource_name"], now)
            elif kind == "stage_complete":
                self._handle_stage_complete(payload["request_id"], payload["stage_name"], now)
            elif kind == "private_stage_complete":
                self._advance_after_private(payload["request_id"], payload["stage_name"], now)
            else:
                raise RuntimeError(f"Unknown event kind {kind}")

        request_rows = []
        for state in self.requests.values():
            spec = state.spec
            request_rows.append(
                {
                    "run_kind": self.run_kind,
                    "workload_name": self.workload_name,
                    "trial_id": self.trial_id,
                    "configuration_id": self.configuration.configuration_id,
                    "request_id": spec.request_id,
                    "tenant": spec.tenant,
                    "request_index": spec.request_index,
                    "release_ns": spec.ready_ns,
                    "success": state.success,
                    "epr_acquired_ns": state.epr_acquired_ns,
                    "epr_wait_ns": state.epr_wait_ns,
                    "epr_miss": state.epr_miss,
                    "epr_pool_key": state.epr_pool_key,
                    "epr_pool_level_at_arrival": state.epr_pool_level_at_arrival,
                    "epr_direct_cross_tenant_generation_at_arrival": state.epr_direct_cross_tenant_generation_at_arrival,
                    "epr_last_consumption_tenant_at_arrival": state.epr_last_consumption_tenant_at_arrival,
                    "epr_last_consumption_ns_at_arrival": state.epr_last_consumption_ns_at_arrival,
                    "epr_persistent_victim_state": state.epr_persistent_victim_state,
                    "endpoint_acquired_ns": state.endpoint_acquired_ns,
                    "external_completion_ns": state.external_completion_ns,
                    "cleanup_completion_ns": state.cleanup_completion_ns,
                    "turnaround_ns": state.external_completion_ns - spec.ready_ns if math.isfinite(state.external_completion_ns) else math.nan,
                    "cleanup_turnaround_ns": state.cleanup_completion_ns - spec.ready_ns if math.isfinite(state.cleanup_completion_ns) else math.nan,
                    **{f"wait_{r}_ns": float(state.resource_wait_ns.get(r, 0.0)) for r in RESOURCE_NAMES if r != "epr_pool"},
                }
            )

        return (
            pd.DataFrame(request_rows),
            pd.DataFrame(self.resource_wait_rows),
            pd.DataFrame(self.resource_interval_rows),
            pd.DataFrame(self.epr_event_rows),
            pd.DataFrame(self.generation_rows),
        )

    def _handle_resource_granted(self, request_id: str, resource_name: str, now: float) -> None:
        req = self.requests[request_id]
        if resource_name == "endpoint":
            req.endpoint_acquired_ns = now
            self.schedule(
                now + STAGE_DURATIONS_NS["endpoint_prepare"],
                "stage_complete",
                {"request_id": request_id, "stage_name": "endpoint_prepare"},
            )
        elif resource_name in {"switch_path", "link", "readout", "feedforward", "reset"}:
            self.schedule(
                now + STAGE_DURATIONS_NS[resource_name],
                "stage_complete",
                {"request_id": request_id, "stage_name": resource_name},
            )
        else:
            raise RuntimeError(resource_name)

    def _handle_stage_complete(self, request_id: str, stage_name: str, now: float) -> None:
        req = self.requests[request_id]
        if stage_name == "endpoint_prepare":
            # Endpoint remains held through post-completion reset.
            self.acquire(request_id, "switch_path", now)
        elif stage_name == "switch_path":
            self.release(request_id, "switch_path", now)
            self.acquire(request_id, "link", now)
        elif stage_name == "link":
            self.release(request_id, "link", now)
            self.schedule(
                now + STAGE_DURATIONS_NS["receiver_private"],
                "private_stage_complete",
                {"request_id": request_id, "stage_name": "receiver_private"},
            )
        elif stage_name == "readout":
            self.release(request_id, "readout", now)
            self.acquire(request_id, "feedforward", now)
        elif stage_name == "feedforward":
            self.release(request_id, "feedforward", now)
            self.schedule(
                now + STAGE_DURATIONS_NS["conditional_private"],
                "private_stage_complete",
                {"request_id": request_id, "stage_name": "conditional_private"},
            )
        elif stage_name == "reset":
            self.release(request_id, "reset", now)
            req.cleanup_completion_ns = now
            self.release(request_id, "endpoint", now)
        else:
            raise RuntimeError(stage_name)

    def _advance_after_private(self, request_id: str, stage_name: str, now: float) -> None:
        req = self.requests[request_id]
        if stage_name == "receiver_private":
            self.acquire(request_id, "readout", now)
        elif stage_name == "conditional_private":
            req.external_completion_ns = now
            self.acquire(request_id, "reset", now)
        else:
            raise RuntimeError(stage_name)


# =============================================================================
# Paired trace construction and attribution
# =============================================================================


def paired_attacker_trace(
    configuration: AblationConfiguration,
    workload_name: str,
    trial_id: int,
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    a = attacker_only[attacker_only["tenant"] == "attacker"].copy()
    c = combined[combined["tenant"] == "attacker"].copy()
    keep_a = [
        "request_index", "release_ns", "success", "external_completion_ns", "turnaround_ns",
    ]
    keep_c = [
        "request_index", "success", "external_completion_ns", "turnaround_ns",
        "epr_miss", "epr_persistent_victim_state", "epr_direct_cross_tenant_generation_at_arrival",
    ] + [f"wait_{r}_ns" for r in RESOURCE_NAMES if r != "epr_pool"]
    a = a[keep_a].rename(columns={
        "success": "attacker_only_success",
        "external_completion_ns": "attacker_only_completion_ns",
        "turnaround_ns": "attacker_only_turnaround_ns",
    })
    c = c[keep_c].rename(columns={
        "success": "combined_success",
        "external_completion_ns": "combined_completion_ns",
        "turnaround_ns": "combined_turnaround_ns",
    })
    df = a.merge(c, on="request_index", how="inner", validate="one_to_one")
    df["excess_turnaround_ns"] = df["combined_turnaround_ns"] - df["attacker_only_turnaround_ns"]
    df["affected"] = df["excess_turnaround_ns"] > AFFECTED_THRESHOLD_NS
    df["speedup"] = df["excess_turnaround_ns"] < -AFFECTED_THRESHOLD_NS
    df["failure_transition"] = df["attacker_only_success"] != df["combined_success"]
    df["configuration_id"] = configuration.configuration_id
    df["workload_name"] = workload_name
    df["trial_id"] = trial_id
    df["trace_id"] = [
        f"{configuration.configuration_id}::{workload_name}::{trial_id}::{idx}"
        for idx in df["request_index"]
    ]
    df = df.rename(columns={"request_index": "probe_index"})
    return df


def add_cross_tenant_wait_attribution(
    wait_df: pd.DataFrame,
    interval_df: pd.DataFrame,
) -> pd.DataFrame:
    if wait_df.empty:
        return wait_df.assign(cross_tenant_wait_ns=pd.Series(dtype=float))
    if interval_df.empty:
        out = wait_df.copy()
        out["cross_tenant_wait_ns"] = 0.0
        return out

    intervals_by_key: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    for row in interval_df.itertuples(index=False):
        intervals_by_key[(row.resource_name, row.resource_key)].append(
            (row.tenant, float(row.start_ns), float(row.end_ns))
        )

    rows = []
    for row in wait_df.itertuples(index=False):
        cross = 0.0
        if float(row.wait_ns) > FLOAT_TOLERANCE_NS:
            for tenant, start, end in intervals_by_key[(row.resource_name, row.resource_key)]:
                if tenant == row.tenant:
                    continue
                overlap = max(
                    0.0,
                    min(float(row.service_start_ns), end) - max(float(row.arrival_ns), start),
                )
                cross += overlap
        d = row._asdict()
        d["cross_tenant_wait_ns"] = min(float(row.wait_ns), cross)
        rows.append(d)
    return pd.DataFrame(rows)


def longest_true_run(values: Iterable[bool]) -> int:
    best = 0
    cur = 0
    for value in values:
        if bool(value):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def true_run_count(values: Iterable[bool]) -> int:
    count = 0
    prev = False
    for value in values:
        current = bool(value)
        if current and not prev:
            count += 1
        prev = current
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
        "configuration_id": str(trace["configuration_id"].iloc[0]),
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
        "epr_miss_fraction": float(np.mean(trace["epr_miss"])) if len(trace) else 0.0,
        "persistent_epr_state_fraction": float(np.mean(trace["epr_persistent_victim_state"])) if len(trace) else 0.0,
        "direct_epr_generation_overlap_fraction": float(np.mean(trace["epr_direct_cross_tenant_generation_at_arrival"])) if len(trace) else 0.0,
        **{
            f"mean_wait_{r}_ns": float(np.mean(trace[f"wait_{r}_ns"])) if len(trace) else 0.0
            for r in RESOURCE_NAMES if r != "epr_pool"
        },
    }


def victim_slowdown_metrics(victim_only: pd.DataFrame, combined: pd.DataFrame) -> dict[str, float]:
    v = victim_only[victim_only["tenant"] == "victim"].copy()
    c = combined[combined["tenant"] == "victim"].copy()
    if v.empty or c.empty:
        return {
            "victim_mean_request_slowdown": 1.0,
            "victim_makespan_slowdown": 1.0,
            "victim_mean_added_turnaround_ns": 0.0,
        }
    m = v[["request_index", "turnaround_ns", "external_completion_ns"]].merge(
        c[["request_index", "turnaround_ns", "external_completion_ns"]],
        on="request_index",
        suffixes=("_victim_only", "_combined"),
        validate="one_to_one",
    )
    ratios = np.divide(
        m["turnaround_ns_combined"].to_numpy(dtype=float),
        m["turnaround_ns_victim_only"].to_numpy(dtype=float),
        out=np.ones(len(m), dtype=float),
        where=m["turnaround_ns_victim_only"].to_numpy(dtype=float) > 0,
    )
    base_start = float(v["release_ns"].min())
    base_makespan = float(v["external_completion_ns"].max() - base_start)
    comb_makespan = float(c["external_completion_ns"].max() - base_start)
    return {
        "victim_mean_request_slowdown": float(np.mean(ratios)),
        "victim_makespan_slowdown": comb_makespan / base_makespan if base_makespan > 0 else 1.0,
        "victim_mean_added_turnaround_ns": float(
            np.mean(m["turnaround_ns_combined"] - m["turnaround_ns_victim_only"])
        ),
    }


# =============================================================================
# Classification separability
# =============================================================================


CLASSIFIER_FEATURES = [
    "affected_probe_fraction",
    "mean_absolute_timing_change_ns",
    "cumulative_positive_excess_ns",
    "maximum_positive_excess_ns",
    "p95_positive_excess_ns",
    "std_excess_turnaround_ns",
    "longest_affected_run",
    "affected_run_count",
    "lag1_excess_autocorrelation",
    "epr_miss_fraction",
    "persistent_epr_state_fraction",
    "failure_transition_fraction",
]


def nearest_centroid_fingerprinting(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for config_id, group in features.groupby("configuration_id", sort=True):
        group = group.copy()
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
            centroids = {
                label: np.mean(
                    (train[train["workload_name"] == label][CLASSIFIER_FEATURES].astype(float).to_numpy() - mean) / std,
                    axis=0,
                )
                for label in labels
            }
            for row in test.itertuples(index=False):
                x = np.array([float(getattr(row, f)) for f in CLASSIFIER_FEATURES])
                x = (x - mean) / std
                distances = {label: float(np.linalg.norm(x - centroid)) for label, centroid in centroids.items()}
                prediction = min(distances.items(), key=lambda kv: (kv[1], kv[0]))[0]
                actual = row.workload_name
                is_correct = prediction == actual
                correct += int(is_correct)
                total += 1
                prediction_rows.append(
                    {
                        "configuration_id": config_id,
                        "trial_id": int(row.trial_id),
                        "actual_workload": actual,
                        "predicted_workload": prediction,
                        "correct": is_correct,
                        "distance": distances[prediction],
                    }
                )
        metric_rows.append(
            {
                "configuration_id": config_id,
                "classifier": "leave_one_trial_out_nearest_centroid",
                "sample_count": int(total),
                "workload_count": int(len(labels)),
                "chance_accuracy": 1.0 / len(labels) if labels else math.nan,
                "accuracy": correct / total if total else math.nan,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


# =============================================================================
# Causal ablation analyses
# =============================================================================


CAUSAL_METRICS = (
    "affected_probe_fraction",
    "mean_absolute_timing_change_ns",
    "cumulative_positive_excess_ns",
    "maximum_positive_excess_ns",
    "longest_affected_run",
    "victim_mean_request_slowdown",
    "victim_makespan_slowdown",
    "classification_accuracy",
)


def configuration_metric_summary(
    trial_summary: pd.DataFrame,
    classifier_metrics: pd.DataFrame,
    configurations: pd.DataFrame,
) -> pd.DataFrame:
    agg_cols = [
        "affected_probe_fraction",
        "speedup_probe_fraction",
        "failure_transition_fraction",
        "mean_excess_turnaround_ns",
        "mean_absolute_timing_change_ns",
        "cumulative_positive_excess_ns",
        "maximum_positive_excess_ns",
        "p95_positive_excess_ns",
        "std_excess_turnaround_ns",
        "longest_affected_run",
        "affected_run_count",
        "lag1_excess_autocorrelation",
        "epr_miss_fraction",
        "persistent_epr_state_fraction",
        "direct_epr_generation_overlap_fraction",
        "victim_mean_request_slowdown",
        "victim_makespan_slowdown",
        "victim_mean_added_turnaround_ns",
    ]
    summary = trial_summary.groupby("configuration_id", as_index=False)[agg_cols].mean()
    cls = classifier_metrics[["configuration_id", "accuracy"]].rename(columns={"accuracy": "classification_accuracy"})
    summary = summary.merge(cls, on="configuration_id", how="left")
    summary = configurations.merge(summary, on="configuration_id", how="left", validate="one_to_one")
    return summary


def main_effects_full_factorial(config_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    flag_cols = [f"shared_{r}" for r in RESOURCE_NAMES]
    index = {
        tuple(bool(row[c]) for c in flag_cols): row
        for _, row in config_summary.iterrows()
    }
    for r_idx, resource in enumerate(RESOURCE_NAMES):
        deltas: dict[str, list[float]] = {m: [] for m in CAUSAL_METRICS}
        for bits in itertools.product((False, True), repeat=len(RESOURCE_NAMES) - 1):
            b0 = list(bits)
            b0.insert(r_idx, False)
            b1 = list(b0)
            b1[r_idx] = True
            row0 = index.get(tuple(b0))
            row1 = index.get(tuple(b1))
            if row0 is None or row1 is None:
                continue
            for metric in CAUSAL_METRICS:
                deltas[metric].append(float(row1[metric]) - float(row0[metric]))
        for metric, values in deltas.items():
            if values:
                rows.append(
                    {
                        "resource_name": resource,
                        "metric": metric,
                        "paired_context_count": len(values),
                        "mean_marginal_effect": float(np.mean(values)),
                        "median_marginal_effect": float(np.median(values)),
                        "min_marginal_effect": float(np.min(values)),
                        "max_marginal_effect": float(np.max(values)),
                        "positive_context_fraction": float(np.mean(np.array(values) > 1e-12)),
                    }
                )
    return pd.DataFrame(rows)


def pairwise_interactions_full_factorial(config_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    flag_cols = [f"shared_{r}" for r in RESOURCE_NAMES]
    index = {
        tuple(bool(row[c]) for c in flag_cols): row
        for _, row in config_summary.iterrows()
    }
    for i, resource_a in enumerate(RESOURCE_NAMES):
        for j in range(i + 1, len(RESOURCE_NAMES)):
            resource_b = RESOURCE_NAMES[j]
            other_indices = [k for k in range(len(RESOURCE_NAMES)) if k not in (i, j)]
            interaction_values: dict[str, list[float]] = {m: [] for m in CAUSAL_METRICS}
            for other_bits in itertools.product((False, True), repeat=len(other_indices)):
                base = [False] * len(RESOURCE_NAMES)
                for k, value in zip(other_indices, other_bits):
                    base[k] = value
                b00 = list(base)
                b10 = list(base); b10[i] = True
                b01 = list(base); b01[j] = True
                b11 = list(base); b11[i] = True; b11[j] = True
                r00 = index.get(tuple(b00)); r10 = index.get(tuple(b10)); r01 = index.get(tuple(b01)); r11 = index.get(tuple(b11))
                if any(x is None for x in (r00, r10, r01, r11)):
                    continue
                for metric in CAUSAL_METRICS:
                    value = float(r11[metric]) - float(r10[metric]) - float(r01[metric]) + float(r00[metric])
                    interaction_values[metric].append(value)
            for metric, values in interaction_values.items():
                if values:
                    rows.append(
                        {
                            "resource_a": resource_a,
                            "resource_b": resource_b,
                            "metric": metric,
                            "context_count": len(values),
                            "mean_difference_in_differences": float(np.mean(values)),
                            "median_difference_in_differences": float(np.median(values)),
                            "min_interaction": float(np.min(values)),
                            "max_interaction": float(np.max(values)),
                            "positive_interaction_fraction": float(np.mean(np.array(values) > 1e-12)),
                            "negative_interaction_fraction": float(np.mean(np.array(values) < -1e-12)),
                        }
                    )
    return pd.DataFrame(rows)


def necessity_sufficiency_summary(config_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    isolated = config_summary[config_summary["shared_count"] == 0]
    all_shared = config_summary[config_summary["shared_count"] == len(RESOURCE_NAMES)]
    if isolated.empty or all_shared.empty:
        return pd.DataFrame()
    iso = isolated.iloc[0]
    full = all_shared.iloc[0]

    for resource in RESOURCE_NAMES:
        single = config_summary[
            (config_summary["shared_count"] == 1) & (config_summary[f"shared_{resource}"])
        ]
        minus = config_summary[
            (config_summary["shared_count"] == len(RESOURCE_NAMES) - 1)
            & (~config_summary[f"shared_{resource}"])
        ]
        for metric in CAUSAL_METRICS:
            single_value = float(single.iloc[0][metric]) if not single.empty else math.nan
            minus_value = float(minus.iloc[0][metric]) if not minus.empty else math.nan
            iso_value = float(iso[metric])
            full_value = float(full[metric])
            rows.append(
                {
                    "resource_name": resource,
                    "metric": metric,
                    "isolated_value": iso_value,
                    "single_resource_value": single_value,
                    "sufficiency_increment_over_isolated": single_value - iso_value if math.isfinite(single_value) else math.nan,
                    "all_shared_value": full_value,
                    "all_shared_minus_resource_value": minus_value,
                    "necessity_drop_when_privatized": full_value - minus_value if math.isfinite(minus_value) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def pair_dominance_summary(config_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    isolated = config_summary[config_summary["shared_count"] == 0]
    if isolated.empty:
        return pd.DataFrame()
    iso = isolated.iloc[0]
    for a, b in itertools.combinations(RESOURCE_NAMES, 2):
        single_a = config_summary[(config_summary["shared_count"] == 1) & config_summary[f"shared_{a}"]]
        single_b = config_summary[(config_summary["shared_count"] == 1) & config_summary[f"shared_{b}"]]
        pair = config_summary[
            (config_summary["shared_count"] == 2)
            & config_summary[f"shared_{a}"]
            & config_summary[f"shared_{b}"]
        ]
        if single_a.empty or single_b.empty or pair.empty:
            continue
        for metric in CAUSAL_METRICS:
            va = float(single_a.iloc[0][metric]); vb = float(single_b.iloc[0][metric]); vp = float(pair.iloc[0][metric]); vi = float(iso[metric])
            strongest_single = max(va, vb)
            additive_expected = va + vb - vi
            rows.append(
                {
                    "resource_a": a,
                    "resource_b": b,
                    "metric": metric,
                    "resource_a_single": va,
                    "resource_b_single": vb,
                    "pair_value": vp,
                    "increment_over_strongest_single": vp - strongest_single,
                    "difference_from_additive_expected": vp - additive_expected,
                    "interpretation": (
                        "synergistic_amplification" if vp - strongest_single > 1e-9
                        else "masked_or_dominated" if abs(vp - strongest_single) <= 1e-9
                        else "interference_or_backpressure_masking"
                    ),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Validation
# =============================================================================


def assertion(group: str, name: str, passed: bool, expected: Any, observed: Any, details: str = "") -> ValidationAssertion:
    return ValidationAssertion(group, name, bool(passed), str(expected), str(observed), details)


def build_validations(
    configurations: pd.DataFrame,
    trial_summary: pd.DataFrame,
    blackbox: pd.DataFrame,
    wait_events: pd.DataFrame,
    config_summary: pd.DataFrame,
    matrix_mode: str,
) -> pd.DataFrame:
    out: list[ValidationAssertion] = []

    out.append(assertion(
        "matrix",
        "configuration_count",
        len(configurations) == (128 if matrix_mode == "full" else len(build_core_configurations())),
        128 if matrix_mode == "full" else len(build_core_configurations()),
        len(configurations),
    ))
    out.append(assertion(
        "matrix",
        "everything_isolated_present",
        bool((configurations["shared_count"] == 0).any()),
        True,
        bool((configurations["shared_count"] == 0).any()),
    ))
    out.append(assertion(
        "matrix",
        "everything_shared_present",
        bool((configurations["shared_count"] == len(RESOURCE_NAMES)).any()),
        True,
        bool((configurations["shared_count"] == len(RESOURCE_NAMES)).any()),
    ))

    isolated_ids = configurations.loc[configurations["shared_count"] == 0, "configuration_id"].tolist()
    iso = trial_summary[trial_summary["configuration_id"].isin(isolated_ids)]
    iso_affected = float(iso["affected_probe_fraction"].max()) if not iso.empty else math.nan
    iso_abs = float(iso["mean_absolute_timing_change_ns"].max()) if not iso.empty else math.nan
    out.append(assertion(
        "negative_control",
        "everything_isolated_has_no_cross_tenant_timing_change",
        bool(iso_affected <= 1e-12 and iso_abs <= 1e-9),
        "0 affected and 0 timing change",
        f"affected={iso_affected}, abs_change={iso_abs}",
    ))

    full_ids = configurations.loc[configurations["shared_count"] == len(RESOURCE_NAMES), "configuration_id"].tolist()
    full = trial_summary[trial_summary["configuration_id"].isin(full_ids)]
    full_affected = float(full["affected_probe_fraction"].mean()) if not full.empty else 0.0
    out.append(assertion(
        "positive_control",
        "everything_shared_produces_nonzero_channel",
        full_affected > 0.0,
        "> 0 affected fraction",
        full_affected,
    ))

    # Private resources must never produce cross-tenant wait attribution.
    if not wait_events.empty:
        merged = wait_events.merge(
            configurations[["configuration_id"] + [f"shared_{r}" for r in RESOURCE_NAMES]],
            on="configuration_id",
            how="left",
        )
        bad_private = 0
        for resource in RESOURCE_NAMES:
            if resource == "epr_pool":
                continue
            rows = merged[(merged["resource_name"] == resource) & (~merged[f"shared_{resource}"])]
            bad_private += int((rows["cross_tenant_wait_ns"] > 1e-9).sum())
        out.append(assertion(
            "attribution",
            "private_resources_have_zero_cross_tenant_wait",
            bad_private == 0,
            0,
            bad_private,
        ))

    blackbox_extra = sorted(set(blackbox.columns) - BLACKBOX_ALLOWED_COLUMNS - {"workload_name", "trial_id"})
    # workload/trial are evaluator keys retained in the master file.  The strict
    # attacker-facing export below removes them.
    out.append(assertion(
        "blackbox",
        "blackbox_master_has_only_timing_plus_evaluator_keys",
        len(blackbox_extra) == 0,
        "no internal columns",
        blackbox_extra,
    ))

    finite_cols = [
        "affected_probe_fraction", "mean_absolute_timing_change_ns",
        "cumulative_positive_excess_ns", "victim_mean_request_slowdown",
        "victim_makespan_slowdown",
    ]
    finite_ok = np.isfinite(trial_summary[finite_cols].astype(float).to_numpy()).all()
    out.append(assertion("numerics", "trial_summary_finite", finite_ok, True, finite_ok))

    # EPR-only should expose a persistent state channel for at least one trial.
    epr_only = configurations[(configurations["shared_count"] == 1) & configurations["shared_epr_pool"]]
    if not epr_only.empty:
        x = trial_summary[trial_summary["configuration_id"].isin(epr_only["configuration_id"])]
        obs = float(x["persistent_epr_state_fraction"].max()) if not x.empty else 0.0
        out.append(assertion(
            "mechanism",
            "shared_epr_pool_alone_exposes_persistent_state",
            obs > 0.0,
            "> 0 persistent EPR state fraction",
            obs,
        ))

    # Each single-resource configuration should exist in full/core modes.
    single_missing = []
    for r in RESOURCE_NAMES:
        rows = configurations[(configurations["shared_count"] == 1) & configurations[f"shared_{r}"]]
        if rows.empty:
            single_missing.append(r)
    out.append(assertion(
        "matrix",
        "all_single_resource_ablations_present",
        len(single_missing) == 0,
        "all seven singles",
        single_missing,
    ))

    # No negative slowdown ratios.
    slow_ok = bool((trial_summary["victim_mean_request_slowdown"] > 0).all() and (trial_summary["victim_makespan_slowdown"] > 0).all())
    out.append(assertion("numerics", "victim_slowdown_positive", slow_ok, True, slow_ok))

    return pd.DataFrame(asdict(x) for x in out)


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(
    *,
    output_dir: Path,
    trials: int,
    seed: int,
    observation_window_ns: float,
    matrix_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations_tuple = build_full_configurations() if matrix_mode == "full" else build_core_configurations()
    configurations_df = pd.DataFrame(asdict(c) for c in configurations_tuple)
    workloads = build_workloads()

    phase_rows = []
    baseline_cache: dict[tuple[str, int, str], pd.DataFrame] = {}

    # Detailed rows can be large, so they are accumulated once and compressed.
    blackbox_rows: list[pd.DataFrame] = []
    evaluator_trace_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    wait_rows: list[pd.DataFrame] = []
    interval_rows: list[pd.DataFrame] = []
    epr_rows: list[pd.DataFrame] = []
    generation_rows: list[pd.DataFrame] = []

    isolated_config = config_from_bits((0,) * len(RESOURCE_NAMES))

    # Baselines are single-tenant runs and therefore independent of sharing flags.
    for workload in workloads:
        for trial_id in range(trials):
            phase_ns = deterministic_trial_phase(seed, workload.workload_name, trial_id)
            sched_seed = deterministic_schedule_seed(seed, workload.workload_name, trial_id)
            phase_rows.append({
                "workload_name": workload.workload_name,
                "trial_id": trial_id,
                "victim_phase_ns": phase_ns,
                "schedule_seed": sched_seed,
            })
            a_specs = attacker_specs(trial_id, workload.workload_name, observation_window_ns)
            v_specs = victim_specs(workload, trial_id, phase_ns, sched_seed, observation_window_ns)

            for run_kind, specs in (("attacker_only", a_specs), ("victim_only", v_specs)):
                sim = RemoteOperationSimulator(
                    configuration=isolated_config,
                    specs=specs,
                    run_kind=run_kind,
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                )
                req, waits, intervals, epr, gens = sim.run()
                baseline_cache[(workload.workload_name, trial_id, run_kind)] = req

    total_cfg = len(configurations_tuple)
    for cfg_idx, configuration in enumerate(configurations_tuple, start=1):
        print(f"[{cfg_idx:3d}/{total_cfg}] {configuration.configuration_id}  shared={configuration.shared_resources}", flush=True)
        for workload in workloads:
            for trial_id in range(trials):
                phase_ns = deterministic_trial_phase(seed, workload.workload_name, trial_id)
                sched_seed = deterministic_schedule_seed(seed, workload.workload_name, trial_id)
                a_specs = attacker_specs(trial_id, workload.workload_name, observation_window_ns)
                v_specs = victim_specs(workload, trial_id, phase_ns, sched_seed, observation_window_ns)
                combined_specs = sorted(a_specs + v_specs, key=lambda s: (s.ready_ns, s.tenant, s.request_index))

                sim = RemoteOperationSimulator(
                    configuration=configuration,
                    specs=combined_specs,
                    run_kind="combined",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                )
                combined, waits, intervals, epr, gens = sim.run()
                waits_attr = add_cross_tenant_wait_attribution(waits, intervals)

                attacker_only = baseline_cache[(workload.workload_name, trial_id, "attacker_only")]
                victim_only = baseline_cache[(workload.workload_name, trial_id, "victim_only")]
                trace = paired_attacker_trace(
                    configuration, workload.workload_name, trial_id, attacker_only, combined
                )
                evaluator_trace_rows.append(trace)
                blackbox_cols = [
                    "trace_id", "configuration_id", "workload_name", "trial_id",
                    "probe_index", "release_ns", "attacker_only_success", "combined_success",
                    "attacker_only_completion_ns", "combined_completion_ns",
                    "attacker_only_turnaround_ns", "combined_turnaround_ns",
                    "excess_turnaround_ns", "affected", "speedup", "failure_transition",
                ]
                blackbox_rows.append(trace[blackbox_cols].copy())
                features = trace_feature_row(trace)
                slowdown = victim_slowdown_metrics(victim_only, combined)

                # Resource-specific cross-tenant wait totals for attacker requests.
                attacker_waits = waits_attr[waits_attr["tenant"] == "attacker"] if not waits_attr.empty else pd.DataFrame()
                cross_wait = {
                    f"attacker_cross_tenant_wait_{r}_ns": float(
                        attacker_waits.loc[attacker_waits["resource_name"] == r, "cross_tenant_wait_ns"].sum()
                    ) if not attacker_waits.empty else 0.0
                    for r in RESOURCE_NAMES if r != "epr_pool"
                }

                row = {
                    **features,
                    **slowdown,
                    **cross_wait,
                }
                trial_rows.append(row)
                feature_rows.append(features)
                if not waits_attr.empty:
                    wait_rows.append(waits_attr)
                if not intervals.empty:
                    interval_rows.append(intervals)
                if not epr.empty:
                    epr_rows.append(epr)
                if not gens.empty:
                    generation_rows.append(gens)

    blackbox = pd.concat(blackbox_rows, ignore_index=True) if blackbox_rows else pd.DataFrame()
    evaluator_trace = pd.concat(evaluator_trace_rows, ignore_index=True) if evaluator_trace_rows else pd.DataFrame()
    features_df = pd.DataFrame(feature_rows)
    trial_summary = pd.DataFrame(trial_rows)
    wait_events = pd.concat(wait_rows, ignore_index=True) if wait_rows else pd.DataFrame()
    interval_events = pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame()
    epr_events = pd.concat(epr_rows, ignore_index=True) if epr_rows else pd.DataFrame()
    generation_events = pd.concat(generation_rows, ignore_index=True) if generation_rows else pd.DataFrame()

    classifier_metrics, classifier_predictions = nearest_centroid_fingerprinting(features_df)
    config_summary = configuration_metric_summary(
        trial_summary,
        classifier_metrics,
        configurations_df,
    )

    # Resource-level summaries.
    resource_wait_summary = pd.DataFrame()
    if not wait_events.empty:
        resource_wait_summary = (
            wait_events.groupby(["configuration_id", "resource_name"], as_index=False)
            .agg(
                acquisition_count=("request_id", "size"),
                total_wait_ns=("wait_ns", "sum"),
                total_cross_tenant_wait_ns=("cross_tenant_wait_ns", "sum"),
                mean_wait_ns=("wait_ns", "mean"),
                max_wait_ns=("wait_ns", "max"),
            )
        )

    main_effects = pd.DataFrame()
    interactions = pd.DataFrame()
    necessity_sufficiency = necessity_sufficiency_summary(config_summary)
    pair_dominance = pair_dominance_summary(config_summary)
    if matrix_mode == "full":
        main_effects = main_effects_full_factorial(config_summary)
        interactions = pairwise_interactions_full_factorial(config_summary)

    validations = build_validations(
        configurations_df,
        trial_summary,
        blackbox,
        wait_events,
        config_summary,
        matrix_mode,
    )
    all_passed = bool(validations["passed"].all()) if not validations.empty else False
    validation_summary = pd.DataFrame([
        {
            "validation_assertion_count": int(len(validations)),
            "passed_assertions": int(validations["passed"].sum()),
            "failed_assertions": int((~validations["passed"]).sum()),
            "all_validations_passed": all_passed,
        }
    ])

    # Write outputs.
    configurations_df.to_csv(output_dir / "phase2_06_configuration_table.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(output_dir / "phase2_06_trial_phase_schedule.csv", index=False)
    trial_summary.to_csv(output_dir / "phase2_06_trial_summary.csv", index=False)
    features_df.to_csv(output_dir / "phase2_06_trace_features.csv", index=False)
    config_summary.to_csv(output_dir / "phase2_06_configuration_summary.csv", index=False)
    classifier_metrics.to_csv(output_dir / "phase2_06_workload_fingerprint_metrics.csv", index=False)
    classifier_predictions.to_csv(output_dir / "phase2_06_workload_fingerprint_predictions.csv", index=False)
    necessity_sufficiency.to_csv(output_dir / "phase2_06_necessity_sufficiency_summary.csv", index=False)
    pair_dominance.to_csv(output_dir / "phase2_06_pair_dominance_summary.csv", index=False)
    if not main_effects.empty:
        main_effects.to_csv(output_dir / "phase2_06_main_effects.csv", index=False)
    if not interactions.empty:
        interactions.to_csv(output_dir / "phase2_06_pairwise_interactions.csv", index=False)
    if not resource_wait_summary.empty:
        resource_wait_summary.to_csv(output_dir / "phase2_06_resource_wait_summary.csv", index=False)

    # Master black-box file retains configuration/workload/trial evaluator keys.
    blackbox.to_csv(output_dir / "phase2_06_blackbox_trace_summary.csv", index=False)
    if not evaluator_trace.empty:
        evaluator_trace.to_csv(output_dir / "phase2_06_evaluator_trace_attribution.csv.gz", index=False, compression=GZIP_COMPRESSION)
    attacker_export_cols = [c for c in BLACKBOX_ALLOWED_COLUMNS if c in blackbox.columns and c not in {"configuration_id"}]
    attacker_export = blackbox[[c for c in attacker_export_cols if c not in {"workload_name", "trial_id"}]].copy()
    attacker_export.to_csv(output_dir / "phase2_06_attacker_visible_trace.csv", index=False)

    if not wait_events.empty:
        wait_events.to_csv(output_dir / "phase2_06_resource_wait_events.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not interval_events.empty:
        interval_events.to_csv(output_dir / "phase2_06_resource_intervals.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not epr_events.empty:
        epr_events.to_csv(output_dir / "phase2_06_epr_state_events.csv.gz", index=False, compression=GZIP_COMPRESSION)
    if not generation_events.empty:
        generation_events.to_csv(output_dir / "phase2_06_epr_generation_events.csv.gz", index=False, compression=GZIP_COMPRESSION)

    validations.to_csv(output_dir / "phase2_06_validation_assertions.csv", index=False)
    validation_summary.to_csv(output_dir / "phase2_06_validation_summary.csv", index=False)

    manifest = {
        "experiment": "Phase 2.6 — Resource Ablation Matrix",
        "output_directory": str(output_dir),
        "matrix_mode": matrix_mode,
        "resource_count": len(RESOURCE_NAMES),
        "resources": list(RESOURCE_NAMES),
        "configuration_count": len(configurations_df),
        "workload_count": len(workloads),
        "trial_count_per_workload_configuration": trials,
        "scenario_workload_trial_tuples": int(len(configurations_df) * len(workloads) * trials),
        "observation_window_ns": observation_window_ns,
        "probe_period_ns": ATTACKER_PERIOD_NS,
        "critical_latency_after_epr_ns": CRITICAL_LATENCY_AFTER_EPR_NS,
        "cleanup_latency_after_epr_ns": CLEANUP_LATENCY_AFTER_EPR_NS,
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
    (output_dir / "phase2_06_run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nPhase 2.6 complete")
    print(f"Output directory: {output_dir}")
    print(f"Configurations: {len(configurations_df)}")
    print(f"Trial tuples: {len(trial_summary)}")
    print(f"Validation: {int(validations['passed'].sum())}/{len(validations)} passed")
    if not all_passed:
        failed = validations[~validations["passed"]]
        print("Failed validations:")
        print(failed[["validation_group", "assertion_name", "expected", "observed"]].to_string(index=False))
        if FAIL_ON_VALIDATION_ERROR:
            raise RuntimeError("Phase 2.6 validation failed")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.6 resource ablation matrix")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--observation-window-ns", type=float, default=DEFAULT_OBSERVATION_WINDOW_NS)
    p.add_argument("--matrix", choices=("full", "core"), default="full")
    return p.parse_args()


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
        matrix_mode=args.matrix,
    )


if __name__ == "__main__":
    main()
