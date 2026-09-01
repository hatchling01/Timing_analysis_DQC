#!/usr/bin/env python3
"""
Phase 2.5 — EPR Generation, Storage, and Prefetch
=================================================

Purpose
-------
Embed an explicitly entanglement-mediated remote-operation realization inside
Phase 2's validated staged timing model.  Unlike Phases 2.1–2.4, a remote
operation in this experiment must first acquire an EPR pair.  The pair can be:

  * generated on demand,
  * consumed from a prefetched shared pool,
  * consumed from a finite buffered pool,
  * consumed from a per-tenant reserved pool.

Pairs have finite storage capacity and lifetime.  Generation has finite latency
and capacity.  Selected robustness scenarios also include deterministic failed
generation attempts.  Background refill/prefetch is decoupled from request
consumption, allowing a victim to change the EPR-resource state seen by a later
attacker probe even when the two requests never directly overlap at the
entanglement generator.

Security question
-----------------
The main question is not simply whether a shared generator queues.  The deeper
question is whether decoupling EPR generation from consumption moves victim
information into persistent entanglement-management state:

    victim consumes prefetched pair
        -> shared pool becomes depleted
        -> background refill begins or is delayed
        -> later attacker arrives after the victim request
        -> attacker misses in the EPR pool and waits for regeneration

That is a cache-like persistent resource-state channel rather than ordinary
simultaneous queue contention.

Black-box boundary
------------------
The attacker-visible file contains only its own release, completion/failure,
and turnaround timing.  Pool occupancy, generation ownership, pair identifiers,
sharing mode, depletion cause, and victim workload labels are evaluator-only.

Default output directory
------------------------
blackbox_window_results/phase2/phase2_05_epr_generation_storage_prefetch/

Run
---
    python phase2_05_epr_generation_storage_prefetch.py
    python phase2_05_epr_generation_storage_prefetch.py --trials 4

Notes
-----
* The numerical stage durations are controlled architectural parameters, not
  claims about a particular vendor implementation.
* Primary causal scenarios use deterministic successful EPR generation.
  Failed attempts are introduced only in an explicit robustness scenario.
* All non-EPR remote-operation resources are tenant-dedicated in this phase so
  that observed cross-tenant effects are attributable to EPR management.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Global configuration
# =============================================================================

DEFAULT_TRIALS = 10
DEFAULT_SEED = 2505
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
    / "phase2_05_epr_generation_storage_prefetch"
)

# Entanglement-mediated remote operation after a pair has been acquired.
# The EPR pair is consumed before these local/feedback stages execute.
REMOTE_STAGE_DURATIONS_NS = {
    "epr_bind_and_local_prepare": 20.0,
    "bell_measurement": 70.0,
    "classical_feedforward": 40.0,
    "conditional_control": 20.0,
    "reset_recovery": 120.0,
}
CRITICAL_REMOTE_STAGES = (
    "epr_bind_and_local_prepare",
    "bell_measurement",
    "classical_feedforward",
    "conditional_control",
)
POSTCOMPLETION_REMOTE_STAGES = ("reset_recovery",)
REMOTE_CRITICAL_LATENCY_NS = float(
    sum(REMOTE_STAGE_DURATIONS_NS[s] for s in CRITICAL_REMOTE_STAGES)
)
REMOTE_CLEANUP_LATENCY_NS = float(sum(REMOTE_STAGE_DURATIONS_NS.values()))

BLACKBOX_ALLOWED_COLUMNS = {
    "trace_id",
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
class EPRScenario:
    scenario_name: str
    mode: str
    shared_generation: bool
    generation_capacity: int
    generation_latency_ns: float
    generation_success_probability: float
    max_generation_retries: int
    shared_storage: bool
    storage_capacity: int
    prefetch_target: int
    refill_policy: str
    low_water_mark: int
    pair_lifetime_ns: float
    strict_reserved_admission: bool
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
class RequestState:
    spec: RequestSpec
    success: Optional[bool] = None
    failure_reason: str = ""
    failure_observed_ns: float = math.nan
    epr_acquired_ns: float = math.nan
    epr_wait_ns: float = math.nan
    acquisition_source: str = ""
    pair_id: str = ""
    pair_age_ns: float = math.nan
    pool_key: str = ""
    pool_level_at_arrival: int = 0
    active_generation_tenants_at_arrival: str = ""
    generation_queue_depth_at_arrival: int = 0
    direct_generation_contention: bool = False
    victim_depleted_pool_at_arrival: bool = False
    persistent_victim_depletion: bool = False
    persistent_without_direct_generation: bool = False
    last_victim_consumption_ns_at_arrival: float = math.nan
    last_consumption_tenant_at_arrival: str = ""
    generation_trigger_tenant: str = ""
    generation_attempts_for_request: int = 0


@dataclass
class EPRPair:
    pair_id: str
    pool_key: str
    generated_ns: float
    expires_ns: float
    generation_attempt_id: str
    generation_trigger_tenant: str
    stored: bool = True
    consumed: bool = False
    consumed_ns: float = math.nan
    consumed_by_tenant: str = ""
    expired: bool = False


@dataclass
class GenerationJob:
    job_id: str
    pool_key: str
    trigger_kind: str
    trigger_tenant: str
    requester_id: str
    enqueued_ns: float
    blocking_tenants_at_enqueue: str = ""
    retry_index: int = 0


@dataclass
class ActiveGeneration:
    attempt_id: str
    job: GenerationJob
    lane: int
    start_ns: float
    end_ns: float
    success: bool
    blocking_owner_tenants_at_enqueue: str


@dataclass
class ValidationAssertion:
    validation_group: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


# =============================================================================
# Scenario and workload definitions
# =============================================================================


def _scenario(
    name: str,
    *,
    mode: str,
    shared_generation: bool = True,
    generation_capacity: int = 1,
    generation_latency_ns: float = 240.0,
    generation_success_probability: float = 1.0,
    max_generation_retries: int = 3,
    shared_storage: bool = True,
    storage_capacity: int = 0,
    prefetch_target: int = 0,
    refill_policy: str = "none",
    low_water_mark: int = 0,
    pair_lifetime_ns: float = 1_200.0,
    strict_reserved_admission: bool = False,
    description: str,
    mechanism: str,
    family: str,
) -> EPRScenario:
    return EPRScenario(
        scenario_name=name,
        mode=mode,
        shared_generation=shared_generation,
        generation_capacity=generation_capacity,
        generation_latency_ns=generation_latency_ns,
        generation_success_probability=generation_success_probability,
        max_generation_retries=max_generation_retries,
        shared_storage=shared_storage,
        storage_capacity=storage_capacity,
        prefetch_target=prefetch_target,
        refill_policy=refill_policy,
        low_water_mark=low_water_mark,
        pair_lifetime_ns=pair_lifetime_ns,
        strict_reserved_admission=strict_reserved_admission,
        description=description,
        expected_mechanism=mechanism,
        scenario_family=family,
    )


def build_scenarios() -> tuple[EPRScenario, ...]:
    """Primary modes plus isolation, capacity, expiration, and failure controls."""

    return (
        _scenario(
            "isolated_on_demand",
            mode="on_demand",
            shared_generation=False,
            generation_capacity=1,
            description="Tenant-dedicated on-demand EPR generators; no shared EPR state.",
            mechanism="negative_control",
            family="isolated",
        ),
        _scenario(
            "shared_on_demand_gen1",
            mode="on_demand",
            shared_generation=True,
            generation_capacity=1,
            description="One shared generator; every remote operation requests a fresh pair.",
            mechanism="direct_generation_queueing",
            family="on_demand",
        ),
        _scenario(
            "shared_on_demand_gen2",
            mode="on_demand",
            shared_generation=True,
            generation_capacity=2,
            description="Two shared generation lanes for the tested two-tenant concurrency.",
            mechanism="generation_capacity_control",
            family="on_demand",
        ),
        _scenario(
            "shared_prefetch_pool1",
            mode="prefetch",
            shared_generation=True,
            generation_capacity=1,
            shared_storage=True,
            storage_capacity=1,
            prefetch_target=1,
            refill_policy="maintain_target",
            description="Single-pair shared prefetch cache with asynchronous refill.",
            mechanism="prefetch_depletion_refill",
            family="prefetch",
        ),
        _scenario(
            "shared_prefetch_pool2",
            mode="prefetch",
            shared_generation=True,
            generation_capacity=1,
            shared_storage=True,
            storage_capacity=2,
            prefetch_target=2,
            refill_policy="maintain_target",
            description="Two-pair shared prefetch cache with one refill lane.",
            mechanism="prefetch_depletion_refill",
            family="prefetch",
        ),
        _scenario(
            "shared_prefetch_pool2_gen2",
            mode="prefetch",
            shared_generation=True,
            generation_capacity=2,
            shared_storage=True,
            storage_capacity=2,
            prefetch_target=2,
            refill_policy="maintain_target",
            description="Two-pair shared cache and two generation lanes.",
            mechanism="prefetch_capacity_control",
            family="prefetch",
        ),
        _scenario(
            "shared_buffer4_lowwater1",
            mode="buffered",
            shared_generation=True,
            generation_capacity=1,
            shared_storage=True,
            storage_capacity=4,
            prefetch_target=4,
            refill_policy="low_water_batch",
            low_water_mark=1,
            description="Finite four-pair shared buffer refilled in batches only at low water.",
            mechanism="buffer_depletion_batch_refill",
            family="buffered",
        ),
        _scenario(
            "reserved_pool1_shared_generator",
            mode="reserved",
            shared_generation=True,
            generation_capacity=1,
            shared_storage=False,
            storage_capacity=1,
            prefetch_target=1,
            refill_policy="maintain_target",
            description="One reserved pair slot per tenant; refill generator remains shared.",
            mechanism="reserved_storage_shared_refill",
            family="reserved",
        ),
        _scenario(
            "reserved_pool1_dedicated_generators",
            mode="reserved",
            shared_generation=False,
            generation_capacity=1,
            shared_storage=False,
            storage_capacity=1,
            prefetch_target=1,
            refill_policy="maintain_target",
            description="One reserved pair slot and dedicated refill generator per tenant.",
            mechanism="full_entanglement_isolation_control",
            family="reserved",
        ),
        _scenario(
            "strict_reserved_pool1_shared_generator",
            mode="reserved",
            shared_generation=True,
            generation_capacity=1,
            shared_storage=False,
            storage_capacity=1,
            prefetch_target=1,
            refill_policy="maintain_target",
            strict_reserved_admission=True,
            description="Reserved slots cannot borrow; an empty reservation causes immediate admission failure.",
            mechanism="reservation_failure_channel",
            family="reserved",
        ),
        _scenario(
            "shared_prefetch_short_lifetime",
            mode="prefetch",
            shared_generation=True,
            generation_capacity=1,
            shared_storage=True,
            storage_capacity=2,
            prefetch_target=2,
            refill_policy="maintain_target",
            pair_lifetime_ns=350.0,
            description="Short-lived prefetched pairs create expiration/refill churn.",
            mechanism="expiration_state_channel",
            family="lifetime",
        ),
        _scenario(
            "shared_prefetch_unreliable_generation",
            mode="prefetch",
            shared_generation=True,
            generation_capacity=1,
            generation_latency_ns=180.0,
            generation_success_probability=0.75,
            max_generation_retries=4,
            shared_storage=True,
            storage_capacity=2,
            prefetch_target=2,
            refill_policy="maintain_target",
            pair_lifetime_ns=1_200.0,
            description="Prefetch with deterministic reproducible failed generation attempts and retry.",
            mechanism="generation_failure_refill_jitter",
            family="generation_failure",
        ),
    )


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


def attacker_specs(
    trial_id: int,
    workload_name: str,
    observation_window_ns: float,
) -> list[RequestSpec]:
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
    rng: np.random.Generator,
    observation_window_ns: float,
) -> list[RequestSpec]:
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


# =============================================================================
# Entanglement resource manager
# =============================================================================


class EPRManager:
    """Event-driven finite EPR generation/storage manager."""

    def __init__(
        self,
        *,
        scenario: EPRScenario,
        run_kind: str,
        workload_name: str,
        trial_id: int,
        seed: int,
        observation_window_ns: float,
        specs: Iterable[RequestSpec],
    ) -> None:
        self.scenario = scenario
        self.run_kind = run_kind
        self.workload_name = workload_name
        self.trial_id = trial_id
        self.seed = seed
        self.observation_window_ns = observation_window_ns
        self.refill_cutoff_ns = observation_window_ns
        self.specs = sorted(list(specs), key=lambda x: (x.ready_ns, x.tenant, x.request_index))
        self.requests: dict[str, RequestState] = {
            s.request_id: RequestState(spec=s) for s in self.specs
        }

        self.events: list[tuple[float, int, int, str, object]] = []
        self.event_seq = 0
        self.job_seq = 0
        self.attempt_seq = 0
        self.pair_seq = 0

        self.pools: dict[str, list[EPRPair]] = defaultdict(list)
        self.waiters: dict[str, Deque[str]] = defaultdict(deque)
        self.generation_queue: Deque[GenerationJob] = deque()
        self.active_generation: dict[int, ActiveGeneration] = {}
        self.inflight_refills: dict[str, int] = defaultdict(int)

        self.last_consumption_tenant: dict[str, str] = defaultdict(str)
        self.last_consumption_ns: dict[str, float] = defaultdict(lambda: math.nan)
        self.last_victim_consumption_ns: dict[str, float] = defaultdict(lambda: math.nan)

        self.pairs: dict[str, EPRPair] = {}
        self.generation_rows: list[dict[str, object]] = []
        self.pool_event_rows: list[dict[str, object]] = []
        self.request_event_rows: list[dict[str, object]] = []

        self._initialize_storage()
        for spec in self.specs:
            self._push_event(spec.ready_ns, 2, "request_arrival", spec.request_id)

    # ------------------------------------------------------------------
    # Physical naming / pool structure
    # ------------------------------------------------------------------

    def pool_key_for_tenant(self, tenant: str) -> str:
        if self.scenario.mode == "on_demand":
            return f"{tenant}::direct"
        if self.scenario.shared_storage:
            return "shared::epr_pool"
        return f"{tenant}::reserved_epr_pool"

    def generator_scope_for_tenant(self, tenant: str) -> str:
        return "shared" if self.scenario.shared_generation else tenant

    def generation_lane_count(self, tenant: str) -> int:
        return self.scenario.generation_capacity

    def _physical_lane_key(self, tenant: str, lane: int) -> tuple[str, int]:
        scope = self.generator_scope_for_tenant(tenant)
        # The integer lane is sufficient for shared generators.  Dedicated
        # generators are multiplexed into a synthetic large lane index so their
        # active states cannot block the other tenant.
        if scope == "shared":
            return ("shared", lane)
        offset = 0 if tenant == "attacker" else 1000
        return (tenant, offset + lane)

    # ------------------------------------------------------------------
    # Event utilities
    # ------------------------------------------------------------------

    def _push_event(self, time_ns: float, priority: int, kind: str, payload: object) -> None:
        self.event_seq += 1
        heapq.heappush(self.events, (float(time_ns), priority, self.event_seq, kind, payload))

    def _initialize_storage(self) -> None:
        if self.scenario.mode == "on_demand" or self.scenario.storage_capacity <= 0:
            return
        pool_keys = ["shared::epr_pool"] if self.scenario.shared_storage else [
            "attacker::reserved_epr_pool",
            "victim::reserved_epr_pool",
        ]
        for pool_key in pool_keys:
            initial = min(self.scenario.prefetch_target, self.scenario.storage_capacity)
            for _ in range(initial):
                self._create_stored_pair(
                    pool_key=pool_key,
                    now_ns=0.0,
                    attempt_id="initial_prefill",
                    trigger_tenant="prefill",
                )

    def _record_pool_event(
        self,
        *,
        now_ns: float,
        pool_key: str,
        event_kind: str,
        tenant: str = "",
        request_id: str = "",
        pair_id: str = "",
        details: str = "",
    ) -> None:
        capacity = self.scenario.storage_capacity if self.scenario.mode != "on_demand" else 0
        self.pool_event_rows.append(
            {
                "run_kind": self.run_kind,
                "scenario_name": self.scenario.scenario_name,
                "workload_name": self.workload_name,
                "trial_id": self.trial_id,
                "time_ns": float(now_ns),
                "pool_key": pool_key,
                "event_kind": event_kind,
                "tenant": tenant,
                "request_id": request_id,
                "pair_id": pair_id,
                "occupancy_after": len(self.pools.get(pool_key, [])),
                "capacity": capacity,
                "waiting_requests": len(self.waiters.get(pool_key, [])),
                "inflight_refills": self.inflight_refills.get(pool_key, 0),
                "details": details,
            }
        )

    # ------------------------------------------------------------------
    # Pair lifecycle
    # ------------------------------------------------------------------

    def _create_stored_pair(
        self,
        *,
        pool_key: str,
        now_ns: float,
        attempt_id: str,
        trigger_tenant: str,
    ) -> Optional[EPRPair]:
        if self.scenario.storage_capacity <= 0:
            return None
        if len(self.pools[pool_key]) >= self.scenario.storage_capacity:
            self._record_pool_event(
                now_ns=now_ns,
                pool_key=pool_key,
                event_kind="generated_pair_dropped_full",
                details="storage capacity already full",
            )
            return None

        self.pair_seq += 1
        pair = EPRPair(
            pair_id=f"pair::{self.trial_id}::{self.pair_seq}",
            pool_key=pool_key,
            generated_ns=float(now_ns),
            expires_ns=float(now_ns + self.scenario.pair_lifetime_ns),
            generation_attempt_id=attempt_id,
            generation_trigger_tenant=trigger_tenant,
        )
        self.pairs[pair.pair_id] = pair
        self.pools[pool_key].append(pair)
        self._push_event(pair.expires_ns, 1, "pair_expire", pair.pair_id)
        self._record_pool_event(
            now_ns=now_ns,
            pool_key=pool_key,
            event_kind="pair_stored",
            pair_id=pair.pair_id,
            tenant=trigger_tenant,
        )
        return pair

    def _consume_pair(self, *, request_id: str, pair: EPRPair, now_ns: float, source: str) -> None:
        state = self.requests[request_id]
        if pair.expired or pair.consumed or now_ns > pair.expires_ns + FLOAT_TOLERANCE_NS:
            raise RuntimeError(f"Invalid pair consumption: {pair.pair_id}")

        pool = self.pools[pair.pool_key]
        if pair in pool:
            pool.remove(pair)
        pair.stored = False
        pair.consumed = True
        pair.consumed_ns = float(now_ns)
        pair.consumed_by_tenant = state.spec.tenant

        self.last_consumption_tenant[pair.pool_key] = state.spec.tenant
        self.last_consumption_ns[pair.pool_key] = float(now_ns)
        if state.spec.tenant == "victim":
            self.last_victim_consumption_ns[pair.pool_key] = float(now_ns)

        state.success = True
        state.epr_acquired_ns = float(now_ns)
        state.epr_wait_ns = float(now_ns - state.spec.ready_ns)
        state.acquisition_source = source
        state.pair_id = pair.pair_id
        state.pair_age_ns = float(now_ns - pair.generated_ns)
        state.generation_trigger_tenant = pair.generation_trigger_tenant

        if state.spec.tenant == "attacker" and state.epr_wait_ns > AFFECTED_THRESHOLD_NS:
            last_victim = state.last_victim_consumption_ns_at_arrival
            if math.isfinite(last_victim) and last_victim < state.spec.ready_ns - FLOAT_TOLERANCE_NS:
                state.persistent_victim_depletion = state.victim_depleted_pool_at_arrival
                state.persistent_without_direct_generation = (
                    state.persistent_victim_depletion and not state.direct_generation_contention
                )

        self._record_pool_event(
            now_ns=now_ns,
            pool_key=pair.pool_key,
            event_kind="pair_consumed",
            tenant=state.spec.tenant,
            request_id=request_id,
            pair_id=pair.pair_id,
            details=source,
        )
        self._record_request_event(state, "epr_acquired", now_ns)
        self._maybe_refill(pair.pool_key, now_ns, trigger_tenant=state.spec.tenant)

    def _expire_pair(self, pair_id: str, now_ns: float) -> None:
        pair = self.pairs.get(pair_id)
        if pair is None or pair.consumed or pair.expired:
            return
        pool = self.pools[pair.pool_key]
        if pair not in pool:
            return
        pool.remove(pair)
        pair.expired = True
        pair.stored = False
        self._record_pool_event(
            now_ns=now_ns,
            pool_key=pair.pool_key,
            event_kind="pair_expired",
            pair_id=pair.pair_id,
        )
        self._maybe_refill(pair.pool_key, now_ns, trigger_tenant="expiration")

    # ------------------------------------------------------------------
    # Generation lifecycle
    # ------------------------------------------------------------------

    def _generation_success(self, attempt_id: str) -> bool:
        p = self.scenario.generation_success_probability
        if p >= 1.0 - 1e-15:
            return True
        token = (
            f"{self.seed}|{self.scenario.scenario_name}|{self.trial_id}|"
            f"{self.run_kind}|{attempt_id}"
        ).encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=8).digest()
        value = int.from_bytes(digest, "big") / float(2**64 - 1)
        return value < p

    def _active_jobs_for_scope(self, tenant: str) -> list[ActiveGeneration]:
        scope = self.generator_scope_for_tenant(tenant)
        out: list[ActiveGeneration] = []
        for active in self.active_generation.values():
            if self.generator_scope_for_tenant(active.job.trigger_tenant if active.job.trigger_tenant in {"attacker", "victim"} else tenant) == scope:
                # Dedicated prefill/expiration jobs are associated with pool key;
                # infer tenant from reserved pool when needed.
                if scope != "shared":
                    job_tenant = self._job_effective_tenant(active.job)
                    if job_tenant != tenant:
                        continue
                out.append(active)
        return out

    def _job_effective_tenant(self, job: GenerationJob) -> str:
        if job.trigger_tenant in {"attacker", "victim"}:
            return job.trigger_tenant
        if job.pool_key.startswith("attacker::"):
            return "attacker"
        if job.pool_key.startswith("victim::"):
            return "victim"
        return "attacker"  # shared pool only; scope is shared, value is immaterial

    def _enqueue_generation(
        self,
        *,
        pool_key: str,
        trigger_kind: str,
        trigger_tenant: str,
        requester_id: str,
        now_ns: float,
        retry_index: int = 0,
    ) -> None:
        self.job_seq += 1
        preview = GenerationJob(
            job_id="preview",
            pool_key=pool_key,
            trigger_kind=trigger_kind,
            trigger_tenant=trigger_tenant,
            requester_id=requester_id,
            enqueued_ns=float(now_ns),
            retry_index=retry_index,
        )
        blockers_at_enqueue = self._blocking_tenants_for_job(preview)
        job = GenerationJob(
            job_id=f"job::{self.trial_id}::{self.job_seq}",
            pool_key=pool_key,
            trigger_kind=trigger_kind,
            trigger_tenant=trigger_tenant,
            requester_id=requester_id,
            enqueued_ns=float(now_ns),
            blocking_tenants_at_enqueue=";".join(sorted(blockers_at_enqueue)),
            retry_index=retry_index,
        )
        if requester_id and trigger_tenant == "attacker" and self.scenario.shared_generation:
            direct_victim_active = any(
                a.job.trigger_kind == "demand"
                and self._job_effective_tenant(a.job) == "victim"
                for a in self.active_generation.values()
            )
            direct_victim_queued = any(
                j.trigger_kind == "demand"
                and self._job_effective_tenant(j) == "victim"
                for j in self.generation_queue
            )
            if direct_victim_active or direct_victim_queued:
                self.requests[requester_id].direct_generation_contention = True
        self.generation_queue.append(job)
        if trigger_kind == "refill":
            self.inflight_refills[pool_key] += 1
        if requester_id:
            self.requests[requester_id].generation_attempts_for_request += 1
        self._dispatch_generation(now_ns)

    def _find_free_lane(self, job: GenerationJob) -> Optional[int]:
        tenant = self._job_effective_tenant(job)
        scope = self.generator_scope_for_tenant(tenant)
        for lane in range(self.scenario.generation_capacity):
            key = lane if scope == "shared" else (0 if tenant == "attacker" else 1000) + lane
            if key not in self.active_generation:
                return key
        return None

    def _blocking_tenants_for_job(self, job: GenerationJob) -> set[str]:
        tenant = self._job_effective_tenant(job)
        scope = self.generator_scope_for_tenant(tenant)
        blockers: set[str] = set()
        for active in self.active_generation.values():
            active_tenant = self._job_effective_tenant(active.job)
            active_scope = self.generator_scope_for_tenant(active_tenant)
            if active_scope == scope:
                blockers.add(active.job.trigger_tenant if active.job.trigger_tenant in {"attacker", "victim"} else active_tenant)
        return blockers

    def _dispatch_generation(self, now_ns: float) -> None:
        if not self.generation_queue:
            return

        # Repeatedly scan the queue.  This allows a dedicated attacker generator
        # job to start even if a victim job is waiting for the victim's dedicated
        # lane, while preserving FCFS within each physical scope.
        made_progress = True
        while made_progress and self.generation_queue:
            made_progress = False
            q_len = len(self.generation_queue)
            for _ in range(q_len):
                job = self.generation_queue.popleft()
                lane = self._find_free_lane(job)
                if lane is None:
                    self.generation_queue.append(job)
                    continue

                blockers = {x for x in job.blocking_tenants_at_enqueue.split(";") if x}
                self.attempt_seq += 1
                attempt_id = f"attempt::{self.trial_id}::{self.attempt_seq}"
                success = self._generation_success(attempt_id)
                end_ns = now_ns + self.scenario.generation_latency_ns
                active = ActiveGeneration(
                    attempt_id=attempt_id,
                    job=job,
                    lane=lane,
                    start_ns=float(now_ns),
                    end_ns=float(end_ns),
                    success=success,
                    blocking_owner_tenants_at_enqueue=";".join(sorted(blockers)),
                )
                self.active_generation[lane] = active
                self._push_event(end_ns, 0, "generation_complete", lane)
                made_progress = True

                self.generation_rows.append(
                    {
                        "run_kind": self.run_kind,
                        "scenario_name": self.scenario.scenario_name,
                        "workload_name": self.workload_name,
                        "trial_id": self.trial_id,
                        "attempt_id": attempt_id,
                        "job_id": job.job_id,
                        "trigger_kind": job.trigger_kind,
                        "trigger_tenant": job.trigger_tenant,
                        "effective_tenant": self._job_effective_tenant(job),
                        "requester_id": job.requester_id,
                        "pool_key": job.pool_key,
                        "lane": lane,
                        "enqueue_ns": job.enqueued_ns,
                        "start_ns": float(now_ns),
                        "end_ns": float(end_ns),
                        "queue_wait_ns": float(now_ns - job.enqueued_ns),
                        "service_ns": self.scenario.generation_latency_ns,
                        "success": success,
                        "retry_index": job.retry_index,
                        "blocking_owner_tenants_at_enqueue": ";".join(sorted(blockers)),
                        "cross_tenant_blocked": bool(
                            job.trigger_tenant in {"attacker", "victim"}
                            and any(b != job.trigger_tenant for b in blockers)
                        ),
                    }
                )

            # If no queued job could start, all applicable lanes are busy.

    def _complete_generation(self, lane: int, now_ns: float) -> None:
        active = self.active_generation.pop(lane, None)
        if active is None:
            return
        job = active.job
        if job.trigger_kind == "refill":
            self.inflight_refills[job.pool_key] = max(0, self.inflight_refills[job.pool_key] - 1)

        if active.success:
            if job.trigger_kind == "demand":
                # Fresh pair is generated specifically for this request and is
                # consumed immediately; it never occupies storage.
                state = self.requests[job.requester_id]
                if state.success is None:
                    self.pair_seq += 1
                    pair = EPRPair(
                        pair_id=f"pair::{self.trial_id}::{self.pair_seq}",
                        pool_key=job.pool_key,
                        generated_ns=float(now_ns),
                        expires_ns=float(now_ns + self.scenario.pair_lifetime_ns),
                        generation_attempt_id=active.attempt_id,
                        generation_trigger_tenant=job.trigger_tenant,
                        stored=False,
                    )
                    self.pairs[pair.pair_id] = pair
                    pair.consumed = True
                    pair.consumed_ns = float(now_ns)
                    pair.consumed_by_tenant = state.spec.tenant
                    state.success = True
                    state.epr_acquired_ns = float(now_ns)
                    state.epr_wait_ns = float(now_ns - state.spec.ready_ns)
                    state.acquisition_source = "on_demand_generation"
                    state.pair_id = pair.pair_id
                    state.pair_age_ns = 0.0
                    state.generation_trigger_tenant = job.trigger_tenant
                    self._record_request_event(state, "epr_acquired", now_ns)
            else:
                # Refill completion first serves a waiting request; otherwise it
                # populates the finite store.
                while self.waiters[job.pool_key]:
                    requester = self.waiters[job.pool_key].popleft()
                    state = self.requests[requester]
                    if state.success is not None:
                        continue
                    self.pair_seq += 1
                    pair = EPRPair(
                        pair_id=f"pair::{self.trial_id}::{self.pair_seq}",
                        pool_key=job.pool_key,
                        generated_ns=float(now_ns),
                        expires_ns=float(now_ns + self.scenario.pair_lifetime_ns),
                        generation_attempt_id=active.attempt_id,
                        generation_trigger_tenant=job.trigger_tenant,
                        stored=False,
                    )
                    self.pairs[pair.pair_id] = pair
                    # Temporary insert lets the common consume path enforce the
                    # same single-use/lifetime checks and update causal state.
                    self.pools[job.pool_key].append(pair)
                    self._consume_pair(
                        request_id=requester,
                        pair=pair,
                        now_ns=now_ns,
                        source="refill_generation",
                    )
                    break
                else:
                    self._create_stored_pair(
                        pool_key=job.pool_key,
                        now_ns=now_ns,
                        attempt_id=active.attempt_id,
                        trigger_tenant=job.trigger_tenant,
                    )
        else:
            # Failed attempts retry demand requests or refill deficits.
            if job.retry_index < self.scenario.max_generation_retries:
                self._enqueue_generation(
                    pool_key=job.pool_key,
                    trigger_kind=job.trigger_kind,
                    trigger_tenant=job.trigger_tenant,
                    requester_id=job.requester_id,
                    now_ns=now_ns,
                    retry_index=job.retry_index + 1,
                )
            elif job.trigger_kind == "demand" and job.requester_id:
                state = self.requests[job.requester_id]
                if state.success is None:
                    self._fail_request(state, now_ns, "generation_retries_exhausted")

        if job.trigger_kind == "refill":
            self._maybe_refill(job.pool_key, now_ns, trigger_tenant=job.trigger_tenant)
        self._dispatch_generation(now_ns)

    # ------------------------------------------------------------------
    # Refill policy
    # ------------------------------------------------------------------

    def _queued_refills_for_pool(self, pool_key: str) -> int:
        return sum(1 for job in self.generation_queue if job.trigger_kind == "refill" and job.pool_key == pool_key)

    def _desired_refills(self, pool_key: str) -> int:
        if self.scenario.mode == "on_demand" or self.scenario.storage_capacity <= 0:
            return 0
        occupancy = len(self.pools[pool_key])
        inflight = self.inflight_refills.get(pool_key, 0) + self._queued_refills_for_pool(pool_key)
        waiting = sum(1 for rid in self.waiters[pool_key] if self.requests[rid].success is None)

        if self.scenario.refill_policy == "maintain_target":
            desired_available = max(self.scenario.prefetch_target, waiting)
            return max(0, desired_available - occupancy - inflight)

        if self.scenario.refill_policy == "low_water_batch":
            # Only launch a batch after the store reaches the low-water mark;
            # once launched, refill toward the configured target.
            if occupancy > self.scenario.low_water_mark and waiting == 0 and inflight == 0:
                return 0
            desired_available = max(self.scenario.prefetch_target, waiting)
            return max(0, desired_available - occupancy - inflight)

        return 0

    def _maybe_refill(self, pool_key: str, now_ns: float, trigger_tenant: str) -> None:
        if now_ns > self.refill_cutoff_ns + FLOAT_TOLERANCE_NS:
            return
        count = self._desired_refills(pool_key)
        for _ in range(count):
            self._enqueue_generation(
                pool_key=pool_key,
                trigger_kind="refill",
                trigger_tenant=trigger_tenant,
                requester_id="",
                now_ns=now_ns,
            )

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _active_generation_tenants(self, tenant: str) -> set[str]:
        scope = self.generator_scope_for_tenant(tenant)
        out: set[str] = set()
        for active in self.active_generation.values():
            eff = self._job_effective_tenant(active.job)
            if self.generator_scope_for_tenant(eff) != scope:
                continue
            out.add(active.job.trigger_tenant if active.job.trigger_tenant in {"attacker", "victim"} else eff)
        return out

    def _record_request_event(self, state: RequestState, event_kind: str, now_ns: float) -> None:
        self.request_event_rows.append(
            {
                "run_kind": self.run_kind,
                "scenario_name": self.scenario.scenario_name,
                "workload_name": self.workload_name,
                "trial_id": self.trial_id,
                "request_id": state.spec.request_id,
                "tenant": state.spec.tenant,
                "request_index": state.spec.request_index,
                "event_kind": event_kind,
                "time_ns": float(now_ns),
                "pool_key": state.pool_key,
                "pool_level_at_arrival": state.pool_level_at_arrival,
            }
        )

    def _fail_request(self, state: RequestState, now_ns: float, reason: str) -> None:
        if state.success is not None:
            return
        state.success = False
        state.failure_reason = reason
        state.failure_observed_ns = float(now_ns)
        state.epr_wait_ns = float(now_ns - state.spec.ready_ns)
        self._record_request_event(state, "request_failed", now_ns)

    def _request_arrival(self, request_id: str, now_ns: float) -> None:
        state = self.requests[request_id]
        tenant = state.spec.tenant
        pool_key = self.pool_key_for_tenant(tenant)
        state.pool_key = pool_key
        state.pool_level_at_arrival = len(self.pools.get(pool_key, []))
        active_tenants = self._active_generation_tenants(tenant)
        state.active_generation_tenants_at_arrival = ";".join(sorted(active_tenants))
        state.generation_queue_depth_at_arrival = len(self.generation_queue)
        state.last_consumption_tenant_at_arrival = self.last_consumption_tenant[pool_key]
        state.last_victim_consumption_ns_at_arrival = self.last_victim_consumption_ns[pool_key]
        state.victim_depleted_pool_at_arrival = (
            tenant == "attacker"
            and state.pool_level_at_arrival == 0
            and self.last_consumption_tenant[pool_key] == "victim"
            and math.isfinite(self.last_victim_consumption_ns[pool_key])
            and self.last_victim_consumption_ns[pool_key] < now_ns - FLOAT_TOLERANCE_NS
        )
        if tenant == "attacker" and self.scenario.shared_generation:
            shared_active = [
                a for a in self.active_generation.values()
                if self.generator_scope_for_tenant(self._job_effective_tenant(a.job)) == "shared"
            ]
            victim_active = any(
                a.job.trigger_kind == "demand"
                and self._job_effective_tenant(a.job) == "victim"
                for a in shared_active
            )
            victim_queued = any(
                j.trigger_kind == "demand"
                and self._job_effective_tenant(j) == "victim"
                for j in self.generation_queue
            )
            if len(shared_active) >= self.scenario.generation_capacity and (victim_active or victim_queued):
                state.direct_generation_contention = True

        self._record_request_event(state, "request_arrival", now_ns)

        if self.scenario.mode == "on_demand":
            self._enqueue_generation(
                pool_key=pool_key,
                trigger_kind="demand",
                trigger_tenant=tenant,
                requester_id=request_id,
                now_ns=now_ns,
            )
            return

        # Remove any stale pair that somehow reached the request handler before
        # its expiration event due floating-point ties.
        valid_pairs = [
            p for p in self.pools[pool_key]
            if not p.expired and not p.consumed and p.expires_ns >= now_ns - FLOAT_TOLERANCE_NS
        ]
        self.pools[pool_key] = valid_pairs

        if valid_pairs:
            # Oldest-first consumption makes pair age/lifetime behavior explicit.
            pair = min(valid_pairs, key=lambda p: (p.generated_ns, p.pair_id))
            source = "prefetched_pool_hit" if self.scenario.mode == "prefetch" else (
                "buffer_hit" if self.scenario.mode == "buffered" else "reserved_pool_hit"
            )
            self._consume_pair(request_id=request_id, pair=pair, now_ns=now_ns, source=source)
            return

        if self.scenario.strict_reserved_admission:
            self._fail_request(state, now_ns, "reserved_pool_empty_no_borrow")
            self._maybe_refill(pool_key, now_ns, trigger_tenant=tenant)
            return

        self.waiters[pool_key].append(request_id)
        self._record_pool_event(
            now_ns=now_ns,
            pool_key=pool_key,
            event_kind="request_waits_for_pair",
            tenant=tenant,
            request_id=request_id,
        )
        self._maybe_refill(pool_key, now_ns, trigger_tenant=tenant)

    # ------------------------------------------------------------------
    # Main loop / exports
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        while self.events:
            now_ns, _, _, kind, payload = heapq.heappop(self.events)
            if kind == "generation_complete":
                self._complete_generation(int(payload), now_ns)
            elif kind == "pair_expire":
                self._expire_pair(str(payload), now_ns)
            elif kind == "request_arrival":
                self._request_arrival(str(payload), now_ns)
            else:
                raise RuntimeError(kind)

        # Any residual demand waiter after generation cutoff is a model failure;
        # mark it rather than silently dropping the request.
        for state in self.requests.values():
            if state.success is None:
                self._fail_request(state, self.observation_window_ns, "unresolved_epr_request")

        request_df = self._request_dataframe()
        stage_df = build_stage_records(request_df)
        generation_df = pd.DataFrame(self.generation_rows)
        pool_events_df = pd.DataFrame(self.pool_event_rows)
        pairs_df = pd.DataFrame(self._pair_rows())
        request_events_df = pd.DataFrame(self.request_event_rows)
        return request_df, stage_df, generation_df, pool_events_df, pairs_df, request_events_df

    def _request_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for state in sorted(self.requests.values(), key=lambda x: (x.spec.tenant, x.spec.request_index)):
            success = bool(state.success)
            external_completion = (
                state.epr_acquired_ns + REMOTE_CRITICAL_LATENCY_NS if success else math.nan
            )
            cleanup_completion = (
                state.epr_acquired_ns + REMOTE_CLEANUP_LATENCY_NS if success else math.nan
            )
            rows.append(
                {
                    "run_kind": self.run_kind,
                    "scenario_name": self.scenario.scenario_name,
                    "workload_name": self.workload_name,
                    "trial_id": self.trial_id,
                    "request_id": state.spec.request_id,
                    "tenant": state.spec.tenant,
                    "request_index": state.spec.request_index,
                    "ready_ns": state.spec.ready_ns,
                    "success": success,
                    "failure_reason": state.failure_reason,
                    "failure_observed_ns": state.failure_observed_ns,
                    "epr_acquired_ns": state.epr_acquired_ns,
                    "epr_wait_ns": state.epr_wait_ns,
                    "acquisition_source": state.acquisition_source,
                    "pair_id": state.pair_id,
                    "pair_age_ns": state.pair_age_ns,
                    "pool_key": state.pool_key,
                    "pool_level_at_arrival": state.pool_level_at_arrival,
                    "active_generation_tenants_at_arrival": state.active_generation_tenants_at_arrival,
                    "generation_queue_depth_at_arrival": state.generation_queue_depth_at_arrival,
                    "direct_generation_contention": state.direct_generation_contention,
                    "victim_depleted_pool_at_arrival": state.victim_depleted_pool_at_arrival,
                    "persistent_victim_depletion": state.persistent_victim_depletion,
                    "persistent_without_direct_generation": state.persistent_without_direct_generation,
                    "last_victim_consumption_ns_at_arrival": state.last_victim_consumption_ns_at_arrival,
                    "last_consumption_tenant_at_arrival": state.last_consumption_tenant_at_arrival,
                    "generation_trigger_tenant": state.generation_trigger_tenant,
                    "generation_attempts_for_request": state.generation_attempts_for_request,
                    "external_completion_ns": external_completion,
                    "cleanup_completion_ns": cleanup_completion,
                    "external_turnaround_ns": (
                        external_completion - state.spec.ready_ns if success else math.nan
                    ),
                    "cleanup_turnaround_ns": (
                        cleanup_completion - state.spec.ready_ns if success else math.nan
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _pair_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for pair in self.pairs.values():
            rows.append(
                {
                    "run_kind": self.run_kind,
                    "scenario_name": self.scenario.scenario_name,
                    "workload_name": self.workload_name,
                    "trial_id": self.trial_id,
                    "pair_id": pair.pair_id,
                    "pool_key": pair.pool_key,
                    "generated_ns": pair.generated_ns,
                    "expires_ns": pair.expires_ns,
                    "generation_attempt_id": pair.generation_attempt_id,
                    "generation_trigger_tenant": pair.generation_trigger_tenant,
                    "consumed": pair.consumed,
                    "consumed_ns": pair.consumed_ns,
                    "consumed_by_tenant": pair.consumed_by_tenant,
                    "expired": pair.expired,
                    "stored_at_end": pair.stored,
                }
            )
        return rows


# =============================================================================
# Remote-operation stage expansion
# =============================================================================


def build_stage_records(request_df: pd.DataFrame) -> pd.DataFrame:
    """Expand successful requests through the validated local/feedback pipeline."""
    rows: list[dict[str, object]] = []
    for r in request_df.itertuples(index=False):
        # Dynamic EPR acquisition is represented as a request stage whose wait is
        # the externally relevant EPR-manager delay.
        rows.append(
            {
                "run_kind": r.run_kind,
                "scenario_name": r.scenario_name,
                "workload_name": r.workload_name,
                "trial_id": r.trial_id,
                "request_id": r.request_id,
                "tenant": r.tenant,
                "request_index": r.request_index,
                "stage_index": 0,
                "stage_name": "epr_pair_acquisition",
                "causal_component": "epr_resource_manager",
                "stage_ready_ns": r.ready_ns,
                "stage_start_ns": r.ready_ns,
                "stage_end_ns": r.epr_acquired_ns if r.success else r.failure_observed_ns,
                "wait_ns": r.epr_wait_ns,
                "service_ns": 0.0,
                "critical_path": True,
                "external_completion_here": False,
                "postcompletion": False,
            }
        )
        if not r.success:
            continue
        t = float(r.epr_acquired_ns)
        stage_index = 1
        for stage_name, duration in REMOTE_STAGE_DURATIONS_NS.items():
            start = t
            end = start + duration
            is_post = stage_name in POSTCOMPLETION_REMOTE_STAGES
            rows.append(
                {
                    "run_kind": r.run_kind,
                    "scenario_name": r.scenario_name,
                    "workload_name": r.workload_name,
                    "trial_id": r.trial_id,
                    "request_id": r.request_id,
                    "tenant": r.tenant,
                    "request_index": r.request_index,
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "causal_component": {
                        "epr_bind_and_local_prepare": "tenant_local_endpoint",
                        "bell_measurement": "tenant_local_bell_measurement",
                        "classical_feedforward": "tenant_local_feedforward",
                        "conditional_control": "tenant_local_conditional_control",
                        "reset_recovery": "tenant_local_reset",
                    }[stage_name],
                    "stage_ready_ns": start,
                    "stage_start_ns": start,
                    "stage_end_ns": end,
                    "wait_ns": 0.0,
                    "service_ns": duration,
                    "critical_path": not is_post,
                    "external_completion_here": stage_name == "conditional_control",
                    "postcompletion": is_post,
                }
            )
            t = end
            stage_index += 1
    return pd.DataFrame(rows)


# =============================================================================
# Paired black-box traces and summaries
# =============================================================================


def paired_attacker_trace(attacker_only: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    a = attacker_only[attacker_only["tenant"] == "attacker"].copy()
    c = combined[combined["tenant"] == "attacker"].copy()
    cols = [
        "request_index",
        "ready_ns",
        "success",
        "external_completion_ns",
        "external_turnaround_ns",
    ]
    a = a[cols].rename(
        columns={
            "ready_ns": "release_ns",
            "success": "attacker_only_success",
            "external_completion_ns": "attacker_only_completion_ns",
            "external_turnaround_ns": "attacker_only_turnaround_ns",
        }
    )
    c = c[cols].rename(
        columns={
            "ready_ns": "combined_release_ns",
            "success": "combined_success",
            "external_completion_ns": "combined_completion_ns",
            "external_turnaround_ns": "combined_turnaround_ns",
        }
    )
    out = a.merge(c, on="request_index", how="outer", validate="one_to_one")
    out["probe_index"] = out["request_index"]
    both_success = out["attacker_only_success"].fillna(False) & out["combined_success"].fillna(False)
    out["excess_turnaround_ns"] = np.where(
        both_success,
        out["combined_turnaround_ns"] - out["attacker_only_turnaround_ns"],
        np.nan,
    )
    out["affected"] = both_success & (out["excess_turnaround_ns"] > AFFECTED_THRESHOLD_NS)
    out["speedup"] = both_success & (out["excess_turnaround_ns"] < -AFFECTED_THRESHOLD_NS)
    out["failure_transition"] = (
        out["attacker_only_success"].fillna(False) & ~out["combined_success"].fillna(False)
    )
    return out.sort_values("probe_index").reset_index(drop=True)


def attacker_causal_table(combined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "request_index",
        "epr_wait_ns",
        "acquisition_source",
        "pool_level_at_arrival",
        "direct_generation_contention",
        "victim_depleted_pool_at_arrival",
        "persistent_victim_depletion",
        "persistent_without_direct_generation",
        "last_victim_consumption_ns_at_arrival",
        "active_generation_tenants_at_arrival",
        "generation_queue_depth_at_arrival",
        "generation_attempts_for_request",
    ]
    return combined[combined["tenant"] == "attacker"][cols].copy()


def summarize_trial(
    *,
    scenario: EPRScenario,
    workload: VictimWorkload,
    trial_id: int,
    phase_ns: float,
    trace: pd.DataFrame,
    causal: pd.DataFrame,
    victim_only: pd.DataFrame,
    combined: pd.DataFrame,
) -> dict[str, object]:
    merged = trace.merge(causal, left_on="probe_index", right_on="request_index", how="left")
    successful = merged[merged["attacker_only_success"] & merged["combined_success"]]
    abs_change = successful["excess_turnaround_ns"].abs() if len(successful) else pd.Series(dtype=float)

    v0 = victim_only[victim_only["tenant"] == "victim"].sort_values("request_index")
    vc = combined[combined["tenant"] == "victim"].sort_values("request_index")
    vpair = v0[["request_index", "success", "external_turnaround_ns"]].merge(
        vc[["request_index", "success", "external_turnaround_ns"]],
        on="request_index",
        suffixes=("_alone", "_combined"),
        how="outer",
    )
    v_success = vpair["success_alone"].fillna(False) & vpair["success_combined"].fillna(False)
    victim_slowdown = np.nan
    if v_success.any():
        denom = vpair.loc[v_success, "external_turnaround_ns_alone"].mean()
        num = vpair.loc[v_success, "external_turnaround_ns_combined"].mean()
        victim_slowdown = float(num / denom) if denom > 0 else np.nan

    return {
        "scenario_name": scenario.scenario_name,
        "scenario_family": scenario.scenario_family,
        "mode": scenario.mode,
        "workload_name": workload.workload_name,
        "trial_id": trial_id,
        "victim_phase_ns": phase_ns,
        "probe_count": len(merged),
        "successful_probe_pair_count": len(successful),
        "affected_probe_fraction": float(merged["affected"].mean()),
        "speedup_probe_fraction": float(merged["speedup"].mean()),
        "failure_transition_fraction": float(merged["failure_transition"].mean()),
        "mean_excess_turnaround_ns": float(successful["excess_turnaround_ns"].mean()) if len(successful) else np.nan,
        "mean_abs_turnaround_change_ns": float(abs_change.mean()) if len(abs_change) else np.nan,
        "max_abs_turnaround_change_ns": float(abs_change.max()) if len(abs_change) else np.nan,
        "mean_combined_epr_wait_ns": float(merged["epr_wait_ns"].mean()),
        "direct_generation_contention_fraction": float(merged["direct_generation_contention"].fillna(False).mean()),
        "victim_depleted_pool_arrival_fraction": float(merged["victim_depleted_pool_at_arrival"].fillna(False).mean()),
        "persistent_victim_depletion_fraction": float(merged["persistent_victim_depletion"].fillna(False).mean()),
        "persistent_without_direct_generation_fraction": float(merged["persistent_without_direct_generation"].fillna(False).mean()),
        "pool_miss_fraction": float((merged["pool_level_at_arrival"].fillna(0) == 0).mean()),
        "attacker_combined_failure_fraction": float((~merged["combined_success"].fillna(False)).mean()),
        "victim_alone_failure_fraction": float((~vpair["success_alone"].fillna(False)).mean()) if len(vpair) else 0.0,
        "victim_combined_failure_fraction": float((~vpair["success_combined"].fillna(False)).mean()) if len(vpair) else 0.0,
        "victim_successful_request_slowdown": victim_slowdown,
    }


# =============================================================================
# Aggregate analyses
# =============================================================================


def scenario_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "affected_probe_fraction",
        "speedup_probe_fraction",
        "failure_transition_fraction",
        "mean_excess_turnaround_ns",
        "mean_abs_turnaround_change_ns",
        "max_abs_turnaround_change_ns",
        "mean_combined_epr_wait_ns",
        "direct_generation_contention_fraction",
        "victim_depleted_pool_arrival_fraction",
        "persistent_victim_depletion_fraction",
        "persistent_without_direct_generation_fraction",
        "pool_miss_fraction",
        "attacker_combined_failure_fraction",
        "victim_combined_failure_fraction",
        "victim_successful_request_slowdown",
    ]
    out = trial_df.groupby(["scenario_name", "scenario_family", "mode"], as_index=False)[metrics].mean()
    counts = trial_df.groupby("scenario_name", as_index=False).agg(trial_count=("trial_id", "count"))
    return out.merge(counts, on="scenario_name", how="left")


def workload_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    return trial_df.groupby("workload_name", as_index=False).agg(
        mean_affected_probe_fraction=("affected_probe_fraction", "mean"),
        mean_abs_turnaround_change_ns=("mean_abs_turnaround_change_ns", "mean"),
        mean_persistent_state_fraction=("persistent_without_direct_generation_fraction", "mean"),
        mean_failure_transition_fraction=("failure_transition_fraction", "mean"),
        mean_victim_slowdown=("victim_successful_request_slowdown", "mean"),
    )


def generation_summary(generation_df: pd.DataFrame) -> pd.DataFrame:
    if generation_df.empty:
        return pd.DataFrame()
    return generation_df.groupby(
        ["scenario_name", "workload_name", "run_kind"], as_index=False
    ).agg(
        generation_attempts=("attempt_id", "count"),
        successful_attempts=("success", "sum"),
        mean_queue_wait_ns=("queue_wait_ns", "mean"),
        max_queue_wait_ns=("queue_wait_ns", "max"),
        cross_tenant_blocked_fraction=("cross_tenant_blocked", "mean"),
    ).assign(
        generation_success_rate=lambda x: x["successful_attempts"] / x["generation_attempts"]
    )


def pool_state_summary(pool_events_df: pd.DataFrame, observation_window_ns: float) -> pd.DataFrame:
    if pool_events_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["scenario_name", "workload_name", "trial_id", "run_kind", "pool_key"]
    for key, group in pool_events_df.groupby(keys):
        g = group.sort_values(["time_ns"]).copy()
        capacity = int(g["capacity"].max())
        times = g["time_ns"].to_numpy(dtype=float)
        occ = g["occupancy_after"].to_numpy(dtype=float)
        horizon = max(observation_window_ns, float(times.max()) if len(times) else observation_window_ns)
        area = 0.0
        prev_t = 0.0
        prev_occ = 0.0
        for t, o in zip(times, occ):
            area += prev_occ * max(0.0, t - prev_t)
            prev_t = t
            prev_occ = o
        area += prev_occ * max(0.0, horizon - prev_t)
        mean_occ = area / horizon if horizon > 0 else 0.0
        events = g["event_kind"]
        rows.append(
            {
                "scenario_name": key[0],
                "workload_name": key[1],
                "trial_id": key[2],
                "run_kind": key[3],
                "pool_key": key[4],
                "capacity": capacity,
                "max_occupancy": float(g["occupancy_after"].max()),
                "mean_occupancy": float(mean_occ),
                "unused_capacity_fraction": float(1.0 - mean_occ / capacity) if capacity > 0 else np.nan,
                "pair_store_events": int((events == "pair_stored").sum()),
                "pair_consumption_events": int((events == "pair_consumed").sum()),
                "pair_expiration_events": int((events == "pair_expired").sum()),
                "pool_wait_events": int((events == "request_waits_for_pair").sum()),
                "dropped_full_events": int((events == "generated_pair_dropped_full").sum()),
            }
        )
    return pd.DataFrame(rows)


def persistent_state_summary(request_df: pd.DataFrame) -> pd.DataFrame:
    if request_df.empty:
        return pd.DataFrame()
    a = request_df[(request_df["run_kind"] == "combined") & (request_df["tenant"] == "attacker")].copy()
    if a.empty:
        return pd.DataFrame()
    a["pool_miss"] = a["pool_level_at_arrival"] == 0
    return a.groupby(["scenario_name", "workload_name"], as_index=False).agg(
        attacker_requests=("request_id", "count"),
        mean_epr_wait_ns=("epr_wait_ns", "mean"),
        pool_miss_fraction=("pool_miss", "mean"),
        direct_generation_contention_fraction=("direct_generation_contention", "mean"),
        victim_depleted_pool_fraction=("victim_depleted_pool_at_arrival", "mean"),
        persistent_victim_depletion_fraction=("persistent_victim_depletion", "mean"),
        persistent_without_direct_generation_fraction=("persistent_without_direct_generation", "mean"),
        failure_fraction=("success", lambda x: float((~x.astype(bool)).mean())),
    )


def pair_lifecycle_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    return pair_df.groupby(["scenario_name", "workload_name", "run_kind"], as_index=False).agg(
        pair_count=("pair_id", "count"),
        consumed_pairs=("consumed", "sum"),
        expired_pairs=("expired", "sum"),
        stored_at_end=("stored_at_end", "sum"),
        mean_pair_age_at_consumption_ns=(
            "consumed_ns",
            lambda s: np.nan,
        ),
    ).drop(columns=["mean_pair_age_at_consumption_ns"])


def build_trace_feature_table(blackbox_df: pd.DataFrame, trace_key_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trace_id, group in blackbox_df.groupby("trace_id"):
        g = group.sort_values("probe_index")
        successful = g[g["attacker_only_success"] & g["combined_success"]]
        x = successful["excess_turnaround_ns"].dropna().to_numpy(dtype=float)
        affected = g["affected"].astype(float).to_numpy()
        longest = 0
        current = 0
        for flag in affected:
            if flag > 0.5:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        rows.append(
            {
                "trace_id": trace_id,
                "mean_excess_ns": float(np.mean(x)) if len(x) else 0.0,
                "mean_abs_excess_ns": float(np.mean(np.abs(x))) if len(x) else 0.0,
                "std_excess_ns": float(np.std(x)) if len(x) else 0.0,
                "max_excess_ns": float(np.max(x)) if len(x) else 0.0,
                "min_excess_ns": float(np.min(x)) if len(x) else 0.0,
                "affected_fraction": float(g["affected"].mean()),
                "speedup_fraction": float(g["speedup"].mean()),
                "failure_transition_fraction": float(g["failure_transition"].mean()),
                "longest_affected_run": longest,
            }
        )
    return pd.DataFrame(rows).merge(trace_key_df, on="trace_id", how="left", validate="one_to_one")


def leave_one_trial_out_nearest_centroid(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple timing-only workload classifier; no internal EPR labels are used."""
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    feature_cols = [
        "mean_excess_ns",
        "mean_abs_excess_ns",
        "std_excess_ns",
        "max_excess_ns",
        "min_excess_ns",
        "affected_fraction",
        "speedup_fraction",
        "failure_transition_fraction",
        "longest_affected_run",
    ]
    pred_rows: list[dict[str, object]] = []
    for scenario_name, sf in features.groupby("scenario_name"):
        for test_trial in sorted(sf["trial_id"].unique()):
            train = sf[sf["trial_id"] != test_trial].copy()
            test = sf[sf["trial_id"] == test_trial].copy()
            if train.empty or test.empty:
                continue
            mu = train[feature_cols].mean()
            sigma = train[feature_cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
            centroids = {}
            for label, g in train.groupby("workload_name"):
                centroids[label] = ((g[feature_cols].mean() - mu) / sigma).to_numpy(dtype=float)
            for row in test.itertuples(index=False):
                vec = ((pd.Series({c: getattr(row, c) for c in feature_cols}) - mu) / sigma).to_numpy(dtype=float)
                distances = {label: float(np.linalg.norm(vec - centroid)) for label, centroid in centroids.items()}
                prediction = min(distances, key=distances.get)
                pred_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "trace_id": row.trace_id,
                        "trial_id": row.trial_id,
                        "true_workload": row.workload_name,
                        "predicted_workload": prediction,
                        "correct": prediction == row.workload_name,
                        "nearest_distance": distances[prediction],
                    }
                )
    predictions = pd.DataFrame(pred_rows)
    if predictions.empty:
        return pd.DataFrame(), predictions
    metrics = predictions.groupby("scenario_name", as_index=False).agg(
        sample_count=("correct", "count"),
        accuracy=("correct", "mean"),
        mean_nearest_distance=("nearest_distance", "mean"),
    )
    metrics["chance_accuracy"] = 1.0 / max(1, features["workload_name"].nunique())
    return metrics, predictions


# =============================================================================
# Validation
# =============================================================================


def assertion(group: str, name: str, passed: bool, expected: object, observed: object, details: str = "") -> ValidationAssertion:
    return ValidationAssertion(
        validation_group=group,
        assertion_name=name,
        passed=bool(passed),
        expected=str(expected),
        observed=str(observed),
        details=details,
    )


def validate(
    *,
    scenarios: tuple[EPRScenario, ...],
    request_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    pool_events_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    blackbox_df: pd.DataFrame,
) -> list[ValidationAssertion]:
    out: list[ValidationAssertion] = []

    out.append(assertion(
        "blackbox_boundary",
        "attacker_visible_columns_are_external_only",
        set(blackbox_df.columns).issubset(BLACKBOX_ALLOWED_COLUMNS),
        sorted(BLACKBOX_ALLOWED_COLUMNS),
        sorted(blackbox_df.columns),
    ))

    successful = request_df[request_df["success"]].copy()
    if len(successful):
        completion_err = (
            successful["external_completion_ns"]
            - successful["epr_acquired_ns"]
            - REMOTE_CRITICAL_LATENCY_NS
        ).abs()
        out.append(assertion(
            "pipeline_timing",
            "successful_remote_completion_matches_fixed_post_epr_pipeline",
            bool((completion_err <= 1e-7).all()),
            "absolute error <= 1e-7 ns",
            float(completion_err.max()),
        ))

    out.append(assertion(
        "request_integrity",
        "all_requests_resolved_success_or_failure",
        bool(request_df["success"].notna().all()),
        "all resolved",
        int(request_df["success"].notna().sum()),
    ))

    if not pair_df.empty:
        consumed = pair_df[pair_df["consumed"]].copy()
        valid_lifetime = (
            consumed["consumed_ns"] <= consumed["expires_ns"] + FLOAT_TOLERANCE_NS
        ).all() if len(consumed) else True
        out.append(assertion(
            "pair_lifecycle",
            "no_pair_consumed_after_expiration",
            bool(valid_lifetime),
            True,
            bool(valid_lifetime),
        ))
        out.append(assertion(
            "pair_lifecycle",
            "pair_identifiers_unique",
            bool(~pair_df.duplicated(["run_kind", "scenario_name", "workload_name", "trial_id", "pair_id"]).any()),
            True,
            bool(~pair_df.duplicated(["run_kind", "scenario_name", "workload_name", "trial_id", "pair_id"]).any()),
        ))

    if not pool_events_df.empty:
        capacity_ok = (
            pool_events_df["occupancy_after"] <= pool_events_df["capacity"].clip(lower=0)
        ) | (pool_events_df["capacity"] == 0)
        out.append(assertion(
            "storage_capacity",
            "finite_pool_never_exceeds_capacity",
            bool(capacity_ok.all()),
            True,
            int((~capacity_ok).sum()),
        ))
        out.append(assertion(
            "storage_capacity",
            "pool_occupancy_never_negative",
            bool((pool_events_df["occupancy_after"] >= 0).all()),
            True,
            int((pool_events_df["occupancy_after"] < 0).sum()),
        ))

    # Scenario-level causal controls.
    sc = trial_df.groupby("scenario_name", as_index=True).mean(numeric_only=True)
    if "isolated_on_demand" in sc.index:
        obs = float(sc.loc["isolated_on_demand", "affected_probe_fraction"])
        out.append(assertion(
            "isolation_control",
            "dedicated_on_demand_generators_remove_cross_tenant_timing_change",
            obs <= 1e-12,
            0.0,
            obs,
        ))

    if "reserved_pool1_dedicated_generators" in sc.index:
        obs = float(sc.loc["reserved_pool1_dedicated_generators", "affected_probe_fraction"])
        fail = float(sc.loc["reserved_pool1_dedicated_generators", "failure_transition_fraction"])
        out.append(assertion(
            "isolation_control",
            "reserved_storage_plus_dedicated_generation_removes_cross_tenant_channel",
            obs <= 1e-12 and fail <= 1e-12,
            "0 timing/failure transition",
            {"affected": obs, "failure_transition": fail},
        ))

    if "shared_on_demand_gen1" in sc.index and "shared_on_demand_gen2" in sc.index:
        one = float(sc.loc["shared_on_demand_gen1", "mean_abs_turnaround_change_ns"])
        two = float(sc.loc["shared_on_demand_gen2", "mean_abs_turnaround_change_ns"])
        out.append(assertion(
            "generation_capacity",
            "second_generation_lane_does_not_increase_on_demand_leakage",
            two <= one + 1e-7,
            "gen2 <= gen1",
            {"gen1": one, "gen2": two},
        ))

    if "shared_prefetch_pool2" in sc.index:
        persistent = float(sc.loc["shared_prefetch_pool2", "persistent_without_direct_generation_fraction"])
        out.append(assertion(
            "persistent_state",
            "prefetch_exposes_nonzero_delayed_depletion_state",
            persistent > 0.0,
            "> 0",
            persistent,
            "A later attacker probe should sometimes wait after victim-induced pool depletion without direct generator overlap.",
        ))

    if "shared_prefetch_short_lifetime" in set(pool_events_df.get("scenario_name", pd.Series(dtype=str))):
        expirations = int(
            ((pool_events_df["scenario_name"] == "shared_prefetch_short_lifetime") &
             (pool_events_df["event_kind"] == "pair_expired")).sum()
        )
        out.append(assertion(
            "pair_lifetime",
            "short_lifetime_scenario_exercises_expiration",
            expirations > 0,
            "> 0 expiration events",
            expirations,
        ))

    if not generation_df.empty and "shared_prefetch_unreliable_generation" in set(generation_df["scenario_name"]):
        failures = int(
            ((generation_df["scenario_name"] == "shared_prefetch_unreliable_generation") &
             (~generation_df["success"].astype(bool))).sum()
        )
        out.append(assertion(
            "generation_failure",
            "unreliable_generation_scenario_exercises_failed_attempts",
            failures > 0,
            "> 0 failed attempts",
            failures,
        ))

    if "strict_reserved_pool1_shared_generator" in sc.index:
        failures = float(sc.loc["strict_reserved_pool1_shared_generator", "attacker_combined_failure_fraction"])
        out.append(assertion(
            "reservation",
            "strict_reservation_exercises_admission_failures",
            failures > 0.0,
            "> 0 attacker failure fraction",
            failures,
        ))

    # EPR acquisition is the only stage allowed to have wait in this phase.
    non_epr_wait = stage_df[stage_df["stage_name"] != "epr_pair_acquisition"]["wait_ns"].abs()
    out.append(assertion(
        "causal_isolation",
        "non_epr_pipeline_resources_have_zero_queue_wait",
        bool((non_epr_wait <= FLOAT_TOLERANCE_NS).all()),
        0.0,
        float(non_epr_wait.max()) if len(non_epr_wait) else 0.0,
    ))

    return out


# =============================================================================
# Experiment driver
# =============================================================================


def simulate(
    *,
    scenario: EPRScenario,
    run_kind: str,
    workload_name: str,
    trial_id: int,
    seed: int,
    observation_window_ns: float,
    specs: Iterable[RequestSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return EPRManager(
        scenario=scenario,
        run_kind=run_kind,
        workload_name=workload_name,
        trial_id=trial_id,
        seed=seed,
        observation_window_ns=observation_window_ns,
        specs=specs,
    ).run()


def protocol_definition_table() -> pd.DataFrame:
    rows = [
        {
            "stage_index": 0,
            "stage_name": "epr_pair_acquisition",
            "causal_component": "epr_resource_manager",
            "duration_ns": "dynamic",
            "critical_path": True,
            "external_completion_here": False,
            "postcompletion": False,
            "description": "Acquire one valid EPR pair from on-demand generation or finite storage.",
        }
    ]
    for i, stage in enumerate(REMOTE_STAGE_DURATIONS_NS, start=1):
        rows.append(
            {
                "stage_index": i,
                "stage_name": stage,
                "causal_component": {
                    "epr_bind_and_local_prepare": "tenant_local_endpoint",
                    "bell_measurement": "tenant_local_bell_measurement",
                    "classical_feedforward": "tenant_local_feedforward",
                    "conditional_control": "tenant_local_conditional_control",
                    "reset_recovery": "tenant_local_reset",
                }[stage],
                "duration_ns": REMOTE_STAGE_DURATIONS_NS[stage],
                "critical_path": stage in CRITICAL_REMOTE_STAGES,
                "external_completion_here": stage == "conditional_control",
                "postcompletion": stage in POSTCOMPLETION_REMOTE_STAGES,
                "description": "Tenant-dedicated stage; retained to embed EPR management in the validated remote-operation pipeline.",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window-ns", type=float, default=DEFAULT_OBSERVATION_WINDOW_NS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 2:
        raise ValueError("Use at least two trials for leave-one-trial-out fingerprinting.")
    if args.window_ns <= ATTACKER_FIRST_RELEASE_NS + ATTACKER_PERIOD_NS:
        raise ValueError("Observation window is too short for the probe train.")

    outdir: Path = args.output_dir
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    workloads = build_workloads()
    rng = np.random.default_rng(args.seed)

    pd.DataFrame(asdict(s) for s in scenarios).to_csv(
        outdir / "phase2_05_configuration_table.csv", index=False
    )
    protocol_definition_table().to_csv(
        outdir / "phase2_05_protocol_definition.csv", index=False
    )

    # Paired release phases: every EPR-management scenario sees the same victim
    # arrival schedule for a given workload/trial.
    phases: dict[tuple[str, int], float] = {}
    phase_rows: list[dict[str, object]] = []
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
    pd.DataFrame(phase_rows).to_csv(
        outdir / "phase2_05_trial_phase_schedule.csv", index=False
    )

    request_frames: list[pd.DataFrame] = []
    stage_frames: list[pd.DataFrame] = []
    generation_frames: list[pd.DataFrame] = []
    pool_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    request_event_frames: list[pd.DataFrame] = []
    blackbox_frames: list[pd.DataFrame] = []
    trial_rows: list[dict[str, object]] = []
    trace_key_rows: list[dict[str, object]] = []

    total = len(scenarios) * len(workloads) * args.trials
    done = 0

    for sidx, scenario in enumerate(scenarios):
        for widx, workload in enumerate(workloads):
            for trial_id in range(args.trials):
                phase = phases[(workload.workload_name, trial_id)]
                schedule_rng = np.random.default_rng(
                    args.seed + 100_000 * trial_id + 1_000 * widx
                )
                a_specs = attacker_specs(trial_id, workload.workload_name, args.window_ns)
                v_specs = victim_specs(
                    workload,
                    trial_id,
                    phase,
                    schedule_rng,
                    args.window_ns,
                )

                run_seed = args.seed + 10_000_000 * sidx + 100_000 * trial_id + 1_000 * widx
                a = simulate(
                    scenario=scenario,
                    run_kind="attacker_only",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                    seed=run_seed,
                    observation_window_ns=args.window_ns,
                    specs=a_specs,
                )
                v = simulate(
                    scenario=scenario,
                    run_kind="victim_only",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                    seed=run_seed + 17,
                    observation_window_ns=args.window_ns,
                    specs=v_specs,
                )
                c = simulate(
                    scenario=scenario,
                    run_kind="combined",
                    workload_name=workload.workload_name,
                    trial_id=trial_id,
                    seed=run_seed + 31,
                    observation_window_ns=args.window_ns,
                    specs=a_specs + v_specs,
                )

                for frame_list, idx in [
                    (request_frames, 0),
                    (stage_frames, 1),
                    (generation_frames, 2),
                    (pool_frames, 3),
                    (pair_frames, 4),
                    (request_event_frames, 5),
                ]:
                    for run in (a, v, c):
                        if not run[idx].empty:
                            frame_list.append(run[idx])

                a_req, v_req, c_req = a[0], v[0], c[0]
                trace = paired_attacker_trace(a_req, c_req)
                causal = attacker_causal_table(c_req)
                trace_id = f"{scenario.scenario_name}::{workload.workload_name}::trial{trial_id}"

                bb = trace[[
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
                ]].copy()
                bb.insert(0, "trace_id", trace_id)
                blackbox_frames.append(bb)

                trial_rows.append(
                    summarize_trial(
                        scenario=scenario,
                        workload=workload,
                        trial_id=trial_id,
                        phase_ns=phase,
                        trace=trace,
                        causal=causal,
                        victim_only=v_req,
                        combined=c_req,
                    )
                )
                trace_key_rows.append(
                    {
                        "trace_id": trace_id,
                        "scenario_name": scenario.scenario_name,
                        "scenario_family": scenario.scenario_family,
                        "mode": scenario.mode,
                        "workload_name": workload.workload_name,
                        "trial_id": trial_id,
                        "victim_phase_ns": phase,
                    }
                )

                done += 1
                if done % max(1, total // 10) == 0 or done == total:
                    print(f"  completed {done}/{total} scenario-workload-trial tuples")

    request_df = pd.concat(request_frames, ignore_index=True) if request_frames else pd.DataFrame()
    stage_df = pd.concat(stage_frames, ignore_index=True) if stage_frames else pd.DataFrame()
    generation_df = pd.concat(generation_frames, ignore_index=True) if generation_frames else pd.DataFrame()
    pool_events_df = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()
    pair_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    request_events_df = pd.concat(request_event_frames, ignore_index=True) if request_event_frames else pd.DataFrame()
    blackbox_df = pd.concat(blackbox_frames, ignore_index=True)
    trial_df = pd.DataFrame(trial_rows)
    trace_key_df = pd.DataFrame(trace_key_rows)

    # Raw evaluator-only records.
    request_df.to_csv(rawdir / "phase2_05_request_records.csv.gz", index=False, compression=GZIP_COMPRESSION)
    stage_df.to_csv(rawdir / "phase2_05_stage_records.csv.gz", index=False, compression=GZIP_COMPRESSION)
    generation_df.to_csv(rawdir / "phase2_05_generation_attempts.csv.gz", index=False, compression=GZIP_COMPRESSION)
    pool_events_df.to_csv(rawdir / "phase2_05_pool_events.csv.gz", index=False, compression=GZIP_COMPRESSION)
    pair_df.to_csv(rawdir / "phase2_05_pair_lifecycle.csv.gz", index=False, compression=GZIP_COMPRESSION)
    request_events_df.to_csv(rawdir / "phase2_05_request_events.csv.gz", index=False, compression=GZIP_COMPRESSION)

    # Attacker-visible timing and evaluator key are deliberately separated.
    blackbox_df.to_csv(outdir / "phase2_05_blackbox_trace_summary.csv", index=False)
    trace_key_df.to_csv(outdir / "phase2_05_trace_key.csv", index=False)
    trial_df.to_csv(outdir / "phase2_05_trial_summary.csv", index=False)

    sc_summary = scenario_summary(trial_df)
    wl_summary = workload_summary(trial_df)
    gen_summary = generation_summary(generation_df)
    pool_summary = pool_state_summary(pool_events_df, args.window_ns)
    persistent = persistent_state_summary(request_df)
    pair_summary = pair_lifecycle_summary(pair_df)

    sc_summary.to_csv(outdir / "phase2_05_scenario_summary.csv", index=False)
    wl_summary.to_csv(outdir / "phase2_05_workload_summary.csv", index=False)
    gen_summary.to_csv(outdir / "phase2_05_generation_summary.csv", index=False)
    pool_summary.to_csv(outdir / "phase2_05_pool_state_summary.csv", index=False)
    persistent.to_csv(outdir / "phase2_05_persistent_state_summary.csv", index=False)
    pair_summary.to_csv(outdir / "phase2_05_pair_lifecycle_summary.csv", index=False)

    features = build_trace_feature_table(blackbox_df, trace_key_df)
    features.to_csv(outdir / "phase2_05_trace_features.csv", index=False)
    fp_metrics, fp_predictions = leave_one_trial_out_nearest_centroid(features)
    fp_metrics.to_csv(outdir / "phase2_05_workload_fingerprint_metrics.csv", index=False)
    fp_predictions.to_csv(outdir / "phase2_05_workload_fingerprint_predictions.csv", index=False)

    validations = validate(
        scenarios=scenarios,
        request_df=request_df,
        stage_df=stage_df,
        generation_df=generation_df,
        pool_events_df=pool_events_df,
        pair_df=pair_df,
        trial_df=trial_df,
        blackbox_df=blackbox_df,
    )
    validation_df = pd.DataFrame(asdict(v) for v in validations)
    validation_df.to_csv(outdir / "phase2_05_validation_assertions.csv", index=False)
    validation_summary = validation_df.groupby("validation_group", as_index=False).agg(
        assertion_count=("assertion_name", "count"),
        passed_count=("passed", "sum"),
    )
    validation_summary["failed_count"] = validation_summary["assertion_count"] - validation_summary["passed_count"]
    validation_summary["pass_rate"] = validation_summary["passed_count"] / validation_summary["assertion_count"]
    validation_summary.to_csv(outdir / "phase2_05_validation_summary.csv", index=False)

    manifest = {
        "experiment": "Phase 2.5 — EPR Generation, Storage, and Prefetch",
        "output_directory": str(outdir),
        "trial_count_per_workload_configuration": args.trials,
        "scenario_count": len(scenarios),
        "workload_count": len(workloads),
        "scenario_workload_trial_tuples": total,
        "probe_period_ns": ATTACKER_PERIOD_NS,
        "observation_window_ns": args.window_ns,
        "remote_critical_latency_after_epr_ns": REMOTE_CRITICAL_LATENCY_NS,
        "remote_cleanup_latency_after_epr_ns": REMOTE_CLEANUP_LATENCY_NS,
        "conceptual_modes": ["on_demand", "prefetch", "buffered", "reserved"],
        "validation_assertion_count": int(len(validation_df)),
        "passed_assertions": int(validation_df["passed"].sum()),
        "failed_assertions": int((~validation_df["passed"]).sum()),
        "all_validations_passed": bool(validation_df["passed"].all()),
        "blackbox_columns": list(blackbox_df.columns),
    }
    with open(outdir / "phase2_05_run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nPhase 2.5 — EPR Generation, Storage, and Prefetch")
    print("=" * 66)
    print(validation_summary.to_string(index=False))
    print("\nRemote critical latency after EPR acquisition:", REMOTE_CRITICAL_LATENCY_NS, "ns")
    print("Remote cleanup latency after EPR acquisition:", REMOTE_CLEANUP_LATENCY_NS, "ns")
    print("\nResults saved to:", outdir)

    if manifest["all_validations_passed"]:
        print("\nAll Phase 2.5 causal/resource validations passed.")
    else:
        failed = validation_df[~validation_df["passed"]]
        print("\nFAILED VALIDATIONS:")
        print(failed[["validation_group", "assertion_name", "expected", "observed"]].to_string(index=False))
        if FAIL_ON_VALIDATION_ERROR:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
