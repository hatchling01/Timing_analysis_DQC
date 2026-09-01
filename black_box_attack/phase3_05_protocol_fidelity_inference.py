#!/usr/bin/env python3
"""
Phase 3.5 — Protocol and Fidelity-Demand Inference
==================================================

Research question
-----------------
Can attacker-only black-box timing reveal HOW a hidden victim implements remote
operations, or what reliability/fidelity-oriented service policy it requests?

Progression
-----------
3.1: how much communication?
3.2: when does it occur?
3.3: where does it occur?
3.4: which intermodule edges / graph are active?
3.5: how is the communication implemented, and what service policy is used?

Phase 3.5 is an inference / attack-characterization experiment, not a defense
experiment.

Primary tasks
-------------
A. protocol_inference
   direct_transfer, on_demand_epr, prefetched_epr, telegate, teledata

B. distillation_depth_inference
   distill_0, distill_1, distill_2
   plus ordinal distillation-depth estimation

C. retry_policy_inference
   retry_1, retry_2, retry_4
   plus actual retry-count estimation

D. service_class_inference
   low_latency, high_fidelity

Negative controls
-----------------
protocol_label_only_control and service_label_only_control assign different
hidden evaluator labels to identical runtime conditions.  They should remain at
chance and therefore test that the ML pipeline is not learning labels that have
no timing semantics.

Architecture semantics
----------------------
The experiment imports Phase-2.7 timing constants and preserves its two core
remote-operation realizations:

* Direct coherent transfer:
  endpoint prepare 30 ns -> switch setup 15 ns -> link 80 ns -> receiver 25 ns
  -> external completion -> reset 120 ns.  Endpoint is held through reset;
  switch path is held through link transfer.

* Entanglement-assisted operation:
  EPR prerequisite -> scoped endpoint access -> readout -> feedforward ->
  correction -> external completion -> reset.

Phase 3.5 adds controlled service extensions:

* on-demand EPR: one fresh raw pair per logical request;
* prefetched EPR: warm finite two-pair pool with asynchronous refill;
* TeleGate / TeleData: two distinct post-EPR resource-lifetime profiles;
* r-round distillation: 2**r raw pairs followed by r exclusive 90 ns
  distillation rounds;
* retry-limited modes: at most 1/2/4 raw-pair attempts per required pair;
* low-latency service: warm pair, no distillation, short post-EPR path;
* high-fidelity service: four raw pairs, two distillation rounds, larger retry
  allowance, and longer post-EPR path.

These are controlled architectural resource models for inference experiments.
They are not claims about measured commercial-hardware fidelity or a specific
vendor protocol implementation.

Link-success generalization
---------------------------
The same hidden schedules are crossed with several raw-pair generation success
probabilities (default 0.65, 0.80, 0.95).  The script reports ordinary grouped
held-out metrics, workload-held-out generalization, and link-probability-held-
out generalization.

Attacker boundary
-----------------
The attacker observes ONLY its own:
  probe index/path, release/completion/turnaround, differential timing,
  success/failure transition.

Hidden victim schedule, protocol/service label, resource waits, EPR pool state,
distillation depth, retries, and link-success outcomes are evaluator-only.

Features
--------
Delay-tail statistics, burst/run repetition, long-gap frequency, consecutive
delayed probes, inferred busy-period duration, lag/spectral structure, and
cross-correlation across three interleaved attacker-chosen logical probe paths.

Outputs
-------
Protocol accuracy, fidelity/service-class accuracy, distillation-depth
classification + MAE, retry-policy classification + retry-count MAE, probe-
budget curves, workload generalization, link-success generalization, confusion
matrices, evaluator mechanism logs, validations, and manifest.

Run
---
python phase3_05_protocol_fidelity_inference.py

Smoke
-----
python phase3_05_protocol_fidelity_inference.py \
  --base-schedules 6 --repeats-per-schedule 1 \
  --observation-window-ns 6000 --link-success-probabilities 0.7,0.9 \
  --rf-trees 50 --output-dir /tmp/p35_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SEED = 3501
DEFAULT_BASE_SCHEDULES = 18
DEFAULT_REPEATS_PER_SCHEDULE = 2
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_LINK_SUCCESS_PROBABILITIES = "0.65,0.80,0.95"
DEFAULT_TEST_SIZE = 0.33
DEFAULT_RF_TREES = 500
DEFAULT_PROBE_BUDGETS = "12,24,36"
DEFAULT_OUTPUT_DIR = Path("blackbox_window_results") / "phase3" / "phase3.5"
EPS = 1e-12
AFFECTED_THRESHOLD_NS = 1e-9
PROBE_PATHS = ("probe_path_0", "probe_path_1", "probe_path_2")

TASK_PROTOCOL = "protocol_inference"
TASK_PROTOCOL_CONTROL = "protocol_label_only_control"
TASK_DISTILL = "distillation_depth_inference"
TASK_RETRY = "retry_policy_inference"
TASK_SERVICE = "service_class_inference"
TASK_SERVICE_CONTROL = "service_label_only_control"

ATTACKER_VISIBLE_COLUMNS = [
    "trace_id", "probe_index", "probe_path", "release_ns",
    "attacker_only_success", "combined_success",
    "attacker_only_completion_ns", "combined_completion_ns",
    "attacker_only_turnaround_ns", "combined_turnaround_ns",
    "excess_turnaround_ns", "delayed", "speedup", "failure_transition",
]


# =============================================================================
# Phase-2.7 constants
# =============================================================================

def load_phase2_07_module():
    candidates = [
        Path(__file__).resolve().parent / "phase2_07_remote_protocol_comparison.py",
        Path.cwd() / "phase2_07_remote_protocol_comparison.py",
        Path(__file__).resolve().parent.parent / "phase2_07_remote_protocol_comparison.py",
    ]
    source = next((x for x in candidates if x.exists()), None)
    if source is None:
        raise FileNotFoundError("Could not locate phase2_07_remote_protocol_comparison.py")
    name = "phase2_07_remote_protocol_comparison"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, source


def stable_seed(*parts: Any, modulus: int = 2**32 - 1) -> int:
    token = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % modulus


def parse_probability_list(text: str) -> list[float]:
    vals = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or any(x < 0.05 or x > 1.0 for x in vals):
        raise ValueError("link success probabilities must lie in [0.05, 1.0]")
    return vals


# =============================================================================
# Experimental data model
# =============================================================================

@dataclass(frozen=True)
class BaseSchedule:
    base_schedule_id: str
    base_index: int
    schedule_profile: str
    base_phase_ns: float
    interval_ns: float
    burst_period_ns: float
    burst_size: int
    burst_spacing_ns: float

@dataclass(frozen=True)
class PostProfile:
    name: str
    endpoint_access_ns: float
    readout_ns: float
    feedforward_ns: float
    correction_ns: float
    reset_ns: float

@dataclass(frozen=True)
class Condition:
    condition_id: str
    task_name: str
    evaluator_label: str
    runtime_mode: str
    post_profile: str
    distillation_depth: int
    retry_limit: int
    uses_prefetch: bool
    description: str

@dataclass
class RequestState:
    request_id: str
    tenant: str
    release_ns: float
    request_index: int
    probe_path: str = ""
    success: bool = True
    failure_reason: str = ""
    completion_ns: float = math.nan
    cleanup_ns: float = math.nan
    resource_wait_ns: dict[str, float] = field(default_factory=lambda: defaultdict(float))

@dataclass
class Waiter:
    actor_id: str
    tenant: str
    stage: str
    duration_ns: float
    release_on_done: bool

@dataclass
class Pair:
    pair_id: str
    expires_ns: float


def clone_request_state(x: RequestState) -> RequestState:
    """
    Clone a request template with fresh mutable per-run state.

    Do not use dataclasses.asdict() here: RequestState.resource_wait_ns is a
    defaultdict, and Python 3.11's recursive asdict reconstruction attempts to
    instantiate defaultdict from an iterator, raising:
        TypeError: first argument must be callable or None
    """
    return RequestState(
        request_id=x.request_id,
        tenant=x.tenant,
        release_ns=float(x.release_ns),
        request_index=int(x.request_index),
        probe_path=str(x.probe_path),
        success=bool(x.success),
        failure_reason=str(x.failure_reason),
        completion_ns=float(x.completion_ns),
        cleanup_ns=float(x.cleanup_ns),
        resource_wait_ns=defaultdict(float),
    )


# =============================================================================
# Base victim schedules
# =============================================================================

def build_base_schedules(seed: int, count: int) -> list[BaseSchedule]:
    if count < 6:
        raise ValueError("--base-schedules must be >= 6")
    profiles = ("sparse_periodic", "dense_periodic", "synchronization_bursty")
    rows = []
    for i in range(count):
        profile = profiles[i % 3]
        rng = np.random.default_rng(stable_seed(seed, "base", i))
        phase = float(rng.uniform(100.0, 420.0))
        if profile == "sparse_periodic":
            interval, bp, bs, bsp = float(rng.uniform(900, 1200)), 0.0, 1, 0.0
        elif profile == "dense_periodic":
            interval, bp, bs, bsp = float(rng.uniform(430, 560)), 0.0, 1, 0.0
        else:
            interval = 0.0
            bp = float(rng.uniform(1400, 1850))
            bs = int(rng.integers(3, 7))
            bsp = float(rng.uniform(55, 95))
        rows.append(BaseSchedule(
            f"base_{i:04d}", i, profile, phase, interval, bp, bs, bsp
        ))
    return rows


def schedule_releases(s: BaseSchedule, repeat_id: int, seed: int, window_ns: float) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, s.base_schedule_id, repeat_id, "release"))
    phase = s.base_phase_ns + float(rng.uniform(-12, 12))
    if s.schedule_profile in {"sparse_periodic", "dense_periodic"}:
        rel = np.arange(phase, window_ns, s.interval_ns, dtype=float)
    else:
        vals = []
        base = phase
        while base < window_ns:
            vals.extend(base + j * s.burst_spacing_ns for j in range(s.burst_size))
            base += s.burst_period_ns
        rel = np.asarray(vals, dtype=float)
    rel = rel[(rel >= 0.0) & (rel < window_ns)]
    if len(rel):
        rel = np.clip(rel + rng.uniform(-4, 4, len(rel)), 0.0, window_ns - 1e-6)
    return np.sort(rel)


# =============================================================================
# Conditions
# =============================================================================

def build_post_profiles(p27) -> dict[str, PostProfile]:
    return {
        "baseline": PostProfile("baseline", p27.ENT_ENDPOINT_ACCESS_NS, p27.ENT_READOUT_NS, p27.ENT_FEEDFORWARD_NS, p27.ENT_CORRECTION_NS, p27.RESET_NS),
        "prefetched": PostProfile("prefetched", 20.0, 55.0, 30.0, 30.0, 110.0),
        "telegate": PostProfile("telegate", 20.0, 70.0, 40.0, 20.0, 120.0),
        "teledata": PostProfile("teledata", 30.0, 85.0, 50.0, 45.0, 135.0),
        "low_latency": PostProfile("low_latency", 15.0, 50.0, 25.0, 15.0, 80.0),
        "high_fidelity": PostProfile("high_fidelity", 25.0, 95.0, 55.0, 30.0, 160.0),
    }


def build_conditions() -> list[Condition]:
    c = [
        Condition("protocol_direct", TASK_PROTOCOL, "direct_transfer", "direct", "baseline", 0, 1, False, "Direct coherent transfer."),
        Condition("protocol_on_demand", TASK_PROTOCOL, "on_demand_epr", "custom_epr", "baseline", 0, 4, False, "Fresh raw pair for every request."),
        Condition("protocol_prefetch", TASK_PROTOCOL, "prefetched_epr", "prefetch", "prefetched", 0, 4, True, "Warm finite EPR pool."),
        Condition("protocol_telegate", TASK_PROTOCOL, "telegate", "prefetch", "telegate", 0, 4, True, "Gate-teleportation-like service."),
        Condition("protocol_teledata", TASK_PROTOCOL, "teledata", "prefetch", "teledata", 0, 4, True, "State-transfer-like service."),
        Condition("distill_0", TASK_DISTILL, "distill_0", "custom_epr", "baseline", 0, 6, False, "One raw pair, no distillation."),
        Condition("distill_1", TASK_DISTILL, "distill_1", "custom_epr", "baseline", 1, 6, False, "Two raw pairs, one round."),
        Condition("distill_2", TASK_DISTILL, "distill_2", "custom_epr", "baseline", 2, 6, False, "Four raw pairs, two rounds."),
        Condition("retry_1", TASK_RETRY, "retry_1", "custom_epr", "baseline", 0, 1, False, "One attempt per raw pair."),
        Condition("retry_2", TASK_RETRY, "retry_2", "custom_epr", "baseline", 0, 2, False, "Two attempts per raw pair."),
        Condition("retry_4", TASK_RETRY, "retry_4", "custom_epr", "baseline", 0, 4, False, "Four attempts per raw pair."),
        Condition("service_low_latency", TASK_SERVICE, "low_latency", "prefetch", "low_latency", 0, 1, True, "Latency-oriented service."),
        Condition("service_high_fidelity", TASK_SERVICE, "high_fidelity", "custom_epr", "high_fidelity", 2, 6, False, "Fidelity/reliability-oriented service."),
    ]
    for label in ("direct_transfer", "on_demand_epr", "prefetched_epr", "telegate", "teledata"):
        c.append(Condition(f"protocol_control_{label}", TASK_PROTOCOL_CONTROL, label, "prefetch", "telegate", 0, 4, True, "Label-only protocol control."))
    for label in ("low_latency", "high_fidelity"):
        c.append(Condition(f"service_control_{label}", TASK_SERVICE_CONTROL, label, "prefetch", "telegate", 0, 4, True, "Label-only service control."))
    return c


# =============================================================================
# Causal resource simulator
# =============================================================================

class Resource:
    def __init__(self, name: str):
        self.name = name
        self.owner: str | None = None
        self.queue: deque[Waiter] = deque()
        self.start_ns: float = math.nan

    def request(self, sim: "Simulator", waiter: Waiter, now: float) -> None:
        if self.owner is None:
            self._grant(sim, waiter, now)
        else:
            self.queue.append(waiter)
            sim.wait_arrival[(waiter.actor_id, self.name, waiter.stage)] = float(now)

    def _grant(self, sim: "Simulator", waiter: Waiter, now: float) -> None:
        self.owner = waiter.actor_id
        self.start_ns = float(now)
        arrival = sim.wait_arrival.pop((waiter.actor_id, self.name, waiter.stage), now)
        wait = max(0.0, float(now) - float(arrival))
        if waiter.actor_id in sim.requests:
            sim.requests[waiter.actor_id].resource_wait_ns[self.name] += wait
        sim.wait_rows.append({
            "run_kind": sim.run_kind, "resource_name": self.name,
            "actor_id": waiter.actor_id, "tenant": waiter.tenant,
            "stage": waiter.stage, "arrival_ns": float(arrival),
            "grant_ns": float(now), "wait_ns": wait,
        })
        sim.schedule(now + waiter.duration_ns, "resource_stage_done", {
            "resource_name": self.name, "actor_id": waiter.actor_id,
            "tenant": waiter.tenant, "stage": waiter.stage,
            "release_on_done": waiter.release_on_done,
            "duration_ns": waiter.duration_ns,
        })

    def release(self, sim: "Simulator", actor_id: str, now: float) -> None:
        if self.owner != actor_id:
            return
        sim.interval_rows.append({
            "run_kind": sim.run_kind, "resource_name": self.name,
            "actor_id": actor_id, "start_ns": self.start_ns,
            "end_ns": float(now), "duration_ns": float(now) - self.start_ns,
        })
        self.owner = None
        self.start_ns = math.nan
        if self.queue:
            self._grant(sim, self.queue.popleft(), now)


class Simulator:
    def __init__(self, p27, condition: Condition, profile: PostProfile,
                 link_p: float, seed: int, schedule_id: str,
                 repeat_id: int, run_kind: str):
        self.p27 = p27
        self.condition = condition
        self.profile = profile
        self.link_p = float(link_p)
        self.seed = int(seed)
        self.schedule_id = schedule_id
        self.repeat_id = int(repeat_id)
        self.run_kind = run_kind
        self.events: list[tuple[float, int, str, dict[str, Any]]] = []
        self.seq = 0
        self.requests: dict[str, RequestState] = {}
        self.resources: dict[str, Resource] = {}
        self.wait_arrival: dict[tuple[str, str, str], float] = {}
        self.wait_rows: list[dict[str, Any]] = []
        self.interval_rows: list[dict[str, Any]] = []
        self.stage_rows: list[dict[str, Any]] = []
        self.retry_rows: list[dict[str, Any]] = []
        self.distill_rows: list[dict[str, Any]] = []
        self.pool_events: list[dict[str, Any]] = []
        self.pool: deque[Pair] = deque()
        self.pool_waiters: deque[str] = deque()
        self.refill_inflight = 0
        self.refill_counter = 0
        self.pair_counter = 0
        self.custom: dict[str, dict[str, Any]] = {}
        self.background_tenant: dict[str, str] = {}

    def resource(self, name: str) -> Resource:
        if name not in self.resources:
            self.resources[name] = Resource(name)
        return self.resources[name]

    def schedule(self, t: float, event: str, payload: dict[str, Any]) -> None:
        self.seq += 1
        heapq.heappush(self.events, (float(t), self.seq, event, payload))

    def request_resource(self, actor_id: str, tenant: str, name: str,
                         stage: str, now: float, duration: float,
                         release_on_done: bool = True) -> None:
        self.resource(name).request(self, Waiter(actor_id, tenant, stage, float(duration), release_on_done), now)

    def add_request(self, r: RequestState) -> None:
        self.requests[r.request_id] = r
        self.schedule(r.release_ns, "request_release", {"request_id": r.request_id})

    def initialize_prefetch(self) -> None:
        if not self.condition.uses_prefetch:
            return
        for _ in range(2):
            self.pair_counter += 1
            self.pool.append(Pair(f"warm_{self.pair_counter}", float(self.p27.EPR_PAIR_LIFETIME_NS)))
        self.pool_events.append({"run_kind": self.run_kind, "time_ns": 0.0, "event": "warm_prefetch", "pool_level": len(self.pool)})

    def prune_pool(self, now: float) -> None:
        kept = deque()
        expired = 0
        while self.pool:
            pair = self.pool.popleft()
            if pair.expires_ns <= now:
                expired += 1
            else:
                kept.append(pair)
        self.pool = kept
        if expired:
            self.pool_events.append({"run_kind": self.run_kind, "time_ns": now, "event": "pair_expired", "pool_level": len(self.pool), "count": expired})

    def ensure_refill(self, now: float) -> None:
        if not self.condition.uses_prefetch:
            return
        while len(self.pool) + self.refill_inflight < 2:
            self.refill_counter += 1
            actor = f"refill::{self.refill_counter}"
            self.background_tenant[actor] = "victim"
            self.refill_inflight += 1
            self.request_resource(actor, "victim", "epr_generator", "refill_generator", now, float(self.p27.EPR_GENERATOR_SETUP_NS), True)

    def acquire_prefetch(self, request_id: str, now: float) -> None:
        self.prune_pool(now)
        if self.pool:
            pair = self.pool.popleft()
            self.pool_events.append({"run_kind": self.run_kind, "time_ns": now, "event": "pair_consumed", "request_id": request_id, "pair_id": pair.pair_id, "pool_level": len(self.pool)})
            self.ensure_refill(now)
            self.start_post_epr(request_id, now)
        else:
            self.pool_waiters.append(request_id)
            self.pool_events.append({"run_kind": self.run_kind, "time_ns": now, "event": "pool_miss", "request_id": request_id, "pool_level": 0})
            self.ensure_refill(now)

    def generation_success(self, token: str, attempt: int) -> bool:
        # Label-only controls must share the exact same random realization.
        # Primary conditions retain condition-specific deterministic streams.
        if self.condition.task_name == TASK_PROTOCOL_CONTROL:
            semantic_key = TASK_PROTOCOL_CONTROL
        elif self.condition.task_name == TASK_SERVICE_CONTROL:
            semantic_key = TASK_SERVICE_CONTROL
        else:
            semantic_key = self.condition.condition_id
        rng = np.random.default_rng(stable_seed(self.seed, semantic_key, self.schedule_id, self.repeat_id, self.link_p, token, attempt))
        return bool(rng.random() < self.link_p)

    def start_custom_epr(self, request_id: str, now: float) -> None:
        needed = 2 ** int(max(0, self.condition.distillation_depth))
        self.custom[request_id] = {"needed": needed, "ready": 0, "pair_attempt": 0, "total_attempts": 0, "failed_attempts": 0, "distill_round": 0}
        self.start_custom_attempt(request_id, now)

    def start_custom_attempt(self, request_id: str, now: float) -> None:
        st = self.custom[request_id]
        st["pair_attempt"] += 1
        st["total_attempts"] += 1
        self.request_resource(request_id, "victim", "epr_generator", "custom_generator", now, float(self.p27.EPR_GENERATOR_SETUP_NS), True)

    def start_post_epr(self, request_id: str, now: float) -> None:
        self.request_resource(request_id, "victim", "endpoint", "ent_endpoint", now, self.profile.endpoint_access_ns, True)

    def fail_victim_request(self, request_id: str, now: float) -> None:
        r = self.requests[request_id]
        r.success = False
        r.failure_reason = "retry_limit_exhausted"
        r.completion_ns = float(now)
        r.cleanup_ns = float(now)

    def run(self, request_rows: list[RequestState]):
        for r in request_rows:
            self.add_request(r)
        self.initialize_prefetch()
        while self.events:
            now, _, event, payload = heapq.heappop(self.events)
            self.handle_event(now, event, payload)
        req = pd.DataFrame([{
            "request_id": r.request_id, "tenant": r.tenant,
            "request_index": r.request_index, "release_ns": r.release_ns,
            "success": r.success, "failure_reason": r.failure_reason,
            "external_completion_ns": r.completion_ns,
            "cleanup_completion_ns": r.cleanup_ns,
            "turnaround_ns": r.completion_ns - r.release_ns,
        } for r in self.requests.values()])
        return (
            req, pd.DataFrame(self.wait_rows), pd.DataFrame(self.interval_rows),
            pd.DataFrame(self.stage_rows), pd.DataFrame(self.retry_rows),
            pd.DataFrame(self.distill_rows), pd.DataFrame(self.pool_events),
        )

    def log_stage(self, request_id: str, stage: str, start: float, end: float) -> None:
        self.stage_rows.append({"run_kind": self.run_kind, "request_id": request_id, "tenant": self.requests[request_id].tenant if request_id in self.requests else self.background_tenant.get(request_id, "system"), "stage": stage, "start_ns": start, "end_ns": end, "duration_ns": end-start})

    def handle_event(self, now: float, event: str, payload: dict[str, Any]) -> None:
        if event == "request_release":
            rid = payload["request_id"]
            r = self.requests[rid]
            if r.tenant == "attacker" or self.condition.runtime_mode == "direct":
                self.request_resource(rid, r.tenant, "endpoint", "direct_endpoint_prepare", now, float(self.p27.DIRECT_ENDPOINT_PREP_NS), False)
            elif self.condition.runtime_mode == "prefetch":
                self.acquire_prefetch(rid, now)
            elif self.condition.runtime_mode == "custom_epr":
                self.start_custom_epr(rid, now)
            else:
                raise RuntimeError(self.condition.runtime_mode)
            return

        if event == "resource_stage_done":
            name = payload["resource_name"]
            actor = payload["actor_id"]
            stage = payload["stage"]
            duration = float(payload["duration_ns"])
            if payload["release_on_done"]:
                self.resource(name).release(self, actor, now)
            self.handle_stage_done(now, actor, stage, duration)
            return
        raise RuntimeError(event)

    def handle_stage_done(self, now: float, actor: str, stage: str, duration: float) -> None:
        # Background prefetched refill.
        if stage == "refill_generator":
            self.request_resource(actor, "victim", "quantum_link", "refill_link", now, float(self.p27.EPR_LINK_GENERATION_NS), True)
            return
        if stage == "refill_link":
            success = self.generation_success(actor, 1)
            self.retry_rows.append({"run_kind": self.run_kind, "tenant": "victim", "request_id": "", "attempt_kind": "prefetch_refill", "success": success, "time_ns": now})
            self.refill_inflight = max(0, self.refill_inflight - 1)
            if success:
                self.pair_counter += 1
                pair = Pair(f"pair_{self.pair_counter}", now + float(self.p27.EPR_PAIR_LIFETIME_NS))
                if self.pool_waiters:
                    rid = self.pool_waiters.popleft()
                    self.pool_events.append({"run_kind": self.run_kind, "time_ns": now, "event": "refill_served_waiter", "request_id": rid, "pair_id": pair.pair_id, "pool_level": len(self.pool)})
                    self.start_post_epr(rid, now)
                else:
                    self.pool.append(pair)
                    self.pool_events.append({"run_kind": self.run_kind, "time_ns": now, "event": "pair_stored", "pair_id": pair.pair_id, "pool_level": len(self.pool)})
            self.ensure_refill(now)
            return

        # Direct coherent path.
        if stage == "direct_endpoint_prepare":
            self.log_stage(actor, "endpoint_prepare", now-duration, now)
            self.request_resource(actor, self.requests[actor].tenant, "switch_path", "direct_switch_setup", now, float(self.p27.DIRECT_SWITCH_SETUP_NS), False)
            return
        if stage == "direct_switch_setup":
            self.log_stage(actor, "switch_setup", now-duration, now)
            self.request_resource(actor, self.requests[actor].tenant, "quantum_link", "direct_link", now, float(self.p27.DIRECT_LINK_TRANSFER_NS), True)
            return
        if stage == "direct_link":
            self.log_stage(actor, "synchronous_quantum_link_transfer", now-duration, now)
            self.resource("switch_path").release(self, actor, now)
            self.schedule(now + float(self.p27.DIRECT_RECEIVER_GATE_NS), "resource_stage_done", {"resource_name": "local_receiver", "actor_id": actor, "tenant": self.requests[actor].tenant, "stage": "direct_receiver", "release_on_done": False, "duration_ns": float(self.p27.DIRECT_RECEIVER_GATE_NS)})
            return
        if stage == "direct_receiver":
            self.log_stage(actor, "receiver_side_gate", now-duration, now)
            self.requests[actor].completion_ns = now
            self.request_resource(actor, self.requests[actor].tenant, "reset", "direct_reset", now, float(self.p27.RESET_NS), True)
            return
        if stage == "direct_reset":
            self.log_stage(actor, "postcompletion_reset", now-duration, now)
            self.resource("endpoint").release(self, actor, now)
            self.requests[actor].cleanup_ns = now
            return

        # Custom raw-pair generation.
        if stage == "custom_generator":
            self.request_resource(actor, "victim", "quantum_link", "custom_link", now, float(self.p27.EPR_LINK_GENERATION_NS), True)
            return
        if stage == "custom_link":
            st = self.custom[actor]
            success = self.generation_success(actor, st["total_attempts"])
            self.retry_rows.append({"run_kind": self.run_kind, "tenant": "victim", "request_id": actor, "attempt_kind": "on_demand_raw_pair", "attempt_index_for_pair": st["pair_attempt"], "success": success, "time_ns": now})
            if success:
                st["ready"] += 1
                st["pair_attempt"] = 0
                if st["ready"] < st["needed"]:
                    self.start_custom_attempt(actor, now)
                elif self.condition.distillation_depth > 0:
                    self.request_resource(actor, "victim", "distillation", "distill_round", now, 90.0, True)
                else:
                    self.start_post_epr(actor, now)
            else:
                st["failed_attempts"] += 1
                if st["pair_attempt"] >= self.condition.retry_limit:
                    self.fail_victim_request(actor, now)
                else:
                    self.start_custom_attempt(actor, now)
            return
        if stage == "distill_round":
            st = self.custom[actor]
            st["distill_round"] += 1
            self.distill_rows.append({"run_kind": self.run_kind, "request_id": actor, "round_index": st["distill_round"], "completion_ns": now})
            if st["distill_round"] < self.condition.distillation_depth:
                self.request_resource(actor, "victim", "distillation", "distill_round", now, 90.0, True)
            else:
                self.start_post_epr(actor, now)
            return

        # Entanglement-assisted post-EPR path.
        if stage == "ent_endpoint":
            self.log_stage(actor, "stored_epr_endpoint_access", now-duration, now)
            self.request_resource(actor, "victim", "readout", "ent_readout", now, self.profile.readout_ns, True)
            return
        if stage == "ent_readout":
            self.log_stage(actor, "bell_measurement_readout", now-duration, now)
            self.request_resource(actor, "victim", "feedforward", "ent_feedforward", now, self.profile.feedforward_ns, True)
            return
        if stage == "ent_feedforward":
            self.log_stage(actor, "classical_feedforward", now-duration, now)
            self.schedule(now + self.profile.correction_ns, "resource_stage_done", {"resource_name": "local_correction", "actor_id": actor, "tenant": "victim", "stage": "ent_correction", "release_on_done": False, "duration_ns": self.profile.correction_ns})
            return
        if stage == "ent_correction":
            self.log_stage(actor, "receiver_correction", now-duration, now)
            self.requests[actor].completion_ns = now
            self.request_resource(actor, "victim", "reset", "ent_reset", now, self.profile.reset_ns, True)
            return
        if stage == "ent_reset":
            self.log_stage(actor, "postcompletion_reset", now-duration, now)
            self.requests[actor].cleanup_ns = now
            return
        raise RuntimeError(stage)


# =============================================================================
# Request generation, black-box pairing, features
# =============================================================================

def make_attacker_requests(p27, repeat_id: int, window_ns: float) -> list[RequestState]:
    rel = np.arange(float(p27.ATTACKER_FIRST_RELEASE_NS), window_ns, float(p27.ATTACKER_PERIOD_NS))
    rows = []
    for i, t in enumerate(rel):
        rows.append(RequestState(
            request_id=f"attacker::{repeat_id:02d}::{i}", tenant="attacker",
            release_ns=float(t), request_index=i,
            probe_path=PROBE_PATHS[i % len(PROBE_PATHS)]
        ))
    return rows


def make_victim_requests(releases: np.ndarray, schedule_id: str, repeat_id: int) -> list[RequestState]:
    return [RequestState(
        request_id=f"victim::{schedule_id}::{repeat_id:02d}::{i}", tenant="victim",
        release_ns=float(t), request_index=i
    ) for i, t in enumerate(releases)]


def pair_attacker_trace(attacker_only: pd.DataFrame, combined: pd.DataFrame,
                        attacker_rows: list[RequestState], trace_id: str) -> pd.DataFrame:
    a = attacker_only[attacker_only["tenant"] == "attacker"].copy()
    c = combined[combined["tenant"] == "attacker"].copy()
    m = a.merge(c, on="request_index", suffixes=("_attacker_only", "_combined"), validate="one_to_one")
    excess = m["turnaround_ns_combined"].to_numpy(float) - m["turnaround_ns_attacker_only"].to_numpy(float)
    path_map = {r.request_index: r.probe_path for r in attacker_rows}
    return pd.DataFrame({
        "trace_id": trace_id,
        "probe_index": m["request_index"].astype(int),
        "probe_path": m["request_index"].map(path_map),
        "release_ns": m["release_ns_attacker_only"].astype(float),
        "attacker_only_success": m["success_attacker_only"].astype(bool),
        "combined_success": m["success_combined"].astype(bool),
        "attacker_only_completion_ns": m["external_completion_ns_attacker_only"].astype(float),
        "combined_completion_ns": m["external_completion_ns_combined"].astype(float),
        "attacker_only_turnaround_ns": m["turnaround_ns_attacker_only"].astype(float),
        "combined_turnaround_ns": m["turnaround_ns_combined"].astype(float),
        "excess_turnaround_ns": excess,
        "delayed": excess > AFFECTED_THRESHOLD_NS,
        "speedup": excess < -AFFECTED_THRESHOLD_NS,
        "failure_transition": m["success_attacker_only"].astype(bool).to_numpy() != m["success_combined"].astype(bool).to_numpy(),
    })


def run_lengths(mask: np.ndarray) -> list[int]:
    out, cur = [], 0
    for v in mask.astype(bool):
        if v:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur: out.append(cur)
    return out


def autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag: return 0.0
    a, b = x[:-lag], x[lag:]
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    v = float(np.corrcoef(a, b)[0, 1])
    return v if np.isfinite(v) else 0.0


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2: return 0.0
    a, b = a[:n], b[:n]
    if np.std(a) < EPS or np.std(b) < EPS: return 0.0
    v = float(np.corrcoef(a, b)[0, 1])
    return v if np.isfinite(v) else 0.0


def spectral_features(x: np.ndarray) -> dict[str, float]:
    if len(x) < 4:
        return dict(spectral_dominant_bin_fraction=0.0, spectral_centroid_fraction=0.0,
                    spectral_entropy=0.0, spectral_low_frequency_power_fraction=0.0)
    y = x - np.mean(x)
    power = np.abs(np.fft.rfft(y)) ** 2
    if len(power): power[0] = 0
    total = float(power.sum())
    if total <= EPS:
        return dict(spectral_dominant_bin_fraction=0.0, spectral_centroid_fraction=0.0,
                    spectral_entropy=0.0, spectral_low_frequency_power_fraction=0.0)
    bins = np.arange(len(power), dtype=float)
    denom = max(len(power)-1, 1)
    probs = power / total
    nz = probs[probs > EPS]
    ent = -float(np.sum(nz*np.log(nz))) / (math.log(len(power)) if len(power) > 1 else 1.0)
    cutoff = max(2, int(math.ceil(len(power)*0.25)))
    return {
        "spectral_dominant_bin_fraction": float(np.argmax(power))/denom,
        "spectral_centroid_fraction": float(np.sum(bins*power)/total)/denom,
        "spectral_entropy": ent,
        "spectral_low_frequency_power_fraction": float(np.sum(power[1:cutoff])/total),
    }


BASE_FEATURES = [
    "probe_count", "mean_excess_ns", "median_excess_ns", "mean_abs_excess_ns",
    "std_excess_ns", "max_excess_ns", "min_excess_ns", "p10_excess_ns",
    "p25_excess_ns", "p50_excess_ns", "p75_excess_ns", "p90_excess_ns",
    "p95_excess_ns", "p99_excess_ns", "tail_mass_gt_p90_fraction",
    "tail_mass_gt_2x_median_abs_fraction", "delayed_fraction", "speedup_fraction",
    "failure_transition_fraction", "cumulative_positive_excess_ns",
    "cumulative_negative_magnitude_ns", "cumulative_abs_excess_ns",
    "longest_delayed_run", "longest_speedup_run", "delayed_run_count",
    "speedup_run_count", "mean_delayed_run_length", "mean_speedup_run_length",
    "long_gap_frequency", "mean_inter_affected_gap_probes",
    "std_inter_affected_gap_probes", "inferred_busy_period_ns",
    "lag1_autocorrelation", "lag2_autocorrelation", "lag3_autocorrelation",
    "spectral_dominant_bin_fraction", "spectral_centroid_fraction",
    "spectral_entropy", "spectral_low_frequency_power_fraction",
    "early_mean_abs_ns", "middle_mean_abs_ns", "late_mean_abs_ns",
    "cross_path_corr_mean", "cross_path_corr_max", "cross_path_corr_min",
]
PATH_FEATURES = []
for path in PROBE_PATHS:
    PATH_FEATURES += [
        f"{path}__mean_excess_ns", f"{path}__mean_abs_excess_ns",
        f"{path}__delayed_fraction", f"{path}__speedup_fraction",
        f"{path}__cumulative_abs_excess_ns", f"{path}__cumulative_positive_excess_ns",
        f"{path}__p95_abs_excess_ns",
    ]
FEATURE_COLUMNS = BASE_FEATURES + PATH_FEATURES


def extract_features(trace: pd.DataFrame, attacker_period_ns: float,
                     probe_budget: int | None = None) -> dict[str, float]:
    t = trace.sort_values("probe_index")
    if probe_budget is not None: t = t.head(int(probe_budget))
    x = t["excess_turnaround_ns"].to_numpy(float)
    ax = np.abs(x)
    delayed = x > AFFECTED_THRESHOLD_NS
    speedup = x < -AFFECTED_THRESHOLD_NS
    affected = ax > AFFECTED_THRESHOLD_NS
    fail = t["failure_transition"].to_numpy(bool)
    dr, sr, ar = run_lengths(delayed), run_lengths(speedup), run_lengths(affected)
    idx = np.flatnonzero(affected)
    gaps = np.diff(idx).astype(float) if len(idx) >= 2 else np.array([], float)
    med_abs = float(np.median(ax)) if len(ax) else 0.0
    p90_abs = float(np.quantile(ax, .9)) if len(ax) else 0.0
    q = lambda p: float(np.quantile(x, p)) if len(x) else 0.0
    thirds = np.array_split(np.arange(len(x)), 3)
    thirds_mean = [float(np.mean(ax[z])) if len(z) else 0.0 for z in thirds]

    arrays, path_feats = {}, {}
    for path in PROBE_PATHS:
        vals = t.loc[t["probe_path"] == path, "excess_turnaround_ns"].to_numpy(float)
        arrays[path] = vals
        av = np.abs(vals)
        path_feats.update({
            f"{path}__mean_excess_ns": float(np.mean(vals)) if len(vals) else 0.0,
            f"{path}__mean_abs_excess_ns": float(np.mean(av)) if len(vals) else 0.0,
            f"{path}__delayed_fraction": float(np.mean(vals > AFFECTED_THRESHOLD_NS)) if len(vals) else 0.0,
            f"{path}__speedup_fraction": float(np.mean(vals < -AFFECTED_THRESHOLD_NS)) if len(vals) else 0.0,
            f"{path}__cumulative_abs_excess_ns": float(np.sum(av)),
            f"{path}__cumulative_positive_excess_ns": float(np.sum(np.maximum(vals, 0))),
            f"{path}__p95_abs_excess_ns": float(np.quantile(av,.95)) if len(vals) else 0.0,
        })
    corrs = np.array([safe_corr(arrays[PROBE_PATHS[i]], arrays[PROBE_PATHS[j]])
                      for i in range(3) for j in range(i+1,3)], float)
    if not len(corrs): corrs = np.array([0.0])
    row = {
        "probe_count": float(len(x)),
        "mean_excess_ns": float(np.mean(x)) if len(x) else 0.0,
        "median_excess_ns": float(np.median(x)) if len(x) else 0.0,
        "mean_abs_excess_ns": float(np.mean(ax)) if len(x) else 0.0,
        "std_excess_ns": float(np.std(x)) if len(x) else 0.0,
        "max_excess_ns": float(np.max(x)) if len(x) else 0.0,
        "min_excess_ns": float(np.min(x)) if len(x) else 0.0,
        "p10_excess_ns": q(.10), "p25_excess_ns": q(.25), "p50_excess_ns": q(.50),
        "p75_excess_ns": q(.75), "p90_excess_ns": q(.90), "p95_excess_ns": q(.95), "p99_excess_ns": q(.99),
        "tail_mass_gt_p90_fraction": float(np.mean(ax > p90_abs)) if len(ax) else 0.0,
        "tail_mass_gt_2x_median_abs_fraction": float(np.mean(ax > 2*med_abs)) if len(ax) and med_abs > 0 else 0.0,
        "delayed_fraction": float(np.mean(delayed)) if len(x) else 0.0,
        "speedup_fraction": float(np.mean(speedup)) if len(x) else 0.0,
        "failure_transition_fraction": float(np.mean(fail)) if len(x) else 0.0,
        "cumulative_positive_excess_ns": float(np.sum(np.maximum(x,0))),
        "cumulative_negative_magnitude_ns": float(np.sum(np.maximum(-x,0))),
        "cumulative_abs_excess_ns": float(np.sum(ax)),
        "longest_delayed_run": float(max(dr) if dr else 0),
        "longest_speedup_run": float(max(sr) if sr else 0),
        "delayed_run_count": float(len(dr)), "speedup_run_count": float(len(sr)),
        "mean_delayed_run_length": float(np.mean(dr)) if dr else 0.0,
        "mean_speedup_run_length": float(np.mean(sr)) if sr else 0.0,
        "long_gap_frequency": float(np.mean(gaps >= 3)) if len(gaps) else 0.0,
        "mean_inter_affected_gap_probes": float(np.mean(gaps)) if len(gaps) else 0.0,
        "std_inter_affected_gap_probes": float(np.std(gaps)) if len(gaps) else 0.0,
        "inferred_busy_period_ns": float(max(ar)*attacker_period_ns) if ar else 0.0,
        "lag1_autocorrelation": autocorr(x,1), "lag2_autocorrelation": autocorr(x,2), "lag3_autocorrelation": autocorr(x,3),
        "early_mean_abs_ns": thirds_mean[0], "middle_mean_abs_ns": thirds_mean[1], "late_mean_abs_ns": thirds_mean[2],
        "cross_path_corr_mean": float(np.mean(corrs)), "cross_path_corr_max": float(np.max(corrs)), "cross_path_corr_min": float(np.min(corrs)),
    }
    row.update(spectral_features(x)); row.update(path_feats)
    return row


# =============================================================================
# Dataset generation
# =============================================================================

def victim_slowdown(victim_only: pd.DataFrame, combined: pd.DataFrame) -> dict[str,float]:
    v = victim_only[victim_only["tenant"]=="victim"].copy()
    c = combined[combined["tenant"]=="victim"].copy()
    if v.empty or c.empty: return {"victim_mean_request_slowdown":1.0,"victim_makespan_slowdown":1.0}
    m = v[["request_index","turnaround_ns"]].merge(c[["request_index","turnaround_ns"]], on="request_index", suffixes=("_v","_c"))
    base = m["turnaround_ns_v"].to_numpy(float); comb = m["turnaround_ns_c"].to_numpy(float)
    good = np.isfinite(base) & np.isfinite(comb) & (base>0)
    req_slow = float(np.mean(comb[good]/base[good])) if np.any(good) else 1.0
    goodv = v[np.isfinite(v["external_completion_ns"])]
    goodc = c[np.isfinite(c["external_completion_ns"])]
    if goodv.empty or goodc.empty: make=1.0
    else:
        start = float(v["release_ns"].min())
        b = float(goodv["external_completion_ns"].max()-start); cc=float(goodc["external_completion_ns"].max()-start)
        make = cc/b if b>0 else 1.0
    return {"victim_mean_request_slowdown":req_slow,"victim_makespan_slowdown":make}


def simulate_dataset(p27, schedules, conditions, profiles, link_probs,
                     repeats, window_ns, seed):
    trace_parts=[]; feat_rows=[]; truth_rows=[]; trial_rows=[]
    retry_parts=[]; distill_parts=[]; eval_rows=[]; rel_rows=[]
    total=len(schedules)*len(conditions)*len(link_probs)*repeats; done=0

    # attacker-only baseline by repeat
    attacker_cache={}
    base_cond = conditions[0]
    for r in range(repeats):
        ars=make_attacker_requests(p27,r,window_ns)
        sim=Simulator(p27,base_cond,profiles[base_cond.post_profile],1.0,seed,"attacker_only",r,"attacker_only")
        req,*_ = sim.run([clone_request_state(x) for x in ars])
        attacker_cache[r]=(req,ars)

    for s in schedules:
        for r in range(repeats):
            releases=schedule_releases(s,r,seed,window_ns)
            for i,t in enumerate(releases):
                rel_rows.append({"base_schedule_id":s.base_schedule_id,"repeat_id":r,"schedule_profile":s.schedule_profile,"victim_request_index":i,"release_ns":float(t)})
            ars=make_attacker_requests(p27,r,window_ns)
            vrs=make_victim_requests(releases,s.base_schedule_id,r)
            for lp in link_probs:
                for cond in conditions:
                    prof=profiles[cond.post_profile]
                    sv=Simulator(p27,cond,prof,lp,seed,s.base_schedule_id,r,"victim_only")
                    vreq,vwait,vint,vstage,vretry,vdist,vpool=sv.run([clone_request_state(x) for x in vrs])
                    sc=Simulator(p27,cond,prof,lp,seed,s.base_schedule_id,r,"combined")
                    combined=[clone_request_state(x) for x in (ars+vrs)]
                    creq,cwait,cint,cstage,cretry,cdist,cpool=sc.run(combined)
                    trace_id=hashlib.sha256(f"{s.base_schedule_id}|{r}|{lp:.6f}|{cond.condition_id}|{seed}".encode()).hexdigest()[:20]
                    trace=pair_attacker_trace(attacker_cache[r][0],creq,ars,trace_id)
                    trace_parts.append(trace)
                    f=extract_features(trace,float(p27.ATTACKER_PERIOD_NS))
                    feat_rows.append({"trace_id":trace_id,"base_schedule_id":s.base_schedule_id,"repeat_id":r,"link_success_probability":lp,**f})
                    vr = cretry[cretry.get("tenant",pd.Series(dtype=str))=="victim"] if not cretry.empty and "tenant" in cretry else pd.DataFrame()
                    actual_attempts=len(vr)
                    failed=int((~vr["success"].astype(bool)).sum()) if len(vr) else 0
                    vsuccess=creq.loc[creq["tenant"]=="victim","success"].mean() if len(vrs) else 1.0
                    slow=victim_slowdown(vreq,creq)
                    truth={
                        "trace_id":trace_id,"base_schedule_id":s.base_schedule_id,"repeat_id":r,
                        "schedule_profile":s.schedule_profile,"link_success_probability":lp,
                        "condition_id":cond.condition_id,"task_name":cond.task_name,"evaluator_label":cond.evaluator_label,
                        "runtime_mode":cond.runtime_mode,"post_profile":cond.post_profile,
                        "distillation_depth":cond.distillation_depth,"retry_limit":cond.retry_limit,"uses_prefetch":cond.uses_prefetch,
                        "victim_remote_operation_count":len(vrs),"victim_success_fraction":float(vsuccess),
                        "actual_generation_attempt_count":actual_attempts,"actual_retry_count":failed,**slow,
                    }
                    truth_rows.append(truth); trial_rows.append({**truth,**f})
                    if not cretry.empty:
                        z=cretry.copy(); z["trace_id"]=trace_id; retry_parts.append(z)
                    if not cdist.empty:
                        z=cdist.copy(); z["trace_id"]=trace_id; distill_parts.append(z)
                    eval_rows.append({"trace_id":trace_id,"condition_id":cond.condition_id,"task_name":cond.task_name,
                                      "evaluator_label":cond.evaluator_label,"link_success_probability":lp,
                                      "combined_wait_events":len(cwait),"combined_resource_intervals":len(cint),
                                      "combined_stage_records":len(cstage),"combined_pool_events":len(cpool),
                                      "actual_generation_attempt_count":actual_attempts,"actual_retry_count":failed,
                                      "distillation_event_count":len(cdist)})
                    done+=1
                    if done%max(1,total//20)==0 or done==total: print(f"[Phase 3.5] Generated {done}/{total} traces")
    return (
        pd.concat(trace_parts,ignore_index=True), pd.DataFrame(feat_rows), pd.DataFrame(truth_rows), pd.DataFrame(trial_rows),
        pd.concat(retry_parts,ignore_index=True) if retry_parts else pd.DataFrame(),
        pd.concat(distill_parts,ignore_index=True) if distill_parts else pd.DataFrame(),
        pd.DataFrame(eval_rows), pd.DataFrame(rel_rows).drop_duplicates()
    )


# =============================================================================
# Group split and ML
# =============================================================================

def build_group_split(schedules: list[BaseSchedule], seed: int, test_size: float) -> pd.DataFrame:
    tab=pd.DataFrame([{"base_schedule_id":s.base_schedule_id,"schedule_profile":s.schedule_profile} for s in schedules])
    rng=np.random.default_rng(stable_seed(seed,"split")); test_ids=set()
    for profile,g in tab.groupby("schedule_profile",sort=True):
        ids=g["base_schedule_id"].to_numpy().copy(); rng.shuffle(ids)
        if len(ids)<2: raise ValueError(f"Need >=2 schedules for {profile}")
        n=max(1,min(len(ids)-1,int(round(len(ids)*test_size))))
        test_ids.update(ids[:n].tolist())
    tab["split"]=tab["base_schedule_id"].map(lambda x:"test" if x in test_ids else "train")
    return tab.sort_values(["schedule_profile","base_schedule_id"]).reset_index(drop=True)


def classifiers(seed:int, trees:int):
    return {
        "logistic_regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=4000,class_weight="balanced",random_state=seed))]),
        "random_forest":RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1),
        "hist_gradient_boosting":HistGradientBoostingClassifier(random_state=seed,max_iter=250,l2_regularization=1e-3),
    }


def regressors(seed:int, trees:int):
    return {
        "random_forest":RandomForestRegressor(n_estimators=trees,random_state=seed,n_jobs=-1),
        "hist_gradient_boosting":HistGradientBoostingRegressor(random_state=seed,max_iter=250,l2_regularization=1e-3),
        "elastic_net":Pipeline([("scale",StandardScaler()),("model",ElasticNet(alpha=.01,l1_ratio=.25,random_state=seed,max_iter=10000))]),
    }


def evaluate_classification(analysis:pd.DataFrame, split:pd.DataFrame, seed:int, trees:int):
    sm=split.set_index("base_schedule_id")["split"]; data=analysis.copy(); data["split"]=data["base_schedule_id"].map(sm)
    metrics=[]; preds=[]; cms=[]
    for task,g in data.groupby("task_name",sort=True):
        tr=g[g["split"]=="train"]; te=g[g["split"]=="test"]
        labels=sorted(g["evaluator_label"].unique()); chance=1/len(labels)
        Xtr=tr[FEATURE_COLUMNS].astype(float).to_numpy(); Xte=te[FEATURE_COLUMNS].astype(float).to_numpy()
        ytr=tr["evaluator_label"].astype(str).to_numpy(); yte=te["evaluator_label"].astype(str).to_numpy()
        for name,model in classifiers(seed,trees).items():
            model.fit(Xtr,ytr); yp=model.predict(Xte)
            auc=math.nan
            if len(labels)==2 and hasattr(model,"predict_proba") and len(np.unique(yte))==2:
                pos=labels[-1]; classes=list(model.classes_); pr=model.predict_proba(Xte)
                auc=float(roc_auc_score((yte==pos).astype(int),pr[:,classes.index(pos)]))
            metrics.append({"task_name":task,"model_name":name,"sample_count":len(te),"class_count":len(labels),"chance_accuracy":chance,
                            "accuracy":accuracy_score(yte,yp),"balanced_accuracy":balanced_accuracy_score(yte,yp),
                            "macro_f1":f1_score(yte,yp,average="macro",zero_division=0),"binary_roc_auc":auc})
            ter=te.reset_index(drop=True)
            for i,row in ter.iterrows():
                preds.append({"trace_id":row.trace_id,"base_schedule_id":row.base_schedule_id,"repeat_id":int(row.repeat_id),
                              "schedule_profile":row.schedule_profile,"link_success_probability":float(row.link_success_probability),
                              "task_name":task,"true_label":yte[i],"predicted_label":yp[i],"correct":bool(yte[i]==yp[i]),"model_name":name})
            cm=confusion_matrix(yte,yp,labels=labels)
            for i,tl in enumerate(labels):
                den=cm[i].sum()
                for j,pl in enumerate(labels):
                    cms.append({"task_name":task,"model_name":name,"true_label":tl,"predicted_label":pl,
                                "count":int(cm[i,j]),"true_normalized_fraction":float(cm[i,j]/den) if den else 0.0})
    return pd.DataFrame(metrics),pd.DataFrame(preds),pd.DataFrame(cms)


def evaluate_regression(analysis,split,seed,trees):
    sm=split.set_index("base_schedule_id")["split"]; data=analysis.copy(); data["split"]=data["base_schedule_id"].map(sm)
    metrics=[]; preds=[]
    specs=[("distillation_depth_estimation",TASK_DISTILL,"distillation_depth"),("retry_count_estimation",TASK_RETRY,"actual_retry_count")]
    for task,source,target in specs:
        g=data[data["task_name"]==source]; tr=g[g["split"]=="train"]; te=g[g["split"]=="test"]
        Xtr=tr[FEATURE_COLUMNS].astype(float).to_numpy(); Xte=te[FEATURE_COLUMNS].astype(float).to_numpy()
        ytr=tr[target].astype(float).to_numpy(); yte=te[target].astype(float).to_numpy()
        for name,model in regressors(seed,trees).items():
            model.fit(Xtr,ytr); yp=np.asarray(model.predict(Xte),float)
            metrics.append({"task_name":task,"source_task":source,"target":target,"model_name":name,"sample_count":len(te),
                            "mae":mean_absolute_error(yte,yp),"rmse":math.sqrt(mean_squared_error(yte,yp)),
                            "r2":r2_score(yte,yp) if len(np.unique(yte))>1 else math.nan})
            ter=te.reset_index(drop=True)
            for i,row in ter.iterrows():
                preds.append({"trace_id":row.trace_id,"base_schedule_id":row.base_schedule_id,"repeat_id":int(row.repeat_id),
                              "schedule_profile":row.schedule_profile,"link_success_probability":float(row.link_success_probability),
                              "task_name":task,"target":target,"true_value":float(yte[i]),"predicted_value":float(yp[i]),
                              "absolute_error":float(abs(yp[i]-yte[i])),"model_name":name})
    return pd.DataFrame(metrics),pd.DataFrame(preds)


def workload_generalization(analysis,seed,trees):
    rows=[]
    for task,g in analysis.groupby("task_name",sort=True):
        labels=sorted(g["evaluator_label"].unique())
        for held in sorted(g["schedule_profile"].unique()):
            tr=g[g["schedule_profile"]!=held]; te=g[g["schedule_profile"]==held]
            model=RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1)
            model.fit(tr[FEATURE_COLUMNS].astype(float),tr["evaluator_label"].astype(str)); yp=model.predict(te[FEATURE_COLUMNS].astype(float))
            rows.append({"task_name":task,"model_name":"random_forest","held_out_schedule_profile":held,"sample_count":len(te),
                         "chance_accuracy":1/len(labels),"accuracy":accuracy_score(te.evaluator_label.astype(str),yp),
                         "balanced_accuracy":balanced_accuracy_score(te.evaluator_label.astype(str),yp),
                         "macro_f1":f1_score(te.evaluator_label.astype(str),yp,average="macro",zero_division=0)})
    return pd.DataFrame(rows)


def link_generalization(analysis,seed,trees):
    rows=[]
    probs=sorted(analysis["link_success_probability"].unique())
    if len(probs)<2: return pd.DataFrame(columns=["task_name","model_name","held_out_link_success_probability","sample_count","chance_accuracy","accuracy","balanced_accuracy","macro_f1"])
    for task,g in analysis.groupby("task_name",sort=True):
        labels=sorted(g["evaluator_label"].unique())
        for held in probs:
            tr=g[~np.isclose(g["link_success_probability"],held)]; te=g[np.isclose(g["link_success_probability"],held)]
            model=RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1)
            model.fit(tr[FEATURE_COLUMNS].astype(float),tr.evaluator_label.astype(str)); yp=model.predict(te[FEATURE_COLUMNS].astype(float))
            rows.append({"task_name":task,"model_name":"random_forest","held_out_link_success_probability":float(held),"sample_count":len(te),
                         "chance_accuracy":1/len(labels),"accuracy":accuracy_score(te.evaluator_label.astype(str),yp),
                         "balanced_accuracy":balanced_accuracy_score(te.evaluator_label.astype(str),yp),
                         "macro_f1":f1_score(te.evaluator_label.astype(str),yp,average="macro",zero_division=0)})
    return pd.DataFrame(rows)


def probe_budget_metrics(traces,truth,split,p27,budgets,seed,trees):
    ti=truth.set_index("trace_id"); sm=split.set_index("base_schedule_id")["split"]; rows=[]
    for budget in budgets:
        fr=[]
        for tid,tr in traces.groupby("trace_id",sort=False):
            meta=ti.loc[tid]
            fr.append({"trace_id":tid,"base_schedule_id":meta.base_schedule_id,"task_name":meta.task_name,"evaluator_label":meta.evaluator_label,
                       **extract_features(tr,float(p27.ATTACKER_PERIOD_NS),budget)})
        d=pd.DataFrame(fr); d["split"]=d.base_schedule_id.map(sm)
        for task,g in d.groupby("task_name",sort=True):
            tr=g[g.split=="train"]; te=g[g.split=="test"]; labels=sorted(g.evaluator_label.unique())
            model=RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1)
            model.fit(tr[FEATURE_COLUMNS].astype(float),tr.evaluator_label.astype(str)); yp=model.predict(te[FEATURE_COLUMNS].astype(float))
            rows.append({"task_name":task,"model_name":"random_forest","probe_budget":budget,"sample_count":len(te),"chance_accuracy":1/len(labels),
                         "accuracy":accuracy_score(te.evaluator_label.astype(str),yp),"balanced_accuracy":balanced_accuracy_score(te.evaluator_label.astype(str),yp),
                         "macro_f1":f1_score(te.evaluator_label.astype(str),yp,average="macro",zero_division=0)})
    return pd.DataFrame(rows)


def best_summary(metrics):
    rows=[]
    for task,g in metrics.groupby("task_name",sort=True):
        b=g.sort_values(["accuracy","macro_f1"],ascending=False).iloc[0]
        rows.append({"task_name":task,"best_model":b.model_name,"class_count":int(b.class_count),"chance_accuracy":float(b.chance_accuracy),
                     "best_accuracy":float(b.accuracy),"best_balanced_accuracy":float(b.balanced_accuracy),"best_macro_f1":float(b.macro_f1),
                     "best_binary_roc_auc":float(b.binary_roc_auc) if np.isfinite(b.binary_roc_auc) else math.nan})
    return pd.DataFrame(rows)


def signal_summary(trials):
    return trials.groupby(["task_name","evaluator_label","schedule_profile","link_success_probability"],sort=True).agg(
        trace_count=("trace_id","count"),mean_abs_excess_ns=("mean_abs_excess_ns","mean"),mean_signed_excess_ns=("mean_excess_ns","mean"),
        delayed_fraction=("delayed_fraction","mean"),speedup_fraction=("speedup_fraction","mean"),p95_excess_ns=("p95_excess_ns","mean"),
        inferred_busy_period_ns=("inferred_busy_period_ns","mean"),actual_retry_count=("actual_retry_count","mean"),
        victim_success_fraction=("victim_success_fraction","mean")).reset_index()


# =============================================================================
# Validation
# =============================================================================

def build_validations(p27,traces,features,truth,split,schedules,conditions,profiles,link_probs):
    rows=[]
    def add(group,name,passed,expected,observed,details=""):
        rows.append({"validation_group":group,"assertion_name":name,"passed":bool(passed),"expected":str(expected),"observed":str(observed),"details":details})
    add("blackbox","attacker_trace_schema_exact",list(traces.columns)==ATTACKER_VISIBLE_COLUMNS,ATTACKER_VISIBLE_COLUMNS,list(traces.columns))
    forbidden=("victim","protocol","service","fidelity","distill","retry","resource","epr","label","condition","schedule_profile","link_success_probability")
    bad=[c for c in FEATURE_COLUMNS if any(tok in c.lower() for tok in forbidden)]
    add("blackbox","features_exclude_evaluator_state",not bad,[],bad)
    encoded=traces.trace_id.astype(str).str.contains("direct|epr|tele|distill|retry|fidelity|service|victim",case=False,regex=True)
    add("blackbox","trace_ids_are_opaque",not encoded.any(),0,int(encoded.sum()))
    add("blackbox","probe_paths_are_attacker_known_only",set(traces.probe_path.unique())==set(PROBE_PATHS),set(PROBE_PATHS),set(traces.probe_path.unique()))
    add("architecture","phase2_07_direct_timing_retained",abs(float(p27.DIRECT_CRITICAL_NS)-150)<1e-9 and abs(float(p27.RESET_NS)-120)<1e-9,"critical=150, cleanup=120",f"{p27.DIRECT_CRITICAL_NS},{p27.RESET_NS}")
    b=profiles["baseline"]
    add("architecture","phase2_07_entangled_baseline_retained",abs(b.endpoint_access_ns-p27.ENT_ENDPOINT_ACCESS_NS)<1e-9 and abs(b.readout_ns-p27.ENT_READOUT_NS)<1e-9 and abs(b.feedforward_ns-p27.ENT_FEEDFORWARD_NS)<1e-9 and abs(b.correction_ns-p27.ENT_CORRECTION_NS)<1e-9 and abs(b.reset_ns-p27.RESET_NS)<1e-9,"Phase2.7 baseline",asdict(b))
    add("architecture","attacker_protocol_fixed_direct",True,"direct","direct")
    labels={t:set(g.evaluator_label) for t,g in truth.groupby("task_name")}
    add("design","protocol_classes_complete",labels.get(TASK_PROTOCOL,set())=={"direct_transfer","on_demand_epr","prefetched_epr","telegate","teledata"},"5 requested",labels.get(TASK_PROTOCOL,set()))
    add("design","distillation_classes_complete",labels.get(TASK_DISTILL,set())=={"distill_0","distill_1","distill_2"},"0/1/2",labels.get(TASK_DISTILL,set()))
    add("design","retry_classes_complete",labels.get(TASK_RETRY,set())=={"retry_1","retry_2","retry_4"},"1/2/4",labels.get(TASK_RETRY,set()))
    add("design","service_classes_complete",labels.get(TASK_SERVICE,set())=={"low_latency","high_fidelity"},"low/high",labels.get(TASK_SERVICE,set()))
    cross=truth.groupby(["base_schedule_id","repeat_id","link_success_probability"]).condition_id.nunique()
    add("schedule","every_schedule_crossed_with_all_conditions",cross.min()==len(conditions) and cross.max()==len(conditions),len(conditions),f"min={cross.min()},max={cross.max()}")
    ops=truth.groupby(["base_schedule_id","repeat_id","link_success_probability"]).victim_remote_operation_count.nunique()
    add("schedule","logical_operation_count_fixed_across_conditions",ops.max()==1,1,int(ops.max()))
    allp={"sparse_periodic","dense_periodic","synchronization_bursty"}
    add("schedule","three_workload_profiles_present",set(s.schedule_profile for s in schedules)==allp,allp,set(s.schedule_profile for s in schedules))
    add("schedule","all_link_probabilities_present",set(np.round(truth.link_success_probability.unique(),12))==set(np.round(link_probs,12)),link_probs,sorted(truth.link_success_probability.unique()))
    train=set(split.loc[split.split=="train","base_schedule_id"]); test=set(split.loc[split.split=="test","base_schedule_id"])
    add("evaluation","no_group_overlap",not(train&test),[],sorted(train&test))
    trp=set(split.loc[split.split=="train","schedule_profile"]); tep=set(split.loc[split.split=="test","schedule_profile"])
    add("evaluation","all_profiles_in_train_and_test",trp==allp and tep==allp,allp,f"train={trp},test={tep}")
    add("evaluation","features_truth_one_to_one",len(features)==len(truth)==features.trace_id.nunique()==truth.trace_id.nunique(),len(truth),f"features={len(features)},truth={len(truth)}")
    merged=features.merge(truth[["trace_id","base_schedule_id","repeat_id","link_success_probability","task_name","evaluator_label"]],on=["trace_id","base_schedule_id","repeat_id","link_success_probability"])
    for task,name in [(TASK_PROTOCOL_CONTROL,"protocol_label_control_identical"),(TASK_SERVICE_CONTROL,"service_label_control_identical")]:
        spans=[]
        for _,g in merged[merged.task_name==task].groupby(["base_schedule_id","repeat_id","link_success_probability"]):
            arr=g[FEATURE_COLUMNS].astype(float).to_numpy(); spans.append(float(np.max(np.ptp(arr,axis=0))) if len(arr) else 0)
        ms=max(spans) if spans else math.inf; add("negative_control",name,ms<=1e-9,"<=1e-9",ms)
    dmap=truth[truth.task_name==TASK_DISTILL].groupby("evaluator_label").distillation_depth.first().to_dict()
    add("mechanism","distillation_depth_mapping",dmap=={"distill_0":0,"distill_1":1,"distill_2":2},{"distill_0":0,"distill_1":1,"distill_2":2},dmap)
    rmap=truth[truth.task_name==TASK_RETRY].groupby("evaluator_label").retry_limit.first().to_dict()
    add("mechanism","retry_limit_mapping",rmap=={"retry_1":1,"retry_2":2,"retry_4":4},{"retry_1":1,"retry_2":2,"retry_4":4},rmap)
    h=int(truth[(truth.task_name==TASK_SERVICE)&(truth.evaluator_label=="high_fidelity")].distillation_depth.iloc[0]); l=int(truth[(truth.task_name==TASK_SERVICE)&(truth.evaluator_label=="low_latency")].distillation_depth.iloc[0])
    add("mechanism","high_fidelity_uses_more_distillation",h>l,"high > low",f"{h}>{l}")
    pc=traces.groupby(["trace_id","probe_path"]).size().unstack(fill_value=0); md=int((pc.max(axis=1)-pc.min(axis=1)).max())
    add("probe_policy","balanced_probe_paths",md<=1,"<=1",md)
    add("execution","attacker_only_success",bool(traces.attacker_only_success.all()),True,bool(traces.attacker_only_success.all()))
    add("execution","combined_attacker_completions_finite",bool(np.isfinite(traces.combined_completion_ns).all()),True,bool(np.isfinite(traces.combined_completion_ns).all()))
    return pd.DataFrame(rows)


# =============================================================================
# Driver
# =============================================================================

def run_experiment(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    p27,source=load_phase2_07_module(); profiles=build_post_profiles(p27); conditions=build_conditions()
    schedules=build_base_schedules(args.seed,args.base_schedules); link_probs=parse_probability_list(args.link_success_probabilities)
    print(f"[Phase 3.5] schedules={len(schedules)}, conditions={len(conditions)}, link_probs={link_probs}, repeats={args.repeats_per_schedule}")
    print(f"[Phase 3.5] Reusing Phase-2.7 timing constants from: {source}")
    print("[Phase 3.5] Attacker protocol is fixed direct coherent; victim protocol/service is evaluator-only.")
    traces,features,truth,trials,retry_log,distill_log,evaluator,releases=simulate_dataset(
        p27,schedules,conditions,profiles,link_probs,args.repeats_per_schedule,args.observation_window_ns,args.seed)
    split=build_group_split(schedules,args.seed,args.test_size)
    analysis=features.merge(truth,on=["trace_id","base_schedule_id","repeat_id","link_success_probability"],validate="one_to_one")
    metrics,preds,cm=evaluate_classification(analysis,split,args.seed,args.rf_trees)
    regm,regp=evaluate_regression(analysis,split,args.seed,args.rf_trees)
    wg=workload_generalization(analysis,args.seed,args.rf_trees); lg=link_generalization(analysis,args.seed,args.rf_trees)
    maxp=int(traces.groupby("trace_id").probe_index.count().max()); budgets=sorted({min(maxp,int(x)) for x in args.probe_budgets.split(",") if x.strip() and int(x)>0});
    if maxp not in budgets: budgets.append(maxp)
    pb=probe_budget_metrics(traces,truth,split,p27,budgets,args.seed,args.rf_trees)
    summary=best_summary(metrics); signal=signal_summary(trials)
    val=build_validations(p27,traces,features,truth,split,schedules,conditions,profiles,link_probs)
    vals=pd.DataFrame([{"assertion_count":len(val),"passed_assertions":int(val.passed.sum()),"failed_assertions":int((~val.passed).sum()),"all_passed":bool(val.passed.all())}])

    traces.to_csv(out/"phase3_05_attacker_visible_trace.csv",index=False)
    features.to_csv(out/"phase3_05_trace_features.csv",index=False)
    truth.to_csv(out/"phase3_05_evaluator_ground_truth.csv",index=False)
    trials.to_csv(out/"phase3_05_trial_summary.csv",index=False)
    evaluator.to_csv(out/"phase3_05_evaluator_mechanism_summary.csv",index=False)
    releases.to_csv(out/"phase3_05_victim_release_schedule.csv",index=False)
    pd.DataFrame([asdict(x) for x in schedules]).to_csv(out/"phase3_05_base_schedule_table.csv",index=False)
    pd.DataFrame([asdict(x) for x in conditions]).to_csv(out/"phase3_05_condition_table.csv",index=False)
    pd.DataFrame([asdict(x) for x in profiles.values()]).to_csv(out/"phase3_05_post_epr_profile_table.csv",index=False)
    split.to_csv(out/"phase3_05_group_split.csv",index=False)
    metrics.to_csv(out/"phase3_05_inference_metrics.csv",index=False)
    preds.to_csv(out/"phase3_05_inference_predictions.csv",index=False)
    cm.to_csv(out/"phase3_05_confusion_matrix.csv",index=False)
    regm.to_csv(out/"phase3_05_regression_metrics.csv",index=False)
    regp.to_csv(out/"phase3_05_regression_predictions.csv",index=False)
    wg.to_csv(out/"phase3_05_workload_generalization_metrics.csv",index=False)
    lg.to_csv(out/"phase3_05_link_success_generalization_metrics.csv",index=False)
    pb.to_csv(out/"phase3_05_probe_budget_metrics.csv",index=False)
    summary.to_csv(out/"phase3_05_protocol_fidelity_summary.csv",index=False)
    signal.to_csv(out/"phase3_05_signal_summary.csv",index=False)
    val.to_csv(out/"phase3_05_validation_assertions.csv",index=False); vals.to_csv(out/"phase3_05_validation_summary.csv",index=False)
    if not retry_log.empty: retry_log.to_csv(out/"phase3_05_retry_attempt_log_evaluator.csv.gz",index=False,compression="gzip")
    if not distill_log.empty: distill_log.to_csv(out/"phase3_05_distillation_log_evaluator.csv.gz",index=False,compression="gzip")
    manifest={
        "experiment":"phase3_05_protocol_fidelity_inference","seed":args.seed,"phase2_07_source":str(source),"output_dir":str(out),
        "research_progression":{"3.1":"how much","3.2":"when","3.3":"where","3.4":"which edge/graph","3.5":"how implemented / requested service policy"},
        "attacker_protocol":"direct_coherent_remote_cx","logical_operation":getattr(p27,"LOGICAL_OPERATION","logical_remote_cx"),
        "tasks":{t:sorted(truth[truth.task_name==t].evaluator_label.unique().tolist()) for t in [TASK_PROTOCOL,TASK_DISTILL,TASK_RETRY,TASK_SERVICE]},
        "base_schedule_count":len(schedules),"schedule_profiles":sorted({s.schedule_profile for s in schedules}),
        "condition_count":len(conditions),"repeats_per_schedule":args.repeats_per_schedule,"observation_window_ns":args.observation_window_ns,
        "attacker_first_release_ns":float(p27.ATTACKER_FIRST_RELEASE_NS),"attacker_period_ns":float(p27.ATTACKER_PERIOD_NS),
        "probe_paths":list(PROBE_PATHS),"link_success_probabilities":link_probs,"test_size":args.test_size,"rf_trees":args.rf_trees,
        "probe_budgets":budgets,"trace_count":int(traces.trace_id.nunique()),"probe_row_count":len(traces),
        "training_base_schedule_count":int((split.split=="train").sum()),"test_base_schedule_count":int((split.split=="test").sum()),
        "feature_columns":FEATURE_COLUMNS,"attacker_visible_columns":ATTACKER_VISIBLE_COLUMNS,
        "validation_assertions":len(val),"validation_passed":int(val.passed.sum()),"all_validation_passed":bool(val.passed.all()),
        "notes":[
            "Phase 3.5 is an inference/attack-characterization experiment, not a defense experiment.",
            "Attacker protocol is fixed; hidden victim protocol/service semantics are evaluator-only.",
            "Distillation and service classes are controlled architectural resource models, not vendor fidelity measurements.",
            "Base schedules are crossed with all conditions and all link-success probabilities.",
            "Grouped train/test split is by base schedule; separate workload-held-out and link-probability-held-out metrics are reported.",
            "Label-only controls deliberately remove runtime semantics while retaining evaluator labels."
        ]}
    (out/"phase3_05_run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("\n[Phase 3.5] Validation"); print(vals.to_string(index=False))
    print("\n[Phase 3.5] Best held-out classification result per task"); print(summary.to_string(index=False))
    print("\n[Phase 3.5] Regression results"); print(regm.to_string(index=False))
    print(f"\n[Phase 3.5] Wrote outputs to: {out}")
    if args.fail_on_validation_error and not val.passed.all():
        raise AssertionError("Phase 3.5 validation failed:\n"+val.loc[~val.passed,["assertion_name","observed"]].to_string(index=False))


def parse_args():
    ap=argparse.ArgumentParser(description="Phase 3.5 — Protocol and Fidelity-Demand Inference")
    ap.add_argument("--output-dir",default=str(DEFAULT_OUTPUT_DIR)); ap.add_argument("--seed",type=int,default=DEFAULT_SEED)
    ap.add_argument("--base-schedules",type=int,default=DEFAULT_BASE_SCHEDULES); ap.add_argument("--repeats-per-schedule",type=int,default=DEFAULT_REPEATS_PER_SCHEDULE)
    ap.add_argument("--observation-window-ns",type=float,default=DEFAULT_OBSERVATION_WINDOW_NS)
    ap.add_argument("--link-success-probabilities",default=DEFAULT_LINK_SUCCESS_PROBABILITIES)
    ap.add_argument("--test-size",type=float,default=DEFAULT_TEST_SIZE); ap.add_argument("--rf-trees",type=int,default=DEFAULT_RF_TREES)
    ap.add_argument("--probe-budgets",default=DEFAULT_PROBE_BUDGETS)
    ap.add_argument("--fail-on-validation-error",action=argparse.BooleanOptionalAction,default=True)
    return ap.parse_args()


def main():
    args=parse_args()
    if args.base_schedules<6: raise ValueError("--base-schedules must be >=6")
    if args.repeats_per_schedule<1: raise ValueError("--repeats-per-schedule must be >=1")
    if not .05<=args.test_size<=.5: raise ValueError("--test-size must be in [.05,.5]")
    if args.rf_trees<10: raise ValueError("--rf-trees must be >=10")
    run_experiment(args)

if __name__=="__main__": main()
