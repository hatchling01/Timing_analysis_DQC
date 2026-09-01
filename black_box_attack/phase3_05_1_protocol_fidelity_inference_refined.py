#!/usr/bin/env python3
"""
Phase 3.5.1 — Refined Protocol and Fidelity-Demand Inference
============================================================

Purpose
-------
Phase 3.5.1 is a methodology-refinement rerun of Phase 3.5.  It keeps the
Phase-3 progression unchanged:

    3.1  how much communication?
    3.2  when does it occur?
    3.3  where does it occur?
    3.4  which intermodule edges / graph are active?
    3.5  how is communication implemented / what service policy is requested?

The refinement addresses three limitations identified after the Phase-3.5 run:

1. TeleGate and TeleData now have genuinely different causal resource paths,
   rather than only different durations on the same post-EPR sequence.

2. The low-latency vs high-fidelity service comparison is now a controlled
   one-factor comparison.  Both classes use the same on-demand EPR mechanism,
   the same retry limit, and the same post-EPR path.  The only semantic change
   is distillation depth (0 vs 2), which necessarily changes raw-pair demand
   from 1 to 4 pairs.  Thus the service-class result can be interpreted as a
   fidelity/reliability-demand effect rather than a bundle of unrelated policy
   changes.

3. probe_path_0/1/2 are now physical route contexts.  Each has distinct
   endpoint, switch/link, gate/state-transfer, and EPR-refill link resources.
   The attacker interleaves probes across these self-chosen routes, while the
   victim uses the same balanced route schedule in every condition.  Cross-path
   correlation features therefore correspond to physically distinct resource
   calendars rather than three labels over one shared queue.

This remains an inference / attack-characterization experiment, not a defense
experiment.

Primary tasks
-------------
A. protocol_inference
   direct_transfer
   on_demand_epr
   prefetched_epr
   telegate
   teledata

B. distillation_depth_inference
   distill_0
   distill_1
   distill_2
   + ordinal depth estimation

C. retry_policy_inference
   retry_1
   retry_2
   retry_4
   + actual retry-count estimation

D. service_class_inference
   low_latency   = no distillation
   high_fidelity = two distillation rounds
   Both otherwise use exactly the same runtime configuration.

Protocol semantics
------------------
Direct transfer:
    endpoint[path]
      -> switch_path[path]
      -> quantum_link[path]
      -> receiver gate[path]
      -> external completion
      -> reset

Generic EPR remote operation:
    EPR ready
      -> endpoint[path]
      -> readout
      -> feedforward
      -> correction[path]
      -> external completion
      -> reset

TeleGate:
    EPR ready
      -> endpoint[path]
      -> gate_interaction[path]
      -> readout
      -> feedforward
      -> gate_correction[path]
      -> external completion
      -> reset

TeleData:
    EPR ready
      -> state_load[path]
      -> endpoint[path]
      -> readout
      -> feedforward
      -> state_reconstruct[path]
      -> external completion
      -> reset

EPR management:
    * on-demand: fresh raw pair(s) before every request;
    * prefetched: warm two-pair pool PER ROUTE + asynchronous refill;
    * r-round distillation: 2**r raw pairs + r exclusive 90 ns rounds;
    * retry-limited: 1/2/4 attempts per required raw pair.

The exact durations are controlled architectural simulation parameters, not
claims about a commercial modular quantum computer.

Attacker boundary
-----------------
Attacker-visible output contains only the attacker's own chosen probe path and
its own timing/success observables.  Hidden victim route schedule, protocol,
service class, EPR state, retry outcomes, distillation depth, resource waits,
and evaluator attribution remain evaluator-only.

Default output directory
------------------------
blackbox_window_results/phase3/phase3.5.1/

Run
---
    python phase3_05_1_protocol_fidelity_inference_refined.py

Smoke
-----
    python phase3_05_1_protocol_fidelity_inference_refined.py \
        --base-schedules 6 \
        --repeats-per-schedule 1 \
        --observation-window-ns 6000 \
        --link-success-probabilities 0.7,0.9 \
        --rf-trees 50 \
        --output-dir /tmp/p351_smoke
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

DEFAULT_SEED = 3511
DEFAULT_BASE_SCHEDULES = 18
DEFAULT_REPEATS_PER_SCHEDULE = 2
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_LINK_SUCCESS_PROBABILITIES = "0.65,0.80,0.95"
DEFAULT_TEST_SIZE = 0.33
DEFAULT_RF_TREES = 500
DEFAULT_PROBE_BUDGETS = "12,24,36"
DEFAULT_OUTPUT_DIR = Path("blackbox_window_results") / "phase3" / "phase3.5.1"

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
    victim_route_offset: int


@dataclass(frozen=True)
class Condition:
    condition_id: str
    task_name: str
    evaluator_label: str
    runtime_mode: str
    post_pipeline: str
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
    route_path: str
    success: bool = True
    failure_reason: str = ""
    completion_ns: float = math.nan
    cleanup_ns: float = math.nan
    resource_wait_ns: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def clone_request_state(x: RequestState) -> RequestState:
    """Python-3.11-safe request cloning; avoids dataclasses.asdict(defaultdict)."""
    return RequestState(
        request_id=x.request_id,
        tenant=x.tenant,
        release_ns=float(x.release_ns),
        request_index=int(x.request_index),
        route_path=str(x.route_path),
        success=bool(x.success),
        failure_reason=str(x.failure_reason),
        completion_ns=float(x.completion_ns),
        cleanup_ns=float(x.cleanup_ns),
        resource_wait_ns=defaultdict(float),
    )


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


# =============================================================================
# Schedules and conditions
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
        route_offset = int(rng.integers(0, len(PROBE_PATHS)))
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
            f"base_{i:04d}", i, profile, phase, interval, bp, bs, bsp, route_offset
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


def build_conditions() -> list[Condition]:
    c = [
        Condition("protocol_direct", TASK_PROTOCOL, "direct_transfer", "direct", "direct", 0, 1, False, "Direct coherent transfer."),
        Condition("protocol_on_demand", TASK_PROTOCOL, "on_demand_epr", "custom_epr", "generic_epr", 0, 4, False, "Fresh raw pair for every request."),
        Condition("protocol_prefetch", TASK_PROTOCOL, "prefetched_epr", "prefetch", "generic_epr", 0, 4, True, "Warm route-local EPR pool."),
        Condition("protocol_telegate", TASK_PROTOCOL, "telegate", "prefetch", "telegate", 0, 4, True, "Gate-teleportation-specific pipeline."),
        Condition("protocol_teledata", TASK_PROTOCOL, "teledata", "prefetch", "teledata", 0, 4, True, "State-teleportation-specific pipeline."),
        Condition("distill_0", TASK_DISTILL, "distill_0", "custom_epr", "generic_epr", 0, 6, False, "One raw pair; no distillation."),
        Condition("distill_1", TASK_DISTILL, "distill_1", "custom_epr", "generic_epr", 1, 6, False, "Two raw pairs; one distillation round."),
        Condition("distill_2", TASK_DISTILL, "distill_2", "custom_epr", "generic_epr", 2, 6, False, "Four raw pairs; two distillation rounds."),
        Condition("retry_1", TASK_RETRY, "retry_1", "custom_epr", "generic_epr", 0, 1, False, "One attempt per required raw pair."),
        Condition("retry_2", TASK_RETRY, "retry_2", "custom_epr", "generic_epr", 0, 2, False, "Two attempts per required raw pair."),
        Condition("retry_4", TASK_RETRY, "retry_4", "custom_epr", "generic_epr", 0, 4, False, "Four attempts per required raw pair."),
        # Controlled service comparison: same mode, same retry limit, same post path.
        Condition("service_low_latency", TASK_SERVICE, "low_latency", "custom_epr", "generic_epr", 0, 4, False, "No-distillation low-latency service."),
        Condition("service_high_fidelity", TASK_SERVICE, "high_fidelity", "custom_epr", "generic_epr", 2, 4, False, "Two-round high-fidelity service."),
    ]
    for label in ("direct_transfer", "on_demand_epr", "prefetched_epr", "telegate", "teledata"):
        c.append(Condition(f"protocol_control_{label}", TASK_PROTOCOL_CONTROL, label, "prefetch", "generic_epr", 0, 4, True, "Label-only protocol control."))
    for label in ("low_latency", "high_fidelity"):
        c.append(Condition(f"service_control_{label}", TASK_SERVICE_CONTROL, label, "custom_epr", "generic_epr", 1, 4, False, "Label-only service control."))
    return c


# =============================================================================
# Physical resource simulator
# =============================================================================

class Resource:
    def __init__(self, name: str):
        self.name = name
        self.owner: str | None = None
        self.queue: deque[Waiter] = deque()
        self.start_ns = math.nan

    def request(self, sim: "Simulator", waiter: Waiter, now: float) -> None:
        key = (waiter.actor_id, self.name, waiter.stage)
        sim.wait_arrival[key] = float(now)
        if self.owner is None:
            self._grant(sim, waiter, now)
        else:
            self.queue.append(waiter)

    def _grant(self, sim: "Simulator", waiter: Waiter, now: float) -> None:
        self.owner = waiter.actor_id
        self.start_ns = float(now)
        key = (waiter.actor_id, self.name, waiter.stage)
        arrival = sim.wait_arrival.pop(key, now)
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
            "end_ns": float(now), "duration_ns": float(now)-self.start_ns,
        })
        self.owner = None
        self.start_ns = math.nan
        if self.queue:
            self._grant(sim, self.queue.popleft(), now)


class Simulator:
    def __init__(self, p27, condition: Condition, link_p: float, seed: int,
                 schedule_id: str, repeat_id: int, run_kind: str):
        self.p27 = p27
        self.condition = condition
        self.link_p = float(link_p)
        self.seed = int(seed)
        self.schedule_id = schedule_id
        self.repeat_id = int(repeat_id)
        self.run_kind = run_kind
        self.events: list[tuple[float,int,str,dict[str,Any]]] = []
        self.seq = 0
        self.requests: dict[str,RequestState] = {}
        self.resources: dict[str,Resource] = {}
        self.wait_arrival: dict[tuple[str,str,str],float] = {}
        self.wait_rows=[]; self.interval_rows=[]; self.stage_rows=[]
        self.retry_rows=[]; self.distill_rows=[]; self.pool_events=[]
        self.pools: dict[str,deque[Pair]] = {p:deque() for p in PROBE_PATHS}
        self.pool_waiters: dict[str,deque[str]] = {p:deque() for p in PROBE_PATHS}
        self.refill_inflight: dict[str,int] = {p:0 for p in PROBE_PATHS}
        self.refill_counter=0; self.pair_counter=0
        self.custom: dict[str,dict[str,Any]] = {}
        self.background_path: dict[str,str] = {}

    def semantic_random_key(self) -> str:
        if self.condition.task_name == TASK_PROTOCOL_CONTROL:
            return TASK_PROTOCOL_CONTROL
        if self.condition.task_name == TASK_SERVICE_CONTROL:
            return TASK_SERVICE_CONTROL
        return self.condition.condition_id

    def resource(self, name: str) -> Resource:
        if name not in self.resources:
            self.resources[name]=Resource(name)
        return self.resources[name]

    def schedule(self,t,event,payload):
        self.seq+=1; heapq.heappush(self.events,(float(t),self.seq,event,payload))

    def request_resource(self,actor,tenant,name,stage,now,duration,release_on_done=True):
        self.resource(name).request(self,Waiter(actor,tenant,stage,float(duration),release_on_done),now)

    def path_resource(self, base: str, path: str) -> str:
        return f"{base}::{path}"

    def add_request(self,r:RequestState):
        self.requests[r.request_id]=r
        self.schedule(r.release_ns,"request_release",{"request_id":r.request_id})

    def initialize_prefetch(self):
        if not self.condition.uses_prefetch: return
        for path in PROBE_PATHS:
            for _ in range(2):
                self.pair_counter+=1
                self.pools[path].append(Pair(f"warm::{path}::{self.pair_counter}",float(self.p27.EPR_PAIR_LIFETIME_NS)))
            self.pool_events.append({"run_kind":self.run_kind,"time_ns":0.0,"event":"warm_prefetch","route_path":path,"pool_level":len(self.pools[path])})

    def prune_pool(self,path,now):
        kept=deque(); expired=0
        while self.pools[path]:
            pair=self.pools[path].popleft()
            if pair.expires_ns<=now: expired+=1
            else: kept.append(pair)
        self.pools[path]=kept
        if expired:
            self.pool_events.append({"run_kind":self.run_kind,"time_ns":now,"event":"pair_expired","route_path":path,"pool_level":len(kept),"count":expired})

    def ensure_refill(self,path,now):
        if not self.condition.uses_prefetch: return
        while len(self.pools[path])+self.refill_inflight[path]<2:
            self.refill_counter+=1
            actor=f"refill::{path}::{self.refill_counter}"
            self.background_path[actor]=path
            self.refill_inflight[path]+=1
            self.request_resource(actor,"victim","epr_generator","refill_generator",now,float(self.p27.EPR_GENERATOR_SETUP_NS),True)

    def acquire_prefetch(self,rid,now):
        path=self.requests[rid].route_path
        self.prune_pool(path,now)
        if self.pools[path]:
            pair=self.pools[path].popleft()
            self.pool_events.append({"run_kind":self.run_kind,"time_ns":now,"event":"pair_consumed","route_path":path,"request_id":rid,"pair_id":pair.pair_id,"pool_level":len(self.pools[path])})
            self.ensure_refill(path,now)
            self.start_post_epr(rid,now)
        else:
            self.pool_waiters[path].append(rid)
            self.pool_events.append({"run_kind":self.run_kind,"time_ns":now,"event":"pool_miss","route_path":path,"request_id":rid,"pool_level":0})
            self.ensure_refill(path,now)

    def generation_success(self,token,attempt):
        rng=np.random.default_rng(stable_seed(self.seed,self.semantic_random_key(),self.schedule_id,self.repeat_id,self.link_p,token,attempt))
        return bool(rng.random()<self.link_p)

    def start_custom_epr(self,rid,now):
        needed=2**int(max(0,self.condition.distillation_depth))
        self.custom[rid]={"needed":needed,"ready":0,"pair_attempt":0,"total_attempts":0,"failed_attempts":0,"distill_round":0}
        self.start_custom_attempt(rid,now)

    def start_custom_attempt(self,rid,now):
        st=self.custom[rid]; st["pair_attempt"]+=1; st["total_attempts"]+=1
        self.request_resource(rid,"victim","epr_generator","custom_generator",now,float(self.p27.EPR_GENERATOR_SETUP_NS),True)

    def fail_victim(self,rid,now):
        r=self.requests[rid]; r.success=False; r.failure_reason="retry_limit_exhausted"; r.completion_ns=float(now); r.cleanup_ns=float(now)

    def start_post_epr(self,rid,now):
        path=self.requests[rid].route_path
        if self.condition.post_pipeline == "generic_epr":
            self.request_resource(rid,"victim",self.path_resource("endpoint",path),"generic_endpoint",now,float(self.p27.ENT_ENDPOINT_ACCESS_NS),True)
        elif self.condition.post_pipeline == "telegate":
            self.request_resource(rid,"victim",self.path_resource("endpoint",path),"telegate_endpoint",now,20.0,True)
        elif self.condition.post_pipeline == "teledata":
            self.request_resource(rid,"victim",self.path_resource("state_load",path),"teledata_state_load",now,35.0,True)
        else:
            raise RuntimeError(self.condition.post_pipeline)

    def log_stage(self,rid,stage,start,end,resource_name=""):
        tenant=self.requests[rid].tenant if rid in self.requests else "system"
        path=self.requests[rid].route_path if rid in self.requests else self.background_path.get(rid,"")
        self.stage_rows.append({"run_kind":self.run_kind,"request_id":rid,"tenant":tenant,"route_path":path,"stage":stage,"resource_name":resource_name,"start_ns":start,"end_ns":end,"duration_ns":end-start})

    def run(self,rows:list[RequestState]):
        for r in rows: self.add_request(r)
        self.initialize_prefetch()
        while self.events:
            now,_,event,payload=heapq.heappop(self.events); self.handle_event(now,event,payload)
        req=pd.DataFrame([{"request_id":r.request_id,"tenant":r.tenant,"request_index":r.request_index,"route_path":r.route_path,"release_ns":r.release_ns,"success":r.success,"failure_reason":r.failure_reason,"external_completion_ns":r.completion_ns,"cleanup_completion_ns":r.cleanup_ns,"turnaround_ns":r.completion_ns-r.release_ns} for r in self.requests.values()])
        return req,pd.DataFrame(self.wait_rows),pd.DataFrame(self.interval_rows),pd.DataFrame(self.stage_rows),pd.DataFrame(self.retry_rows),pd.DataFrame(self.distill_rows),pd.DataFrame(self.pool_events)


    # ------------------------------------------------------------------
    # Event semantics
    # ------------------------------------------------------------------
    def handle_event(self,now,event,payload):
        if event=="request_release":
            rid=payload["request_id"]; r=self.requests[rid]
            if r.tenant=="attacker" or self.condition.runtime_mode=="direct":
                ep=self.path_resource("endpoint",r.route_path)
                self.request_resource(rid,r.tenant,ep,"direct_endpoint_prepare",now,float(self.p27.DIRECT_ENDPOINT_PREP_NS),False)
            elif self.condition.runtime_mode=="prefetch":
                self.acquire_prefetch(rid,now)
            elif self.condition.runtime_mode=="custom_epr":
                self.start_custom_epr(rid,now)
            else:
                raise RuntimeError(self.condition.runtime_mode)
            return

        if event=="resource_stage_done":
            name=payload["resource_name"]; actor=payload["actor_id"]; stage=payload["stage"]; duration=float(payload["duration_ns"])
            if payload["release_on_done"]:
                self.resource(name).release(self,actor,now)
            self.handle_stage_done(now,actor,stage,duration,name)
            return
        raise RuntimeError(event)

    def handle_stage_done(self,now,actor,stage,duration,resource_name):
        # --------------------------------------------------------------
        # Prefetch refill: common generator, route-specific quantum link
        # --------------------------------------------------------------
        if stage=="refill_generator":
            path=self.background_path[actor]
            ql=self.path_resource("quantum_link",path)
            self.request_resource(actor,"victim",ql,"refill_link",now,float(self.p27.EPR_LINK_GENERATION_NS),True)
            return
        if stage=="refill_link":
            path=self.background_path[actor]
            success=self.generation_success(actor,1)
            self.retry_rows.append({"run_kind":self.run_kind,"tenant":"victim","route_path":path,"request_id":"","attempt_kind":"prefetch_refill","success":success,"time_ns":now})
            self.refill_inflight[path]=max(0,self.refill_inflight[path]-1)
            if success:
                self.pair_counter+=1
                pair=Pair(f"pair::{path}::{self.pair_counter}",now+float(self.p27.EPR_PAIR_LIFETIME_NS))
                if self.pool_waiters[path]:
                    rid=self.pool_waiters[path].popleft()
                    self.pool_events.append({"run_kind":self.run_kind,"time_ns":now,"event":"refill_served_waiter","route_path":path,"request_id":rid,"pair_id":pair.pair_id,"pool_level":len(self.pools[path])})
                    self.start_post_epr(rid,now)
                else:
                    self.pools[path].append(pair)
                    self.pool_events.append({"run_kind":self.run_kind,"time_ns":now,"event":"pair_stored","route_path":path,"pair_id":pair.pair_id,"pool_level":len(self.pools[path])})
            else:
                self.pool_events.append({"run_kind":self.run_kind,"time_ns":now,"event":"refill_failed","route_path":path,"pool_level":len(self.pools[path])})
            self.ensure_refill(path,now)
            return

        # --------------------------------------------------------------
        # Direct coherent path: endpoint[path] held through reset.
        # switch_path[path] held through quantum_link[path].
        # --------------------------------------------------------------
        if stage=="direct_endpoint_prepare":
            path=self.requests[actor].route_path
            self.log_stage(actor,"endpoint_prepare",now-duration,now,resource_name)
            sw=self.path_resource("switch_path",path)
            self.request_resource(actor,self.requests[actor].tenant,sw,"direct_switch_setup",now,float(self.p27.DIRECT_SWITCH_SETUP_NS),False)
            return
        if stage=="direct_switch_setup":
            path=self.requests[actor].route_path
            self.log_stage(actor,"switch_setup",now-duration,now,resource_name)
            ql=self.path_resource("quantum_link",path)
            self.request_resource(actor,self.requests[actor].tenant,ql,"direct_link",now,float(self.p27.DIRECT_LINK_TRANSFER_NS),True)
            return
        if stage=="direct_link":
            path=self.requests[actor].route_path
            self.log_stage(actor,"synchronous_quantum_link_transfer",now-duration,now,resource_name)
            self.resource(self.path_resource("switch_path",path)).release(self,actor,now)
            recv=self.path_resource("receiver_gate",path)
            self.request_resource(actor,self.requests[actor].tenant,recv,"direct_receiver",now,float(self.p27.DIRECT_RECEIVER_GATE_NS),True)
            return
        if stage=="direct_receiver":
            self.log_stage(actor,"receiver_side_gate",now-duration,now,resource_name)
            self.requests[actor].completion_ns=float(now)
            self.request_resource(actor,self.requests[actor].tenant,"reset","direct_reset",now,float(self.p27.RESET_NS),True)
            return
        if stage=="direct_reset":
            path=self.requests[actor].route_path
            self.log_stage(actor,"postcompletion_reset",now-duration,now,resource_name)
            self.resource(self.path_resource("endpoint",path)).release(self,actor,now)
            self.requests[actor].cleanup_ns=float(now)
            return

        # --------------------------------------------------------------
        # On-demand raw pair generation: common generator, route link.
        # --------------------------------------------------------------
        if stage=="custom_generator":
            path=self.requests[actor].route_path
            ql=self.path_resource("quantum_link",path)
            self.request_resource(actor,"victim",ql,"custom_link",now,float(self.p27.EPR_LINK_GENERATION_NS),True)
            return
        if stage=="custom_link":
            st=self.custom[actor]
            success=self.generation_success(actor,st["total_attempts"])
            self.retry_rows.append({"run_kind":self.run_kind,"tenant":"victim","route_path":self.requests[actor].route_path,"request_id":actor,"attempt_kind":"on_demand_raw_pair","attempt_index_for_pair":st["pair_attempt"],"success":success,"time_ns":now})
            if success:
                st["ready"]+=1; st["pair_attempt"]=0
                if st["ready"]<st["needed"]:
                    self.start_custom_attempt(actor,now)
                elif self.condition.distillation_depth>0:
                    self.request_resource(actor,"victim","distillation","distill_round",now,90.0,True)
                else:
                    self.start_post_epr(actor,now)
            else:
                st["failed_attempts"]+=1
                if st["pair_attempt"]>=self.condition.retry_limit:
                    self.fail_victim(actor,now)
                else:
                    self.start_custom_attempt(actor,now)
            return
        if stage=="distill_round":
            st=self.custom[actor]; st["distill_round"]+=1
            self.log_stage(actor,"entanglement_distillation_round",now-duration,now,resource_name)
            self.distill_rows.append({"run_kind":self.run_kind,"request_id":actor,"route_path":self.requests[actor].route_path,"round_index":st["distill_round"],"completion_ns":now})
            if st["distill_round"]<self.condition.distillation_depth:
                self.request_resource(actor,"victim","distillation","distill_round",now,90.0,True)
            else:
                self.start_post_epr(actor,now)
            return

        # --------------------------------------------------------------
        # Generic EPR post path
        # endpoint[path] -> readout -> feedforward -> correction[path]
        # --------------------------------------------------------------
        if stage=="generic_endpoint":
            path=self.requests[actor].route_path
            self.log_stage(actor,"generic_endpoint_access",now-duration,now,resource_name)
            self.request_resource(actor,"victim","readout","generic_readout",now,float(self.p27.ENT_READOUT_NS),True)
            return
        if stage=="generic_readout":
            self.log_stage(actor,"generic_bell_readout",now-duration,now,resource_name)
            self.request_resource(actor,"victim","feedforward","generic_feedforward",now,float(self.p27.ENT_FEEDFORWARD_NS),True)
            return
        if stage=="generic_feedforward":
            path=self.requests[actor].route_path
            self.log_stage(actor,"generic_feedforward",now-duration,now,resource_name)
            corr=self.path_resource("correction",path)
            self.request_resource(actor,"victim",corr,"generic_correction",now,float(self.p27.ENT_CORRECTION_NS),True)
            return
        if stage=="generic_correction":
            self.log_stage(actor,"generic_receiver_correction",now-duration,now,resource_name)
            self.requests[actor].completion_ns=float(now)
            self.request_resource(actor,"victim","reset","ent_reset",now,float(self.p27.RESET_NS),True)
            return

        # --------------------------------------------------------------
        # TeleGate: distinct gate-interaction resource and gate correction
        # --------------------------------------------------------------
        if stage=="telegate_endpoint":
            path=self.requests[actor].route_path
            self.log_stage(actor,"telegate_endpoint_access",now-duration,now,resource_name)
            gi=self.path_resource("gate_interaction",path)
            self.request_resource(actor,"victim",gi,"telegate_gate_interaction",now,30.0,True)
            return
        if stage=="telegate_gate_interaction":
            self.log_stage(actor,"telegate_nonlocal_gate_interaction",now-duration,now,resource_name)
            self.request_resource(actor,"victim","readout","telegate_readout",now,70.0,True)
            return
        if stage=="telegate_readout":
            self.log_stage(actor,"telegate_bell_readout",now-duration,now,resource_name)
            self.request_resource(actor,"victim","feedforward","telegate_feedforward",now,40.0,True)
            return
        if stage=="telegate_feedforward":
            path=self.requests[actor].route_path
            self.log_stage(actor,"telegate_feedforward",now-duration,now,resource_name)
            gc=self.path_resource("gate_correction",path)
            self.request_resource(actor,"victim",gc,"telegate_gate_correction",now,20.0,True)
            return
        if stage=="telegate_gate_correction":
            self.log_stage(actor,"telegate_gate_correction",now-duration,now,resource_name)
            self.requests[actor].completion_ns=float(now)
            self.request_resource(actor,"victim","reset","ent_reset",now,120.0,True)
            return

        # --------------------------------------------------------------
        # TeleData: distinct state-load and state-reconstruction resources.
        # --------------------------------------------------------------
        if stage=="teledata_state_load":
            path=self.requests[actor].route_path
            self.log_stage(actor,"teledata_state_load",now-duration,now,resource_name)
            ep=self.path_resource("endpoint",path)
            self.request_resource(actor,"victim",ep,"teledata_endpoint",now,20.0,True)
            return
        if stage=="teledata_endpoint":
            self.log_stage(actor,"teledata_endpoint_access",now-duration,now,resource_name)
            self.request_resource(actor,"victim","readout","teledata_readout",now,70.0,True)
            return
        if stage=="teledata_readout":
            self.log_stage(actor,"teledata_bell_readout",now-duration,now,resource_name)
            self.request_resource(actor,"victim","feedforward","teledata_feedforward",now,40.0,True)
            return
        if stage=="teledata_feedforward":
            path=self.requests[actor].route_path
            self.log_stage(actor,"teledata_feedforward",now-duration,now,resource_name)
            sr=self.path_resource("state_reconstruct",path)
            self.request_resource(actor,"victim",sr,"teledata_state_reconstruct",now,55.0,True)
            return
        if stage=="teledata_state_reconstruct":
            self.log_stage(actor,"teledata_state_reconstruct",now-duration,now,resource_name)
            self.requests[actor].completion_ns=float(now)
            self.request_resource(actor,"victim","reset","ent_reset",now,120.0,True)
            return

        if stage=="ent_reset":
            self.log_stage(actor,"postcompletion_reset",now-duration,now,resource_name)
            self.requests[actor].cleanup_ns=float(now)
            return

        raise RuntimeError(stage)


# =============================================================================
# Request generation and black-box traces
# =============================================================================

def make_attacker_requests(p27,repeat_id,window_ns):
    rel=np.arange(float(p27.ATTACKER_FIRST_RELEASE_NS),window_ns,float(p27.ATTACKER_PERIOD_NS))
    return [RequestState(f"attacker::{repeat_id:02d}::{i}","attacker",float(t),i,PROBE_PATHS[i%len(PROBE_PATHS)]) for i,t in enumerate(rel)]


def make_victim_requests(releases,schedule:BaseSchedule,repeat_id):
    return [RequestState(f"victim::{schedule.base_schedule_id}::{repeat_id:02d}::{i}","victim",float(t),i,PROBE_PATHS[(i+schedule.victim_route_offset)%len(PROBE_PATHS)]) for i,t in enumerate(releases)]


def pair_attacker_trace(attacker_only,combined,attacker_rows,trace_id):
    a=attacker_only[attacker_only.tenant=="attacker"].copy(); c=combined[combined.tenant=="attacker"].copy()
    m=a.merge(c,on="request_index",suffixes=("_attacker_only","_combined"),validate="one_to_one")
    excess=m.turnaround_ns_combined.to_numpy(float)-m.turnaround_ns_attacker_only.to_numpy(float)
    path_map={r.request_index:r.route_path for r in attacker_rows}
    return pd.DataFrame({
        "trace_id":trace_id,"probe_index":m.request_index.astype(int),"probe_path":m.request_index.map(path_map),
        "release_ns":m.release_ns_attacker_only.astype(float),"attacker_only_success":m.success_attacker_only.astype(bool),
        "combined_success":m.success_combined.astype(bool),"attacker_only_completion_ns":m.external_completion_ns_attacker_only.astype(float),
        "combined_completion_ns":m.external_completion_ns_combined.astype(float),"attacker_only_turnaround_ns":m.turnaround_ns_attacker_only.astype(float),
        "combined_turnaround_ns":m.turnaround_ns_combined.astype(float),"excess_turnaround_ns":excess,
        "delayed":excess>AFFECTED_THRESHOLD_NS,"speedup":excess<-AFFECTED_THRESHOLD_NS,
        "failure_transition":m.success_attacker_only.astype(bool).to_numpy()!=m.success_combined.astype(bool).to_numpy(),
    })


def run_lengths(mask):
    out=[]; cur=0
    for v in np.asarray(mask,bool):
        if v: cur+=1
        elif cur: out.append(cur); cur=0
    if cur: out.append(cur)
    return out


def autocorr(x,lag):
    if len(x)<=lag:return 0.0
    a,b=x[:-lag],x[lag:]
    if np.std(a)<EPS or np.std(b)<EPS:return 0.0
    v=float(np.corrcoef(a,b)[0,1]); return v if np.isfinite(v) else 0.0


def safe_corr(a,b):
    n=min(len(a),len(b))
    if n<2:return 0.0
    a,b=a[:n],b[:n]
    if np.std(a)<EPS or np.std(b)<EPS:return 0.0
    v=float(np.corrcoef(a,b)[0,1]); return v if np.isfinite(v) else 0.0


def spectral_features(x):
    if len(x)<4:return {"spectral_dominant_bin_fraction":0.0,"spectral_centroid_fraction":0.0,"spectral_entropy":0.0,"spectral_low_frequency_power_fraction":0.0}
    y=x-np.mean(x); power=np.abs(np.fft.rfft(y))**2
    if len(power):power[0]=0
    total=float(power.sum())
    if total<=EPS:return {"spectral_dominant_bin_fraction":0.0,"spectral_centroid_fraction":0.0,"spectral_entropy":0.0,"spectral_low_frequency_power_fraction":0.0}
    bins=np.arange(len(power),dtype=float); denom=max(len(power)-1,1); probs=power/total; nz=probs[probs>EPS]
    ent=-float(np.sum(nz*np.log(nz)))/(math.log(len(power)) if len(power)>1 else 1.0); cutoff=max(2,int(math.ceil(len(power)*.25)))
    return {"spectral_dominant_bin_fraction":float(np.argmax(power))/denom,"spectral_centroid_fraction":float(np.sum(bins*power)/total)/denom,"spectral_entropy":ent,"spectral_low_frequency_power_fraction":float(np.sum(power[1:cutoff])/total)}


BASE_FEATURES=["probe_count","mean_excess_ns","median_excess_ns","mean_abs_excess_ns","std_excess_ns","max_excess_ns","min_excess_ns","p10_excess_ns","p25_excess_ns","p50_excess_ns","p75_excess_ns","p90_excess_ns","p95_excess_ns","p99_excess_ns","tail_mass_gt_p90_fraction","tail_mass_gt_2x_median_abs_fraction","delayed_fraction","speedup_fraction","failure_transition_fraction","cumulative_positive_excess_ns","cumulative_negative_magnitude_ns","cumulative_abs_excess_ns","longest_delayed_run","longest_speedup_run","delayed_run_count","speedup_run_count","mean_delayed_run_length","mean_speedup_run_length","long_gap_frequency","mean_inter_affected_gap_probes","std_inter_affected_gap_probes","inferred_busy_period_ns","lag1_autocorrelation","lag2_autocorrelation","lag3_autocorrelation","spectral_dominant_bin_fraction","spectral_centroid_fraction","spectral_entropy","spectral_low_frequency_power_fraction","early_mean_abs_ns","middle_mean_abs_ns","late_mean_abs_ns","cross_path_corr_mean","cross_path_corr_max","cross_path_corr_min"]
PATH_FEATURES=[]
for path in PROBE_PATHS:
    PATH_FEATURES += [f"{path}__mean_excess_ns",f"{path}__mean_abs_excess_ns",f"{path}__delayed_fraction",f"{path}__speedup_fraction",f"{path}__cumulative_abs_excess_ns",f"{path}__cumulative_positive_excess_ns",f"{path}__p95_abs_excess_ns"]
FEATURE_COLUMNS=BASE_FEATURES+PATH_FEATURES


def extract_features(trace,attacker_period_ns,probe_budget=None):
    t=trace.sort_values("probe_index")
    if probe_budget is not None:t=t.head(int(probe_budget))
    x=t.excess_turnaround_ns.to_numpy(float); ax=np.abs(x); delayed=x>AFFECTED_THRESHOLD_NS; speedup=x<-AFFECTED_THRESHOLD_NS; affected=ax>AFFECTED_THRESHOLD_NS; fail=t.failure_transition.to_numpy(bool)
    dr,sr,ar=run_lengths(delayed),run_lengths(speedup),run_lengths(affected); idx=np.flatnonzero(affected); gaps=np.diff(idx).astype(float) if len(idx)>=2 else np.array([],float)
    med=float(np.median(ax)) if len(ax) else 0.0; p90=float(np.quantile(ax,.9)) if len(ax) else 0.0; q=lambda z:float(np.quantile(x,z)) if len(x) else 0.0
    thirds=np.array_split(np.arange(len(x)),3); tm=[float(np.mean(ax[z])) if len(z) else 0.0 for z in thirds]
    arrays={}; pf={}
    for path in PROBE_PATHS:
        vals=t.loc[t.probe_path==path,"excess_turnaround_ns"].to_numpy(float); arrays[path]=vals; av=np.abs(vals)
        pf.update({f"{path}__mean_excess_ns":float(np.mean(vals)) if len(vals) else 0.0,f"{path}__mean_abs_excess_ns":float(np.mean(av)) if len(vals) else 0.0,f"{path}__delayed_fraction":float(np.mean(vals>AFFECTED_THRESHOLD_NS)) if len(vals) else 0.0,f"{path}__speedup_fraction":float(np.mean(vals<-AFFECTED_THRESHOLD_NS)) if len(vals) else 0.0,f"{path}__cumulative_abs_excess_ns":float(np.sum(av)),f"{path}__cumulative_positive_excess_ns":float(np.sum(np.maximum(vals,0))),f"{path}__p95_abs_excess_ns":float(np.quantile(av,.95)) if len(vals) else 0.0})
    corrs=np.array([safe_corr(arrays[PROBE_PATHS[i]],arrays[PROBE_PATHS[j]]) for i in range(3) for j in range(i+1,3)],float)
    row={"probe_count":float(len(x)),"mean_excess_ns":float(np.mean(x)) if len(x) else 0.0,"median_excess_ns":float(np.median(x)) if len(x) else 0.0,"mean_abs_excess_ns":float(np.mean(ax)) if len(x) else 0.0,"std_excess_ns":float(np.std(x)) if len(x) else 0.0,"max_excess_ns":float(np.max(x)) if len(x) else 0.0,"min_excess_ns":float(np.min(x)) if len(x) else 0.0,"p10_excess_ns":q(.1),"p25_excess_ns":q(.25),"p50_excess_ns":q(.5),"p75_excess_ns":q(.75),"p90_excess_ns":q(.9),"p95_excess_ns":q(.95),"p99_excess_ns":q(.99),"tail_mass_gt_p90_fraction":float(np.mean(ax>p90)) if len(ax) else 0.0,"tail_mass_gt_2x_median_abs_fraction":float(np.mean(ax>2*med)) if len(ax) and med>0 else 0.0,"delayed_fraction":float(np.mean(delayed)) if len(x) else 0.0,"speedup_fraction":float(np.mean(speedup)) if len(x) else 0.0,"failure_transition_fraction":float(np.mean(fail)) if len(x) else 0.0,"cumulative_positive_excess_ns":float(np.sum(np.maximum(x,0))),"cumulative_negative_magnitude_ns":float(np.sum(np.maximum(-x,0))),"cumulative_abs_excess_ns":float(np.sum(ax)),"longest_delayed_run":float(max(dr) if dr else 0),"longest_speedup_run":float(max(sr) if sr else 0),"delayed_run_count":float(len(dr)),"speedup_run_count":float(len(sr)),"mean_delayed_run_length":float(np.mean(dr)) if dr else 0.0,"mean_speedup_run_length":float(np.mean(sr)) if sr else 0.0,"long_gap_frequency":float(np.mean(gaps>=3)) if len(gaps) else 0.0,"mean_inter_affected_gap_probes":float(np.mean(gaps)) if len(gaps) else 0.0,"std_inter_affected_gap_probes":float(np.std(gaps)) if len(gaps) else 0.0,"inferred_busy_period_ns":float(max(ar)*attacker_period_ns) if ar else 0.0,"lag1_autocorrelation":autocorr(x,1),"lag2_autocorrelation":autocorr(x,2),"lag3_autocorrelation":autocorr(x,3),"early_mean_abs_ns":tm[0],"middle_mean_abs_ns":tm[1],"late_mean_abs_ns":tm[2],"cross_path_corr_mean":float(np.mean(corrs)) if len(corrs) else 0.0,"cross_path_corr_max":float(np.max(corrs)) if len(corrs) else 0.0,"cross_path_corr_min":float(np.min(corrs)) if len(corrs) else 0.0}
    row.update(spectral_features(x)); row.update(pf); return row


# =============================================================================
# Dataset generation
# =============================================================================

def victim_slowdown(victim_only,combined):
    v=victim_only[victim_only.tenant=="victim"].copy(); c=combined[combined.tenant=="victim"].copy()
    if v.empty or c.empty:return {"victim_mean_request_slowdown":1.0,"victim_makespan_slowdown":1.0}
    m=v[["request_index","turnaround_ns"]].merge(c[["request_index","turnaround_ns"]],on="request_index",suffixes=("_v","_c"))
    a=m.turnaround_ns_v.to_numpy(float); b=m.turnaround_ns_c.to_numpy(float); good=np.isfinite(a)&np.isfinite(b)&(a>0)
    req=float(np.mean(b[good]/a[good])) if np.any(good) else 1.0
    gv=v[np.isfinite(v.external_completion_ns)]; gc=c[np.isfinite(c.external_completion_ns)]
    if gv.empty or gc.empty:make=1.0
    else:
        start=float(v.release_ns.min()); ba=float(gv.external_completion_ns.max()-start); bc=float(gc.external_completion_ns.max()-start); make=bc/ba if ba>0 else 1.0
    return {"victim_mean_request_slowdown":req,"victim_makespan_slowdown":make}


def simulate_dataset(p27,schedules,conditions,link_probs,repeats,window_ns,seed):
    trace_parts=[]; feat_rows=[]; truth_rows=[]; trial_rows=[]; retry_parts=[]; dist_parts=[]; eval_rows=[]; rel_rows=[]; stage_parts=[]
    total=len(schedules)*len(conditions)*len(link_probs)*repeats; done=0
    attacker_cache={}; base_cond=conditions[0]
    for r in range(repeats):
        ars=make_attacker_requests(p27,r,window_ns); sim=Simulator(p27,base_cond,1.0,seed,"attacker_only",r,"attacker_only")
        req,*_=sim.run([clone_request_state(x) for x in ars]); attacker_cache[r]=(req,ars)
    for s in schedules:
        for r in range(repeats):
            releases=schedule_releases(s,r,seed,window_ns); ars=make_attacker_requests(p27,r,window_ns); vrs=make_victim_requests(releases,s,r)
            for vr in vrs: rel_rows.append({"base_schedule_id":s.base_schedule_id,"repeat_id":r,"schedule_profile":s.schedule_profile,"victim_request_index":vr.request_index,"route_path":vr.route_path,"release_ns":vr.release_ns})
            for lp in link_probs:
                for cond in conditions:
                    sv=Simulator(p27,cond,lp,seed,s.base_schedule_id,r,"victim_only")
                    vreq,vwait,vint,vstage,vretry,vdist,vpool=sv.run([clone_request_state(x) for x in vrs])
                    sc=Simulator(p27,cond,lp,seed,s.base_schedule_id,r,"combined")
                    creq,cwait,cint,cstage,cretry,cdist,cpool=sc.run([clone_request_state(x) for x in ars+vrs])
                    tid=hashlib.sha256(f"{s.base_schedule_id}|{r}|{lp:.6f}|{cond.condition_id}|{seed}".encode()).hexdigest()[:20]
                    tr=pair_attacker_trace(attacker_cache[r][0],creq,ars,tid); trace_parts.append(tr)
                    f=extract_features(tr,float(p27.ATTACKER_PERIOD_NS)); feat_rows.append({"trace_id":tid,"base_schedule_id":s.base_schedule_id,"repeat_id":r,"link_success_probability":lp,**f})
                    vrtry=cretry[(cretry.tenant=="victim")] if not cretry.empty and "tenant" in cretry else pd.DataFrame(); attempts=len(vrtry); failed=int((~vrtry.success.astype(bool)).sum()) if len(vrtry) else 0
                    vs=float(creq.loc[creq.tenant=="victim","success"].mean()) if len(vrs) else 1.0; slow=victim_slowdown(vreq,creq)
                    truth={"trace_id":tid,"base_schedule_id":s.base_schedule_id,"repeat_id":r,"schedule_profile":s.schedule_profile,"link_success_probability":lp,"condition_id":cond.condition_id,"task_name":cond.task_name,"evaluator_label":cond.evaluator_label,"runtime_mode":cond.runtime_mode,"post_pipeline":cond.post_pipeline,"distillation_depth":cond.distillation_depth,"retry_limit":cond.retry_limit,"uses_prefetch":cond.uses_prefetch,"victim_route_offset":s.victim_route_offset,"victim_remote_operation_count":len(vrs),"victim_success_fraction":vs,"actual_generation_attempt_count":attempts,"actual_retry_count":failed,**slow}
                    truth_rows.append(truth); trial_rows.append({**truth,**f})
                    if not cretry.empty: z=cretry.copy(); z["trace_id"]=tid; retry_parts.append(z)
                    if not cdist.empty: z=cdist.copy(); z["trace_id"]=tid; dist_parts.append(z)
                    if not cstage.empty: z=cstage.copy(); z["trace_id"]=tid; z["condition_id"]=cond.condition_id; stage_parts.append(z)
                    eval_rows.append({"trace_id":tid,"condition_id":cond.condition_id,"task_name":cond.task_name,"evaluator_label":cond.evaluator_label,"link_success_probability":lp,"combined_wait_events":len(cwait),"combined_intervals":len(cint),"combined_stage_records":len(cstage),"combined_pool_events":len(cpool),"actual_generation_attempt_count":attempts,"actual_retry_count":failed,"distillation_event_count":len(cdist)})
                    done+=1
                    if done%max(1,total//20)==0 or done==total: print(f"[Phase 3.5.1] Generated {done}/{total} traces")
    return pd.concat(trace_parts,ignore_index=True),pd.DataFrame(feat_rows),pd.DataFrame(truth_rows),pd.DataFrame(trial_rows),(pd.concat(retry_parts,ignore_index=True) if retry_parts else pd.DataFrame()),(pd.concat(dist_parts,ignore_index=True) if dist_parts else pd.DataFrame()),pd.DataFrame(eval_rows),pd.DataFrame(rel_rows).drop_duplicates(),(pd.concat(stage_parts,ignore_index=True) if stage_parts else pd.DataFrame())


# =============================================================================
# ML helpers
# =============================================================================

def build_group_split(schedules,seed,test_size):
    tab=pd.DataFrame([{"base_schedule_id":s.base_schedule_id,"schedule_profile":s.schedule_profile} for s in schedules]); rng=np.random.default_rng(stable_seed(seed,"split")); test_ids=set()
    for profile,g in tab.groupby("schedule_profile",sort=True):
        ids=g.base_schedule_id.to_numpy().copy(); rng.shuffle(ids); n=max(1,min(len(ids)-1,int(round(len(ids)*test_size)))); test_ids.update(ids[:n].tolist())
    tab["split"]=tab.base_schedule_id.map(lambda x:"test" if x in test_ids else "train"); return tab.sort_values(["schedule_profile","base_schedule_id"]).reset_index(drop=True)


def classifiers(seed,trees):
    return {"logistic_regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=4000,class_weight="balanced",random_state=seed))]),"random_forest":RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1),"hist_gradient_boosting":HistGradientBoostingClassifier(random_state=seed,max_iter=250,l2_regularization=1e-3)}


def regressors(seed,trees):
    return {"random_forest":RandomForestRegressor(n_estimators=trees,random_state=seed,n_jobs=-1),"hist_gradient_boosting":HistGradientBoostingRegressor(random_state=seed,max_iter=250,l2_regularization=1e-3),"elastic_net":Pipeline([("scale",StandardScaler()),("model",ElasticNet(alpha=.01,l1_ratio=.25,random_state=seed,max_iter=10000))])}


def evaluate_classification(analysis,split,seed,trees):
    sm=split.set_index("base_schedule_id").split; d=analysis.copy(); d["split"]=d.base_schedule_id.map(sm); mets=[]; preds=[]; cms=[]
    for task,g in d.groupby("task_name",sort=True):
        tr=g[g.split=="train"]; te=g[g.split=="test"]; labels=sorted(g.evaluator_label.unique()); Xtr=tr[FEATURE_COLUMNS].astype(float).to_numpy(); Xte=te[FEATURE_COLUMNS].astype(float).to_numpy(); ytr=tr.evaluator_label.astype(str).to_numpy(); yte=te.evaluator_label.astype(str).to_numpy()
        for name,m in classifiers(seed,trees).items():
            m.fit(Xtr,ytr); yp=m.predict(Xte); auc=math.nan
            if len(labels)==2 and hasattr(m,"predict_proba") and len(np.unique(yte))==2:
                classes=list(m.classes_); pos=labels[-1]; auc=float(roc_auc_score((yte==pos).astype(int),m.predict_proba(Xte)[:,classes.index(pos)]))
            mets.append({"task_name":task,"model_name":name,"sample_count":len(te),"class_count":len(labels),"chance_accuracy":1/len(labels),"accuracy":accuracy_score(yte,yp),"balanced_accuracy":balanced_accuracy_score(yte,yp),"macro_f1":f1_score(yte,yp,average="macro",zero_division=0),"binary_roc_auc":auc})
            ter=te.reset_index(drop=True)
            for i,row in ter.iterrows(): preds.append({"trace_id":row.trace_id,"base_schedule_id":row.base_schedule_id,"repeat_id":int(row.repeat_id),"schedule_profile":row.schedule_profile,"link_success_probability":float(row.link_success_probability),"task_name":task,"true_label":yte[i],"predicted_label":yp[i],"correct":bool(yte[i]==yp[i]),"model_name":name})
            cm=confusion_matrix(yte,yp,labels=labels)
            for i,tl in enumerate(labels):
                den=cm[i].sum()
                for j,pl in enumerate(labels): cms.append({"task_name":task,"model_name":name,"true_label":tl,"predicted_label":pl,"count":int(cm[i,j]),"true_normalized_fraction":float(cm[i,j]/den) if den else 0.0})
    return pd.DataFrame(mets),pd.DataFrame(preds),pd.DataFrame(cms)


def evaluate_regression(analysis,split,seed,trees):
    sm=split.set_index("base_schedule_id").split; d=analysis.copy(); d["split"]=d.base_schedule_id.map(sm); mets=[]; preds=[]
    for task,source,target in [("distillation_depth_estimation",TASK_DISTILL,"distillation_depth"),("retry_count_estimation",TASK_RETRY,"actual_retry_count")]:
        g=d[d.task_name==source]; tr=g[g.split=="train"]; te=g[g.split=="test"]; Xtr=tr[FEATURE_COLUMNS].astype(float).to_numpy(); Xte=te[FEATURE_COLUMNS].astype(float).to_numpy(); ytr=tr[target].astype(float).to_numpy(); yte=te[target].astype(float).to_numpy()
        for name,m in regressors(seed,trees).items():
            m.fit(Xtr,ytr); yp=np.asarray(m.predict(Xte),float); mets.append({"task_name":task,"source_task":source,"target":target,"model_name":name,"sample_count":len(te),"mae":mean_absolute_error(yte,yp),"rmse":math.sqrt(mean_squared_error(yte,yp)),"r2":r2_score(yte,yp) if len(np.unique(yte))>1 else math.nan})
            ter=te.reset_index(drop=True)
            for i,row in ter.iterrows(): preds.append({"trace_id":row.trace_id,"base_schedule_id":row.base_schedule_id,"repeat_id":int(row.repeat_id),"schedule_profile":row.schedule_profile,"link_success_probability":float(row.link_success_probability),"task_name":task,"target":target,"true_value":float(yte[i]),"predicted_value":float(yp[i]),"absolute_error":float(abs(yp[i]-yte[i])),"model_name":name})
    return pd.DataFrame(mets),pd.DataFrame(preds)


def holdout_generalization(analysis,axis,seed,trees):
    rows=[]
    for task,g in analysis.groupby("task_name",sort=True):
        labels=sorted(g.evaluator_label.unique())
        for held in sorted(g[axis].unique()):
            mask=np.isclose(g[axis],held) if axis=="link_success_probability" else (g[axis]==held); tr=g[~mask]; te=g[mask]
            m=RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1); m.fit(tr[FEATURE_COLUMNS].astype(float),tr.evaluator_label.astype(str)); yp=m.predict(te[FEATURE_COLUMNS].astype(float))
            rows.append({"task_name":task,"model_name":"random_forest",f"held_out_{axis}":held,"sample_count":len(te),"chance_accuracy":1/len(labels),"accuracy":accuracy_score(te.evaluator_label.astype(str),yp),"balanced_accuracy":balanced_accuracy_score(te.evaluator_label.astype(str),yp),"macro_f1":f1_score(te.evaluator_label.astype(str),yp,average="macro",zero_division=0)})
    return pd.DataFrame(rows)


def probe_budget_metrics(traces,truth,split,p27,budgets,seed,trees):
    ti=truth.set_index("trace_id"); sm=split.set_index("base_schedule_id").split; rows=[]
    for budget in budgets:
        fr=[]
        for tid,tr in traces.groupby("trace_id",sort=False):
            meta=ti.loc[tid]; fr.append({"trace_id":tid,"base_schedule_id":meta.base_schedule_id,"task_name":meta.task_name,"evaluator_label":meta.evaluator_label,**extract_features(tr,float(p27.ATTACKER_PERIOD_NS),budget)})
        d=pd.DataFrame(fr); d["split"]=d.base_schedule_id.map(sm)
        for task,g in d.groupby("task_name",sort=True):
            tr=g[g.split=="train"]; te=g[g.split=="test"]; labels=sorted(g.evaluator_label.unique()); m=RandomForestClassifier(n_estimators=trees,class_weight="balanced",random_state=seed,n_jobs=-1); m.fit(tr[FEATURE_COLUMNS].astype(float),tr.evaluator_label.astype(str)); yp=m.predict(te[FEATURE_COLUMNS].astype(float))
            rows.append({"task_name":task,"model_name":"random_forest","probe_budget":budget,"sample_count":len(te),"chance_accuracy":1/len(labels),"accuracy":accuracy_score(te.evaluator_label.astype(str),yp),"balanced_accuracy":balanced_accuracy_score(te.evaluator_label.astype(str),yp),"macro_f1":f1_score(te.evaluator_label.astype(str),yp,average="macro",zero_division=0)})
    return pd.DataFrame(rows)


def best_summary(metrics):
    rows=[]
    for task,g in metrics.groupby("task_name",sort=True):
        b=g.sort_values(["accuracy","macro_f1"],ascending=False).iloc[0]; rows.append({"task_name":task,"best_model":b.model_name,"class_count":int(b.class_count),"chance_accuracy":float(b.chance_accuracy),"best_accuracy":float(b.accuracy),"best_balanced_accuracy":float(b.balanced_accuracy),"best_macro_f1":float(b.macro_f1),"best_binary_roc_auc":float(b.binary_roc_auc) if np.isfinite(b.binary_roc_auc) else math.nan})
    return pd.DataFrame(rows)


def signal_summary(trials):
    return trials.groupby(["task_name","evaluator_label","schedule_profile","link_success_probability"],sort=True).agg(trace_count=("trace_id","count"),mean_abs_excess_ns=("mean_abs_excess_ns","mean"),mean_signed_excess_ns=("mean_excess_ns","mean"),delayed_fraction=("delayed_fraction","mean"),speedup_fraction=("speedup_fraction","mean"),p95_excess_ns=("p95_excess_ns","mean"),inferred_busy_period_ns=("inferred_busy_period_ns","mean"),actual_retry_count=("actual_retry_count","mean"),victim_success_fraction=("victim_success_fraction","mean")).reset_index()


# =============================================================================
# Methodology tables and validation
# =============================================================================

def protocol_resource_signature_table() -> pd.DataFrame:
    rows = [
        {"protocol_class":"direct_transfer","epr_mode":"none","ordered_stage_sequence":"endpoint[path] -> switch_path[path] -> quantum_link[path] -> receiver_gate[path] -> completion -> reset","distinctive_resources":"switch_path[path]|quantum_link[path]|receiver_gate[path]"},
        {"protocol_class":"on_demand_epr","epr_mode":"on_demand","ordered_stage_sequence":"epr_generator -> quantum_link[path] -> endpoint[path] -> readout -> feedforward -> correction[path] -> completion -> reset","distinctive_resources":"epr_generator|quantum_link[path]|correction[path]"},
        {"protocol_class":"prefetched_epr","epr_mode":"route_local_prefetch","ordered_stage_sequence":"EPR_pool[path] -> endpoint[path] -> readout -> feedforward -> correction[path] -> completion -> reset; async refill uses epr_generator + quantum_link[path]","distinctive_resources":"EPR_pool[path]|async_refill|correction[path]"},
        {"protocol_class":"telegate","epr_mode":"route_local_prefetch","ordered_stage_sequence":"EPR_pool[path] -> endpoint[path] -> gate_interaction[path] -> readout -> feedforward -> gate_correction[path] -> completion -> reset","distinctive_resources":"gate_interaction[path]|gate_correction[path]"},
        {"protocol_class":"teledata","epr_mode":"route_local_prefetch","ordered_stage_sequence":"EPR_pool[path] -> state_load[path] -> endpoint[path] -> readout -> feedforward -> state_reconstruct[path] -> completion -> reset","distinctive_resources":"state_load[path]|state_reconstruct[path]"},
    ]
    return pd.DataFrame(rows)


def service_factor_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"service_class":"low_latency","runtime_mode":"custom_epr","post_pipeline":"generic_epr","retry_limit":4,"distillation_depth":0,"raw_pairs_required":1,"controlled_difference":"distillation depth / required raw-pair count only"},
        {"service_class":"high_fidelity","runtime_mode":"custom_epr","post_pipeline":"generic_epr","retry_limit":4,"distillation_depth":2,"raw_pairs_required":4,"controlled_difference":"distillation depth / required raw-pair count only"},
    ])


def path_resource_table() -> pd.DataFrame:
    rows=[]
    for path in PROBE_PATHS:
        rows.append({"probe_path":path,"attacker_knows_path":True,"path_specific_resources":"|".join([f"endpoint::{path}",f"switch_path::{path}",f"quantum_link::{path}",f"receiver_gate::{path}",f"correction::{path}",f"gate_interaction::{path}",f"gate_correction::{path}",f"state_load::{path}",f"state_reconstruct::{path}"]),"common_cross_path_resources":"epr_generator|readout|feedforward|reset|distillation"})
    return pd.DataFrame(rows)


def build_validations(p27,traces,features,truth,split,schedules,conditions,link_probs,releases,stages):
    rows=[]
    def add(group,name,passed,expected,observed,details=""):
        rows.append({"validation_group":group,"assertion_name":name,"passed":bool(passed),"expected":str(expected),"observed":str(observed),"details":details})
    add("blackbox","attacker_trace_schema_exact",list(traces.columns)==ATTACKER_VISIBLE_COLUMNS,ATTACKER_VISIBLE_COLUMNS,list(traces.columns))
    forbidden=("victim","protocol","service","fidelity","distill","retry","resource","epr","label","condition","schedule_profile","link_success_probability","route_offset")
    bad=[c for c in FEATURE_COLUMNS if any(tok in c.lower() for tok in forbidden)]
    add("blackbox","features_exclude_evaluator_state",not bad,[],bad)
    encoded=traces.trace_id.astype(str).str.contains("direct|epr|tele|distill|retry|fidelity|service|victim",case=False,regex=True)
    add("blackbox","trace_ids_are_opaque",not bool(encoded.any()),0,int(encoded.sum()))
    add("blackbox","attacker_probe_path_is_self_chosen_observable",set(traces.probe_path.unique())==set(PROBE_PATHS),set(PROBE_PATHS),set(traces.probe_path.unique()))

    add("architecture","phase2_07_direct_timing_retained",abs(float(p27.DIRECT_CRITICAL_NS)-150)<1e-9 and abs(float(p27.RESET_NS)-120)<1e-9,"critical=150 ns, cleanup=120 ns",f"critical={p27.DIRECT_CRITICAL_NS}, cleanup={p27.RESET_NS}")
    labels={t:set(g.evaluator_label) for t,g in truth.groupby("task_name")}
    add("design","protocol_classes_complete",labels.get(TASK_PROTOCOL,set())=={"direct_transfer","on_demand_epr","prefetched_epr","telegate","teledata"},"5 requested classes",labels.get(TASK_PROTOCOL,set()))
    add("design","distillation_classes_complete",labels.get(TASK_DISTILL,set())=={"distill_0","distill_1","distill_2"},"distill_0/1/2",labels.get(TASK_DISTILL,set()))
    add("design","retry_classes_complete",labels.get(TASK_RETRY,set())=={"retry_1","retry_2","retry_4"},"retry_1/2/4",labels.get(TASK_RETRY,set()))
    add("design","service_classes_complete",labels.get(TASK_SERVICE,set())=={"low_latency","high_fidelity"},"low_latency/high_fidelity",labels.get(TASK_SERVICE,set()))

    # Service methodology: exactly one semantic axis is changed.
    service=[c for c in conditions if c.task_name==TASK_SERVICE]
    low=next(c for c in service if c.evaluator_label=="low_latency"); high=next(c for c in service if c.evaluator_label=="high_fidelity")
    same_runtime=(low.runtime_mode==high.runtime_mode and low.post_pipeline==high.post_pipeline and low.retry_limit==high.retry_limit and low.uses_prefetch==high.uses_prefetch)
    add("methodology","service_classes_match_runtime_retry_pipeline",same_runtime,"all non-distillation fields identical",{"runtime_mode":(low.runtime_mode,high.runtime_mode),"post_pipeline":(low.post_pipeline,high.post_pipeline),"retry_limit":(low.retry_limit,high.retry_limit),"uses_prefetch":(low.uses_prefetch,high.uses_prefetch)})
    add("methodology","service_classes_differ_only_in_distillation_depth",same_runtime and low.distillation_depth==0 and high.distillation_depth==2,"depth 0 vs 2",f"{low.distillation_depth} vs {high.distillation_depth}")

    # TeleGate and TeleData must have distinct actual evaluator stage/resource signatures.
    if not stages.empty:
        prot_truth=truth[truth.task_name==TASK_PROTOCOL][["trace_id","evaluator_label"]]
        ss=stages.merge(prot_truth,on="trace_id",how="inner")
        tg=set(ss[ss.evaluator_label=="telegate"].stage.unique()); td=set(ss[ss.evaluator_label=="teledata"].stage.unique())
        tg_res=set(ss[ss.evaluator_label=="telegate"].resource_name.dropna().unique()); td_res=set(ss[ss.evaluator_label=="teledata"].resource_name.dropna().unique())
    else:
        tg=td=tg_res=td_res=set()
    add("methodology","telegate_teledata_stage_sequences_are_distinct",tg!=td and "telegate_nonlocal_gate_interaction" in tg and "teledata_state_load" in td and "teledata_state_reconstruct" in td,"distinct causal stage sets",{"telegate":sorted(tg),"teledata":sorted(td)})
    add("methodology","telegate_teledata_use_distinct_path_resources",tg_res!=td_res and any("gate_interaction::" in x for x in tg_res) and any("state_load::" in x for x in td_res) and any("state_reconstruct::" in x for x in td_res),"distinct gate vs state-transfer resources",{"telegate":sorted(tg_res),"teledata":sorted(td_res)})

    # Physical path semantics.
    stage_resources=set(stages.resource_name.dropna().astype(str).unique()) if not stages.empty else set()
    for path in PROBE_PATHS:
        add("methodology",f"physical_resources_exist_for_{path}",f"endpoint::{path}" in stage_resources and f"quantum_link::{path}" in stage_resources,"path-specific endpoint and quantum link present",sorted([x for x in stage_resources if path in x]))
    pc=traces.groupby(["trace_id","probe_path"]).size().unstack(fill_value=0); md=int((pc.max(axis=1)-pc.min(axis=1)).max())
    add("probe_policy","balanced_attacker_probe_paths",md<=1,"max count difference <=1",md)
    vc=releases.groupby(["base_schedule_id","repeat_id","route_path"]).size().unstack(fill_value=0); vdiff=int((vc.max(axis=1)-vc.min(axis=1)).max()) if len(vc) else 0
    add("probe_policy","balanced_victim_route_assignment",vdiff<=1,"max route-count difference <=1",vdiff)

    # Crossing and split discipline.
    cross=truth.groupby(["base_schedule_id","repeat_id","link_success_probability"]).condition_id.nunique(); add("schedule","all_conditions_crossed_with_every_schedule_probability",cross.min()==len(conditions) and cross.max()==len(conditions),len(conditions),f"min={cross.min()},max={cross.max()}")
    ops=truth.groupby(["base_schedule_id","repeat_id","link_success_probability"]).victim_remote_operation_count.nunique(); add("schedule","logical_request_count_condition_independent",ops.max()==1,1,int(ops.max()))
    route_sig=(releases.sort_values(["base_schedule_id","repeat_id","victim_request_index"]).groupby(["base_schedule_id","repeat_id"],sort=False)["route_path"].agg(tuple))
    add("schedule","victim_route_schedule_defined_once_per_schedule_repeat",len(route_sig)==releases[["base_schedule_id","repeat_id"]].drop_duplicates().shape[0],"one route signature per schedule/repeat",len(route_sig))
    allp={"sparse_periodic","dense_periodic","synchronization_bursty"}; add("schedule","three_workload_profiles_present",set(s.schedule_profile for s in schedules)==allp,allp,set(s.schedule_profile for s in schedules))
    add("schedule","all_link_probabilities_present",set(np.round(truth.link_success_probability.unique(),12))==set(np.round(link_probs,12)),link_probs,sorted(truth.link_success_probability.unique()))
    train=set(split.loc[split.split=="train","base_schedule_id"]); test=set(split.loc[split.split=="test","base_schedule_id"]); add("evaluation","no_group_overlap",not(train&test),[],sorted(train&test))
    trp=set(split.loc[split.split=="train","schedule_profile"]); tep=set(split.loc[split.split=="test","schedule_profile"]); add("evaluation","all_profiles_in_train_and_test",trp==allp and tep==allp,allp,f"train={trp},test={tep}")
    add("evaluation","features_truth_one_to_one",len(features)==len(truth)==features.trace_id.nunique()==truth.trace_id.nunique(),len(truth),f"features={len(features)},truth={len(truth)}")

    # Negative controls.
    merged=features.merge(truth[["trace_id","base_schedule_id","repeat_id","link_success_probability","task_name","evaluator_label"]],on=["trace_id","base_schedule_id","repeat_id","link_success_probability"])
    for task,name in [(TASK_PROTOCOL_CONTROL,"protocol_label_control_identical"),(TASK_SERVICE_CONTROL,"service_label_control_identical")]:
        spans=[]
        for _,g in merged[merged.task_name==task].groupby(["base_schedule_id","repeat_id","link_success_probability"]):
            arr=g[FEATURE_COLUMNS].astype(float).to_numpy(); spans.append(float(np.max(np.ptp(arr,axis=0))) if len(arr) else 0)
        ms=max(spans) if spans else math.inf; add("negative_control",name,ms<=1e-9,"<=1e-9",ms)

    dmap=truth[truth.task_name==TASK_DISTILL].groupby("evaluator_label").distillation_depth.first().to_dict(); add("mechanism","distillation_depth_mapping",dmap=={"distill_0":0,"distill_1":1,"distill_2":2},{"distill_0":0,"distill_1":1,"distill_2":2},dmap)
    rmap=truth[truth.task_name==TASK_RETRY].groupby("evaluator_label").retry_limit.first().to_dict(); add("mechanism","retry_limit_mapping",rmap=={"retry_1":1,"retry_2":2,"retry_4":4},{"retry_1":1,"retry_2":2,"retry_4":4},rmap)
    add("execution","all_attacker_only_requests_succeed",bool(traces.attacker_only_success.all()),True,bool(traces.attacker_only_success.all()))
    add("execution","combined_attacker_completion_finite",bool(np.isfinite(traces.combined_completion_ns).all()),True,bool(np.isfinite(traces.combined_completion_ns).all()))
    return pd.DataFrame(rows)


# =============================================================================
# Driver / outputs
# =============================================================================

def run_experiment(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    p27,source=load_phase2_07_module(); schedules=build_base_schedules(args.seed,args.base_schedules); conditions=build_conditions(); link_probs=parse_probability_list(args.link_success_probabilities)
    print(f"[Phase 3.5.1] schedules={len(schedules)}, conditions={len(conditions)}, link_probs={link_probs}, repeats={args.repeats_per_schedule}")
    print(f"[Phase 3.5.1] Reusing Phase-2.7 timing constants from: {source}")
    print("[Phase 3.5.1] Refined TeleGate/TeleData pipelines, controlled service comparison, and physical probe paths enabled.")
    traces,features,truth,trials,retry_log,dist_log,evaluator,releases,stages=simulate_dataset(p27,schedules,conditions,link_probs,args.repeats_per_schedule,args.observation_window_ns,args.seed)
    split=build_group_split(schedules,args.seed,args.test_size); analysis=features.merge(truth,on=["trace_id","base_schedule_id","repeat_id","link_success_probability"],validate="one_to_one")
    metrics,preds,cm=evaluate_classification(analysis,split,args.seed,args.rf_trees); regm,regp=evaluate_regression(analysis,split,args.seed,args.rf_trees)
    wg=holdout_generalization(analysis,"schedule_profile",args.seed,args.rf_trees); lg=holdout_generalization(analysis,"link_success_probability",args.seed,args.rf_trees)
    maxp=int(traces.groupby("trace_id").probe_index.count().max()); budgets=sorted({min(maxp,int(x)) for x in args.probe_budgets.split(",") if x.strip() and int(x)>0});
    if maxp not in budgets:budgets.append(maxp)
    pb=probe_budget_metrics(traces,truth,split,p27,budgets,args.seed,args.rf_trees); summary=best_summary(metrics); signal=signal_summary(trials)
    val=build_validations(p27,traces,features,truth,split,schedules,conditions,link_probs,releases,stages); vals=pd.DataFrame([{"assertion_count":len(val),"passed_assertions":int(val.passed.sum()),"failed_assertions":int((~val.passed).sum()),"all_passed":bool(val.passed.all())}])

    prefix="phase3_05_1"
    traces.to_csv(out/f"{prefix}_attacker_visible_trace.csv",index=False); features.to_csv(out/f"{prefix}_trace_features.csv",index=False); truth.to_csv(out/f"{prefix}_evaluator_ground_truth.csv",index=False); trials.to_csv(out/f"{prefix}_trial_summary.csv",index=False)
    evaluator.to_csv(out/f"{prefix}_evaluator_mechanism_summary.csv",index=False); releases.to_csv(out/f"{prefix}_victim_release_schedule.csv",index=False); stages.to_csv(out/f"{prefix}_stage_records_evaluator.csv.gz",index=False,compression="gzip")
    pd.DataFrame([asdict(x) for x in schedules]).to_csv(out/f"{prefix}_base_schedule_table.csv",index=False); pd.DataFrame([asdict(x) for x in conditions]).to_csv(out/f"{prefix}_condition_table.csv",index=False)
    protocol_resource_signature_table().to_csv(out/f"{prefix}_protocol_resource_signatures.csv",index=False); service_factor_table().to_csv(out/f"{prefix}_service_factor_table.csv",index=False); path_resource_table().to_csv(out/f"{prefix}_path_resource_table.csv",index=False)
    split.to_csv(out/f"{prefix}_group_split.csv",index=False); metrics.to_csv(out/f"{prefix}_inference_metrics.csv",index=False); preds.to_csv(out/f"{prefix}_inference_predictions.csv",index=False); cm.to_csv(out/f"{prefix}_confusion_matrix.csv",index=False)
    regm.to_csv(out/f"{prefix}_regression_metrics.csv",index=False); regp.to_csv(out/f"{prefix}_regression_predictions.csv",index=False); wg.to_csv(out/f"{prefix}_workload_generalization_metrics.csv",index=False); lg.to_csv(out/f"{prefix}_link_success_generalization_metrics.csv",index=False); pb.to_csv(out/f"{prefix}_probe_budget_metrics.csv",index=False); summary.to_csv(out/f"{prefix}_protocol_fidelity_summary.csv",index=False); signal.to_csv(out/f"{prefix}_signal_summary.csv",index=False)
    val.to_csv(out/f"{prefix}_validation_assertions.csv",index=False); vals.to_csv(out/f"{prefix}_validation_summary.csv",index=False)
    if not retry_log.empty:retry_log.to_csv(out/f"{prefix}_retry_attempt_log_evaluator.csv.gz",index=False,compression="gzip")
    if not dist_log.empty:dist_log.to_csv(out/f"{prefix}_distillation_log_evaluator.csv.gz",index=False,compression="gzip")

    manifest={"experiment":"phase3_05_1_protocol_fidelity_inference_refined","seed":args.seed,"phase2_07_source":str(source),"output_dir":str(out),"methodology_refinements":["TeleGate and TeleData have distinct causal stage/resource sequences","low-latency vs high-fidelity service classes differ only by distillation depth/raw-pair requirement","three attacker probe paths map to physically distinct route resources with route-local EPR pools/refill links"],"attacker_protocol":"direct_coherent_remote_cx","tasks":{t:sorted(truth[truth.task_name==t].evaluator_label.unique().tolist()) for t in [TASK_PROTOCOL,TASK_DISTILL,TASK_RETRY,TASK_SERVICE]},"base_schedule_count":len(schedules),"schedule_profiles":sorted({s.schedule_profile for s in schedules}),"condition_count":len(conditions),"repeats_per_schedule":args.repeats_per_schedule,"observation_window_ns":args.observation_window_ns,"attacker_first_release_ns":float(p27.ATTACKER_FIRST_RELEASE_NS),"attacker_period_ns":float(p27.ATTACKER_PERIOD_NS),"probe_paths":list(PROBE_PATHS),"link_success_probabilities":link_probs,"test_size":args.test_size,"rf_trees":args.rf_trees,"probe_budgets":budgets,"trace_count":int(traces.trace_id.nunique()),"probe_row_count":len(traces),"training_base_schedule_count":int((split.split=="train").sum()),"test_base_schedule_count":int((split.split=="test").sum()),"feature_columns":FEATURE_COLUMNS,"attacker_visible_columns":ATTACKER_VISIBLE_COLUMNS,"validation_assertions":len(val),"validation_passed":int(val.passed.sum()),"all_validation_passed":bool(val.passed.all()),"notes":["Phase 3.5.1 remains inference/attack characterization, not defense.","Victim route assignment is generated once per base schedule/repeat and reused across every condition.","Service-class comparison holds runtime mode, retry limit, prefetch status, and post-EPR pipeline fixed; only distillation depth changes.","TeleGate and TeleData use distinct stage orders and distinct path-specific physical resources.","Probe paths are physically distinct route calendars, not labels over one shared queue."]}
    (out/f"{prefix}_run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("\n[Phase 3.5.1] Validation"); print(vals.to_string(index=False)); print("\n[Phase 3.5.1] Best held-out classification result per task"); print(summary.to_string(index=False)); print("\n[Phase 3.5.1] Regression results"); print(regm.to_string(index=False)); print(f"\n[Phase 3.5.1] Wrote outputs to: {out}")
    if args.fail_on_validation_error and not bool(val.passed.all()):raise AssertionError("Phase 3.5.1 validation failed:\n"+val.loc[~val.passed,["assertion_name","observed"]].to_string(index=False))


def parse_args():
    ap=argparse.ArgumentParser(description="Phase 3.5.1 — Refined Protocol and Fidelity-Demand Inference")
    ap.add_argument("--output-dir",default=str(DEFAULT_OUTPUT_DIR)); ap.add_argument("--seed",type=int,default=DEFAULT_SEED); ap.add_argument("--base-schedules",type=int,default=DEFAULT_BASE_SCHEDULES); ap.add_argument("--repeats-per-schedule",type=int,default=DEFAULT_REPEATS_PER_SCHEDULE); ap.add_argument("--observation-window-ns",type=float,default=DEFAULT_OBSERVATION_WINDOW_NS); ap.add_argument("--link-success-probabilities",default=DEFAULT_LINK_SUCCESS_PROBABILITIES); ap.add_argument("--test-size",type=float,default=DEFAULT_TEST_SIZE); ap.add_argument("--rf-trees",type=int,default=DEFAULT_RF_TREES); ap.add_argument("--probe-budgets",default=DEFAULT_PROBE_BUDGETS); ap.add_argument("--fail-on-validation-error",action=argparse.BooleanOptionalAction,default=True); return ap.parse_args()


def main():
    args=parse_args()
    if args.base_schedules<6:raise ValueError("--base-schedules must be >=6")
    if args.repeats_per_schedule<1:raise ValueError("--repeats-per-schedule must be >=1")
    if not .05<=args.test_size<=.5:raise ValueError("--test-size must be in [.05,.5]")
    if args.rf_trees<10:raise ValueError("--rf-trees must be >=10")
    run_experiment(args)


if __name__=="__main__":main()
