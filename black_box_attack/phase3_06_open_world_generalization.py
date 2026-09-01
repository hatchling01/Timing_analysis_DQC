#!/usr/bin/env python3
"""
Phase 3.6 — Closed-World / Open-World Generalization
====================================================

Research question
-----------------
After Phase 3.5.1 showed that attacker-only timing can reveal semantic
properties of distributed quantum communication under a refined architecture,
how much of that inference survives when the hidden victim moves outside the
training distribution?

Progression
-----------
3.1  how much communication?
3.2  when does it occur?
3.3  where does it occur?
3.4  which intermodule edges / graph are active?
3.5  how is communication implemented / what service policy is requested?
3.6  does semantic inference survive unseen workloads, link conditions,
     placements, implementation variants, and combined domain shift?

Phase 3.6 remains an inference / privacy-leakage characterization experiment.
It is NOT a defense experiment.

Architecture foundation
-----------------------
This script imports and reuses the refined Phase-3.5.1 simulator:

    phase3_05_1_protocol_fidelity_inference_refined.py

Therefore it preserves:
    * distinct TeleGate and TeleData causal resource paths;
    * the controlled low-latency vs high-fidelity service comparison;
    * physically distinct attacker probe paths;
    * the strict attacker/evaluator separation;
    * the Phase-2.7 calibrated remote-operation timing constants.

Training domain
---------------
Known workload families:
    sparse_periodic
    dense_periodic
    synchronization_bursty

Known link-success probabilities:
    0.65, 0.80, 0.95

Known placement policy:
    balanced_round_robin

Known implementation:
    nominal timing, EPR prefetch target = 2

Base schedules are split by schedule instance into:
    train
    validation
    closed_test

Model family is selected ONLY on the validation split and is then refit on
train+validation before closed-world and open-world evaluation.

Open-world evaluation domains
-----------------------------
1. closed_world
   Held-out victim schedule instances drawn from the known workload families.

2. unseen_workload
   Completely unseen temporal communication families:
       jittered_periodic
       alternating_rate
       clustered_bursty

3. unseen_link
   Link-success probabilities not used in training:
       0.55, 0.725, 0.875, 0.99

4. unseen_placement
   Hidden victim route policies not used in training:
       single_path_hotspot
       two_path_skewed
       markov_locality

5. unseen_implementation
   In-family implementation shifts not used in training:
       timing_fast      (victim service times x0.85)
       timing_slow      (victim service times x1.15)
       prefetch_small   (route-local prefetch target 1)
       prefetch_large   (route-local prefetch target 3)

6. joint_shift
   Unseen workload + unseen link probability + unseen route policy + unseen
   implementation timing/pool configuration at the same time.

The hidden semantic labels remain the same across domains.  Thus 3.6 measures
whether the semantic timing fingerprint transfers to unseen execution domains,
not whether an open-set classifier can name a completely new protocol family.

Primary tasks
-------------
    protocol_inference
    distillation_depth_inference
    retry_policy_inference
    service_class_inference

Negative-control tasks from Phase 3.5.1 are retained:
    protocol_label_only_control
    service_label_only_control

Regression tasks are also evaluated under domain shift:
    distillation_depth_estimation
    retry_count_estimation

Attacker boundary
-----------------
The model uses only Phase-3.5.1 attacker-visible timing features.  No hidden
workload family, link probability, placement policy, implementation variant,
protocol label, victim route, resource wait, retry outcome, or distillation
state is included in the feature vector.

Outputs
-------
    phase3_06_domain_classification_metrics.csv
    phase3_06_domain_regression_metrics.csv
    phase3_06_generalization_drop_summary.csv
    phase3_06_selected_models.csv
    phase3_06_validation_model_selection.csv
    phase3_06_domain_predictions.csv
    phase3_06_confusion_matrix.csv
    phase3_06_domain_signal_summary.csv
    phase3_06_domain_table.csv
    phase3_06_base_schedule_table.csv
    phase3_06_group_split.csv
    phase3_06_attacker_visible_trace.csv.gz
    phase3_06_trace_features.csv
    phase3_06_evaluator_ground_truth.csv
    phase3_06_validation_assertions.csv
    phase3_06_validation_summary.csv
    phase3_06_run_manifest.json

Default output directory
------------------------
blackbox_window_results/phase3/phase3.6/

Run
---
    python phase3_06_open_world_generalization.py

Smoke
-----
    python phase3_06_open_world_generalization.py \
        --known-schedules 9 \
        --ood-workload-schedules 3 \
        --ood-shift-schedules 3 \
        --repeats 1 \
        --observation-window-ns 6000 \
        --rf-trees 50 \
        --output-dir /tmp/p36_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
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


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SEED = 3601
DEFAULT_KNOWN_SCHEDULES = 18
DEFAULT_OOD_WORKLOAD_SCHEDULES = 9
DEFAULT_OOD_SHIFT_SCHEDULES = 6
DEFAULT_REPEATS = 1
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_RF_TREES = 500
DEFAULT_OUTPUT_DIR = Path("blackbox_window_results") / "phase3" / "phase3.6"

TRAIN_LINK_PROBS = (0.65, 0.80, 0.95)
OOD_LINK_PROBS = (0.55, 0.725, 0.875, 0.99)
KNOWN_WORKLOADS = (
    "sparse_periodic",
    "dense_periodic",
    "synchronization_bursty",
)
OOD_WORKLOADS = (
    "jittered_periodic",
    "alternating_rate",
    "clustered_bursty",
)
OOD_PLACEMENTS = (
    "single_path_hotspot",
    "two_path_skewed",
    "markov_locality",
)

TASK_PROTOCOL = "protocol_inference"
TASK_PROTOCOL_CONTROL = "protocol_label_only_control"
TASK_DISTILL = "distillation_depth_inference"
TASK_RETRY = "retry_policy_inference"
TASK_SERVICE = "service_class_inference"
TASK_SERVICE_CONTROL = "service_label_only_control"
ALL_CLASSIFICATION_TASKS = (
    TASK_PROTOCOL,
    TASK_PROTOCOL_CONTROL,
    TASK_DISTILL,
    TASK_RETRY,
    TASK_SERVICE,
    TASK_SERVICE_CONTROL,
)
PRIMARY_TASKS = (TASK_PROTOCOL, TASK_DISTILL, TASK_RETRY, TASK_SERVICE)

EPS = 1e-12


# =============================================================================
# Load Phase 3.5.1
# =============================================================================


def load_phase351_module():
    candidates = [
        Path(__file__).resolve().parent / "phase3_05_1_protocol_fidelity_inference_refined.py",
        Path.cwd() / "phase3_05_1_protocol_fidelity_inference_refined.py",
        Path(__file__).resolve().parent.parent / "phase3_05_1_protocol_fidelity_inference_refined.py",
    ]
    source = next((x for x in candidates if x.exists()), None)
    if source is None:
        searched = "\n".join(f"  - {x}" for x in candidates)
        raise FileNotFoundError(
            "Could not locate phase3_05_1_protocol_fidelity_inference_refined.py\n"
            f"Searched:\n{searched}"
        )
    name = "phase3_05_1_protocol_fidelity_inference_refined"
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


# =============================================================================
# Domain and schedule definitions
# =============================================================================


@dataclass(frozen=True)
class OWBaseSchedule:
    base_schedule_id: str
    schedule_group: str
    schedule_profile: str
    base_phase_ns: float
    interval_ns: float
    interval2_ns: float
    burst_period_ns: float
    burst_size: int
    burst_spacing_ns: float
    victim_route_offset: int


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    domain_family: str
    description: str
    workload_mode: str
    link_mode: str
    placement_policy: str
    timing_scale: float
    prefetch_target: int


DOMAINS = [
    DomainSpec(
        "known_domain", "closed_world",
        "Known workloads, known links, balanced placement, nominal implementation.",
        "known", "train_grid", "balanced_round_robin", 1.0, 2,
    ),
    DomainSpec(
        "unseen_workload", "open_world",
        "Unseen temporal workload families; known links and nominal architecture.",
        "ood", "train_grid", "balanced_round_robin", 1.0, 2,
    ),
    DomainSpec(
        "unseen_link", "open_world",
        "Known workload families with unseen link-success probabilities.",
        "known", "ood_grid", "balanced_round_robin", 1.0, 2,
    ),
    DomainSpec(
        "unseen_placement_single", "open_world",
        "Known workloads with single-path hidden victim placement.",
        "known", "nominal_link", "single_path_hotspot", 1.0, 2,
    ),
    DomainSpec(
        "unseen_placement_two_path", "open_world",
        "Known workloads with two-path skewed hidden victim placement.",
        "known", "nominal_link", "two_path_skewed", 1.0, 2,
    ),
    DomainSpec(
        "unseen_placement_markov", "open_world",
        "Known workloads with Markov-local hidden victim placement.",
        "known", "nominal_link", "markov_locality", 1.0, 2,
    ),
    DomainSpec(
        "unseen_impl_fast", "open_world",
        "Known workloads with victim service durations scaled by 0.85.",
        "known", "nominal_link", "balanced_round_robin", 0.85, 2,
    ),
    DomainSpec(
        "unseen_impl_slow", "open_world",
        "Known workloads with victim service durations scaled by 1.15.",
        "known", "nominal_link", "balanced_round_robin", 1.15, 2,
    ),
    DomainSpec(
        "unseen_prefetch_small", "open_world",
        "Known workloads with route-local prefetch target reduced to 1.",
        "known", "nominal_link", "balanced_round_robin", 1.0, 1,
    ),
    DomainSpec(
        "unseen_prefetch_large", "open_world",
        "Known workloads with route-local prefetch target increased to 3.",
        "known", "nominal_link", "balanced_round_robin", 1.0, 3,
    ),
    DomainSpec(
        "joint_shift", "open_world",
        "Unseen workload + unseen link + unseen placement + unseen implementation.",
        "ood", "joint_link", "markov_locality", 1.15, 1,
    ),
]


def build_schedule_bank(seed: int, count: int, group: str, workload_mode: str) -> list[OWBaseSchedule]:
    if count < 3:
        raise ValueError("Each schedule bank must contain at least three schedules.")
    profiles = KNOWN_WORKLOADS if workload_mode == "known" else OOD_WORKLOADS
    rows: list[OWBaseSchedule] = []
    for i in range(count):
        profile = profiles[i % len(profiles)]
        rng = np.random.default_rng(stable_seed(seed, group, i))
        phase = float(rng.uniform(100.0, 420.0))
        route_offset = int(rng.integers(0, 3))
        interval = 0.0
        interval2 = 0.0
        burst_period = 0.0
        burst_size = 1
        burst_spacing = 0.0

        if profile == "sparse_periodic":
            interval = float(rng.uniform(900.0, 1200.0))
        elif profile == "dense_periodic":
            interval = float(rng.uniform(430.0, 560.0))
        elif profile == "synchronization_bursty":
            burst_period = float(rng.uniform(1400.0, 1850.0))
            burst_size = int(rng.integers(3, 7))
            burst_spacing = float(rng.uniform(55.0, 95.0))
        elif profile == "jittered_periodic":
            interval = float(rng.uniform(600.0, 900.0))
        elif profile == "alternating_rate":
            interval = float(rng.uniform(430.0, 560.0))
            interval2 = float(rng.uniform(900.0, 1200.0))
        elif profile == "clustered_bursty":
            burst_period = float(rng.uniform(1100.0, 1700.0))
            burst_size = int(rng.integers(2, 6))
            burst_spacing = float(rng.uniform(35.0, 80.0))
        else:
            raise RuntimeError(profile)

        rows.append(
            OWBaseSchedule(
                base_schedule_id=f"{group}_{i:04d}",
                schedule_group=group,
                schedule_profile=profile,
                base_phase_ns=phase,
                interval_ns=interval,
                interval2_ns=interval2,
                burst_period_ns=burst_period,
                burst_size=burst_size,
                burst_spacing_ns=burst_spacing,
                victim_route_offset=route_offset,
            )
        )
    return rows


def generate_releases(s: OWBaseSchedule, repeat_id: int, seed: int, window_ns: float) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, s.base_schedule_id, repeat_id, "release"))
    phase = s.base_phase_ns + float(rng.uniform(-12.0, 12.0))
    values: list[float] = []

    if s.schedule_profile in {"sparse_periodic", "dense_periodic"}:
        values = list(np.arange(phase, window_ns, s.interval_ns, dtype=float))

    elif s.schedule_profile == "synchronization_bursty":
        base = phase
        while base < window_ns:
            values.extend(base + j * s.burst_spacing_ns for j in range(s.burst_size))
            base += s.burst_period_ns

    elif s.schedule_profile == "jittered_periodic":
        t = phase
        while t < window_ns:
            values.append(t)
            step = max(120.0, s.interval_ns + float(rng.normal(0.0, 0.22 * s.interval_ns)))
            t += step

    elif s.schedule_profile == "alternating_rate":
        t = phase
        toggle = 0
        while t < window_ns:
            values.append(t)
            base_step = s.interval_ns if toggle == 0 else s.interval2_ns
            t += max(120.0, base_step + float(rng.normal(0.0, 0.06 * base_step)))
            toggle ^= 1

    elif s.schedule_profile == "clustered_bursty":
        base = phase
        while base < window_ns:
            size = max(1, s.burst_size + int(rng.integers(-1, 2)))
            for j in range(size):
                values.append(base + j * s.burst_spacing_ns + float(rng.uniform(-8.0, 8.0)))
            base += max(350.0, s.burst_period_ns + float(rng.normal(0.0, 0.18 * s.burst_period_ns)))
    else:
        raise RuntimeError(s.schedule_profile)

    rel = np.asarray(values, dtype=float)
    rel = rel[(rel >= 0.0) & (rel < window_ns)]
    if len(rel):
        rel = rel + rng.uniform(-4.0, 4.0, size=len(rel))
        rel = np.clip(rel, 0.0, window_ns - 1e-6)
    return np.sort(rel)


def route_sequence(
    schedule: OWBaseSchedule,
    count: int,
    repeat_id: int,
    seed: int,
    placement_policy: str,
    paths: tuple[str, ...],
) -> list[str]:
    if count <= 0:
        return []
    offset = schedule.victim_route_offset % len(paths)

    if placement_policy == "balanced_round_robin":
        return [paths[(i + offset) % len(paths)] for i in range(count)]

    if placement_policy == "single_path_hotspot":
        return [paths[offset]] * count

    if placement_policy == "two_path_skewed":
        p0 = paths[offset]
        p1 = paths[(offset + 1) % len(paths)]
        # Deterministic 3:1 skew.
        pattern = [p0, p0, p0, p1]
        return [pattern[i % len(pattern)] for i in range(count)]

    if placement_policy == "markov_locality":
        rng = np.random.default_rng(
            stable_seed(seed, schedule.base_schedule_id, repeat_id, "route_markov")
        )
        cur = offset
        out: list[str] = []
        for _ in range(count):
            out.append(paths[cur])
            if rng.random() >= 0.75:
                choices = [j for j in range(len(paths)) if j != cur]
                cur = int(rng.choice(choices))
        return out

    raise RuntimeError(placement_policy)


def domain_link_probs(domain: DomainSpec) -> list[float]:
    if domain.link_mode == "train_grid":
        return list(TRAIN_LINK_PROBS)
    if domain.link_mode == "ood_grid":
        return list(OOD_LINK_PROBS)
    if domain.link_mode == "nominal_link":
        return [0.80]
    if domain.link_mode == "joint_link":
        return [0.725]
    raise RuntimeError(domain.link_mode)


# =============================================================================
# Domain-shift simulator
# =============================================================================


def build_ow_simulator_class(p351):
    class OWSimulator(p351.Simulator):
        def __init__(
            self,
            p27,
            condition,
            link_p: float,
            seed: int,
            schedule_id: str,
            repeat_id: int,
            run_kind: str,
            *,
            timing_scale: float,
            prefetch_target: int,
        ):
            self.ow_timing_scale = float(timing_scale)
            self.ow_prefetch_target = int(prefetch_target)
            super().__init__(
                p27,
                condition,
                link_p,
                seed,
                schedule_id,
                repeat_id,
                run_kind,
            )

        def request_resource(
            self,
            actor,
            tenant,
            name,
            stage,
            now,
            duration,
            release_on_done=True,
        ):
            # Only the hidden victim / victim-triggered background work is shifted.
            # The attacker's own direct-coherent probe remains fixed in every domain.
            scale = self.ow_timing_scale if tenant == "victim" else 1.0
            return super().request_resource(
                actor,
                tenant,
                name,
                stage,
                now,
                float(duration) * scale,
                release_on_done,
            )

        def initialize_prefetch(self):
            if not self.condition.uses_prefetch:
                return
            for path in p351.PROBE_PATHS:
                for _ in range(self.ow_prefetch_target):
                    self.pair_counter += 1
                    self.pools[path].append(
                        p351.Pair(
                            f"warm::{path}::{self.pair_counter}",
                            float(self.p27.EPR_PAIR_LIFETIME_NS),
                        )
                    )
                self.pool_events.append(
                    {
                        "run_kind": self.run_kind,
                        "time_ns": 0.0,
                        "event": "warm_prefetch",
                        "route_path": path,
                        "pool_level": len(self.pools[path]),
                        "prefetch_target": self.ow_prefetch_target,
                    }
                )

        def ensure_refill(self, path, now):
            if not self.condition.uses_prefetch:
                return
            while len(self.pools[path]) + self.refill_inflight[path] < self.ow_prefetch_target:
                self.refill_counter += 1
                actor = f"refill::{path}::{self.refill_counter}"
                self.background_path[actor] = path
                self.refill_inflight[path] += 1
                self.request_resource(
                    actor,
                    "victim",
                    "epr_generator",
                    "refill_generator",
                    now,
                    float(self.p27.EPR_GENERATOR_SETUP_NS),
                    True,
                )

    return OWSimulator


# =============================================================================
# Request helpers
# =============================================================================


def make_attacker_requests(p351, p27, repeat_id: int, window_ns: float):
    return p351.make_attacker_requests(p27, repeat_id, window_ns)


def make_victim_requests(
    p351,
    releases: np.ndarray,
    schedule: OWBaseSchedule,
    repeat_id: int,
    placement_policy: str,
    seed: int,
):
    routes = route_sequence(
        schedule,
        len(releases),
        repeat_id,
        seed,
        placement_policy,
        p351.PROBE_PATHS,
    )
    return [
        p351.RequestState(
            request_id=f"victim::{schedule.base_schedule_id}::{repeat_id:02d}::{i}",
            tenant="victim",
            release_ns=float(t),
            request_index=i,
            route_path=routes[i],
        )
        for i, t in enumerate(releases)
    ]


# =============================================================================
# Dataset generation
# =============================================================================


def schedule_bank_for_domain(
    domain: DomainSpec,
    known_test: list[OWBaseSchedule],
    ood_workload: list[OWBaseSchedule],
    ood_shift: list[OWBaseSchedule],
) -> list[OWBaseSchedule]:
    if domain.domain_id == "known_domain":
        raise RuntimeError("known_domain is generated from the entire known bank separately")
    if domain.workload_mode == "ood":
        return ood_workload if domain.domain_id == "unseen_workload" else ood_shift
    return known_test


def simulate_domain_dataset(
    p351,
    p27,
    OWSimulator,
    *,
    domain: DomainSpec,
    schedules: list[OWBaseSchedule],
    conditions: list[Any],
    link_probs: list[float],
    repeats: int,
    window_ns: float,
    seed: int,
    attacker_cache: dict[int, tuple[pd.DataFrame, list[Any]]],
):
    trace_parts: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []

    total = len(schedules) * len(conditions) * len(link_probs) * repeats
    done = 0

    for schedule in schedules:
        for repeat_id in range(repeats):
            releases = generate_releases(schedule, repeat_id, seed, window_ns)
            attacker_rows = make_attacker_requests(p351, p27, repeat_id, window_ns)
            victim_rows = make_victim_requests(
                p351,
                releases,
                schedule,
                repeat_id,
                domain.placement_policy,
                seed,
            )

            for lp in link_probs:
                for condition in conditions:
                    sim = OWSimulator(
                        p27,
                        condition,
                        lp,
                        seed,
                        schedule.base_schedule_id,
                        repeat_id,
                        "combined",
                        timing_scale=domain.timing_scale,
                        prefetch_target=domain.prefetch_target,
                    )
                    creq, cwait, cint, cstage, cretry, cdist, cpool = sim.run(
                        [p351.clone_request_state(x) for x in attacker_rows + victim_rows]
                    )

                    trace_id = hashlib.sha256(
                        (
                            f"{domain.domain_id}|{schedule.base_schedule_id}|{repeat_id}|"
                            f"{lp:.6f}|{condition.condition_id}|{seed}"
                        ).encode()
                    ).hexdigest()[:20]
                    trace = p351.pair_attacker_trace(
                        attacker_cache[repeat_id][0],
                        creq,
                        attacker_rows,
                        trace_id,
                    )
                    trace.insert(1, "domain_id", domain.domain_id)
                    trace_parts.append(trace)

                    feats = p351.extract_features(
                        trace,
                        float(p27.ATTACKER_PERIOD_NS),
                    )
                    feature_rows.append(
                        {
                            "trace_id": trace_id,
                            "domain_id": domain.domain_id,
                            "domain_family": domain.domain_family,
                            "base_schedule_id": schedule.base_schedule_id,
                            "repeat_id": repeat_id,
                            **feats,
                        }
                    )

                    vcomb = creq[creq["tenant"] == "victim"]
                    vretry = (
                        cretry[cretry["tenant"] == "victim"]
                        if not cretry.empty and "tenant" in cretry.columns
                        else pd.DataFrame()
                    )
                    failed_attempts = (
                        int((~vretry["success"].astype(bool)).sum())
                        if len(vretry)
                        else 0
                    )
                    truth = {
                        "trace_id": trace_id,
                        "domain_id": domain.domain_id,
                        "domain_family": domain.domain_family,
                        "base_schedule_id": schedule.base_schedule_id,
                        "schedule_group": schedule.schedule_group,
                        "repeat_id": repeat_id,
                        "schedule_profile": schedule.schedule_profile,
                        "link_success_probability": float(lp),
                        "placement_policy": domain.placement_policy,
                        "timing_scale": domain.timing_scale,
                        "prefetch_target": domain.prefetch_target,
                        "condition_id": condition.condition_id,
                        "task_name": condition.task_name,
                        "evaluator_label": condition.evaluator_label,
                        "distillation_depth": int(condition.distillation_depth),
                        "retry_limit": int(condition.retry_limit),
                        "victim_request_count": len(victim_rows),
                        "victim_success_fraction": float(vcomb["success"].mean()) if len(vcomb) else 1.0,
                        "actual_retry_count": failed_attempts,
                        "combined_wait_event_count": len(cwait),
                        "combined_interval_count": len(cint),
                        "combined_stage_count": len(cstage),
                        "combined_pool_event_count": len(cpool),
                    }
                    truth_rows.append(truth)
                    signal_rows.append({**truth, **feats})
                    done += 1
                    if done % max(1, total // 10) == 0 or done == total:
                        print(
                            f"[Phase 3.6] {domain.domain_id}: {done}/{total} traces"
                        )

    return (
        pd.concat(trace_parts, ignore_index=True),
        pd.DataFrame(feature_rows),
        pd.DataFrame(truth_rows),
        pd.DataFrame(signal_rows),
    )


# =============================================================================
# Known-domain split
# =============================================================================


def build_known_split(schedules: list[OWBaseSchedule], seed: int) -> pd.DataFrame:
    """Per-profile 3:1:2 train/validation/closed-test split when count=18."""
    table = pd.DataFrame([asdict(s) for s in schedules])
    rows = []
    rng = np.random.default_rng(stable_seed(seed, "known_split"))
    for profile, g in table.groupby("schedule_profile", sort=True):
        ids = g["base_schedule_id"].astype(str).to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError(f"Need >=3 known schedules for profile {profile}")
        n_val = max(1, int(round(n / 6)))
        n_test = max(1, int(round(n / 3)))
        if n_val + n_test >= n:
            n_test = 1
            n_val = 1
        val_ids = set(ids[:n_val])
        test_ids = set(ids[n_val:n_val + n_test])
        for sid in ids:
            split = "validation" if sid in val_ids else ("closed_test" if sid in test_ids else "train")
            rows.append({
                "base_schedule_id": sid,
                "schedule_profile": profile,
                "split": split,
            })
    return pd.DataFrame(rows).sort_values(["schedule_profile", "split", "base_schedule_id"]).reset_index(drop=True)


# =============================================================================
# Model selection and evaluation
# =============================================================================


def classifier_models(seed: int, rf_trees: int):
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_trees,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=seed,
            max_iter=250,
            l2_regularization=1e-3,
        ),
    }


def regression_models(seed: int, rf_trees: int):
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=rf_trees,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=seed,
            max_iter=250,
            l2_regularization=1e-3,
        ),
        "elastic_net": Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.25, random_state=seed, max_iter=10000)),
        ]),
    }


def evaluate_classifier_predictions(y_true, y_pred, labels, probabilities=None):
    auc = math.nan
    if len(labels) == 2 and probabilities is not None and len(np.unique(y_true)) == 2:
        positive = labels[-1]
        auc = float(roc_auc_score((y_true == positive).astype(int), probabilities[:, labels.index(positive)]))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "binary_roc_auc": auc,
    }


def select_classification_models(
    p351,
    known_analysis: pd.DataFrame,
    known_split: pd.DataFrame,
    *,
    seed: int,
    rf_trees: int,
):
    split_map = known_split.set_index("base_schedule_id")["split"]
    data = known_analysis.copy()
    data["split"] = data["base_schedule_id"].map(split_map)
    selection_rows = []
    selected: dict[str, str] = {}

    for task, g in data.groupby("task_name", sort=True):
        train = g[g["split"] == "train"]
        val = g[g["split"] == "validation"]
        labels = sorted(g["evaluator_label"].unique())
        Xtr = train[p351.FEATURE_COLUMNS].astype(float).to_numpy()
        Xv = val[p351.FEATURE_COLUMNS].astype(float).to_numpy()
        ytr = train["evaluator_label"].astype(str).to_numpy()
        yv = val["evaluator_label"].astype(str).to_numpy()

        task_rows = []
        for name, model in classifier_models(seed, rf_trees).items():
            model.fit(Xtr, ytr)
            yp = model.predict(Xv)
            probs = None
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(Xv)
                classes = list(model.classes_)
                probs = np.column_stack([raw[:, classes.index(lbl)] for lbl in labels])
            m = evaluate_classifier_predictions(yv, yp, labels, probs)
            row = {
                "task_name": task,
                "model_name": name,
                "validation_sample_count": len(val),
                "class_count": len(labels),
                "chance_accuracy": 1.0 / len(labels),
                **m,
            }
            selection_rows.append(row)
            task_rows.append(row)

        best = sorted(task_rows, key=lambda r: (r["macro_f1"], r["accuracy"]), reverse=True)[0]
        selected[task] = best["model_name"]

    return pd.DataFrame(selection_rows), selected


def fit_selected_classifiers(
    p351,
    known_analysis: pd.DataFrame,
    known_split: pd.DataFrame,
    selected: dict[str, str],
    *, seed: int, rf_trees: int,
):
    split_map = known_split.set_index("base_schedule_id")["split"]
    data = known_analysis.copy()
    data["split"] = data["base_schedule_id"].map(split_map)
    fitted = {}
    label_map = {}
    for task, model_name in selected.items():
        g = data[(data["task_name"] == task) & (data["split"].isin(["train", "validation"]))]
        model = classifier_models(seed, rf_trees)[model_name]
        model.fit(g[p351.FEATURE_COLUMNS].astype(float), g["evaluator_label"].astype(str))
        fitted[task] = model
        label_map[task] = sorted(g["evaluator_label"].unique())
    return fitted, label_map


def select_regression_models(
    p351,
    known_analysis: pd.DataFrame,
    known_split: pd.DataFrame,
    *, seed: int, rf_trees: int,
):
    split_map = known_split.set_index("base_schedule_id")["split"]
    data = known_analysis.copy()
    data["split"] = data["base_schedule_id"].map(split_map)
    specs = [
        ("distillation_depth_estimation", TASK_DISTILL, "distillation_depth"),
        ("retry_count_estimation", TASK_RETRY, "actual_retry_count"),
    ]
    rows = []
    selected = {}
    for task_name, source_task, target in specs:
        g = data[data["task_name"] == source_task]
        tr = g[g["split"] == "train"]
        va = g[g["split"] == "validation"]
        task_rows = []
        for name, model in regression_models(seed, rf_trees).items():
            model.fit(tr[p351.FEATURE_COLUMNS].astype(float), tr[target].astype(float))
            yp = np.asarray(model.predict(va[p351.FEATURE_COLUMNS].astype(float)), float)
            row = {
                "task_name": task_name,
                "source_task": source_task,
                "target": target,
                "model_name": name,
                "validation_sample_count": len(va),
                "mae": mean_absolute_error(va[target].astype(float), yp),
                "rmse": math.sqrt(mean_squared_error(va[target].astype(float), yp)),
                "r2": r2_score(va[target].astype(float), yp) if va[target].nunique() > 1 else math.nan,
            }
            rows.append(row)
            task_rows.append(row)
        best = sorted(task_rows, key=lambda r: r["mae"])[0]
        selected[task_name] = {
            "source_task": source_task,
            "target": target,
            "model_name": best["model_name"],
        }
    return pd.DataFrame(rows), selected


def fit_selected_regressors(
    p351,
    known_analysis,
    known_split,
    selected,
    *, seed, rf_trees,
):
    split_map = known_split.set_index("base_schedule_id")["split"]
    data = known_analysis.copy(); data["split"] = data["base_schedule_id"].map(split_map)
    fitted = {}
    for task_name, spec in selected.items():
        g = data[(data["task_name"] == spec["source_task"]) & (data["split"].isin(["train", "validation"]))]
        model = regression_models(seed, rf_trees)[spec["model_name"]]
        model.fit(g[p351.FEATURE_COLUMNS].astype(float), g[spec["target"]].astype(float))
        fitted[task_name] = model
    return fitted


def evaluate_domains(
    p351,
    datasets: dict[str, pd.DataFrame],
    fitted_classifiers,
    label_map,
    selected_classifiers,
):
    metric_rows=[]; pred_rows=[]; cm_rows=[]
    for domain_id, data in datasets.items():
        for task, model in fitted_classifiers.items():
            g=data[data.task_name==task]
            if g.empty: continue
            labels=label_map[task]
            X=g[p351.FEATURE_COLUMNS].astype(float)
            y=g.evaluator_label.astype(str).to_numpy(); yp=model.predict(X)
            probs=None
            if hasattr(model,"predict_proba"):
                raw=model.predict_proba(X); classes=list(model.classes_)
                probs=np.column_stack([raw[:,classes.index(lbl)] for lbl in labels])
            m=evaluate_classifier_predictions(y,yp,labels,probs)
            metric_rows.append({"domain_id":domain_id,"domain_family":g.domain_family.iloc[0],"task_name":task,"model_name":selected_classifiers[task],"sample_count":len(g),"class_count":len(labels),"chance_accuracy":1/len(labels),**m})
            gr=g.reset_index(drop=True)
            for i,row in gr.iterrows():
                pred_rows.append({"trace_id":row.trace_id,"domain_id":domain_id,"task_name":task,"true_label":y[i],"predicted_label":yp[i],"correct":bool(y[i]==yp[i]),"model_name":selected_classifiers[task],"schedule_profile":row.schedule_profile,"link_success_probability":row.link_success_probability,"placement_policy":row.placement_policy,"timing_scale":row.timing_scale,"prefetch_target":row.prefetch_target})
            cm=confusion_matrix(y,yp,labels=labels)
            for i,tl in enumerate(labels):
                den=cm[i].sum()
                for j,pl in enumerate(labels):
                    cm_rows.append({"domain_id":domain_id,"task_name":task,"model_name":selected_classifiers[task],"true_label":tl,"predicted_label":pl,"count":int(cm[i,j]),"true_normalized_fraction":float(cm[i,j]/den) if den else 0.0})
    return pd.DataFrame(metric_rows),pd.DataFrame(pred_rows),pd.DataFrame(cm_rows)


def evaluate_regression_domains(p351,datasets,fitted,selected):
    rows=[]; preds=[]
    for domain_id,data in datasets.items():
        for task_name,model in fitted.items():
            spec=selected[task_name]; g=data[data.task_name==spec["source_task"]]
            if g.empty: continue
            y=g[spec["target"]].astype(float).to_numpy(); yp=np.asarray(model.predict(g[p351.FEATURE_COLUMNS].astype(float)),float)
            rows.append({"domain_id":domain_id,"domain_family":g.domain_family.iloc[0],"task_name":task_name,"source_task":spec["source_task"],"target":spec["target"],"model_name":spec["model_name"],"sample_count":len(g),"mae":mean_absolute_error(y,yp),"rmse":math.sqrt(mean_squared_error(y,yp)),"r2":r2_score(y,yp) if len(np.unique(y))>1 else math.nan})
            gr=g.reset_index(drop=True)
            for i,row in gr.iterrows(): preds.append({"trace_id":row.trace_id,"domain_id":domain_id,"task_name":task_name,"target":spec["target"],"true_value":float(y[i]),"predicted_value":float(yp[i]),"absolute_error":float(abs(yp[i]-y[i])),"model_name":spec["model_name"]})
    return pd.DataFrame(rows),pd.DataFrame(preds)


# =============================================================================
# Summaries / validation
# =============================================================================


def generalization_drop_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    closed = metrics[metrics["domain_id"] == "closed_world"].set_index("task_name")
    rows=[]
    for _,r in metrics.iterrows():
        if r.task_name not in closed.index: continue
        c=closed.loc[r.task_name]
        rows.append({"domain_id":r.domain_id,"domain_family":r.domain_family,"task_name":r.task_name,"closed_world_accuracy":float(c.accuracy),"domain_accuracy":float(r.accuracy),"accuracy_drop_pp":100*(float(c.accuracy)-float(r.accuracy)),"closed_world_macro_f1":float(c.macro_f1),"domain_macro_f1":float(r.macro_f1),"macro_f1_drop":float(c.macro_f1)-float(r.macro_f1),"chance_accuracy":float(r.chance_accuracy)})
    return pd.DataFrame(rows)


def domain_signal_summary(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.groupby(["domain_id","domain_family","task_name","evaluator_label"],sort=True).agg(trace_count=("trace_id","count"),mean_abs_excess_ns=("mean_abs_excess_ns","mean"),mean_signed_excess_ns=("mean_excess_ns","mean"),delayed_fraction=("delayed_fraction","mean"),speedup_fraction=("speedup_fraction","mean"),p95_excess_ns=("p95_excess_ns","mean"),inferred_busy_period_ns=("inferred_busy_period_ns","mean"),victim_success_fraction=("victim_success_fraction","mean"),actual_retry_count=("actual_retry_count","mean")).reset_index()


def build_validation_assertions(
    p351,
    all_features,
    all_truth,
    known_split,
    domain_table,
    selected_models,
):
    rows=[]
    def add(group,name,passed,expected,observed):
        rows.append({"validation_group":group,"assertion_name":name,"passed":bool(passed),"expected":str(expected),"observed":str(observed)})

    forbidden=("victim","protocol","service","fidelity","distill","retry","resource","epr","condition","schedule_profile","link_success_probability","placement","timing_scale","prefetch_target","domain")
    bad=[c for c in p351.FEATURE_COLUMNS if any(tok in c.lower() for tok in forbidden)]
    add("blackbox","feature_vector_excludes_domain_and_evaluator_state",not bad,[],bad)
    add("blackbox","feature_columns_match_phase351",set(p351.FEATURE_COLUMNS).issubset(all_features.columns),"all Phase3.5.1 features present",len(set(p351.FEATURE_COLUMNS)-set(all_features.columns)))

    split_sets={s:set(g.base_schedule_id) for s,g in known_split.groupby("split")}
    disjoint=not(split_sets.get("train",set())&split_sets.get("validation",set())) and not(split_sets.get("train",set())&split_sets.get("closed_test",set())) and not(split_sets.get("validation",set())&split_sets.get("closed_test",set()))
    add("evaluation","known_train_validation_test_groups_disjoint",disjoint,True,disjoint)
    for split_name in ("train","validation","closed_test"):
        profiles=set(known_split.loc[known_split.split==split_name,"schedule_profile"])
        add("evaluation",f"all_known_profiles_in_{split_name}",profiles==set(KNOWN_WORKLOADS),set(KNOWN_WORKLOADS),profiles)

    train_ids=set(known_split.loc[known_split.split.isin(["train","validation"]),"base_schedule_id"])
    ood_ids=set(all_truth.loc[all_truth.domain_id!="known_domain","base_schedule_id"])
    add("evaluation","open_world_schedule_ids_do_not_overlap_training",not(train_ids&ood_ids),[],sorted(train_ids&ood_ids))

    expected_domains={d.domain_id for d in DOMAINS}
    observed_domains=set(domain_table.domain_id)
    add("design","all_open_world_domains_defined",observed_domains==expected_domains,expected_domains,observed_domains)
    add("design","ood_workload_families_are_unseen",set(OOD_WORKLOADS).isdisjoint(set(KNOWN_WORKLOADS)),True,set(OOD_WORKLOADS)&set(KNOWN_WORKLOADS))
    add("design","ood_link_probabilities_unseen",set(OOD_LINK_PROBS).isdisjoint(set(TRAIN_LINK_PROBS)),True,set(OOD_LINK_PROBS)&set(TRAIN_LINK_PROBS))

    primary_selected=set(selected_models[selected_models.selection_type=="classification"].task_name)
    add("evaluation","classification_model_selected_for_every_task",primary_selected==set(ALL_CLASSIFICATION_TASKS),set(ALL_CLASSIFICATION_TASKS),primary_selected)

    # Controls should remain exactly paired within a domain/schedule/repeat/link block.
    merged=all_features.merge(all_truth[["trace_id","domain_id","base_schedule_id","repeat_id","link_success_probability","task_name","evaluator_label"]],on=["trace_id","domain_id","base_schedule_id","repeat_id"],how="inner")
    for task,name in [(TASK_PROTOCOL_CONTROL,"protocol_control_features_identical"),(TASK_SERVICE_CONTROL,"service_control_features_identical")]:
        spans=[]
        g0=merged[merged.task_name==task]
        for _,g in g0.groupby(["domain_id","base_schedule_id","repeat_id","link_success_probability"]):
            arr=g[p351.FEATURE_COLUMNS].astype(float).to_numpy(); spans.append(float(np.max(np.ptp(arr,axis=0))) if len(arr) else 0.0)
        mx=max(spans) if spans else math.inf
        add("negative_control",name,mx<=1e-9,"<=1e-9",mx)

    return pd.DataFrame(rows)


# =============================================================================
# Main driver
# =============================================================================


def run_experiment(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    p351,source351=load_phase351_module(); p27,source27=p351.load_phase2_07_module(); OWSimulator=build_ow_simulator_class(p351)
    conditions=p351.build_conditions()

    known_bank=build_schedule_bank(args.seed,args.known_schedules,"known","known")
    ood_workload_bank=build_schedule_bank(args.seed,args.ood_workload_schedules,"oodwork","ood")
    ood_shift_bank=build_schedule_bank(args.seed,args.ood_shift_schedules,"oodshift","ood")
    known_split=build_known_split(known_bank,args.seed)
    known_test_ids=set(known_split.loc[known_split.split=="closed_test","base_schedule_id"])
    known_test=[s for s in known_bank if s.base_schedule_id in known_test_ids]

    # Attacker-only baseline, independent of victim domain.
    attacker_cache={}
    base_condition=conditions[0]
    for repeat_id in range(args.repeats):
        ars=make_attacker_requests(p351,p27,repeat_id,args.observation_window_ns)
        sim=OWSimulator(p27,base_condition,1.0,args.seed,"attacker_only",repeat_id,"attacker_only",timing_scale=1.0,prefetch_target=2)
        req,*_=sim.run([p351.clone_request_state(x) for x in ars]); attacker_cache[repeat_id]=(req,ars)

    print(f"[Phase 3.6] Phase-3.5.1 source: {source351}")
    print(f"[Phase 3.6] Known schedules={len(known_bank)}, OOD-workload schedules={len(ood_workload_bank)}, OOD-shift schedules={len(ood_shift_bank)}")

    domain_feature_parts=[]; domain_truth_parts=[]; domain_trace_parts=[]; domain_signal_parts=[]

    # Known domain: entire bank so train/validation/closed-test are all available.
    known_domain=DOMAINS[0]
    kt,kf,kg,ks=simulate_domain_dataset(p351,p27,OWSimulator,domain=known_domain,schedules=known_bank,conditions=conditions,link_probs=list(TRAIN_LINK_PROBS),repeats=args.repeats,window_ns=args.observation_window_ns,seed=args.seed,attacker_cache=attacker_cache)
    domain_trace_parts.append(kt); domain_feature_parts.append(kf); domain_truth_parts.append(kg); domain_signal_parts.append(ks)

    for domain in DOMAINS[1:]:
        bank=schedule_bank_for_domain(domain,known_test,ood_workload_bank,ood_shift_bank)
        links=domain_link_probs(domain)
        dt,df,dg,ds=simulate_domain_dataset(p351,p27,OWSimulator,domain=domain,schedules=bank,conditions=conditions,link_probs=links,repeats=args.repeats,window_ns=args.observation_window_ns,seed=args.seed,attacker_cache=attacker_cache)
        domain_trace_parts.append(dt); domain_feature_parts.append(df); domain_truth_parts.append(dg); domain_signal_parts.append(ds)

    traces=pd.concat(domain_trace_parts,ignore_index=True); features=pd.concat(domain_feature_parts,ignore_index=True); truth=pd.concat(domain_truth_parts,ignore_index=True); signal=pd.concat(domain_signal_parts,ignore_index=True)
    analysis=features.merge(truth,on=["trace_id","domain_id","domain_family","base_schedule_id","repeat_id"],validate="one_to_one")

    known_analysis=analysis[analysis.domain_id=="known_domain"].copy()
    csel,selected_cls=select_classification_models(p351,known_analysis,known_split,seed=args.seed,rf_trees=args.rf_trees)
    fitted_cls,label_map=fit_selected_classifiers(p351,known_analysis,known_split,selected_cls,seed=args.seed,rf_trees=args.rf_trees)
    rsel,selected_reg=select_regression_models(p351,known_analysis,known_split,seed=args.seed,rf_trees=args.rf_trees)
    fitted_reg=fit_selected_regressors(p351,known_analysis,known_split,selected_reg,seed=args.seed,rf_trees=args.rf_trees)

    # Construct evaluation datasets. Closed world uses only known closed-test schedules.
    split_map=known_split.set_index("base_schedule_id").split
    known_eval=known_analysis[known_analysis.base_schedule_id.map(split_map)=="closed_test"].copy(); known_eval["domain_id"]="closed_world"; known_eval["domain_family"]="closed_world"
    datasets={"closed_world":known_eval}
    for did,g in analysis[analysis.domain_id!="known_domain"].groupby("domain_id"):
        datasets[did]=g.copy()

    cls_metrics,cls_preds,cm=evaluate_domains(p351,datasets,fitted_cls,label_map,selected_cls)
    reg_metrics,reg_preds=evaluate_regression_domains(p351,datasets,fitted_reg,selected_reg)
    drops=generalization_drop_summary(cls_metrics)
    sigsum=domain_signal_summary(signal)

    selected_rows=[{"selection_type":"classification","task_name":task,"source_task":task,"target":"evaluator_label","model_name":name} for task,name in selected_cls.items()]
    selected_rows += [{"selection_type":"regression","task_name":task,"source_task":spec["source_task"],"target":spec["target"],"model_name":spec["model_name"]} for task,spec in selected_reg.items()]
    selected_df=pd.DataFrame(selected_rows)
    model_selection=pd.concat([csel.assign(selection_type="classification"),rsel.assign(selection_type="regression")],ignore_index=True,sort=False)

    domain_table=pd.DataFrame([asdict(d) for d in DOMAINS])
    schedule_table=pd.concat([pd.DataFrame([asdict(s) for s in known_bank]),pd.DataFrame([asdict(s) for s in ood_workload_bank]),pd.DataFrame([asdict(s) for s in ood_shift_bank])],ignore_index=True)
    validations=build_validation_assertions(p351,features,truth,known_split,domain_table,selected_df)
    vals=pd.DataFrame([{"assertion_count":len(validations),"passed_assertions":int(validations.passed.sum()),"failed_assertions":int((~validations.passed).sum()),"all_passed":bool(validations.passed.all())}])

    # Save.
    traces.to_csv(out/"phase3_06_attacker_visible_trace.csv.gz",index=False,compression="gzip")
    features.to_csv(out/"phase3_06_trace_features.csv",index=False)
    truth.to_csv(out/"phase3_06_evaluator_ground_truth.csv",index=False)
    cls_metrics.to_csv(out/"phase3_06_domain_classification_metrics.csv",index=False)
    reg_metrics.to_csv(out/"phase3_06_domain_regression_metrics.csv",index=False)
    cls_preds.to_csv(out/"phase3_06_domain_predictions.csv",index=False)
    reg_preds.to_csv(out/"phase3_06_regression_predictions.csv",index=False)
    cm.to_csv(out/"phase3_06_confusion_matrix.csv",index=False)
    drops.to_csv(out/"phase3_06_generalization_drop_summary.csv",index=False)
    sigsum.to_csv(out/"phase3_06_domain_signal_summary.csv",index=False)
    domain_table.to_csv(out/"phase3_06_domain_table.csv",index=False)
    schedule_table.to_csv(out/"phase3_06_base_schedule_table.csv",index=False)
    known_split.to_csv(out/"phase3_06_group_split.csv",index=False)
    selected_df.to_csv(out/"phase3_06_selected_models.csv",index=False)
    model_selection.to_csv(out/"phase3_06_validation_model_selection.csv",index=False)
    validations.to_csv(out/"phase3_06_validation_assertions.csv",index=False)
    vals.to_csv(out/"phase3_06_validation_summary.csv",index=False)

    manifest={
        "experiment":"phase3_06_open_world_generalization",
        "seed":args.seed,
        "phase3_05_1_source":str(source351),
        "phase2_07_source":str(source27),
        "output_dir":str(out),
        "research_question":"Does semantic attacker-only timing inference survive unseen workload, link, placement, implementation, and joint domain shift?",
        "known_workloads":list(KNOWN_WORKLOADS),
        "ood_workloads":list(OOD_WORKLOADS),
        "training_link_success_probabilities":list(TRAIN_LINK_PROBS),
        "ood_link_success_probabilities":list(OOD_LINK_PROBS),
        "ood_placements":list(OOD_PLACEMENTS),
        "known_schedule_count":len(known_bank),
        "ood_workload_schedule_count":len(ood_workload_bank),
        "ood_shift_schedule_count":len(ood_shift_bank),
        "repeats":args.repeats,
        "observation_window_ns":args.observation_window_ns,
        "rf_trees":args.rf_trees,
        "trace_count":int(traces.trace_id.nunique()),
        "probe_row_count":int(len(traces)),
        "feature_columns":list(p351.FEATURE_COLUMNS),
        "selected_classification_models":selected_cls,
        "selected_regression_models":selected_reg,
        "validation_assertions":len(validations),
        "validation_passed":int(validations.passed.sum()),
        "all_validation_passed":bool(validations.passed.all()),
        "notes":[
            "Phase 3.6 is inference/privacy-leakage characterization, not defense.",
            "Model family selection uses only the known-domain validation split.",
            "Closed-world and every open-world domain are untouched during model selection.",
            "The attacker protocol and attacker timing are fixed across domains.",
            "Open-world labels retain the same semantics; the experiment measures domain-shift generalization rather than open-set naming of a new protocol family.",
            "Negative-control tasks are retained to detect spurious label learning under domain shift.",
        ],
    }
    (out/"phase3_06_run_manifest.json").write_text(json.dumps(manifest,indent=2))

    print("\n[Phase 3.6] Validation")
    print(vals.to_string(index=False))
    print("\n[Phase 3.6] Selected classification models")
    print(selected_df[selected_df.selection_type=="classification"].to_string(index=False))
    print("\n[Phase 3.6] Domain classification metrics")
    print(cls_metrics.to_string(index=False))
    print(f"\n[Phase 3.6] Wrote outputs to: {out}")

    if args.fail_on_validation_error and not bool(validations.passed.all()):
        raise AssertionError("Phase 3.6 validation failed:\n"+validations.loc[~validations.passed,["assertion_name","observed"]].to_string(index=False))


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    ap=argparse.ArgumentParser(description="Phase 3.6 — Closed-World / Open-World Generalization")
    ap.add_argument("--output-dir",default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--seed",type=int,default=DEFAULT_SEED)
    ap.add_argument("--known-schedules",type=int,default=DEFAULT_KNOWN_SCHEDULES)
    ap.add_argument("--ood-workload-schedules",type=int,default=DEFAULT_OOD_WORKLOAD_SCHEDULES)
    ap.add_argument("--ood-shift-schedules",type=int,default=DEFAULT_OOD_SHIFT_SCHEDULES)
    ap.add_argument("--repeats",type=int,default=DEFAULT_REPEATS)
    ap.add_argument("--observation-window-ns",type=float,default=DEFAULT_OBSERVATION_WINDOW_NS)
    ap.add_argument("--rf-trees",type=int,default=DEFAULT_RF_TREES)
    ap.add_argument("--fail-on-validation-error",action=argparse.BooleanOptionalAction,default=True)
    return ap.parse_args()


def main():
    args=parse_args()
    if args.known_schedules<9: raise ValueError("--known-schedules must be >= 9")
    if args.ood_workload_schedules<3: raise ValueError("--ood-workload-schedules must be >= 3")
    if args.ood_shift_schedules<3: raise ValueError("--ood-shift-schedules must be >= 3")
    if args.repeats<1: raise ValueError("--repeats must be >= 1")
    if args.rf_trees<10: raise ValueError("--rf-trees must be >= 10")
    run_experiment(args)


if __name__=="__main__":
    main()
