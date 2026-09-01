#!/usr/bin/env python3
"""
Phase 3.5 — Protocol / Fidelity-Demand Inference
=================================================

Research question
-----------------
Can an observer infer a hidden tenant's remote-operation realization or a
modeled service/fidelity-demand class using only the observer's own black-box
remote-operation timing?

This experiment is an inference/attack-characterization study, not a defense
study.  It reuses the validated Phase-2.7 remote-operation simulator and keeps
the observer protocol fixed to the direct-coherent realization.  Only the
hidden tenant's runtime realization/service demand changes.

Two primary inference tasks
---------------------------
1. protocol_inference
   Hidden tenant executes the same logical remote-CX schedule using either:
       * direct_coherent_remote_cx
       * entanglement_assisted_remote_cx
   The observer always executes direct_coherent_remote_cx.

2. fidelity_demand_inference
   Hidden tenant always uses the entanglement-assisted realization, but the
   service profile requested by that tenant is one of three controlled tiers:
       * low_demand
       * medium_demand
       * high_demand
   The tiers change hidden-tenant resource occupancy / EPR-refill service cost
   while preserving the logical remote-operation count and release schedule.
   These are controlled architectural service-demand tiers; they are NOT claims
   about measured physical fidelity on a commercial platform.

Label-only negative controls
----------------------------
For each task, a matched label-only control is included.  Different evaluator
labels are assigned to otherwise identical runtime conditions.  Therefore a
black-box classifier should return to chance when the physical/service
semantics carrying the label are removed.

Threat boundary
---------------
The attacker-visible trace contains only:
    trace_id
    probe_index
    release_ns
    attacker_only_success
    combined_success
    attacker_only_completion_ns
    combined_completion_ns
    attacker_only_turnaround_ns
    combined_turnaround_ns
    excess_turnaround_ns
    delayed
    speedup
    failure_transition

Hidden protocol, service class, victim schedule, resource waits, EPR state,
scenario identity, and evaluator attribution remain evaluator-only.

Evaluation discipline
---------------------
Base victim schedules are crossed with every experimental condition.  Grouped
train/test splitting is performed by base schedule instance, so no timing
realization of one hidden schedule can appear in both train and test.

Default output directory
------------------------
blackbox_window_results/phase3/phase3.5/

Run
---
    python phase3_05_protocol_fidelity_demand_inference.py

Smoke test
----------
    python phase3_05_protocol_fidelity_demand_inference.py \
        --base-schedules 9 \
        --repeats-per-schedule 1 \
        --observation-window-ns 8000 \
        --rf-trees 100 \
        --output-dir /tmp/p35_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Global settings
# =============================================================================

DEFAULT_SEED = 3501
DEFAULT_BASE_SCHEDULES = 18
DEFAULT_REPEATS_PER_SCHEDULE = 2
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_TEST_SIZE = 0.33
DEFAULT_RF_TREES = 500
DEFAULT_OUTPUT_DIR = Path("blackbox_window_results") / "phase3" / "phase3.5"

AFFECTED_THRESHOLD_NS = 1e-9
EPS = 1e-12

ATTACKER_VISIBLE_COLUMNS = [
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
    "delayed",
    "speedup",
    "failure_transition",
]

FEATURE_COLUMNS = [
    "probe_count",
    "mean_excess_ns",
    "median_excess_ns",
    "mean_abs_excess_ns",
    "std_excess_ns",
    "max_excess_ns",
    "min_excess_ns",
    "p10_excess_ns",
    "p25_excess_ns",
    "p50_excess_ns",
    "p75_excess_ns",
    "p90_excess_ns",
    "p95_excess_ns",
    "p99_excess_ns",
    "delayed_fraction",
    "speedup_fraction",
    "failure_transition_fraction",
    "cumulative_positive_excess_ns",
    "cumulative_negative_magnitude_ns",
    "cumulative_abs_excess_ns",
    "longest_delayed_run",
    "longest_speedup_run",
    "delayed_run_count",
    "speedup_run_count",
    "lag1_autocorrelation",
    "lag2_autocorrelation",
    "lag3_autocorrelation",
    "spectral_dominant_bin_fraction",
    "spectral_centroid_fraction",
    "spectral_entropy",
    "spectral_low_frequency_power_fraction",
    "early_mean_abs_ns",
    "middle_mean_abs_ns",
    "late_mean_abs_ns",
]

PROTOCOL_TASK = "protocol_inference"
PROTOCOL_CONTROL_TASK = "protocol_label_only_control"
FIDELITY_TASK = "fidelity_demand_inference"
FIDELITY_CONTROL_TASK = "fidelity_label_only_control"


# =============================================================================
# Data model
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
class FidelityProfile:
    name: str
    endpoint_access_ns: float
    readout_ns: float
    feedforward_ns: float
    correction_ns: float
    reset_ns: float
    epr_generator_setup_ns: float
    epr_link_generation_ns: float


@dataclass(frozen=True)
class Condition:
    condition_id: str
    task_name: str
    evaluator_label: str
    victim_runtime_protocol: str
    runtime_fidelity_profile: str
    control: bool
    description: str


@dataclass(frozen=True)
class ValidationAssertion:
    validation_group: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


# =============================================================================
# Phase-2.7 loading
# =============================================================================


def load_phase2_07_module():
    candidates = [
        Path(__file__).resolve().parent / "phase2_07_remote_protocol_comparison.py",
        Path.cwd() / "phase2_07_remote_protocol_comparison.py",
        Path(__file__).resolve().parent.parent / "phase2_07_remote_protocol_comparison.py",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        searched = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            "Could not locate phase2_07_remote_protocol_comparison.py.\n"
            "Phase 3.5 intentionally reuses the validated Phase-2.7 simulator.\n"
            f"Searched:\n{searched}"
        )
    module_name = "phase2_07_remote_protocol_comparison"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, source


# =============================================================================
# Deterministic helpers and schedule generation
# =============================================================================


def stable_seed(*parts: Any, modulus: int = 2**32 - 1) -> int:
    token = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % modulus


def build_base_schedules(
    *,
    seed: int,
    count: int,
) -> list[BaseSchedule]:
    if count < 6:
        raise ValueError("--base-schedules must be at least 6")

    profiles = (
        "sparse_periodic",
        "dense_periodic",
        "synchronization_bursty",
    )
    rows: list[BaseSchedule] = []
    for idx in range(count):
        profile = profiles[idx % len(profiles)]
        rng = np.random.default_rng(stable_seed(seed, "base_schedule", idx))
        phase = float(rng.uniform(100.0, 420.0))
        if profile == "sparse_periodic":
            interval = float(rng.uniform(900.0, 1200.0))
            burst_period = 0.0
            burst_size = 1
            burst_spacing = 0.0
        elif profile == "dense_periodic":
            interval = float(rng.uniform(430.0, 560.0))
            burst_period = 0.0
            burst_size = 1
            burst_spacing = 0.0
        else:
            interval = 0.0
            burst_period = float(rng.uniform(1400.0, 1800.0))
            burst_size = int(rng.integers(3, 7))
            burst_spacing = float(rng.uniform(55.0, 95.0))
        rows.append(
            BaseSchedule(
                base_schedule_id=f"base_{idx:04d}",
                base_index=idx,
                schedule_profile=profile,
                base_phase_ns=phase,
                interval_ns=interval,
                burst_period_ns=burst_period,
                burst_size=burst_size,
                burst_spacing_ns=burst_spacing,
            )
        )
    return rows


def release_times_for_schedule(
    schedule: BaseSchedule,
    *,
    repeat_id: int,
    seed: int,
    observation_window_ns: float,
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed(seed, schedule.base_schedule_id, repeat_id, "release")
    )
    phase = schedule.base_phase_ns + float(rng.uniform(-12.0, 12.0))

    if schedule.schedule_profile in {"sparse_periodic", "dense_periodic"}:
        releases = np.arange(
            phase,
            observation_window_ns,
            schedule.interval_ns,
            dtype=float,
        )
    else:
        values: list[float] = []
        base = phase
        while base < observation_window_ns:
            for j in range(schedule.burst_size):
                values.append(base + j * schedule.burst_spacing_ns)
            base += schedule.burst_period_ns
        releases = np.asarray(values, dtype=float)

    releases = releases[(releases >= 0.0) & (releases < observation_window_ns)]
    if len(releases):
        releases = releases + rng.uniform(-4.0, 4.0, size=len(releases))
        releases = np.clip(releases, 0.0, observation_window_ns - 1e-6)
    return np.sort(releases)


# =============================================================================
# Conditions and service profiles
# =============================================================================


def build_fidelity_profiles(p27) -> dict[str, FidelityProfile]:
    # Controlled service-demand profiles.  Medium reproduces the Phase-2.7
    # entanglement-assisted durations.  Low/high reduce/increase hidden-tenant
    # service occupancy while keeping the logical schedule fixed.
    return {
        "low_demand": FidelityProfile(
            "low_demand",
            endpoint_access_ns=float(p27.ENT_ENDPOINT_ACCESS_NS),
            readout_ns=55.0,
            feedforward_ns=30.0,
            correction_ns=15.0,
            reset_ns=90.0,
            epr_generator_setup_ns=30.0,
            epr_link_generation_ns=150.0,
        ),
        "medium_demand": FidelityProfile(
            "medium_demand",
            endpoint_access_ns=float(p27.ENT_ENDPOINT_ACCESS_NS),
            readout_ns=float(p27.ENT_READOUT_NS),
            feedforward_ns=float(p27.ENT_FEEDFORWARD_NS),
            correction_ns=float(p27.ENT_CORRECTION_NS),
            reset_ns=float(p27.RESET_NS),
            epr_generator_setup_ns=float(p27.EPR_GENERATOR_SETUP_NS),
            epr_link_generation_ns=float(p27.EPR_LINK_GENERATION_NS),
        ),
        "high_demand": FidelityProfile(
            "high_demand",
            endpoint_access_ns=25.0,
            readout_ns=95.0,
            feedforward_ns=55.0,
            correction_ns=30.0,
            reset_ns=160.0,
            epr_generator_setup_ns=55.0,
            epr_link_generation_ns=280.0,
        ),
    }


def build_conditions(p27) -> list[Condition]:
    # Exactly 10 conditions: 2 protocol + 2 protocol-label controls +
    # 3 service-demand + 3 service-label controls.
    return [
        Condition(
            "protocol_direct",
            PROTOCOL_TASK,
            "direct_coherent",
            p27.DIRECT_PROTOCOL,
            "medium_demand",
            False,
            "Hidden tenant uses direct coherent remote CX.",
        ),
        Condition(
            "protocol_entanglement_assisted",
            PROTOCOL_TASK,
            "entanglement_assisted",
            p27.ENTANGLED_PROTOCOL,
            "medium_demand",
            False,
            "Hidden tenant uses prefetched entanglement-assisted remote CX.",
        ),
        Condition(
            "protocol_control_label_direct",
            PROTOCOL_CONTROL_TASK,
            "direct_coherent",
            p27.DIRECT_PROTOCOL,
            "medium_demand",
            True,
            "Label-only control: runtime fixed to direct coherent.",
        ),
        Condition(
            "protocol_control_label_entangled",
            PROTOCOL_CONTROL_TASK,
            "entanglement_assisted",
            p27.DIRECT_PROTOCOL,
            "medium_demand",
            True,
            "Label-only control: identical runtime, different evaluator label.",
        ),
        Condition(
            "fidelity_low",
            FIDELITY_TASK,
            "low_demand",
            p27.ENTANGLED_PROTOCOL,
            "low_demand",
            False,
            "Entanglement-assisted hidden tenant with lower controlled service demand.",
        ),
        Condition(
            "fidelity_medium",
            FIDELITY_TASK,
            "medium_demand",
            p27.ENTANGLED_PROTOCOL,
            "medium_demand",
            False,
            "Entanglement-assisted hidden tenant with baseline Phase-2.7 service demand.",
        ),
        Condition(
            "fidelity_high",
            FIDELITY_TASK,
            "high_demand",
            p27.ENTANGLED_PROTOCOL,
            "high_demand",
            False,
            "Entanglement-assisted hidden tenant with higher controlled service demand.",
        ),
        Condition(
            "fidelity_control_label_low",
            FIDELITY_CONTROL_TASK,
            "low_demand",
            p27.ENTANGLED_PROTOCOL,
            "medium_demand",
            True,
            "Label-only control: runtime fixed to medium demand.",
        ),
        Condition(
            "fidelity_control_label_medium",
            FIDELITY_CONTROL_TASK,
            "medium_demand",
            p27.ENTANGLED_PROTOCOL,
            "medium_demand",
            True,
            "Label-only control: runtime fixed to medium demand.",
        ),
        Condition(
            "fidelity_control_label_high",
            FIDELITY_CONTROL_TASK,
            "high_demand",
            p27.ENTANGLED_PROTOCOL,
            "medium_demand",
            True,
            "Label-only control: identical runtime, different evaluator label.",
        ),
    ]


# =============================================================================
# Mixed-protocol simulator
# =============================================================================


def make_mixed_protocol_definition(p27, uses_epr: bool):
    resources = (
        "endpoint",
        "switch_path",
        "quantum_link",
        "epr_pool",
        "epr_generator",
        "readout",
        "feedforward",
        "reset",
    )
    return p27.ProtocolDefinition(
        protocol_name="phase3_05_mixed_runtime",
        protocol_family="mixed_observer_hidden_tenant",
        logical_operation=p27.LOGICAL_OPERATION,
        description=(
            "Fixed direct-coherent observer with evaluator-selected hidden-tenant "
            "remote-operation realization/service demand."
        ),
        nominal_critical_latency_ns=float(p27.DIRECT_CRITICAL_NS),
        postcompletion_cleanup_ns=float(p27.RESET_NS),
        uses_epr=bool(uses_epr),
        used_resources=resources,
    )


def make_shared_scenario(p27, victim_protocol_name: str):
    if victim_protocol_name == p27.DIRECT_PROTOCOL:
        shared = ("endpoint", "switch_path", "quantum_link", "reset")
    else:
        shared = (
            "endpoint",
            "switch_path",
            "quantum_link",
            "epr_pool",
            "epr_generator",
            "readout",
            "feedforward",
            "reset",
        )
    return p27.ProtocolScenario(
        scenario_id="phase3_05_all_relevant_shared",
        protocol_name="phase3_05_mixed_runtime",
        scenario_class="all_relevant_shared",
        shared_resources=tuple(shared),
        description="All resources relevant to the fixed observer / hidden-tenant pair are shared.",
    )


def build_mixed_simulator_class(p27):
    class ServiceEPRSubsystem(p27.EPRSubsystem):
        def _profile(self, job) -> FidelityProfile:
            return self.sim.victim_fidelity_profile

        def on_generator_granted(self, job_id, resource_key, now, wait_ns):
            job = self.jobs[job_id]
            job.generator_wait_ns += wait_ns
            if not math.isfinite(job.start_ns):
                job.start_ns = float(now)
            duration = (
                self._profile(job).epr_generator_setup_ns
                if job.trigger_tenant == "victim"
                else float(p27.EPR_GENERATOR_SETUP_NS)
            )
            self.sim.schedule(
                now + duration,
                "epr_generator_done",
                {"job_id": job_id, "resource_key": resource_key},
            )

        def on_link_granted(self, job_id, resource_key, now, wait_ns):
            job = self.jobs[job_id]
            job.link_wait_ns += wait_ns
            duration = (
                self._profile(job).epr_link_generation_ns
                if job.trigger_tenant == "victim"
                else float(p27.EPR_LINK_GENERATION_NS)
            )
            self.sim.schedule(
                now + duration,
                "epr_link_done",
                {"job_id": job_id, "resource_key": resource_key},
            )

    class MixedProtocolSimulator(p27.ProtocolSimulator):
        def __init__(
            self,
            *,
            victim_protocol_name: str,
            victim_fidelity_profile: FidelityProfile,
            scenario,
            workload_name: str,
            trial_id: int,
            run_kind: str,
        ):
            self.victim_protocol_name = victim_protocol_name
            self.victim_fidelity_profile = victim_fidelity_profile
            mixed_protocol = make_mixed_protocol_definition(
                p27, uses_epr=(victim_protocol_name == p27.ENTANGLED_PROTOCOL)
            )
            super().__init__(
                protocol=mixed_protocol,
                scenario=scenario,
                workload_name=workload_name,
                trial_id=trial_id,
                run_kind=run_kind,
            )
            if mixed_protocol.uses_epr:
                self.epr = ServiceEPRSubsystem(self)

        def request_protocol(self, request_id: str) -> str:
            tenant = self.requests[request_id].spec.tenant
            if tenant == "attacker":
                return p27.DIRECT_PROTOCOL
            return self.victim_protocol_name

        def ent_duration(self, request_id: str, field: str, baseline: float) -> float:
            if self.requests[request_id].spec.tenant != "victim":
                return float(baseline)
            return float(getattr(self.victim_fidelity_profile, field))

        def _handle_event(self, now, event_type, payload):
            if event_type == "request_release":
                request_id = payload["request_id"]
                if self.request_protocol(request_id) == p27.DIRECT_PROTOCOL:
                    self.request_resource(request_id, "endpoint", "direct_endpoint_hold", now)
                else:
                    assert self.epr is not None
                    self.epr.acquire(
                        request_id,
                        self.requests[request_id].spec.tenant,
                        now,
                    )
                return

            if event_type == "epr_ready":
                request_id = payload["request_id"]
                self.get_resource(
                    "local_entangled_slot",
                    self.requests[request_id].spec.tenant,
                ).request(
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
                if self.request_protocol(actor_id) == p27.DIRECT_PROTOCOL:
                    self._handle_direct_grant(now, actor_id, stage_tag, resource_key)
                else:
                    self._handle_service_entangled_grant(
                        now, actor_id, stage_tag, resource_key
                    )
                return

            if event_type == "epr_generator_done":
                assert self.epr is not None
                self.epr.on_generator_done(payload["job_id"], payload["resource_key"], now)
                return

            if event_type == "epr_link_done":
                assert self.epr is not None
                self.epr.on_link_done(payload["job_id"], payload["resource_key"], now)
                return

            if event_type.startswith("direct_"):
                self._handle_direct_event(now, event_type, payload)
            elif event_type.startswith("entangled_"):
                self._handle_service_entangled_event(now, event_type, payload)
            else:
                raise RuntimeError(event_type)

        def _handle_service_entangled_grant(self, now, request_id, stage_tag, resource_key):
            if stage_tag == "entangled_local_slot_hold":
                self.request_resource(
                    request_id,
                    "endpoint",
                    "entangled_endpoint_access",
                    now,
                )
            elif stage_tag == "entangled_endpoint_access":
                dur = self.ent_duration(
                    request_id,
                    "endpoint_access_ns",
                    p27.ENT_ENDPOINT_ACCESS_NS,
                )
                self.schedule(
                    now + dur,
                    "entangled_endpoint_done",
                    {"request_id": request_id, "resource_key": resource_key},
                )
            elif stage_tag == "entangled_readout":
                dur = self.ent_duration(
                    request_id,
                    "readout_ns",
                    p27.ENT_READOUT_NS,
                )
                self.schedule(
                    now + dur,
                    "entangled_readout_done",
                    {"request_id": request_id, "resource_key": resource_key},
                )
            elif stage_tag == "entangled_feedforward":
                dur = self.ent_duration(
                    request_id,
                    "feedforward_ns",
                    p27.ENT_FEEDFORWARD_NS,
                )
                self.schedule(
                    now + dur,
                    "entangled_feedforward_done",
                    {"request_id": request_id, "resource_key": resource_key},
                )
            elif stage_tag == "entangled_reset":
                dur = self.ent_duration(request_id, "reset_ns", p27.RESET_NS)
                self.schedule(
                    now + dur,
                    "entangled_reset_done",
                    {"request_id": request_id, "resource_key": resource_key},
                )
            else:
                raise RuntimeError(stage_tag)

        def _handle_service_entangled_event(self, now, event_type, payload):
            request_id = payload["request_id"]
            if event_type == "entangled_endpoint_done":
                endpoint_key = payload["resource_key"]
                dur = self.ent_duration(
                    request_id,
                    "endpoint_access_ns",
                    p27.ENT_ENDPOINT_ACCESS_NS,
                )
                self._log_stage(
                    request_id,
                    "stored_epr_endpoint_access",
                    now - dur,
                    now,
                )
                self.resources[("endpoint", endpoint_key)].release(self, request_id, now)
                self.request_resource(request_id, "readout", "entangled_readout", now)
            elif event_type == "entangled_readout_done":
                readout_key = payload["resource_key"]
                dur = self.ent_duration(request_id, "readout_ns", p27.ENT_READOUT_NS)
                self._log_stage(
                    request_id,
                    "bell_measurement_readout",
                    now - dur,
                    now,
                )
                self.resources[("readout", readout_key)].release(self, request_id, now)
                self.request_resource(
                    request_id,
                    "feedforward",
                    "entangled_feedforward",
                    now,
                )
            elif event_type == "entangled_feedforward_done":
                ff_key = payload["resource_key"]
                dur = self.ent_duration(
                    request_id,
                    "feedforward_ns",
                    p27.ENT_FEEDFORWARD_NS,
                )
                self._log_stage(
                    request_id,
                    "classical_feedforward",
                    now - dur,
                    now,
                )
                self.resources[("feedforward", ff_key)].release(self, request_id, now)
                correction = self.ent_duration(
                    request_id,
                    "correction_ns",
                    p27.ENT_CORRECTION_NS,
                )
                self.schedule(
                    now + correction,
                    "entangled_external_complete",
                    {"request_id": request_id},
                )
            elif event_type == "entangled_external_complete":
                correction = self.ent_duration(
                    request_id,
                    "correction_ns",
                    p27.ENT_CORRECTION_NS,
                )
                self._log_stage(
                    request_id,
                    "receiver_correction",
                    now - correction,
                    now,
                )
                self.requests[request_id].external_completion_ns = float(now)
                self.request_resource(request_id, "reset", "entangled_reset", now)
            elif event_type == "entangled_reset_done":
                reset_key = payload["resource_key"]
                dur = self.ent_duration(request_id, "reset_ns", p27.RESET_NS)
                self._log_stage(
                    request_id,
                    "postcompletion_reset",
                    now - dur,
                    now,
                )
                self.resources[("reset", reset_key)].release(self, request_id, now)
                slot_key = f"{self.requests[request_id].spec.tenant}::local_entangled_slot"
                self.resources[("local_entangled_slot", slot_key)].release(
                    self, request_id, now
                )
                self.requests[request_id].cleanup_completion_ns = float(now)
            else:
                raise RuntimeError(event_type)

    return MixedProtocolSimulator


# =============================================================================
# Request generation and one-condition simulation
# =============================================================================


def attacker_specs(p27, repeat_id: int, observation_window_ns: float):
    releases = np.arange(
        p27.ATTACKER_FIRST_RELEASE_NS,
        observation_window_ns,
        p27.ATTACKER_PERIOD_NS,
        dtype=float,
    )
    return [
        p27.RequestSpec(
            request_id=f"attacker::{repeat_id:02d}::{idx}",
            tenant="attacker",
            ready_ns=float(t),
            request_index=idx,
            workload_name="opaque",
            trial_id=repeat_id,
        )
        for idx, t in enumerate(releases)
    ]


def victim_specs(p27, releases: np.ndarray, schedule_id: str, repeat_id: int):
    return [
        p27.RequestSpec(
            request_id=f"victim::{schedule_id}::{repeat_id:02d}::{idx}",
            tenant="victim",
            ready_ns=float(t),
            request_index=idx,
            workload_name="hidden",
            trial_id=repeat_id,
        )
        for idx, t in enumerate(releases)
    ]


def run_mixed(
    p27,
    MixedProtocolSimulator,
    *,
    condition: Condition,
    fidelity_profile: FidelityProfile,
    schedule_profile: str,
    repeat_id: int,
    run_kind: str,
    specs: list[Any],
):
    scenario = make_shared_scenario(p27, condition.victim_runtime_protocol)
    sim = MixedProtocolSimulator(
        victim_protocol_name=condition.victim_runtime_protocol,
        victim_fidelity_profile=fidelity_profile,
        scenario=scenario,
        workload_name=schedule_profile,
        trial_id=repeat_id,
        run_kind=run_kind,
    )
    return sim.run(specs)


def pair_attacker_trace(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    trace_id: str,
) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "trace_id": trace_id,
            "probe_index": merged["request_index"].astype(int),
            "release_ns": merged["release_ns_attacker_only"].astype(float),
            "attacker_only_success": merged["success_attacker_only"].astype(bool),
            "combined_success": merged["success_combined"].astype(bool),
            "attacker_only_completion_ns": merged[
                "external_completion_ns_attacker_only"
            ].astype(float),
            "combined_completion_ns": merged[
                "external_completion_ns_combined"
            ].astype(float),
            "attacker_only_turnaround_ns": merged[
                "turnaround_ns_attacker_only"
            ].astype(float),
            "combined_turnaround_ns": merged[
                "turnaround_ns_combined"
            ].astype(float),
            "excess_turnaround_ns": excess,
            "delayed": excess > AFFECTED_THRESHOLD_NS,
            "speedup": excess < -AFFECTED_THRESHOLD_NS,
            "failure_transition": (
                merged["success_attacker_only"].astype(bool).to_numpy()
                != merged["success_combined"].astype(bool).to_numpy()
            ),
        }
    )


# =============================================================================
# Black-box features
# =============================================================================


def run_lengths(mask: np.ndarray) -> list[int]:
    out: list[int] = []
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def run_count(mask: np.ndarray) -> int:
    return len(run_lengths(mask))


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def spectral_features(values: np.ndarray) -> dict[str, float]:
    if len(values) < 4:
        return {
            "spectral_dominant_bin_fraction": 0.0,
            "spectral_centroid_fraction": 0.0,
            "spectral_entropy": 0.0,
            "spectral_low_frequency_power_fraction": 0.0,
        }
    x = np.asarray(values, dtype=float) - float(np.mean(values))
    power = np.abs(np.fft.rfft(x)) ** 2
    if len(power):
        power[0] = 0.0
    total = float(np.sum(power))
    if total <= EPS:
        return {
            "spectral_dominant_bin_fraction": 0.0,
            "spectral_centroid_fraction": 0.0,
            "spectral_entropy": 0.0,
            "spectral_low_frequency_power_fraction": 0.0,
        }
    bins = np.arange(len(power), dtype=float)
    max_bin = max(len(power) - 1, 1)
    dominant = int(np.argmax(power)) / max_bin
    centroid = float(np.sum(bins * power) / total) / max_bin
    probs = power / total
    nz = probs[probs > EPS]
    entropy = -float(np.sum(nz * np.log(nz)))
    if len(power) > 1:
        entropy /= math.log(len(power))
    cutoff = max(2, int(math.ceil(len(power) * 0.25)))
    low = float(np.sum(power[1:cutoff]) / total)
    return {
        "spectral_dominant_bin_fraction": dominant,
        "spectral_centroid_fraction": centroid,
        "spectral_entropy": entropy,
        "spectral_low_frequency_power_fraction": low,
    }


def extract_features(trace: pd.DataFrame, probe_budget: int | None = None) -> dict[str, float]:
    t = trace.sort_values("probe_index")
    if probe_budget is not None:
        t = t.head(int(probe_budget))
    x = t["excess_turnaround_ns"].to_numpy(dtype=float)
    delayed = x > AFFECTED_THRESHOLD_NS
    speedup = x < -AFFECTED_THRESHOLD_NS
    failure = t["failure_transition"].to_numpy(dtype=bool)
    absx = np.abs(x)
    druns = run_lengths(delayed)
    sruns = run_lengths(speedup)

    def q(p: float) -> float:
        return float(np.quantile(x, p)) if len(x) else 0.0

    thirds = np.array_split(np.arange(len(x)), 3)
    third_vals = [
        float(np.mean(absx[idx])) if len(idx) else 0.0
        for idx in thirds
    ]

    row = {
        "probe_count": float(len(x)),
        "mean_excess_ns": float(np.mean(x)) if len(x) else 0.0,
        "median_excess_ns": float(np.median(x)) if len(x) else 0.0,
        "mean_abs_excess_ns": float(np.mean(absx)) if len(x) else 0.0,
        "std_excess_ns": float(np.std(x)) if len(x) else 0.0,
        "max_excess_ns": float(np.max(x)) if len(x) else 0.0,
        "min_excess_ns": float(np.min(x)) if len(x) else 0.0,
        "p10_excess_ns": q(0.10),
        "p25_excess_ns": q(0.25),
        "p50_excess_ns": q(0.50),
        "p75_excess_ns": q(0.75),
        "p90_excess_ns": q(0.90),
        "p95_excess_ns": q(0.95),
        "p99_excess_ns": q(0.99),
        "delayed_fraction": float(np.mean(delayed)) if len(x) else 0.0,
        "speedup_fraction": float(np.mean(speedup)) if len(x) else 0.0,
        "failure_transition_fraction": float(np.mean(failure)) if len(x) else 0.0,
        "cumulative_positive_excess_ns": float(np.sum(np.maximum(x, 0.0))),
        "cumulative_negative_magnitude_ns": float(np.sum(np.maximum(-x, 0.0))),
        "cumulative_abs_excess_ns": float(np.sum(absx)),
        "longest_delayed_run": float(max(druns) if druns else 0),
        "longest_speedup_run": float(max(sruns) if sruns else 0),
        "delayed_run_count": float(run_count(delayed)),
        "speedup_run_count": float(run_count(speedup)),
        "lag1_autocorrelation": autocorrelation(x, 1),
        "lag2_autocorrelation": autocorrelation(x, 2),
        "lag3_autocorrelation": autocorrelation(x, 3),
        "early_mean_abs_ns": third_vals[0],
        "middle_mean_abs_ns": third_vals[1],
        "late_mean_abs_ns": third_vals[2],
    }
    row.update(spectral_features(x))
    return row


# =============================================================================
# Dataset generation
# =============================================================================


def build_dataset(
    p27,
    MixedProtocolSimulator,
    *,
    schedules: list[BaseSchedule],
    conditions: list[Condition],
    fidelity_profiles: dict[str, FidelityProfile],
    repeats_per_schedule: int,
    observation_window_ns: float,
    seed: int,
):
    trace_parts: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []

    total = len(schedules) * len(conditions) * repeats_per_schedule
    done = 0

    # Attacker-only is independent of hidden condition.  Cache one baseline per repeat.
    attacker_cache: dict[int, pd.DataFrame] = {}
    baseline_condition = next(c for c in conditions if c.condition_id == "protocol_direct")
    for repeat_id in range(repeats_per_schedule):
        a_specs = attacker_specs(p27, repeat_id, observation_window_ns)
        a_req, *_ = run_mixed(
            p27,
            MixedProtocolSimulator,
            condition=baseline_condition,
            fidelity_profile=fidelity_profiles["medium_demand"],
            schedule_profile="opaque",
            repeat_id=repeat_id,
            run_kind="attacker_only",
            specs=list(a_specs),
        )
        attacker_cache[repeat_id] = a_req

    for schedule in schedules:
        for repeat_id in range(repeats_per_schedule):
            releases = release_times_for_schedule(
                schedule,
                repeat_id=repeat_id,
                seed=seed,
                observation_window_ns=observation_window_ns,
            )
            a_specs = attacker_specs(p27, repeat_id, observation_window_ns)
            v_specs = victim_specs(
                p27,
                releases,
                schedule.base_schedule_id,
                repeat_id,
            )

            for condition in conditions:
                profile = fidelity_profiles[condition.runtime_fidelity_profile]

                # Victim-only evaluator run.
                v_req, v_wait, v_intervals, v_stages, v_epr, v_gen = run_mixed(
                    p27,
                    MixedProtocolSimulator,
                    condition=condition,
                    fidelity_profile=profile,
                    schedule_profile=schedule.schedule_profile,
                    repeat_id=repeat_id,
                    run_kind="victim_only",
                    specs=list(v_specs),
                )

                combined_specs = sorted(
                    list(a_specs) + list(v_specs),
                    key=lambda s: (s.ready_ns, s.tenant, s.request_index),
                )
                c_req, c_wait, c_intervals, c_stages, c_epr, c_gen = run_mixed(
                    p27,
                    MixedProtocolSimulator,
                    condition=condition,
                    fidelity_profile=profile,
                    schedule_profile=schedule.schedule_profile,
                    repeat_id=repeat_id,
                    run_kind="combined",
                    specs=combined_specs,
                )

                trace_id = hashlib.sha256(
                    (
                        f"{schedule.base_schedule_id}|{repeat_id}|"
                        f"{condition.condition_id}|{seed}"
                    ).encode()
                ).hexdigest()[:20]

                trace = pair_attacker_trace(
                    attacker_cache[repeat_id],
                    c_req,
                    trace_id=trace_id,
                )
                trace_parts.append(trace)
                feats = extract_features(trace)
                feature_rows.append(
                    {
                        "trace_id": trace_id,
                        "base_schedule_id": schedule.base_schedule_id,
                        "repeat_id": repeat_id,
                        **feats,
                    }
                )

                slowdown = p27.victim_slowdown_metrics(v_req, c_req)
                truth = {
                    "trace_id": trace_id,
                    "base_schedule_id": schedule.base_schedule_id,
                    "repeat_id": repeat_id,
                    "schedule_profile": schedule.schedule_profile,
                    "condition_id": condition.condition_id,
                    "task_name": condition.task_name,
                    "evaluator_label": condition.evaluator_label,
                    "control": condition.control,
                    "victim_runtime_protocol": condition.victim_runtime_protocol,
                    "runtime_fidelity_profile": condition.runtime_fidelity_profile,
                    "victim_remote_operation_count": len(v_specs),
                    **slowdown,
                }
                truth_rows.append(truth)
                trial_rows.append({**truth, **feats})

                # Compact evaluator-only mechanism summary.
                victim_comb = c_req[c_req["tenant"] == "victim"]
                attacker_comb = c_req[c_req["tenant"] == "attacker"]
                evaluator_rows.append(
                    {
                        "trace_id": trace_id,
                        "condition_id": condition.condition_id,
                        "victim_runtime_protocol": condition.victim_runtime_protocol,
                        "runtime_fidelity_profile": condition.runtime_fidelity_profile,
                        "victim_mean_turnaround_ns": float(victim_comb["turnaround_ns"].mean()) if len(victim_comb) else 0.0,
                        "attacker_combined_mean_turnaround_ns": float(attacker_comb["turnaround_ns"].mean()) if len(attacker_comb) else 0.0,
                        "combined_epr_generation_jobs": int(len(c_gen)),
                        "combined_epr_state_events": int(len(c_epr)),
                        "combined_resource_wait_events": int(len(c_wait)),
                        "combined_resource_interval_count": int(len(c_intervals)),
                        "combined_stage_record_count": int(len(c_stages)),
                    }
                )

                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print(f"[Phase 3.5] Generated {done}/{total} traces")

    traces = pd.concat(trace_parts, ignore_index=True)
    features = pd.DataFrame(feature_rows)
    truth = pd.DataFrame(truth_rows)
    trials = pd.DataFrame(trial_rows)
    evaluator = pd.DataFrame(evaluator_rows)
    return traces, features, truth, trials, evaluator


# =============================================================================
# Group split and ML
# =============================================================================


def build_group_split(
    schedules: list[BaseSchedule],
    *,
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    """Profile-stratified grouped split with at least one train/test schedule per profile."""
    table = pd.DataFrame(
        [
            {
                "base_schedule_id": s.base_schedule_id,
                "schedule_profile": s.schedule_profile,
            }
            for s in schedules
        ]
    )

    rng = np.random.default_rng(stable_seed(seed, "group_split"))
    test_ids: set[str] = set()
    for profile, block in table.groupby("schedule_profile", sort=True):
        ids = block["base_schedule_id"].astype(str).to_numpy().copy()
        rng.shuffle(ids)
        if len(ids) < 2:
            raise ValueError(
                f"Schedule profile {profile!r} has fewer than two base schedules; "
                "cannot place it in both train and test."
            )
        n_test = int(round(len(ids) * test_size))
        n_test = max(1, min(len(ids) - 1, n_test))
        test_ids.update(ids[:n_test].tolist())

    out = table.copy()
    out["split"] = out["base_schedule_id"].map(
        lambda x: "test" if x in test_ids else "train"
    )
    return out.sort_values(["schedule_profile", "base_schedule_id"]).reset_index(drop=True)


def classifier_models(seed: int, rf_trees: int):
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_trees,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=seed,
            max_iter=250,
            l2_regularization=1e-3,
        ),
    }


def positive_probability(model, x: np.ndarray, positive_label: str) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    if positive_label not in classes:
        return None
    return probs[:, classes.index(positive_label)]


def evaluate_tasks(
    analysis: pd.DataFrame,
    split: pd.DataFrame,
    *,
    seed: int,
    rf_trees: int,
):
    split_map = split.set_index("base_schedule_id")["split"]
    data = analysis.copy()
    data["split"] = data["base_schedule_id"].map(split_map)

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for task_name, group in data.groupby("task_name", sort=True):
        train = group[group["split"] == "train"].copy()
        test = group[group["split"] == "test"].copy()
        labels = sorted(group["evaluator_label"].unique())
        chance = 1.0 / len(labels)
        x_train = train[FEATURE_COLUMNS].astype(float).to_numpy()
        x_test = test[FEATURE_COLUMNS].astype(float).to_numpy()
        y_train = train["evaluator_label"].astype(str).to_numpy()
        y_test = test["evaluator_label"].astype(str).to_numpy()

        for model_name, model in classifier_models(seed, rf_trees).items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            bal = balanced_accuracy_score(y_test, y_pred)
            mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            auc = math.nan
            if len(labels) == 2:
                positive_label = labels[-1]
                prob = positive_probability(model, x_test, positive_label)
                if prob is not None and len(np.unique(y_test)) == 2:
                    y_bin = (y_test == positive_label).astype(int)
                    auc = float(roc_auc_score(y_bin, prob))

            metric_rows.append(
                {
                    "task_name": task_name,
                    "model_name": model_name,
                    "sample_count": len(test),
                    "class_count": len(labels),
                    "chance_accuracy": chance,
                    "accuracy": acc,
                    "balanced_accuracy": bal,
                    "macro_f1": mf1,
                    "binary_roc_auc": auc,
                }
            )

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_test)
                model_classes = list(model.classes_)
            else:
                probs = None
                model_classes = []

            for i, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
                out = {
                    "trace_id": row["trace_id"],
                    "base_schedule_id": row["base_schedule_id"],
                    "repeat_id": int(row["repeat_id"]),
                    "schedule_profile": row["schedule_profile"],
                    "task_name": task_name,
                    "true_label": str(y_test[i]),
                    "predicted_label": str(y_pred[i]),
                    "correct": bool(y_test[i] == y_pred[i]),
                    "model_name": model_name,
                }
                if probs is not None:
                    for j, cls in enumerate(model_classes):
                        out[f"prob::{cls}"] = float(probs[i, j])
                pred_rows.append(out)

            cm = confusion_matrix(y_test, y_pred, labels=labels)
            for i, true_label in enumerate(labels):
                denom = cm[i].sum()
                for j, pred_label in enumerate(labels):
                    confusion_rows.append(
                        {
                            "task_name": task_name,
                            "model_name": model_name,
                            "true_label": true_label,
                            "predicted_label": pred_label,
                            "count": int(cm[i, j]),
                            "true_normalized_fraction": float(cm[i, j] / denom) if denom else 0.0,
                        }
                    )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(pred_rows),
        pd.DataFrame(confusion_rows),
    )


def probe_budget_metrics(
    traces: pd.DataFrame,
    truth: pd.DataFrame,
    split: pd.DataFrame,
    *,
    budgets: list[int],
    seed: int,
    rf_trees: int,
):
    split_map = split.set_index("base_schedule_id")["split"]
    truth_index = truth.set_index("trace_id")
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        feat_rows = []
        for trace_id, trace in traces.groupby("trace_id", sort=False):
            meta = truth_index.loc[trace_id]
            feat_rows.append(
                {
                    "trace_id": trace_id,
                    "base_schedule_id": meta["base_schedule_id"],
                    "task_name": meta["task_name"],
                    "evaluator_label": meta["evaluator_label"],
                    **extract_features(trace, probe_budget=budget),
                }
            )
        data = pd.DataFrame(feat_rows)
        data["split"] = data["base_schedule_id"].map(split_map)

        for task_name, group in data.groupby("task_name", sort=True):
            train = group[group["split"] == "train"]
            test = group[group["split"] == "test"]
            x_train = train[FEATURE_COLUMNS].astype(float).to_numpy()
            x_test = test[FEATURE_COLUMNS].astype(float).to_numpy()
            y_train = train["evaluator_label"].astype(str).to_numpy()
            y_test = test["evaluator_label"].astype(str).to_numpy()
            model = RandomForestClassifier(
                n_estimators=rf_trees,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            labels = sorted(group["evaluator_label"].unique())
            auc = math.nan
            if len(labels) == 2:
                positive = labels[-1]
                prob = positive_probability(model, x_test, positive)
                if prob is not None and len(np.unique(y_test)) == 2:
                    auc = float(roc_auc_score((y_test == positive).astype(int), prob))
            rows.append(
                {
                    "task_name": task_name,
                    "model_name": "random_forest",
                    "probe_budget": int(budget),
                    "sample_count": len(test),
                    "chance_accuracy": 1.0 / len(labels),
                    "accuracy": accuracy_score(y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                    "binary_roc_auc": auc,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Summaries and validation
# =============================================================================


def build_task_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_name, group in metrics.groupby("task_name", sort=True):
        best = group.sort_values(
            ["accuracy", "macro_f1"], ascending=[False, False]
        ).iloc[0]
        rows.append(
            {
                "task_name": task_name,
                "best_model": best["model_name"],
                "class_count": int(best["class_count"]),
                "chance_accuracy": float(best["chance_accuracy"]),
                "best_accuracy": float(best["accuracy"]),
                "best_balanced_accuracy": float(best["balanced_accuracy"]),
                "best_macro_f1": float(best["macro_f1"]),
                "best_binary_roc_auc": float(best["binary_roc_auc"]) if np.isfinite(best["binary_roc_auc"]) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_signal_summary(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(
            ["task_name", "evaluator_label", "condition_id", "schedule_profile"],
            sort=True,
        )
        .agg(
            trace_count=("trace_id", "count"),
            mean_abs_excess_ns=("mean_abs_excess_ns", "mean"),
            mean_signed_excess_ns=("mean_excess_ns", "mean"),
            delayed_fraction=("delayed_fraction", "mean"),
            speedup_fraction=("speedup_fraction", "mean"),
            failure_transition_fraction=("failure_transition_fraction", "mean"),
            victim_mean_request_slowdown=("victim_mean_request_slowdown", "mean"),
            victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
        )
        .reset_index()
    )


def build_validations(
    p27,
    *,
    traces: pd.DataFrame,
    features: pd.DataFrame,
    truth: pd.DataFrame,
    schedules: list[BaseSchedule],
    conditions: list[Condition],
    split: pd.DataFrame,
    fidelity_profiles: dict[str, FidelityProfile],
) -> pd.DataFrame:
    a: list[ValidationAssertion] = []

    def add(group, name, passed, expected, observed, details=""):
        a.append(
            ValidationAssertion(
                group,
                name,
                bool(passed),
                str(expected),
                str(observed),
                str(details),
            )
        )

    add(
        "blackbox_boundary",
        "attacker_trace_schema_exact",
        list(traces.columns) == ATTACKER_VISIBLE_COLUMNS,
        ATTACKER_VISIBLE_COLUMNS,
        list(traces.columns),
    )
    forbidden_feature_tokens = (
        "victim",
        "condition",
        "task",
        "label",
        "protocol",
        "fidelity",
        "resource",
        "epr",
        "scenario",
        "schedule_profile",
    )
    bad_features = [
        c
        for c in FEATURE_COLUMNS
        if any(tok in c.lower() for tok in forbidden_feature_tokens)
    ]
    add(
        "blackbox_boundary",
        "model_features_exclude_evaluator_state",
        len(bad_features) == 0,
        [],
        bad_features,
    )
    encoded = traces["trace_id"].astype(str).str.contains(
        "direct|entang|low|medium|high|fidelity|protocol|victim", case=False, regex=True
    )
    add(
        "blackbox_boundary",
        "trace_ids_are_opaque",
        not bool(encoded.any()),
        0,
        int(encoded.sum()),
    )

    # Schedule crossing and split discipline.
    expected_conditions = len(conditions)
    per_group = truth.groupby(["base_schedule_id", "repeat_id"])["condition_id"].nunique()
    add(
        "schedule",
        "every_schedule_repeat_crossed_with_all_conditions",
        int(per_group.min()) == expected_conditions and int(per_group.max()) == expected_conditions,
        expected_conditions,
        f"min={int(per_group.min())}, max={int(per_group.max())}",
    )
    op_counts = truth.groupby(["base_schedule_id", "repeat_id"])["victim_remote_operation_count"].nunique()
    add(
        "schedule",
        "logical_remote_operation_count_fixed_across_conditions",
        int(op_counts.max()) == 1,
        1,
        int(op_counts.max()),
    )
    add(
        "schedule",
        "all_three_schedule_profiles_present",
        set(s.schedule_profile for s in schedules)
        == {"sparse_periodic", "dense_periodic", "synchronization_bursty"},
        {"sparse_periodic", "dense_periodic", "synchronization_bursty"},
        set(s.schedule_profile for s in schedules),
    )

    train_ids = set(split.loc[split["split"] == "train", "base_schedule_id"])
    test_ids = set(split.loc[split["split"] == "test", "base_schedule_id"])
    add(
        "evaluation",
        "group_split_has_no_base_schedule_overlap",
        len(train_ids & test_ids) == 0,
        [],
        sorted(train_ids & test_ids),
    )
    train_profiles = set(split.loc[split["split"] == "train", "schedule_profile"])
    test_profiles = set(split.loc[split["split"] == "test", "schedule_profile"])
    all_profiles = {"sparse_periodic", "dense_periodic", "synchronization_bursty"}
    add(
        "evaluation",
        "schedule_profiles_present_in_train_and_test",
        train_profiles == all_profiles and test_profiles == all_profiles,
        sorted(all_profiles),
        f"train={sorted(train_profiles)}, test={sorted(test_profiles)}",
    )
    add(
        "evaluation",
        "feature_and_truth_rows_pair_one_to_one",
        features["trace_id"].nunique() == truth["trace_id"].nunique() == len(features) == len(truth),
        f"equal unique rows ({len(truth)})",
        f"features={len(features)}, truth={len(truth)}",
    )

    # Task and condition structure.
    task_counts = pd.Series([c.task_name for c in conditions]).value_counts().to_dict()
    add(
        "design",
        "exactly_ten_conditions_present",
        len(conditions) == 10,
        10,
        len(conditions),
    )
    add(
        "design",
        "protocol_task_has_two_classes",
        task_counts.get(PROTOCOL_TASK, 0) == 2,
        2,
        task_counts.get(PROTOCOL_TASK, 0),
    )
    add(
        "design",
        "protocol_control_has_two_classes",
        task_counts.get(PROTOCOL_CONTROL_TASK, 0) == 2,
        2,
        task_counts.get(PROTOCOL_CONTROL_TASK, 0),
    )
    add(
        "design",
        "fidelity_task_has_three_classes",
        task_counts.get(FIDELITY_TASK, 0) == 3,
        3,
        task_counts.get(FIDELITY_TASK, 0),
    )
    add(
        "design",
        "fidelity_control_has_three_classes",
        task_counts.get(FIDELITY_CONTROL_TASK, 0) == 3,
        3,
        task_counts.get(FIDELITY_CONTROL_TASK, 0),
    )

    # Runtime semantics / negative controls.
    protocol_control_runtime = {
        (c.victim_runtime_protocol, c.runtime_fidelity_profile)
        for c in conditions
        if c.task_name == PROTOCOL_CONTROL_TASK
    }
    add(
        "negative_control",
        "protocol_label_control_runtime_identical",
        len(protocol_control_runtime) == 1,
        1,
        len(protocol_control_runtime),
    )
    fidelity_control_runtime = {
        (c.victim_runtime_protocol, c.runtime_fidelity_profile)
        for c in conditions
        if c.task_name == FIDELITY_CONTROL_TASK
    }
    add(
        "negative_control",
        "fidelity_label_control_runtime_identical",
        len(fidelity_control_runtime) == 1,
        1,
        len(fidelity_control_runtime),
    )
    protocol_runtimes = {
        c.victim_runtime_protocol
        for c in conditions
        if c.task_name == PROTOCOL_TASK
    }
    add(
        "architecture",
        "protocol_task_uses_both_phase2_07_realizations",
        protocol_runtimes == {p27.DIRECT_PROTOCOL, p27.ENTANGLED_PROTOCOL},
        {p27.DIRECT_PROTOCOL, p27.ENTANGLED_PROTOCOL},
        protocol_runtimes,
    )
    fidelity_runtimes = {
        c.victim_runtime_protocol
        for c in conditions
        if c.task_name == FIDELITY_TASK
    }
    add(
        "architecture",
        "fidelity_task_uses_entanglement_assisted_runtime_only",
        fidelity_runtimes == {p27.ENTANGLED_PROTOCOL},
        {p27.ENTANGLED_PROTOCOL},
        fidelity_runtimes,
    )
    add(
        "architecture",
        "observer_protocol_fixed_direct_coherent",
        True,
        p27.DIRECT_PROTOCOL,
        p27.DIRECT_PROTOCOL,
    )
    add(
        "architecture",
        "phase2_07_direct_observer_timing_retained",
        abs(float(p27.DIRECT_CRITICAL_NS) - 150.0) < 1e-9
        and abs(float(p27.RESET_NS) - 120.0) < 1e-9,
        "direct critical=150 ns, cleanup=120 ns",
        f"critical={p27.DIRECT_CRITICAL_NS}, cleanup={p27.RESET_NS}",
    )

    medium = fidelity_profiles["medium_demand"]
    add(
        "architecture",
        "medium_service_profile_equals_phase2_07_entangled_baseline",
        abs(medium.readout_ns - p27.ENT_READOUT_NS) < 1e-9
        and abs(medium.feedforward_ns - p27.ENT_FEEDFORWARD_NS) < 1e-9
        and abs(medium.reset_ns - p27.RESET_NS) < 1e-9
        and abs(medium.epr_link_generation_ns - p27.EPR_LINK_GENERATION_NS) < 1e-9,
        "Phase-2.7 entangled baseline",
        asdict(medium),
    )

    low = fidelity_profiles["low_demand"]
    high = fidelity_profiles["high_demand"]
    add(
        "architecture",
        "service_demand_profiles_order_resource_cost",
        low.readout_ns < medium.readout_ns < high.readout_ns
        and low.epr_link_generation_ns < medium.epr_link_generation_ns < high.epr_link_generation_ns
        and low.reset_ns < medium.reset_ns < high.reset_ns,
        "low < medium < high",
        (
            f"readout={low.readout_ns},{medium.readout_ns},{high.readout_ns}; "
            f"epr_link={low.epr_link_generation_ns},{medium.epr_link_generation_ns},{high.epr_link_generation_ns}; "
            f"reset={low.reset_ns},{medium.reset_ns},{high.reset_ns}"
        ),
    )

    # Black-box operational sanity.
    add(
        "execution",
        "all_attacker_only_requests_succeed",
        bool(traces["attacker_only_success"].all()),
        True,
        bool(traces["attacker_only_success"].all()),
    )
    add(
        "execution",
        "all_combined_attacker_requests_succeed",
        bool(traces["combined_success"].all()),
        True,
        bool(traces["combined_success"].all()),
    )
    add(
        "execution",
        "no_failure_transition_channel_required",
        int(traces["failure_transition"].sum()) == 0,
        0,
        int(traces["failure_transition"].sum()),
    )

    # Exact label-control feature equality: because each base schedule/repeat is
    # crossed with identical runtime controls, paired control feature vectors
    # should be identical to numerical precision.
    merged = features.merge(
        truth[["trace_id", "base_schedule_id", "repeat_id", "task_name", "evaluator_label"]],
        on=["trace_id", "base_schedule_id", "repeat_id"],
        validate="one_to_one",
    )
    max_spans = {}
    for task in (PROTOCOL_CONTROL_TASK, FIDELITY_CONTROL_TASK):
        g = merged[merged["task_name"] == task]
        spans = []
        for _, block in g.groupby(["base_schedule_id", "repeat_id"]):
            arr = block[FEATURE_COLUMNS].astype(float).to_numpy()
            spans.append(float(np.max(np.ptp(arr, axis=0))) if len(arr) else 0.0)
        max_spans[task] = max(spans) if spans else math.inf
    add(
        "negative_control",
        "protocol_label_control_blackbox_features_identical",
        max_spans[PROTOCOL_CONTROL_TASK] <= 1e-9,
        "<=1e-9",
        max_spans[PROTOCOL_CONTROL_TASK],
    )
    add(
        "negative_control",
        "fidelity_label_control_blackbox_features_identical",
        max_spans[FIDELITY_CONTROL_TASK] <= 1e-9,
        "<=1e-9",
        max_spans[FIDELITY_CONTROL_TASK],
    )

    return pd.DataFrame([asdict(x) for x in a])


# =============================================================================
# Experiment driver
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p27, p27_source = load_phase2_07_module()
    MixedProtocolSimulator = build_mixed_simulator_class(p27)
    fidelity_profiles = build_fidelity_profiles(p27)
    conditions = build_conditions(p27)
    schedules = build_base_schedules(seed=args.seed, count=args.base_schedules)

    print(
        f"[Phase 3.5] schedules={len(schedules)}, "
        f"conditions={len(conditions)}, repeats={args.repeats_per_schedule}"
    )
    print(f"[Phase 3.5] Reusing Phase-2.7 simulator: {p27_source}")
    print(
        "[Phase 3.5] Attacker protocol is fixed to direct coherent; "
        "victim protocol/service class is evaluator-only."
    )

    traces, features, truth, trials, evaluator = build_dataset(
        p27,
        MixedProtocolSimulator,
        schedules=schedules,
        conditions=conditions,
        fidelity_profiles=fidelity_profiles,
        repeats_per_schedule=args.repeats_per_schedule,
        observation_window_ns=args.observation_window_ns,
        seed=args.seed,
    )

    split = build_group_split(
        schedules,
        seed=args.seed,
        test_size=args.test_size,
    )
    analysis = features.merge(
        truth,
        on=["trace_id", "base_schedule_id", "repeat_id"],
        validate="one_to_one",
    )

    metrics, predictions, confusion = evaluate_tasks(
        analysis,
        split,
        seed=args.seed,
        rf_trees=args.rf_trees,
    )
    task_summary = build_task_summary(metrics)

    max_probes = int(traces.groupby("trace_id")["probe_index"].count().max())
    budgets = sorted(
        {
            min(max_probes, int(x.strip()))
            for x in args.probe_budgets.split(",")
            if x.strip() and int(x.strip()) > 0
        }
    )
    if max_probes not in budgets:
        budgets.append(max_probes)
    budget_metrics = probe_budget_metrics(
        traces,
        truth,
        split,
        budgets=budgets,
        seed=args.seed,
        rf_trees=args.rf_trees,
    )

    signal_summary = build_signal_summary(trials)
    validation = build_validations(
        p27,
        traces=traces,
        features=features,
        truth=truth,
        schedules=schedules,
        conditions=conditions,
        split=split,
        fidelity_profiles=fidelity_profiles,
    )
    validation_summary = pd.DataFrame(
        [
            {
                "assertion_count": len(validation),
                "passed_assertions": int(validation["passed"].sum()),
                "failed_assertions": int((~validation["passed"]).sum()),
                "all_passed": bool(validation["passed"].all()),
            }
        ]
    )

    schedule_table = pd.DataFrame([asdict(s) for s in schedules])
    condition_table = pd.DataFrame([asdict(c) for c in conditions])
    profile_table = pd.DataFrame([asdict(x) for x in fidelity_profiles.values()])

    # Save outputs.
    traces.to_csv(output_dir / "phase3_05_attacker_visible_trace.csv", index=False)
    features.to_csv(output_dir / "phase3_05_trace_features.csv", index=False)
    truth.to_csv(output_dir / "phase3_05_evaluator_ground_truth.csv", index=False)
    trials.to_csv(output_dir / "phase3_05_trial_summary.csv", index=False)
    evaluator.to_csv(output_dir / "phase3_05_evaluator_mechanism_summary.csv", index=False)
    schedule_table.to_csv(output_dir / "phase3_05_base_schedule_table.csv", index=False)
    condition_table.to_csv(output_dir / "phase3_05_condition_table.csv", index=False)
    profile_table.to_csv(output_dir / "phase3_05_service_profile_table.csv", index=False)
    split.to_csv(output_dir / "phase3_05_group_split.csv", index=False)
    metrics.to_csv(output_dir / "phase3_05_inference_metrics.csv", index=False)
    predictions.to_csv(output_dir / "phase3_05_inference_predictions.csv", index=False)
    confusion.to_csv(output_dir / "phase3_05_confusion_matrix.csv", index=False)
    task_summary.to_csv(output_dir / "phase3_05_protocol_fidelity_summary.csv", index=False)
    budget_metrics.to_csv(output_dir / "phase3_05_probe_budget_metrics.csv", index=False)
    signal_summary.to_csv(output_dir / "phase3_05_signal_summary.csv", index=False)
    validation.to_csv(output_dir / "phase3_05_validation_assertions.csv", index=False)
    validation_summary.to_csv(output_dir / "phase3_05_validation_summary.csv", index=False)

    manifest = {
        "experiment": "phase3_05_protocol_fidelity_demand_inference",
        "seed": args.seed,
        "phase2_07_source": str(p27_source),
        "output_dir": str(output_dir),
        "observer_protocol": p27.DIRECT_PROTOCOL,
        "logical_operation": p27.LOGICAL_OPERATION,
        "base_schedule_count": len(schedules),
        "schedule_profiles": sorted(schedule_table["schedule_profile"].unique().tolist()),
        "condition_count": len(conditions),
        "tasks": sorted(condition_table["task_name"].unique().tolist()),
        "repeats_per_schedule": args.repeats_per_schedule,
        "observation_window_ns": args.observation_window_ns,
        "attacker_first_release_ns": float(p27.ATTACKER_FIRST_RELEASE_NS),
        "attacker_period_ns": float(p27.ATTACKER_PERIOD_NS),
        "test_size": args.test_size,
        "rf_trees": args.rf_trees,
        "probe_budgets": budgets,
        "trace_count": int(traces["trace_id"].nunique()),
        "probe_row_count": int(len(traces)),
        "training_base_schedule_count": int((split["split"] == "train").sum()),
        "test_base_schedule_count": int((split["split"] == "test").sum()),
        "feature_columns": FEATURE_COLUMNS,
        "attacker_visible_columns": ATTACKER_VISIBLE_COLUMNS,
        "validation_assertions": len(validation),
        "validation_passed": int(validation["passed"].sum()),
        "all_validation_passed": bool(validation["passed"].all()),
        "notes": [
            "This is an inference / privacy-leakage characterization experiment, not a defense experiment.",
            "Observer protocol remains fixed to direct coherent for every trace.",
            "Hidden protocol/service label is evaluator-only.",
            "Base victim schedules are crossed with all labels and conditions.",
            "Grouped train/test splitting is by base schedule instance.",
            "Service-demand tiers are controlled simulation parameters, not measured commercial-hardware fidelity levels.",
            "Label-only controls remove physical semantics while retaining evaluator labels.",
        ],
    }
    (output_dir / "phase3_05_run_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\n[Phase 3.5] Validation")
    print(validation_summary.to_string(index=False))
    print("\n[Phase 3.5] Best held-out result per task")
    print(task_summary.to_string(index=False))
    print(f"\n[Phase 3.5] Wrote outputs to: {output_dir}")

    if args.fail_on_validation_error and not bool(validation["passed"].all()):
        failed = validation.loc[~validation["passed"], ["assertion_name", "observed"]]
        raise AssertionError(
            "Phase 3.5 validation failed:\n" + failed.to_string(index=False)
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.5 — Protocol / Fidelity-Demand Inference"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--base-schedules",
        type=int,
        default=DEFAULT_BASE_SCHEDULES,
        help="Number of distinct hidden logical release schedules.",
    )
    parser.add_argument(
        "--repeats-per-schedule",
        type=int,
        default=DEFAULT_REPEATS_PER_SCHEDULE,
    )
    parser.add_argument(
        "--observation-window-ns",
        type=float,
        default=DEFAULT_OBSERVATION_WINDOW_NS,
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--rf-trees", type=int, default=DEFAULT_RF_TREES)
    parser.add_argument(
        "--probe-budgets",
        default="8,16,24,32,40",
        help="Comma-separated prefix probe budgets; full length is appended automatically.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_schedules < 6:
        raise ValueError("--base-schedules must be >= 6")
    if args.repeats_per_schedule < 1:
        raise ValueError("--repeats-per-schedule must be >= 1")
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.5")
    if args.rf_trees < 10:
        raise ValueError("--rf-trees must be >= 10")
    run_experiment(args)


if __name__ == "__main__":
    main()
