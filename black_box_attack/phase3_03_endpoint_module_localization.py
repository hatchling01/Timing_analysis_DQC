#!/usr/bin/env python3
"""
Phase 3.3 — Endpoint / Module Localization
==========================================

Research question
-----------------
Phase 3.1 established that attacker-only timing can reveal whether remote
communication occurs and how much communication the victim performs.
Phase 3.2 established that the same black-box timing can reveal when the victim
changes execution regime.

Phase 3.3 asks the next spatial question:

    Can an attacker determine WHICH remote endpoint / module is active using
    only the timing of its own deliberately targeted remote probes?

The victim communicates through one hidden remote-module region for an entire
trace.  The attacker source is fixed at ``module_0`` and interleaves probes to
candidate remote modules ``module_1`` ... ``module_N``.  The attacker knows the
module it chose for each of its own probes, but it receives no victim module,
resource-wait, stage, EPR-state, or blocking-owner information.

Architecture foundation
-----------------------
This experiment reuses the validated Phase-2.7 protocol simulator rather than
creating a new monolithic timing model.  Two protocol contexts are evaluated:

* direct_coherent_remote_cx
* entanglement_assisted_remote_cx

Both retain Phase-2.7 normalization (150 ns nominal critical latency after the
protocol prerequisite is ready and 120 ns cleanup).

Spatial resource model
----------------------
The Phase-2.7 simulator represents one contention domain at a time.  Phase 3.3
builds a multi-region black-box scan by assigning each attacker probe to one
candidate region and evaluating that probe against the corresponding physical
contention domain.  Only one attacker probe is released every 420 ns, so the
candidate streams are temporally interleaved rather than concurrent.

Three spatial-visibility modes are provided:

1. ``localized_only``
   * matching probe/victim region -> every protocol resource used by that route
     is shared;
   * nonmatching probe region -> tenant-private resources.
   This is the clean causal positive control.

2. ``hybrid_local_global`` (primary realistic localization context)
   * matching region -> all protocol resources shared;
   * nonmatching regions -> only a protocol-specific global/common stack is
     shared.
   Direct coherent uses the Phase-2.7 interconnect stack
   (switch path + quantum link) as common background.
   Entanglement-assisted uses the measurement/feedforward stack as common
   background, while endpoint/EPR-management state remains route-local.
   Thus all probe regions may see some activity, but the physically matching
   region should carry an additional local signature.

3. ``global_only_control``
   * every probe region sees exactly the same protocol-specific common stack,
     independent of the hidden victim region.
   The victim-location label is therefore intentionally unobservable and
   localization should return to the 1/N chance baseline.

This control is critical: remote-activity leakage alone must not be mistaken
for spatial localization.

Victim schedules
----------------
Base communication schedules are generated independently of the hidden module
label and then CROSSED with every candidate victim region.  The same base
schedule/repeat is also reused across both protocols and all spatial modes.
Three schedule profiles are used:

* sparse_periodic
* dense_periodic
* synchronization_bursty

Train/test splitting is grouped by BASE SCHEDULE INSTANCE, so every spatial
copy and timing repeat of one schedule remains in the same split.  A model
therefore cannot memorize one timing realization of a victim schedule and see
another spatial copy of that same schedule in the opposite partition.

Attacker probe policy
---------------------
The attacker retains the 420 ns Phase-2/3 probe period.  Candidate modules are
sampled in balanced, randomly permuted blocks: every block of N probes contains
one probe to each candidate module.  The block permutations depend only on the
base schedule instance/repeat, never on the hidden victim module.

Primary inference tasks
-----------------------
* N-way hidden endpoint/module classification.
* Top-2 localization accuracy.
* Interpretable argmax spatial-score baselines.
* Accuracy versus total probe budget.
* Protocol comparison.
* Schedule-profile comparison.
* Spatial signal matrix: attacker probe region x hidden victim region.
* Global-only negative control.

Strict black-box boundary
-------------------------
Attacker-visible rows contain only:

* opaque trace id,
* attacker's known protocol context,
* probe index,
* attacker-chosen probe region,
* release/completion/turnaround of the attacker's own requests,
* paired excess turnaround,
* success/failure/delay flags.

Hidden victim region, victim schedule profile, victim releases, Phase-2.7
scenario ids, shared-resource names, resource waits, EPR state, and evaluator
attribution are stored separately.

Default output directory
------------------------
blackbox_window_results/phase3/phase3.3/

Run
---
From the repository directory containing
``phase2_07_remote_protocol_comparison.py``:

    python phase3_03_endpoint_module_localization.py

Quick smoke test:

    python phase3_03_endpoint_module_localization.py \
        --base-instances 6 \
        --repeats-per-instance 1 \
        --observation-window-ns 6000 \
        --rf-trees 100 \
        --output-dir /tmp/phase3_03_smoke

Notes
-----
* Stage durations and capacities are controlled simulation parameters, not
  vendor measurements.
* Phase 3.3 localizes one persistent active communication region per trace.
  Time-varying multi-edge reconstruction belongs to Phase 3.4.
* ``global_only_control`` intentionally destroys spatial identifiability while
  preserving non-spatial communication timing.  Chance performance there is a
  positive causal control, not a failed attack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Global settings
# =============================================================================

DEFAULT_SEED = 3301
DEFAULT_BASE_INSTANCES = 18
DEFAULT_REPEATS = 2
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_TEST_SIZE = 0.33
DEFAULT_REGION_COUNT = 4
DEFAULT_RF_TREES = 500
DEFAULT_OUTPUT_DIR = Path("blackbox_window_results") / "phase3" / "phase3.3"

AFFECTED_THRESHOLD_NS = 1e-9
EPS = 1e-12

SCHEDULE_PROFILES = (
    "sparse_periodic",
    "dense_periodic",
    "synchronization_bursty",
)

SPATIAL_MODES = (
    "localized_only",
    "hybrid_local_global",
    "global_only_control",
)

ATTACKER_VISIBLE_COLUMNS = (
    "trace_id",
    "protocol_context",
    "probe_index",
    "probe_region",
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
)

PER_REGION_FEATURE_SUFFIXES = (
    "probe_count",
    "mean_excess_ns",
    "median_excess_ns",
    "mean_abs_excess_ns",
    "std_excess_ns",
    "max_excess_ns",
    "p90_excess_ns",
    "p95_excess_ns",
    "delayed_fraction",
    "speedup_fraction",
    "failure_fraction",
    "cumulative_positive_excess_ns",
    "cumulative_abs_excess_ns",
    "longest_delayed_run",
    "delayed_run_count",
    "lag1_autocorrelation",
    "early_mean_abs_ns",
    "late_mean_abs_ns",
)


# =============================================================================
# Phase-2.7 loader
# =============================================================================


def load_phase2_07_module():
    candidates = [
        Path(__file__).resolve().parent / "phase2_07_remote_protocol_comparison.py",
        Path.cwd() / "phase2_07_remote_protocol_comparison.py",
        Path(__file__).resolve().parent.parent / "phase2_07_remote_protocol_comparison.py",
        Path("/mnt/data/phase2_07_remote_protocol_comparison.py"),
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        searched = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            "Could not locate phase2_07_remote_protocol_comparison.py.\n"
            "Phase 3.3 intentionally reuses the validated Phase-2.7 simulator.\n"
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
class SpatialContext:
    protocol_context: str
    protocol_name: str
    spatial_mode: str
    matching_scenario_id: str
    nonmatching_scenario_id: str
    common_stack_description: str


@dataclass(frozen=True)
class ValidationAssertion:
    validation_group: str
    assertion_name: str
    passed: bool
    expected: str
    observed: str
    details: str = ""


# =============================================================================
# Deterministic helpers
# =============================================================================


def stable_seed(*parts: Any, modulus: int = 2**32 - 1) -> int:
    token = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % modulus


def opaque_trace_id(*parts: Any) -> str:
    token = "|".join(str(x) for x in parts).encode()
    return hashlib.sha256(token).hexdigest()[:20]


def region_names(count: int) -> tuple[str, ...]:
    return tuple(f"module_{i}" for i in range(1, count + 1))


# =============================================================================
# Base victim schedules
# =============================================================================


def make_base_schedules(
    *,
    seed: int,
    base_instances: int,
) -> list[BaseSchedule]:
    if base_instances < len(SCHEDULE_PROFILES) * 2:
        raise ValueError(
            f"--base-instances must be at least {len(SCHEDULE_PROFILES) * 2} "
            "so schedule profiles can appear in both grouped splits."
        )

    rows: list[BaseSchedule] = []
    for idx in range(base_instances):
        profile = SCHEDULE_PROFILES[idx % len(SCHEDULE_PROFILES)]
        rng = np.random.default_rng(stable_seed(seed, "base_schedule", idx, profile))
        base_phase = float(rng.uniform(80.0, 420.0))

        if profile == "sparse_periodic":
            interval = float(rng.uniform(900.0, 1_220.0))
            burst_period = 0.0
            burst_size = 1
            burst_spacing = 0.0
        elif profile == "dense_periodic":
            interval = float(rng.uniform(430.0, 570.0))
            burst_period = 0.0
            burst_size = 1
            burst_spacing = 0.0
        elif profile == "synchronization_bursty":
            interval = 0.0
            burst_period = float(rng.uniform(1_350.0, 1_850.0))
            burst_size = int(rng.integers(3, 7))
            burst_spacing = float(rng.uniform(55.0, 95.0))
        else:
            raise ValueError(profile)

        rows.append(
            BaseSchedule(
                base_schedule_id=f"base_{idx:04d}",
                base_index=idx,
                schedule_profile=profile,
                base_phase_ns=base_phase,
                interval_ns=interval,
                burst_period_ns=burst_period,
                burst_size=burst_size,
                burst_spacing_ns=burst_spacing,
            )
        )
    return rows


def generate_victim_releases(
    schedule: BaseSchedule,
    *,
    repeat_id: int,
    seed: int,
    observation_window_ns: float,
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed(seed, schedule.base_schedule_id, repeat_id, "victim_repeat")
    )
    phase = schedule.base_phase_ns + float(rng.uniform(-12.0, 12.0))

    if schedule.schedule_profile in {"sparse_periodic", "dense_periodic"}:
        releases = np.arange(
            phase,
            observation_window_ns,
            schedule.interval_ns,
            dtype=float,
        )
    elif schedule.schedule_profile == "synchronization_bursty":
        values: list[float] = []
        base = phase
        while base < observation_window_ns:
            for j in range(schedule.burst_size):
                values.append(base + j * schedule.burst_spacing_ns)
            base += schedule.burst_period_ns
        releases = np.asarray(values, dtype=float)
    else:
        raise ValueError(schedule.schedule_profile)

    releases = releases[(releases >= 0.0) & (releases < observation_window_ns)]
    if len(releases):
        releases = releases + rng.uniform(-4.0, 4.0, size=len(releases))
        releases = np.clip(releases, 0.0, observation_window_ns - 1e-6)
    return np.sort(releases)


def make_victim_specs(
    p27,
    *,
    releases: np.ndarray,
    schedule: BaseSchedule,
    repeat_id: int,
) -> list[Any]:
    return [
        p27.RequestSpec(
            request_id=(
                f"victim::{schedule.base_schedule_id}::repeat_{repeat_id:02d}::{i}"
            ),
            tenant="victim",
            ready_ns=float(t),
            request_index=i,
            workload_name=schedule.schedule_profile,
            trial_id=repeat_id,
        )
        for i, t in enumerate(releases)
    ]


# =============================================================================
# Attacker target schedule
# =============================================================================


def attacker_probe_plan(
    p27,
    *,
    regions: tuple[str, ...],
    schedule: BaseSchedule,
    repeat_id: int,
    seed: int,
    observation_window_ns: float,
) -> pd.DataFrame:
    releases = np.arange(
        p27.ATTACKER_FIRST_RELEASE_NS,
        observation_window_ns,
        p27.ATTACKER_PERIOD_NS,
        dtype=float,
    )
    rng = np.random.default_rng(
        stable_seed(seed, schedule.base_schedule_id, repeat_id, "attacker_region_plan")
    )

    assignments: list[str] = []
    while len(assignments) < len(releases):
        block = list(regions)
        rng.shuffle(block)
        assignments.extend(block)
    assignments = assignments[: len(releases)]

    return pd.DataFrame(
        {
            "probe_index": np.arange(len(releases), dtype=int),
            "release_ns": releases.astype(float),
            "probe_region": assignments,
        }
    )


def make_attacker_specs_for_region(
    p27,
    *,
    plan: pd.DataFrame,
    probe_region: str,
    schedule: BaseSchedule,
    repeat_id: int,
) -> list[Any]:
    subset = plan[plan["probe_region"] == probe_region].sort_values("probe_index")
    return [
        p27.RequestSpec(
            request_id=(
                f"attacker::{schedule.base_schedule_id}::repeat_{repeat_id:02d}::"
                f"{probe_region}::{int(row.probe_index)}"
            ),
            tenant="attacker",
            ready_ns=float(row.release_ns),
            request_index=int(row.probe_index),
            workload_name="opaque",
            trial_id=repeat_id,
        )
        for row in subset.itertuples(index=False)
    ]


# =============================================================================
# Protocol / spatial contexts
# =============================================================================


def select_protocols(p27, protocol_choice: str):
    protocols = p27.build_protocols()
    out: list[tuple[str, Any]] = []
    if protocol_choice in {"direct", "both"}:
        out.append(("direct_coherent", protocols[p27.DIRECT_PROTOCOL]))
    if protocol_choice in {"entangled", "both"}:
        out.append(("entanglement_assisted", protocols[p27.ENTANGLED_PROTOCOL]))
    return out


def parse_spatial_modes(raw: str) -> tuple[str, ...]:
    requested = tuple(x.strip() for x in raw.split(",") if x.strip())
    unknown = sorted(set(requested) - set(SPATIAL_MODES))
    if unknown:
        raise ValueError(f"Unknown spatial mode(s): {unknown}; choices={SPATIAL_MODES}")
    if not requested:
        raise ValueError("At least one spatial mode must be selected")
    return requested


def build_spatial_contexts(p27, protocol_choice: str, spatial_modes: tuple[str, ...]):
    protocols = p27.build_protocols()
    scenarios = {s.scenario_id: s for s in p27.build_scenarios(protocols)}

    rows: list[tuple[SpatialContext, Any, Any, Any]] = []

    for short_name, protocol in select_protocols(p27, protocol_choice):
        pid = protocol.protocol_name
        all_shared_sid = f"{pid}__all_used_shared"
        isolated_sid = f"{pid}__isolated"

        if pid == p27.DIRECT_PROTOCOL:
            common_sid = f"{pid}__share_interconnect_stack"
            common_desc = "global switch-path + synchronous quantum-link background"
        elif pid == p27.ENTANGLED_PROTOCOL:
            common_sid = f"{pid}__share_measurement_stack"
            common_desc = "global readout + feedforward backend background"
        else:
            raise ValueError(pid)

        for mode in spatial_modes:
            if mode == "localized_only":
                match_sid = all_shared_sid
                nonmatch_sid = isolated_sid
            elif mode == "hybrid_local_global":
                match_sid = all_shared_sid
                nonmatch_sid = common_sid
            elif mode == "global_only_control":
                match_sid = common_sid
                nonmatch_sid = common_sid
            else:
                raise ValueError(mode)

            ctx = SpatialContext(
                protocol_context=f"{short_name}_all_shared_local_route",
                protocol_name=pid,
                spatial_mode=mode,
                matching_scenario_id=match_sid,
                nonmatching_scenario_id=nonmatch_sid,
                common_stack_description=common_desc,
            )
            rows.append(
                (
                    ctx,
                    protocol,
                    scenarios[match_sid],
                    scenarios[nonmatch_sid],
                )
            )
    return rows


# =============================================================================
# Pair attacker observations
# =============================================================================


def pair_attacker_region_trace(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    trace_id: str,
    protocol_context: str,
    probe_region: str,
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
            "protocol_context": protocol_context,
            "probe_index": merged["request_index"].astype(int),
            "probe_region": probe_region,
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
            "combined_turnaround_ns": merged["turnaround_ns_combined"].astype(float),
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
# Feature extraction
# =============================================================================


def run_lengths(mask: np.ndarray) -> list[int]:
    out: list[int] = []
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
        elif current:
            out.append(current)
            current = 0
    if current:
        out.append(current)
    return out


def autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    if len(values) <= lag:
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def summarize_region(sub: pd.DataFrame) -> dict[str, float]:
    sub = sub.sort_values("probe_index")
    x = sub["excess_turnaround_ns"].to_numpy(dtype=float)
    delayed = x > AFFECTED_THRESHOLD_NS
    speedup = x < -AFFECTED_THRESHOLD_NS
    failure = sub["failure_transition"].to_numpy(dtype=bool)
    absx = np.abs(x)
    pos = np.maximum(x, 0.0)
    runs = run_lengths(delayed)

    if len(x):
        half = max(1, len(x) // 2)
        early = absx[:half]
        late = absx[-half:]
    else:
        early = np.asarray([], dtype=float)
        late = np.asarray([], dtype=float)

    q = lambda p: float(np.quantile(x, p)) if len(x) else 0.0

    return {
        "probe_count": float(len(x)),
        "mean_excess_ns": float(np.mean(x)) if len(x) else 0.0,
        "median_excess_ns": float(np.median(x)) if len(x) else 0.0,
        "mean_abs_excess_ns": float(np.mean(absx)) if len(x) else 0.0,
        "std_excess_ns": float(np.std(x)) if len(x) else 0.0,
        "max_excess_ns": float(np.max(x)) if len(x) else 0.0,
        "p90_excess_ns": q(0.90),
        "p95_excess_ns": q(0.95),
        "delayed_fraction": float(np.mean(delayed)) if len(x) else 0.0,
        "speedup_fraction": float(np.mean(speedup)) if len(x) else 0.0,
        "failure_fraction": float(np.mean(failure)) if len(x) else 0.0,
        "cumulative_positive_excess_ns": float(np.sum(pos)),
        "cumulative_abs_excess_ns": float(np.sum(absx)),
        "longest_delayed_run": float(max(runs) if runs else 0),
        "delayed_run_count": float(len(runs)),
        "lag1_autocorrelation": autocorrelation(x, 1),
        "early_mean_abs_ns": float(np.mean(early)) if len(early) else 0.0,
        "late_mean_abs_ns": float(np.mean(late)) if len(late) else 0.0,
    }


def trace_feature_row(
    trace: pd.DataFrame,
    *,
    regions: tuple[str, ...],
    probe_budget: int | None = None,
) -> dict[str, Any]:
    t = trace.sort_values("probe_index")
    if probe_budget is not None:
        t = t.head(int(probe_budget))

    row: dict[str, Any] = {
        "trace_id": str(t["trace_id"].iloc[0]) if len(t) else "",
        "protocol_context": str(t["protocol_context"].iloc[0]) if len(t) else "",
    }

    cumulative_scores: list[float] = []
    abs_scores: list[float] = []
    delay_scores: list[float] = []

    for region in regions:
        stats = summarize_region(t[t["probe_region"] == region])
        for suffix, value in stats.items():
            row[f"region_{region}__{suffix}"] = value
        cumulative_scores.append(stats["cumulative_positive_excess_ns"])
        abs_scores.append(stats["cumulative_abs_excess_ns"])
        delay_scores.append(stats["delayed_fraction"])

    def top_gap(values: list[float]) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        order = np.sort(arr)[::-1]
        top1 = float(order[0])
        top2 = float(order[1]) if len(order) > 1 else 0.0
        gap = top1 - top2
        denom = float(np.sum(np.maximum(arr, 0.0)))
        share = top1 / denom if denom > EPS else 0.0
        return top1, gap, share

    c_top, c_gap, c_share = top_gap(cumulative_scores)
    a_top, a_gap, a_share = top_gap(abs_scores)
    d_top, d_gap, d_share = top_gap(delay_scores)
    row.update(
        {
            "global_probe_count": float(len(t)),
            "spatial_cumulative_top1": c_top,
            "spatial_cumulative_top1_minus_top2": c_gap,
            "spatial_cumulative_top1_share": c_share,
            "spatial_abs_top1": a_top,
            "spatial_abs_top1_minus_top2": a_gap,
            "spatial_abs_top1_share": a_share,
            "spatial_delay_top1": d_top,
            "spatial_delay_top1_minus_top2": d_gap,
            "spatial_delay_top1_share": d_share,
        }
    )
    return row


def model_feature_columns(regions: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for region in regions:
        for suffix in PER_REGION_FEATURE_SUFFIXES:
            cols.append(f"region_{region}__{suffix}")
    cols.extend(
        [
            "global_probe_count",
            "spatial_cumulative_top1",
            "spatial_cumulative_top1_minus_top2",
            "spatial_cumulative_top1_share",
            "spatial_abs_top1",
            "spatial_abs_top1_minus_top2",
            "spatial_abs_top1_share",
            "spatial_delay_top1",
            "spatial_delay_top1_minus_top2",
            "spatial_delay_top1_share",
        ]
    )
    return cols


# =============================================================================
# Dataset generation
# =============================================================================


def scenario_for_probe(
    *,
    spatial_mode: str,
    probe_region: str,
    hidden_region: str,
    matching_scenario: Any,
    nonmatching_scenario: Any,
):
    if spatial_mode == "global_only_control":
        return nonmatching_scenario
    return matching_scenario if probe_region == hidden_region else nonmatching_scenario


def run_dataset(
    p27,
    *,
    contexts,
    schedules: list[BaseSchedule],
    regions: tuple[str, ...],
    repeats_per_instance: int,
    seed: int,
    observation_window_ns: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blackbox_parts: list[pd.DataFrame] = []
    ground_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    release_rows: list[dict[str, Any]] = []

    total = len(contexts) * len(schedules) * len(regions) * repeats_per_instance
    done = 0

    # Attacker-only execution can be reused for all hidden-region copies that
    # select the same scenario for a given base/repeat/probe region.
    attacker_cache: dict[tuple[str, str, str, int, str], pd.DataFrame] = {}

    for ctx, protocol, matching_scenario, nonmatching_scenario in contexts:
        for schedule in schedules:
            for repeat_id in range(repeats_per_instance):
                plan = attacker_probe_plan(
                    p27,
                    regions=regions,
                    schedule=schedule,
                    repeat_id=repeat_id,
                    seed=seed,
                    observation_window_ns=observation_window_ns,
                )

                # Save one evaluator-side copy of the probe plan per context,
                # base schedule, repeat.  probe_region itself remains attacker-known.
                for row in plan.itertuples(index=False):
                    plan_rows.append(
                        {
                            "protocol_context": ctx.protocol_context,
                            "spatial_mode": ctx.spatial_mode,
                            "base_schedule_id": schedule.base_schedule_id,
                            "repeat_id": repeat_id,
                            "probe_index": int(row.probe_index),
                            "release_ns": float(row.release_ns),
                            "probe_region": str(row.probe_region),
                        }
                    )

                victim_releases = generate_victim_releases(
                    schedule,
                    repeat_id=repeat_id,
                    seed=seed,
                    observation_window_ns=observation_window_ns,
                )
                victim_specs = make_victim_specs(
                    p27,
                    releases=victim_releases,
                    schedule=schedule,
                    repeat_id=repeat_id,
                )

                # One schedule copy; hidden region does not alter releases.
                for i, t in enumerate(victim_releases):
                    release_rows.append(
                        {
                            "base_schedule_id": schedule.base_schedule_id,
                            "repeat_id": repeat_id,
                            "schedule_profile": schedule.schedule_profile,
                            "victim_request_index": i,
                            "release_ns": float(t),
                        }
                    )

                attacker_specs_by_region = {
                    region: make_attacker_specs_for_region(
                        p27,
                        plan=plan,
                        probe_region=region,
                        schedule=schedule,
                        repeat_id=repeat_id,
                    )
                    for region in regions
                }

                for hidden_region in regions:
                    trace_id = opaque_trace_id(
                        "phase3_03",
                        seed,
                        ctx.protocol_context,
                        ctx.spatial_mode,
                        schedule.base_schedule_id,
                        repeat_id,
                        hidden_region,
                    )
                    trace_parts: list[pd.DataFrame] = []

                    for probe_region in regions:
                        scenario = scenario_for_probe(
                            spatial_mode=ctx.spatial_mode,
                            probe_region=probe_region,
                            hidden_region=hidden_region,
                            matching_scenario=matching_scenario,
                            nonmatching_scenario=nonmatching_scenario,
                        )
                        attacker_specs = attacker_specs_by_region[probe_region]

                        cache_key = (
                            ctx.protocol_context,
                            scenario.scenario_id,
                            schedule.base_schedule_id,
                            repeat_id,
                            probe_region,
                        )
                        if cache_key not in attacker_cache:
                            attacker_only, *_ = p27.run_one(
                                protocol,
                                scenario,
                                "opaque",
                                repeat_id,
                                "attacker_only",
                                list(attacker_specs),
                            )
                            attacker_cache[cache_key] = attacker_only
                        attacker_only = attacker_cache[cache_key]

                        combined_specs = sorted(
                            list(attacker_specs) + list(victim_specs),
                            key=lambda x: (x.ready_ns, x.tenant, x.request_index),
                        )
                        combined, *_ = p27.run_one(
                            protocol,
                            scenario,
                            schedule.schedule_profile,
                            repeat_id,
                            "combined",
                            combined_specs,
                        )

                        paired = pair_attacker_region_trace(
                            attacker_only,
                            combined,
                            trace_id=trace_id,
                            protocol_context=ctx.protocol_context,
                            probe_region=probe_region,
                        )
                        trace_parts.append(paired)

                    trace = pd.concat(trace_parts, ignore_index=True).sort_values(
                        "probe_index"
                    )
                    blackbox_parts.append(trace)

                    ground_rows.append(
                        {
                            "trace_id": trace_id,
                            "protocol_context": ctx.protocol_context,
                            "protocol_name": ctx.protocol_name,
                            "spatial_mode": ctx.spatial_mode,
                            "base_schedule_id": schedule.base_schedule_id,
                            "schedule_profile": schedule.schedule_profile,
                            "repeat_id": repeat_id,
                            "hidden_victim_region": hidden_region,
                            "remote_operation_count": int(len(victim_specs)),
                            "matching_scenario_id": ctx.matching_scenario_id,
                            "nonmatching_scenario_id": ctx.nonmatching_scenario_id,
                        }
                    )
                    feature_rows.append(trace_feature_row(trace, regions=regions))

                    done += 1
                    if done % max(1, total // 20) == 0 or done == total:
                        print(f"[Phase 3.3] Generated {done}/{total} traces")

    blackbox = (
        pd.concat(blackbox_parts, ignore_index=True)
        if blackbox_parts
        else pd.DataFrame(columns=ATTACKER_VISIBLE_COLUMNS)
    )
    ground = pd.DataFrame(ground_rows)
    features = pd.DataFrame(feature_rows)
    plans = pd.DataFrame(plan_rows).drop_duplicates()
    releases = pd.DataFrame(release_rows).drop_duplicates()
    return blackbox, ground, features, plans, releases


# =============================================================================
# Grouped split
# =============================================================================


def build_group_split(
    schedules: list[BaseSchedule],
    *,
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    """
    Deterministic grouped split stratified by schedule profile.

    We select test groups independently within each profile so even small smoke
    tests retain at least one train and one test base schedule per profile.
    """
    table = pd.DataFrame(
        [
            {
                "base_schedule_id": s.base_schedule_id,
                "schedule_profile": s.schedule_profile,
            }
            for s in schedules
        ]
    )

    test_set: set[str] = set()
    train_set: set[str] = set()
    for profile, g in table.groupby("schedule_profile", sort=True):
        ids = g["base_schedule_id"].astype(str).tolist()
        if len(ids) < 2:
            raise ValueError(
                f"Schedule profile {profile!r} has only {len(ids)} base instance(s); "
                "need at least 2 for grouped train/test evaluation."
            )
        rng = np.random.default_rng(stable_seed(seed, "group_split", profile))
        ids = list(np.asarray(ids)[rng.permutation(len(ids))])
        n_test = int(round(len(ids) * test_size))
        n_test = min(len(ids) - 1, max(1, n_test))
        test_set.update(ids[:n_test])
        train_set.update(ids[n_test:])

    out = table.copy()
    out["split"] = out["base_schedule_id"].map(
        lambda x: "test" if x in test_set else "train"
    )
    assert not (train_set & test_set)
    assert train_set | test_set == set(table["base_schedule_id"].astype(str))
    return out.sort_values(["schedule_profile", "base_schedule_id"]).reset_index(drop=True)


# =============================================================================
# ML models and evaluation
# =============================================================================


def classification_models(seed: int, rf_trees: int):
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
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            l2_regularization=0.5,
            random_state=seed,
        ),
    }


def safe_top2(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) < 2 or len(y_true) == 0:
        return math.nan
    return float(
        top_k_accuracy_score(
            y_true,
            proba,
            k=min(2, len(classes)),
            labels=classes,
        )
    )


def argmax_rule_prediction(
    row: pd.Series,
    *,
    regions: tuple[str, ...],
    suffix: str,
) -> str:
    scores = np.asarray(
        [float(row[f"region_{r}__{suffix}"]) for r in regions], dtype=float
    )
    # Deterministic first-region tie break.  In global-only control, balanced
    # labels ensure this returns exactly chance in expectation.
    return regions[int(np.argmax(scores))]


def evaluate_localization(
    analysis: pd.DataFrame,
    split: pd.DataFrame,
    *,
    regions: tuple[str, ...],
    feature_columns: list[str],
    seed: int,
    rf_trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_map = dict(zip(split.base_schedule_id, split.split))
    data = analysis.copy()
    data["split"] = data["base_schedule_id"].map(split_map)

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    group_cols = ["protocol_context", "spatial_mode"]
    for (protocol_context, spatial_mode), group in data.groupby(group_cols, sort=True):
        train = group[group["split"] == "train"].copy()
        test = group[group["split"] == "test"].copy()
        if train.empty or test.empty:
            continue

        X_train = train[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y_train = train["hidden_victim_region"].astype(str).to_numpy()
        X_test = test[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y_test = test["hidden_victim_region"].astype(str).to_numpy()

        for model_name, model in classification_models(seed, rf_trees).items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test).astype(str)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                classes = np.asarray(model.classes_, dtype=str)
                top2 = safe_top2(y_test, proba, classes)
                max_prob = np.max(proba, axis=1)
                order = np.sort(proba, axis=1)[:, ::-1]
                margins = order[:, 0] - order[:, 1] if order.shape[1] > 1 else order[:, 0]
            else:
                classes = np.asarray(regions)
                proba = np.zeros((len(test), len(classes)), dtype=float)
                top2 = math.nan
                max_prob = np.full(len(test), math.nan)
                margins = np.full(len(test), math.nan)

            acc = float(accuracy_score(y_test, pred))
            bacc = float(balanced_accuracy_score(y_test, pred))
            mf1 = float(f1_score(y_test, pred, average="macro", labels=list(regions)))
            metric_rows.append(
                {
                    "protocol_context": protocol_context,
                    "spatial_mode": spatial_mode,
                    "model_name": model_name,
                    "sample_count": len(test),
                    "location_class_count": len(regions),
                    "chance_accuracy": 1.0 / len(regions),
                    "accuracy": acc,
                    "balanced_accuracy": bacc,
                    "macro_f1": mf1,
                    "top2_accuracy": top2,
                }
            )

            for i, row in enumerate(test.itertuples(index=False)):
                out = {
                    "trace_id": row.trace_id,
                    "protocol_context": protocol_context,
                    "spatial_mode": spatial_mode,
                    "base_schedule_id": row.base_schedule_id,
                    "schedule_profile": row.schedule_profile,
                    "repeat_id": int(row.repeat_id),
                    "true_hidden_region": y_test[i],
                    "predicted_region": pred[i],
                    "correct": bool(pred[i] == y_test[i]),
                    "model_name": model_name,
                    "prediction_confidence": float(max_prob[i]),
                    "prediction_margin": float(margins[i]),
                }
                for j, cls in enumerate(classes):
                    out[f"prob_{cls}"] = float(proba[i, j])
                pred_rows.append(out)

            cm = confusion_matrix(y_test, pred, labels=list(regions))
            for i, true_region in enumerate(regions):
                denom = int(np.sum(cm[i, :]))
                for j, predicted_region in enumerate(regions):
                    confusion_rows.append(
                        {
                            "protocol_context": protocol_context,
                            "spatial_mode": spatial_mode,
                            "model_name": model_name,
                            "true_region": true_region,
                            "predicted_region": predicted_region,
                            "count": int(cm[i, j]),
                            "true_normalized_fraction": (
                                float(cm[i, j] / denom) if denom else 0.0
                            ),
                        }
                    )

            for profile, pg in test.assign(_pred=pred).groupby(
                "schedule_profile", sort=True
            ):
                profile_rows.append(
                    {
                        "protocol_context": protocol_context,
                        "spatial_mode": spatial_mode,
                        "model_name": model_name,
                        "schedule_profile": profile,
                        "sample_count": len(pg),
                        "accuracy": float(
                            accuracy_score(pg["hidden_victim_region"], pg["_pred"])
                        ),
                        "macro_f1": float(
                            f1_score(
                                pg["hidden_victim_region"],
                                pg["_pred"],
                                average="macro",
                                labels=list(regions),
                                zero_division=0,
                            )
                        ),
                    }
                )

        # Interpretable, training-free spatial-score baselines.
        for rule_name, suffix in [
            ("argmax_cumulative_positive", "cumulative_positive_excess_ns"),
            ("argmax_cumulative_abs", "cumulative_abs_excess_ns"),
            ("argmax_delayed_fraction", "delayed_fraction"),
        ]:
            pred = test.apply(
                lambda r: argmax_rule_prediction(r, regions=regions, suffix=suffix),
                axis=1,
            ).astype(str).to_numpy()
            metric_rows.append(
                {
                    "protocol_context": protocol_context,
                    "spatial_mode": spatial_mode,
                    "model_name": rule_name,
                    "sample_count": len(test),
                    "location_class_count": len(regions),
                    "chance_accuracy": 1.0 / len(regions),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "macro_f1": float(
                        f1_score(y_test, pred, average="macro", labels=list(regions))
                    ),
                    "top2_accuracy": math.nan,
                }
            )
            for i, row in enumerate(test.itertuples(index=False)):
                pred_rows.append(
                    {
                        "trace_id": row.trace_id,
                        "protocol_context": protocol_context,
                        "spatial_mode": spatial_mode,
                        "base_schedule_id": row.base_schedule_id,
                        "schedule_profile": row.schedule_profile,
                        "repeat_id": int(row.repeat_id),
                        "true_hidden_region": y_test[i],
                        "predicted_region": pred[i],
                        "correct": bool(pred[i] == y_test[i]),
                        "model_name": rule_name,
                        "prediction_confidence": math.nan,
                        "prediction_margin": math.nan,
                    }
                )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(pred_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(profile_rows),
    )


# =============================================================================
# Probe-budget analysis
# =============================================================================


def evaluate_probe_budgets(
    blackbox: pd.DataFrame,
    ground: pd.DataFrame,
    split: pd.DataFrame,
    *,
    regions: tuple[str, ...],
    budgets: list[int],
    seed: int,
    rf_trees: int,
) -> pd.DataFrame:
    split_map = dict(zip(split.base_schedule_id, split.split))
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        feat_rows = [
            trace_feature_row(g, regions=regions, probe_budget=budget)
            for _, g in blackbox.groupby("trace_id", sort=False)
        ]
        feats = pd.DataFrame(feat_rows)
        data = feats.merge(ground, on=["trace_id", "protocol_context"], validate="one_to_one")
        data["split"] = data["base_schedule_id"].map(split_map)
        fcols = model_feature_columns(regions)

        for (protocol_context, spatial_mode), group in data.groupby(
            ["protocol_context", "spatial_mode"], sort=True
        ):
            train = group[group.split == "train"]
            test = group[group.split == "test"]
            if train.empty or test.empty:
                continue
            X_train = train[fcols].fillna(0.0).to_numpy(dtype=float)
            y_train = train.hidden_victim_region.astype(str).to_numpy()
            X_test = test[fcols].fillna(0.0).to_numpy(dtype=float)
            y_test = test.hidden_victim_region.astype(str).to_numpy()

            rf = RandomForestClassifier(
                n_estimators=rf_trees,
                class_weight="balanced_subsample",
                random_state=seed + budget,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test).astype(str)
            proba = rf.predict_proba(X_test)
            rows.append(
                {
                    "protocol_context": protocol_context,
                    "spatial_mode": spatial_mode,
                    "model_name": "random_forest",
                    "probe_budget": budget,
                    "probes_per_region_nominal": budget / len(regions),
                    "sample_count": len(test),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "macro_f1": float(
                        f1_score(y_test, pred, average="macro", labels=list(regions))
                    ),
                    "top2_accuracy": safe_top2(
                        y_test, proba, np.asarray(rf.classes_, dtype=str)
                    ),
                }
            )

            # Training-free baseline at same budget.
            rule_pred = test.apply(
                lambda r: argmax_rule_prediction(
                    r,
                    regions=regions,
                    suffix="cumulative_positive_excess_ns",
                ),
                axis=1,
            ).astype(str).to_numpy()
            rows.append(
                {
                    "protocol_context": protocol_context,
                    "spatial_mode": spatial_mode,
                    "model_name": "argmax_cumulative_positive",
                    "probe_budget": budget,
                    "probes_per_region_nominal": budget / len(regions),
                    "sample_count": len(test),
                    "accuracy": float(accuracy_score(y_test, rule_pred)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, rule_pred)
                    ),
                    "macro_f1": float(
                        f1_score(
                            y_test,
                            rule_pred,
                            average="macro",
                            labels=list(regions),
                        )
                    ),
                    "top2_accuracy": math.nan,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Spatial signal / causal summaries
# =============================================================================


def build_spatial_signal_matrix(
    blackbox: pd.DataFrame,
    ground: pd.DataFrame,
    split: pd.DataFrame,
) -> pd.DataFrame:
    meta = ground[
        [
            "trace_id",
            "protocol_context",
            "spatial_mode",
            "base_schedule_id",
            "schedule_profile",
            "hidden_victim_region",
        ]
    ].copy()
    m = blackbox.merge(meta, on=["trace_id", "protocol_context"], validate="many_to_one")
    split_map = dict(zip(split.base_schedule_id, split.split))
    m["split"] = m["base_schedule_id"].map(split_map)
    m["abs_excess_ns"] = np.abs(m["excess_turnaround_ns"].astype(float))
    m["positive_excess_ns"] = np.maximum(m["excess_turnaround_ns"].astype(float), 0.0)
    return (
        m.groupby(
            [
                "protocol_context",
                "spatial_mode",
                "split",
                "schedule_profile",
                "hidden_victim_region",
                "probe_region",
            ],
            sort=True,
        )
        .agg(
            probe_count=("probe_index", "count"),
            mean_excess_ns=("excess_turnaround_ns", "mean"),
            mean_abs_excess_ns=("abs_excess_ns", "mean"),
            delayed_fraction=("delayed", "mean"),
            mean_positive_excess_ns=("positive_excess_ns", "mean"),
            failure_transition_fraction=("failure_transition", "mean"),
        )
        .reset_index()
    )


def build_diagonal_contrast(signal_matrix: pd.DataFrame) -> pd.DataFrame:
    test = signal_matrix[signal_matrix["split"] == "test"].copy()
    test["is_matching_region"] = test["hidden_victim_region"] == test["probe_region"]
    rows: list[dict[str, Any]] = []
    for (protocol_context, spatial_mode, schedule_profile), g in test.groupby(
        ["protocol_context", "spatial_mode", "schedule_profile"], sort=True
    ):
        diag = g[g.is_matching_region]
        off = g[~g.is_matching_region]
        d = float(diag.mean_abs_excess_ns.mean()) if len(diag) else 0.0
        o = float(off.mean_abs_excess_ns.mean()) if len(off) else 0.0
        dd = float(diag.delayed_fraction.mean()) if len(diag) else 0.0
        od = float(off.delayed_fraction.mean()) if len(off) else 0.0
        rows.append(
            {
                "protocol_context": protocol_context,
                "spatial_mode": spatial_mode,
                "schedule_profile": schedule_profile,
                "matching_mean_abs_excess_ns": d,
                "nonmatching_mean_abs_excess_ns": o,
                "mean_abs_diagonal_minus_offdiagonal_ns": d - o,
                "mean_abs_diagonal_ratio": d / o if o > EPS else math.inf if d > 0 else 1.0,
                "matching_delayed_fraction": dd,
                "nonmatching_delayed_fraction": od,
                "delayed_fraction_diagonal_minus_offdiagonal": dd - od,
            }
        )

    # Aggregate across schedule profiles as an explicit ALL row.
    for (protocol_context, spatial_mode), g in test.groupby(
        ["protocol_context", "spatial_mode"], sort=True
    ):
        diag = g[g.is_matching_region]
        off = g[~g.is_matching_region]
        d = float(diag.mean_abs_excess_ns.mean()) if len(diag) else 0.0
        o = float(off.mean_abs_excess_ns.mean()) if len(off) else 0.0
        dd = float(diag.delayed_fraction.mean()) if len(diag) else 0.0
        od = float(off.delayed_fraction.mean()) if len(off) else 0.0
        rows.append(
            {
                "protocol_context": protocol_context,
                "spatial_mode": spatial_mode,
                "schedule_profile": "ALL",
                "matching_mean_abs_excess_ns": d,
                "nonmatching_mean_abs_excess_ns": o,
                "mean_abs_diagonal_minus_offdiagonal_ns": d - o,
                "mean_abs_diagonal_ratio": d / o if o > EPS else math.inf if d > 0 else 1.0,
                "matching_delayed_fraction": dd,
                "nonmatching_delayed_fraction": od,
                "delayed_fraction_diagonal_minus_offdiagonal": dd - od,
            }
        )
    return pd.DataFrame(rows)


def build_protocol_comparison_summary(
    metrics: pd.DataFrame,
    contrast: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    c_all = contrast[contrast.schedule_profile == "ALL"].copy()
    for (protocol_context, spatial_mode), g in metrics.groupby(
        ["protocol_context", "spatial_mode"], sort=True
    ):
        # Primary learned model summary: choose best among the three ML models.
        learned = g[g.model_name.isin(
            ["logistic_regression", "random_forest", "hist_gradient_boosting"]
        )].copy()
        best = learned.sort_values(
            ["accuracy", "macro_f1"], ascending=[False, False]
        ).iloc[0]
        rule = g[g.model_name == "argmax_cumulative_positive"]
        cr = c_all[
            (c_all.protocol_context == protocol_context)
            & (c_all.spatial_mode == spatial_mode)
        ]
        rows.append(
            {
                "protocol_context": protocol_context,
                "spatial_mode": spatial_mode,
                "best_model": best.model_name,
                "chance_accuracy": float(best.chance_accuracy),
                "best_accuracy": float(best.accuracy),
                "best_macro_f1": float(best.macro_f1),
                "best_top2_accuracy": float(best.top2_accuracy),
                "argmax_cumulative_positive_accuracy": (
                    float(rule.iloc[0].accuracy) if len(rule) else math.nan
                ),
                "matching_mean_abs_excess_ns": (
                    float(cr.iloc[0].matching_mean_abs_excess_ns) if len(cr) else math.nan
                ),
                "nonmatching_mean_abs_excess_ns": (
                    float(cr.iloc[0].nonmatching_mean_abs_excess_ns) if len(cr) else math.nan
                ),
                "mean_abs_diagonal_minus_offdiagonal_ns": (
                    float(cr.iloc[0].mean_abs_diagonal_minus_offdiagonal_ns)
                    if len(cr)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Validation
# =============================================================================


def add_assertion(rows: list[ValidationAssertion], group: str, name: str, passed: bool, expected: Any, observed: Any, details: str = ""):
    rows.append(
        ValidationAssertion(
            validation_group=group,
            assertion_name=name,
            passed=bool(passed),
            expected=str(expected),
            observed=str(observed),
            details=details,
        )
    )


def build_validations(
    *,
    p27,
    blackbox: pd.DataFrame,
    ground: pd.DataFrame,
    features: pd.DataFrame,
    plans: pd.DataFrame,
    releases: pd.DataFrame,
    split: pd.DataFrame,
    schedules: list[BaseSchedule],
    regions: tuple[str, ...],
    contexts,
) -> pd.DataFrame:
    rows: list[ValidationAssertion] = []

    # 1 strict black-box schema
    observed_cols = list(blackbox.columns)
    add_assertion(
        rows,
        "blackbox_boundary",
        "attacker_trace_schema_exact",
        observed_cols == list(ATTACKER_VISIBLE_COLUMNS),
        list(ATTACKER_VISIBLE_COLUMNS),
        observed_cols,
    )

    # 2 hidden/evaluator labels absent from attacker trace
    forbidden = {
        "hidden_victim_region",
        "base_schedule_id",
        "schedule_profile",
        "spatial_mode",
        "matching_scenario_id",
        "nonmatching_scenario_id",
        "resource_name",
        "resource_key",
        "epr_pool_level",
        "blocking_tenant",
    }
    leaked = sorted(forbidden & set(blackbox.columns))
    add_assertion(
        rows,
        "blackbox_boundary",
        "hidden_location_and_evaluator_state_excluded",
        len(leaked) == 0,
        [],
        leaked,
    )

    # 3 trace ids are opaque rather than plaintext labels
    bad_trace_ids = [
        x for x in ground.trace_id.astype(str)
        if any(r in x for r in regions) or "localized" in x or "global" in x
    ]
    add_assertion(
        rows,
        "blackbox_boundary",
        "trace_ids_do_not_encode_location_or_mode",
        len(bad_trace_ids) == 0,
        0,
        len(bad_trace_ids),
    )

    # 4 each base/repeat/context/mode crosses all hidden locations
    cross = ground.groupby(
        ["protocol_context", "spatial_mode", "base_schedule_id", "repeat_id"]
    )["hidden_victim_region"].nunique()
    add_assertion(
        rows,
        "schedule",
        "every_schedule_crossed_with_all_regions",
        bool((cross == len(regions)).all()),
        len(regions),
        f"min={cross.min()}, max={cross.max()}",
    )

    # 5 balanced attacker target counts in each plan
    pc = plans.groupby(
        ["protocol_context", "spatial_mode", "base_schedule_id", "repeat_id", "probe_region"]
    ).size().reset_index(name="count")
    spread = pc.groupby(
        ["protocol_context", "spatial_mode", "base_schedule_id", "repeat_id"]
    )["count"].agg(lambda x: int(x.max() - x.min()))
    add_assertion(
        rows,
        "attacker_probe_policy",
        "balanced_probe_regions",
        bool((spread <= 1).all()),
        "max count difference <= 1",
        int(spread.max()) if len(spread) else 0,
    )

    # 6 victim releases independent of hidden region by construction: only one
    # evaluator schedule copy exists per base/repeat.
    release_key_dupes = releases.duplicated(
        ["base_schedule_id", "repeat_id", "victim_request_index"], keep=False
    )
    # duplicates would only arise if schedule was accidentally copied by location
    add_assertion(
        rows,
        "schedule",
        "victim_release_schedule_not_location_dependent",
        not bool(release_key_dupes.any()),
        False,
        bool(release_key_dupes.any()),
    )

    # 7 group split no overlap
    train_ids = set(split.loc[split.split == "train", "base_schedule_id"])
    test_ids = set(split.loc[split.split == "test", "base_schedule_id"])
    overlap = sorted(train_ids & test_ids)
    add_assertion(
        rows,
        "evaluation",
        "group_split_has_no_base_schedule_overlap",
        len(overlap) == 0,
        [],
        overlap,
    )

    # 8 all profiles in train/test
    train_profiles = set(split.loc[split.split == "train", "schedule_profile"])
    test_profiles = set(split.loc[split.split == "test", "schedule_profile"])
    add_assertion(
        rows,
        "evaluation",
        "all_schedule_profiles_present_in_train_and_test",
        train_profiles == set(SCHEDULE_PROFILES) and test_profiles == set(SCHEDULE_PROFILES),
        sorted(SCHEDULE_PROFILES),
        f"train={sorted(train_profiles)}, test={sorted(test_profiles)}",
    )

    # 9 all hidden locations in train/test after merge
    gsplit = ground.merge(split[["base_schedule_id", "split"]], on="base_schedule_id")
    train_regions = set(gsplit.loc[gsplit.split == "train", "hidden_victim_region"])
    test_regions = set(gsplit.loc[gsplit.split == "test", "hidden_victim_region"])
    add_assertion(
        rows,
        "evaluation",
        "all_hidden_regions_present_in_train_and_test",
        train_regions == set(regions) and test_regions == set(regions),
        sorted(regions),
        f"train={sorted(train_regions)}, test={sorted(test_regions)}",
    )

    # 10 feature/ground rows one to one
    add_assertion(
        rows,
        "evaluation",
        "feature_and_ground_truth_rows_pair_one_to_one",
        len(features) == len(ground)
        and features.trace_id.nunique() == len(features)
        and ground.trace_id.nunique() == len(ground),
        f"equal unique rows ({len(ground)})",
        f"features={len(features)}, ground_truth={len(ground)}",
    )

    # 11 Phase2.7 protocol normalization retained
    protocols = p27.build_protocols()
    critical = sorted(
        {float(x.nominal_critical_latency_ns) for x in protocols.values()}
    )
    cleanup = sorted({float(x.postcompletion_cleanup_ns) for x in protocols.values()})
    add_assertion(
        rows,
        "architecture",
        "phase2_07_protocol_normalization_retained",
        critical == [150.0] and cleanup == [120.0],
        "critical=[150.0], cleanup=[120.0]",
        f"critical={critical}, cleanup={cleanup}",
    )

    # 12 global-only control invariant across hidden location for identical
    # protocol/base/repeat.  Compare feature vectors excluding trace metadata.
    global_ids = set(
        ground.loc[ground.spatial_mode == "global_only_control", "trace_id"]
    )
    if global_ids:
        fm = features.merge(
            ground[
                [
                    "trace_id",
                    "protocol_context",
                    "spatial_mode",
                    "base_schedule_id",
                    "repeat_id",
                    "hidden_victim_region",
                ]
            ],
            on=["trace_id", "protocol_context"],
            validate="one_to_one",
        )
        gm = fm[fm.spatial_mode == "global_only_control"].copy()
        fcols = [c for c in features.columns if c not in {"trace_id", "protocol_context"}]
        max_span = 0.0
        for _, gg in gm.groupby(["protocol_context", "base_schedule_id", "repeat_id"]):
            vals = gg[fcols].fillna(0.0).to_numpy(dtype=float)
            if len(vals):
                span = float(np.max(np.ptp(vals, axis=0)))
                max_span = max(max_span, span)
        add_assertion(
            rows,
            "causal_spatial_control",
            "global_only_features_are_location_invariant",
            max_span <= 1e-9,
            "max feature span <= 1e-9",
            max_span,
        )
    else:
        add_assertion(
            rows,
            "causal_spatial_control",
            "global_only_features_are_location_invariant",
            True,
            "not requested",
            "not requested",
        )

    # 13 localized-only mismatching regions must be zero differential timing.
    loc_ground = ground[ground.spatial_mode == "localized_only"]
    if len(loc_ground):
        meta = loc_ground[["trace_id", "hidden_victim_region"]]
        loc = blackbox[blackbox.trace_id.isin(set(loc_ground.trace_id))].merge(
            meta, on="trace_id", validate="many_to_one"
        )
        mismatch = loc[loc.probe_region != loc.hidden_victim_region]
        max_mismatch = (
            float(np.max(np.abs(mismatch.excess_turnaround_ns.to_numpy(dtype=float))))
            if len(mismatch)
            else 0.0
        )
        add_assertion(
            rows,
            "causal_spatial_control",
            "localized_only_nonmatching_regions_zero_differential_timing",
            max_mismatch <= 1e-9,
            "<= 1e-9 ns",
            max_mismatch,
        )
    else:
        add_assertion(
            rows,
            "causal_spatial_control",
            "localized_only_nonmatching_regions_zero_differential_timing",
            True,
            "not requested",
            "not requested",
        )

    # 14 primary hybrid match/nonmatch scenario ids differ.
    hybrid = [c for c, *_ in contexts if c.spatial_mode == "hybrid_local_global"]
    hybrid_ok = all(c.matching_scenario_id != c.nonmatching_scenario_id for c in hybrid)
    add_assertion(
        rows,
        "architecture",
        "hybrid_mode_contains_distinct_local_and_global_resource_paths",
        hybrid_ok,
        True,
        hybrid_ok,
    )

    # 15 all requested protocols present
    observed_protocols = sorted(ground.protocol_context.unique().tolist())
    expected_protocols = sorted({c.protocol_context for c, *_ in contexts})
    add_assertion(
        rows,
        "architecture",
        "requested_protocol_contexts_present",
        observed_protocols == expected_protocols,
        expected_protocols,
        observed_protocols,
    )

    return pd.DataFrame([asdict(x) for x in rows])


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p27, phase2_source = load_phase2_07_module()
    regions = region_names(args.region_count)
    modes = parse_spatial_modes(args.spatial_modes)
    contexts = build_spatial_contexts(p27, args.protocol, modes)
    schedules = make_base_schedules(seed=args.seed, base_instances=args.base_instances)

    print(
        f"[Phase 3.3] protocols={len({c.protocol_context for c, *_ in contexts})}, "
        f"spatial_modes={len(modes)}, base_schedules={len(schedules)}, "
        f"regions={len(regions)}, repeats={args.repeats_per_instance}"
    )
    print(f"[Phase 3.3] Reusing Phase-2.7 simulator: {phase2_source}")

    blackbox, ground, features, plans, releases = run_dataset(
        p27,
        contexts=contexts,
        schedules=schedules,
        regions=regions,
        repeats_per_instance=args.repeats_per_instance,
        seed=args.seed,
        observation_window_ns=args.observation_window_ns,
    )

    split = build_group_split(
        schedules,
        seed=args.seed,
        test_size=args.test_size,
    )

    analysis = features.merge(
        ground,
        on=["trace_id", "protocol_context"],
        validate="one_to_one",
    )
    fcols = model_feature_columns(regions)

    metrics, predictions, confusion, profile_metrics = evaluate_localization(
        analysis,
        split,
        regions=regions,
        feature_columns=fcols,
        seed=args.seed,
        rf_trees=args.rf_trees,
    )

    max_probes = int(blackbox.groupby("trace_id")["probe_index"].count().max())
    budgets = sorted(
        {
            min(max_probes, int(x))
            for x in args.probe_budgets.split(",")
            if x.strip() and int(x) > 0
        }
    )
    budgets = [b for b in budgets if b >= len(regions)]
    # Prefer complete balanced blocks.  User-supplied partial budgets are still
    # legal, but default budgets are multiples of region_count.
    if max_probes not in budgets:
        budgets.append(max_probes)

    budget_metrics = evaluate_probe_budgets(
        blackbox,
        ground,
        split,
        regions=regions,
        budgets=budgets,
        seed=args.seed,
        rf_trees=max(100, min(args.rf_trees, 300)),
    )

    signal_matrix = build_spatial_signal_matrix(blackbox, ground, split)
    contrast = build_diagonal_contrast(signal_matrix)
    protocol_summary = build_protocol_comparison_summary(metrics, contrast)

    validations = build_validations(
        p27=p27,
        blackbox=blackbox,
        ground=ground,
        features=features,
        plans=plans,
        releases=releases,
        split=split,
        schedules=schedules,
        regions=regions,
        contexts=contexts,
    )

    # Context / configuration table
    context_table = pd.DataFrame(
        [
            {
                **asdict(c),
                "matching_semantics": (
                    "probe target equals hidden victim region -> matching scenario"
                ),
                "nonmatching_semantics": (
                    "probe target differs from hidden victim region -> nonmatching scenario"
                ),
            }
            for c, *_ in contexts
        ]
    )
    region_table = pd.DataFrame(
        [
            {
                "attacker_source_module": "module_0",
                "candidate_region_index": i,
                "candidate_probe_region": r,
                "attacker_knows_probe_region": True,
                "victim_region_hidden_from_attacker": True,
            }
            for i, r in enumerate(regions)
        ]
    )
    schedule_table = pd.DataFrame([asdict(s) for s in schedules])

    # Save outputs
    blackbox.to_csv(output_dir / "phase3_03_attacker_visible_trace.csv", index=False)
    ground.to_csv(output_dir / "phase3_03_evaluator_ground_truth.csv", index=False)
    features.to_csv(output_dir / "phase3_03_trace_features.csv", index=False)
    plans.to_csv(output_dir / "phase3_03_attacker_probe_plan_evaluator.csv", index=False)
    releases.to_csv(output_dir / "phase3_03_victim_release_schedule.csv", index=False)
    split.to_csv(output_dir / "phase3_03_group_split.csv", index=False)
    context_table.to_csv(output_dir / "phase3_03_spatial_context_table.csv", index=False)
    region_table.to_csv(output_dir / "phase3_03_probe_target_table.csv", index=False)
    schedule_table.to_csv(output_dir / "phase3_03_base_schedule_table.csv", index=False)
    metrics.to_csv(output_dir / "phase3_03_localization_metrics.csv", index=False)
    predictions.to_csv(output_dir / "phase3_03_localization_predictions.csv", index=False)
    confusion.to_csv(output_dir / "phase3_03_confusion_matrix.csv", index=False)
    profile_metrics.to_csv(output_dir / "phase3_03_profile_metrics.csv", index=False)
    budget_metrics.to_csv(output_dir / "phase3_03_probe_budget_metrics.csv", index=False)
    signal_matrix.to_csv(output_dir / "phase3_03_spatial_signal_matrix.csv", index=False)
    contrast.to_csv(output_dir / "phase3_03_diagonal_contrast_summary.csv", index=False)
    protocol_summary.to_csv(output_dir / "phase3_03_protocol_comparison_summary.csv", index=False)
    validations.to_csv(output_dir / "phase3_03_validation_assertions.csv", index=False)

    validation_summary = pd.DataFrame(
        [
            {
                "assertion_count": len(validations),
                "passed_assertions": int(validations.passed.sum()),
                "failed_assertions": int((~validations.passed.astype(bool)).sum()),
                "all_passed": bool(validations.passed.all()),
            }
        ]
    )
    validation_summary.to_csv(
        output_dir / "phase3_03_validation_summary.csv", index=False
    )

    manifest = {
        "experiment": "phase3_03_endpoint_module_localization",
        "seed": args.seed,
        "phase2_07_source": str(phase2_source),
        "output_dir": str(output_dir),
        "protocol_contexts": sorted(ground.protocol_context.unique().tolist()),
        "spatial_modes": list(modes),
        "candidate_regions": list(regions),
        "attacker_source_module": "module_0",
        "base_schedule_count": len(schedules),
        "schedule_profiles": list(SCHEDULE_PROFILES),
        "repeats_per_instance": args.repeats_per_instance,
        "observation_window_ns": args.observation_window_ns,
        "attacker_first_release_ns": float(p27.ATTACKER_FIRST_RELEASE_NS),
        "attacker_period_ns": float(p27.ATTACKER_PERIOD_NS),
        "test_size": args.test_size,
        "rf_trees": args.rf_trees,
        "probe_budgets": budgets,
        "trace_count": int(ground.trace_id.nunique()),
        "probe_row_count": int(len(blackbox)),
        "training_base_schedule_count": int((split.split == "train").sum()),
        "test_base_schedule_count": int((split.split == "test").sum()),
        "model_feature_columns": fcols,
        "attacker_visible_columns": list(ATTACKER_VISIBLE_COLUMNS),
        "validation_assertions": int(len(validations)),
        "validation_passed": int(validations.passed.sum()),
        "all_validation_passed": bool(validations.passed.all()),
        "notes": [
            "Hidden victim region is evaluator-only.",
            "Probe region is attacker-known because the attacker chooses its own target module.",
            "Base victim schedules are crossed with every hidden region, preventing schedule/location confounding.",
            "Grouped train/test splitting is by base schedule instance.",
            "global_only_control intentionally removes spatial identifiability while preserving common communication timing.",
            "Phase 3.3 localizes one persistent active region per trace; time-varying graph reconstruction is deferred to Phase 3.4.",
            "Architecture timings are controlled simulation parameters, not vendor measurements.",
        ],
    }
    with open(output_dir / "phase3_03_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[Phase 3.3] Validation")
    print(validation_summary.to_string(index=False))

    print("\n[Phase 3.3] Protocol / spatial-mode summary")
    if len(protocol_summary):
        print(protocol_summary.to_string(index=False))

    print(f"\n[Phase 3.3] Wrote outputs to: {output_dir}")

    if args.fail_on_validation_error and not bool(validations.passed.all()):
        failed = validations[~validations.passed.astype(bool)]
        raise AssertionError(
            "Phase 3.3 validation failed:\n" + failed.to_string(index=False)
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.3 — Endpoint / Module Localization"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Default: blackbox_window_results/phase3/phase3.3/",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--base-instances",
        type=int,
        default=DEFAULT_BASE_INSTANCES,
        help="Distinct base victim communication schedules before crossing location labels.",
    )
    parser.add_argument(
        "--repeats-per-instance",
        type=int,
        default=DEFAULT_REPEATS,
    )
    parser.add_argument(
        "--observation-window-ns",
        type=float,
        default=DEFAULT_OBSERVATION_WINDOW_NS,
    )
    parser.add_argument(
        "--region-count",
        type=int,
        default=DEFAULT_REGION_COUNT,
        help="Number of candidate remote modules. Default: 4.",
    )
    parser.add_argument(
        "--protocol",
        choices=("direct", "entangled", "both"),
        default="both",
    )
    parser.add_argument(
        "--spatial-modes",
        default=",".join(SPATIAL_MODES),
        help=(
            "Comma-separated subset of: localized_only,hybrid_local_global,"
            "global_only_control"
        ),
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--rf-trees", type=int, default=DEFAULT_RF_TREES)
    parser.add_argument(
        "--probe-budgets",
        default="8,16,24,32,40",
        help="Comma-separated total attacker-probe prefix budgets; full trace is appended.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.region_count < 3:
        raise ValueError("--region-count must be >= 3")
    if args.repeats_per_instance < 1:
        raise ValueError("--repeats-per-instance must be >= 1")
    if not 0.10 <= args.test_size <= 0.50:
        raise ValueError("--test-size must be between 0.10 and 0.50")
    if args.rf_trees < 10:
        raise ValueError("--rf-trees must be >= 10")
    run_experiment(args)


if __name__ == "__main__":
    main()
