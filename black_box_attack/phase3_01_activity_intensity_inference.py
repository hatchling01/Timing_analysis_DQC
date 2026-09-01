#!/usr/bin/env python3
"""
Phase 3.1 — Activity and Communication-Intensity Inference
===========================================================

Purpose
-------
Move beyond Phase-2 workload bar charts and test a consequential black-box
inference objective:

    Can an attacker infer whether the victim is performing remote
    communication, classify its communication-intensity regime, and estimate
    continuous communication demand using only its own probe timing?

This experiment REUSES the validated Phase-2.7 simulator rather than
reimplementing the architecture.  Therefore the Phase-3 inference results are
grounded in the same resource semantics already validated in Phase 2:

* direct coherent remote-CX realization,
* prefetched entanglement-assisted remote-CX realization,
* endpoint/link/switch/reset semantics,
* EPR generation/storage/refill semantics,
* strict attacker-visible vs evaluator-only separation.

Primary inference contexts
--------------------------
By default, Phase 3.1 evaluates two fixed protocol-specific architectures:

1. direct_coherent_remote_cx__all_used_shared
2. entanglement_assisted_remote_cx__all_used_shared

The models are trained and evaluated separately within each context.  This is
intentional: Phase 3.1 asks whether communication activity/intensity is
inferable under a fixed known architecture.  Unknown scheduling, placement,
mapping, and architecture parameters belong to the later open-world robustness
experiments.

Victim labels
-------------
Six activity classes are generated:

1. no_victim
2. local_only
3. sparse_remote
4. moderate_remote
5. heavy_remote
6. bursty_remote

Important negative control:
``local_only`` generates no remote-operation requests in this remote-resource
simulator.  Therefore it SHOULD be indistinguishable from ``no_victim`` through
this channel.  The binary activity label is consequently:

    remote communication active vs. no remote communication

rather than generic victim-presence detection.

Continuous ground-truth targets
-------------------------------
For each trace/circuit instance:

* remote_operation_count
* remote_operation_rate_per_us
* remote_activity_fraction
* total_endpoint_occupancy_ns

Ground truth is derived from the victim logical schedule and victim-only
simulator execution.  These labels are evaluator-only and are never included
in attacker features.

Attacker-observable features only
---------------------------------
Features are computed exclusively from paired attacker timing/success traces:

* mean / median / max / standard deviation of excess latency
* mean absolute timing change
* delayed-probe fraction
* cumulative positive excess
* consecutive delayed-probe statistics
* inter-delay interval statistics
* excess-latency quantiles
* lag-1 / lag-2 / lag-3 autocorrelation
* failure-transition fraction
* optional spectral features from the uniformly sampled probe trace

No victim event times, workload labels, resource waits, EPR state, stage names,
or evaluator attribution are used as model inputs.

Models
------
Classification:
* Logistic regression
* Random forest
* Histogram gradient-boosted trees
* Simple temporal logistic classifier over coarse time bins

Regression:
* Random forest regressor
* Histogram gradient-boosted regressor
* Elastic-net linear regressor

Evaluation discipline
---------------------
Train/test splitting is GROUPED BY VICTIM CIRCUIT INSTANCE.  All repeated
timing realizations of one victim instance remain in the same split.  Splits
are stratified at the circuit-instance level by the six activity classes.

Outputs include:
* binary remote-activity AUC
* six-class accuracy and macro-F1
* remote-operation-count MAE
* regression metrics for all continuous targets
* held-out calibration curves
* performance vs probe budget
* victim slowdown vs inference correctness
* test-set predictions
* split assignments
* black-box trace export
* evaluator-only labels/ground truth
* validation assertions and manifest

Default output directory
------------------------
blackbox_window_results/phase3/phase3.1/

Run
---
From the repository directory containing
``phase2_07_remote_protocol_comparison.py``:

    python phase3_01_activity_intensity_inference.py

Faster smoke test:

    python phase3_01_activity_intensity_inference.py \
        --instances-per-class 4 \
        --repeats-per-instance 1 \
        --observation-window-ns 6000 \
        --probe-budgets 6,10,14

Notes
-----
* All architecture timings are controlled simulation parameters, not measured
  vendor values.
* Phase 3.1 intentionally establishes the minimum inference capability under
  fixed architecture/protocol conditions.
* Open-world, unknown-placement, unknown-scheduler, and architecture-shift
  generalization should be evaluated later rather than mixed into this first
  inference experiment.
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

from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# =============================================================================
# Global settings
# =============================================================================

DEFAULT_SEED = 3101
DEFAULT_INSTANCES_PER_CLASS = 18
DEFAULT_REPEATS_PER_INSTANCE = 2
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_TEST_SIZE = 0.30
DEFAULT_TEMPORAL_BINS = 12

DEFAULT_OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase3"
    / "phase3.1"
)

ACTIVITY_CLASSES = (
    "no_victim",
    "local_only",
    "sparse_remote",
    "moderate_remote",
    "heavy_remote",
    "bursty_remote",
)

REMOTE_CLASSES = {
    "sparse_remote",
    "moderate_remote",
    "heavy_remote",
    "bursty_remote",
}

AFFECTED_THRESHOLD_NS = 1e-9
EPS = 1e-12

# Fields that may be exported to the attacker-facing trace file.
ATTACKER_VISIBLE_COLUMNS = (
    "trace_id",
    "protocol_context",
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
)

# Summary features used by the non-temporal ML models.
SUMMARY_FEATURES = [
    "probe_count",
    "mean_excess_latency_ns",
    "median_excess_latency_ns",
    "maximum_excess_latency_ns",
    "minimum_excess_latency_ns",
    "std_excess_latency_ns",
    "mean_absolute_timing_change_ns",
    "delayed_probe_fraction",
    "speedup_probe_fraction",
    "failure_transition_fraction",
    "cumulative_positive_excess_ns",
    "cumulative_absolute_timing_change_ns",
    "p10_excess_latency_ns",
    "p25_excess_latency_ns",
    "p50_excess_latency_ns",
    "p75_excess_latency_ns",
    "p90_excess_latency_ns",
    "p95_excess_latency_ns",
    "p99_excess_latency_ns",
    "longest_delayed_run",
    "delayed_run_count",
    "mean_delayed_run_length",
    "max_delayed_run_length",
    "mean_inter_delay_probes",
    "std_inter_delay_probes",
    "median_inter_delay_probes",
    "min_inter_delay_probes",
    "max_inter_delay_probes",
    "lag1_excess_autocorrelation",
    "lag2_excess_autocorrelation",
    "lag3_excess_autocorrelation",
    "spectral_dominant_bin_fraction",
    "spectral_centroid_fraction",
    "spectral_entropy",
    "spectral_low_frequency_power_fraction",
]

CONTINUOUS_TARGETS = (
    "remote_operation_count",
    "remote_operation_rate_per_us",
    "remote_activity_fraction",
    "total_endpoint_occupancy_ns",
)


# =============================================================================
# Phase-2.7 simulator loading
# =============================================================================


def load_phase2_07_module():
    """
    Reuse the exact Phase-2.7 simulator.

    We intentionally depend on the validated Phase-2 implementation so Phase 3
    changes only the inference objective, victim schedules, and ML analysis.
    """
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
            "Phase 3.1 intentionally reuses the validated Phase-2.7 simulator.\n"
            f"Searched:\n{searched}"
        )

    module_name = "phase2_07_remote_protocol_comparison"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {source}")

    module = importlib.util.module_from_spec(spec)
    # dataclasses needs the module registered before exec_module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, source


# =============================================================================
# Victim-instance model
# =============================================================================


@dataclass(frozen=True)
class VictimInstance:
    victim_instance_id: str
    activity_class: str
    instance_index: int
    base_phase_ns: float
    interval_ns: float
    burst_period_ns: float
    burst_size: int
    burst_spacing_ns: float
    local_activity_fraction: float

    @property
    def remote_activity_label(self) -> int:
        return int(self.activity_class in REMOTE_CLASSES)


def stable_seed(*parts: Any, modulus: int = 2**32 - 1) -> int:
    token = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % modulus


def make_victim_instances(
    *,
    seed: int,
    instances_per_class: int,
    observation_window_ns: float,
) -> list[VictimInstance]:
    rows: list[VictimInstance] = []

    for class_name in ACTIVITY_CLASSES:
        for idx in range(instances_per_class):
            rng = np.random.default_rng(stable_seed(seed, class_name, idx, "instance"))

            # Keep phase away from exact attacker alignment while still varying
            # meaningfully across circuit instances.
            base_phase = float(rng.uniform(80.0, 420.0))
            local_fraction = float(rng.uniform(0.45, 0.95))

            if class_name == "sparse_remote":
                interval = float(rng.uniform(950.0, 1250.0))
                burst_period = 0.0
                burst_size = 1
                burst_spacing = 0.0
            elif class_name == "moderate_remote":
                interval = float(rng.uniform(650.0, 820.0))
                burst_period = 0.0
                burst_size = 1
                burst_spacing = 0.0
            elif class_name == "heavy_remote":
                interval = float(rng.uniform(410.0, 520.0))
                burst_period = 0.0
                burst_size = 1
                burst_spacing = 0.0
            elif class_name == "bursty_remote":
                interval = 0.0
                burst_period = float(rng.uniform(1350.0, 1850.0))
                burst_size = int(rng.integers(3, 7))
                burst_spacing = float(rng.uniform(45.0, 100.0))
            else:
                interval = 0.0
                burst_period = 0.0
                burst_size = 0
                burst_spacing = 0.0

            rows.append(
                VictimInstance(
                    victim_instance_id=f"{class_name}::instance_{idx:03d}",
                    activity_class=class_name,
                    instance_index=idx,
                    base_phase_ns=base_phase,
                    interval_ns=interval,
                    burst_period_ns=burst_period,
                    burst_size=burst_size,
                    burst_spacing_ns=burst_spacing,
                    local_activity_fraction=local_fraction,
                )
            )

    return rows


def generate_remote_release_times(
    instance: VictimInstance,
    *,
    repeat_id: int,
    seed: int,
    observation_window_ns: float,
) -> np.ndarray:
    """
    Generate one timing realization of the victim circuit instance.

    Repeats share the same high-level circuit instance parameters but receive
    small timing jitter.  Grouped splitting prevents repeats of one instance
    from crossing train/test.
    """
    if instance.activity_class not in REMOTE_CLASSES:
        return np.array([], dtype=float)

    rng = np.random.default_rng(
        stable_seed(seed, instance.victim_instance_id, repeat_id, "repeat")
    )

    phase = instance.base_phase_ns + float(rng.uniform(-12.0, 12.0))

    if instance.activity_class in {
        "sparse_remote",
        "moderate_remote",
        "heavy_remote",
    }:
        releases = np.arange(
            phase,
            observation_window_ns,
            instance.interval_ns,
            dtype=float,
        )
    elif instance.activity_class == "bursty_remote":
        values: list[float] = []
        base = phase
        while base < observation_window_ns:
            for j in range(instance.burst_size):
                values.append(base + j * instance.burst_spacing_ns)
            base += instance.burst_period_ns
        releases = np.asarray(values, dtype=float)
    else:
        raise ValueError(instance.activity_class)

    releases = releases[(releases >= 0.0) & (releases < observation_window_ns)]

    if len(releases):
        # Small hardware/scheduler-scale jitter while preserving the circuit
        # instance's communication structure.
        releases = releases + rng.uniform(-4.0, 4.0, size=len(releases))
        releases = np.clip(releases, 0.0, observation_window_ns - 1e-6)

    return np.sort(releases)


def make_request_specs(
    p27,
    *,
    tenant: str,
    releases: np.ndarray,
    activity_class: str,
    instance_id: str,
    repeat_id: int,
) -> list[Any]:
    return [
        p27.RequestSpec(
            request_id=f"{tenant}::{instance_id}::repeat_{repeat_id:02d}::{i}",
            tenant=tenant,
            ready_ns=float(t),
            request_index=i,
            workload_name=activity_class,
            trial_id=repeat_id,
        )
        for i, t in enumerate(releases)
    ]


# =============================================================================
# Trace pairing and attacker-only feature extraction
# =============================================================================


def attacker_probe_specs(p27, *, repeat_id: int, observation_window_ns: float) -> list[Any]:
    releases = np.arange(
        p27.ATTACKER_FIRST_RELEASE_NS,
        observation_window_ns,
        p27.ATTACKER_PERIOD_NS,
        dtype=float,
    )
    return [
        p27.RequestSpec(
            request_id=f"attacker::repeat_{repeat_id:02d}::{i}",
            tenant="attacker",
            ready_ns=float(t),
            request_index=i,
            workload_name="opaque",
            trial_id=repeat_id,
        )
        for i, t in enumerate(releases)
    ]


def pair_attacker_blackbox(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    trace_id: str,
    protocol_context: str,
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


def run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


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
    """
    Spectral features are justified because attacker probes are uniformly spaced
    in time.  We use normalized FFT-bin quantities rather than converting them
    to a vendor-specific physical frequency.
    """
    if len(values) < 4:
        return {
            "spectral_dominant_bin_fraction": 0.0,
            "spectral_centroid_fraction": 0.0,
            "spectral_entropy": 0.0,
            "spectral_low_frequency_power_fraction": 0.0,
        }

    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2

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
    dominant = int(np.argmax(power))
    dominant_fraction = dominant / max_bin

    centroid = float(np.sum(bins * power) / total) / max_bin

    probs = power / total
    nz = probs[probs > EPS]
    entropy = -float(np.sum(nz * np.log(nz)))
    entropy /= math.log(len(power)) if len(power) > 1 else 1.0

    cutoff = max(2, int(math.ceil(len(power) * 0.25)))
    low_fraction = float(np.sum(power[1:cutoff]) / total)

    return {
        "spectral_dominant_bin_fraction": dominant_fraction,
        "spectral_centroid_fraction": centroid,
        "spectral_entropy": entropy,
        "spectral_low_frequency_power_fraction": low_fraction,
    }


def extract_summary_features(
    trace: pd.DataFrame,
    *,
    probe_budget: int | None = None,
    include_spectral: bool = True,
) -> dict[str, float]:
    t = trace.sort_values("probe_index")
    if probe_budget is not None:
        t = t.head(int(probe_budget))

    excess = t["excess_turnaround_ns"].to_numpy(dtype=float)
    delayed = excess > AFFECTED_THRESHOLD_NS
    speedup = excess < -AFFECTED_THRESHOLD_NS
    failures = t["failure_transition"].to_numpy(dtype=bool)

    positive = np.maximum(excess, 0.0)
    abs_change = np.abs(excess)

    delayed_idx = np.flatnonzero(delayed)
    if len(delayed_idx) >= 2:
        inter = np.diff(delayed_idx).astype(float)
    else:
        inter = np.array([], dtype=float)

    runs = run_lengths(delayed)

    def q(p: float) -> float:
        return float(np.quantile(excess, p)) if len(excess) else 0.0

    row = {
        "probe_count": float(len(t)),
        "mean_excess_latency_ns": float(np.mean(excess)) if len(excess) else 0.0,
        "median_excess_latency_ns": float(np.median(excess)) if len(excess) else 0.0,
        "maximum_excess_latency_ns": float(np.max(excess)) if len(excess) else 0.0,
        "minimum_excess_latency_ns": float(np.min(excess)) if len(excess) else 0.0,
        "std_excess_latency_ns": float(np.std(excess)) if len(excess) else 0.0,
        "mean_absolute_timing_change_ns": float(np.mean(abs_change)) if len(excess) else 0.0,
        "delayed_probe_fraction": float(np.mean(delayed)) if len(excess) else 0.0,
        "speedup_probe_fraction": float(np.mean(speedup)) if len(excess) else 0.0,
        "failure_transition_fraction": float(np.mean(failures)) if len(excess) else 0.0,
        "cumulative_positive_excess_ns": float(np.sum(positive)),
        "cumulative_absolute_timing_change_ns": float(np.sum(abs_change)),
        "p10_excess_latency_ns": q(0.10),
        "p25_excess_latency_ns": q(0.25),
        "p50_excess_latency_ns": q(0.50),
        "p75_excess_latency_ns": q(0.75),
        "p90_excess_latency_ns": q(0.90),
        "p95_excess_latency_ns": q(0.95),
        "p99_excess_latency_ns": q(0.99),
        "longest_delayed_run": float(max(runs) if runs else 0),
        "delayed_run_count": float(len(runs)),
        "mean_delayed_run_length": float(np.mean(runs)) if runs else 0.0,
        "max_delayed_run_length": float(max(runs) if runs else 0.0),
        "mean_inter_delay_probes": float(np.mean(inter)) if len(inter) else 0.0,
        "std_inter_delay_probes": float(np.std(inter)) if len(inter) else 0.0,
        "median_inter_delay_probes": float(np.median(inter)) if len(inter) else 0.0,
        "min_inter_delay_probes": float(np.min(inter)) if len(inter) else 0.0,
        "max_inter_delay_probes": float(np.max(inter)) if len(inter) else 0.0,
        "lag1_excess_autocorrelation": autocorrelation(excess, 1),
        "lag2_excess_autocorrelation": autocorrelation(excess, 2),
        "lag3_excess_autocorrelation": autocorrelation(excess, 3),
    }

    if include_spectral:
        row.update(spectral_features(excess))
    else:
        row.update(
            {
                "spectral_dominant_bin_fraction": 0.0,
                "spectral_centroid_fraction": 0.0,
                "spectral_entropy": 0.0,
                "spectral_low_frequency_power_fraction": 0.0,
            }
        )

    return row


def extract_temporal_features(
    trace: pd.DataFrame,
    *,
    temporal_bins: int,
    probe_budget: int | None = None,
) -> dict[str, float]:
    t = trace.sort_values("probe_index")
    if probe_budget is not None:
        t = t.head(int(probe_budget))

    excess = t["excess_turnaround_ns"].to_numpy(dtype=float)
    failure = t["failure_transition"].to_numpy(dtype=float)

    chunks = np.array_split(np.arange(len(t)), temporal_bins)
    row: dict[str, float] = {}
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            mean_pos = 0.0
            delayed_fraction = 0.0
            mean_abs = 0.0
            failure_fraction = 0.0
        else:
            vals = excess[idx]
            mean_pos = float(np.mean(np.maximum(vals, 0.0)))
            delayed_fraction = float(np.mean(vals > AFFECTED_THRESHOLD_NS))
            mean_abs = float(np.mean(np.abs(vals)))
            failure_fraction = float(np.mean(failure[idx]))
        row[f"temporal_bin_{i:02d}_mean_positive_ns"] = mean_pos
        row[f"temporal_bin_{i:02d}_delayed_fraction"] = delayed_fraction
        row[f"temporal_bin_{i:02d}_mean_abs_ns"] = mean_abs
        row[f"temporal_bin_{i:02d}_failure_fraction"] = failure_fraction
    return row


# =============================================================================
# Ground truth from victim-only execution
# =============================================================================


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    spans = sorted(
        (float(a), float(b))
        for a, b in intervals
        if np.isfinite(a) and np.isfinite(b) and b > a
    )
    if not spans:
        return 0.0

    total = 0.0
    cur_a, cur_b = spans[0]
    for a, b in spans[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    total += cur_b - cur_a
    return float(total)


def victim_ground_truth(
    victim_specs: list[Any],
    victim_only_requests: pd.DataFrame,
    victim_only_intervals: pd.DataFrame,
    *,
    observation_window_ns: float,
) -> dict[str, float]:
    remote_count = int(len(victim_specs))
    window_us = observation_window_ns / 1000.0
    remote_rate = remote_count / window_us if window_us > 0 else 0.0

    victim_req = victim_only_requests[
        victim_only_requests["tenant"] == "victim"
    ].copy()

    active_spans: list[tuple[float, float]] = []
    if not victim_req.empty:
        for row in victim_req.itertuples(index=False):
            if np.isfinite(row.release_ns) and np.isfinite(row.external_completion_ns):
                active_spans.append(
                    (
                        max(0.0, float(row.release_ns)),
                        min(observation_window_ns, float(row.external_completion_ns)),
                    )
                )

    active_ns = union_duration(active_spans)
    activity_fraction = (
        active_ns / observation_window_ns if observation_window_ns > 0 else 0.0
    )

    endpoint = victim_only_intervals[
        (victim_only_intervals["tenant"] == "victim")
        & (victim_only_intervals["resource_name"] == "endpoint")
    ]
    endpoint_occupancy = (
        float(endpoint["duration_ns"].sum()) if not endpoint.empty else 0.0
    )

    return {
        "remote_operation_count": float(remote_count),
        "remote_operation_rate_per_us": float(remote_rate),
        "remote_activity_fraction": float(activity_fraction),
        "total_endpoint_occupancy_ns": float(endpoint_occupancy),
    }


# =============================================================================
# Dataset generation
# =============================================================================


def select_contexts(p27, protocol_choice: str):
    protocols = p27.build_protocols()
    scenarios = {s.scenario_id: s for s in p27.build_scenarios(protocols)}

    wanted: list[tuple[str, Any, Any]] = []

    if protocol_choice in {"direct", "both"}:
        pid = p27.DIRECT_PROTOCOL
        sid = f"{pid}__all_used_shared"
        wanted.append(("direct_coherent_all_shared", protocols[pid], scenarios[sid]))

    if protocol_choice in {"entangled", "both"}:
        pid = p27.ENTANGLED_PROTOCOL
        sid = f"{pid}__all_used_shared"
        wanted.append(("entanglement_assisted_all_shared", protocols[pid], scenarios[sid]))

    return wanted


def run_dataset(
    p27,
    *,
    contexts,
    instances: list[VictimInstance],
    repeats_per_instance: int,
    seed: int,
    observation_window_ns: float,
    temporal_bins: int,
    include_spectral: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blackbox_parts: list[pd.DataFrame] = []
    label_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    total = len(contexts) * len(instances) * repeats_per_instance
    done = 0

    # Attacker-only timing depends on protocol/context/repeat, not victim
    # instance.  Cache it to avoid redundant simulation.
    attacker_only_cache: dict[tuple[str, int], pd.DataFrame] = {}

    for protocol_context, protocol, scenario in contexts:
        for repeat_id in range(repeats_per_instance):
            attacker_specs = attacker_probe_specs(
                p27,
                repeat_id=repeat_id,
                observation_window_ns=observation_window_ns,
            )
            attacker_only, *_ = p27.run_one(
                protocol,
                scenario,
                "opaque",
                repeat_id,
                "attacker_only",
                list(attacker_specs),
            )
            attacker_only_cache[(protocol_context, repeat_id)] = attacker_only

        for instance in instances:
            for repeat_id in range(repeats_per_instance):
                attacker_specs = attacker_probe_specs(
                    p27,
                    repeat_id=repeat_id,
                    observation_window_ns=observation_window_ns,
                )
                release_times = generate_remote_release_times(
                    instance,
                    repeat_id=repeat_id,
                    seed=seed,
                    observation_window_ns=observation_window_ns,
                )
                victim_specs = make_request_specs(
                    p27,
                    tenant="victim",
                    releases=release_times,
                    activity_class=instance.activity_class,
                    instance_id=instance.victim_instance_id,
                    repeat_id=repeat_id,
                )

                # Victim-only is needed for ground truth and victim slowdown.
                if victim_specs:
                    (
                        victim_only,
                        _vw,
                        victim_intervals,
                        _vst,
                        _ve,
                        _vg,
                    ) = p27.run_one(
                        protocol,
                        scenario,
                        instance.activity_class,
                        repeat_id,
                        "victim_only",
                        list(victim_specs),
                    )
                else:
                    victim_only = pd.DataFrame()
                    victim_intervals = pd.DataFrame(
                        columns=["tenant", "resource_name", "duration_ns"]
                    )

                combined_specs = sorted(
                    list(attacker_specs) + list(victim_specs),
                    key=lambda x: (x.ready_ns, x.tenant, x.request_index),
                )
                combined, *_ = p27.run_one(
                    protocol,
                    scenario,
                    instance.activity_class,
                    repeat_id,
                    "combined",
                    combined_specs,
                )

                attacker_only = attacker_only_cache[(protocol_context, repeat_id)]

                trace_id = hashlib.sha256(
                    (
                        f"{protocol_context}|{instance.victim_instance_id}|"
                        f"{repeat_id}|{seed}"
                    ).encode()
                ).hexdigest()[:20]

                trace = pair_attacker_blackbox(
                    attacker_only,
                    combined,
                    trace_id=trace_id,
                    protocol_context=protocol_context,
                )
                blackbox_parts.append(trace)

                summary = extract_summary_features(
                    trace,
                    include_spectral=include_spectral,
                )
                temporal = extract_temporal_features(
                    trace,
                    temporal_bins=temporal_bins,
                )

                if victim_specs:
                    gt = victim_ground_truth(
                        victim_specs,
                        victim_only,
                        victim_intervals,
                        observation_window_ns=observation_window_ns,
                    )
                    slowdown = p27.victim_slowdown_metrics(victim_only, combined)
                else:
                    gt = {
                        "remote_operation_count": 0.0,
                        "remote_operation_rate_per_us": 0.0,
                        "remote_activity_fraction": 0.0,
                        "total_endpoint_occupancy_ns": 0.0,
                    }
                    slowdown = {
                        "victim_mean_request_slowdown": 1.0,
                        "victim_makespan_slowdown": 1.0,
                        "victim_mean_added_turnaround_ns": 0.0,
                    }

                # Labels / evaluator truth are stored separately from the
                # attacker-facing trace.
                label_row = {
                    "trace_id": trace_id,
                    "protocol_context": protocol_context,
                    "victim_instance_id": instance.victim_instance_id,
                    "repeat_id": repeat_id,
                    "activity_class": instance.activity_class,
                    "remote_activity_label": instance.remote_activity_label,
                    "victim_present_label": int(instance.activity_class != "no_victim"),
                    "local_activity_fraction_evaluator_only": instance.local_activity_fraction,
                    **gt,
                    **slowdown,
                }
                label_rows.append(label_row)

                feature_rows.append(
                    {
                        "trace_id": trace_id,
                        "protocol_context": protocol_context,
                        "victim_instance_id": instance.victim_instance_id,
                        "repeat_id": repeat_id,
                        **summary,
                        **temporal,
                    }
                )

                trial_rows.append(
                    {
                        **label_row,
                        **summary,
                    }
                )

                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print(f"[Phase 3.1] Generated {done}/{total} traces")

    blackbox = (
        pd.concat(blackbox_parts, ignore_index=True)
        if blackbox_parts
        else pd.DataFrame(columns=ATTACKER_VISIBLE_COLUMNS)
    )
    labels = pd.DataFrame(label_rows)
    features = pd.DataFrame(feature_rows)
    trials = pd.DataFrame(trial_rows)

    return blackbox, labels, features, trials


# =============================================================================
# Grouped train/test split
# =============================================================================


def build_group_split(
    instances: list[VictimInstance],
    *,
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    group_table = pd.DataFrame(
        [
            {
                "victim_instance_id": x.victim_instance_id,
                "activity_class": x.activity_class,
            }
            for x in instances
        ]
    ).drop_duplicates()

    train_ids, test_ids = train_test_split(
        group_table["victim_instance_id"],
        test_size=test_size,
        random_state=seed,
        stratify=group_table["activity_class"],
    )

    train_set = set(train_ids)
    test_set = set(test_ids)

    out = group_table.copy()
    out["split"] = out["victim_instance_id"].map(
        lambda x: "train" if x in train_set else "test"
    )
    assert not (train_set & test_set)
    return out.sort_values(["activity_class", "victim_instance_id"]).reset_index(drop=True)


# =============================================================================
# ML helpers
# =============================================================================


def classification_models(seed: int):
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "gradient_boosted_trees": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        ),
    }


def temporal_model(seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def regression_models(seed: int):
    return {
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=350,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "gradient_boosted_regressor": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        ),
        "elastic_net": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.05,
                        l1_ratio=0.25,
                        max_iter=20000,
                        tol=1e-3,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def safe_binary_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, prob))


def probability_for_positive_class(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        if 1 in classes:
            return probs[:, classes.index(1)]
        return np.zeros(len(X), dtype=float)

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))

    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def evaluate_models(
    data: pd.DataFrame,
    split: pd.DataFrame,
    *,
    seed: int,
    temporal_feature_columns: list[str],
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    split_map = split.set_index("victim_instance_id")["split"].to_dict()
    data = data.copy()
    data["split"] = data["victim_instance_id"].map(split_map)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    regression_metric_rows: list[dict[str, Any]] = []
    regression_prediction_rows: list[dict[str, Any]] = []

    for protocol_context, group in data.groupby("protocol_context", sort=True):
        train = group[group["split"] == "train"].copy()
        test = group[group["split"] == "test"].copy()

        X_train = train[feature_columns].astype(float)
        X_test = test[feature_columns].astype(float)

        # -------------------------------------------------------------
        # Binary: remote communication active vs no remote communication
        # -------------------------------------------------------------
        y_train_bin = train["remote_activity_label"].astype(int).to_numpy()
        y_test_bin = test["remote_activity_label"].astype(int).to_numpy()

        models = classification_models(seed)
        for model_name, model in models.items():
            model.fit(X_train, y_train_bin)
            prob = probability_for_positive_class(model, X_test)
            pred = (prob >= 0.5).astype(int)

            auc = safe_binary_auc(y_test_bin, prob)
            acc = float(accuracy_score(y_test_bin, pred))
            f1 = float(f1_score(y_test_bin, pred, zero_division=0))

            metric_rows.append(
                {
                    "protocol_context": protocol_context,
                    "task": "binary_remote_activity",
                    "model": model_name,
                    "train_trace_count": len(train),
                    "test_trace_count": len(test),
                    "metric_primary": "roc_auc",
                    "roc_auc": auc,
                    "accuracy": acc,
                    "macro_f1": f1,
                }
            )

            for row, actual, predicted, p in zip(
                test.itertuples(index=False), y_test_bin, pred, prob
            ):
                prediction_rows.append(
                    {
                        "trace_id": row.trace_id,
                        "protocol_context": protocol_context,
                        "task": "binary_remote_activity",
                        "model": model_name,
                        "actual": int(actual),
                        "predicted": int(predicted),
                        "score": float(p),
                        "victim_instance_id": row.victim_instance_id,
                        "activity_class": row.activity_class,
                        "victim_makespan_slowdown": float(row.victim_makespan_slowdown),
                    }
                )

            # Held-out calibration curve.
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_test_bin,
                    prob,
                    n_bins=min(10, max(3, len(test) // 8)),
                    strategy="quantile",
                )
                for bin_idx, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
                    calibration_rows.append(
                        {
                            "protocol_context": protocol_context,
                            "model": model_name,
                            "bin_index": bin_idx,
                            "mean_predicted_probability": float(mp),
                            "observed_remote_activity_fraction": float(fp),
                        }
                    )
            except ValueError:
                pass

        # Simple temporal classifier.
        X_train_t = train[temporal_feature_columns].astype(float)
        X_test_t = test[temporal_feature_columns].astype(float)
        tmodel = temporal_model(seed)
        tmodel.fit(X_train_t, y_train_bin)
        tprob = probability_for_positive_class(tmodel, X_test_t)
        tpred = (tprob >= 0.5).astype(int)

        metric_rows.append(
            {
                "protocol_context": protocol_context,
                "task": "binary_remote_activity",
                "model": "temporal_logistic",
                "train_trace_count": len(train),
                "test_trace_count": len(test),
                "metric_primary": "roc_auc",
                "roc_auc": safe_binary_auc(y_test_bin, tprob),
                "accuracy": float(accuracy_score(y_test_bin, tpred)),
                "macro_f1": float(f1_score(y_test_bin, tpred, zero_division=0)),
            }
        )

        for row, actual, predicted, p in zip(
            test.itertuples(index=False), y_test_bin, tpred, tprob
        ):
            prediction_rows.append(
                {
                    "trace_id": row.trace_id,
                    "protocol_context": protocol_context,
                    "task": "binary_remote_activity",
                    "model": "temporal_logistic",
                    "actual": int(actual),
                    "predicted": int(predicted),
                    "score": float(p),
                    "victim_instance_id": row.victim_instance_id,
                    "activity_class": row.activity_class,
                    "victim_makespan_slowdown": float(row.victim_makespan_slowdown),
                }
            )

        # -------------------------------------------------------------
        # Six-class communication-intensity/activity regime
        # -------------------------------------------------------------
        y_train_multi = train["activity_class"].astype(str)
        y_test_multi = test["activity_class"].astype(str)

        for model_name, model in classification_models(seed + 17).items():
            model.fit(X_train, y_train_multi)
            pred = model.predict(X_test)

            metric_rows.append(
                {
                    "protocol_context": protocol_context,
                    "task": "six_class_activity_intensity",
                    "model": model_name,
                    "train_trace_count": len(train),
                    "test_trace_count": len(test),
                    "metric_primary": "accuracy",
                    "roc_auc": math.nan,
                    "accuracy": float(accuracy_score(y_test_multi, pred)),
                    "macro_f1": float(
                        f1_score(y_test_multi, pred, average="macro", zero_division=0)
                    ),
                }
            )

            for row, actual, predicted in zip(
                test.itertuples(index=False), y_test_multi, pred
            ):
                prediction_rows.append(
                    {
                        "trace_id": row.trace_id,
                        "protocol_context": protocol_context,
                        "task": "six_class_activity_intensity",
                        "model": model_name,
                        "actual": str(actual),
                        "predicted": str(predicted),
                        "score": math.nan,
                        "victim_instance_id": row.victim_instance_id,
                        "activity_class": row.activity_class,
                        "victim_makespan_slowdown": float(row.victim_makespan_slowdown),
                    }
                )

        # Temporal multi-class model.
        tmulti = temporal_model(seed + 17)
        tmulti.fit(X_train_t, y_train_multi)
        tpred_multi = tmulti.predict(X_test_t)
        metric_rows.append(
            {
                "protocol_context": protocol_context,
                "task": "six_class_activity_intensity",
                "model": "temporal_logistic",
                "train_trace_count": len(train),
                "test_trace_count": len(test),
                "metric_primary": "accuracy",
                "roc_auc": math.nan,
                "accuracy": float(accuracy_score(y_test_multi, tpred_multi)),
                "macro_f1": float(
                    f1_score(
                        y_test_multi,
                        tpred_multi,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

        # -------------------------------------------------------------
        # Continuous communication-intensity regression
        # -------------------------------------------------------------
        for target in CONTINUOUS_TARGETS:
            y_train = train[target].astype(float).to_numpy()
            y_test = test[target].astype(float).to_numpy()

            for model_name, model in regression_models(seed + 31).items():
                model.fit(X_train, y_train)
                pred = np.asarray(model.predict(X_test), dtype=float)

                # Targets are physically nonnegative.
                pred = np.maximum(pred, 0.0)

                regression_metric_rows.append(
                    {
                        "protocol_context": protocol_context,
                        "target": target,
                        "model": model_name,
                        "train_trace_count": len(train),
                        "test_trace_count": len(test),
                        "mae": float(mean_absolute_error(y_test, pred)),
                        "rmse": float(
                            math.sqrt(mean_squared_error(y_test, pred))
                        ),
                        "r2": float(r2_score(y_test, pred)),
                    }
                )

                for row, actual, predicted in zip(
                    test.itertuples(index=False), y_test, pred
                ):
                    regression_prediction_rows.append(
                        {
                            "trace_id": row.trace_id,
                            "protocol_context": protocol_context,
                            "target": target,
                            "model": model_name,
                            "actual": float(actual),
                            "predicted": float(predicted),
                            "absolute_error": float(abs(actual - predicted)),
                            "victim_instance_id": row.victim_instance_id,
                            "activity_class": row.activity_class,
                        }
                    )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(calibration_rows),
        pd.DataFrame(regression_metric_rows),
        pd.DataFrame(regression_prediction_rows),
    )


# =============================================================================
# Probe-budget analysis
# =============================================================================


def rebuild_budget_features(
    blackbox: pd.DataFrame,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    *,
    budgets: list[int],
    temporal_bins: int,
    include_spectral: bool,
    seed: int,
) -> pd.DataFrame:
    label_lookup = labels.set_index("trace_id")
    split_map = split.set_index("victim_instance_id")["split"].to_dict()
    rows: list[dict[str, Any]] = []

    for budget in budgets:
        budget_feature_rows: list[dict[str, Any]] = []

        for trace_id, trace in blackbox.groupby("trace_id", sort=False):
            lab = label_lookup.loc[trace_id]
            summary = extract_summary_features(
                trace,
                probe_budget=budget,
                include_spectral=include_spectral,
            )
            temporal = extract_temporal_features(
                trace,
                temporal_bins=temporal_bins,
                probe_budget=budget,
            )
            budget_feature_rows.append(
                {
                    "trace_id": trace_id,
                    "protocol_context": lab["protocol_context"],
                    "victim_instance_id": lab["victim_instance_id"],
                    "activity_class": lab["activity_class"],
                    "remote_activity_label": int(lab["remote_activity_label"]),
                    "remote_operation_count": float(lab["remote_operation_count"]),
                    **summary,
                    **temporal,
                }
            )

        df = pd.DataFrame(budget_feature_rows)
        df["split"] = df["victim_instance_id"].map(split_map)

        temporal_cols = [
            c for c in df.columns if c.startswith("temporal_bin_")
        ]

        for protocol_context, group in df.groupby("protocol_context", sort=True):
            train = group[group["split"] == "train"]
            test = group[group["split"] == "test"]

            Xtr = train[SUMMARY_FEATURES].astype(float)
            Xte = test[SUMMARY_FEATURES].astype(float)
            ytr = train["remote_activity_label"].astype(int).to_numpy()
            yte = test["remote_activity_label"].astype(int).to_numpy()

            # Classification budget curves for all requested model families.
            for model_name, model in classification_models(seed + budget).items():
                model.fit(Xtr, ytr)
                prob = probability_for_positive_class(model, Xte)
                pred = (prob >= 0.5).astype(int)
                rows.append(
                    {
                        "probe_budget": budget,
                        "protocol_context": protocol_context,
                        "task": "binary_remote_activity",
                        "model": model_name,
                        "roc_auc": safe_binary_auc(yte, prob),
                        "accuracy": float(accuracy_score(yte, pred)),
                        "mae": math.nan,
                    }
                )

            tmodel = temporal_model(seed + budget)
            tmodel.fit(
                train[temporal_cols].astype(float),
                ytr,
            )
            prob = probability_for_positive_class(
                tmodel,
                test[temporal_cols].astype(float),
            )
            pred = (prob >= 0.5).astype(int)
            rows.append(
                {
                    "probe_budget": budget,
                    "protocol_context": protocol_context,
                    "task": "binary_remote_activity",
                    "model": "temporal_logistic",
                    "roc_auc": safe_binary_auc(yte, prob),
                    "accuracy": float(accuracy_score(yte, pred)),
                    "mae": math.nan,
                }
            )

            # Remote-operation count is the primary continuous budget target.
            reg = RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed + budget,
            )
            reg.fit(
                Xtr,
                train["remote_operation_count"].astype(float),
            )
            pred_count = np.maximum(reg.predict(Xte), 0.0)
            rows.append(
                {
                    "probe_budget": budget,
                    "protocol_context": protocol_context,
                    "task": "remote_operation_count",
                    "model": "random_forest_regressor",
                    "roc_auc": math.nan,
                    "accuracy": math.nan,
                    "mae": float(
                        mean_absolute_error(
                            test["remote_operation_count"].astype(float),
                            pred_count,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Slowdown vs inference
# =============================================================================


def slowdown_accuracy_bins(predictions: pd.DataFrame) -> pd.DataFrame:
    binary = predictions[
        predictions["task"] == "binary_remote_activity"
    ].copy()
    if binary.empty:
        return pd.DataFrame()

    binary["correct"] = (
        binary["actual"].astype(str) == binary["predicted"].astype(str)
    )

    rows: list[dict[str, Any]] = []
    for (context, model), group in binary.groupby(
        ["protocol_context", "model"], sort=True
    ):
        g = group.copy()
        # qcut can collapse bins when slowdown is exactly 1 for many samples.
        try:
            g["slowdown_bin"] = pd.qcut(
                g["victim_makespan_slowdown"],
                q=4,
                duplicates="drop",
            ).astype(str)
        except ValueError:
            g["slowdown_bin"] = "all"

        for bin_name, sub in g.groupby("slowdown_bin", sort=False):
            rows.append(
                {
                    "protocol_context": context,
                    "model": model,
                    "slowdown_bin": str(bin_name),
                    "sample_count": len(sub),
                    "mean_victim_makespan_slowdown": float(
                        sub["victim_makespan_slowdown"].mean()
                    ),
                    "binary_accuracy": float(sub["correct"].mean()),
                    "mean_activity_score": float(
                        pd.to_numeric(sub["score"], errors="coerce").mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Validation
# =============================================================================


def build_validations(
    *,
    p27,
    blackbox: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    split: pd.DataFrame,
    contexts,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, expected: str, observed: Any, details: str = ""):
        rows.append(
            {
                "assertion_name": name,
                "passed": bool(passed),
                "expected": expected,
                "observed": str(observed),
                "details": details,
            }
        )

    add(
        "blackbox_columns_are_attacker_only",
        set(blackbox.columns) == set(ATTACKER_VISIBLE_COLUMNS),
        str(sorted(ATTACKER_VISIBLE_COLUMNS)),
        sorted(blackbox.columns),
    )

    forbidden_feature_tokens = (
        "activity_class",
        "remote_operation_count",
        "victim_",
        "resource",
        "epr_",
        "wait_",
        "stage",
    )
    actual_model_features = set(SUMMARY_FEATURES) | {
        c for c in features.columns if c.startswith("temporal_bin_")
    }
    bad_features = sorted(
        f
        for f in actual_model_features
        if any(tok in f for tok in forbidden_feature_tokens)
    )
    add(
        "model_features_exclude_ground_truth_and_internal_state",
        len(bad_features) == 0,
        "no forbidden evaluator/internal tokens",
        bad_features,
    )

    local = labels[labels["activity_class"] == "local_only"]
    novictim = labels[labels["activity_class"] == "no_victim"]
    remote = labels[labels["activity_class"].isin(REMOTE_CLASSES)]

    add(
        "no_victim_has_zero_remote_operations",
        bool((novictim["remote_operation_count"] == 0).all()),
        "all zero",
        novictim["remote_operation_count"].unique().tolist(),
    )
    add(
        "local_only_has_zero_remote_operations",
        bool((local["remote_operation_count"] == 0).all()),
        "all zero",
        local["remote_operation_count"].unique().tolist(),
    )
    add(
        "remote_classes_have_positive_remote_operations",
        bool((remote["remote_operation_count"] > 0).all()),
        "all > 0",
        float(remote["remote_operation_count"].min()) if len(remote) else math.nan,
    )

    # In this remote-resource-only architecture, no-victim and local-only
    # should produce no differential remote timing.
    merged = labels[["trace_id", "activity_class"]].merge(
        features[["trace_id", "mean_absolute_timing_change_ns"]],
        on="trace_id",
        validate="one_to_one",
    )
    neg = merged[merged["activity_class"].isin(["no_victim", "local_only"])]
    max_negative_control = (
        float(neg["mean_absolute_timing_change_ns"].max()) if len(neg) else math.nan
    )
    add(
        "no_victim_and_local_only_are_remote_channel_negative_controls",
        max_negative_control <= 1e-9,
        "max mean absolute timing change <= 1e-9 ns",
        max_negative_control,
    )

    train_ids = set(split.loc[split["split"] == "train", "victim_instance_id"])
    test_ids = set(split.loc[split["split"] == "test", "victim_instance_id"])
    add(
        "grouped_split_has_no_instance_overlap",
        len(train_ids & test_ids) == 0,
        "empty train/test victim-instance intersection",
        sorted(train_ids & test_ids),
    )

    class_split_counts = (
        split.groupby(["activity_class", "split"]).size().unstack(fill_value=0)
    )
    all_classes_both = bool(
        (class_split_counts.get("train", 0) > 0).all()
        and (class_split_counts.get("test", 0) > 0).all()
    )
    add(
        "every_activity_class_present_in_train_and_test",
        all_classes_both,
        "positive count in both splits",
        class_split_counts.to_dict(),
    )

    expected_contexts = {x[0] for x in contexts}
    observed_contexts = set(labels["protocol_context"].unique())
    add(
        "all_requested_protocol_contexts_present",
        observed_contexts == expected_contexts,
        sorted(expected_contexts),
        sorted(observed_contexts),
    )

    protocol_defs = p27.build_protocols()
    normalized = all(
        math.isclose(
            protocol_defs[p.protocol_name].nominal_critical_latency_ns,
            150.0,
            abs_tol=1e-9,
        )
        for _, p, _ in contexts
    )
    add(
        "phase2_07_protocol_normalization_preserved",
        normalized,
        "150 ns nominal post-prerequisite critical latency",
        [
            (
                name,
                protocol_defs[p.protocol_name].nominal_critical_latency_ns,
            )
            for name, p, _ in contexts
        ],
    )

    return pd.DataFrame(rows)


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p27, phase2_source = load_phase2_07_module()
    contexts = select_contexts(p27, args.protocol)

    instances = make_victim_instances(
        seed=args.seed,
        instances_per_class=args.instances_per_class,
        observation_window_ns=args.observation_window_ns,
    )

    print(
        f"[Phase 3.1] contexts={len(contexts)}, "
        f"victim_instances={len(instances)}, "
        f"repeats={args.repeats_per_instance}"
    )
    print(f"[Phase 3.1] Reusing Phase-2.7 simulator: {phase2_source}")

    blackbox, labels, features, trials = run_dataset(
        p27,
        contexts=contexts,
        instances=instances,
        repeats_per_instance=args.repeats_per_instance,
        seed=args.seed,
        observation_window_ns=args.observation_window_ns,
        temporal_bins=args.temporal_bins,
        include_spectral=not args.no_spectral,
    )

    split = build_group_split(
        instances,
        seed=args.seed,
        test_size=args.test_size,
    )

    analysis = features.merge(
        labels,
        on=[
            "trace_id",
            "protocol_context",
            "victim_instance_id",
            "repeat_id",
        ],
        validate="one_to_one",
    )

    temporal_cols = [
        c for c in analysis.columns if c.startswith("temporal_bin_")
    ]

    (
        classification_metrics,
        classification_predictions,
        calibration,
        regression_metrics,
        regression_predictions,
    ) = evaluate_models(
        analysis,
        split,
        seed=args.seed,
        temporal_feature_columns=temporal_cols,
        feature_columns=SUMMARY_FEATURES,
    )

    max_probes = int(
        blackbox.groupby("trace_id")["probe_index"].count().max()
        if len(blackbox)
        else 0
    )
    budgets = sorted(
        {
            min(max_probes, int(x))
            for x in args.probe_budgets.split(",")
            if x.strip() and int(x) > 0
        }
    )
    budgets = [x for x in budgets if x > 0]
    if max_probes and max_probes not in budgets:
        budgets.append(max_probes)

    budget_metrics = rebuild_budget_features(
        blackbox,
        labels,
        split,
        budgets=budgets,
        temporal_bins=args.temporal_bins,
        include_spectral=not args.no_spectral,
        seed=args.seed,
    )

    slowdown_bins = slowdown_accuracy_bins(classification_predictions)

    validation = build_validations(
        p27=p27,
        blackbox=blackbox,
        labels=labels,
        features=features,
        split=split,
        contexts=contexts,
    )

    # -----------------------------------------------------------------
    # Helpful compact summaries
    # -----------------------------------------------------------------
    label_summary = (
        labels.groupby(["protocol_context", "activity_class"], sort=True)
        .agg(
            trace_count=("trace_id", "count"),
            mean_remote_operation_count=("remote_operation_count", "mean"),
            mean_remote_operation_rate_per_us=("remote_operation_rate_per_us", "mean"),
            mean_remote_activity_fraction=("remote_activity_fraction", "mean"),
            mean_total_endpoint_occupancy_ns=("total_endpoint_occupancy_ns", "mean"),
            mean_victim_makespan_slowdown=("victim_makespan_slowdown", "mean"),
        )
        .reset_index()
    )

    feature_summary = analysis.groupby(
        ["protocol_context", "activity_class"], sort=True
    ).agg(
        trace_count=("trace_id", "count"),
        mean_excess_latency_ns=("mean_excess_latency_ns", "mean"),
        mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
        delayed_probe_fraction=("delayed_probe_fraction", "mean"),
        cumulative_positive_excess_ns=("cumulative_positive_excess_ns", "mean"),
        maximum_excess_latency_ns=("maximum_excess_latency_ns", "mean"),
        longest_delayed_run=("longest_delayed_run", "mean"),
    ).reset_index()

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    blackbox.to_csv(output_dir / "phase3_01_attacker_visible_trace.csv", index=False)
    labels.to_csv(output_dir / "phase3_01_evaluator_ground_truth.csv", index=False)
    features.to_csv(output_dir / "phase3_01_trace_features.csv", index=False)
    trials.to_csv(output_dir / "phase3_01_trial_summary.csv", index=False)
    split.to_csv(output_dir / "phase3_01_group_split.csv", index=False)

    label_summary.to_csv(output_dir / "phase3_01_activity_class_summary.csv", index=False)
    feature_summary.to_csv(output_dir / "phase3_01_blackbox_feature_summary.csv", index=False)

    classification_metrics.to_csv(
        output_dir / "phase3_01_classification_metrics.csv",
        index=False,
    )
    classification_predictions.to_csv(
        output_dir / "phase3_01_classification_predictions.csv",
        index=False,
    )
    calibration.to_csv(
        output_dir / "phase3_01_binary_calibration_curve.csv",
        index=False,
    )
    regression_metrics.to_csv(
        output_dir / "phase3_01_regression_metrics.csv",
        index=False,
    )
    regression_predictions.to_csv(
        output_dir / "phase3_01_regression_predictions.csv",
        index=False,
    )
    budget_metrics.to_csv(
        output_dir / "phase3_01_probe_budget_metrics.csv",
        index=False,
    )
    slowdown_bins.to_csv(
        output_dir / "phase3_01_slowdown_accuracy_bins.csv",
        index=False,
    )
    validation.to_csv(
        output_dir / "phase3_01_validation_assertions.csv",
        index=False,
    )

    validation_summary = pd.DataFrame(
        [
            {
                "assertion_count": len(validation),
                "passed_assertions": int(validation["passed"].sum()),
                "failed_assertions": int((~validation["passed"]).sum()),
                "all_validations_passed": bool(validation["passed"].all()),
            }
        ]
    )
    validation_summary.to_csv(
        output_dir / "phase3_01_validation_summary.csv",
        index=False,
    )

    instance_table = pd.DataFrame([asdict(x) for x in instances])
    instance_table.to_csv(
        output_dir / "phase3_01_victim_instance_table.csv",
        index=False,
    )

    context_table = pd.DataFrame(
        [
            {
                "protocol_context": name,
                "protocol_name": protocol.protocol_name,
                "scenario_id": scenario.scenario_id,
                "shared_resources": "+".join(scenario.shared_resources),
                "nominal_critical_latency_ns": protocol.nominal_critical_latency_ns,
                "postcompletion_cleanup_ns": protocol.postcompletion_cleanup_ns,
                "uses_epr": protocol.uses_epr,
            }
            for name, protocol, scenario in contexts
        ]
    )
    context_table.to_csv(
        output_dir / "phase3_01_protocol_context_table.csv",
        index=False,
    )

    manifest = {
        "experiment": "Phase 3.1 — Activity and Communication-Intensity Inference",
        "output_directory": str(output_dir),
        "phase2_07_simulator_source": str(phase2_source),
        "seed": args.seed,
        "instances_per_class": args.instances_per_class,
        "activity_class_count": len(ACTIVITY_CLASSES),
        "victim_instance_count": len(instances),
        "repeats_per_instance": args.repeats_per_instance,
        "protocol_choice": args.protocol,
        "protocol_context_count": len(contexts),
        "trace_count": int(len(labels)),
        "observation_window_ns": args.observation_window_ns,
        "attacker_probe_period_ns": p27.ATTACKER_PERIOD_NS,
        "maximum_probe_count": max_probes,
        "probe_budgets": budgets,
        "temporal_bins": args.temporal_bins,
        "spectral_features_enabled": not args.no_spectral,
        "test_size": args.test_size,
        "group_split_unit": "victim_instance_id",
        "binary_task_definition": (
            "remote communication active vs no remote communication; "
            "no_victim and local_only are negative"
        ),
        "multi_class_labels": list(ACTIVITY_CLASSES),
        "continuous_targets": list(CONTINUOUS_TARGETS),
        "summary_feature_count": len(SUMMARY_FEATURES),
        "validation_assertion_count": len(validation),
        "passed_assertions": int(validation["passed"].sum()),
        "failed_assertions": int((~validation["passed"]).sum()),
        "all_validations_passed": bool(validation["passed"].all()),
        "attacker_visible_columns": list(ATTACKER_VISIBLE_COLUMNS),
    }
    (output_dir / "phase3_01_run_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\n[Phase 3.1] Classification metrics")
    print(
        classification_metrics[
            [
                "protocol_context",
                "task",
                "model",
                "roc_auc",
                "accuracy",
                "macro_f1",
            ]
        ].to_string(index=False)
    )

    print("\n[Phase 3.1] Remote-operation-count regression")
    print(
        regression_metrics[
            regression_metrics["target"] == "remote_operation_count"
        ][
            [
                "protocol_context",
                "model",
                "mae",
                "rmse",
                "r2",
            ]
        ].to_string(index=False)
    )

    print("\n[Phase 3.1] Validation")
    print(validation_summary.to_string(index=False))
    print(f"\n[Phase 3.1] Results saved to: {output_dir}")

    if args.fail_on_validation_error and not bool(validation["passed"].all()):
        failed = validation[~validation["passed"]]
        raise RuntimeError(
            "Phase 3.1 validation failed:\n"
            + failed.to_string(index=False)
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.1 — Activity and Communication-Intensity Inference"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output directory. Default: "
            "blackbox_window_results/phase3/phase3.1/"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--instances-per-class",
        type=int,
        default=DEFAULT_INSTANCES_PER_CLASS,
        help="Distinct victim circuit instances per activity class.",
    )
    parser.add_argument(
        "--repeats-per-instance",
        type=int,
        default=DEFAULT_REPEATS_PER_INSTANCE,
        help="Repeated timing realizations per victim circuit instance.",
    )
    parser.add_argument(
        "--observation-window-ns",
        type=float,
        default=DEFAULT_OBSERVATION_WINDOW_NS,
    )
    parser.add_argument(
        "--protocol",
        choices=("direct", "entangled", "both"),
        default="both",
        help=(
            "Fixed inference context(s). Models are evaluated separately per "
            "protocol even when 'both' is selected."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of victim circuit instances held out for testing.",
    )
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=DEFAULT_TEMPORAL_BINS,
    )
    parser.add_argument(
        "--probe-budgets",
        default="8,16,24,32,40",
        help=(
            "Comma-separated prefix probe budgets. The full trace length is "
            "automatically appended."
        ),
    )
    parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Disable FFT-derived attacker timing features.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.instances_per_class < 4:
        raise ValueError(
            "--instances-per-class must be at least 4 so every class can "
            "appear in both grouped train and test splits."
        )
    if args.repeats_per_instance < 1:
        raise ValueError("--repeats-per-instance must be >= 1")
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.5")
    if args.temporal_bins < 2:
        raise ValueError("--temporal-bins must be >= 2")

    run_experiment(args)


if __name__ == "__main__":
    main()
