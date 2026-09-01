#!/usr/bin/env python3
"""
Phase 3.2 — Execution-Phase Segmentation
========================================

Research question
-----------------
Can an attacker reconstruct WHEN a victim enters and leaves qualitatively
 different execution phases using only the timing/success behavior of the
attacker's own periodic remote-operation probes?

This experiment is the temporal-structure follow-on to Phase 3.1.  Phase 3.1
showed that aggregate communication activity and intensity are inferable.  In
Phase 3.2, every victim circuit instance contains several hidden execution
segments drawn from four logical phase classes:

    local_compute
    sparse_remote
    dense_remote
    synchronization_bursty

The order and durations of the phases are randomized independently for every
victim circuit instance.  This is important: a classifier cannot solve the
experiment simply by learning that, for example, the second quarter of every
trace is always "dense".  Absolute probe time / probe index is NOT an input to
any primary segmentation model.  A separate time-only diagnostic baseline is
reported so this possible confound is visible rather than hidden.

Architecture
------------
The script reuses the exact validated Phase-2.7 simulator:

    phase2_07_remote_protocol_comparison.py

and evaluates the same two fixed all-shared protocol contexts used in Phase
3.1:

    direct_coherent_all_shared
    entanglement_assisted_all_shared

The logical operation remains ``logical_remote_cx``.  Both protocols retain
the Phase-2.7 normalization of 150 ns nominal critical latency after the
protocol prerequisite is ready and 120 ns post-completion cleanup.

Black-box boundary
------------------
The attacker-facing trace contains ONLY the attacker's own:

* probe release time,
* completion / turnaround time,
* success/failure,
* paired attacker-only calibration,
* differential turnaround.

No victim release time, phase label, phase boundary, resource wait, blocking
owner, EPR state, stage identity, or evaluator attribution is exposed to the
segmentation models.

Per-probe inference features are causal.  For probe i they use only probe i
and a short history of earlier probes.  They do not use future probes,
absolute probe index, or absolute release time.  This lets the raw model be
interpreted as online phase inference.  A simple one-probe-island debounce is
also reported as an OPTIONAL offline cleanup of the predicted label sequence;
raw and debounced results are always stored separately.

Evaluation discipline
---------------------
* Train/test split is grouped by victim circuit instance.
* All repeats of one victim instance, under both protocols, remain in the same
  split.
* Models are trained separately per protocol context.
* Per-probe accuracy, balanced accuracy, macro-F1, and macro-IoU are reported.
* Exact phase-transition boundary recall/precision and timing error are
  reported.
* Remote-vs-local onset/offset boundary metrics are reported separately.
* A time-only baseline and majority baseline are reported as diagnostics.

Outputs
-------
phase3_02_attacker_visible_trace.csv
phase3_02_probe_features.csv
phase3_02_evaluator_probe_ground_truth.csv
phase3_02_victim_phase_schedule.csv
phase3_02_victim_release_schedule.csv
phase3_02_victim_instance_table.csv
phase3_02_group_split.csv
phase3_02_segmentation_metrics.csv
phase3_02_segmentation_predictions.csv
phase3_02_confusion_matrix.csv
phase3_02_boundary_metrics.csv
phase3_02_boundary_predictions.csv
phase3_02_phase_class_summary.csv
phase3_02_protocol_comparison_summary.csv
phase3_02_validation_assertions.csv
phase3_02_validation_summary.csv
phase3_02_run_manifest.json

Default run
-----------
Place this script next to ``phase2_07_remote_protocol_comparison.py`` and run:

    python phase3_02_execution_phase_segmentation.py

Fast smoke test:

    python phase3_02_execution_phase_segmentation.py \
        --instances 8 \
        --repeats-per-instance 1 \
        --observation-window-ns 8000 \
        --rf-trees 120 \
        --output-dir /tmp/phase3_02_smoke

Evidence boundary
-----------------
All timing values and resource semantics are controlled architectural
simulation parameters.  The experiment tests whether the validated modeled
channel contains temporal execution-structure information; it is not a claim
about timestamp resolution or stage durations of a specific commercial
machine.
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
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Global settings
# =============================================================================

DEFAULT_SEED = 3201
DEFAULT_INSTANCES = 36
DEFAULT_REPEATS_PER_INSTANCE = 2
DEFAULT_OBSERVATION_WINDOW_NS = 20_000.0
DEFAULT_TEST_SIZE = 0.30
DEFAULT_CAUSAL_WINDOW_PROBES = 5
DEFAULT_LAG_COUNT = 4
DEFAULT_RF_TREES = 500
DEFAULT_BOUNDARY_TOLERANCE_PROBES = 2.0
DEFAULT_MIN_SEGMENT_NS = 2_200.0

DEFAULT_OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase3"
    / "phase3.2"
)

PHASE_LABELS = (
    "local_compute",
    "sparse_remote",
    "dense_remote",
    "synchronization_bursty",
)
REMOTE_PHASES = {
    "sparse_remote",
    "dense_remote",
    "synchronization_bursty",
}

AFFECTED_THRESHOLD_NS = 1e-9
EPS = 1e-12

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

# Metadata retained in the feature table for joins/grouping.  These are NOT
# passed to the primary classifiers.
FEATURE_METADATA_COLUMNS = (
    "trace_id",
    "protocol_context",
    "victim_instance_id",
    "repeat_id",
    "probe_index",
)

# Filled after feature construction.  Kept explicit so validation can assert
# that time/index and evaluator labels never enter a primary model.
MODEL_FEATURE_COLUMNS: list[str] = []


# =============================================================================
# Phase-2.7 loading
# =============================================================================


def load_phase2_07_module():
    """Load the validated Phase-2.7 simulator from a nearby path."""
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
            "Phase 3.2 intentionally reuses the validated Phase-2.7 simulator.\n"
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
# Victim schedule model
# =============================================================================


@dataclass(frozen=True)
class PhaseSegment:
    victim_instance_id: str
    phase_index: int
    phase_label: str
    start_ns: float
    end_ns: float

    @property
    def duration_ns(self) -> float:
        return float(self.end_ns - self.start_ns)

    @property
    def remote_active(self) -> int:
        return int(self.phase_label in REMOTE_PHASES)


@dataclass(frozen=True)
class VictimInstance:
    victim_instance_id: str
    instance_index: int
    phase_count: int
    schedule_signature: str


def stable_seed(*parts: Any, modulus: int = 2**32 - 1) -> int:
    token = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % modulus


def _valid_phase_order(labels: Sequence[str]) -> bool:
    return all(labels[i] != labels[i - 1] for i in range(1, len(labels)))


def make_phase_order(rng: np.random.Generator, phase_count: int) -> list[str]:
    """
    Every instance contains every phase class at least once.  Additional phase
    labels are randomized.  Adjacent identical phases are rejected so every
    schedule boundary is a true semantic transition.
    """
    if phase_count < len(PHASE_LABELS):
        raise ValueError("phase_count must be >= number of phase labels")

    base = list(PHASE_LABELS)
    extras = [
        str(rng.choice(PHASE_LABELS))
        for _ in range(phase_count - len(PHASE_LABELS))
    ]
    labels = base + extras

    for _ in range(10_000):
        rng.shuffle(labels)
        if _valid_phase_order(labels):
            return list(labels)

    # Extremely unlikely fallback: greedily choose a non-equal next label.
    remaining = list(labels)
    out: list[str] = []
    while remaining:
        candidates = [x for x in remaining if not out or x != out[-1]]
        if not candidates:
            raise RuntimeError("Could not construct non-adjacent phase order")
        choice = str(rng.choice(candidates))
        out.append(choice)
        remaining.remove(choice)
    return out


def make_instances_and_schedules(
    *,
    seed: int,
    instance_count: int,
    observation_window_ns: float,
    min_segment_ns: float,
) -> tuple[list[VictimInstance], pd.DataFrame]:
    instances: list[VictimInstance] = []
    phase_rows: list[dict[str, Any]] = []

    for idx in range(instance_count):
        rng = np.random.default_rng(stable_seed(seed, idx, "phase_schedule"))
        phase_count = int(rng.integers(5, 7))  # five or six segments
        labels = make_phase_order(rng, phase_count)

        # Keep every phase long enough to contain multiple probes.  For shorter
        # smoke-test windows, automatically scale the lower bound down instead
        # of making the requested window impossible.
        feasible_min = min(
            float(min_segment_ns),
            0.55 * observation_window_ns / phase_count,
        )
        feasible_min = max(feasible_min, 2.0)

        remaining = observation_window_ns - feasible_min * phase_count
        if remaining < -1e-9:
            raise ValueError("Observation window too short for requested phase schedule")
        remaining = max(0.0, remaining)

        weights = rng.dirichlet(np.full(phase_count, 2.0))
        durations = feasible_min + remaining * weights
        # Exact final coverage despite floating point.
        durations[-1] += observation_window_ns - float(np.sum(durations))

        instance_id = f"victim_{idx:04d}"
        signature = hashlib.sha256(
            (
                instance_id
                + "|"
                + "|".join(labels)
                + "|"
                + "|".join(f"{x:.6f}" for x in durations)
            ).encode()
        ).hexdigest()[:16]

        instances.append(
            VictimInstance(
                victim_instance_id=instance_id,
                instance_index=idx,
                phase_count=phase_count,
                schedule_signature=signature,
            )
        )

        t = 0.0
        for phase_idx, (label, duration) in enumerate(zip(labels, durations)):
            start = float(t)
            end = (
                float(observation_window_ns)
                if phase_idx == phase_count - 1
                else float(t + duration)
            )
            phase_rows.append(
                {
                    "victim_instance_id": instance_id,
                    "phase_index": phase_idx,
                    "phase_label": label,
                    "remote_active": int(label in REMOTE_PHASES),
                    "start_ns": start,
                    "end_ns": end,
                    "duration_ns": end - start,
                    "schedule_signature": signature,
                }
            )
            t = end

    schedule = pd.DataFrame(phase_rows)
    return instances, schedule


def schedule_for_instance(schedule: pd.DataFrame, instance_id: str) -> pd.DataFrame:
    return (
        schedule[schedule["victim_instance_id"] == instance_id]
        .sort_values("phase_index")
        .reset_index(drop=True)
    )


def phase_at_time(schedule_rows: pd.DataFrame, time_ns: float) -> tuple[int, str, float, float]:
    rows = schedule_rows
    mask = (rows["start_ns"] <= time_ns) & (time_ns < rows["end_ns"])
    if mask.any():
        row = rows.loc[mask].iloc[0]
    else:
        # Last probe can only miss due to sub-ns floating point at the upper edge.
        row = rows.iloc[-1]
    return (
        int(row["phase_index"]),
        str(row["phase_label"]),
        float(row["start_ns"]),
        float(row["end_ns"]),
    )


def generate_remote_releases_for_instance(
    schedule_rows: pd.DataFrame,
    *,
    seed: int,
    victim_instance_id: str,
    repeat_id: int,
    observation_window_ns: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Convert the logical phase schedule into victim logical remote-CX release
    times.  Repeats retain the same phase boundaries but receive small timing
    jitter and phase-local rate variation.
    """
    release_rows: list[dict[str, Any]] = []

    for row in schedule_rows.itertuples(index=False):
        label = str(row.phase_label)
        start = float(row.start_ns)
        end = float(row.end_ns)
        duration = end - start

        if label == "local_compute":
            continue

        rng = np.random.default_rng(
            stable_seed(seed, victim_instance_id, repeat_id, int(row.phase_index), "releases")
        )

        # Ensure the first release is well inside the segment and that even a
        # short smoke-test segment receives at least one remote operation.
        first_offset = min(float(rng.uniform(90.0, 260.0)), max(20.0, 0.22 * duration))
        first = start + first_offset

        times: list[float] = []
        if label == "sparse_remote":
            interval = float(rng.uniform(950.0, 1250.0))
            times = list(np.arange(first, end, interval, dtype=float))
        elif label == "dense_remote":
            interval = float(rng.uniform(400.0, 520.0))
            times = list(np.arange(first, end, interval, dtype=float))
        elif label == "synchronization_bursty":
            burst_period = float(rng.uniform(1150.0, 1550.0))
            burst_size = int(rng.integers(3, 6))
            burst_spacing = float(rng.uniform(60.0, 95.0))
            base = first
            while base < end:
                for j in range(burst_size):
                    t = base + j * burst_spacing
                    if t < end:
                        times.append(float(t))
                base += burst_period
        else:
            raise ValueError(label)

        if not times and first < end:
            times = [first]

        # Tiny repeat-level jitter; clip into the same logical phase so ground
        # truth is not accidentally changed by the timing realization.
        for local_idx, t in enumerate(times):
            jitter = float(rng.uniform(-4.0, 4.0))
            t2 = min(max(t + jitter, start + 1e-6), end - 1e-6)
            if 0.0 <= t2 < observation_window_ns:
                release_rows.append(
                    {
                        "victim_instance_id": victim_instance_id,
                        "repeat_id": repeat_id,
                        "phase_index": int(row.phase_index),
                        "phase_label": label,
                        "phase_local_request_index": local_idx,
                        "release_ns": float(t2),
                    }
                )

    release_df = pd.DataFrame(release_rows)
    if release_df.empty:
        return np.array([], dtype=float), release_df

    release_df = release_df.sort_values("release_ns").reset_index(drop=True)
    release_df["victim_request_index"] = np.arange(len(release_df), dtype=int)
    return release_df["release_ns"].to_numpy(dtype=float), release_df


# =============================================================================
# Phase-2.7 protocol contexts and request construction
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


def make_request_specs(
    p27,
    *,
    tenant: str,
    releases: np.ndarray,
    victim_instance_id: str,
    repeat_id: int,
) -> list[Any]:
    return [
        p27.RequestSpec(
            request_id=f"{tenant}::{victim_instance_id}::repeat_{repeat_id:02d}::{i}",
            tenant=tenant,
            ready_ns=float(t),
            request_index=i,
            workload_name="phase_segmentation",
            trial_id=repeat_id,
        )
        for i, t in enumerate(releases)
    ]


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


# =============================================================================
# Causal per-probe feature extraction
# =============================================================================


def causal_probe_feature_frame(
    trace: pd.DataFrame,
    *,
    victim_instance_id: str,
    repeat_id: int,
    causal_window_probes: int,
    lag_count: int,
) -> pd.DataFrame:
    """
    Build one feature vector per attacker probe using only current/past probes.

    IMPORTANT: probe_index/release_ns are retained only as join metadata and
    are never placed in MODEL_FEATURE_COLUMNS.
    """
    t = trace.sort_values("probe_index").reset_index(drop=True)
    excess = t["excess_turnaround_ns"].to_numpy(dtype=float)
    delayed = t["delayed"].to_numpy(dtype=bool)
    speedup = t["speedup"].to_numpy(dtype=bool)
    failure = t["failure_transition"].to_numpy(dtype=bool)

    rows: list[dict[str, Any]] = []
    for i in range(len(t)):
        begin = max(0, i - causal_window_probes + 1)
        w = excess[begin : i + 1]
        wd = delayed[begin : i + 1]
        ws = speedup[begin : i + 1]
        wf = failure[begin : i + 1]

        xidx = np.arange(len(w), dtype=float)
        if len(w) >= 2 and np.std(xidx) > EPS:
            slope = float(np.polyfit(xidx, w, 1)[0])
        else:
            slope = 0.0

        previous_delays = np.flatnonzero(delayed[: i + 1])
        if len(previous_delays):
            since_delay = float(i - previous_delays[-1])
        else:
            since_delay = float(causal_window_probes + 1)

        row: dict[str, Any] = {
            "trace_id": str(t.loc[i, "trace_id"]),
            "protocol_context": str(t.loc[i, "protocol_context"]),
            "victim_instance_id": victim_instance_id,
            "repeat_id": int(repeat_id),
            "probe_index": int(t.loc[i, "probe_index"]),
            # Primary black-box model inputs start here.
            "current_excess_ns": float(excess[i]),
            "current_abs_excess_ns": float(abs(excess[i])),
            "current_delayed": float(delayed[i]),
            "current_speedup": float(speedup[i]),
            "current_failure_transition": float(failure[i]),
            "delta_excess_1_ns": float(excess[i] - excess[i - 1]) if i >= 1 else 0.0,
            "rolling_mean_excess_ns": float(np.mean(w)),
            "rolling_mean_abs_excess_ns": float(np.mean(np.abs(w))),
            "rolling_std_excess_ns": float(np.std(w)),
            "rolling_max_excess_ns": float(np.max(w)),
            "rolling_min_excess_ns": float(np.min(w)),
            "rolling_max_abs_excess_ns": float(np.max(np.abs(w))),
            "rolling_delayed_fraction": float(np.mean(wd)),
            "rolling_speedup_fraction": float(np.mean(ws)),
            "rolling_failure_fraction": float(np.mean(wf)),
            "rolling_positive_sum_ns": float(np.sum(np.maximum(w, 0.0))),
            "rolling_abs_sum_ns": float(np.sum(np.abs(w))),
            "rolling_excess_slope_ns_per_probe": slope,
            "rolling_nonzero_fraction": float(np.mean(np.abs(w) > AFFECTED_THRESHOLD_NS)),
            "probes_since_last_delay": since_delay,
        }

        for lag in range(1, lag_count + 1):
            if i >= lag:
                row[f"lag{lag}_excess_ns"] = float(excess[i - lag])
                row[f"lag{lag}_delayed"] = float(delayed[i - lag])
                row[f"lag{lag}_failure"] = float(failure[i - lag])
            else:
                row[f"lag{lag}_excess_ns"] = 0.0
                row[f"lag{lag}_delayed"] = 0.0
                row[f"lag{lag}_failure"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def model_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = set(FEATURE_METADATA_COLUMNS)
    return [c for c in feature_df.columns if c not in excluded]


# =============================================================================
# Dataset generation
# =============================================================================


def build_probe_ground_truth(
    trace: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    *,
    victim_instance_id: str,
    repeat_id: int,
) -> pd.DataFrame:
    boundaries = schedule_rows["start_ns"].to_numpy(dtype=float)[1:]
    out: list[dict[str, Any]] = []

    for row in trace.itertuples(index=False):
        phase_index, label, start, end = phase_at_time(schedule_rows, float(row.release_ns))
        if len(boundaries):
            distance = float(np.min(np.abs(boundaries - float(row.release_ns))))
        else:
            distance = math.nan
        out.append(
            {
                "trace_id": row.trace_id,
                "protocol_context": row.protocol_context,
                "victim_instance_id": victim_instance_id,
                "repeat_id": repeat_id,
                "probe_index": int(row.probe_index),
                "release_ns": float(row.release_ns),
                "phase_index": phase_index,
                "phase_label": label,
                "remote_active_label": int(label in REMOTE_PHASES),
                "phase_start_ns": start,
                "phase_end_ns": end,
                "distance_to_nearest_true_boundary_ns": distance,
            }
        )
    return pd.DataFrame(out)


def run_dataset(
    p27,
    *,
    contexts,
    instances: list[VictimInstance],
    schedule: pd.DataFrame,
    repeats_per_instance: int,
    seed: int,
    observation_window_ns: float,
    causal_window_probes: int,
    lag_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blackbox_parts: list[pd.DataFrame] = []
    gt_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    release_schedule_parts: list[pd.DataFrame] = []

    # Release schedules are architecture-independent; generate each once and
    # use exactly the same logical victim timing under both protocols.
    release_cache: dict[tuple[str, int], tuple[np.ndarray, pd.DataFrame]] = {}
    for instance in instances:
        inst_sched = schedule_for_instance(schedule, instance.victim_instance_id)
        for repeat_id in range(repeats_per_instance):
            releases, release_df = generate_remote_releases_for_instance(
                inst_sched,
                seed=seed,
                victim_instance_id=instance.victim_instance_id,
                repeat_id=repeat_id,
                observation_window_ns=observation_window_ns,
            )
            release_cache[(instance.victim_instance_id, repeat_id)] = (releases, release_df)
            if not release_df.empty:
                release_schedule_parts.append(release_df)

    total = len(contexts) * len(instances) * repeats_per_instance
    done = 0

    # Attacker-only timing is independent of the victim instance.  Cache once
    # per protocol/repeat to avoid unnecessary simulation.
    attacker_only_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for protocol_context, protocol, scenario in contexts:
        for repeat_id in range(repeats_per_instance):
            a_specs = attacker_probe_specs(
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
                list(a_specs),
            )
            attacker_only_cache[(protocol_context, repeat_id)] = attacker_only

        for instance in instances:
            inst_sched = schedule_for_instance(schedule, instance.victim_instance_id)
            for repeat_id in range(repeats_per_instance):
                a_specs = attacker_probe_specs(
                    p27,
                    repeat_id=repeat_id,
                    observation_window_ns=observation_window_ns,
                )
                releases, _release_df = release_cache[(instance.victim_instance_id, repeat_id)]
                v_specs = make_request_specs(
                    p27,
                    tenant="victim",
                    releases=releases,
                    victim_instance_id=instance.victim_instance_id,
                    repeat_id=repeat_id,
                )

                combined_specs = sorted(
                    list(a_specs) + list(v_specs),
                    key=lambda x: (x.ready_ns, x.tenant, x.request_index),
                )
                combined, *_ = p27.run_one(
                    protocol,
                    scenario,
                    "phase_segmentation",
                    repeat_id,
                    "combined",
                    combined_specs,
                )

                trace_id = hashlib.sha256(
                    (
                        f"{protocol_context}|{instance.victim_instance_id}|"
                        f"{repeat_id}|{seed}|phase3.2"
                    ).encode()
                ).hexdigest()[:20]

                trace = pair_attacker_blackbox(
                    attacker_only_cache[(protocol_context, repeat_id)],
                    combined,
                    trace_id=trace_id,
                    protocol_context=protocol_context,
                )
                blackbox_parts.append(trace)

                gt = build_probe_ground_truth(
                    trace,
                    inst_sched,
                    victim_instance_id=instance.victim_instance_id,
                    repeat_id=repeat_id,
                )
                gt_parts.append(gt)

                feat = causal_probe_feature_frame(
                    trace,
                    victim_instance_id=instance.victim_instance_id,
                    repeat_id=repeat_id,
                    causal_window_probes=causal_window_probes,
                    lag_count=lag_count,
                )
                feature_parts.append(feat)

                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print(f"[Phase 3.2] Generated {done}/{total} traces")

    blackbox = pd.concat(blackbox_parts, ignore_index=True)
    ground_truth = pd.concat(gt_parts, ignore_index=True)
    features = pd.concat(feature_parts, ignore_index=True)
    release_schedule = (
        pd.concat(release_schedule_parts, ignore_index=True)
        if release_schedule_parts
        else pd.DataFrame(
            columns=[
                "victim_instance_id",
                "repeat_id",
                "phase_index",
                "phase_label",
                "phase_local_request_index",
                "release_ns",
                "victim_request_index",
            ]
        )
    )
    return blackbox, ground_truth, features, release_schedule


# =============================================================================
# Grouped split
# =============================================================================


def build_group_split(
    instances: list[VictimInstance],
    *,
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    ids = np.array([x.victim_instance_id for x in instances], dtype=object)
    rng = np.random.default_rng(stable_seed(seed, "group_split"))
    perm = rng.permutation(ids)
    n_test = max(1, int(round(len(ids) * test_size)))
    n_test = min(n_test, len(ids) - 1)
    test_ids = set(perm[:n_test].tolist())
    rows = []
    for instance in instances:
        rows.append(
            {
                "victim_instance_id": instance.victim_instance_id,
                "split": "test" if instance.victim_instance_id in test_ids else "train",
            }
        )
    return pd.DataFrame(rows).sort_values("victim_instance_id").reset_index(drop=True)


# =============================================================================
# Models and segmentation evaluation
# =============================================================================


def primary_models(seed: int, rf_trees: int):
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
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.07,
            l2_regularization=0.20,
            random_state=seed,
        ),
    }


def sample_weights_balanced(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    n = len(y)
    k = max(len(counts), 1)
    return np.asarray([n / (k * counts[v]) for v in y], dtype=float)


def debounce_one_probe_islands(labels: Sequence[str]) -> list[str]:
    """
    Offline diagnostic cleanup: replace A,B,A one-probe islands with A.
    Iterate until stable.  This uses one future label, so it is NEVER presented
    as the online/raw model result.
    """
    out = list(labels)
    changed = True
    while changed and len(out) >= 3:
        changed = False
        new = list(out)
        for i in range(1, len(out) - 1):
            if out[i - 1] == out[i + 1] and out[i] != out[i - 1]:
                new[i] = out[i - 1]
                changed = True
        out = new
    return out


def macro_iou(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(
        jaccard_score(
            y_true,
            y_pred,
            labels=list(PHASE_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def segmentation_metric_row(
    *,
    protocol_context: str,
    model_name: str,
    model_variant: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    true_remote = np.asarray([x in REMOTE_PHASES for x in yt], dtype=int)
    pred_remote = np.asarray([x in REMOTE_PHASES for x in yp], dtype=int)

    return {
        "protocol_context": protocol_context,
        "model_name": model_name,
        "model_variant": model_variant,
        "sample_count": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=list(PHASE_LABELS), average="macro", zero_division=0)),
        "macro_iou": macro_iou(yt, yp),
        "remote_vs_local_accuracy": float(accuracy_score(true_remote, pred_remote)),
        "remote_vs_local_f1": float(f1_score(true_remote, pred_remote, zero_division=0)),
    }


def add_prediction_variants(pred: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    raw = pred.copy()
    raw["model_variant"] = "raw_online"
    parts.append(raw)

    debounced_parts: list[pd.DataFrame] = []
    for _trace_id, group in pred.groupby("trace_id", sort=False):
        g = group.sort_values("probe_index").copy()
        g["predicted_phase_label"] = debounce_one_probe_islands(
            g["predicted_phase_label"].astype(str).tolist()
        )
        g["model_variant"] = "debounced_offline"
        debounced_parts.append(g)
    parts.append(pd.concat(debounced_parts, ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def fit_and_predict_models(
    analysis: pd.DataFrame,
    split: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int,
    rf_trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = analysis.merge(split, on="victim_instance_id", validate="many_to_one")
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for protocol_context, pdat in merged.groupby("protocol_context", sort=True):
        train = pdat[pdat["split"] == "train"].copy()
        test = pdat[pdat["split"] == "test"].copy()
        X_train = train[feature_columns].to_numpy(dtype=float)
        X_test = test[feature_columns].to_numpy(dtype=float)
        y_train = train["phase_label"].astype(str)
        y_test = test["phase_label"].astype(str)

        for model_name, model in primary_models(seed, rf_trees).items():
            if model_name == "hist_gradient_boosting":
                model.fit(X_train, y_train, sample_weight=sample_weights_balanced(y_train))
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test).astype(str)

            base = test[
                [
                    "trace_id",
                    "protocol_context",
                    "victim_instance_id",
                    "repeat_id",
                    "probe_index",
                    "release_ns",
                    "phase_label",
                    "remote_active_label",
                ]
            ].copy()
            base = base.rename(columns={"phase_label": "true_phase_label"})
            base["predicted_phase_label"] = y_pred
            base["model_name"] = model_name
            variants = add_prediction_variants(base)
            prediction_parts.append(variants)

            for variant_name, vdat in variants.groupby("model_variant", sort=True):
                metric_rows.append(
                    segmentation_metric_row(
                        protocol_context=protocol_context,
                        model_name=model_name,
                        model_variant=variant_name,
                        y_true=vdat["true_phase_label"].astype(str),
                        y_pred=vdat["predicted_phase_label"].astype(str),
                    )
                )

        # -------------------------------------------------------------
        # Majority baseline: does not use timing.
        # -------------------------------------------------------------
        majority = str(y_train.value_counts().idxmax())
        majority_pred = np.full(len(test), majority, dtype=object)
        metric_rows.append(
            segmentation_metric_row(
                protocol_context=protocol_context,
                model_name="majority_baseline",
                model_variant="raw_online",
                y_true=y_test,
                y_pred=majority_pred,
            )
        )

        # -------------------------------------------------------------
        # Time-only diagnostic baseline.  This intentionally uses probe index
        # to verify that randomized phase order/duration does not itself encode
        # the answer.  It is NOT one of the primary attack models.
        # -------------------------------------------------------------
        max_probe = max(float(merged["probe_index"].max()), 1.0)
        Xt_train = (train[["probe_index"]].to_numpy(dtype=float) / max_probe)
        Xt_test = (test[["probe_index"]].to_numpy(dtype=float) / max_probe)
        time_model = RandomForestClassifier(
            n_estimators=min(rf_trees, 300),
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed + 77,
            n_jobs=-1,
        )
        time_model.fit(Xt_train, y_train)
        time_pred = time_model.predict(Xt_test).astype(str)
        metric_rows.append(
            segmentation_metric_row(
                protocol_context=protocol_context,
                model_name="time_only_diagnostic",
                model_variant="raw_online",
                y_true=y_test,
                y_pred=time_pred,
            )
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    return metrics, predictions


# =============================================================================
# Confusion matrices
# =============================================================================


def build_confusion_output(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = list(PHASE_LABELS)
    for keys, dat in predictions.groupby(
        ["protocol_context", "model_name", "model_variant"],
        sort=True,
    ):
        protocol_context, model_name, model_variant = keys
        cm = confusion_matrix(
            dat["true_phase_label"],
            dat["predicted_phase_label"],
            labels=labels,
        )
        for i, true_label in enumerate(labels):
            denom = int(np.sum(cm[i]))
            for j, pred_label in enumerate(labels):
                count = int(cm[i, j])
                rows.append(
                    {
                        "protocol_context": protocol_context,
                        "model_name": model_name,
                        "model_variant": model_variant,
                        "true_phase_label": true_label,
                        "predicted_phase_label": pred_label,
                        "count": count,
                        "true_normalized_fraction": count / denom if denom else 0.0,
                    }
                )
    return pd.DataFrame(rows)


# =============================================================================
# Boundary extraction and evaluation
# =============================================================================


def predicted_boundaries_from_trace(dat: pd.DataFrame) -> list[dict[str, Any]]:
    g = dat.sort_values("probe_index").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(g)):
        a = str(g.loc[i - 1, "predicted_phase_label"])
        b = str(g.loc[i, "predicted_phase_label"])
        if a == b:
            continue
        time_ns = 0.5 * (float(g.loc[i - 1, "release_ns"]) + float(g.loc[i, "release_ns"]))
        rows.append(
            {
                "predicted_boundary_ns": time_ns,
                "predicted_from_label": a,
                "predicted_to_label": b,
            }
        )
    return rows


def true_boundaries_for_instance(schedule_rows: pd.DataFrame) -> list[dict[str, Any]]:
    s = schedule_rows.sort_values("phase_index").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(s)):
        rows.append(
            {
                "true_boundary_ns": float(s.loc[i, "start_ns"]),
                "true_from_label": str(s.loc[i - 1, "phase_label"]),
                "true_to_label": str(s.loc[i, "phase_label"]),
            }
        )
    return rows


def is_remote_label(label: str) -> bool:
    return label in REMOTE_PHASES


def _greedy_boundary_match(
    true_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    tolerance_ns: float,
    exact_transition_type: bool,
    binary_remote_only: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Greedy nearest one-to-one matching is sufficient for these ordered, sparse boundaries."""
    used: set[int] = set()
    match_rows: list[dict[str, Any]] = []

    for ti, tr in enumerate(true_rows):
        true_from = str(tr["true_from_label"])
        true_to = str(tr["true_to_label"])

        if binary_remote_only and is_remote_label(true_from) == is_remote_label(true_to):
            continue

        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for pi, pr in enumerate(pred_rows):
            if pi in used:
                continue
            pred_from = str(pr["predicted_from_label"])
            pred_to = str(pr["predicted_to_label"])

            if binary_remote_only:
                if is_remote_label(pred_from) == is_remote_label(pred_to):
                    continue
                if (
                    is_remote_label(pred_from) != is_remote_label(true_from)
                    or is_remote_label(pred_to) != is_remote_label(true_to)
                ):
                    continue
            elif exact_transition_type:
                if pred_from != true_from or pred_to != true_to:
                    continue

            error = float(pr["predicted_boundary_ns"] - tr["true_boundary_ns"])
            if abs(error) <= tolerance_ns:
                candidates.append((abs(error), pi, pr))

        if candidates:
            _, pi, pr = min(candidates, key=lambda x: x[0])
            used.add(pi)
            signed_error = float(pr["predicted_boundary_ns"] - tr["true_boundary_ns"])
            match_rows.append(
                {
                    "true_boundary_index": ti,
                    **tr,
                    **pr,
                    "matched": True,
                    "signed_error_ns": signed_error,
                    "absolute_error_ns": abs(signed_error),
                }
            )
        else:
            match_rows.append(
                {
                    "true_boundary_index": ti,
                    **tr,
                    "predicted_boundary_ns": math.nan,
                    "predicted_from_label": "",
                    "predicted_to_label": "",
                    "matched": False,
                    "signed_error_ns": math.nan,
                    "absolute_error_ns": math.nan,
                }
            )

    return match_rows, len(used)


def evaluate_boundaries(
    predictions: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    tolerance_ns: float,
    attacker_period_ns: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    group_cols = ["protocol_context", "model_name", "model_variant", "trace_id"]
    for keys, dat in predictions.groupby(group_cols, sort=True):
        protocol_context, model_name, model_variant, trace_id = keys
        instance_id = str(dat["victim_instance_id"].iloc[0])
        repeat_id = int(dat["repeat_id"].iloc[0])
        s = schedule_for_instance(schedule, instance_id)
        true_rows = true_boundaries_for_instance(s)
        pred_rows = predicted_boundaries_from_trace(dat)

        for boundary_mode, exact, binary in [
            ("phase_exact", True, False),
            ("phase_any", False, False),
            ("remote_on_off", False, True),
        ]:
            matches, matched_pred_count = _greedy_boundary_match(
                true_rows,
                pred_rows,
                tolerance_ns=tolerance_ns,
                exact_transition_type=exact,
                binary_remote_only=binary,
            )

            if binary:
                true_count = sum(
                    is_remote_label(str(x["true_from_label"]))
                    != is_remote_label(str(x["true_to_label"]))
                    for x in true_rows
                )
                pred_count = sum(
                    is_remote_label(str(x["predicted_from_label"]))
                    != is_remote_label(str(x["predicted_to_label"]))
                    for x in pred_rows
                )
            else:
                true_count = len(true_rows)
                pred_count = len(pred_rows)

            matched = [x for x in matches if bool(x["matched"])]
            errors = np.asarray([x["absolute_error_ns"] for x in matched], dtype=float)
            signed = np.asarray([x["signed_error_ns"] for x in matched], dtype=float)

            recall = len(matched) / true_count if true_count else 1.0
            precision = matched_pred_count / pred_count if pred_count else (1.0 if true_count == 0 else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            metric_rows.append(
                {
                    "protocol_context": protocol_context,
                    "model_name": model_name,
                    "model_variant": model_variant,
                    "trace_id": trace_id,
                    "victim_instance_id": instance_id,
                    "repeat_id": repeat_id,
                    "boundary_mode": boundary_mode,
                    "true_boundary_count": true_count,
                    "predicted_boundary_count": pred_count,
                    "matched_boundary_count": len(matched),
                    "boundary_recall_within_tolerance": recall,
                    "boundary_precision_within_tolerance": precision,
                    "boundary_f1_within_tolerance": f1,
                    "boundary_mae_ns": float(np.mean(errors)) if len(errors) else math.nan,
                    "boundary_median_absolute_error_ns": float(np.median(errors)) if len(errors) else math.nan,
                    "boundary_mean_signed_error_ns": float(np.mean(signed)) if len(signed) else math.nan,
                    "fraction_within_one_probe": float(np.mean(errors <= attacker_period_ns)) if len(errors) else 0.0,
                    "fraction_within_two_probes": float(np.mean(errors <= 2.0 * attacker_period_ns)) if len(errors) else 0.0,
                    "tolerance_ns": tolerance_ns,
                }
            )

            for mr in matches:
                detail_rows.append(
                    {
                        "protocol_context": protocol_context,
                        "model_name": model_name,
                        "model_variant": model_variant,
                        "trace_id": trace_id,
                        "victim_instance_id": instance_id,
                        "repeat_id": repeat_id,
                        "boundary_mode": boundary_mode,
                        **mr,
                    }
                )

    per_trace_metrics = pd.DataFrame(metric_rows)
    details = pd.DataFrame(detail_rows)

    # Add aggregate rows over test traces, weighted by boundaries for recall/
    # precision but with timing errors averaged over matched boundaries.
    aggregate_rows: list[dict[str, Any]] = []
    for keys, dat in per_trace_metrics.groupby(
        ["protocol_context", "model_name", "model_variant", "boundary_mode"],
        sort=True,
    ):
        protocol_context, model_name, model_variant, boundary_mode = keys
        true_count = int(dat["true_boundary_count"].sum())
        pred_count = int(dat["predicted_boundary_count"].sum())
        matched_count = int(dat["matched_boundary_count"].sum())
        recall = matched_count / true_count if true_count else 1.0
        precision = matched_count / pred_count if pred_count else (1.0 if true_count == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        d = details[
            (details["protocol_context"] == protocol_context)
            & (details["model_name"] == model_name)
            & (details["model_variant"] == model_variant)
            & (details["boundary_mode"] == boundary_mode)
            & (details["matched"] == True)  # noqa: E712
        ]
        errors = d["absolute_error_ns"].dropna().to_numpy(dtype=float)
        signed = d["signed_error_ns"].dropna().to_numpy(dtype=float)

        aggregate_rows.append(
            {
                "protocol_context": protocol_context,
                "model_name": model_name,
                "model_variant": model_variant,
                "trace_id": "__aggregate__",
                "victim_instance_id": "__aggregate__",
                "repeat_id": -1,
                "boundary_mode": boundary_mode,
                "true_boundary_count": true_count,
                "predicted_boundary_count": pred_count,
                "matched_boundary_count": matched_count,
                "boundary_recall_within_tolerance": recall,
                "boundary_precision_within_tolerance": precision,
                "boundary_f1_within_tolerance": f1,
                "boundary_mae_ns": float(np.mean(errors)) if len(errors) else math.nan,
                "boundary_median_absolute_error_ns": float(np.median(errors)) if len(errors) else math.nan,
                "boundary_mean_signed_error_ns": float(np.mean(signed)) if len(signed) else math.nan,
                "fraction_within_one_probe": float(np.mean(errors <= attacker_period_ns)) if len(errors) else 0.0,
                "fraction_within_two_probes": float(np.mean(errors <= 2.0 * attacker_period_ns)) if len(errors) else 0.0,
                "tolerance_ns": tolerance_ns,
            }
        )

    boundary_metrics = pd.concat(
        [per_trace_metrics, pd.DataFrame(aggregate_rows)],
        ignore_index=True,
    )
    return boundary_metrics, details


# =============================================================================
# Summaries
# =============================================================================


def phase_class_summary(
    ground_truth: pd.DataFrame,
    blackbox: pd.DataFrame,
    split: pd.DataFrame,
) -> pd.DataFrame:
    gt = ground_truth.merge(split, on="victim_instance_id", validate="many_to_one")
    b = blackbox[
        ["trace_id", "probe_index", "excess_turnaround_ns", "delayed", "failure_transition"]
    ]
    d = gt.merge(b, on=["trace_id", "probe_index"], validate="one_to_one")
    return (
        d.groupby(["protocol_context", "split", "phase_label"], sort=True)
        .agg(
            probe_count=("probe_index", "count"),
            mean_excess_ns=("excess_turnaround_ns", "mean"),
            mean_abs_excess_ns=("excess_turnaround_ns", lambda x: float(np.mean(np.abs(x)))),
            delayed_fraction=("delayed", "mean"),
            failure_transition_fraction=("failure_transition", "mean"),
        )
        .reset_index()
    )


def protocol_comparison_summary(
    segmentation_metrics: pd.DataFrame,
    boundary_metrics: pd.DataFrame,
) -> pd.DataFrame:
    # Pick the best PRIMARY raw model by macro-F1 within each protocol.  Do not
    # let diagnostic baselines or offline debounce define the headline row.
    primary = segmentation_metrics[
        (segmentation_metrics["model_variant"] == "raw_online")
        & segmentation_metrics["model_name"].isin(
            ["logistic_regression", "random_forest", "hist_gradient_boosting"]
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    for protocol_context, dat in primary.groupby("protocol_context", sort=True):
        best = dat.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        bm = boundary_metrics[
            (boundary_metrics["protocol_context"] == protocol_context)
            & (boundary_metrics["model_name"] == best["model_name"])
            & (boundary_metrics["model_variant"] == "raw_online")
            & (boundary_metrics["trace_id"] == "__aggregate__")
        ]
        exact = bm[bm["boundary_mode"] == "phase_exact"]
        remote = bm[bm["boundary_mode"] == "remote_on_off"]
        erow = exact.iloc[0] if len(exact) else None
        rrow = remote.iloc[0] if len(remote) else None
        rows.append(
            {
                "protocol_context": protocol_context,
                "best_raw_model": str(best["model_name"]),
                "phase_accuracy": float(best["accuracy"]),
                "phase_macro_f1": float(best["macro_f1"]),
                "phase_macro_iou": float(best["macro_iou"]),
                "remote_vs_local_f1": float(best["remote_vs_local_f1"]),
                "exact_boundary_recall": float(erow["boundary_recall_within_tolerance"]) if erow is not None else math.nan,
                "exact_boundary_precision": float(erow["boundary_precision_within_tolerance"]) if erow is not None else math.nan,
                "exact_boundary_mae_ns": float(erow["boundary_mae_ns"]) if erow is not None else math.nan,
                "remote_on_off_boundary_recall": float(rrow["boundary_recall_within_tolerance"]) if rrow is not None else math.nan,
                "remote_on_off_boundary_mae_ns": float(rrow["boundary_mae_ns"]) if rrow is not None else math.nan,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Validation
# =============================================================================


def validation_row(group: str, name: str, passed: bool, expected: str, observed: str, details: str = ""):
    return {
        "validation_group": group,
        "assertion_name": name,
        "passed": bool(passed),
        "expected": expected,
        "observed": observed,
        "details": details,
    }


def build_validations(
    *,
    p27,
    contexts,
    instances: list[VictimInstance],
    schedule: pd.DataFrame,
    release_schedule: pd.DataFrame,
    blackbox: pd.DataFrame,
    ground_truth: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    split: pd.DataFrame,
    repeats_per_instance: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # 1. Strict attacker-visible schema.
    observed_cols = tuple(blackbox.columns)
    rows.append(
        validation_row(
            "blackbox_boundary",
            "attacker_trace_schema_exact",
            set(observed_cols) == set(ATTACKER_VISIBLE_COLUMNS),
            str(sorted(ATTACKER_VISIBLE_COLUMNS)),
            str(sorted(observed_cols)),
        )
    )

    # 2. Primary model features contain no absolute time/index or evaluator labels.
    forbidden_tokens = (
        "phase_label",
        "phase_index",
        "remote_active",
        "release_ns",
        "probe_index",
        "victim_",
        "resource",
        "epr",
        "wait_",
        "boundary",
    )
    bad_features = [
        c for c in feature_columns if any(token in c.lower() for token in forbidden_tokens)
    ]
    rows.append(
        validation_row(
            "blackbox_boundary",
            "primary_features_exclude_time_and_evaluator_state",
            len(bad_features) == 0,
            "[]",
            str(bad_features),
        )
    )

    # 3. Every instance includes every logical phase class.
    counts = schedule.groupby("victim_instance_id")["phase_label"].nunique()
    rows.append(
        validation_row(
            "schedule",
            "every_instance_contains_all_phase_classes",
            bool((counts == len(PHASE_LABELS)).all()),
            f"{len(PHASE_LABELS)} unique labels per instance",
            f"min={int(counts.min())}, max={int(counts.max())}",
        )
    )

    # 4. Adjacent phase labels always differ.
    adjacent_ok = True
    for _iid, g in schedule.groupby("victim_instance_id"):
        labels = g.sort_values("phase_index")["phase_label"].astype(str).tolist()
        adjacent_ok &= _valid_phase_order(labels)
    rows.append(
        validation_row(
            "schedule",
            "adjacent_phase_labels_differ",
            adjacent_ok,
            "True",
            str(bool(adjacent_ok)),
        )
    )

    # 5. Phase order is not fixed globally.
    orders = (
        schedule.sort_values(["victim_instance_id", "phase_index"])
        .groupby("victim_instance_id")["phase_label"]
        .apply(lambda x: "|".join(x.astype(str)))
    )
    rows.append(
        validation_row(
            "schedule",
            "phase_order_varies_across_instances",
            int(orders.nunique()) >= max(3, min(8, len(instances) // 2)),
            "multiple randomized phase orders",
            f"unique_orders={int(orders.nunique())}",
        )
    )

    # 6. Boundaries vary across instances (not fixed quartiles).
    boundary_signatures = (
        schedule[schedule["phase_index"] > 0]
        .groupby("victim_instance_id")["start_ns"]
        .apply(lambda x: "|".join(f"{v:.1f}" for v in x))
    )
    rows.append(
        validation_row(
            "schedule",
            "phase_boundaries_vary_across_instances",
            int(boundary_signatures.nunique()) >= max(3, min(8, len(instances) // 2)),
            "multiple boundary schedules",
            f"unique_boundary_schedules={int(boundary_signatures.nunique())}",
        )
    )

    # 7. No victim request is logically released during local_compute.
    local_release_count = 0
    if not release_schedule.empty:
        local_release_count = int((release_schedule["phase_label"] == "local_compute").sum())
    rows.append(
        validation_row(
            "schedule",
            "local_compute_releases_no_remote_operations",
            local_release_count == 0,
            "0",
            str(local_release_count),
        )
    )

    # 8. Every remote phase segment generates at least one release per repeat.
    expected_remote_segments = int((schedule["phase_label"].isin(REMOTE_PHASES)).sum()) * repeats_per_instance
    if release_schedule.empty:
        observed_remote_segments = 0
    else:
        observed_remote_segments = int(
            release_schedule[
                release_schedule["phase_label"].isin(REMOTE_PHASES)
            ][["victim_instance_id", "repeat_id", "phase_index"]]
            .drop_duplicates()
            .shape[0]
        )
    rows.append(
        validation_row(
            "schedule",
            "every_remote_phase_segment_has_remote_demand",
            observed_remote_segments == expected_remote_segments,
            str(expected_remote_segments),
            str(observed_remote_segments),
        )
    )

    # 9. Grouped train/test split has no victim-instance overlap.
    train_ids = set(split.loc[split["split"] == "train", "victim_instance_id"])
    test_ids = set(split.loc[split["split"] == "test", "victim_instance_id"])
    overlap = sorted(train_ids & test_ids)
    rows.append(
        validation_row(
            "evaluation",
            "group_split_has_no_instance_overlap",
            len(overlap) == 0,
            "[]",
            str(overlap),
        )
    )

    # 10. Every phase class appears in both train and test probe labels.
    gt_split = ground_truth.merge(split, on="victim_instance_id", validate="many_to_one")
    train_labels = set(gt_split.loc[gt_split["split"] == "train", "phase_label"])
    test_labels = set(gt_split.loc[gt_split["split"] == "test", "phase_label"])
    rows.append(
        validation_row(
            "evaluation",
            "all_phase_classes_present_in_train_and_test",
            train_labels == set(PHASE_LABELS) and test_labels == set(PHASE_LABELS),
            str(sorted(PHASE_LABELS)),
            f"train={sorted(train_labels)}, test={sorted(test_labels)}",
        )
    )

    # 11. Requested Phase-2.7 contexts are actually present.
    expected_contexts = {x[0] for x in contexts}
    observed_contexts = set(blackbox["protocol_context"].unique())
    rows.append(
        validation_row(
            "architecture",
            "requested_protocol_contexts_present",
            observed_contexts == expected_contexts,
            str(sorted(expected_contexts)),
            str(sorted(observed_contexts)),
        )
    )

    # 12. Identical logical phase ground truth appears under each protocol.
    # Compare labels by instance/repeat/probe, ignoring trace/protocol IDs.
    same_ground_truth = True
    if len(expected_contexts) == 2:
        contexts_sorted = sorted(expected_contexts)
        a = ground_truth[ground_truth["protocol_context"] == contexts_sorted[0]][
            ["victim_instance_id", "repeat_id", "probe_index", "phase_label"]
        ].sort_values(["victim_instance_id", "repeat_id", "probe_index"]).reset_index(drop=True)
        b = ground_truth[ground_truth["protocol_context"] == contexts_sorted[1]][
            ["victim_instance_id", "repeat_id", "probe_index", "phase_label"]
        ].sort_values(["victim_instance_id", "repeat_id", "probe_index"]).reset_index(drop=True)
        same_ground_truth = a.equals(b)
    rows.append(
        validation_row(
            "architecture",
            "same_logical_phase_schedule_across_protocols",
            same_ground_truth,
            "True",
            str(bool(same_ground_truth)),
        )
    )

    # 13. Phase-2.7 protocol normalization retained.
    protocols = p27.build_protocols()
    critical = [float(p.nominal_critical_latency_ns) for p in protocols.values()]
    cleanup = [float(p.postcompletion_cleanup_ns) for p in protocols.values()]
    norm_ok = all(abs(x - 150.0) < 1e-9 for x in critical) and all(abs(x - 120.0) < 1e-9 for x in cleanup)
    rows.append(
        validation_row(
            "architecture",
            "phase2_07_protocol_normalization_retained",
            norm_ok,
            "critical=150 ns and cleanup=120 ns for both protocols",
            f"critical={critical}, cleanup={cleanup}",
        )
    )

    # 14. Feature rows and ground-truth rows pair one-to-one without labels
    # embedded in the saved feature table.
    key = ["trace_id", "protocol_context", "victim_instance_id", "repeat_id", "probe_index"]
    pair_ok = (
        len(features) == len(ground_truth)
        and not features.duplicated(key).any()
        and not ground_truth.duplicated(key).any()
    )
    rows.append(
        validation_row(
            "evaluation",
            "feature_and_ground_truth_rows_pair_one_to_one",
            pair_ok,
            f"equal unique rows ({len(features)})",
            f"features={len(features)}, ground_truth={len(ground_truth)}",
        )
    )

    return pd.DataFrame(rows)


# =============================================================================
# Experiment driver
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p27, p27_source = load_phase2_07_module()
    contexts = select_contexts(p27, args.protocol)

    instances, schedule = make_instances_and_schedules(
        seed=args.seed,
        instance_count=args.instances,
        observation_window_ns=args.observation_window_ns,
        min_segment_ns=args.min_segment_ns,
    )

    print(
        f"[Phase 3.2] contexts={len(contexts)}, instances={len(instances)}, "
        f"repeats={args.repeats_per_instance}, window={args.observation_window_ns:g} ns"
    )
    print(f"[Phase 3.2] Reusing Phase-2.7 simulator: {p27_source}")

    blackbox, ground_truth, features, release_schedule = run_dataset(
        p27,
        contexts=contexts,
        instances=instances,
        schedule=schedule,
        repeats_per_instance=args.repeats_per_instance,
        seed=args.seed,
        observation_window_ns=args.observation_window_ns,
        causal_window_probes=args.causal_window_probes,
        lag_count=args.lag_count,
    )

    global MODEL_FEATURE_COLUMNS
    MODEL_FEATURE_COLUMNS = model_feature_columns(features)

    split = build_group_split(
        instances,
        seed=args.seed,
        test_size=args.test_size,
    )

    # Merge only inside evaluator-side analysis.  The saved attacker feature
    # file remains label-free.
    analysis = features.merge(
        ground_truth,
        on=[
            "trace_id",
            "protocol_context",
            "victim_instance_id",
            "repeat_id",
            "probe_index",
        ],
        validate="one_to_one",
    )

    segmentation_metrics, predictions = fit_and_predict_models(
        analysis,
        split,
        feature_columns=MODEL_FEATURE_COLUMNS,
        seed=args.seed,
        rf_trees=args.rf_trees,
    )

    confusion = build_confusion_output(predictions)

    boundary_tolerance_ns = (
        args.boundary_tolerance_probes * float(p27.ATTACKER_PERIOD_NS)
    )
    boundary_metrics, boundary_predictions = evaluate_boundaries(
        predictions,
        schedule,
        tolerance_ns=boundary_tolerance_ns,
        attacker_period_ns=float(p27.ATTACKER_PERIOD_NS),
    )

    phase_summary = phase_class_summary(ground_truth, blackbox, split)
    protocol_summary = protocol_comparison_summary(segmentation_metrics, boundary_metrics)

    validations = build_validations(
        p27=p27,
        contexts=contexts,
        instances=instances,
        schedule=schedule,
        release_schedule=release_schedule,
        blackbox=blackbox,
        ground_truth=ground_truth,
        features=features,
        feature_columns=MODEL_FEATURE_COLUMNS,
        split=split,
        repeats_per_instance=args.repeats_per_instance,
    )

    instance_table = pd.DataFrame([asdict(x) for x in instances])

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    blackbox.to_csv(output_dir / "phase3_02_attacker_visible_trace.csv", index=False)
    features.to_csv(output_dir / "phase3_02_probe_features.csv", index=False)
    ground_truth.to_csv(output_dir / "phase3_02_evaluator_probe_ground_truth.csv", index=False)
    schedule.to_csv(output_dir / "phase3_02_victim_phase_schedule.csv", index=False)
    release_schedule.to_csv(output_dir / "phase3_02_victim_release_schedule.csv", index=False)
    instance_table.to_csv(output_dir / "phase3_02_victim_instance_table.csv", index=False)
    split.to_csv(output_dir / "phase3_02_group_split.csv", index=False)
    segmentation_metrics.to_csv(output_dir / "phase3_02_segmentation_metrics.csv", index=False)
    predictions.to_csv(output_dir / "phase3_02_segmentation_predictions.csv", index=False)
    confusion.to_csv(output_dir / "phase3_02_confusion_matrix.csv", index=False)
    boundary_metrics.to_csv(output_dir / "phase3_02_boundary_metrics.csv", index=False)
    boundary_predictions.to_csv(output_dir / "phase3_02_boundary_predictions.csv", index=False)
    phase_summary.to_csv(output_dir / "phase3_02_phase_class_summary.csv", index=False)
    protocol_summary.to_csv(output_dir / "phase3_02_protocol_comparison_summary.csv", index=False)
    validations.to_csv(output_dir / "phase3_02_validation_assertions.csv", index=False)

    validation_summary = pd.DataFrame(
        [
            {
                "assertion_count": int(len(validations)),
                "passed_assertions": int(validations["passed"].sum()),
                "failed_assertions": int((~validations["passed"]).sum()),
                "all_passed": bool(validations["passed"].all()),
            }
        ]
    )
    validation_summary.to_csv(output_dir / "phase3_02_validation_summary.csv", index=False)

    manifest = {
        "experiment": "phase3_02_execution_phase_segmentation",
        "seed": args.seed,
        "phase2_07_source": str(p27_source),
        "output_dir": str(output_dir),
        "protocol_contexts": [x[0] for x in contexts],
        "phase_labels": list(PHASE_LABELS),
        "instance_count": args.instances,
        "repeats_per_instance": args.repeats_per_instance,
        "observation_window_ns": args.observation_window_ns,
        "attacker_first_release_ns": float(p27.ATTACKER_FIRST_RELEASE_NS),
        "attacker_period_ns": float(p27.ATTACKER_PERIOD_NS),
        "causal_window_probes": args.causal_window_probes,
        "lag_count": args.lag_count,
        "test_size": args.test_size,
        "rf_trees": args.rf_trees,
        "boundary_tolerance_probes": args.boundary_tolerance_probes,
        "boundary_tolerance_ns": boundary_tolerance_ns,
        "primary_model_feature_columns": MODEL_FEATURE_COLUMNS,
        "attacker_visible_columns": list(ATTACKER_VISIBLE_COLUMNS),
        "trace_count": int(blackbox["trace_id"].nunique()),
        "probe_row_count": int(len(blackbox)),
        "training_instance_count": int((split["split"] == "train").sum()),
        "test_instance_count": int((split["split"] == "test").sum()),
        "validation_assertions": int(len(validations)),
        "validation_passed": int(validations["passed"].sum()),
        "all_validation_passed": bool(validations["passed"].all()),
        "notes": [
            "Primary models use only current/past attacker timing observations.",
            "Absolute probe time/index is excluded from primary model features.",
            "Time-only classifier is a diagnostic baseline, not a primary attack model.",
            "Debounced predictions are offline diagnostic cleanup and are reported separately from raw online predictions.",
            "Phase order and phase duration are randomized independently per victim circuit instance.",
            "Logical victim phase schedules are identical across the two protocol contexts.",
            "Architecture timings are controlled simulation parameters, not vendor measurements.",
        ],
    }
    with open(output_dir / "phase3_02_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    print("\n[Phase 3.2] Validation")
    print(validation_summary.to_string(index=False))

    print("\n[Phase 3.2] Best raw model per protocol")
    if len(protocol_summary):
        print(protocol_summary.to_string(index=False))

    print("\n[Phase 3.2] Diagnostic baselines")
    baselines = segmentation_metrics[
        segmentation_metrics["model_name"].isin(["majority_baseline", "time_only_diagnostic"])
    ][
        ["protocol_context", "model_name", "accuracy", "macro_f1"]
    ]
    print(baselines.to_string(index=False))

    print(f"\n[Phase 3.2] Wrote outputs to: {output_dir.resolve()}")

    if args.fail_on_validation_error and not bool(validations["passed"].all()):
        failed = validations[~validations["passed"]]
        raise RuntimeError(
            "Phase 3.2 validation failed:\n" + failed.to_string(index=False)
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.2 — Execution-Phase Segmentation"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--instances",
        type=int,
        default=DEFAULT_INSTANCES,
        help="Number of distinct victim circuit instances.",
    )
    parser.add_argument(
        "--repeats-per-instance",
        type=int,
        default=DEFAULT_REPEATS_PER_INSTANCE,
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
        help="Models remain separate per selected protocol context.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
    )
    parser.add_argument(
        "--causal-window-probes",
        type=int,
        default=DEFAULT_CAUSAL_WINDOW_PROBES,
    )
    parser.add_argument(
        "--lag-count",
        type=int,
        default=DEFAULT_LAG_COUNT,
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=DEFAULT_RF_TREES,
    )
    parser.add_argument(
        "--boundary-tolerance-probes",
        type=float,
        default=DEFAULT_BOUNDARY_TOLERANCE_PROBES,
        help="Tolerance for boundary matching, in attacker probe periods.",
    )
    parser.add_argument(
        "--min-segment-ns",
        type=float,
        default=DEFAULT_MIN_SEGMENT_NS,
        help="Preferred minimum logical phase duration; auto-scaled for short smoke windows.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.instances < 6:
        raise ValueError("--instances must be at least 6 for grouped train/test evaluation")
    if args.repeats_per_instance < 1:
        raise ValueError("--repeats-per-instance must be >= 1")
    if not 0.10 <= args.test_size <= 0.50:
        raise ValueError("--test-size must be between 0.10 and 0.50")
    if args.observation_window_ns <= 2_000.0:
        raise ValueError("--observation-window-ns must be > 2000 ns")
    if args.causal_window_probes < 1:
        raise ValueError("--causal-window-probes must be >= 1")
    if args.lag_count < 0:
        raise ValueError("--lag-count must be >= 0")
    if args.rf_trees < 20:
        raise ValueError("--rf-trees must be >= 20")
    if args.boundary_tolerance_probes <= 0:
        raise ValueError("--boundary-tolerance-probes must be > 0")

    run_experiment(args)


if __name__ == "__main__":
    main()
