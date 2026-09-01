#!/usr/bin/env python3
"""
Phase 3.4 — Intermodule Communication-Graph Reconstruction
==========================================================

Research question
-----------------
Phase 3.1 asked HOW MUCH remote communication occurs.
Phase 3.2 asked WHEN the victim changes communication regime.
Phase 3.3 asked WHERE one persistent active region is located.

Phase 3.4 combines the temporal machinery of 3.2 with the active spatial
probing idea of 3.3 and asks:

    Can an attacker reconstruct which intermodule communication edges are
    active, and recover the hidden communication graph over time, using only
    the timing of its own deliberately targeted remote probes?

Architecture foundation
-----------------------
The experiment REUSES the validated Phase-2.7 protocol simulator rather than
introducing a new monolithic latency model.  The two protocol contexts are:

* direct_coherent_remote_cx
* entanglement_assisted_remote_cx

Both retain the Phase-2.7 normalization:

* 150 ns nominal post-prerequisite critical latency
* 120 ns post-completion cleanup

The attacker-visible / evaluator-only separation is retained.

Controlled graph model
----------------------
The default hidden victim region set is:

    module_1, module_2, module_3, module_4

which yields six undirected candidate communication edges:

    (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)

Each victim graph contains 2, 3, or 4 of those edges.  During one execution
phase, at most ONE hidden edge generates remote operations.  Two local/no-
remote phases are interspersed.  Thus the dynamic graph is piecewise constant,
while the union of active edges over the whole trace is a nontrivial hidden
communication graph.

The one-active-edge-per-phase restriction is deliberate.  It lets this first
graph-reconstruction experiment compose the already validated temporal and
spatial mechanisms without inventing unvalidated simultaneous multi-edge
resource semantics.  Concurrent multi-edge graph activity is a later
robustness extension.

Attacker capability
-------------------
The attacker actively scans candidate intermodule paths.  It is assumed to
have probe-capable allocations on the candidate module pairs (or an equivalent
API that allows it to submit its own remote operation through a known candidate
path).  The attacker knows the edge chosen for each of its own probes; the
victim graph and current active victim edge remain hidden.

This is a stronger active-probing capability than the single-source 3.3
experiment and should be stated explicitly in any paper claim.  If a platform
only exposes probes from one fixed attacker source, this experiment reduces to
reconstructing the star of edges visible from that source.

Probe policy
------------
The Phase-2/3 420 ns probe period is retained.  Candidate edges are sampled in
balanced randomly permuted scan blocks.  Every complete block contains one
probe to each candidate edge.

With the default 40,000 ns observation window:

    96 total probes
    / 6 candidate edges
    = 16 probes per edge

The longer window than 3.1–3.3 is intentional: the attacker now scans six
candidate paths rather than one global stream or four regions.

Two causal visibility modes
---------------------------
1. ``edge_localized`` — primary graph-reconstruction mode

   For a probe edge e, only victim requests whose hidden active edge is e are
   placed in that candidate edge's validated Phase-2.7 contention domain.
   Other victim edges do not occupy that route-local domain.

   This preserves the full protocol state evolution for edge e across the
   trace, including reset/EPR persistence after that edge was active.

2. ``global_only_control`` — negative control

   Every candidate edge sees the same NON-spatial victim communication demand
   through a protocol-specific common/global stack:

   * direct coherent: switch path + synchronous quantum link
   * entanglement-assisted: readout + feedforward

   Edge identities are randomized independently of temporal traffic profiles,
   so common timing leakage may remain, but there is no target-selective
   resource mapping from the hidden edge label to the probe edge.

Phase 3.3 already established that one-region localization survives a hybrid
local+global background.  Phase 3.4 therefore uses the clean route-local model
plus a global-only control instead of adding delays from separately simulated
local/global systems, which would violate Phase-2.6's demonstrated
sub-additive/masking semantics.

Inference tasks
---------------
A. Dynamic state reconstruction (causal, per probe)

   Seven-way classification by default:

       no_remote
       edge_1_2
       edge_1_3
       edge_1_4
       edge_2_3
       edge_2_4
       edge_3_4

   Features contain only the current and PAST attacker observations.  No
   future timing and no absolute probe time/index are model inputs.

B. Whole-trace communication-graph reconstruction

   Predicted dynamic edge states are aggregated into a predicted edge set.
   A presence threshold is calibrated using TRAINING traces only and then
   applied unchanged to held-out graph instances.

   Reported graph metrics include:

   * exact graph match
   * edge precision / recall / F1
   * edge-set Jaccard similarity
   * edge edit distance (false additions + missing edges)
   * graph edge-count error
   * node-degree MAE

C. Temporal edge-transition localization

   Predicted state changes are compared with true hidden phase boundaries.
   Boundary recall, precision, F1, and timing error are reported using a
   tolerance of one complete edge-scan cycle by default.

D. Causal edge-signal matrix

   Evaluator-side analysis compares attacker probe edge x true active victim
   edge.  This is used to explain why reconstruction succeeds; it is never an
   attacker input.

Evaluation discipline
---------------------
Train/test splitting is GROUPED BY HIDDEN GRAPH INSTANCE.  Both timing repeats
of one graph stay in the same split.  The split is additionally checked so all
candidate edges and all graph cardinalities appear in train and test.

Default output directory
------------------------
blackbox_window_results/phase3/phase3.4/

Run
---
From the repository directory containing
``phase2_07_remote_protocol_comparison.py``:

    python phase3_04_intermodule_graph_reconstruction.py

Quick smoke test:

    python phase3_04_intermodule_graph_reconstruction.py \
        --graph-instances 9 \
        --repeats-per-instance 1 \
        --observation-window-ns 16000 \
        --rf-trees 100 \
        --output-dir /tmp/phase3_04_smoke

Notes
-----
* All stage times/capacities are controlled simulation parameters, not vendor
  measurements.
* ``edge_localized`` is a controlled route-domain model built from the exact
  validated Phase-2.7 protocol semantics.
* ``global_only_control`` preserves common timing activity while intentionally
  removing edge-selective mapping.
* This experiment reconstructs a piecewise-constant dynamic graph with at most
  one actively communicating edge per phase.  Simultaneously active multiple
  victim edges are not claimed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from collections import defaultdict, deque
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
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Global settings
# =============================================================================

DEFAULT_SEED = 3401
DEFAULT_GRAPH_INSTANCES = 30
DEFAULT_REPEATS = 2
DEFAULT_OBSERVATION_WINDOW_NS = 40_000.0
DEFAULT_TEST_SIZE = 0.30
DEFAULT_MODULE_COUNT = 4
DEFAULT_RF_TREES = 500
DEFAULT_HISTORY = 3
DEFAULT_OUTPUT_DIR = Path("blackbox_window_results") / "phase3" / "phase3.4"

AFFECTED_THRESHOLD_NS = 1e-9
EPS = 1e-12

GRAPH_MODES = (
    "edge_localized",
    "global_only_control",
)

REMOTE_PROFILES = (
    "sparse_periodic",
    "dense_periodic",
    "synchronization_bursty",
)

ATTACKER_VISIBLE_COLUMNS = (
    "trace_id",
    "protocol_context",
    "probe_index",
    "probe_edge",
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
            "Phase 3.4 intentionally reuses the validated Phase-2.7 simulator.\n"
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
class DynamicPhase:
    phase_index: int
    phase_label: str
    active_edge: str
    remote_profile: str
    start_ns: float
    end_ns: float


@dataclass(frozen=True)
class GraphInstance:
    graph_instance_id: str
    instance_index: int
    graph_edges: tuple[str, ...]
    edge_count: int
    phases: tuple[DynamicPhase, ...]
    graph_signature: str


@dataclass(frozen=True)
class GraphContext:
    protocol_context: str
    protocol_name: str
    graph_mode: str
    scenario_id: str
    scenario_description: str


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


def opaque_id(*parts: Any, n: int = 20) -> str:
    token = "|".join(str(x) for x in parts).encode()
    return hashlib.sha256(token).hexdigest()[:n]


def module_names(count: int) -> tuple[str, ...]:
    return tuple(f"module_{i}" for i in range(1, count + 1))


def edge_label(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{x}--{y}"


def candidate_edges(modules: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(edge_label(a, b) for a, b in itertools.combinations(modules, 2))


def split_edge(edge: str) -> tuple[str, str]:
    a, b = edge.split("--", 1)
    return a, b


def graph_signature(edges: Iterable[str]) -> str:
    return opaque_id("graph", *sorted(edges), n=16)


# =============================================================================
# Hidden graph and temporal phase generation
# =============================================================================


def choose_edge_sets(
    *,
    seed: int,
    graph_instances: int,
    edges: tuple[str, ...],
) -> list[tuple[str, ...]]:
    """Balanced 2/3/4-edge hidden graphs from the candidate edge universe."""
    if graph_instances < 9:
        raise ValueError(
            "--graph-instances must be at least 9 so 2/3/4-edge graphs can "
            "appear in both grouped train and test splits."
        )

    by_k: dict[int, list[tuple[str, ...]]] = {}
    for k in (2, 3, 4):
        combos = list(itertools.combinations(edges, k))
        rng = np.random.default_rng(stable_seed(seed, "edge_sets", k))
        order = rng.permutation(len(combos))
        by_k[k] = [combos[i] for i in order]

    counters = defaultdict(int)
    out: list[tuple[str, ...]] = []
    for idx in range(graph_instances):
        k = 2 + (idx % 3)
        pool = by_k[k]
        j = counters[k]
        if j and j % len(pool) == 0:
            rng = np.random.default_rng(stable_seed(seed, "edge_sets_cycle", k, j))
            order = rng.permutation(len(pool))
            pool = [pool[i] for i in order]
            by_k[k] = pool
        out.append(tuple(sorted(pool[j % len(pool)])))
        counters[k] += 1
    return out


def _phase_token_order(
    *,
    rng: np.random.Generator,
    graph_edges: tuple[str, ...],
) -> list[str]:
    tokens = list(graph_edges) + ["no_remote", "no_remote"]
    for _ in range(500):
        perm = [tokens[i] for i in rng.permutation(len(tokens))]
        if all(not (perm[i] == perm[i + 1] == "no_remote") for i in range(len(perm) - 1)):
            return perm
    # Deterministic fallback.
    out: list[str] = []
    rem = list(graph_edges)
    out.append("no_remote")
    while rem:
        out.append(rem.pop(0))
        if len(out) == 2 and "no_remote" not in out[-1:]:
            out.append("no_remote")
    if out.count("no_remote") < 2:
        out.append("no_remote")
    return out[: len(tokens)]


def make_graph_instances(
    *,
    seed: int,
    graph_instances: int,
    edges: tuple[str, ...],
    observation_window_ns: float,
    scan_cycle_ns: float,
) -> list[GraphInstance]:
    edge_sets = choose_edge_sets(seed=seed, graph_instances=graph_instances, edges=edges)
    rows: list[GraphInstance] = []

    for idx, graph_edges in enumerate(edge_sets):
        gid = f"graph_{idx:04d}"
        rng = np.random.default_rng(stable_seed(seed, "graph_instance", idx))
        tokens = _phase_token_order(rng=rng, graph_edges=graph_edges)
        phase_count = len(tokens)

        # Every phase should last long enough to be sampled across a substantial
        # fraction of one full six-edge scan.  With defaults, minimum is 4.5 us.
        desired_min = max(4_500.0, 1.55 * scan_cycle_ns)
        feasible_min = 0.72 * observation_window_ns / phase_count
        min_duration = min(desired_min, feasible_min)
        min_duration = max(min_duration, 1.05 * scan_cycle_ns)

        reserved = phase_count * min_duration
        if reserved >= observation_window_ns:
            min_duration = 0.80 * observation_window_ns / phase_count
            reserved = phase_count * min_duration
        residual = max(0.0, observation_window_ns - reserved)
        weights = rng.dirichlet(np.ones(phase_count) * 2.5)
        durations = min_duration + residual * weights
        durations[-1] += observation_window_ns - float(np.sum(durations))

        profile_cycle = list(REMOTE_PROFILES)
        rng.shuffle(profile_cycle)
        profile_index = 0
        phases: list[DynamicPhase] = []
        now = 0.0
        for pidx, (token, duration) in enumerate(zip(tokens, durations)):
            end = observation_window_ns if pidx == phase_count - 1 else now + float(duration)
            if token == "no_remote":
                profile = "local_compute"
                active = ""
                label = "no_remote"
            else:
                profile = profile_cycle[profile_index % len(profile_cycle)]
                profile_index += 1
                active = token
                label = token
            phases.append(
                DynamicPhase(
                    phase_index=pidx,
                    phase_label=label,
                    active_edge=active,
                    remote_profile=profile,
                    start_ns=float(now),
                    end_ns=float(end),
                )
            )
            now = float(end)

        rows.append(
            GraphInstance(
                graph_instance_id=gid,
                instance_index=idx,
                graph_edges=tuple(sorted(graph_edges)),
                edge_count=len(graph_edges),
                phases=tuple(phases),
                graph_signature=graph_signature(graph_edges),
            )
        )

    return rows


def phase_for_time(instance: GraphInstance, time_ns: float) -> DynamicPhase:
    for phase in instance.phases:
        if phase.start_ns <= time_ns < phase.end_ns:
            return phase
    return instance.phases[-1]


def distance_to_boundary(instance: GraphInstance, time_ns: float) -> float:
    boundaries = [p.end_ns for p in instance.phases[:-1]]
    if not boundaries:
        return math.inf
    return float(min(abs(time_ns - b) for b in boundaries))


# =============================================================================
# Victim schedules
# =============================================================================


def generate_phase_releases(
    phase: DynamicPhase,
    *,
    seed: int,
    graph_instance_id: str,
    repeat_id: int,
) -> np.ndarray:
    if not phase.active_edge:
        return np.asarray([], dtype=float)

    rng = np.random.default_rng(
        stable_seed(seed, graph_instance_id, repeat_id, phase.phase_index, "victim_phase")
    )
    start = phase.start_ns + float(rng.uniform(110.0, 320.0))
    end = phase.end_ns - 20.0
    if start >= end:
        return np.asarray([], dtype=float)

    if phase.remote_profile == "sparse_periodic":
        interval = float(rng.uniform(930.0, 1_180.0))
        values = np.arange(start, end, interval, dtype=float)
    elif phase.remote_profile == "dense_periodic":
        interval = float(rng.uniform(455.0, 570.0))
        values = np.arange(start, end, interval, dtype=float)
    elif phase.remote_profile == "synchronization_bursty":
        burst_period = float(rng.uniform(1_450.0, 1_750.0))
        burst_size = int(rng.integers(3, 6))
        spacing = float(rng.uniform(60.0, 90.0))
        vals: list[float] = []
        base = start
        while base < end:
            for j in range(burst_size):
                t = base + j * spacing
                if t < end:
                    vals.append(float(t))
            base += burst_period
        values = np.asarray(vals, dtype=float)
    else:
        raise ValueError(phase.remote_profile)

    if len(values):
        values = values + rng.uniform(-5.0, 5.0, size=len(values))
        values = values[(values >= phase.start_ns) & (values < phase.end_ns)]
    return np.sort(values)


def generate_victim_release_table(
    instance: GraphInstance,
    *,
    seed: int,
    repeat_id: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    req_index = 0
    for phase in instance.phases:
        releases = generate_phase_releases(
            phase,
            seed=seed,
            graph_instance_id=instance.graph_instance_id,
            repeat_id=repeat_id,
        )
        for local_idx, t in enumerate(releases):
            rows.append(
                {
                    "graph_instance_id": instance.graph_instance_id,
                    "repeat_id": repeat_id,
                    "phase_index": phase.phase_index,
                    "active_edge": phase.active_edge,
                    "remote_profile": phase.remote_profile,
                    "phase_local_request_index": local_idx,
                    "victim_request_index": req_index,
                    "release_ns": float(t),
                }
            )
            req_index += 1
    return pd.DataFrame(rows)


def make_victim_specs(p27, release_table: pd.DataFrame, *, repeat_id: int) -> list[Any]:
    specs: list[Any] = []
    if release_table.empty:
        return specs
    for row in release_table.itertuples(index=False):
        specs.append(
            p27.RequestSpec(
                request_id=(
                    f"victim::{row.graph_instance_id}::repeat_{repeat_id:02d}::"
                    f"req_{int(row.victim_request_index):04d}"
                ),
                tenant="victim",
                ready_ns=float(row.release_ns),
                request_index=int(row.victim_request_index),
                workload_name="dynamic_graph",
                trial_id=repeat_id,
            )
        )
    return specs


# =============================================================================
# Attacker scan plan
# =============================================================================


def attacker_probe_plan(
    p27,
    *,
    edges: tuple[str, ...],
    graph_instance_id: str,
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
        stable_seed(seed, graph_instance_id, repeat_id, "attacker_edge_scan")
    )

    assigned: list[str] = []
    block = len(edges)
    while len(assigned) < len(releases):
        perm = [edges[i] for i in rng.permutation(block)]
        assigned.extend(perm)
    assigned = assigned[: len(releases)]

    return pd.DataFrame(
        {
            "probe_index": np.arange(len(releases), dtype=int),
            "release_ns": releases,
            "probe_edge": assigned,
        }
    )


def make_attacker_specs_for_edge(
    p27,
    *,
    plan: pd.DataFrame,
    probe_edge: str,
    graph_instance_id: str,
    repeat_id: int,
) -> list[Any]:
    sub = plan[plan["probe_edge"] == probe_edge]
    return [
        p27.RequestSpec(
            request_id=(
                f"attacker::{graph_instance_id}::repeat_{repeat_id:02d}::"
                f"probe_{int(row.probe_index):04d}"
            ),
            tenant="attacker",
            ready_ns=float(row.release_ns),
            request_index=int(row.probe_index),
            workload_name="dynamic_graph_probe",
            trial_id=repeat_id,
        )
        for row in sub.itertuples(index=False)
    ]


# =============================================================================
# Protocol contexts
# =============================================================================


def select_protocols(p27, protocol_choice: str):
    protocols = p27.build_protocols()
    out = []
    if protocol_choice in {"direct", "both"}:
        out.append(("direct_coherent", protocols[p27.DIRECT_PROTOCOL]))
    if protocol_choice in {"entangled", "both"}:
        out.append(("entanglement_assisted", protocols[p27.ENTANGLED_PROTOCOL]))
    return out


def build_graph_contexts(p27, protocol_choice: str, graph_modes: tuple[str, ...]):
    protocols = p27.build_protocols()
    scenarios = {s.scenario_id: s for s in p27.build_scenarios(protocols)}
    rows: list[tuple[GraphContext, Any, Any]] = []

    for short_name, protocol in select_protocols(p27, protocol_choice):
        pid = protocol.protocol_name
        for mode in graph_modes:
            if mode == "edge_localized":
                sid = f"{pid}__all_used_shared"
                desc = "route-local validated protocol resources for the matching candidate edge"
            elif mode == "global_only_control":
                if pid == p27.DIRECT_PROTOCOL:
                    sid = f"{pid}__share_interconnect_stack"
                    desc = "global switch-path + synchronous quantum-link timing only"
                elif pid == p27.ENTANGLED_PROTOCOL:
                    sid = f"{pid}__share_measurement_stack"
                    desc = "global readout + feedforward timing only"
                else:
                    raise ValueError(pid)
            else:
                raise ValueError(mode)

            ctx = GraphContext(
                protocol_context=f"{short_name}_dynamic_graph",
                protocol_name=pid,
                graph_mode=mode,
                scenario_id=sid,
                scenario_description=desc,
            )
            rows.append((ctx, protocol, scenarios[sid]))
    return rows


# =============================================================================
# Pair attacker timing
# =============================================================================


def pair_attacker_edge_trace(
    attacker_only: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    trace_id: str,
    protocol_context: str,
    probe_edge: str,
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
            "probe_edge": probe_edge,
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
# Dataset generation
# =============================================================================


def run_dataset(
    p27,
    *,
    contexts,
    graph_instances: list[GraphInstance],
    edges: tuple[str, ...],
    repeats_per_instance: int,
    seed: int,
    observation_window_ns: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blackbox_parts: list[pd.DataFrame] = []
    ground_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    release_parts: list[pd.DataFrame] = []
    dispatch_rows: list[dict[str, Any]] = []

    total = len(contexts) * len(graph_instances) * repeats_per_instance
    done = 0

    attacker_cache: dict[tuple[str, str, str, int, str], pd.DataFrame] = {}

    for ctx, protocol, scenario in contexts:
        for instance in graph_instances:
            for repeat_id in range(repeats_per_instance):
                plan = attacker_probe_plan(
                    p27,
                    edges=edges,
                    graph_instance_id=instance.graph_instance_id,
                    repeat_id=repeat_id,
                    seed=seed,
                    observation_window_ns=observation_window_ns,
                )

                for row in plan.itertuples(index=False):
                    plan_rows.append(
                        {
                            "protocol_context": ctx.protocol_context,
                            "graph_mode": ctx.graph_mode,
                            "graph_instance_id": instance.graph_instance_id,
                            "repeat_id": repeat_id,
                            "probe_index": int(row.probe_index),
                            "probe_edge": str(row.probe_edge),
                            "release_ns": float(row.release_ns),
                        }
                    )

                release_table = generate_victim_release_table(
                    instance,
                    seed=seed,
                    repeat_id=repeat_id,
                )
                if not release_table.empty:
                    rel = release_table.copy()
                    rel["protocol_context"] = ctx.protocol_context
                    rel["graph_mode"] = ctx.graph_mode
                    release_parts.append(rel)

                trace_id = opaque_id(
                    "phase3_04",
                    seed,
                    ctx.protocol_context,
                    ctx.graph_mode,
                    instance.graph_instance_id,
                    repeat_id,
                )
                trace_parts: list[pd.DataFrame] = []

                for probe_edge in edges:
                    attacker_specs = make_attacker_specs_for_edge(
                        p27,
                        plan=plan,
                        probe_edge=probe_edge,
                        graph_instance_id=instance.graph_instance_id,
                        repeat_id=repeat_id,
                    )

                    if ctx.graph_mode == "edge_localized":
                        victim_rows = release_table[
                            release_table["active_edge"] == probe_edge
                        ].copy() if not release_table.empty else release_table.copy()
                    elif ctx.graph_mode == "global_only_control":
                        victim_rows = release_table.copy()
                    else:
                        raise ValueError(ctx.graph_mode)

                    victim_specs = make_victim_specs(
                        p27,
                        victim_rows,
                        repeat_id=repeat_id,
                    )

                    dispatch_rows.append(
                        {
                            "protocol_context": ctx.protocol_context,
                            "graph_mode": ctx.graph_mode,
                            "graph_instance_id": instance.graph_instance_id,
                            "repeat_id": repeat_id,
                            "probe_edge": probe_edge,
                            "scenario_id": scenario.scenario_id,
                            "victim_request_count_used": len(victim_specs),
                            "victim_edges_used": "|".join(
                                sorted(set(victim_rows["active_edge"].astype(str)))
                            ) if len(victim_rows) else "",
                        }
                    )

                    cache_key = (
                        ctx.protocol_context,
                        scenario.scenario_id,
                        instance.graph_instance_id,
                        repeat_id,
                        probe_edge,
                    )
                    if cache_key not in attacker_cache:
                        attacker_only, *_ = p27.run_one(
                            protocol,
                            scenario,
                            "dynamic_graph_probe",
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
                        "dynamic_graph",
                        repeat_id,
                        "combined",
                        combined_specs,
                    )

                    paired = pair_attacker_edge_trace(
                        attacker_only,
                        combined,
                        trace_id=trace_id,
                        protocol_context=ctx.protocol_context,
                        probe_edge=probe_edge,
                    )
                    trace_parts.append(paired)

                trace = pd.concat(trace_parts, ignore_index=True).sort_values("probe_index")
                blackbox_parts.append(trace)

                for row in trace.itertuples(index=False):
                    phase = phase_for_time(instance, float(row.release_ns))
                    ground_rows.append(
                        {
                            "trace_id": trace_id,
                            "protocol_context": ctx.protocol_context,
                            "graph_mode": ctx.graph_mode,
                            "graph_instance_id": instance.graph_instance_id,
                            "repeat_id": repeat_id,
                            "probe_index": int(row.probe_index),
                            "probe_edge": str(row.probe_edge),
                            "release_ns": float(row.release_ns),
                            "phase_index": phase.phase_index,
                            "true_state_label": phase.phase_label,
                            "true_active_edge": phase.active_edge,
                            "remote_profile": phase.remote_profile,
                            "remote_active_label": int(bool(phase.active_edge)),
                            "phase_start_ns": phase.start_ns,
                            "phase_end_ns": phase.end_ns,
                            "distance_to_nearest_true_boundary_ns": distance_to_boundary(
                                instance, float(row.release_ns)
                            ),
                        }
                    )

                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    print(f"[Phase 3.4] Generated {done}/{total} high-level traces")

    blackbox = pd.concat(blackbox_parts, ignore_index=True)
    ground = pd.DataFrame(ground_rows)
    plan = pd.DataFrame(plan_rows)
    releases = (
        pd.concat(release_parts, ignore_index=True)
        if release_parts
        else pd.DataFrame(
            columns=[
                "graph_instance_id",
                "repeat_id",
                "phase_index",
                "active_edge",
                "remote_profile",
                "phase_local_request_index",
                "victim_request_index",
                "release_ns",
                "protocol_context",
                "graph_mode",
            ]
        )
    )
    dispatch = pd.DataFrame(dispatch_rows)
    return blackbox, ground, plan, releases, dispatch


# =============================================================================
# Causal temporal-spatial feature construction
# =============================================================================


def model_feature_columns(edges: tuple[str, ...]) -> list[str]:
    cols = [
        "current_excess_ns",
        "current_abs_excess_ns",
        "current_delayed",
        "current_speedup",
        "current_failure_transition",
        "latest_abs_top1_ns",
        "latest_abs_top1_minus_top2_ns",
        "latest_abs_top1_share",
    ]
    for edge in edges:
        prefix = f"edge::{edge}::"
        cols.extend(
            [
                prefix + "latest_excess_ns",
                prefix + "latest_abs_excess_ns",
                prefix + "latest_delayed",
                prefix + "latest_speedup",
                prefix + "latest_failure",
                prefix + "age_probes",
                prefix + "history_mean_excess_ns",
                prefix + "history_mean_abs_ns",
                prefix + "history_delayed_fraction",
                f"current_probe_is::{edge}",
            ]
        )
    return cols


def build_causal_probe_features(
    blackbox: pd.DataFrame,
    *,
    edges: tuple[str, ...],
    history: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unseen_age = len(edges) + 1

    for trace_id, trace in blackbox.groupby("trace_id", sort=False):
        t = trace.sort_values("probe_index")
        histories = {e: deque(maxlen=history) for e in edges}
        last_index = {e: None for e in edges}
        latest = {
            e: {
                "excess": 0.0,
                "abs": 0.0,
                "delayed": 0.0,
                "speedup": 0.0,
                "failure": 0.0,
            }
            for e in edges
        }

        for r in t.itertuples(index=False):
            idx = int(r.probe_index)
            edge = str(r.probe_edge)
            x = float(r.excess_turnaround_ns)
            obs = {
                "excess": x,
                "abs": abs(x),
                "delayed": float(bool(r.delayed)),
                "speedup": float(bool(r.speedup)),
                "failure": float(bool(r.failure_transition)),
            }
            latest[edge] = obs
            histories[edge].append(obs)
            last_index[edge] = idx

            row: dict[str, Any] = {
                "trace_id": trace_id,
                "protocol_context": str(r.protocol_context),
                "probe_index": idx,
                "probe_edge": edge,
                "current_excess_ns": x,
                "current_abs_excess_ns": abs(x),
                "current_delayed": float(bool(r.delayed)),
                "current_speedup": float(bool(r.speedup)),
                "current_failure_transition": float(bool(r.failure_transition)),
            }

            latest_abs_values = np.asarray([latest[e]["abs"] for e in edges], dtype=float)
            order = np.sort(latest_abs_values)[::-1]
            top1 = float(order[0]) if len(order) else 0.0
            top2 = float(order[1]) if len(order) > 1 else 0.0
            total = float(np.sum(latest_abs_values))
            row["latest_abs_top1_ns"] = top1
            row["latest_abs_top1_minus_top2_ns"] = top1 - top2
            row["latest_abs_top1_share"] = top1 / total if total > EPS else 0.0

            for e in edges:
                prefix = f"edge::{e}::"
                h = list(histories[e])
                if h:
                    mean_excess = float(np.mean([z["excess"] for z in h]))
                    mean_abs = float(np.mean([z["abs"] for z in h]))
                    delayed_fraction = float(np.mean([z["delayed"] for z in h]))
                else:
                    mean_excess = 0.0
                    mean_abs = 0.0
                    delayed_fraction = 0.0

                age = (
                    idx - int(last_index[e])
                    if last_index[e] is not None
                    else unseen_age
                )
                row[prefix + "latest_excess_ns"] = latest[e]["excess"]
                row[prefix + "latest_abs_excess_ns"] = latest[e]["abs"]
                row[prefix + "latest_delayed"] = latest[e]["delayed"]
                row[prefix + "latest_speedup"] = latest[e]["speedup"]
                row[prefix + "latest_failure"] = latest[e]["failure"]
                row[prefix + "age_probes"] = float(age)
                row[prefix + "history_mean_excess_ns"] = mean_excess
                row[prefix + "history_mean_abs_ns"] = mean_abs
                row[prefix + "history_delayed_fraction"] = delayed_fraction
                row[f"current_probe_is::{e}"] = float(edge == e)

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Grouped split
# =============================================================================


def build_group_split(
    instances: list[GraphInstance],
    *,
    edges: tuple[str, ...],
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {
                "graph_instance_id": g.graph_instance_id,
                "edge_count": g.edge_count,
                "graph_edges": "|".join(g.graph_edges),
            }
            for g in instances
        ]
    )

    for attempt in range(500):
        train_ids, test_ids = train_test_split(
            table["graph_instance_id"],
            test_size=test_size,
            random_state=seed + attempt,
            stratify=table["edge_count"],
        )
        train_set = set(train_ids)
        test_set = set(test_ids)

        def coverage(ids: set[str]) -> set[str]:
            vals: set[str] = set()
            for g in instances:
                if g.graph_instance_id in ids:
                    vals.update(g.graph_edges)
            return vals

        if coverage(train_set) == set(edges) and coverage(test_set) == set(edges):
            out = table.copy()
            out["split"] = out["graph_instance_id"].map(
                lambda x: "train" if x in train_set else "test"
            )
            return out.sort_values(["edge_count", "graph_instance_id"]).reset_index(drop=True)

    raise RuntimeError(
        "Could not construct a grouped split with complete candidate-edge coverage "
        "in both train and test. Increase --graph-instances."
    )


# =============================================================================
# Models and dynamic state evaluation
# =============================================================================


def state_models(seed: int, rf_trees: int):
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
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def active_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = (y_true != "no_remote").astype(int)
    yp = (y_pred != "no_remote").astype(int)
    return {
        "remote_vs_no_accuracy": float(accuracy_score(yt, yp)),
        "remote_vs_no_f1": float(f1_score(yt, yp, zero_division=0)),
    }


def evaluate_state_models(
    analysis: pd.DataFrame,
    split: pd.DataFrame,
    *,
    feature_columns: list[str],
    state_labels: tuple[str, ...],
    seed: int,
    rf_trees: int,
    scan_cycle_ns: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = analysis.merge(
        split[["graph_instance_id", "split"]],
        on="graph_instance_id",
        how="left",
        validate="many_to_one",
    )

    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for (protocol_context, graph_mode), subset in data.groupby(
        ["protocol_context", "graph_mode"], sort=True
    ):
        train = subset[subset["split"] == "train"].copy()
        test = subset[subset["split"] == "test"].copy()

        X_train = train[feature_columns].to_numpy(dtype=float)
        X_test = test[feature_columns].to_numpy(dtype=float)
        y_train = train["true_state_label"].astype(str).to_numpy()
        y_test = test["true_state_label"].astype(str).to_numpy()

        models = state_models(stable_seed(seed, protocol_context, graph_mode), rf_trees)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred_train = model.predict(X_train).astype(str)
            pred_test = model.predict(X_test).astype(str)

            for split_name, part, pred in (
                ("train", train, pred_train),
                ("test", test, pred_test),
            ):
                for (_, r), p in zip(part.iterrows(), pred):
                    pred_rows.append(
                        {
                            "trace_id": r["trace_id"],
                            "protocol_context": protocol_context,
                            "graph_mode": graph_mode,
                            "graph_instance_id": r["graph_instance_id"],
                            "repeat_id": int(r["repeat_id"]),
                            "probe_index": int(r["probe_index"]),
                            "release_ns": float(r["release_ns"]),
                            "probe_edge": r["probe_edge"],
                            "true_state_label": r["true_state_label"],
                            "predicted_state_label": p,
                            "distance_to_nearest_true_boundary_ns": float(
                                r["distance_to_nearest_true_boundary_ns"]
                            ),
                            "model_name": model_name,
                            "split": split_name,
                        }
                    )

            acc = accuracy_score(y_test, pred_test)
            bal = balanced_accuracy_score(y_test, pred_test)
            macro = f1_score(y_test, pred_test, average="macro", zero_division=0)
            binary = active_binary_metrics(y_test, pred_test)
            remote_mask = y_test != "no_remote"
            remote_edge_accuracy = (
                float(accuracy_score(y_test[remote_mask], pred_test[remote_mask]))
                if remote_mask.any()
                else math.nan
            )
            remote_edge_macro_f1 = (
                float(
                    f1_score(
                        y_test[remote_mask],
                        pred_test[remote_mask],
                        labels=list(state_labels[1:]),
                        average="macro",
                        zero_division=0,
                    )
                )
                if remote_mask.any()
                else math.nan
            )
            stable_mask = (
                test["distance_to_nearest_true_boundary_ns"].to_numpy(dtype=float)
                > scan_cycle_ns
            )
            near_mask = ~stable_mask
            stable_acc = (
                accuracy_score(y_test[stable_mask], pred_test[stable_mask])
                if stable_mask.any()
                else math.nan
            )
            near_acc = (
                accuracy_score(y_test[near_mask], pred_test[near_mask])
                if near_mask.any()
                else math.nan
            )

            metric_rows.append(
                {
                    "protocol_context": protocol_context,
                    "graph_mode": graph_mode,
                    "model_name": model_name,
                    "sample_count": len(test),
                    "state_class_count": len(state_labels),
                    "chance_accuracy": 1.0 / len(state_labels),
                    "accuracy": float(acc),
                    "balanced_accuracy": float(bal),
                    "macro_f1": float(macro),
                    "remote_edge_identity_accuracy": remote_edge_accuracy,
                    "remote_edge_identity_macro_f1": remote_edge_macro_f1,
                    "remote_edge_identity_chance": 1.0 / max(1, len(state_labels) - 1),
                    "stable_interior_accuracy_gt_one_scan": float(stable_acc),
                    "near_boundary_accuracy_within_one_scan": float(near_acc),
                    **binary,
                }
            )

            cm = confusion_matrix(y_test, pred_test, labels=list(state_labels))
            row_sums = cm.sum(axis=1, keepdims=True)
            norm = np.divide(
                cm,
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums > 0,
            )
            for i, true_label in enumerate(state_labels):
                for j, pred_label in enumerate(state_labels):
                    confusion_rows.append(
                        {
                            "protocol_context": protocol_context,
                            "graph_mode": graph_mode,
                            "model_name": model_name,
                            "true_state_label": true_label,
                            "predicted_state_label": pred_label,
                            "count": int(cm[i, j]),
                            "true_normalized_fraction": float(norm[i, j]),
                        }
                    )

        # Majority diagnostic baseline.
        majority = train["true_state_label"].value_counts().idxmax()
        pred_majority = np.repeat(str(majority), len(test))
        metric_rows.append(
            {
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "model_name": "majority_baseline",
                "sample_count": len(test),
                "state_class_count": len(state_labels),
                "chance_accuracy": 1.0 / len(state_labels),
                "accuracy": float(accuracy_score(y_test, pred_majority)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_majority)),
                "macro_f1": float(
                    f1_score(y_test, pred_majority, average="macro", zero_division=0)
                ),
                "remote_edge_identity_accuracy": float(
                    accuracy_score(y_test[y_test != "no_remote"], pred_majority[y_test != "no_remote"])
                ) if np.any(y_test != "no_remote") else math.nan,
                "remote_edge_identity_macro_f1": float(
                    f1_score(
                        y_test[y_test != "no_remote"],
                        pred_majority[y_test != "no_remote"],
                        labels=list(state_labels[1:]),
                        average="macro",
                        zero_division=0,
                    )
                ) if np.any(y_test != "no_remote") else math.nan,
                "remote_edge_identity_chance": 1.0 / max(1, len(state_labels) - 1),
                "stable_interior_accuracy_gt_one_scan": math.nan,
                "near_boundary_accuracy_within_one_scan": math.nan,
                **active_binary_metrics(y_test, pred_majority),
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(confusion_rows)


# =============================================================================
# Graph reconstruction from dynamic predictions
# =============================================================================


def edge_degrees(edge_set: set[str], modules: tuple[str, ...]) -> dict[str, int]:
    deg = {m: 0 for m in modules}
    for edge in edge_set:
        a, b = split_edge(edge)
        if a in deg:
            deg[a] += 1
        if b in deg:
            deg[b] += 1
    return deg


def set_metrics(true_edges: set[str], pred_edges: set[str], modules: tuple[str, ...]) -> dict[str, float]:
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not true_edges else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(true_edges | pred_edges)
    jaccard = len(true_edges & pred_edges) / union if union else 1.0
    td = edge_degrees(true_edges, modules)
    pdg = edge_degrees(pred_edges, modules)
    degree_mae = float(np.mean([abs(td[m] - pdg[m]) for m in modules]))
    return {
        "true_edge_count": len(true_edges),
        "predicted_edge_count": len(pred_edges),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(f1),
        "edge_jaccard": float(jaccard),
        "exact_graph_match": float(true_edges == pred_edges),
        "edge_edit_distance": float(fp + fn),
        "edge_count_absolute_error": float(abs(len(true_edges) - len(pred_edges))),
        "node_degree_mae": degree_mae,
    }


def trace_prediction_edge_fractions(pred: pd.DataFrame, edges: tuple[str, ...]) -> dict[str, float]:
    n = max(1, len(pred))
    counts = pred["predicted_state_label"].value_counts()
    return {edge: float(counts.get(edge, 0) / n) for edge in edges}


def true_graph_lookup(instances: list[GraphInstance]) -> dict[str, set[str]]:
    return {g.graph_instance_id: set(g.graph_edges) for g in instances}


def calibrate_presence_thresholds(
    predictions: pd.DataFrame,
    *,
    instances: list[GraphInstance],
    edges: tuple[str, ...],
    modules: tuple[str, ...],
) -> pd.DataFrame:
    true_lookup = true_graph_lookup(instances)
    grid = np.unique(np.concatenate([
        np.linspace(0.01, 0.20, 20),
        np.asarray([0.025, 0.035, 0.045, 0.055, 0.075, 0.10, 0.125, 0.15]),
    ]))
    rows: list[dict[str, Any]] = []

    train = predictions[predictions["split"] == "train"]
    for (protocol_context, graph_mode, model_name), subset in train.groupby(
        ["protocol_context", "graph_mode", "model_name"], sort=True
    ):
        best = None
        for threshold in grid:
            metrics = []
            for trace_id, tr in subset.groupby("trace_id", sort=False):
                gid = str(tr["graph_instance_id"].iloc[0])
                fractions = trace_prediction_edge_fractions(tr, edges)
                pred_edges = {e for e, f in fractions.items() if f >= threshold}
                metrics.append(set_metrics(true_lookup[gid], pred_edges, modules))
            mdf = pd.DataFrame(metrics)
            exact = float(mdf["exact_graph_match"].mean())
            f1 = float(mdf["edge_f1"].mean())
            jac = float(mdf["edge_jaccard"].mean())
            score = (exact, f1, jac, -abs(float(threshold) - 0.05))
            if best is None or score > best[0]:
                best = (score, float(threshold), exact, f1, jac)
        assert best is not None
        rows.append(
            {
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "model_name": model_name,
                "presence_fraction_threshold": best[1],
                "training_exact_graph_match": best[2],
                "training_mean_edge_f1": best[3],
                "training_mean_edge_jaccard": best[4],
            }
        )
    return pd.DataFrame(rows)


def reconstruct_graphs(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    instances: list[GraphInstance],
    edges: tuple[str, ...],
    modules: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_lookup = true_graph_lookup(instances)
    test = predictions[predictions["split"] == "test"].copy()
    threshold_lookup = {
        (r.protocol_context, r.graph_mode, r.model_name): float(r.presence_fraction_threshold)
        for r in thresholds.itertuples(index=False)
    }

    detail_rows: list[dict[str, Any]] = []
    for (protocol_context, graph_mode, model_name, trace_id), tr in test.groupby(
        ["protocol_context", "graph_mode", "model_name", "trace_id"], sort=True
    ):
        gid = str(tr["graph_instance_id"].iloc[0])
        repeat_id = int(tr["repeat_id"].iloc[0])
        threshold = threshold_lookup[(protocol_context, graph_mode, model_name)]
        fractions = trace_prediction_edge_fractions(tr, edges)
        pred_edges = {e for e, f in fractions.items() if f >= threshold}
        true_edges = true_lookup[gid]
        metrics = set_metrics(true_edges, pred_edges, modules)
        detail_rows.append(
            {
                "trace_id": trace_id,
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "model_name": model_name,
                "graph_instance_id": gid,
                "repeat_id": repeat_id,
                "presence_fraction_threshold": threshold,
                "true_edges": "|".join(sorted(true_edges)),
                "predicted_edges": "|".join(sorted(pred_edges)),
                **{f"predicted_fraction::{e}": fractions[e] for e in edges},
                **metrics,
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["protocol_context", "graph_mode", "model_name"], sort=True)
        .agg(
            trace_count=("trace_id", "count"),
            exact_graph_match=("exact_graph_match", "mean"),
            mean_edge_precision=("edge_precision", "mean"),
            mean_edge_recall=("edge_recall", "mean"),
            mean_edge_f1=("edge_f1", "mean"),
            mean_edge_jaccard=("edge_jaccard", "mean"),
            mean_edge_edit_distance=("edge_edit_distance", "mean"),
            mean_edge_count_absolute_error=("edge_count_absolute_error", "mean"),
            mean_node_degree_mae=("node_degree_mae", "mean"),
        )
        .reset_index()
    )
    return detail, summary


# =============================================================================
# Boundary metrics
# =============================================================================


def greedy_match_boundaries(
    true_times: list[float],
    pred_times: list[float],
    tolerance_ns: float,
) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    unmatched_pred = set(range(len(pred_times)))
    matches: list[tuple[float, float]] = []
    missed: list[float] = []

    for t in true_times:
        candidates = [
            (abs(pred_times[j] - t), j)
            for j in unmatched_pred
            if abs(pred_times[j] - t) <= tolerance_ns
        ]
        if not candidates:
            missed.append(t)
            continue
        _, j = min(candidates)
        unmatched_pred.remove(j)
        matches.append((t, pred_times[j]))
    extras = [pred_times[j] for j in sorted(unmatched_pred)]
    return matches, missed, extras


def boundary_metrics(
    predictions: pd.DataFrame,
    instances: list[GraphInstance],
    *,
    tolerance_ns: float,
) -> pd.DataFrame:
    lookup = {g.graph_instance_id: g for g in instances}
    test = predictions[predictions["split"] == "test"]
    rows: list[dict[str, Any]] = []

    for (protocol_context, graph_mode, model_name, trace_id), tr in test.groupby(
        ["protocol_context", "graph_mode", "model_name", "trace_id"], sort=True
    ):
        tr = tr.sort_values("probe_index")
        gid = str(tr["graph_instance_id"].iloc[0])
        inst = lookup[gid]
        true_times = [float(p.end_ns) for p in inst.phases[:-1]]

        labels = tr["predicted_state_label"].astype(str).to_numpy()
        releases = tr["release_ns"].to_numpy(dtype=float)
        pred_times: list[float] = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                pred_times.append(float((releases[i - 1] + releases[i]) / 2.0))

        matches, missed, extras = greedy_match_boundaries(
            true_times, pred_times, tolerance_ns
        )
        recall = len(matches) / len(true_times) if true_times else 1.0
        precision = len(matches) / len(pred_times) if pred_times else (1.0 if not true_times else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        abs_errors = [abs(p - t) for t, p in matches]
        signed_errors = [p - t for t, p in matches]
        rows.append(
            {
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "model_name": model_name,
                "trace_id": trace_id,
                "graph_instance_id": gid,
                "repeat_id": int(tr["repeat_id"].iloc[0]),
                "true_boundary_count": len(true_times),
                "predicted_boundary_count": len(pred_times),
                "matched_boundary_count": len(matches),
                "boundary_recall_within_tolerance": recall,
                "boundary_precision_within_tolerance": precision,
                "boundary_f1_within_tolerance": f1,
                "boundary_mae_ns": float(np.mean(abs_errors)) if abs_errors else math.nan,
                "boundary_median_absolute_error_ns": float(np.median(abs_errors)) if abs_errors else math.nan,
                "boundary_mean_signed_error_ns": float(np.mean(signed_errors)) if signed_errors else math.nan,
                "tolerance_ns": tolerance_ns,
            }
        )

    detail = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, Any]] = []
    for keys, sub in detail.groupby(
        ["protocol_context", "graph_mode", "model_name"], sort=True
    ):
        protocol_context, graph_mode, model_name = keys
        total_true = int(sub["true_boundary_count"].sum())
        total_pred = int(sub["predicted_boundary_count"].sum())
        total_match = int(sub["matched_boundary_count"].sum())
        recall = total_match / total_true if total_true else 1.0
        precision = total_match / total_pred if total_pred else (1.0 if total_true == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate_rows.append(
            {
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "model_name": model_name,
                "trace_id": "__aggregate__",
                "graph_instance_id": "__aggregate__",
                "repeat_id": -1,
                "true_boundary_count": total_true,
                "predicted_boundary_count": total_pred,
                "matched_boundary_count": total_match,
                "boundary_recall_within_tolerance": recall,
                "boundary_precision_within_tolerance": precision,
                "boundary_f1_within_tolerance": f1,
                "boundary_mae_ns": float(sub["boundary_mae_ns"].mean()),
                "boundary_median_absolute_error_ns": float(sub["boundary_median_absolute_error_ns"].median()),
                "boundary_mean_signed_error_ns": float(sub["boundary_mean_signed_error_ns"].mean()),
                "tolerance_ns": tolerance_ns,
            }
        )
    return pd.concat([detail, pd.DataFrame(aggregate_rows)], ignore_index=True)


# =============================================================================
# Evaluator-side causal signal matrix
# =============================================================================


def build_edge_signal_matrix(
    blackbox: pd.DataFrame,
    ground: pd.DataFrame,
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = blackbox.merge(
        ground[
            [
                "trace_id",
                "probe_index",
                "graph_mode",
                "graph_instance_id",
                "true_active_edge",
                "true_state_label",
                "remote_profile",
            ]
        ],
        on=["trace_id", "probe_index"],
        validate="one_to_one",
    ).merge(
        split[["graph_instance_id", "split"]],
        on="graph_instance_id",
        validate="many_to_one",
    )
    paired = paired[(paired["split"] == "test") & (paired["true_active_edge"] != "")].copy()
    paired["abs_excess_ns"] = paired["excess_turnaround_ns"].abs()
    paired["positive_excess_ns"] = paired["excess_turnaround_ns"].clip(lower=0.0)
    paired["matching_edge"] = paired["probe_edge"] == paired["true_active_edge"]

    matrix = (
        paired.groupby(
            [
                "protocol_context",
                "graph_mode",
                "remote_profile",
                "true_active_edge",
                "probe_edge",
            ],
            sort=True,
        )
        .agg(
            probe_count=("probe_index", "count"),
            mean_excess_ns=("excess_turnaround_ns", "mean"),
            mean_abs_excess_ns=("abs_excess_ns", "mean"),
            mean_positive_excess_ns=("positive_excess_ns", "mean"),
            delayed_fraction=("delayed", "mean"),
            speedup_fraction=("speedup", "mean"),
            failure_transition_fraction=("failure_transition", "mean"),
        )
        .reset_index()
    )

    contrast_rows: list[dict[str, Any]] = []
    for (protocol_context, graph_mode), sub in paired.groupby(
        ["protocol_context", "graph_mode"], sort=True
    ):
        match = sub[sub["matching_edge"]]
        non = sub[~sub["matching_edge"]]
        contrast_rows.append(
            {
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "matching_probe_count": len(match),
                "nonmatching_probe_count": len(non),
                "matching_mean_abs_excess_ns": float(match["abs_excess_ns"].mean()) if len(match) else 0.0,
                "nonmatching_mean_abs_excess_ns": float(non["abs_excess_ns"].mean()) if len(non) else 0.0,
                "mean_abs_diagonal_minus_offdiagonal_ns": (
                    float(match["abs_excess_ns"].mean()) - float(non["abs_excess_ns"].mean())
                    if len(match) and len(non)
                    else 0.0
                ),
                "matching_delayed_fraction": float(match["delayed"].mean()) if len(match) else 0.0,
                "nonmatching_delayed_fraction": float(non["delayed"].mean()) if len(non) else 0.0,
                "matching_speedup_fraction": float(match["speedup"].mean()) if len(match) else 0.0,
                "nonmatching_speedup_fraction": float(non["speedup"].mean()) if len(non) else 0.0,
            }
        )
    return matrix, pd.DataFrame(contrast_rows)


# =============================================================================
# Compact protocol comparison
# =============================================================================


def protocol_comparison_summary(
    state_metrics: pd.DataFrame,
    graph_summary: pd.DataFrame,
    boundary: pd.DataFrame,
    contrast: pd.DataFrame,
) -> pd.DataFrame:
    state_models_only = state_metrics[~state_metrics["model_name"].str.contains("baseline")].copy()
    rows: list[dict[str, Any]] = []
    for (protocol_context, graph_mode), sub in state_models_only.groupby(
        ["protocol_context", "graph_mode"], sort=True
    ):
        best_state = sub.sort_values(
            ["macro_f1", "accuracy"], ascending=False
        ).iloc[0]
        gsub = graph_summary[
            (graph_summary["protocol_context"] == protocol_context)
            & (graph_summary["graph_mode"] == graph_mode)
        ]
        if len(gsub):
            best_graph = gsub.sort_values(
                ["exact_graph_match", "mean_edge_f1"], ascending=False
            ).iloc[0]
        else:
            best_graph = None

        bsub = boundary[
            (boundary["protocol_context"] == protocol_context)
            & (boundary["graph_mode"] == graph_mode)
            & (boundary["trace_id"] == "__aggregate__")
            & (boundary["model_name"] == best_state["model_name"])
        ]
        csub = contrast[
            (contrast["protocol_context"] == protocol_context)
            & (contrast["graph_mode"] == graph_mode)
        ]

        rows.append(
            {
                "protocol_context": protocol_context,
                "graph_mode": graph_mode,
                "best_state_model": best_state["model_name"],
                "dynamic_state_accuracy": float(best_state["accuracy"]),
                "dynamic_state_macro_f1": float(best_state["macro_f1"]),
                "remote_edge_identity_accuracy": float(best_state["remote_edge_identity_accuracy"]),
                "remote_edge_identity_macro_f1": float(best_state["remote_edge_identity_macro_f1"]),
                "remote_edge_identity_chance": float(best_state["remote_edge_identity_chance"]),
                "stable_interior_accuracy_gt_one_scan": float(
                    best_state["stable_interior_accuracy_gt_one_scan"]
                ),
                "best_graph_model": str(best_graph["model_name"]) if best_graph is not None else "",
                "exact_graph_match": float(best_graph["exact_graph_match"]) if best_graph is not None else math.nan,
                "mean_graph_edge_f1": float(best_graph["mean_edge_f1"]) if best_graph is not None else math.nan,
                "mean_graph_jaccard": float(best_graph["mean_edge_jaccard"]) if best_graph is not None else math.nan,
                "boundary_recall_one_scan": float(bsub["boundary_recall_within_tolerance"].iloc[0]) if len(bsub) else math.nan,
                "boundary_precision_one_scan": float(bsub["boundary_precision_within_tolerance"].iloc[0]) if len(bsub) else math.nan,
                "boundary_mae_ns": float(bsub["boundary_mae_ns"].iloc[0]) if len(bsub) else math.nan,
                "matching_mean_abs_excess_ns": float(csub["matching_mean_abs_excess_ns"].iloc[0]) if len(csub) else math.nan,
                "nonmatching_mean_abs_excess_ns": float(csub["nonmatching_mean_abs_excess_ns"].iloc[0]) if len(csub) else math.nan,
                "diagonal_contrast_ns": float(csub["mean_abs_diagonal_minus_offdiagonal_ns"].iloc[0]) if len(csub) else math.nan,
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
    ground: pd.DataFrame,
    features: pd.DataFrame,
    split: pd.DataFrame,
    graph_instances: list[GraphInstance],
    edges: tuple[str, ...],
    contexts,
    plan: pd.DataFrame,
    releases: pd.DataFrame,
    dispatch: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[ValidationAssertion] = []

    def add(group, name, passed, expected, observed, details=""):
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

    add(
        "blackbox_boundary",
        "attacker_trace_schema_exact",
        list(blackbox.columns) == list(ATTACKER_VISIBLE_COLUMNS),
        list(ATTACKER_VISIBLE_COLUMNS),
        list(blackbox.columns),
    )

    forbidden_tokens = (
        "true_",
        "hidden",
        "graph_edges",
        "graph_instance",
        "phase_",
        "remote_profile",
        "resource",
        "wait_",
        "epr_",
        "scenario",
        "split",
    )
    leaked_cols = [
        c for c in blackbox.columns if any(tok in c.lower() for tok in forbidden_tokens)
    ]
    add(
        "blackbox_boundary",
        "hidden_graph_and_evaluator_state_excluded",
        not leaked_cols,
        [],
        leaked_cols,
    )

    forbidden_feature_tokens = (
        "release_ns",
        "probe_index",
        "true_",
        "graph_instance",
        "phase_",
        "remote_profile",
        "distance_to",
    )
    leaked_features = [
        c for c in feature_columns if any(tok in c.lower() for tok in forbidden_feature_tokens)
    ]
    add(
        "blackbox_boundary",
        "primary_features_are_causal_and_exclude_absolute_time_evaluator_labels",
        not leaked_features,
        [],
        leaked_features,
    )

    bad_trace_ids = 0
    edge_tokens = [e.replace("--", "") for e in edges]
    for tid in blackbox["trace_id"].astype(str).unique():
        if any(e in tid for e in edges) or "graph_" in tid or "edge_localized" in tid:
            bad_trace_ids += 1
    add(
        "blackbox_boundary",
        "trace_ids_are_opaque",
        bad_trace_ids == 0,
        0,
        bad_trace_ids,
    )

    edge_counts = {g.edge_count for g in graph_instances}
    add(
        "graph_schedule",
        "graph_cardinalities_two_three_four_present",
        edge_counts == {2, 3, 4},
        {2, 3, 4},
        edge_counts,
    )

    max_active_per_phase = max(
        int(bool(p.active_edge)) for g in graph_instances for p in g.phases
    )
    add(
        "graph_schedule",
        "at_most_one_active_edge_per_phase",
        max_active_per_phase <= 1,
        "<=1",
        max_active_per_phase,
    )

    all_have_local = all(any(not p.active_edge for p in g.phases) for g in graph_instances)
    all_have_remote = all(any(bool(p.active_edge) for p in g.phases) for g in graph_instances)
    add(
        "graph_schedule",
        "every_graph_has_local_and_remote_phases",
        all_have_local and all_have_remote,
        True,
        all_have_local and all_have_remote,
    )

    # Every graph edge must receive at least one victim request in every repeat/context copy.
    release_ok = True
    if len(releases):
        for g in graph_instances:
            for edge in g.graph_edges:
                if not (releases["active_edge"] == edge).any():
                    release_ok = False
                    break
    add(
        "graph_schedule",
        "every_true_graph_edge_generates_remote_demand",
        release_ok,
        True,
        release_ok,
    )

    # Balanced attacker scan per high-level trace plan.
    count_spans = []
    for _, sub in plan.groupby(
        ["protocol_context", "graph_mode", "graph_instance_id", "repeat_id"],
        sort=False,
    ):
        counts = sub["probe_edge"].value_counts().reindex(edges, fill_value=0)
        count_spans.append(int(counts.max() - counts.min()))
    max_span = max(count_spans) if count_spans else 0
    add(
        "attacker_probe_policy",
        "balanced_candidate_edge_scan",
        max_span <= 1,
        "max count difference <= 1",
        max_span,
    )

    train_ids = set(split.loc[split["split"] == "train", "graph_instance_id"])
    test_ids = set(split.loc[split["split"] == "test", "graph_instance_id"])
    overlap = sorted(train_ids & test_ids)
    add(
        "evaluation",
        "group_split_has_no_graph_instance_overlap",
        not overlap,
        [],
        overlap,
    )

    train_edge_counts = set(split.loc[split["split"] == "train", "edge_count"])
    test_edge_counts = set(split.loc[split["split"] == "test", "edge_count"])
    add(
        "evaluation",
        "all_graph_cardinalities_in_train_and_test",
        train_edge_counts == {2, 3, 4} and test_edge_counts == {2, 3, 4},
        {2, 3, 4},
        f"train={train_edge_counts}, test={test_edge_counts}",
    )

    def edge_coverage(ids: set[str]) -> set[str]:
        out: set[str] = set()
        for g in graph_instances:
            if g.graph_instance_id in ids:
                out.update(g.graph_edges)
        return out

    add(
        "evaluation",
        "all_candidate_edges_present_in_train_and_test",
        edge_coverage(train_ids) == set(edges) and edge_coverage(test_ids) == set(edges),
        sorted(edges),
        f"train={sorted(edge_coverage(train_ids))}, test={sorted(edge_coverage(test_ids))}",
    )

    paired_unique = (
        ground[["trace_id", "probe_index"]].drop_duplicates().shape[0]
        == features[["trace_id", "probe_index"]].drop_duplicates().shape[0]
        == len(ground)
        == len(features)
    )
    add(
        "evaluation",
        "feature_and_ground_truth_rows_pair_one_to_one",
        paired_unique,
        f"equal unique rows ({len(ground)})",
        f"features={len(features)}, ground={len(ground)}",
    )

    protocols = p27.build_protocols()
    critical = sorted({float(x.nominal_critical_latency_ns) for x in protocols.values()})
    cleanup = sorted({float(x.postcompletion_cleanup_ns) for x in protocols.values()})
    add(
        "architecture",
        "phase2_07_protocol_normalization_retained",
        critical == [150.0] and cleanup == [120.0],
        "critical=[150.0], cleanup=[120.0]",
        f"critical={critical}, cleanup={cleanup}",
    )

    requested_contexts = sorted({ctx.protocol_context for ctx, _, _ in contexts})
    observed_contexts = sorted(blackbox["protocol_context"].unique())
    add(
        "architecture",
        "requested_protocol_contexts_present",
        sorted(set(requested_contexts)) == observed_contexts,
        sorted(set(requested_contexts)),
        observed_contexts,
    )

    observed_modes = sorted(ground["graph_mode"].unique())
    add(
        "architecture",
        "requested_graph_modes_present",
        observed_modes == sorted(GRAPH_MODES),
        sorted(GRAPH_MODES),
        observed_modes,
    )

    # Localized simulator dispatch must include only the matching victim edge.
    localized = dispatch[dispatch["graph_mode"] == "edge_localized"]
    dispatch_local_ok = True
    for r in localized.itertuples(index=False):
        used = set(str(r.victim_edges_used).split("|")) if str(r.victim_edges_used) else set()
        used.discard("")
        if any(e != r.probe_edge for e in used):
            dispatch_local_ok = False
            break
    add(
        "causal_spatial_control",
        "localized_dispatch_uses_only_matching_edge_demand",
        dispatch_local_ok,
        True,
        dispatch_local_ok,
    )

    # If a candidate edge is absent from the hidden graph, edge_localized combined
    # execution contains no victim request for that route and must equal attacker-only.
    graph_lookup = {g.graph_instance_id: set(g.graph_edges) for g in graph_instances}
    loc_ground = ground[ground["graph_mode"] == "edge_localized"]
    loc = blackbox.merge(
        loc_ground[["trace_id", "probe_index", "graph_instance_id"]],
        on=["trace_id", "probe_index"],
        validate="one_to_one",
    )
    absent_mask = loc.apply(
        lambda r: r["probe_edge"] not in graph_lookup[str(r["graph_instance_id"])],
        axis=1,
    )
    max_abs_absent = float(loc.loc[absent_mask, "excess_turnaround_ns"].abs().max()) if absent_mask.any() else 0.0
    add(
        "causal_spatial_control",
        "localized_absent_graph_edges_zero_differential_timing",
        max_abs_absent <= AFFECTED_THRESHOLD_NS,
        f"<= {AFFECTED_THRESHOLD_NS} ns",
        max_abs_absent,
    )

    # Global-only dispatch must use the complete remote victim schedule for each candidate edge.
    global_dispatch = dispatch[dispatch["graph_mode"] == "global_only_control"]
    global_dispatch_ok = True
    for (pc, gid, rep), sub in global_dispatch.groupby(
        ["protocol_context", "graph_instance_id", "repeat_id"], sort=False
    ):
        counts = sub["victim_request_count_used"].to_numpy(dtype=int)
        if len(counts) != len(edges) or len(set(counts.tolist())) != 1:
            global_dispatch_ok = False
            break
    add(
        "causal_spatial_control",
        "global_only_dispatch_uses_same_remote_demand_for_every_probe_edge",
        global_dispatch_ok,
        True,
        global_dispatch_ok,
    )

    return pd.DataFrame([asdict(x) for x in rows])


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p27, phase2_source = load_phase2_07_module()
    modules = module_names(args.module_count)
    edges = candidate_edges(modules)
    if len(edges) < 3:
        raise ValueError("At least 3 candidate edges are required.")

    scan_cycle_ns = len(edges) * float(p27.ATTACKER_PERIOD_NS)
    contexts = build_graph_contexts(p27, args.protocol, GRAPH_MODES)
    instances = make_graph_instances(
        seed=args.seed,
        graph_instances=args.graph_instances,
        edges=edges,
        observation_window_ns=args.observation_window_ns,
        scan_cycle_ns=scan_cycle_ns,
    )

    print(
        f"[Phase 3.4] protocols={len(set(ctx.protocol_context for ctx,_,_ in contexts))}, "
        f"modes={len(GRAPH_MODES)}, graphs={len(instances)}, "
        f"candidate_edges={len(edges)}, repeats={args.repeats_per_instance}"
    )
    print(f"[Phase 3.4] scan_cycle_ns={scan_cycle_ns:.1f}")
    print(f"[Phase 3.4] Reusing Phase-2.7 simulator: {phase2_source}")

    blackbox, ground, plan, releases, dispatch = run_dataset(
        p27,
        contexts=contexts,
        graph_instances=instances,
        edges=edges,
        repeats_per_instance=args.repeats_per_instance,
        seed=args.seed,
        observation_window_ns=args.observation_window_ns,
    )

    features = build_causal_probe_features(
        blackbox,
        edges=edges,
        history=args.history,
    )
    feature_columns = model_feature_columns(edges)

    # Attach evaluator metadata only after feature extraction; these columns are
    # never part of feature_columns.
    analysis = features.merge(
        ground,
        on=["trace_id", "protocol_context", "probe_index", "probe_edge"],
        validate="one_to_one",
    )

    split = build_group_split(
        instances,
        edges=edges,
        seed=args.seed,
        test_size=args.test_size,
    )

    state_labels = ("no_remote",) + edges
    state_metrics, state_predictions, confusion = evaluate_state_models(
        analysis,
        split,
        feature_columns=feature_columns,
        state_labels=state_labels,
        seed=args.seed,
        rf_trees=args.rf_trees,
        scan_cycle_ns=scan_cycle_ns,
    )

    thresholds = calibrate_presence_thresholds(
        state_predictions,
        instances=instances,
        edges=edges,
        modules=modules,
    )
    graph_detail, graph_summary = reconstruct_graphs(
        state_predictions,
        thresholds,
        instances=instances,
        edges=edges,
        modules=modules,
    )

    boundary = boundary_metrics(
        state_predictions,
        instances,
        tolerance_ns=args.boundary_tolerance_scans * scan_cycle_ns,
    )

    edge_matrix, contrast = build_edge_signal_matrix(blackbox, ground, split)
    protocol_summary = protocol_comparison_summary(
        state_metrics, graph_summary, boundary, contrast
    )

    validation = build_validations(
        p27=p27,
        blackbox=blackbox,
        ground=ground,
        features=features,
        split=split,
        graph_instances=instances,
        edges=edges,
        contexts=contexts,
        plan=plan,
        releases=releases,
        dispatch=dispatch,
        feature_columns=feature_columns,
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

    # ------------------------------------------------------------------
    # Evaluator tables
    # ------------------------------------------------------------------
    graph_table = pd.DataFrame(
        [
            {
                "graph_instance_id": g.graph_instance_id,
                "instance_index": g.instance_index,
                "edge_count": g.edge_count,
                "graph_edges": "|".join(g.graph_edges),
                "graph_signature": g.graph_signature,
                "phase_count": len(g.phases),
            }
            for g in instances
        ]
    )
    phase_table = pd.DataFrame(
        [
            {
                "graph_instance_id": g.graph_instance_id,
                "phase_index": p.phase_index,
                "phase_label": p.phase_label,
                "active_edge": p.active_edge,
                "remote_profile": p.remote_profile,
                "start_ns": p.start_ns,
                "end_ns": p.end_ns,
                "duration_ns": p.end_ns - p.start_ns,
            }
            for g in instances
            for p in g.phases
        ]
    )
    context_table = pd.DataFrame(
        [asdict(ctx) for ctx, _, _ in contexts]
    ).drop_duplicates()
    target_table = pd.DataFrame(
        [
            {
                "candidate_edge_index": i,
                "candidate_probe_edge": e,
                "endpoint_a": split_edge(e)[0],
                "endpoint_b": split_edge(e)[1],
                "attacker_knows_probe_edge": True,
                "victim_active_edge_hidden_from_attacker": True,
            }
            for i, e in enumerate(edges)
        ]
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    blackbox.to_csv(output_dir / "phase3_04_attacker_visible_trace.csv", index=False)
    ground.to_csv(output_dir / "phase3_04_evaluator_probe_ground_truth.csv", index=False)
    features.to_csv(output_dir / "phase3_04_probe_features.csv", index=False)
    graph_table.to_csv(output_dir / "phase3_04_graph_instance_table.csv", index=False)
    phase_table.to_csv(output_dir / "phase3_04_victim_phase_schedule.csv", index=False)
    releases.to_csv(output_dir / "phase3_04_victim_release_schedule.csv", index=False)
    plan.to_csv(output_dir / "phase3_04_probe_plan_evaluator.csv", index=False)
    dispatch.to_csv(output_dir / "phase3_04_simulation_dispatch_evaluator.csv", index=False)
    context_table.to_csv(output_dir / "phase3_04_graph_context_table.csv", index=False)
    target_table.to_csv(output_dir / "phase3_04_candidate_edge_table.csv", index=False)
    split.to_csv(output_dir / "phase3_04_group_split.csv", index=False)

    state_metrics.to_csv(output_dir / "phase3_04_state_segmentation_metrics.csv", index=False)
    state_predictions.to_csv(output_dir / "phase3_04_state_predictions.csv", index=False)
    confusion.to_csv(output_dir / "phase3_04_confusion_matrix.csv", index=False)

    thresholds.to_csv(output_dir / "phase3_04_graph_presence_thresholds.csv", index=False)
    graph_detail.to_csv(output_dir / "phase3_04_graph_reconstruction_predictions.csv", index=False)
    graph_summary.to_csv(output_dir / "phase3_04_graph_reconstruction_summary.csv", index=False)
    boundary.to_csv(output_dir / "phase3_04_boundary_metrics.csv", index=False)
    edge_matrix.to_csv(output_dir / "phase3_04_edge_signal_matrix.csv", index=False)
    contrast.to_csv(output_dir / "phase3_04_diagonal_contrast_summary.csv", index=False)
    protocol_summary.to_csv(output_dir / "phase3_04_protocol_comparison_summary.csv", index=False)

    validation.to_csv(output_dir / "phase3_04_validation_assertions.csv", index=False)
    validation_summary.to_csv(output_dir / "phase3_04_validation_summary.csv", index=False)

    manifest = {
        "experiment": "phase3_04_intermodule_graph_reconstruction",
        "seed": args.seed,
        "phase2_07_source": str(phase2_source),
        "output_dir": str(output_dir),
        "protocol_contexts": sorted({ctx.protocol_context for ctx, _, _ in contexts}),
        "graph_modes": list(GRAPH_MODES),
        "candidate_modules": list(modules),
        "candidate_edges": list(edges),
        "graph_instance_count": len(instances),
        "graph_edge_cardinalities": [2, 3, 4],
        "repeats_per_instance": args.repeats_per_instance,
        "observation_window_ns": args.observation_window_ns,
        "attacker_first_release_ns": float(p27.ATTACKER_FIRST_RELEASE_NS),
        "attacker_period_ns": float(p27.ATTACKER_PERIOD_NS),
        "scan_cycle_ns": scan_cycle_ns,
        "causal_history_per_edge": args.history,
        "test_size": args.test_size,
        "rf_trees": args.rf_trees,
        "boundary_tolerance_scans": args.boundary_tolerance_scans,
        "boundary_tolerance_ns": args.boundary_tolerance_scans * scan_cycle_ns,
        "trace_count": int(blackbox["trace_id"].nunique()),
        "probe_row_count": int(len(blackbox)),
        "training_graph_instance_count": int((split["split"] == "train").sum()),
        "test_graph_instance_count": int((split["split"] == "test").sum()),
        "model_feature_columns": feature_columns,
        "attacker_visible_columns": list(ATTACKER_VISIBLE_COLUMNS),
        "validation_assertions": len(validation),
        "validation_passed": int(validation["passed"].sum()),
        "all_validation_passed": bool(validation["passed"].all()),
        "notes": [
            "Primary state models use only current/past attacker timing observations.",
            "Absolute probe time/index is excluded from primary model features.",
            "Probe edge is attacker-known because the attacker chooses its own candidate path.",
            "Hidden graph, active edge, phase labels, victim requests, scenario ids, and evaluator state are not attacker-visible.",
            "Train/test splitting is grouped by hidden graph instance.",
            "edge_localized preserves per-edge protocol state across the complete trace.",
            "global_only_control preserves common timing activity but removes target-selective edge mapping.",
            "At most one victim communication edge is active per phase in this first graph-reconstruction experiment.",
            "Architecture timings are controlled simulation parameters, not vendor measurements.",
        ],
    }
    with open(output_dir / "phase3_04_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n[Phase 3.4] Validation")
    print(validation_summary.to_string(index=False))
    print("\n[Phase 3.4] Protocol / graph-mode summary")
    print(protocol_summary.to_string(index=False))
    print(f"\n[Phase 3.4] Wrote outputs to: {output_dir}")

    if args.fail_on_validation_error and not bool(validation["passed"].all()):
        failed = validation[~validation["passed"]]
        raise RuntimeError(
            "Phase 3.4 validation failed:\n" + failed.to_string(index=False)
        )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.4 — Intermodule Communication-Graph Reconstruction"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--graph-instances",
        type=int,
        default=DEFAULT_GRAPH_INSTANCES,
        help="Distinct hidden graph instances. Default: 30.",
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
        help="Default 40 us so six candidate edges receive 16 probes each.",
    )
    parser.add_argument(
        "--module-count",
        type=int,
        default=DEFAULT_MODULE_COUNT,
        help="Candidate victim modules. Default 4 -> six undirected edges.",
    )
    parser.add_argument(
        "--protocol",
        choices=("direct", "entangled", "both"),
        default="both",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
    )
    parser.add_argument("--rf-trees", type=int, default=DEFAULT_RF_TREES)
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
        help="Number of past observations retained per candidate edge.",
    )
    parser.add_argument(
        "--boundary-tolerance-scans",
        type=float,
        default=1.0,
        help="Boundary matching tolerance measured in complete edge-scan cycles.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.graph_instances < 9:
        raise ValueError("--graph-instances must be >= 9")
    if args.repeats_per_instance < 1:
        raise ValueError("--repeats-per-instance must be >= 1")
    if args.module_count != 4:
        raise ValueError(
            "Phase 3.4 currently fixes --module-count=4 because graph cardinality "
            "generation and seven-way state labels are designed for six candidate edges."
        )
    if not 0.15 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.15 and 0.5")
    if args.history < 1:
        raise ValueError("--history must be >= 1")
    if args.observation_window_ns < 12_000:
        raise ValueError("--observation-window-ns must be at least 12000 ns")
    if args.boundary_tolerance_scans <= 0:
        raise ValueError("--boundary-tolerance-scans must be > 0")
    run_experiment(args)


if __name__ == "__main__":
    main()
