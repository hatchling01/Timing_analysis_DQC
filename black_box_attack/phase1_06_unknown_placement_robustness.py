#!/usr/bin/env python3
"""
phase1_06_unknown_placement_robustness.py

Experiment 1.6 — Unknown placement and allocation state.

Research question
-----------------
Can a black-box attacker still detect victim activity, infer whether it shares
an exclusive resource, and choose a better probe when physical placement and
runtime policy information are hidden?

Dependencies
------------
Keep this file beside:
    phase1_01_job_module_allocation.py
    phase1_04_remote_operation_schedulers.py

Run
---
    python phase1_06_unknown_placement_robustness.py

No terminal arguments are required. All run controls are defined below.

Experiment design
-----------------
For each physical trial, the script:

1. Selects a module count, tenant count, allocator policy, and scheduler policy.
2. Randomizes victim, attacker, and background placement through the Phase 1.1
   allocator.
3. Gives the attacker three logical communication-probe choices across three
   allocated attacker partitions.
4. Executes attacker-only, victim-only, and victim-present controls.
5. Evaluates four probe strategies:
       - fixed one-shot probe;
       - one-shot candidate-set cycling;
       - knowledge-guided one-shot probe;
       - adaptive explore-then-exploit probing.
6. Reuses the same physical observations to create eight attacker knowledge
   views spanning placement-known/unknown and policy-known/unknown states.
7. Evaluates:
       - victim-presence detection;
       - resource-sharing / placement-class inference;
       - probes required for first observable change;
       - false positives under no endpoint/link sharing;
       - adaptive probe-selection accuracy;
       - victim disruption during exploration and exploitation.

Important controls
------------------
- The allocator and scheduler are physically randomized before any knowledge
  is hidden from the attacker.
- Attacker-only and victim-present runs use identical placement and scheduler
  configuration.
- Randomized arbitration uses the same scenario seed in paired controls.
- Knowledge levels alter only the attacker-visible feature set and the
  knowledge-guided probe choice; they never alter the hidden physical state.
- The adaptive strategy uses 12 exploration probes (four per candidate pair)
  and then uses the selected pair for the remaining 36 probes.

Primary outputs
---------------
blackbox_window_results/phase1_06_unknown_placement_robustness/
    unknown_placement_physical_trial_summary.csv
    unknown_placement_strategy_trial_summary.csv
    unknown_placement_samples.csv
    unknown_placement_features.csv
    unknown_placement_detection_metrics.csv
    unknown_placement_detection_predictions.csv
    unknown_placement_sharing_metrics.csv
    unknown_placement_sharing_predictions.csv
    unknown_placement_paired_rule_summary.csv
    unknown_placement_probe_selection_summary.csv
    unknown_placement_no_sharing_false_positive.csv
    unknown_placement_knowledge_summary.csv
    unknown_placement_policy_summary.csv
    unknown_placement_allocator_summary.csv
    unknown_placement_scheduler_summary.csv
    unknown_placement_detection_accuracy.png
    unknown_placement_sharing_accuracy.png
    unknown_placement_probe_efficiency.png

Optional detailed output
------------------------
Set SAVE_REQUEST_LEVEL_RESULTS = True below to save the compressed attacker
request logs. It is disabled by default because the complete experiment can
produce a large file.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import phase1_01_job_module_allocation as p1
import phase1_04_remote_operation_schedulers as p4


# =============================================================================
# Integrated run controls
# =============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_06_unknown_placement_robustness"
)

RUN_QUICK_VALIDATION = False
TRIALS_PER_MODULE_TENANCY_WORKLOAD = 8
MAX_PHYSICAL_TRIALS: int | None = None
SAVE_REQUEST_LEVEL_RESULTS = False

GLOBAL_SEED = 20260731

VICTIM_QASMS = list(p1.VICTIM_QASMS)

NUM_MODULE_OPTIONS = [8, 12, 16]
TENANT_COUNT_OPTIONS = [2, 4, 8]
VICTIM_MODULE_REQUEST_OPTIONS = [2, 3]
ATTACKER_MODULES_REQUESTED = 3
BACKGROUND_MODULES_REQUESTED = 2

ALLOCATION_POLICIES = [
    "uniform_random",
    "load_balanced",
    "communication_minimizing",
    "communication_consolidating",
    "endpoint_separation",
    "first_fit_admission",
]

SCHEDULER_POLICIES = [
    "first_come_first_served",
    "round_robin_tenants",
    "endpoint_aware",
    "randomized_arbitration",
]

# Keep resource provisioning fixed so that Phase 1.6 isolates uncertainty rather
# than repeating the Phase 1.3 and 1.4 capacity sweeps.
QUEUE_DEPTH = 128
HUB_CAPACITY = 4
LINK_CAPACITY = 1
COMMUNICATION_QUBITS_PER_MODULE = 2
PRIORITY_PROFILE = "equal"
DECISION_INTERVAL_NS = 20
LOOKAHEAD_WINDOW_NS = 500
PREEMPTION_ALLOWED = False
EPR_PREFETCH_ENABLED = False

TOTAL_PROBES = 48
EXPLORATION_PROBES = 12
PROBES_PER_CANDIDATE_DURING_EXPLORATION = 4
TIMING_CHANGE_THRESHOLD_NS = 0.0
NUM_TEMPORAL_BINS = 8
CLASSIFICATION_FOLDS = 5

PROBE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (0, 2),
)

PROBE_PAIR_NAMES = {
    (0, 1): "pair_01",
    (1, 2): "pair_12",
    (0, 2): "pair_02",
}

PROBE_STRATEGIES = [
    "one_shot_fixed",
    "one_shot_candidate_cycle",
    "one_shot_knowledge_guided",
    "adaptive_explore_exploit",
]

ALLOCATOR_FEATURE_NAMES = [
    f"k_allocator_{policy}"
    for policy in ALLOCATION_POLICIES
]

SCHEDULER_FEATURE_NAMES = [
    f"k_scheduler_{policy}"
    for policy in SCHEDULER_POLICIES
]


# =============================================================================
# Knowledge profiles
# =============================================================================

@dataclass(frozen=True)
class KnowledgeProfile:
    name: str
    own_physical_known: bool
    victim_physical_known: bool
    logical_identifiers_known: bool
    allocator_policy_known: bool
    scheduler_policy_known: bool


KNOWLEDGE_PROFILES = [
    KnowledgeProfile(
        name="exact_both_policy_known",
        own_physical_known=True,
        victim_physical_known=True,
        logical_identifiers_known=True,
        allocator_policy_known=True,
        scheduler_policy_known=True,
    ),
    KnowledgeProfile(
        name="exact_both_policy_unknown",
        own_physical_known=True,
        victim_physical_known=True,
        logical_identifiers_known=True,
        allocator_policy_known=False,
        scheduler_policy_known=False,
    ),
    KnowledgeProfile(
        name="attacker_known_victim_unknown_policy_known",
        own_physical_known=True,
        victim_physical_known=False,
        logical_identifiers_known=True,
        allocator_policy_known=True,
        scheduler_policy_known=True,
    ),
    KnowledgeProfile(
        name="attacker_known_victim_unknown_policy_unknown",
        own_physical_known=True,
        victim_physical_known=False,
        logical_identifiers_known=True,
        allocator_policy_known=False,
        scheduler_policy_known=False,
    ),
    KnowledgeProfile(
        name="logical_identifiers_only_policy_known",
        own_physical_known=False,
        victim_physical_known=False,
        logical_identifiers_known=True,
        allocator_policy_known=True,
        scheduler_policy_known=True,
    ),
    KnowledgeProfile(
        name="logical_identifiers_only_policy_unknown",
        own_physical_known=False,
        victim_physical_known=False,
        logical_identifiers_known=True,
        allocator_policy_known=False,
        scheduler_policy_known=False,
    ),
    KnowledgeProfile(
        name="both_physical_unknown_policy_known",
        own_physical_known=False,
        victim_physical_known=False,
        logical_identifiers_known=False,
        allocator_policy_known=True,
        scheduler_policy_known=True,
    ),
    KnowledgeProfile(
        name="both_physical_unknown_policy_unknown",
        own_physical_known=False,
        victim_physical_known=False,
        logical_identifiers_known=False,
        allocator_policy_known=False,
        scheduler_policy_known=False,
    ),
]

KNOWLEDGE_ORDER = [profile.name for profile in KNOWLEDGE_PROFILES]


# =============================================================================
# General utilities
# =============================================================================


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return (
        GLOBAL_SEED
        + zlib.crc32(text.encode("utf-8"))
    ) & 0x7FFFFFFF



def safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else 0.0



def safe_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std()) if len(array) else 0.0



def canonical_pair(first: int, second: int) -> tuple[int, int]:
    return (min(first, second), max(first, second))



def pair_name(pair: tuple[int, int]) -> str:
    return PROBE_PAIR_NAMES[canonical_pair(*pair)]



def parse_module_string(value: str) -> tuple[int, ...]:
    if not value:
        return tuple()
    return tuple(sorted(int(item) for item in str(value).split(",") if item != ""))



def role_completion(
    request_log: pd.DataFrame,
    role: str,
) -> tuple[bool, float]:
    frame = request_log[request_log["role"] == role]
    if frame.empty:
        return True, 0.0
    success = bool(frame["completed"].all())
    if not success:
        return False, float("nan")
    return True, float(frame["completion_time_ns"].max())


# =============================================================================
# Attacker probe jobs
# =============================================================================


def attacker_partition_map() -> dict[int, int]:
    # Two logical qubits per attacker partition.
    return {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
    }



def representative_qubit(partition: int) -> int:
    return 2 * partition



def probe_pair_sequence(
    mode: str,
    selected_pair: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    if mode == "candidate_cycle":
        return [
            PROBE_PAIRS[index % len(PROBE_PAIRS)]
            for index in range(TOTAL_PROBES)
        ]

    if mode == "adaptive":
        if selected_pair is None:
            raise ValueError("Adaptive probe schedule requires selected_pair.")
        sequence = [
            PROBE_PAIRS[index % len(PROBE_PAIRS)]
            for index in range(EXPLORATION_PROBES)
        ]
        sequence.extend(
            [selected_pair] * (TOTAL_PROBES - EXPLORATION_PROBES)
        )
        return sequence

    if mode == "single_pair":
        if selected_pair is None:
            raise ValueError("single_pair schedule requires selected_pair.")
        return [selected_pair] * TOTAL_PROBES

    raise ValueError(f"Unknown probe schedule mode: {mode}")



def build_attacker_job(
    *,
    schedule_mode: str,
    selected_pair: tuple[int, int] | None = None,
) -> p1.JobSpec:
    partition_of = attacker_partition_map()

    # Keep the allocation graph identical for all probe schedules. Otherwise,
    # changing the probe strategy could itself change the allocated modules.
    partition_graph = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (0, 2): 1.0,
    }

    pairs = probe_pair_sequence(
        schedule_mode,
        selected_pair,
    )

    events: list[p1.LogicalEvent] = []
    for probe_id, (left, right) in enumerate(pairs):
        events.append(
            p1.LogicalEvent(
                event_id=probe_id,
                release_offset_ns=(
                    p1.PROBE_REMOTE_OFFSET_NS
                    + probe_id * p1.PROBE_PERIOD_NS
                ),
                op_name=f"probe3_{pair_name((left, right))}",
                qubits=(
                    representative_qubit(left),
                    representative_qubit(right),
                ),
            )
        )

    return p1.JobSpec(
        tenant_id="attacker",
        role="attacker",
        logical_qubits=6,
        modules_requested=ATTACKER_MODULES_REQUESTED,
        partition_of_qubit=partition_of,
        partition_graph=partition_graph,
        logical_events=events,
        start_time_ns=p1.ATTACKER_START_NS,
    )



def attacker_allocation_job() -> p1.JobSpec:
    # Allocation is based on the same complete three-partition communication
    # graph but does not depend on the later selected probe sequence.
    return build_attacker_job(
        schedule_mode="candidate_cycle"
    )


# =============================================================================
# Dynamic physical placement
# =============================================================================


def build_background_jobs(
    tenant_count: int,
    seed: int,
) -> list[p1.JobSpec]:
    return [
        p1.build_background_job(
            tenant_index=index,
            modules_requested=BACKGROUND_MODULES_REQUESTED,
            rng=random.Random(
                stable_seed(seed, "background", index)
            ),
        )
        for index in range(max(0, tenant_count - 2))
    ]



def allocate_physical_trial(
    *,
    victim_job: p1.JobSpec,
    attacker_job: p1.JobSpec,
    background_jobs: Sequence[p1.JobSpec],
    allocation_policy: str,
    num_modules: int,
    seed: int,
) -> dict[str, p1.Allocation]:
    jobs = [victim_job, attacker_job, *background_jobs]
    return p1.allocate_jobs(
        jobs,
        allocation_policy,
        num_modules,
        random.Random(seed),
    )


# =============================================================================
# Scheduler request construction for arbitrary module counts
# =============================================================================


def touched_partitions(
    job: p1.JobSpec,
    event: p1.LogicalEvent,
) -> list[int]:
    return sorted(
        {
            job.partition_of_qubit[qubit]
            for qubit in event.qubits
            if qubit in job.partition_of_qubit
        }
    )



def build_scheduler_requests(
    jobs: Sequence[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    *,
    num_modules: int,
) -> list[p4.SchedulerRequest]:
    weights = p4.PRIORITY_WEIGHTS[PRIORITY_PROFILE]
    provisional: list[p4.SchedulerRequest] = []
    request_id = 0

    for job in jobs:
        allocation = allocations[job.tenant_id]
        if not allocation.accepted:
            continue

        last_remote_by_qubit: dict[int, int] = {}
        layer_by_request: dict[int, int] = {}
        tenant_request_ids: list[int] = []

        for event in job.logical_events:
            partitions = touched_partitions(job, event)
            if len(partitions) < 2:
                continue

            pair_index = 0
            for left_partition, right_partition in itertools.combinations(
                partitions,
                2,
            ):
                source = allocation.partition_to_module[left_partition]
                target = allocation.partition_to_module[right_partition]
                if source == target:
                    continue

                logical_qubits = tuple(sorted(set(event.qubits)))
                dependencies = sorted(
                    {
                        last_remote_by_qubit[qubit]
                        for qubit in logical_qubits
                        if qubit in last_remote_by_qubit
                    }
                )
                layer = (
                    0
                    if not dependencies
                    else 1
                    + max(layer_by_request[dependency] for dependency in dependencies)
                )

                route = p4.route_links(
                    source,
                    target,
                    num_modules=num_modules,
                )
                release_time_ns = job.start_time_ns + event.release_offset_ns
                deadline_ns = (
                    release_time_ns
                    + p4.ROLE_DEADLINE_SLACK_NS[job.role]
                )

                provisional.append(
                    p4.SchedulerRequest(
                        request_id=request_id,
                        operation_uid=(
                            f"{job.tenant_id}:"
                            f"{event.event_id}:"
                            f"{pair_index}"
                        ),
                        tenant_id=job.tenant_id,
                        role=job.role,
                        logical_event_id=event.event_id,
                        circuit_order=len(tenant_request_ids),
                        release_time_ns=release_time_ns,
                        source_module=source,
                        target_module=target,
                        route_links=route,
                        base_service_time_ns=p4.base_service_time_ns(route),
                        dependencies=tuple(dependencies),
                        layer=layer,
                        deadline_ns=deadline_ns,
                        priority_weight=weights[job.role],
                        criticality=0,
                    )
                )

                tenant_request_ids.append(request_id)
                layer_by_request[request_id] = layer
                for qubit in logical_qubits:
                    last_remote_by_qubit[qubit] = request_id

                request_id += 1
                pair_index += 1

    successors: dict[int, list[int]] = defaultdict(list)
    for request in provisional:
        for dependency in request.dependencies:
            successors[dependency].append(request.request_id)

    criticality_cache: dict[int, int] = {}

    def criticality(request_id_value: int) -> int:
        if request_id_value in criticality_cache:
            return criticality_cache[request_id_value]
        value = len(successors.get(request_id_value, []))
        value += sum(
            criticality(child)
            for child in successors.get(request_id_value, [])
        )
        criticality_cache[request_id_value] = value
        return value

    requests = [
        p4.SchedulerRequest(
            **{
                **request.__dict__,
                "criticality": criticality(request.request_id),
            }
        )
        for request in provisional
    ]

    role_order = {
        "victim": 0,
        "background": 1,
        "attacker": 2,
    }
    return sorted(
        requests,
        key=lambda request: (
            request.release_time_ns,
            role_order[request.role],
            request.tenant_id,
            request.circuit_order,
            request.request_id,
        ),
    )



def scheduler_configuration(
    scheduler_policy: str,
    tenant_count: int,
) -> p4.SchedulerConfiguration:
    return p4.SchedulerConfiguration(
        subexperiment="unknown_placement_robustness",
        scheduler_policy=scheduler_policy,
        queue_depth=QUEUE_DEPTH,
        tenant_count=tenant_count,
        hub_capacity=HUB_CAPACITY,
        link_capacity=LINK_CAPACITY,
        communication_qubits_per_module=COMMUNICATION_QUBITS_PER_MODULE,
        priority_profile=PRIORITY_PROFILE,
        decision_interval_ns=DECISION_INTERVAL_NS,
        lookahead_window_ns=LOOKAHEAD_WINDOW_NS,
        preemption_allowed=PREEMPTION_ALLOWED,
        epr_prefetch_enabled=EPR_PREFETCH_ENABLED,
    )



def execute_schedule(
    *,
    jobs: Sequence[p1.JobSpec],
    allocations: dict[str, p1.Allocation],
    included_roles: set[str],
    num_modules: int,
    scheduler_policy: str,
    tenant_count: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    requests = build_scheduler_requests(
        jobs,
        allocations,
        num_modules=num_modules,
    )
    configuration = scheduler_configuration(
        scheduler_policy,
        tenant_count,
    )
    return p4.execute_scenario(
        requests,
        included_roles,
        configuration,
        seed,
    )


# =============================================================================
# Physical sharing labels and probe overlap scores
# =============================================================================


def job_remote_module_pairs(
    job: p1.JobSpec,
    allocation: p1.Allocation,
) -> set[tuple[int, int]]:
    if not allocation.accepted:
        return set()
    pairs: set[tuple[int, int]] = set()
    for left_partition, right_partition in job.partition_graph:
        source = allocation.partition_to_module[left_partition]
        target = allocation.partition_to_module[right_partition]
        if source != target:
            pairs.add(canonical_pair(source, target))
    return pairs



def route_set_for_pairs(
    pairs: Iterable[tuple[int, int]],
    num_modules: int,
) -> set[tuple[int, int]]:
    links: set[tuple[int, int]] = set()
    for source, target in pairs:
        links.update(
            p4.route_links(
                source,
                target,
                num_modules=num_modules,
            )
        )
    return links



def physical_sharing_state(
    *,
    victim_job: p1.JobSpec,
    attacker_job: p1.JobSpec,
    victim_allocation: p1.Allocation,
    attacker_allocation: p1.Allocation,
    num_modules: int,
) -> dict[str, Any]:
    if not victim_allocation.accepted or not attacker_allocation.accepted:
        return {
            "placement_class": "rejected",
            "module_overlap": False,
            "endpoint_overlap": False,
            "link_overlap": False,
            "hub_shared": False,
            "shared_module_count": 0,
            "shared_link_count": 0,
        }

    victim_modules = set(victim_allocation.assigned_modules)
    attacker_modules = set(attacker_allocation.assigned_modules)
    victim_endpoint_modules = p1.remote_endpoint_modules(
        victim_job,
        victim_allocation,
    )
    attacker_endpoint_modules = p1.remote_endpoint_modules(
        attacker_job,
        attacker_allocation,
    )

    victim_pairs = job_remote_module_pairs(victim_job, victim_allocation)
    attacker_pairs = job_remote_module_pairs(attacker_job, attacker_allocation)
    victim_links = route_set_for_pairs(victim_pairs, num_modules)
    attacker_links = route_set_for_pairs(attacker_pairs, num_modules)

    module_overlap_set = victim_modules & attacker_modules
    endpoint_overlap_set = victim_endpoint_modules & attacker_endpoint_modules
    shared_links = victim_links & attacker_links

    if endpoint_overlap_set:
        placement_class = "endpoint_overlap"
    elif shared_links:
        placement_class = "link_overlap_only"
    else:
        placement_class = "hub_only"

    return {
        "placement_class": placement_class,
        "module_overlap": bool(module_overlap_set),
        "endpoint_overlap": bool(endpoint_overlap_set),
        "link_overlap": bool(shared_links),
        "hub_shared": True,
        "shared_module_count": len(module_overlap_set),
        "shared_link_count": len(shared_links),
    }



def candidate_physical_pair(
    pair: tuple[int, int],
    allocation: p1.Allocation,
) -> tuple[int, int]:
    left, right = pair
    return canonical_pair(
        allocation.partition_to_module[left],
        allocation.partition_to_module[right],
    )



def candidate_overlap_score(
    *,
    candidate_pair: tuple[int, int],
    victim_job: p1.JobSpec,
    victim_allocation: p1.Allocation,
    attacker_allocation: p1.Allocation,
    num_modules: int,
) -> float:
    physical_pair = candidate_physical_pair(
        candidate_pair,
        attacker_allocation,
    )
    candidate_endpoints = set(physical_pair)
    candidate_links = set(
        p4.route_links(
            *physical_pair,
            num_modules=num_modules,
        )
    )

    victim_endpoints = p1.remote_endpoint_modules(
        victim_job,
        victim_allocation,
    )
    victim_links = route_set_for_pairs(
        job_remote_module_pairs(victim_job, victim_allocation),
        num_modules,
    )

    endpoint_score = 100.0 * len(candidate_endpoints & victim_endpoints)
    link_score = 10.0 * len(candidate_links & victim_links)
    route_exposure = float(len(candidate_links))
    return endpoint_score + link_score + route_exposure



def oracle_best_pair(
    *,
    victim_job: p1.JobSpec,
    victim_allocation: p1.Allocation,
    attacker_allocation: p1.Allocation,
    num_modules: int,
) -> tuple[int, int]:
    return max(
        PROBE_PAIRS,
        key=lambda pair: (
            candidate_overlap_score(
                candidate_pair=pair,
                victim_job=victim_job,
                victim_allocation=victim_allocation,
                attacker_allocation=attacker_allocation,
                num_modules=num_modules,
            ),
            -PROBE_PAIRS.index(pair),
        ),
    )


# =============================================================================
# Attacker comparison and probe selection
# =============================================================================


def attacker_frame(request_log: pd.DataFrame) -> pd.DataFrame:
    frame = request_log[request_log["role"] == "attacker"].copy()
    return frame.sort_values("logical_event_id").reset_index(drop=True)



def compare_attacker_logs(
    attacker_only_log: pd.DataFrame,
    victim_present_log: pd.DataFrame,
) -> pd.DataFrame:
    return p4.compare_attacker_runs(
        attacker_only_log,
        victim_present_log,
    )



def exploration_selected_pair(
    comparison: pd.DataFrame,
) -> tuple[tuple[int, int], dict[str, float]]:
    exploration = comparison[
        comparison["logical_event_id"] < EXPLORATION_PROBES
    ].copy()

    scores: dict[str, float] = {}
    for pair_index, pair in enumerate(PROBE_PAIRS):
        pair_rows = exploration[
            exploration["logical_event_id"] % len(PROBE_PAIRS) == pair_index
        ]
        absolute_change = pair_rows[
            "absolute_turnaround_change_ns"
        ].fillna(0.0)
        transitions = (
            pair_rows["failure_transition_observed"].fillna(False)
            | pair_rows["recovery_transition_observed"].fillna(False)
        )
        score = float(absolute_change.mean())
        score += 10_000.0 * float(transitions.mean())
        scores[pair_name(pair)] = score

    selected = max(
        PROBE_PAIRS,
        key=lambda pair: (
            scores[pair_name(pair)],
            -PROBE_PAIRS.index(pair),
        ),
    )
    return selected, scores



def knowledge_guided_pair(
    *,
    profile: KnowledgeProfile,
    allocation_policy: str,
    victim_job: p1.JobSpec,
    victim_allocation: p1.Allocation,
    attacker_allocation: p1.Allocation,
    num_modules: int,
) -> tuple[int, int]:
    # Exact physical knowledge permits direct targeting of the pair with the
    # strongest expected endpoint/path overlap.
    if profile.own_physical_known and profile.victim_physical_known:
        return oracle_best_pair(
            victim_job=victim_job,
            victim_allocation=victim_allocation,
            attacker_allocation=attacker_allocation,
            num_modules=num_modules,
        )

    # With own placement known but victim placement hidden, a longer route
    # intersects more potential victim routes. Consolidating allocators add a
    # bias toward low-index modules because they fill existing modules first.
    if profile.own_physical_known:
        def prior_score(pair: tuple[int, int]) -> tuple[float, float, int]:
            physical = candidate_physical_pair(pair, attacker_allocation)
            route = p4.route_links(
                *physical,
                num_modules=num_modules,
            )
            route_score = float(len(route))
            consolidation_score = 0.0
            if (
                profile.allocator_policy_known
                and allocation_policy in {
                    "communication_consolidating",
                    "first_fit_admission",
                }
            ):
                consolidation_score = -float(sum(physical))
            return (
                consolidation_score,
                route_score,
                -PROBE_PAIRS.index(pair),
            )

        return max(PROBE_PAIRS, key=prior_score)

    # Logical-only and fully unknown attackers cannot distinguish physical
    # routes before exploration, so they use the canonical selected probe.
    return PROBE_PAIRS[0]



def first_observable_probe(comparison: pd.DataFrame) -> int:
    observed = (
        comparison["any_observable_change"].fillna(False)
    ).to_numpy(dtype=bool)
    indices = np.flatnonzero(observed)
    return int(indices[0] + 1) if len(indices) else TOTAL_PROBES + 1



def comparison_metrics(comparison: pd.DataFrame) -> dict[str, Any]:
    signed = comparison["signed_turnaround_change_ns"].to_numpy(dtype=float)
    absolute = comparison["absolute_turnaround_change_ns"].to_numpy(dtype=float)
    finite_signed = signed[np.isfinite(signed)]
    finite_absolute = absolute[np.isfinite(absolute)]

    return {
        "probe_count": int(len(comparison)),
        "mean_signed_timing_change_ns": (
            float(finite_signed.mean()) if len(finite_signed) else 0.0
        ),
        "mean_absolute_timing_change_ns": (
            float(finite_absolute.mean()) if len(finite_absolute) else 0.0
        ),
        "max_absolute_timing_change_ns": (
            float(finite_absolute.max()) if len(finite_absolute) else 0.0
        ),
        "positive_delay_fraction": float(
            comparison["positive_delay_observed"].mean()
        ),
        "negative_speedup_fraction": float(
            comparison["negative_speedup_observed"].mean()
        ),
        "failure_transition_fraction": float(
            comparison["failure_transition_observed"].mean()
        ),
        "observable_change_fraction": float(
            comparison["any_observable_change"].mean()
        ),
        "paired_rule_detected": bool(
            comparison["any_observable_change"].any()
        ),
        "probes_to_first_observable_change": first_observable_probe(comparison),
    }


# =============================================================================
# Absolute black-box features
# =============================================================================


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest



def safe_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    first = values[:-lag]
    second = values[lag:]
    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    value = np.corrcoef(first, second)[0, 1]
    return float(value) if np.isfinite(value) else 0.0



def absolute_timing_features(request_log: pd.DataFrame) -> dict[str, float]:
    frame = attacker_frame(request_log)
    if frame.empty:
        return {"f_missing_attacker_trace": 1.0}

    completed = frame["completed"].fillna(False).to_numpy(dtype=bool)
    rejected = frame["rejected"].fillna(False).to_numpy(dtype=bool)
    turnaround = frame["turnaround_time_ns"].fillna(0.0).to_numpy(dtype=float)
    queue = frame["queue_delay_ns"].fillna(0.0).to_numpy(dtype=float)
    selection_count = frame["scheduler_selection_count"].fillna(0.0).to_numpy(dtype=float)
    acquisition_failures = frame[
        "resource_acquisition_failure_count"
    ].fillna(0.0).to_numpy(dtype=float)

    features: dict[str, float] = {
        "f_missing_attacker_trace": 0.0,
        "f_completed_fraction": float(completed.mean()),
        "f_rejected_fraction": float(rejected.mean()),
        "f_turnaround_mean_ns": float(turnaround.mean()),
        "f_turnaround_std_ns": float(turnaround.std()),
        "f_turnaround_median_ns": float(np.median(turnaround)),
        "f_turnaround_max_ns": float(turnaround.max()),
        "f_turnaround_p90_ns": float(np.percentile(turnaround, 90)),
        "f_turnaround_p95_ns": float(np.percentile(turnaround, 95)),
        "f_queue_mean_ns": float(queue.mean()),
        "f_queue_std_ns": float(queue.std()),
        "f_queue_max_ns": float(queue.max()),
        "f_queue_p95_ns": float(np.percentile(queue, 95)),
        "f_scheduler_selection_mean": float(selection_count.mean()),
        "f_resource_acquisition_failures_mean": float(acquisition_failures.mean()),
        "f_longest_rejection_run": float(longest_true_run(rejected)),
        "f_rejection_transition_count": float(
            np.count_nonzero(rejected[1:] != rejected[:-1])
        ),
    }

    normalized_positions = np.linspace(0.0, 1.0, len(turnaround))
    if len(turnaround) > 1:
        features["f_turnaround_linear_slope"] = float(
            np.polyfit(normalized_positions, turnaround, 1)[0]
        )
    else:
        features["f_turnaround_linear_slope"] = 0.0

    for lag in range(1, 6):
        features[f"f_turnaround_autocorr_lag_{lag}"] = safe_autocorrelation(
            turnaround,
            lag,
        )

    bins = np.array_split(np.arange(len(frame)), NUM_TEMPORAL_BINS)
    for bin_index, indices in enumerate(bins):
        if len(indices):
            bin_turnaround = turnaround[indices]
            bin_queue = queue[indices]
            bin_rejected = rejected[indices]
        else:
            bin_turnaround = np.array([0.0])
            bin_queue = np.array([0.0])
            bin_rejected = np.array([False])
        features[f"f_bin_{bin_index:02d}_turnaround_mean_ns"] = float(
            bin_turnaround.mean()
        )
        features[f"f_bin_{bin_index:02d}_queue_mean_ns"] = float(
            bin_queue.mean()
        )
        features[f"f_bin_{bin_index:02d}_rejected_fraction"] = float(
            bin_rejected.mean()
        )

    # Preserve the complete attacker-visible sequence. Missing/rejected probes
    # use zero turnaround plus a separate rejection indicator.
    for probe_index in range(TOTAL_PROBES):
        if probe_index < len(frame):
            features[f"f_probe_{probe_index:03d}_turnaround_ns"] = float(
                turnaround[probe_index]
            )
            features[f"f_probe_{probe_index:03d}_rejected"] = float(
                rejected[probe_index]
            )
        else:
            features[f"f_probe_{probe_index:03d}_turnaround_ns"] = 0.0
            features[f"f_probe_{probe_index:03d}_rejected"] = 1.0

    return features


# =============================================================================
# Knowledge-view features
# =============================================================================


def knowledge_features(
    *,
    profile: KnowledgeProfile,
    selected_pair: tuple[int, int],
    allocation_policy: str,
    scheduler_policy: str,
    victim_allocation: p1.Allocation,
    attacker_allocation: p1.Allocation,
    sharing: dict[str, Any],
    num_modules: int,
) -> dict[str, float]:
    features: dict[str, float] = {
        "k_own_physical_known": float(profile.own_physical_known),
        "k_victim_physical_known": float(profile.victim_physical_known),
        "k_logical_identifiers_known": float(profile.logical_identifiers_known),
        "k_allocator_policy_known": float(profile.allocator_policy_known),
        "k_scheduler_policy_known": float(profile.scheduler_policy_known),
        "k_selected_pair_01": float(selected_pair == (0, 1)),
        "k_selected_pair_12": float(selected_pair == (1, 2)),
        "k_selected_pair_02": float(selected_pair == (0, 2)),
        "k_known_shared_module_count": (
            float(sharing["shared_module_count"])
            if profile.victim_physical_known
            else 0.0
        ),
        "k_known_shared_link_count": (
            float(sharing["shared_link_count"])
            if profile.victim_physical_known
            else 0.0
        ),
        "k_known_endpoint_overlap": (
            float(sharing["endpoint_overlap"])
            if profile.victim_physical_known
            else 0.0
        ),
        "k_known_link_overlap": (
            float(sharing["link_overlap"])
            if profile.victim_physical_known
            else 0.0
        ),
    }

    for module in range(max(NUM_MODULE_OPTIONS)):
        features[f"k_own_module_{module:02d}"] = 0.0
        features[f"k_victim_module_{module:02d}"] = 0.0

    if profile.own_physical_known and attacker_allocation.accepted:
        for module in attacker_allocation.assigned_modules:
            features[f"k_own_module_{module:02d}"] = 1.0

    if profile.victim_physical_known and victim_allocation.accepted:
        for module in victim_allocation.assigned_modules:
            features[f"k_victim_module_{module:02d}"] = 1.0

    for name in ALLOCATOR_FEATURE_NAMES:
        features[name] = 0.0
    for name in SCHEDULER_FEATURE_NAMES:
        features[name] = 0.0

    if profile.allocator_policy_known:
        features[f"k_allocator_{allocation_policy}"] = 1.0
    if profile.scheduler_policy_known:
        features[f"k_scheduler_{scheduler_policy}"] = 1.0

    # Encode system size only when either policy or physical state is known.
    features["k_num_modules"] = (
        float(num_modules)
        if (
            profile.own_physical_known
            or profile.victim_physical_known
            or profile.allocator_policy_known
        )
        else 0.0
    )
    return features


# =============================================================================
# Strategy-view construction
# =============================================================================


def strategy_selected_pair(
    *,
    strategy: str,
    profile: KnowledgeProfile,
    adaptive_pair: tuple[int, int],
    allocation_policy: str,
    victim_job: p1.JobSpec,
    victim_allocation: p1.Allocation,
    attacker_allocation: p1.Allocation,
    num_modules: int,
) -> tuple[int, int]:
    if strategy == "one_shot_fixed":
        return PROBE_PAIRS[0]
    if strategy == "one_shot_candidate_cycle":
        # The pair is not singular, but the canonical pair label keeps the
        # knowledge-feature schema fixed. The strategy name indicates cycling.
        return PROBE_PAIRS[0]
    if strategy == "one_shot_knowledge_guided":
        return knowledge_guided_pair(
            profile=profile,
            allocation_policy=allocation_policy,
            victim_job=victim_job,
            victim_allocation=victim_allocation,
            attacker_allocation=attacker_allocation,
            num_modules=num_modules,
        )
    if strategy == "adaptive_explore_exploit":
        return adaptive_pair
    raise ValueError(f"Unknown strategy: {strategy}")



def schedule_key_for_strategy(
    strategy: str,
    selected_pair: tuple[int, int],
) -> str:
    if strategy == "one_shot_fixed":
        return pair_name(PROBE_PAIRS[0])
    if strategy == "one_shot_candidate_cycle":
        return "candidate_cycle"
    if strategy == "one_shot_knowledge_guided":
        return pair_name(selected_pair)
    if strategy == "adaptive_explore_exploit":
        return "adaptive"
    raise ValueError(strategy)


# =============================================================================
# One physical trial
# =============================================================================


def run_physical_trial(
    *,
    physical_trial_id: str,
    victim_qasm: Path,
    num_modules: int,
    tenant_count: int,
    trial_id: int,
    seed: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[pd.DataFrame],
]:
    rng = random.Random(seed)
    allocation_policy = rng.choice(ALLOCATION_POLICIES)
    scheduler_policy = rng.choice(SCHEDULER_POLICIES)
    victim_modules_requested = rng.choice(VICTIM_MODULE_REQUEST_OPTIONS)

    victim_job = copy.deepcopy(
        p1.build_victim_job(
            victim_qasm,
            victim_modules_requested,
        )
    )
    attacker_for_allocation = attacker_allocation_job()
    background_jobs = build_background_jobs(tenant_count, seed)

    allocations = allocate_physical_trial(
        victim_job=victim_job,
        attacker_job=attacker_for_allocation,
        background_jobs=background_jobs,
        allocation_policy=allocation_policy,
        num_modules=num_modules,
        seed=stable_seed(seed, "allocation"),
    )

    victim_allocation = allocations["victim"]
    attacker_allocation = allocations["attacker"]
    sharing = physical_sharing_state(
        victim_job=victim_job,
        attacker_job=attacker_for_allocation,
        victim_allocation=victim_allocation,
        attacker_allocation=attacker_allocation,
        num_modules=num_modules,
    )

    physical_summary: dict[str, Any] = {
        "physical_trial_id": physical_trial_id,
        "victim_qasm": victim_qasm.name,
        "victim_tag": p1.safe_tag(victim_qasm.name),
        "num_modules": num_modules,
        "simultaneous_tenants": tenant_count,
        "trial_id": trial_id,
        "seed": seed,
        "allocation_policy": allocation_policy,
        "scheduler_policy": scheduler_policy,
        "victim_modules_requested": victim_modules_requested,
        "attacker_modules_requested": ATTACKER_MODULES_REQUESTED,
        "victim_accepted": victim_allocation.accepted,
        "attacker_accepted": attacker_allocation.accepted,
        "victim_assigned_modules": ",".join(
            map(str, victim_allocation.assigned_modules)
        ),
        "attacker_assigned_modules": ",".join(
            map(str, attacker_allocation.assigned_modules)
        ),
        **sharing,
    }

    if not victim_allocation.accepted or not attacker_allocation.accepted:
        return physical_summary, [], [], []

    # Victim-only completion is common to every attacker strategy.
    victim_background_jobs = [victim_job, *background_jobs]
    victim_only_log, _, _ = execute_schedule(
        jobs=victim_background_jobs,
        allocations=allocations,
        included_roles={"victim", "background"},
        num_modules=num_modules,
        scheduler_policy=scheduler_policy,
        tenant_count=tenant_count,
        seed=stable_seed(seed, "scenario", "victim_only"),
    )
    victim_only_success, victim_only_completion = role_completion(
        victim_only_log,
        "victim",
    )
    physical_summary["victim_only_success"] = victim_only_success
    physical_summary["victim_only_completion_ns"] = victim_only_completion

    schedule_jobs: dict[str, p1.JobSpec] = {
        pair_name(pair): build_attacker_job(
            schedule_mode="single_pair",
            selected_pair=pair,
        )
        for pair in PROBE_PAIRS
    }
    schedule_jobs["candidate_cycle"] = build_attacker_job(
        schedule_mode="candidate_cycle"
    )

    schedule_results: dict[str, dict[str, Any]] = {}
    request_frames: list[pd.DataFrame] = []

    roles_background = {"background"}

    # Run all three candidate probes and the candidate-cycle exploration trace.
    for schedule_name, attacker_job in schedule_jobs.items():
        jobs = [victim_job, attacker_job, *background_jobs]
        scenario_seed = stable_seed(seed, "scenario", schedule_name)

        attacker_only_log, _, _ = execute_schedule(
            jobs=jobs,
            allocations=allocations,
            included_roles={"attacker", *roles_background},
            num_modules=num_modules,
            scheduler_policy=scheduler_policy,
            tenant_count=tenant_count,
            seed=scenario_seed,
        )
        combined_log, _, _ = execute_schedule(
            jobs=jobs,
            allocations=allocations,
            included_roles={"victim", "attacker", *roles_background},
            num_modules=num_modules,
            scheduler_policy=scheduler_policy,
            tenant_count=tenant_count,
            seed=scenario_seed,
        )
        comparison = compare_attacker_logs(attacker_only_log, combined_log)
        combined_victim_success, combined_victim_completion = role_completion(
            combined_log,
            "victim",
        )

        schedule_results[schedule_name] = {
            "attacker_only_log": attacker_only_log,
            "combined_log": combined_log,
            "comparison": comparison,
            "combined_victim_success": combined_victim_success,
            "combined_victim_completion_ns": combined_victim_completion,
        }

        if SAVE_REQUEST_LEVEL_RESULTS:
            for presence, log in [
                ("victim_absent", attacker_only_log),
                ("victim_present", combined_log),
            ]:
                frame = attacker_frame(log)
                frame.insert(0, "victim_presence", presence)
                frame.insert(0, "schedule_name", schedule_name)
                frame.insert(0, "physical_trial_id", physical_trial_id)
                request_frames.append(frame)

    cycle_comparison = schedule_results["candidate_cycle"]["comparison"]
    adaptive_pair, exploration_scores = exploration_selected_pair(cycle_comparison)

    adaptive_job = build_attacker_job(
        schedule_mode="adaptive",
        selected_pair=adaptive_pair,
    )
    adaptive_schedule_name = "adaptive"
    adaptive_jobs = [victim_job, adaptive_job, *background_jobs]
    adaptive_seed = stable_seed(seed, "scenario", adaptive_schedule_name)

    adaptive_attacker_only, _, _ = execute_schedule(
        jobs=adaptive_jobs,
        allocations=allocations,
        included_roles={"attacker", *roles_background},
        num_modules=num_modules,
        scheduler_policy=scheduler_policy,
        tenant_count=tenant_count,
        seed=adaptive_seed,
    )
    adaptive_combined, _, _ = execute_schedule(
        jobs=adaptive_jobs,
        allocations=allocations,
        included_roles={"victim", "attacker", *roles_background},
        num_modules=num_modules,
        scheduler_policy=scheduler_policy,
        tenant_count=tenant_count,
        seed=adaptive_seed,
    )
    adaptive_comparison = compare_attacker_logs(
        adaptive_attacker_only,
        adaptive_combined,
    )
    adaptive_victim_success, adaptive_victim_completion = role_completion(
        adaptive_combined,
        "victim",
    )
    schedule_results[adaptive_schedule_name] = {
        "attacker_only_log": adaptive_attacker_only,
        "combined_log": adaptive_combined,
        "comparison": adaptive_comparison,
        "combined_victim_success": adaptive_victim_success,
        "combined_victim_completion_ns": adaptive_victim_completion,
    }

    if SAVE_REQUEST_LEVEL_RESULTS:
        for presence, log in [
            ("victim_absent", adaptive_attacker_only),
            ("victim_present", adaptive_combined),
        ]:
            frame = attacker_frame(log)
            frame.insert(0, "victim_presence", presence)
            frame.insert(0, "schedule_name", adaptive_schedule_name)
            frame.insert(0, "physical_trial_id", physical_trial_id)
            request_frames.append(frame)

    oracle_pair = oracle_best_pair(
        victim_job=victim_job,
        victim_allocation=victim_allocation,
        attacker_allocation=attacker_allocation,
        num_modules=num_modules,
    )

    strategy_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for profile in KNOWLEDGE_PROFILES:
        for strategy in PROBE_STRATEGIES:
            selected_pair = strategy_selected_pair(
                strategy=strategy,
                profile=profile,
                adaptive_pair=adaptive_pair,
                allocation_policy=allocation_policy,
                victim_job=victim_job,
                victim_allocation=victim_allocation,
                attacker_allocation=attacker_allocation,
                num_modules=num_modules,
            )
            schedule_name = schedule_key_for_strategy(strategy, selected_pair)
            result = schedule_results[schedule_name]
            comparison = result["comparison"]
            metrics = comparison_metrics(comparison)

            combined_success = bool(result["combined_victim_success"])
            combined_completion = float(result["combined_victim_completion_ns"])
            if victim_only_success and combined_success:
                victim_slowdown_ns = combined_completion - victim_only_completion
                victim_slowdown_ratio = (
                    (combined_completion - victim_job.start_time_ns)
                    / max(victim_only_completion - victim_job.start_time_ns, 1.0)
                )
            else:
                victim_slowdown_ns = float("nan")
                victim_slowdown_ratio = float("nan")

            strategy_row = {
                **physical_summary,
                "knowledge_level": profile.name,
                "probe_strategy": strategy,
                "selected_probe_pair": pair_name(selected_pair),
                "adaptive_selected_pair": pair_name(adaptive_pair),
                "oracle_best_pair": pair_name(oracle_pair),
                "adaptive_selection_matches_oracle": bool(adaptive_pair == oracle_pair),
                "exploration_probe_count": (
                    EXPLORATION_PROBES
                    if strategy == "adaptive_explore_exploit"
                    else 0
                ),
                "one_shot": strategy != "adaptive_explore_exploit",
                "combined_victim_success": combined_success,
                "combined_victim_completion_ns": combined_completion,
                "victim_slowdown_ns": victim_slowdown_ns,
                "victim_slowdown_ratio": victim_slowdown_ratio,
                **metrics,
                **{
                    f"exploration_score_{name}": value
                    for name, value in exploration_scores.items()
                },
            }
            strategy_rows.append(strategy_row)

            k_features = knowledge_features(
                profile=profile,
                selected_pair=selected_pair,
                allocation_policy=allocation_policy,
                scheduler_policy=scheduler_policy,
                victim_allocation=victim_allocation,
                attacker_allocation=attacker_allocation,
                sharing=sharing,
                num_modules=num_modules,
            )

            for presence_label, presence_value, request_log in [
                ("victim_absent", 0, result["attacker_only_log"]),
                ("victim_present", 1, result["combined_log"]),
            ]:
                features = absolute_timing_features(request_log)
                sample_rows.append(
                    {
                        "physical_trial_id": physical_trial_id,
                        "victim_qasm": victim_qasm.name,
                        "victim_tag": p1.safe_tag(victim_qasm.name),
                        "num_modules": num_modules,
                        "simultaneous_tenants": tenant_count,
                        "trial_id": trial_id,
                        "seed": seed,
                        "allocation_policy": allocation_policy,
                        "scheduler_policy": scheduler_policy,
                        "knowledge_level": profile.name,
                        "probe_strategy": strategy,
                        "selected_probe_pair": pair_name(selected_pair),
                        "victim_presence_label": presence_label,
                        "victim_presence": presence_value,
                        "placement_class": (
                            sharing["placement_class"]
                            if presence_value == 1
                            else "no_victim"
                        ),
                        "resource_sharing_binary": int(
                            presence_value == 1
                            and sharing["placement_class"]
                            in {"endpoint_overlap", "link_overlap_only"}
                        ),
                        **features,
                        **k_features,
                    }
                )

    physical_summary.update(
        {
            "adaptive_selected_pair": pair_name(adaptive_pair),
            "oracle_best_pair": pair_name(oracle_pair),
            "adaptive_selection_matches_oracle": bool(adaptive_pair == oracle_pair),
            **{
                f"exploration_score_{name}": value
                for name, value in exploration_scores.items()
            },
        }
    )

    return physical_summary, strategy_rows, sample_rows, request_frames


# =============================================================================
# Grouped nearest-centroid classification
# =============================================================================


def fold_for_group(group: str, folds: int = CLASSIFICATION_FOLDS) -> int:
    return zlib.crc32(group.encode("utf-8")) % folds



def balanced_accuracy(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_order: Sequence[str],
) -> float:
    recalls: list[float] = []
    for label in class_order:
        mask = true_labels == label
        if mask.any():
            recalls.append(float((predicted_labels[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0



def macro_f1(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_order: Sequence[str],
) -> float:
    values: list[float] = []
    for label in class_order:
        true_positive = np.sum((true_labels == label) & (predicted_labels == label))
        false_positive = np.sum((true_labels != label) & (predicted_labels == label))
        false_negative = np.sum((true_labels == label) & (predicted_labels != label))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        value = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0



def nearest_centroid_predictions(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    allowed_classes: Sequence[str] | None = None,
) -> pd.DataFrame:
    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith("f_") or column.startswith("k_")
    ]

    data = dataframe.copy()
    if "sample_row_id" not in data.columns:
        data["sample_row_id"] = data.index.astype(int)
    data = data.reset_index(drop=True)
    if allowed_classes is not None:
        data = data[data[target_column].isin(allowed_classes)].reset_index(drop=True)

    predictions: list[dict[str, Any]] = []
    for fold in range(CLASSIFICATION_FOLDS):
        test_mask = data["physical_trial_id"].map(
            lambda value: fold_for_group(str(value)) == fold
        )
        train = data[~test_mask]
        test = data[test_mask]
        if train.empty or test.empty:
            continue

        classes = sorted(train[target_column].astype(str).unique())
        if len(classes) < 2:
            continue

        train_matrix = (
            train[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        test_matrix = (
            test[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0)
        std[std == 0] = 1.0
        train_scaled = (train_matrix - mean) / std
        test_scaled = (test_matrix - mean) / std

        centroids = {
            label: train_scaled[
                train[target_column].astype(str).to_numpy() == label
            ].mean(axis=0)
            for label in classes
        }

        for local_index, (_, row) in enumerate(test.iterrows()):
            distances = {
                label: float(
                    np.mean(np.square(test_scaled[local_index] - centroid))
                )
                for label, centroid in centroids.items()
            }
            predicted = min(distances, key=distances.get)
            ordered = sorted(distances.items(), key=lambda item: item[1])
            margin = (
                ordered[1][1] - ordered[0][1]
                if len(ordered) > 1
                else 0.0
            )
            predictions.append(
                {
                    "sample_row_id": int(row["sample_row_id"]),
                    "physical_trial_id": row["physical_trial_id"],
                    "knowledge_level": row["knowledge_level"],
                    "probe_strategy": row["probe_strategy"],
                    "true_label": str(row[target_column]),
                    "predicted_label": predicted,
                    "correct": predicted == str(row[target_column]),
                    "classification_margin": float(margin),
                    "fold": fold,
                }
            )

    return pd.DataFrame(predictions)



def classification_metrics(
    predictions: pd.DataFrame,
    *,
    task: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(rows)

    for (knowledge, strategy), frame in predictions.groupby(
        ["knowledge_level", "probe_strategy"],
        observed=True,
    ):
        true_labels = frame["true_label"].to_numpy(dtype=str)
        predicted = frame["predicted_label"].to_numpy(dtype=str)
        classes = sorted(set(true_labels))
        accuracy = float((true_labels == predicted).mean())
        row: dict[str, Any] = {
            "task": task,
            "knowledge_level": knowledge,
            "probe_strategy": strategy,
            "sample_count": int(len(frame)),
            "class_count": len(classes),
            "chance_accuracy": 1.0 / max(len(classes), 1),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy(true_labels, predicted, classes),
            "macro_f1": macro_f1(true_labels, predicted, classes),
        }

        if task == "victim_presence":
            positive = "1"
            negative = "0"
            tp = int(np.sum((true_labels == positive) & (predicted == positive)))
            tn = int(np.sum((true_labels == negative) & (predicted == negative)))
            fp = int(np.sum((true_labels == negative) & (predicted == positive)))
            fn = int(np.sum((true_labels == positive) & (predicted == negative)))
            row.update(
                {
                    "true_positive": tp,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "sensitivity": tp / max(tp + fn, 1),
                    "specificity": tn / max(tn + fp, 1),
                    "false_positive_rate": fp / max(tn + fp, 1),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Optional random-forest classification
# =============================================================================


def optional_random_forest(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    except ImportError:
        print("scikit-learn not installed; random-forest analysis skipped.")
        return pd.DataFrame(), pd.DataFrame()

    feature_columns = [
        column
        for column in samples.columns
        if column.startswith("f_") or column.startswith("k_")
    ]
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for (knowledge, strategy), data in samples.groupby(
        ["knowledge_level", "probe_strategy"],
        observed=True,
    ):
        data = data.reset_index(drop=True)
        true_all: list[str] = []
        predicted_all: list[str] = []
        metadata_rows: list[dict[str, Any]] = []

        for fold in range(CLASSIFICATION_FOLDS):
            test_mask = data["physical_trial_id"].map(
                lambda value: fold_for_group(str(value)) == fold
            )
            train = data[~test_mask]
            test = data[test_mask]
            if train.empty or test.empty:
                continue

            x_train = (
                train[feature_columns]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            x_test = (
                test[feature_columns]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            y_train = train["victim_presence"].astype(str).to_numpy()
            y_test = test["victim_presence"].astype(str).to_numpy()

            classifier = RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                class_weight="balanced",
                random_state=stable_seed(knowledge, strategy, fold),
                n_jobs=-1,
            )
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_test)

            true_all.extend(y_test.tolist())
            predicted_all.extend(predicted.tolist())
            for (_, row), true_value, predicted_value in zip(
                test.iterrows(),
                y_test,
                predicted,
            ):
                metadata_rows.append(
                    {
                        "physical_trial_id": row["physical_trial_id"],
                        "knowledge_level": knowledge,
                        "probe_strategy": strategy,
                        "true_label": true_value,
                        "predicted_label": predicted_value,
                        "correct": true_value == predicted_value,
                        "fold": fold,
                    }
                )

        if not true_all:
            continue
        true_array = np.asarray(true_all)
        predicted_array = np.asarray(predicted_all)
        metric_rows.append(
            {
                "task": "victim_presence",
                "method": "random_forest",
                "knowledge_level": knowledge,
                "probe_strategy": strategy,
                "sample_count": len(true_array),
                "accuracy": accuracy_score(true_array, predicted_array),
                "balanced_accuracy": balanced_accuracy_score(
                    true_array,
                    predicted_array,
                ),
                "macro_f1": f1_score(
                    true_array,
                    predicted_array,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
        prediction_rows.extend(metadata_rows)

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


# =============================================================================
# Aggregation and plots
# =============================================================================


def aggregate_strategy_results(strategy_trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    accepted = strategy_trials[
        strategy_trials["victim_accepted"]
        & strategy_trials["attacker_accepted"]
    ].copy()

    knowledge_summary = (
        accepted.groupby(
            ["knowledge_level", "probe_strategy"],
            as_index=False,
            observed=True,
        )
        .agg(
            trial_count=("physical_trial_id", "count"),
            paired_detection_probability=("paired_rule_detected", "mean"),
            mean_observable_change_fraction=("observable_change_fraction", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            mean_probes_to_first_change=("probes_to_first_observable_change", "mean"),
            median_probes_to_first_change=("probes_to_first_observable_change", "median"),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
            victim_success_probability=("combined_victim_success", "mean"),
            adaptive_or_oracle_match=("adaptive_selection_matches_oracle", "mean"),
        )
    )

    allocator_summary = (
        accepted.groupby(
            ["allocation_policy", "probe_strategy"],
            as_index=False,
            observed=True,
        )
        .agg(
            trial_count=("physical_trial_id", "count"),
            endpoint_overlap_probability=("endpoint_overlap", "mean"),
            link_overlap_probability=("link_overlap", "mean"),
            paired_detection_probability=("paired_rule_detected", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
        )
    )

    scheduler_summary = (
        accepted.groupby(
            ["scheduler_policy", "probe_strategy"],
            as_index=False,
            observed=True,
        )
        .agg(
            trial_count=("physical_trial_id", "count"),
            paired_detection_probability=("paired_rule_detected", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            mean_probes_to_first_change=("probes_to_first_observable_change", "mean"),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
        )
    )

    policy_summary = (
        accepted.groupby(
            ["allocation_policy", "scheduler_policy", "probe_strategy"],
            as_index=False,
            observed=True,
        )
        .agg(
            trial_count=("physical_trial_id", "count"),
            paired_detection_probability=("paired_rule_detected", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            endpoint_overlap_probability=("endpoint_overlap", "mean"),
            link_overlap_probability=("link_overlap", "mean"),
        )
    )

    return {
        "knowledge": knowledge_summary,
        "allocator": allocator_summary,
        "scheduler": scheduler_summary,
        "policy": policy_summary,
    }



def paired_rule_summary(strategy_trials: pd.DataFrame) -> pd.DataFrame:
    accepted = strategy_trials[
        strategy_trials["victim_accepted"]
        & strategy_trials["attacker_accepted"]
    ].copy()

    return (
        accepted.groupby(
            ["knowledge_level", "probe_strategy", "placement_class"],
            as_index=False,
            observed=True,
        )
        .agg(
            trial_count=("physical_trial_id", "count"),
            detection_probability=("paired_rule_detected", "mean"),
            mean_observable_change_fraction=("observable_change_fraction", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            mean_probes_to_first_change=("probes_to_first_observable_change", "mean"),
        )
    )



def no_sharing_false_positive(
    detection_predictions: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    if detection_predictions.empty:
        return pd.DataFrame()

    present = samples[samples["victim_presence"] == 1].copy()
    if "sample_row_id" not in present.columns:
        present["sample_row_id"] = present.index.astype(int)
    merged = detection_predictions.merge(
        present[
            [
                "sample_row_id",
                "physical_trial_id",
                "knowledge_level",
                "probe_strategy",
                "placement_class",
            ]
        ],
        on=[
            "sample_row_id",
            "physical_trial_id",
            "knowledge_level",
            "probe_strategy",
        ],
        how="left",
        validate="one_to_one",
    )

    no_direct_sharing = merged[
        merged["placement_class"] == "hub_only"
    ]
    if no_direct_sharing.empty:
        return pd.DataFrame()

    return (
        no_direct_sharing.groupby(
            ["knowledge_level", "probe_strategy"],
            as_index=False,
            observed=True,
        )
        .agg(
            no_direct_sharing_sample_count=("physical_trial_id", "count"),
            victim_activity_detection_rate=(
                "predicted_label",
                lambda series: float((series.astype(str) == "1").mean()),
            ),
        )
        .rename(
            columns={
                "victim_activity_detection_rate": (
                    "false_positive_rate_if_endpoint_or_link_sharing_is_assumed"
                )
            }
        )
    )



def save_metric_plot(
    metrics: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    if metrics.empty:
        return
    pivot = (
        metrics.pivot(
            index="knowledge_level",
            columns="probe_strategy",
            values=metric,
        )
        .reindex(index=KNOWLEDGE_ORDER, columns=PROBE_STRATEGIES)
    )
    axis = pivot.plot(kind="bar", figsize=(16, 7))
    axis.set_xlabel("Attacker knowledge level")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Probe strategy", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qasm_paths = [p1.resolve_qasm(name) for name in VICTIM_QASMS]

    if RUN_QUICK_VALIDATION:
        module_options = [8]
        tenant_options = [2, 4]
        qasm_paths = qasm_paths[:1]
        trials_per_setting = 1
    else:
        module_options = NUM_MODULE_OPTIONS
        tenant_options = TENANT_COUNT_OPTIONS
        trials_per_setting = TRIALS_PER_MODULE_TENANCY_WORKLOAD

    physical_specs: list[tuple[Path, int, int, int]] = []
    for qasm_path in qasm_paths:
        for num_modules in module_options:
            for tenant_count in tenant_options:
                for trial_id in range(trials_per_setting):
                    physical_specs.append(
                        (qasm_path, num_modules, tenant_count, trial_id)
                    )

    if MAX_PHYSICAL_TRIALS is not None:
        physical_specs = physical_specs[:MAX_PHYSICAL_TRIALS]

    print("Phase 1.6 — Unknown placement and allocation state")
    print(f"Physical trials: {len(physical_specs)}")
    print(f"Knowledge profiles: {len(KNOWLEDGE_PROFILES)}")
    print(f"Probe strategies: {len(PROBE_STRATEGIES)}")

    physical_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    request_frames: list[pd.DataFrame] = []

    for index, (qasm_path, num_modules, tenant_count, trial_id) in enumerate(
        physical_specs,
        start=1,
    ):
        physical_trial_id = (
            f"phys_{zlib.crc32(f'{qasm_path.name}|{num_modules}|{tenant_count}|{trial_id}'.encode('utf-8')):08x}"
        )
        seed = stable_seed(
            qasm_path.name,
            num_modules,
            tenant_count,
            trial_id,
        )

        print(
            f"[{index:04d}/{len(physical_specs):04d}] "
            f"modules={num_modules} | tenants={tenant_count} | "
            f"{qasm_path.name} | trial={trial_id}"
        )

        physical, strategies, samples, logs = run_physical_trial(
            physical_trial_id=physical_trial_id,
            victim_qasm=qasm_path,
            num_modules=num_modules,
            tenant_count=tenant_count,
            trial_id=trial_id,
            seed=seed,
        )
        physical_rows.append(physical)
        strategy_rows.extend(strategies)
        sample_rows.extend(samples)
        request_frames.extend(logs)

    physical_trials = pd.DataFrame(physical_rows)
    strategy_trials = pd.DataFrame(strategy_rows)
    samples = pd.DataFrame(sample_rows)
    if not samples.empty:
        samples.insert(0, "sample_row_id", np.arange(len(samples), dtype=int))

    physical_trials.to_csv(
        OUTPUT_DIR / "unknown_placement_physical_trial_summary.csv",
        index=False,
    )
    strategy_trials.to_csv(
        OUTPUT_DIR / "unknown_placement_strategy_trial_summary.csv",
        index=False,
    )
    samples.to_csv(
        OUTPUT_DIR / "unknown_placement_samples.csv",
        index=False,
    )

    feature_columns = [
        column
        for column in samples.columns
        if column.startswith("f_") or column.startswith("k_")
    ]
    samples[
        [
            "physical_trial_id",
            "victim_qasm",
            "knowledge_level",
            "probe_strategy",
            "victim_presence",
            "placement_class",
            *feature_columns,
        ]
    ].to_csv(
        OUTPUT_DIR / "unknown_placement_features.csv",
        index=False,
    )

    if SAVE_REQUEST_LEVEL_RESULTS and request_frames:
        pd.concat(request_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "unknown_placement_attacker_request_log.csv.gz",
            index=False,
            compression="gzip",
        )

    # -------------------------------------------------------------------------
    # Non-ML victim-presence classification
    # -------------------------------------------------------------------------

    detection_predictions_parts: list[pd.DataFrame] = []
    sharing_predictions_parts: list[pd.DataFrame] = []

    for (knowledge, strategy), subset in samples.groupby(
        ["knowledge_level", "probe_strategy"],
        observed=True,
    ):
        detection_predictions_parts.append(
            nearest_centroid_predictions(
                subset,
                target_column="victim_presence",
                allowed_classes=[0, 1],
            )
        )

        sharing_subset = subset[
            (subset["victim_presence"] == 1)
            & subset["placement_class"].isin(
                ["endpoint_overlap", "link_overlap_only", "hub_only"]
            )
        ]
        sharing_predictions_parts.append(
            nearest_centroid_predictions(
                sharing_subset,
                target_column="placement_class",
                allowed_classes=[
                    "endpoint_overlap",
                    "link_overlap_only",
                    "hub_only",
                ],
            )
        )

    detection_predictions = pd.concat(
        [frame for frame in detection_predictions_parts if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in detection_predictions_parts) else pd.DataFrame()

    sharing_predictions = pd.concat(
        [frame for frame in sharing_predictions_parts if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in sharing_predictions_parts) else pd.DataFrame()

    detection_metrics = classification_metrics(
        detection_predictions,
        task="victim_presence",
    )
    sharing_metrics = classification_metrics(
        sharing_predictions,
        task="placement_class",
    )

    detection_predictions.to_csv(
        OUTPUT_DIR / "unknown_placement_detection_predictions.csv",
        index=False,
    )
    detection_metrics.to_csv(
        OUTPUT_DIR / "unknown_placement_detection_metrics.csv",
        index=False,
    )
    sharing_predictions.to_csv(
        OUTPUT_DIR / "unknown_placement_sharing_predictions.csv",
        index=False,
    )
    sharing_metrics.to_csv(
        OUTPUT_DIR / "unknown_placement_sharing_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Physical and strategy summaries
    # -------------------------------------------------------------------------

    summaries = aggregate_strategy_results(strategy_trials)
    summaries["knowledge"].to_csv(
        OUTPUT_DIR / "unknown_placement_knowledge_summary.csv",
        index=False,
    )
    summaries["allocator"].to_csv(
        OUTPUT_DIR / "unknown_placement_allocator_summary.csv",
        index=False,
    )
    summaries["scheduler"].to_csv(
        OUTPUT_DIR / "unknown_placement_scheduler_summary.csv",
        index=False,
    )
    summaries["policy"].to_csv(
        OUTPUT_DIR / "unknown_placement_policy_summary.csv",
        index=False,
    )

    paired_summary = paired_rule_summary(strategy_trials)
    paired_summary.to_csv(
        OUTPUT_DIR / "unknown_placement_paired_rule_summary.csv",
        index=False,
    )

    adaptive_rows = strategy_trials[
        strategy_trials["probe_strategy"] == "adaptive_explore_exploit"
    ]
    probe_selection_summary = (
        adaptive_rows.groupby(
            ["knowledge_level", "allocation_policy", "scheduler_policy"],
            as_index=False,
            observed=True,
        )
        .agg(
            trial_count=("physical_trial_id", "count"),
            adaptive_oracle_match=("adaptive_selection_matches_oracle", "mean"),
            mean_exploration_probes=("exploration_probe_count", "mean"),
            paired_detection_probability=("paired_rule_detected", "mean"),
            mean_absolute_timing_change_ns=("mean_absolute_timing_change_ns", "mean"),
            mean_probes_to_first_change=("probes_to_first_observable_change", "mean"),
            mean_victim_slowdown_ratio=("victim_slowdown_ratio", "mean"),
        )
    )
    probe_selection_summary.to_csv(
        OUTPUT_DIR / "unknown_placement_probe_selection_summary.csv",
        index=False,
    )

    false_positive = no_sharing_false_positive(
        detection_predictions,
        samples,
    )
    false_positive.to_csv(
        OUTPUT_DIR / "unknown_placement_no_sharing_false_positive.csv",
        index=False,
    )

    # Optional ML comparison. The non-ML results are always produced.
    rf_metrics, rf_predictions = optional_random_forest(samples)
    rf_metrics.to_csv(
        OUTPUT_DIR / "unknown_placement_random_forest_metrics.csv",
        index=False,
    )
    rf_predictions.to_csv(
        OUTPUT_DIR / "unknown_placement_random_forest_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    save_metric_plot(
        detection_metrics,
        metric="balanced_accuracy",
        title="Victim-Activity Detection Under Unknown Placement",
        ylabel="Balanced detection accuracy",
        filename="unknown_placement_detection_accuracy.png",
    )
    save_metric_plot(
        sharing_metrics,
        metric="balanced_accuracy",
        title="Sharing-Condition Inference Under Unknown Placement",
        ylabel="Balanced placement-class accuracy",
        filename="unknown_placement_sharing_accuracy.png",
    )
    save_metric_plot(
        summaries["knowledge"].rename(
            columns={"mean_probes_to_first_change": "balanced_accuracy"}
        ),
        metric="balanced_accuracy",
        title="Probes Required for First Observable Change",
        ylabel="Mean probes",
        filename="unknown_placement_probe_efficiency.png",
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    accepted_count = int(
        (
            physical_trials["victim_accepted"]
            & physical_trials["attacker_accepted"]
        ).sum()
    )
    print("\n=== Phase 1.6 completion ===")
    print(f"Physical trials: {len(physical_trials)}")
    print(f"Accepted victim-attacker trials: {accepted_count}")
    print(f"Strategy/knowledge trial views: {len(strategy_trials)}")
    print(f"Classification samples: {len(samples)}")

    if not detection_metrics.empty:
        display = detection_metrics[
            [
                "knowledge_level",
                "probe_strategy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
            ]
        ].sort_values(
            ["knowledge_level", "probe_strategy"]
        )
        print("\nVictim-presence detection:")
        print(display.to_string(index=False))

    if not sharing_metrics.empty:
        display = sharing_metrics[
            [
                "knowledge_level",
                "probe_strategy",
                "balanced_accuracy",
                "macro_f1",
            ]
        ].sort_values(
            ["knowledge_level", "probe_strategy"]
        )
        print("\nSharing-condition inference:")
        print(display.to_string(index=False))

    print(f"\nSaved all results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
