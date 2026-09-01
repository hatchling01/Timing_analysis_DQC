#!/usr/bin/env python3
"""
run_qaoa_circuit_noise_robustness.py

Test whether the timing fingerprints of the supplied QAOA circuits
(qaoa_n5 through qaoa_n15) survive scheduler jitter and timestamp noise.

Dependencies
------------
1. new_arch_fivenode_traceadded.py
2. run_attack_tier1_p1_static_blackbox_observation_window_sweep.py
3. run_attack_tier1_p1_static_blackbox_qaoa_circuit_fingerprinting.py

The actual QASM files must be in the current directory or /mnt/data.

Fixed attack configuration
--------------------------
- Probe 3
- uniform 420 ns probe spacing
- 20,000 ns observation window
- exact nominal victim-start estimate
- P1 disjoint placement
- static distributed execution
- hub capacity = 1

Noise models
------------
Scheduler jitter:
    Gaussian perturbation applied to:
    - each victim event release time;
    - each attacker Probe-3 round release time.

    Victim event ordering and attacker probe ordering are preserved.

Timestamp noise:
    Gaussian noise applied independently to measured attacker-only and
    victim-present turnaround times.

Identification methods
----------------------
1. Nearest clean template:
   Every noisy timing trace is compared with the 11 clean exact-start traces.

2. Random forest:
   Evaluated separately for each noise profile using leave-one-trial-out
   cross-validation. Each held-out trial contains one trace from every circuit.

Outputs
-------
blackbox_window_results/qaoa_circuit_noise_robustness/

    qaoa_noise_trial_summary.csv
    qaoa_noise_features.csv
    qaoa_noise_observations.csv
    qaoa_clean_templates.csv
    qaoa_clean_template_pairwise_mae.csv

    qaoa_nearest_template_predictions.csv
    qaoa_nearest_template_metrics.csv

    qaoa_random_forest_predictions.csv
    qaoa_random_forest_metrics.csv
    qaoa_random_forest_metrics.json

    qaoa_noise_profile_signal_summary.csv
    qaoa_noise_accuracy_comparison.csv

    qaoa_noise_accuracy_comparison.png
    qaoa_noise_contention_fraction.png
    qaoa_noise_excess_latency.png

    confusion_matrices/
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

import run_atack_tier1_p1_static_blackbox_observation_window_sweep as base
import run_attack_tier1_p1_static_blackbox_qaoa_circuit_fingerprinting as qfp


# ============================================================
# Output and circuit configuration
# ============================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "qaoa_circuit_noise_robustness"
)

CONFUSION_DIR = (
    OUTPUT_DIR
    / "confusion_matrices"
)

SIZES = list(
    range(5, 16)
)

LABELS = [
    f"qaoa_n{num_qubits}"
    for num_qubits in SIZES
]

LABEL_ORDER = {
    label: index
    for index, label
    in enumerate(LABELS)
}


# ============================================================
# Selected black-box configuration
# ============================================================

VICTIM_START_NS = 1_000

ATTACKER_ESTIMATED_START_NS = 1_000

OBSERVATION_DURATION_NS = 20_000

PROBE_PERIOD_NS = 420

HUB_CAPACITY = 1

GLOBAL_SEED = 2026

# Threshold applied to a difference between two independently
# noisy timestamps. Difference noise has std = sqrt(2) * sigma.
DETECTION_SIGMA_MULTIPLIER = 3.0


# ============================================================
# Robustness profiles
# ============================================================

NOISE_PROFILES = [
    {
        "profile_name": "clean",
        "scheduler_jitter_std_ns": 0.0,
        "timestamp_noise_std_ns": 0.0,
        "trial_count": 1,
    },
    {
        "profile_name": "timestamp_5ns",
        "scheduler_jitter_std_ns": 0.0,
        "timestamp_noise_std_ns": 5.0,
        "trial_count": 10,
    },
    {
        "profile_name": "timestamp_20ns",
        "scheduler_jitter_std_ns": 0.0,
        "timestamp_noise_std_ns": 20.0,
        "trial_count": 10,
    },
    {
        "profile_name": "scheduler_5ns",
        "scheduler_jitter_std_ns": 5.0,
        "timestamp_noise_std_ns": 0.0,
        "trial_count": 10,
    },
    {
        "profile_name": "scheduler_20ns",
        "scheduler_jitter_std_ns": 20.0,
        "timestamp_noise_std_ns": 0.0,
        "trial_count": 10,
    },
    {
        "profile_name": "combined_5ns",
        "scheduler_jitter_std_ns": 5.0,
        "timestamp_noise_std_ns": 5.0,
        "trial_count": 10,
    },
    {
        "profile_name": "combined_20ns",
        "scheduler_jitter_std_ns": 20.0,
        "timestamp_noise_std_ns": 20.0,
        "trial_count": 10,
    },
    {
        "profile_name": "combined_50ns",
        "scheduler_jitter_std_ns": 50.0,
        "timestamp_noise_std_ns": 50.0,
        "trial_count": 10,
    },
]

PROFILE_ORDER = [
    profile["profile_name"]
    for profile in NOISE_PROFILES
]


# ============================================================
# Configure imported experiment helpers
# ============================================================

base.VICTIM_TRUE_START_NS = (
    VICTIM_START_NS
)

base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
    ATTACKER_ESTIMATED_START_NS
)

base.PROBE_ROUND_PERIOD_NS = (
    PROBE_PERIOD_NS
)

base.HUB_MAX_CONCURRENT_TRANSFERS = (
    HUB_CAPACITY
)


# ============================================================
# General helpers
# ============================================================

def clean_json_value(
    value: Any,
) -> Any:
    """Convert numpy values to JSON-safe Python values."""

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    return value


def profile_by_name(
    profile_name: str,
) -> dict[str, Any]:
    """Return one configured noise profile."""

    for profile in NOISE_PROFILES:
        if (
            profile["profile_name"]
            == profile_name
        ):
            return profile

    raise KeyError(
        f"Unknown profile: {profile_name}"
    )


def trial_seed(
    *,
    profile_index: int,
    trial_id: int,
    num_qubits: int,
    stream_id: int,
) -> int:
    """Create a deterministic independent seed."""

    return int(
        GLOBAL_SEED
        + profile_index * 1_000_000
        + trial_id * 10_000
        + num_qubits * 100
        + stream_id
    )


def detection_threshold_ns(
    timestamp_noise_std_ns: float,
) -> float:
    """Return the calibrated threshold for baseline subtraction."""

    if timestamp_noise_std_ns <= 0:
        return 0.0

    difference_std_ns = (
        math.sqrt(2.0)
        * timestamp_noise_std_ns
    )

    return (
        DETECTION_SIGMA_MULTIPLIER
        * difference_std_ns
    )


# ============================================================
# Scheduler-jitter generation
# ============================================================

def jitter_victim_schedule(
    victim_trace: list[dict],
    *,
    start_ns: int,
    jitter_std_ns: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Apply independent scheduler jitter to victim events.

    Event order is preserved. Events may share the same release time,
    but a later event is never released before an earlier event.
    """

    scheduled_events: list[dict] = []

    previous_release_ns = start_ns

    for event_index, trace_entry in enumerate(
        victim_trace
    ):
        nominal_release_ns = (
            start_ns
            + event_index
            * base.VICTIM_EVENT_TICK_NS
        )

        if jitter_std_ns > 0:
            jitter_ns = int(
                round(
                    rng.normal(
                        0.0,
                        jitter_std_ns,
                    )
                )
            )
        else:
            jitter_ns = 0

        candidate_release_ns = max(
            0,
            nominal_release_ns
            + jitter_ns,
        )

        actual_release_ns = max(
            previous_release_ns,
            candidate_release_ns,
        )

        scheduled_events.append(
            {
                "release_time_ns": (
                    actual_release_ns
                ),
                "tenant": "victim",
                "sequence_index": (
                    event_index
                ),
                "entry": copy.deepcopy(
                    trace_entry
                ),
            }
        )

        previous_release_ns = (
            actual_release_ns
        )

    return scheduled_events


def jitter_attacker_schedule(
    nominal_schedule: list[dict],
    nominal_metadata: dict[int, dict],
    *,
    jitter_std_ns: float,
    rng: np.random.Generator,
) -> tuple[
    list[dict],
    dict[int, dict],
]:
    """
    Shift each entire Probe-3 round by one Gaussian jitter sample.

    The internal Probe-3 operation spacing remains unchanged.
    Remote-probe ordering is preserved.
    """

    jittered_schedule = copy.deepcopy(
        nominal_schedule
    )

    jittered_metadata = copy.deepcopy(
        nominal_metadata
    )

    metadata_by_probe = {
        int(metadata["probe_id"]): metadata
        for metadata
        in jittered_metadata.values()
    }

    probe_deltas_ns: dict[int, int] = {}

    previous_remote_release_ns: int | None = None

    for probe_id in sorted(
        metadata_by_probe
    ):
        metadata = metadata_by_probe[
            probe_id
        ]

        nominal_remote_release_ns = int(
            metadata[
                "request_release_time_ns"
            ]
        )

        if jitter_std_ns > 0:
            sampled_jitter_ns = int(
                round(
                    rng.normal(
                        0.0,
                        jitter_std_ns,
                    )
                )
            )
        else:
            sampled_jitter_ns = 0

        candidate_remote_release_ns = max(
            0,
            nominal_remote_release_ns
            + sampled_jitter_ns,
        )

        if previous_remote_release_ns is None:
            actual_remote_release_ns = (
                candidate_remote_release_ns
            )
        else:
            actual_remote_release_ns = max(
                previous_remote_release_ns + 1,
                candidate_remote_release_ns,
            )

        delta_ns = (
            actual_remote_release_ns
            - nominal_remote_release_ns
        )

        probe_deltas_ns[
            probe_id
        ] = delta_ns

        metadata[
            "request_release_time_ns"
        ] = actual_remote_release_ns

        metadata[
            "round_start_time_ns"
        ] = (
            int(
                metadata[
                    "round_start_time_ns"
                ]
            )
            + delta_ns
        )

        previous_remote_release_ns = (
            actual_remote_release_ns
        )

    operations_per_round = len(
        base.PROBE_3_OPERATIONS
    )

    for scheduled_event in jittered_schedule:
        attacker_event_index = int(
            scheduled_event[
                "sequence_index"
            ]
        )

        probe_id = (
            attacker_event_index
            // operations_per_round
        )

        scheduled_event[
            "release_time_ns"
        ] = (
            int(
                scheduled_event[
                    "release_time_ns"
                ]
            )
            + probe_deltas_ns[
                probe_id
            ]
        )

    return (
        jittered_schedule,
        jittered_metadata,
    )


# ============================================================
# Timestamp-noise model
# ============================================================

def add_timestamp_noise(
    observations: pd.DataFrame,
    *,
    timestamp_noise_std_ns: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Add noise to an attacker's measured request turnaround.

    Waiting time is reconstructed from:
        observed turnaround - known service time.
    """

    noisy = observations.copy()

    if timestamp_noise_std_ns <= 0:
        return noisy

    noise_ns = rng.normal(
        0.0,
        timestamp_noise_std_ns,
        size=len(noisy),
    )

    observed_turnaround_ns = np.maximum(
        0.0,
        noisy[
            "turnaround_time_ns"
        ].to_numpy(
            dtype=float
        )
        + noise_ns,
    )

    service_time_ns = noisy[
        "service_time_ns"
    ].to_numpy(
        dtype=float
    )

    observed_waiting_ns = np.maximum(
        0.0,
        observed_turnaround_ns
        - service_time_ns,
    )

    noisy[
        "turnaround_time_ns"
    ] = observed_turnaround_ns

    noisy[
        "waiting_time_ns"
    ] = observed_waiting_ns

    noisy[
        "completion_time_ns"
    ] = (
        noisy[
            "completion_time_ns"
        ].to_numpy(
            dtype=float
        )
        + noise_ns
    )

    return noisy


# ============================================================
# Baseline subtraction and denoising
# ============================================================

def compare_and_threshold(
    attacker_only: pd.DataFrame,
    victim_present: pd.DataFrame,
    *,
    threshold_ns: float,
) -> pd.DataFrame:
    """
    Perform request-by-request baseline subtraction.

    Excess delays below the calibrated threshold are set to zero.
    Raw excess delays are retained in separate columns.
    """

    compared = (
        base.compare_attacker_runs(
            attacker_only,
            victim_present,
        )
    )

    compared[
        "raw_excess_turnaround_time_ns"
    ] = compared[
        "excess_turnaround_time_ns"
    ]

    compared[
        "raw_excess_waiting_time_ns"
    ] = compared[
        "excess_waiting_time_ns"
    ]

    raw_excess = compared[
        "raw_excess_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    detected = (
        raw_excess
        > threshold_ns
    )

    compared[
        "excess_turnaround_time_ns"
    ] = np.where(
        detected,
        raw_excess,
        0.0,
    )

    raw_waiting = compared[
        "raw_excess_waiting_time_ns"
    ].to_numpy(
        dtype=float
    )

    compared[
        "excess_waiting_time_ns"
    ] = np.where(
        detected,
        np.maximum(
            raw_waiting,
            0.0,
        ),
        0.0,
    )

    compared[
        "victim_contention_observed"
    ] = detected

    compared[
        "detection_threshold_ns"
    ] = threshold_ns

    return compared


# ============================================================
# Architecture execution
# ============================================================

def execute_attacker_only(
    *,
    victim_mapping: dict[int, str],
    num_qubits: int,
    attacker_mapping: dict[int, str],
    attacker_schedule: list[dict],
    attacker_metadata: dict[int, dict],
) -> tuple[
    Any,
    pd.DataFrame,
]:
    """Run an attacker-only calibration."""

    architecture = base.build_architecture(
        victim_mapping,
        num_qubits,
        attacker_mapping,
    )

    base.execute_timed_schedule(
        architecture,
        copy.deepcopy(
            attacker_schedule
        ),
    )

    observations = (
        base.collect_attacker_observations(
            architecture,
            attacker_metadata,
            "attacker_only",
        )
    )

    return (
        architecture,
        observations,
    )


def execute_victim_present(
    *,
    victim_mapping: dict[int, str],
    num_qubits: int,
    attacker_mapping: dict[int, str],
    victim_schedule: list[dict],
    attacker_schedule: list[dict],
    attacker_metadata: dict[int, dict],
) -> tuple[
    Any,
    pd.DataFrame,
]:
    """Run the victim and attacker together."""

    architecture = base.build_architecture(
        victim_mapping,
        num_qubits,
        attacker_mapping,
    )

    merged_schedule = (
        copy.deepcopy(
            victim_schedule
        )
        + copy.deepcopy(
            attacker_schedule
        )
    )

    base.execute_timed_schedule(
        architecture,
        merged_schedule,
    )

    observations = (
        base.collect_attacker_observations(
            architecture,
            attacker_metadata,
            "victim_present",
        )
    )

    return (
        architecture,
        observations,
    )


# ============================================================
# Clean templates
# ============================================================

def generate_clean_template(
    *,
    label: str,
    num_qubits: int,
    qasm_path: Path,
) -> dict[str, Any]:
    """Generate one clean exact-start timing template."""

    print(
        f"\nGenerating clean template: {label}"
    )

    circuit = (
        QuantumCircuit.from_qasm_file(
            str(qasm_path)
        )
    )

    (
        victim_trace,
        victim_mapping,
        parsed_num_qubits,
        cross_operation_count,
    ) = base.extract_static_victim_trace(
        str(qasm_path)
    )

    if parsed_num_qubits != num_qubits:
        raise ValueError(
            f"{qasm_path.name} contains "
            f"{parsed_num_qubits} qubits, "
            f"expected {num_qubits}."
        )

    (
        nominal_attacker_schedule,
        attacker_mapping,
        nominal_attacker_metadata,
        schedule_metadata,
    ) = base.build_observation_window_schedule(
        {
            "window_name": (
                f"{label}_clean"
            ),
            "observation_duration_ns": (
                OBSERVATION_DURATION_NS
            ),
        },
        num_qubits,
    )

    victim_schedule = (
        jitter_victim_schedule(
            victim_trace,
            start_ns=VICTIM_START_NS,
            jitter_std_ns=0.0,
            rng=np.random.default_rng(
                GLOBAL_SEED
            ),
        )
    )

    (
        attacker_architecture,
        attacker_only,
    ) = execute_attacker_only(
        victim_mapping=victim_mapping,
        num_qubits=num_qubits,
        attacker_mapping=attacker_mapping,
        attacker_schedule=(
            nominal_attacker_schedule
        ),
        attacker_metadata=(
            nominal_attacker_metadata
        ),
    )

    (
        victim_architecture,
        victim_present,
    ) = execute_victim_present(
        victim_mapping=victim_mapping,
        num_qubits=num_qubits,
        attacker_mapping=attacker_mapping,
        victim_schedule=victim_schedule,
        attacker_schedule=(
            nominal_attacker_schedule
        ),
        attacker_metadata=(
            nominal_attacker_metadata
        ),
    )

    compared = compare_and_threshold(
        attacker_only,
        victim_present,
        threshold_ns=0.0,
    )

    clean_trace = compared[
        "excess_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    structure = qfp.structure_row(
        label,
        num_qubits,
        qasm_path,
        circuit,
        victim_trace,
        cross_operation_count,
    )

    return {
        "label": label,
        "num_qubits": num_qubits,
        "qasm_path": qasm_path,
        "victim_trace": victim_trace,
        "victim_mapping": victim_mapping,
        "attacker_mapping": attacker_mapping,
        "nominal_attacker_schedule": (
            nominal_attacker_schedule
        ),
        "nominal_attacker_metadata": (
            nominal_attacker_metadata
        ),
        "schedule_metadata": (
            schedule_metadata
        ),
        "clean_attacker_only": (
            attacker_only
        ),
        "clean_victim_present": (
            victim_present
        ),
        "clean_compared": compared,
        "clean_trace": clean_trace,
        "structure": structure,
        "clean_attacker_hub_makespan_ns": int(
            attacker_architecture
            .hub.current_time_ns
        ),
        "clean_victim_hub_makespan_ns": int(
            victim_architecture
            .hub.current_time_ns
        ),
    }


def save_clean_templates(
    templates: dict[str, dict[str, Any]],
) -> None:
    """Save clean traces and pairwise clean-template distances."""

    trace_length_set = {
        len(
            template[
                "clean_trace"
            ]
        )
        for template
        in templates.values()
    }

    if len(trace_length_set) != 1:
        raise RuntimeError(
            "Clean template lengths differ: "
            f"{trace_length_set}"
        )

    clean_dataframe = pd.DataFrame(
        [
            {
                "qaoa_label": label,
                "num_qubits": (
                    templates[
                        label
                    ][
                        "num_qubits"
                    ]
                ),
                **{
                    (
                        f"probe_{probe_id:03d}_"
                        "excess_ns"
                    ): float(
                        value
                    )
                    for probe_id, value
                    in enumerate(
                        templates[
                            label
                        ][
                            "clean_trace"
                        ]
                    )
                },
            }
            for label in LABELS
        ]
    )

    clean_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_clean_templates.csv",
        index=False,
    )

    distance_matrix = np.zeros(
        (
            len(LABELS),
            len(LABELS),
        ),
        dtype=float,
    )

    for first_index, first_label in enumerate(
        LABELS
    ):
        for second_index, second_label in enumerate(
            LABELS
        ):
            distance_matrix[
                first_index,
                second_index,
            ] = float(
                np.mean(
                    np.abs(
                        templates[
                            first_label
                        ][
                            "clean_trace"
                        ]
                        - templates[
                            second_label
                        ][
                            "clean_trace"
                        ]
                    )
                )
            )

    distance_dataframe = pd.DataFrame(
        distance_matrix,
        index=LABELS,
        columns=LABELS,
    )

    distance_dataframe.index.name = (
        "qaoa_label"
    )

    distance_dataframe.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_clean_template_"
            "pairwise_mae.csv"
        )
    )


# ============================================================
# One noisy trial
# ============================================================

def run_noisy_trial(
    *,
    template: dict[str, Any],
    profile: dict[str, Any],
    profile_index: int,
    trial_id: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
]:
    """Run one noisy observation of one supplied QAOA circuit."""

    label = str(
        template[
            "label"
        ]
    )

    num_qubits = int(
        template[
            "num_qubits"
        ]
    )

    scheduler_std_ns = float(
        profile[
            "scheduler_jitter_std_ns"
        ]
    )

    timestamp_std_ns = float(
        profile[
            "timestamp_noise_std_ns"
        ]
    )

    threshold_ns = (
        detection_threshold_ns(
            timestamp_std_ns
        )
    )

    attacker_rng = np.random.default_rng(
        trial_seed(
            profile_index=profile_index,
            trial_id=trial_id,
            num_qubits=num_qubits,
            stream_id=1,
        )
    )

    victim_rng = np.random.default_rng(
        trial_seed(
            profile_index=profile_index,
            trial_id=trial_id,
            num_qubits=num_qubits,
            stream_id=2,
        )
    )

    attacker_timestamp_rng = (
        np.random.default_rng(
            trial_seed(
                profile_index=profile_index,
                trial_id=trial_id,
                num_qubits=num_qubits,
                stream_id=3,
            )
        )
    )

    victim_timestamp_rng = (
        np.random.default_rng(
            trial_seed(
                profile_index=profile_index,
                trial_id=trial_id,
                num_qubits=num_qubits,
                stream_id=4,
            )
        )
    )

    (
        attacker_schedule,
        attacker_metadata,
    ) = jitter_attacker_schedule(
        template[
            "nominal_attacker_schedule"
        ],
        template[
            "nominal_attacker_metadata"
        ],
        jitter_std_ns=(
            scheduler_std_ns
        ),
        rng=attacker_rng,
    )

    victim_schedule = (
        jitter_victim_schedule(
            template[
                "victim_trace"
            ],
            start_ns=VICTIM_START_NS,
            jitter_std_ns=(
                scheduler_std_ns
            ),
            rng=victim_rng,
        )
    )

    (
        attacker_architecture,
        attacker_only_true,
    ) = execute_attacker_only(
        victim_mapping=(
            template[
                "victim_mapping"
            ]
        ),
        num_qubits=num_qubits,
        attacker_mapping=(
            template[
                "attacker_mapping"
            ]
        ),
        attacker_schedule=(
            attacker_schedule
        ),
        attacker_metadata=(
            attacker_metadata
        ),
    )

    (
        victim_architecture,
        victim_present_true,
    ) = execute_victim_present(
        victim_mapping=(
            template[
                "victim_mapping"
            ]
        ),
        num_qubits=num_qubits,
        attacker_mapping=(
            template[
                "attacker_mapping"
            ]
        ),
        victim_schedule=(
            victim_schedule
        ),
        attacker_schedule=(
            attacker_schedule
        ),
        attacker_metadata=(
            attacker_metadata
        ),
    )

    attacker_only_measured = (
        add_timestamp_noise(
            attacker_only_true,
            timestamp_noise_std_ns=(
                timestamp_std_ns
            ),
            rng=attacker_timestamp_rng,
        )
    )

    victim_present_measured = (
        add_timestamp_noise(
            victim_present_true,
            timestamp_noise_std_ns=(
                timestamp_std_ns
            ),
            rng=victim_timestamp_rng,
        )
    )

    compared = compare_and_threshold(
        attacker_only_measured,
        victim_present_measured,
        threshold_ns=threshold_ns,
    )

    features = qfp.blackbox_features(
        compared
    )

    feature_row = {
        "qaoa_label": label,
        "num_qubits": num_qubits,
        "profile_name": (
            profile[
                "profile_name"
            ]
        ),
        "trial_id": trial_id,
        "scheduler_jitter_std_ns": (
            scheduler_std_ns
        ),
        "timestamp_noise_std_ns": (
            timestamp_std_ns
        ),
        "detection_threshold_ns": (
            threshold_ns
        ),
        **features,
    }

    raw_excess = compared[
        "raw_excess_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    thresholded_excess = compared[
        "excess_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    summary_row = {
        "qaoa_label": label,
        "num_qubits": num_qubits,
        "profile_name": (
            profile[
                "profile_name"
            ]
        ),
        "trial_id": trial_id,
        "scheduler_jitter_std_ns": (
            scheduler_std_ns
        ),
        "timestamp_noise_std_ns": (
            timestamp_std_ns
        ),
        "detection_threshold_ns": (
            threshold_ns
        ),
        "total_attacker_probes": int(
            len(compared)
        ),
        "baseline_true_avg_waiting_ns": float(
            attacker_only_true[
                "waiting_time_ns"
            ].mean()
        ),
        "baseline_true_max_waiting_ns": float(
            attacker_only_true[
                "waiting_time_ns"
            ].max()
        ),
        "raw_avg_excess_latency_ns": float(
            raw_excess.mean()
        ),
        "raw_std_excess_latency_ns": float(
            raw_excess.std()
        ),
        "thresholded_avg_excess_latency_ns": float(
            thresholded_excess.mean()
        ),
        "thresholded_max_excess_latency_ns": float(
            thresholded_excess.max()
        ),
        "thresholded_total_excess_latency_ns": float(
            thresholded_excess.sum()
        ),
        "detected_probe_count": int(
            compared[
                "victim_contention_observed"
            ].sum()
        ),
        "detected_contention_fraction": float(
            compared[
                "victim_contention_observed"
            ].mean()
        ),
        "attacker_hub_makespan_ns": int(
            attacker_architecture
            .hub.current_time_ns
        ),
        "victim_present_hub_makespan_ns": int(
            victim_architecture
            .hub.current_time_ns
        ),
        "cross_module_operation_count_"
        "evaluator_only": int(
            template[
                "structure"
            ][
                "cross_module_operation_count_"
                "evaluator_only"
            ]
        ),
    }

    observation_rows = compared.copy()

    observation_rows.insert(
        0,
        "trial_id",
        trial_id,
    )

    observation_rows.insert(
        0,
        "profile_name",
        profile[
            "profile_name"
        ],
    )

    observation_rows.insert(
        0,
        "num_qubits",
        num_qubits,
    )

    observation_rows.insert(
        0,
        "qaoa_label",
        label,
    )

    return (
        summary_row,
        feature_row,
        observation_rows,
    )


# ============================================================
# Clean-profile sample rows
# ============================================================

def clean_sample_rows(
    template: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
]:
    """Convert the clean template into the clean-profile sample."""

    label = str(
        template[
            "label"
        ]
    )

    num_qubits = int(
        template[
            "num_qubits"
        ]
    )

    compared = template[
        "clean_compared"
    ].copy()

    features = qfp.blackbox_features(
        compared
    )

    feature_row = {
        "qaoa_label": label,
        "num_qubits": num_qubits,
        "profile_name": "clean",
        "trial_id": 0,
        "scheduler_jitter_std_ns": 0.0,
        "timestamp_noise_std_ns": 0.0,
        "detection_threshold_ns": 0.0,
        **features,
    }

    excess = compared[
        "excess_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    summary_row = {
        "qaoa_label": label,
        "num_qubits": num_qubits,
        "profile_name": "clean",
        "trial_id": 0,
        "scheduler_jitter_std_ns": 0.0,
        "timestamp_noise_std_ns": 0.0,
        "detection_threshold_ns": 0.0,
        "total_attacker_probes": int(
            len(compared)
        ),
        "baseline_true_avg_waiting_ns": float(
            template[
                "clean_attacker_only"
            ][
                "waiting_time_ns"
            ].mean()
        ),
        "baseline_true_max_waiting_ns": float(
            template[
                "clean_attacker_only"
            ][
                "waiting_time_ns"
            ].max()
        ),
        "raw_avg_excess_latency_ns": float(
            excess.mean()
        ),
        "raw_std_excess_latency_ns": float(
            excess.std()
        ),
        "thresholded_avg_excess_latency_ns": float(
            excess.mean()
        ),
        "thresholded_max_excess_latency_ns": float(
            excess.max()
        ),
        "thresholded_total_excess_latency_ns": float(
            excess.sum()
        ),
        "detected_probe_count": int(
            compared[
                "victim_contention_observed"
            ].sum()
        ),
        "detected_contention_fraction": float(
            compared[
                "victim_contention_observed"
            ].mean()
        ),
        "attacker_hub_makespan_ns": int(
            template[
                "clean_attacker_hub_makespan_ns"
            ]
        ),
        "victim_present_hub_makespan_ns": int(
            template[
                "clean_victim_hub_makespan_ns"
            ]
        ),
        "cross_module_operation_count_"
        "evaluator_only": int(
            template[
                "structure"
            ][
                "cross_module_operation_count_"
                "evaluator_only"
            ]
        ),
    }

    observation_rows = compared.copy()

    observation_rows.insert(
        0,
        "trial_id",
        0,
    )

    observation_rows.insert(
        0,
        "profile_name",
        "clean",
    )

    observation_rows.insert(
        0,
        "num_qubits",
        num_qubits,
    )

    observation_rows.insert(
        0,
        "qaoa_label",
        label,
    )

    return (
        summary_row,
        feature_row,
        observation_rows,
    )


# ============================================================
# Confusion-matrix plotting
# ============================================================

def save_confusion_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    filename: str,
) -> None:
    """Save one normalized-looking raw-count confusion matrix."""

    try:
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
        )
    except ImportError:
        return

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    ).plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        title
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_DIR
        / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# Nearest-clean-template classification
# ============================================================

def nearest_template_classification(
    feature_dataframe: pd.DataFrame,
    templates: dict[str, dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Identify every sample by minimum mean absolute distance
    to one of the 11 clean timing templates.
    """

    probe_columns = sorted(
        [
            column
            for column
            in feature_dataframe.columns
            if (
                column.startswith(
                    "bb_probe_"
                )
                and column.endswith(
                    "_excess_ns"
                )
            )
        ]
    )

    template_matrix = {
        label: np.asarray(
            templates[
                label
            ][
                "clean_trace"
            ],
            dtype=float,
        )
        for label in LABELS
    }

    prediction_rows: list[dict] = []

    for _, row in feature_dataframe.iterrows():
        observed_trace = row[
            probe_columns
        ].to_numpy(
            dtype=float
        )

        distances = {
            label: float(
                np.mean(
                    np.abs(
                        observed_trace
                        - template_matrix[
                            label
                        ]
                    )
                )
            )
            for label in LABELS
        }

        predicted_label = min(
            distances,
            key=distances.get,
        )

        sorted_distances = sorted(
            distances.items(),
            key=lambda item: item[1],
        )

        best_distance = float(
            sorted_distances[0][1]
        )

        second_distance = float(
            sorted_distances[1][1]
        )

        prediction_rows.append(
            {
                "qaoa_label": row[
                    "qaoa_label"
                ],
                "num_qubits": int(
                    row[
                        "num_qubits"
                    ]
                ),
                "profile_name": row[
                    "profile_name"
                ],
                "trial_id": int(
                    row[
                        "trial_id"
                    ]
                ),
                "predicted_qaoa_label": (
                    predicted_label
                ),
                "correct": (
                    predicted_label
                    == row[
                        "qaoa_label"
                    ]
                ),
                "best_template_mae_ns": (
                    best_distance
                ),
                "second_best_template_mae_ns": (
                    second_distance
                ),
                "template_margin_ns": (
                    second_distance
                    - best_distance
                ),
            }
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    metric_rows: list[dict] = []

    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
        )
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required."
        ) from error

    for profile_name in PROFILE_ORDER:
        profile_predictions = predictions[
            predictions[
                "profile_name"
            ]
            == profile_name
        ]

        true_labels = profile_predictions[
            "qaoa_label"
        ].to_numpy()

        predicted_labels = profile_predictions[
            "predicted_qaoa_label"
        ].to_numpy()

        matrix = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=LABELS,
        )

        accuracy = accuracy_score(
            true_labels,
            predicted_labels,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        )

        macro_f1 = f1_score(
            true_labels,
            predicted_labels,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        metric_rows.append(
            {
                "profile_name": (
                    profile_name
                ),
                "method": (
                    "nearest_clean_template"
                ),
                "sample_count": int(
                    len(
                        profile_predictions
                    )
                ),
                "chance_accuracy": (
                    1.0
                    / len(LABELS)
                ),
                "accuracy": float(
                    accuracy
                ),
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
                "macro_f1": float(
                    macro_f1
                ),
                "mean_template_margin_ns": float(
                    profile_predictions[
                        "template_margin_ns"
                    ].mean()
                ),
            }
        )

        save_confusion_matrix(
            matrix,
            title=(
                "Nearest Clean Template\n"
                f"{profile_name}"
            ),
            filename=(
                f"nearest_template_"
                f"{profile_name}.png"
            ),
        )

    return (
        predictions,
        pd.DataFrame(
            metric_rows
        ),
    )


# ============================================================
# Random-forest classification
# ============================================================

def random_forest_classification(
    feature_dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Evaluate each noisy profile independently.

    LeaveOneGroupOut holds out one complete trial ID,
    containing one sample from every QAOA circuit.
    """

    try:
        from sklearn.ensemble import (
            RandomForestClassifier,
        )
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
        from sklearn.model_selection import (
            LeaveOneGroupOut,
            cross_val_predict,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install scikit-learn before "
            "running this script."
        ) from error

    feature_columns = [
        column
        for column
        in feature_dataframe.columns
        if column.startswith(
            "bb_"
        )
    ]

    all_prediction_rows: list[
        pd.DataFrame
    ] = []

    metric_rows: list[dict] = []

    metrics_json: dict[str, Any] = {
        "task": (
            "identify_exact_supplied_qaoa_"
            "circuit_under_timing_noise"
        ),
        "classes": LABELS,
        "chance_accuracy": (
            1.0
            / len(LABELS)
        ),
        "validation": (
            "leave-one-trial-out within "
            "each noise profile"
        ),
        "profiles": {},
    }

    for profile in NOISE_PROFILES:
        profile_name = profile[
            "profile_name"
        ]

        if int(
            profile[
                "trial_count"
            ]
        ) < 2:
            continue

        profile_data = (
            feature_dataframe[
                feature_dataframe[
                    "profile_name"
                ]
                == profile_name
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        features = (
            profile_data[
                feature_columns
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(
                0.0
            )
            .to_numpy(
                dtype=float
            )
        )

        labels = profile_data[
            "qaoa_label"
        ].to_numpy()

        groups = profile_data[
            "trial_id"
        ].to_numpy()

        classifier = (
            RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                class_weight="balanced",
                random_state=GLOBAL_SEED,
                n_jobs=-1,
            )
        )

        predictions = cross_val_predict(
            classifier,
            features,
            labels,
            groups=groups,
            cv=LeaveOneGroupOut(),
            n_jobs=-1,
        )

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=LABELS,
        )

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                labels,
                predictions,
            )
        )

        macro_f1 = f1_score(
            labels,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            labels,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )

        prediction_dataframe = (
            profile_data[
                [
                    "qaoa_label",
                    "num_qubits",
                    "profile_name",
                    "trial_id",
                ]
            ]
            .copy()
        )

        prediction_dataframe[
            "predicted_qaoa_label"
        ] = predictions

        prediction_dataframe[
            "correct"
        ] = (
            prediction_dataframe[
                "qaoa_label"
            ]
            == prediction_dataframe[
                "predicted_qaoa_label"
            ]
        )

        all_prediction_rows.append(
            prediction_dataframe
        )

        metric_rows.append(
            {
                "profile_name": (
                    profile_name
                ),
                "method": (
                    "random_forest"
                ),
                "sample_count": int(
                    len(
                        profile_data
                    )
                ),
                "trial_count": int(
                    profile_data[
                        "trial_id"
                    ].nunique()
                ),
                "feature_count": int(
                    len(
                        feature_columns
                    )
                ),
                "chance_accuracy": (
                    1.0
                    / len(LABELS)
                ),
                "accuracy": float(
                    accuracy
                ),
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
                "macro_f1": float(
                    macro_f1
                ),
            }
        )

        metrics_json[
            "profiles"
        ][
            profile_name
        ] = {
            "sample_count": int(
                len(
                    profile_data
                )
            ),
            "trial_count": int(
                profile_data[
                    "trial_id"
                ].nunique()
            ),
            "feature_count": int(
                len(
                    feature_columns
                )
            ),
            "accuracy": float(
                accuracy
            ),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
            "macro_f1": float(
                macro_f1
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
            "classification_report": (
                report
            ),
        }

        save_confusion_matrix(
            matrix,
            title=(
                "Random Forest: "
                "Leave-One-Trial-Out\n"
                f"{profile_name}"
            ),
            filename=(
                f"random_forest_"
                f"{profile_name}.png"
            ),
        )

    if all_prediction_rows:
        predictions = pd.concat(
            all_prediction_rows,
            ignore_index=True,
        )
    else:
        predictions = pd.DataFrame()

    return (
        predictions,
        pd.DataFrame(
            metric_rows
        ),
        metrics_json,
    )


# ============================================================
# Aggregate plotting
# ============================================================

def save_accuracy_plot(
    accuracy_dataframe: pd.DataFrame,
) -> None:
    """Compare template and learned identification accuracy."""

    pivot = (
        accuracy_dataframe.pivot(
            index="profile_name",
            columns="method",
            values="accuracy",
        )
        .reindex(
            PROFILE_ORDER
        )
    )

    axis = pivot.plot(
        kind="bar",
        figsize=(14, 6),
    )

    axis.axhline(
        1.0 / len(LABELS),
        linestyle="--",
        linewidth=1,
        label="Random guessing",
    )

    axis.set_xlabel(
        "Noise profile"
    )

    axis.set_ylabel(
        "Circuit-identification accuracy"
    )

    axis.set_title(
        "QAOA Circuit Identification "
        "Under Timing Noise"
    )

    axis.set_ylim(
        0,
        1.05,
    )

    axis.tick_params(
        axis="x",
        rotation=30,
    )

    axis.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            "qaoa_noise_"
            "accuracy_comparison.png"
        ),
        dpi=300,
    )

    plt.close()


def save_profile_signal_plots(
    profile_summary: pd.DataFrame,
) -> None:
    """Save profile-level timing-signal plots."""

    ordered = profile_summary.copy()

    ordered[
        "profile_name"
    ] = pd.Categorical(
        ordered[
            "profile_name"
        ],
        categories=PROFILE_ORDER,
        ordered=True,
    )

    contention = (
        ordered.pivot(
            index="qaoa_label",
            columns="profile_name",
            values=(
                "mean_detected_"
                "contention_fraction"
            ),
        )
        .reindex(
            index=LABELS,
            columns=PROFILE_ORDER,
        )
    )

    axis = contention.plot(
        kind="bar",
        figsize=(15, 6),
    )

    axis.set_xlabel(
        "QAOA circuit"
    )

    axis.set_ylabel(
        "Mean detected contention fraction"
    )

    axis.set_title(
        "Detected QAOA Timing Signal "
        "Under Noise"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.legend(
        title="Noise profile",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            "qaoa_noise_"
            "contention_fraction.png"
        ),
        dpi=300,
    )

    plt.close()

    excess = (
        ordered.pivot(
            index="qaoa_label",
            columns="profile_name",
            values=(
                "mean_thresholded_"
                "excess_latency_ns"
            ),
        )
        .reindex(
            index=LABELS,
            columns=PROFILE_ORDER,
        )
    )

    axis = excess.plot(
        kind="bar",
        figsize=(15, 6),
    )

    axis.set_xlabel(
        "QAOA circuit"
    )

    axis.set_ylabel(
        "Mean thresholded excess latency (ns)"
    )

    axis.set_title(
        "QAOA Excess-Latency Fingerprint "
        "Under Noise"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.legend(
        title="Noise profile",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            "qaoa_noise_"
            "excess_latency.png"
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFUSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qasm_files = (
        qfp.resolve_all_qasms()
    )

    print(
        "Resolved QAOA QASM files:"
    )

    for num_qubits in SIZES:
        print(
            f"  qaoa_n{num_qubits}: "
            f"{qasm_files[num_qubits]}"
        )

    # --------------------------------------------------------
    # Generate clean templates
    # --------------------------------------------------------

    templates: dict[
        str,
        dict[str, Any],
    ] = {}

    structure_rows: list[dict] = []

    for num_qubits in SIZES:
        label = (
            f"qaoa_n{num_qubits}"
        )

        template = (
            generate_clean_template(
                label=label,
                num_qubits=num_qubits,
                qasm_path=(
                    qasm_files[
                        num_qubits
                    ]
                ),
            )
        )

        templates[
            label
        ] = template

        structure_rows.append(
            template[
                "structure"
            ]
        )

    pd.DataFrame(
        structure_rows
    ).sort_values(
        "num_qubits"
    ).to_csv(
        OUTPUT_DIR
        / (
            "qaoa_noise_"
            "structure_summary.csv"
        ),
        index=False,
    )

    save_clean_templates(
        templates
    )

    # --------------------------------------------------------
    # Run robustness trials
    # --------------------------------------------------------

    summary_rows: list[dict] = []

    feature_rows: list[dict] = []

    observation_frames: list[
        pd.DataFrame
    ] = []

    for profile_index, profile in enumerate(
        NOISE_PROFILES
    ):
        profile_name = (
            profile[
                "profile_name"
            ]
        )

        trial_count = int(
            profile[
                "trial_count"
            ]
        )

        print(
            "\n========================================"
        )

        print(
            f"Noise profile: {profile_name}"
        )

        print(
            "Scheduler jitter std: "
            f"{profile['scheduler_jitter_std_ns']} ns"
        )

        print(
            "Timestamp noise std: "
            f"{profile['timestamp_noise_std_ns']} ns"
        )

        print(
            "========================================"
        )

        for label in LABELS:
            template = templates[
                label
            ]

            if profile_name == "clean":
                (
                    summary_row,
                    feature_row,
                    observations,
                ) = clean_sample_rows(
                    template
                )

                summary_rows.append(
                    summary_row
                )

                feature_rows.append(
                    feature_row
                )

                observation_frames.append(
                    observations
                )

                continue

            for trial_id in range(
                trial_count
            ):
                print(
                    f"  {label} | "
                    f"{profile_name} | "
                    f"trial {trial_id:02d}"
                )

                (
                    summary_row,
                    feature_row,
                    observations,
                ) = run_noisy_trial(
                    template=template,
                    profile=profile,
                    profile_index=(
                        profile_index
                    ),
                    trial_id=trial_id,
                )

                summary_rows.append(
                    summary_row
                )

                feature_rows.append(
                    feature_row
                )

                observation_frames.append(
                    observations
                )

    summary_dataframe = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "profile_name",
                "num_qubits",
                "trial_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    feature_dataframe = (
        pd.DataFrame(
            feature_rows
        )
        .sort_values(
            [
                "profile_name",
                "num_qubits",
                "trial_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    observations_dataframe = (
        pd.concat(
            observation_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "profile_name",
                "num_qubits",
                "trial_id",
                "attacker_request_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_noise_trial_summary.csv",
        index=False,
    )

    feature_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_noise_features.csv",
        index=False,
    )

    observations_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_noise_observations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate signal behavior
    # --------------------------------------------------------

    signal_summary = (
        summary_dataframe
        .groupby(
            [
                "profile_name",
                "qaoa_label",
                "num_qubits",
            ],
            as_index=False,
        )
        .agg(
            trial_count=(
                "trial_id",
                "count",
            ),
            mean_baseline_waiting_ns=(
                "baseline_true_avg_waiting_ns",
                "mean",
            ),
            max_baseline_waiting_ns=(
                "baseline_true_max_waiting_ns",
                "max",
            ),
            mean_raw_excess_latency_ns=(
                "raw_avg_excess_latency_ns",
                "mean",
            ),
            std_raw_excess_latency_ns=(
                "raw_avg_excess_latency_ns",
                "std",
            ),
            mean_thresholded_excess_latency_ns=(
                "thresholded_avg_excess_latency_ns",
                "mean",
            ),
            std_thresholded_excess_latency_ns=(
                "thresholded_avg_excess_latency_ns",
                "std",
            ),
            mean_detected_contention_fraction=(
                "detected_contention_fraction",
                "mean",
            ),
            std_detected_contention_fraction=(
                "detected_contention_fraction",
                "std",
            ),
        )
    )

    signal_summary.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_noise_profile_"
            "signal_summary.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Identification methods
    # --------------------------------------------------------

    (
        nearest_predictions,
        nearest_metrics,
    ) = nearest_template_classification(
        feature_dataframe,
        templates,
    )

    nearest_predictions.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_nearest_template_"
            "predictions.csv"
        ),
        index=False,
    )

    nearest_metrics.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_nearest_template_"
            "metrics.csv"
        ),
        index=False,
    )

    (
        rf_predictions,
        rf_metrics,
        rf_metrics_json,
    ) = random_forest_classification(
        feature_dataframe
    )

    rf_predictions.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_random_forest_"
            "predictions.csv"
        ),
        index=False,
    )

    rf_metrics.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_random_forest_"
            "metrics.csv"
        ),
        index=False,
    )

    with (
        OUTPUT_DIR
        / (
            "qaoa_random_forest_"
            "metrics.json"
        )
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            rf_metrics_json,
            output_file,
            indent=2,
            default=clean_json_value,
        )

    accuracy_comparison = (
        pd.concat(
            [
                nearest_metrics[
                    [
                        "profile_name",
                        "method",
                        "sample_count",
                        "chance_accuracy",
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                    ]
                ],
                rf_metrics[
                    [
                        "profile_name",
                        "method",
                        "sample_count",
                        "chance_accuracy",
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                    ]
                ],
            ],
            ignore_index=True,
        )
    )

    accuracy_comparison.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_noise_"
            "accuracy_comparison.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_accuracy_plot(
        accuracy_comparison
    )

    save_profile_signal_plots(
        signal_summary
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\n=== Identification accuracy ==="
    )

    print(
        accuracy_comparison[
            [
                "profile_name",
                "method",
                "sample_count",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRandom-guessing accuracy: "
        f"{1.0 / len(LABELS):.4f}"
    )

    print(
        "\nSaved all results to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
