#!/usr/bin/env python3
"""
Identify the supplied compiled QAOA circuits (qaoa_n5 ... qaoa_n15)
from attacker-visible timing under the selected black-box configuration.

Each QASM is one class. Because there is only one QASM per class, repeated
samples are created by applying the same victim-start offsets to every circuit.
Classification uses leave-one-start-offset-out cross-validation, so every test
fold contains all 11 circuits at an offset that was not used for training.

Keep this script beside:
  run_attack_tier1_p1_static_blackbox_observation_window_sweep.py
  new_arch_fivenode_traceadded.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

import run_atack_tier1_p1_static_blackbox_observation_window_sweep as base


OUTPUT_DIR = Path(
    "blackbox_window_results/qaoa_circuit_fingerprinting"
)

QASM_PREFIX = "qaoa_nativegates_ibm_qiskit_opt3"

QASM_ROOTS = [
    Path("."),
    Path("/mnt/data"),
]

SIZES = list(range(5, 16))

LABELS = [
    f"qaoa_n{num_qubits}"
    for num_qubits in SIZES
]


# ============================================================
# Selected black-box configuration
# ============================================================

VICTIM_REFERENCE_START_NS = 1_000

ATTACKER_ESTIMATED_START_NS = 1_000

OBSERVATION_DURATION_NS = 20_000

PROBE_PERIOD_NS = 420

# Same offsets are used for every QAOA circuit.
START_OFFSETS_NS = [
    -1_000,
    -750,
    -500,
    -250,
    0,
    250,
    500,
    750,
    1_000,
]

NUM_BINS = 8

SEED = 41


base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
    ATTACKER_ESTIMATED_START_NS
)

base.PROBE_ROUND_PERIOD_NS = (
    PROBE_PERIOD_NS
)

base.HUB_MAX_CONCURRENT_TRANSFERS = 1


# ============================================================
# QASM resolution
# ============================================================

def resolve_qasm(
    num_qubits: int,
) -> Path:
    """
    Find the required QASM in the current directory
    or in /mnt/data.
    """

    preferred_names = [
        (
            f"{QASM_PREFIX}_"
            f"{num_qubits}(1).qasm"
        ),
        (
            f"{QASM_PREFIX}_"
            f"{num_qubits}.qasm"
        ),
    ]

    for root in QASM_ROOTS:
        for filename in preferred_names:
            path = root / filename

            if path.exists():
                return path.resolve()

    matches: list[Path] = []

    for root in QASM_ROOTS:
        if root.exists():
            matches.extend(
                root.glob(
                    f"{QASM_PREFIX}_"
                    f"{num_qubits}*.qasm"
                )
            )

    matches = sorted(
        {
            path.resolve()
            for path in matches
        },
        key=lambda path: path.name,
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find the "
            f"{num_qubits}-qubit QAOA QASM."
        )

    return matches[0]


def resolve_all_qasms() -> dict[int, Path]:
    """Resolve and validate all 11 QASM files."""

    qasm_files: dict[int, Path] = {}

    for num_qubits in SIZES:
        path = resolve_qasm(
            num_qubits
        )

        circuit = (
            QuantumCircuit.from_qasm_file(
                str(path)
            )
        )

        if circuit.num_qubits != num_qubits:
            raise ValueError(
                f"{path.name} has "
                f"{circuit.num_qubits} qubits, "
                f"but {num_qubits} were expected."
            )

        qasm_files[num_qubits] = path

    return qasm_files


# ============================================================
# Timing helpers
# ============================================================

def schedule_victim(
    trace: list[dict],
    start_ns: int,
) -> list[dict]:
    """Schedule the victim trace at an explicit start."""

    if start_ns < 0:
        raise ValueError(
            "Victim start cannot be negative: "
            f"{start_ns}"
        )

    return [
        {
            "release_time_ns": (
                start_ns
                + event_index
                * base.VICTIM_EVENT_TICK_NS
            ),
            "tenant": "victim",
            "sequence_index": event_index,
            "entry": copy.deepcopy(
                trace_entry
            ),
        }
        for event_index, trace_entry
        in enumerate(trace)
    ]


def completion_duration(
    dataframe: pd.DataFrame,
    start_ns: int,
) -> float:
    """Return completion time relative to victim start."""

    if dataframe.empty:
        return 0.0

    return float(
        dataframe[
            "completion_time_ns"
        ].max()
        - start_ns
    )


def offset_name(
    offset_ns: int,
) -> str:
    """Create a directory name for one offset."""

    if offset_ns == 0:
        return "offset_exact"

    prefix = (
        "p"
        if offset_ns > 0
        else "m"
    )

    return (
        f"offset_{prefix}"
        f"{abs(offset_ns)}ns"
    )


# ============================================================
# Feature extraction
# ============================================================

def longest_run(
    values: np.ndarray,
) -> int:
    """Return the longest consecutive True run."""

    longest = 0
    current = 0

    for value in values:
        if bool(value):
            current += 1
        else:
            current = 0

        longest = max(
            longest,
            current,
        )

    return longest


def autocorrelation(
    values: np.ndarray,
    lag: int,
) -> float:
    """Compute safe lagged autocorrelation."""

    if len(values) <= lag:
        return 0.0

    first = values[:-lag]
    second = values[lag:]

    if (
        np.std(first) == 0
        or np.std(second) == 0
    ):
        return 0.0

    result = np.corrcoef(
        first,
        second,
    )[0, 1]

    if not np.isfinite(
        result
    ):
        return 0.0

    return float(
        result
    )


def blackbox_features(
    compared: pd.DataFrame,
) -> dict[str, float]:
    """
    Extract features only from attacker-visible
    request timing.
    """

    compared = (
        compared
        .sort_values(
            "attacker_request_id"
        )
        .reset_index(
            drop=True
        )
    )

    excess = compared[
        "excess_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    turnaround = compared[
        "victim_on_turnaround_time_ns"
    ].to_numpy(
        dtype=float
    )

    waiting = compared[
        "victim_on_waiting_time_ns"
    ].to_numpy(
        dtype=float
    )

    delayed = excess > 0

    probe_count = len(
        excess
    )

    if probe_count == 0:
        raise RuntimeError(
            "No attacker observations "
            "were produced."
        )

    features: dict[str, float] = {
        "bb_probe_count": float(
            probe_count
        ),
        "bb_delayed_count": float(
            delayed.sum()
        ),
        "bb_contention_fraction": float(
            delayed.mean()
        ),
        "bb_excess_mean_ns": float(
            excess.mean()
        ),
        "bb_excess_std_ns": float(
            excess.std()
        ),
        "bb_excess_median_ns": float(
            np.median(
                excess
            )
        ),
        "bb_excess_max_ns": float(
            excess.max()
        ),
        "bb_excess_sum_ns": float(
            excess.sum()
        ),
        "bb_turnaround_mean_ns": float(
            turnaround.mean()
        ),
        "bb_turnaround_std_ns": float(
            turnaround.std()
        ),
        "bb_wait_mean_ns": float(
            waiting.mean()
        ),
        "bb_wait_max_ns": float(
            waiting.max()
        ),
        "bb_longest_contention_run": float(
            longest_run(
                delayed
            )
        ),
    }

    for percentile in [
        10,
        25,
        50,
        75,
        90,
        95,
    ]:
        features[
            f"bb_excess_p{percentile}_ns"
        ] = float(
            np.percentile(
                excess,
                percentile,
            )
        )

    delayed_indices = np.flatnonzero(
        delayed
    )

    if len(delayed_indices) > 0:
        denominator = max(
            probe_count - 1,
            1,
        )

        features[
            "bb_first_delayed_position"
        ] = float(
            delayed_indices[0]
            / denominator
        )

        features[
            "bb_last_delayed_position"
        ] = float(
            delayed_indices[-1]
            / denominator
        )

        features[
            "bb_contention_span_fraction"
        ] = float(
            (
                delayed_indices[-1]
                - delayed_indices[0]
                + 1
            )
            / probe_count
        )

    else:
        features[
            "bb_first_delayed_position"
        ] = 1.0

        features[
            "bb_last_delayed_position"
        ] = 0.0

        features[
            "bb_contention_span_fraction"
        ] = 0.0

    transition_count = int(
        np.count_nonzero(
            delayed[1:]
            != delayed[:-1]
        )
    )

    features[
        "bb_contention_transition_count"
    ] = float(
        transition_count
    )

    features[
        "bb_contention_transition_rate"
    ] = float(
        transition_count
        / max(
            probe_count - 1,
            1,
        )
    )

    positive_excess = np.maximum(
        excess,
        0.0,
    )

    normalized_positions = (
        np.arange(
            probe_count,
            dtype=float,
        )
        / max(
            probe_count - 1,
            1,
        )
    )

    if positive_excess.sum() > 0:
        features[
            "bb_delay_centroid"
        ] = float(
            np.average(
                normalized_positions,
                weights=positive_excess,
            )
        )
    else:
        features[
            "bb_delay_centroid"
        ] = 0.0

    if probe_count > 1:
        features[
            "bb_excess_linear_slope"
        ] = float(
            np.polyfit(
                normalized_positions,
                excess,
                1,
            )[0]
        )
    else:
        features[
            "bb_excess_linear_slope"
        ] = 0.0

    for lag in range(
        1,
        6,
    ):
        features[
            f"bb_excess_autocorr_lag_{lag}"
        ] = autocorrelation(
            excess,
            lag,
        )

    temporal_bins = np.array_split(
        np.arange(
            probe_count
        ),
        NUM_BINS,
    )

    for bin_id, bin_indices in enumerate(
        temporal_bins
    ):
        if len(bin_indices) > 0:
            bin_excess = excess[
                bin_indices
            ]

            bin_delayed = delayed[
                bin_indices
            ]

        else:
            bin_excess = np.array(
                [0.0]
            )

            bin_delayed = np.array(
                [False]
            )

        features[
            f"bb_bin_{bin_id:02d}_mean_ns"
        ] = float(
            bin_excess.mean()
        )

        features[
            f"bb_bin_{bin_id:02d}_max_ns"
        ] = float(
            bin_excess.max()
        )

        features[
            f"bb_bin_{bin_id:02d}_sum_ns"
        ] = float(
            bin_excess.sum()
        )

        features[
            f"bb_bin_{bin_id:02d}_"
            "contention_fraction"
        ] = float(
            bin_delayed.mean()
        )

    # Preserve the full attacker-visible temporal trace.
    for probe_id, value in enumerate(
        excess
    ):
        features[
            f"bb_probe_{probe_id:03d}_"
            "excess_ns"
        ] = float(
            value
        )

    return features


# ============================================================
# Evaluator metadata
# ============================================================

def save_trace_metadata(
    trace: list[dict],
    path: Path,
) -> None:
    """Save the compiled victim trace."""

    rows: list[dict] = []

    for event in trace:
        rows.append(
            {
                "step": event[
                    "step"
                ],
                "op_name": event[
                    "op_name"
                ],
                "qubits": ",".join(
                    map(
                        str,
                        event["qubits"],
                    )
                ),
                "modules_touched": ",".join(
                    event[
                        "modules_touched"
                    ]
                ),
                "is_cross_module": bool(
                    event[
                        "is_cross_module"
                    ]
                ),
                "params": json.dumps(
                    event[
                        "params"
                    ]
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False,
    )


def structure_row(
    label: str,
    num_qubits: int,
    path: Path,
    circuit: QuantumCircuit,
    trace: list[dict],
    cross_count: int,
) -> dict[str, Any]:
    """Create structural metadata for one QASM."""

    counts = {
        str(name): int(count)
        for name, count
        in circuit.count_ops().items()
    }

    return {
        "qaoa_label": label,
        "num_qubits": num_qubits,
        "qasm_file": path.name,
        "qasm_path": str(
            path
        ),
        "circuit_depth_evaluator_only": int(
            circuit.depth()
        ),
        "circuit_size_evaluator_only": int(
            circuit.size()
        ),
        "trace_event_count_evaluator_only": (
            len(trace)
        ),
        "cross_module_operation_count_"
        "evaluator_only": int(
            cross_count
        ),
        "cx_count_evaluator_only": int(
            counts.get(
                "cx",
                0,
            )
        ),
        "rz_count_evaluator_only": int(
            counts.get(
                "rz",
                0,
            )
        ),
        "sx_count_evaluator_only": int(
            counts.get(
                "sx",
                0,
            )
        ),
        "x_count_evaluator_only": int(
            counts.get(
                "x",
                0,
            )
        ),
        "measure_count_evaluator_only": int(
            counts.get(
                "measure",
                0,
            )
        ),
        "operation_counts_json_"
        "evaluator_only": json.dumps(
            counts,
            sort_keys=True,
        ),
    }


def trial_summary(
    label: str,
    num_qubits: int,
    path: Path,
    offset_ns: int,
    actual_start_ns: int,
    compared: pd.DataFrame,
    attacker_only: pd.DataFrame,
    victim_only_duration: float,
    victim_on_truth: pd.DataFrame,
    structure: dict[str, Any],
) -> dict[str, Any]:
    """Create one offset-trial summary."""

    victim_on_duration = (
        completion_duration(
            victim_on_truth,
            actual_start_ns,
        )
    )

    slowdown = (
        victim_on_duration
        - victim_only_duration
    )

    if victim_only_duration > 0:
        slowdown_ratio = (
            victim_on_duration
            / victim_only_duration
        )
    else:
        slowdown_ratio = 1.0

    return {
        "qaoa_label": label,
        "num_qubits": num_qubits,
        "qasm_file": path.name,
        "start_offset_ns": offset_ns,
        "actual_victim_start_ns": (
            actual_start_ns
        ),
        "attacker_estimated_start_ns": (
            ATTACKER_ESTIMATED_START_NS
        ),
        "observation_duration_ns": (
            OBSERVATION_DURATION_NS
        ),
        "probe_round_period_ns": (
            PROBE_PERIOD_NS
        ),
        "probe_name": (
            "probe_3_light_periodic"
        ),
        "spacing_pattern": "uniform",
        "placement": "P1_disjoint",
        "execution_mode": (
            "static_distributed"
        ),
        "hub_max_concurrent_transfers": 1,
        "total_attacker_probes": int(
            len(compared)
        ),
        "baseline_avg_waiting_time_ns": float(
            attacker_only[
                "waiting_time_ns"
            ].mean()
        ),
        "baseline_max_waiting_time_ns": float(
            attacker_only[
                "waiting_time_ns"
            ].max()
        ),
        "avg_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].mean()
        ),
        "median_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].median()
        ),
        "max_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].max()
        ),
        "total_excess_turnaround_time_ns": float(
            compared[
                "excess_turnaround_time_ns"
            ].sum()
        ),
        "delayed_probe_count": int(
            compared[
                "victim_contention_observed"
            ].sum()
        ),
        "contention_observed_fraction": float(
            compared[
                "victim_contention_observed"
            ].mean()
        ),
        "victim_only_duration_ns": float(
            victim_only_duration
        ),
        "victim_on_duration_ns": float(
            victim_on_duration
        ),
        "victim_slowdown_ns": float(
            slowdown
        ),
        "victim_slowdown_ratio": float(
            slowdown_ratio
        ),
        "trace_event_count_evaluator_only": int(
            structure[
                "trace_event_count_evaluator_only"
            ]
        ),
        "cross_module_operation_count_"
        "evaluator_only": int(
            structure[
                "cross_module_operation_count_"
                "evaluator_only"
            ]
        ),
        "cx_count_evaluator_only": int(
            structure[
                "cx_count_evaluator_only"
            ]
        ),
        "circuit_depth_evaluator_only": int(
            structure[
                "circuit_depth_evaluator_only"
            ]
        ),
    }


# ============================================================
# Run one QAOA circuit
# ============================================================

def run_circuit(
    num_qubits: int,
    path: Path,
):
    """Run every start-offset trial for one QASM."""

    label = (
        f"qaoa_n{num_qubits}"
    )

    circuit_directory = (
        OUTPUT_DIR / label
    )

    circuit_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\n=== {label}: "
        f"{path.name} ==="
    )

    circuit = (
        QuantumCircuit.from_qasm_file(
            str(path)
        )
    )

    (
        trace,
        victim_mapping,
        parsed_num_qubits,
        cross_count,
    ) = base.extract_static_victim_trace(
        str(path)
    )

    if parsed_num_qubits != num_qubits:
        raise ValueError(
            f"Parsed {parsed_num_qubits} "
            f"qubits from {path.name}, "
            f"expected {num_qubits}."
        )

    structure = structure_row(
        label,
        num_qubits,
        path,
        circuit,
        trace,
        cross_count,
    )

    save_trace_metadata(
        trace,
        circuit_directory
        / "compiled_victim_trace.csv",
    )

    with (
        circuit_directory
        / "circuit_structure.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            structure,
            output_file,
            indent=2,
        )

    # --------------------------------------------------------
    # Fixed attacker schedule
    # --------------------------------------------------------

    base.ATTACKER_ESTIMATED_WINDOW_START_NS = (
        ATTACKER_ESTIMATED_START_NS
    )

    (
        attacker_schedule,
        attacker_mapping,
        step_metadata,
        schedule_metadata,
    ) = base.build_observation_window_schedule(
        {
            "window_name": label,
            "observation_duration_ns": (
                OBSERVATION_DURATION_NS
            ),
        },
        num_qubits,
    )

    with (
        circuit_directory
        / "attacker_schedule_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            schedule_metadata,
            output_file,
            indent=2,
        )

    # --------------------------------------------------------
    # Attacker-only calibration
    # --------------------------------------------------------

    attacker_architecture = (
        base.build_architecture(
            victim_mapping,
            num_qubits,
            attacker_mapping,
        )
    )

    base.execute_timed_schedule(
        attacker_architecture,
        copy.deepcopy(
            attacker_schedule
        ),
    )

    attacker_only = (
        base.collect_attacker_observations(
            attacker_architecture,
            step_metadata,
            "attacker_only",
        )
    )

    attacker_only.to_csv(
        circuit_directory
        / "attacker_only_calibration.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Victim-only control
    # --------------------------------------------------------

    victim_only_schedule = schedule_victim(
        trace,
        VICTIM_REFERENCE_START_NS,
    )

    victim_only_architecture = (
        base.build_architecture(
            victim_mapping,
            num_qubits,
            attacker_mapping,
        )
    )

    base.execute_timed_schedule(
        victim_only_architecture,
        copy.deepcopy(
            victim_only_schedule
        ),
    )

    victim_only_truth = (
        base.collect_victim_ground_truth(
            victim_only_architecture,
            "victim_only",
        )
    )

    victim_only_duration = (
        completion_duration(
            victim_only_truth,
            VICTIM_REFERENCE_START_NS,
        )
    )

    victim_only_truth.to_csv(
        circuit_directory
        / "victim_only_ground_truth.csv",
        index=False,
    )

    summary_rows: list[dict] = []

    feature_rows: list[dict] = []

    exact_trace: np.ndarray | None = None

    # --------------------------------------------------------
    # Start-offset trials
    # --------------------------------------------------------

    for offset_ns in START_OFFSETS_NS:
        actual_start_ns = (
            VICTIM_REFERENCE_START_NS
            + offset_ns
        )

        trial_directory = (
            circuit_directory
            / offset_name(
                offset_ns
            )
        )

        trial_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"  offset {offset_ns:+6d} ns"
        )

        victim_schedule = schedule_victim(
            trace,
            actual_start_ns,
        )

        victim_on_architecture = (
            base.build_architecture(
                victim_mapping,
                num_qubits,
                attacker_mapping,
            )
        )

        base.execute_timed_schedule(
            victim_on_architecture,
            copy.deepcopy(
                victim_schedule
            )
            + copy.deepcopy(
                attacker_schedule
            ),
        )

        victim_present = (
            base.collect_attacker_observations(
                victim_on_architecture,
                step_metadata,
                "victim_present",
            )
        )

        victim_on_truth = (
            base.collect_victim_ground_truth(
                victim_on_architecture,
                "victim_present",
            )
        )

        compared = (
            base.compare_attacker_runs(
                attacker_only,
                victim_present,
            )
        )

        summary = trial_summary(
            label,
            num_qubits,
            path,
            offset_ns,
            actual_start_ns,
            compared,
            attacker_only,
            victim_only_duration,
            victim_on_truth,
            structure,
        )

        features = {
            "qaoa_label": label,
            "num_qubits": num_qubits,
            "qasm_file": path.name,
            "start_offset_ns": offset_ns,
            "actual_victim_start_ns": (
                actual_start_ns
            ),
            **blackbox_features(
                compared
            ),
        }

        compared.to_csv(
            trial_directory
            / "attacker_observations.csv",
            index=False,
        )

        victim_on_truth.to_csv(
            trial_directory
            / "victim_on_ground_truth.csv",
            index=False,
        )

        with (
            trial_directory
            / "trial_summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                summary,
                output_file,
                indent=2,
            )

        if offset_ns == 0:
            exact_trace = compared[
                "excess_turnaround_time_ns"
            ].to_numpy(
                dtype=float
            )

            plt.figure(
                figsize=(13, 5)
            )

            plt.plot(
                compared[
                    "request_release_time_ns"
                ],
                exact_trace,
                marker="o",
                markersize=3,
                linewidth=1,
            )

            plt.axhline(
                0,
                linewidth=1,
            )

            plt.xlabel(
                "Attacker remote-probe "
                "release time (ns)"
            )

            plt.ylabel(
                "Victim-induced "
                "excess latency (ns)"
            )

            plt.title(
                f"{label}: exact-start "
                "black-box fingerprint"
            )

            plt.tight_layout()

            plt.savefig(
                circuit_directory
                / "exact_start_excess_latency.png",
                dpi=300,
            )

            plt.close()

        summary_rows.append(
            summary
        )

        feature_rows.append(
            features
        )

    if exact_trace is None:
        raise RuntimeError(
            "Exact-start trial missing "
            f"for {label}."
        )

    return (
        structure,
        summary_rows,
        feature_rows,
        exact_trace,
    )


# ============================================================
# Aggregate plots
# ============================================================

def metric_plot(
    aggregate: pd.DataFrame,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot a metric against circuit size."""

    data = aggregate.sort_values(
        "num_qubits"
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.errorbar(
        data[
            "num_qubits"
        ],
        data[
            mean_column
        ],
        yerr=data[
            std_column
        ].fillna(
            0.0
        ),
        marker="o",
        capsize=4,
    )

    plt.xticks(
        SIZES
    )

    plt.xlabel(
        "QAOA circuit size (qubits)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# Exact-start comparisons
# ============================================================

def exact_start_outputs(
    exact_traces:
    dict[str, np.ndarray],
) -> None:
    """Compare exact-start traces directly."""

    trace_lengths = {
        len(trace)
        for trace
        in exact_traces.values()
    }

    if len(trace_lengths) != 1:
        raise RuntimeError(
            "Exact-start trace lengths differ: "
            f"{trace_lengths}"
        )

    trace_matrix = np.vstack(
        [
            exact_traces[label]
            for label in LABELS
        ]
    )

    trace_dataframe = pd.DataFrame(
        trace_matrix,
        index=LABELS,
        columns=[
            (
                f"probe_{probe_id:03d}_"
                "excess_ns"
            )
            for probe_id
            in range(
                trace_matrix.shape[1]
            )
        ],
    )

    trace_dataframe.index.name = (
        "qaoa_label"
    )

    trace_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_exact_start_traces.csv"
    )

    plt.figure(
        figsize=(14, 6)
    )

    image = plt.imshow(
        trace_matrix,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label=(
            "Excess turnaround time (ns)"
        ),
    )

    plt.yticks(
        range(
            len(LABELS)
        ),
        LABELS,
    )

    plt.xlabel(
        "Probe index"
    )

    plt.ylabel(
        "QAOA circuit"
    )

    plt.title(
        "Exact-Start QAOA "
        "Timing Fingerprints"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "qaoa_exact_start_trace_heatmap.png",
        dpi=300,
    )

    plt.close()

    distance_matrix = np.zeros(
        (
            len(LABELS),
            len(LABELS),
        )
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
            ] = np.mean(
                np.abs(
                    exact_traces[
                        first_label
                    ]
                    - exact_traces[
                        second_label
                    ]
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
            "qaoa_exact_start_"
            "pairwise_mae_ns.csv"
        )
    )

    plt.figure(
        figsize=(10, 9)
    )

    image = plt.imshow(
        distance_matrix,
        aspect="equal",
    )

    plt.colorbar(
        image,
        label=(
            "Mean absolute trace "
            "difference (ns)"
        ),
    )

    plt.xticks(
        range(
            len(LABELS)
        ),
        LABELS,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(
            len(LABELS)
        ),
        LABELS,
    )

    plt.title(
        "Pairwise Distance Between "
        "Exact-Start Fingerprints"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            "qaoa_exact_start_"
            "pairwise_mae_ns.png"
        ),
        dpi=300,
    )

    plt.close()


# ============================================================
# Classification
# ============================================================

def classify(
    feature_dataframe:
    pd.DataFrame,
) -> None:
    """
    Leave one complete start-offset condition
    out of training.
    """

    try:
        from sklearn.ensemble import (
            RandomForestClassifier,
        )

        from sklearn.metrics import (
            ConfusionMatrixDisplay,
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

    except ImportError:
        print(
            "\nscikit-learn is missing; "
            "classification was skipped."
        )

        print(
            "Install with: "
            "pip install scikit-learn"
        )

        return

    feature_columns = [
        column
        for column
        in feature_dataframe.columns
        if column.startswith(
            "bb_"
        )
    ]

    feature_matrix = (
        feature_dataframe[
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

    labels = feature_dataframe[
        "qaoa_label"
    ].astype(
        str
    ).to_numpy()

    groups = feature_dataframe[
        "start_offset_ns"
    ].to_numpy()

    classifier = (
        RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    )

    predictions = cross_val_predict(
        classifier,
        feature_matrix,
        labels,
        groups=groups,
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
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
        average="macro",
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=LABELS,
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
        feature_dataframe[
            [
                "qaoa_label",
                "num_qubits",
                "qasm_file",
                "start_offset_ns",
                "actual_victim_start_ns",
            ]
        ].copy()
    )

    prediction_dataframe[
        "predicted_qaoa_label"
    ] = predictions

    prediction_dataframe[
        "correct"
    ] = (
        prediction_dataframe[
            "qaoa_label"
        ].astype(
            str
        )
        == predictions
    )

    prediction_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_circuit_predictions.csv",
        index=False,
    )

    metrics = {
        "task": (
            "identify_exact_supplied_"
            "qaoa_circuit"
        ),
        "classes": LABELS,
        "class_count": len(
            LABELS
        ),
        "samples_per_class": len(
            START_OFFSETS_NS
        ),
        "sample_count": len(
            feature_dataframe
        ),
        "feature_count": len(
            feature_columns
        ),
        "classifier": (
            "RandomForestClassifier"
        ),
        "cross_validation": (
            "leave-one-victim-"
            "start-offset-out"
        ),
        "held_out_offsets_ns": (
            START_OFFSETS_NS
        ),
        "chance_accuracy": (
            1.0 / len(LABELS)
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
        "scope_note": (
            "Identification of these exact "
            "compiled QAOA circuits under "
            "unseen start offsets; not "
            "generalization to unseen "
            "QAOA instances."
        ),
    }

    with (
        OUTPUT_DIR
        / (
            "qaoa_circuit_"
            "classification_metrics.json"
        )
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metrics,
            output_file,
            indent=2,
        )

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    ConfusionMatrixDisplay(
        matrix,
        display_labels=LABELS,
    ).plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        "QAOA Circuit Identification\n"
        "Leave-One-Start-Offset-Out"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "qaoa_circuit_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    # Fit all observations only for interpretation.
    classifier.fit(
        feature_matrix,
        labels,
    )

    importance_dataframe = (
        pd.DataFrame(
            {
                "feature": (
                    feature_columns
                ),
                "importance": (
                    classifier
                    .feature_importances_
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    importance_dataframe.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_circuit_"
            "feature_importance.csv"
        ),
        index=False,
    )

    top_features = (
        importance_dataframe
        .head(
            20
        )
        .sort_values(
            "importance"
        )
    )

    axis = top_features.plot(
        kind="barh",
        x="feature",
        y="importance",
        legend=False,
        figsize=(11, 8),
    )

    axis.set_xlabel(
        "Random-forest feature importance"
    )

    axis.set_ylabel(
        "Black-box timing feature"
    )

    axis.set_title(
        "Most Informative QAOA "
        "Fingerprint Features"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            "qaoa_circuit_"
            "feature_importance.png"
        ),
        dpi=300,
    )

    plt.close()

    print(
        "\n=== QAOA circuit identification ==="
    )

    print(
        "Chance accuracy:   "
        f"{1.0 / len(LABELS):.4f}"
    )

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        "Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1:          {macro_f1:.4f}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qasm_files = resolve_all_qasms()

    print(
        "Resolved QAOA circuits:"
    )

    for num_qubits in SIZES:
        print(
            f"  qaoa_n{num_qubits}: "
            f"{qasm_files[num_qubits]}"
        )

    structure_rows: list[dict] = []

    summary_rows: list[dict] = []

    feature_rows: list[dict] = []

    exact_traces: dict[
        str,
        np.ndarray,
    ] = {}

    for num_qubits in SIZES:
        (
            structure,
            circuit_summaries,
            circuit_features,
            exact_trace,
        ) = run_circuit(
            num_qubits,
            qasm_files[
                num_qubits
            ],
        )

        structure_rows.append(
            structure
        )

        summary_rows.extend(
            circuit_summaries
        )

        feature_rows.extend(
            circuit_features
        )

        exact_traces[
            f"qaoa_n{num_qubits}"
        ] = exact_trace

    structure_dataframe = (
        pd.DataFrame(
            structure_rows
        )
        .sort_values(
            "num_qubits"
        )
    )

    summary_dataframe = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "num_qubits",
                "start_offset_ns",
            ]
        )
    )

    feature_dataframe = (
        pd.DataFrame(
            feature_rows
        )
        .sort_values(
            [
                "num_qubits",
                "start_offset_ns",
            ]
        )
    )

    structure_dataframe.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_circuit_"
            "structure_summary.csv"
        ),
        index=False,
    )

    summary_dataframe.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_circuit_"
            "trial_summary.csv"
        ),
        index=False,
    )

    feature_dataframe.to_csv(
        OUTPUT_DIR
        / "qaoa_circuit_features.csv",
        index=False,
    )

    aggregate = (
        summary_dataframe
        .groupby(
            [
                "qaoa_label",
                "num_qubits",
            ],
            as_index=False,
        )
        .agg(
            trial_count=(
                "start_offset_ns",
                "count",
            ),
            mean_avg_excess_latency_ns=(
                "avg_excess_turnaround_time_ns",
                "mean",
            ),
            std_avg_excess_latency_ns=(
                "avg_excess_turnaround_time_ns",
                "std",
            ),
            mean_total_excess_latency_ns=(
                "total_excess_turnaround_time_ns",
                "mean",
            ),
            std_total_excess_latency_ns=(
                "total_excess_turnaround_time_ns",
                "std",
            ),
            mean_contention_fraction=(
                "contention_observed_fraction",
                "mean",
            ),
            std_contention_fraction=(
                "contention_observed_fraction",
                "std",
            ),
            mean_victim_slowdown_ratio=(
                "victim_slowdown_ratio",
                "mean",
            ),
            std_victim_slowdown_ratio=(
                "victim_slowdown_ratio",
                "std",
            ),
            max_victim_slowdown_ratio=(
                "victim_slowdown_ratio",
                "max",
            ),
        )
        .merge(
            structure_dataframe,
            on=[
                "qaoa_label",
                "num_qubits",
            ],
            validate="one_to_one",
        )
        .sort_values(
            "num_qubits"
        )
    )

    aggregate.to_csv(
        OUTPUT_DIR
        / (
            "qaoa_circuit_"
            "aggregate_summary.csv"
        ),
        index=False,
    )

    metric_plot(
        aggregate,
        "mean_avg_excess_latency_ns",
        "std_avg_excess_latency_ns",
        (
            "Average victim-induced "
            "latency (ns)"
        ),
        (
            "QAOA Circuit Size: "
            "Average Black-Box Timing Signal"
        ),
        (
            "qaoa_circuit_"
            "avg_excess_latency.png"
        ),
    )

    metric_plot(
        aggregate,
        "mean_contention_fraction",
        "std_contention_fraction",
        (
            "Fraction of attacker "
            "probes delayed"
        ),
        (
            "QAOA Circuit Size: "
            "Contention Observation Fraction"
        ),
        (
            "qaoa_circuit_"
            "contention_fraction.png"
        ),
    )

    metric_plot(
        aggregate,
        "mean_victim_slowdown_ratio",
        "std_victim_slowdown_ratio",
        (
            "Victim completion-time ratio"
        ),
        (
            "QAOA Circuit Size: "
            "Victim Slowdown"
        ),
        (
            "qaoa_circuit_"
            "victim_slowdown.png"
        ),
    )

    exact_start_outputs(
        exact_traces
    )

    classify(
        feature_dataframe
    )

    display_columns = [
        "qaoa_label",
        "num_qubits",
        "trace_event_count_evaluator_only",
        (
            "cross_module_operation_count_"
            "evaluator_only"
        ),
        "cx_count_evaluator_only",
        "mean_avg_excess_latency_ns",
        "mean_contention_fraction",
        "mean_victim_slowdown_ratio",
    ]

    print(
        "\n=== Aggregate summary ==="
    )

    print(
        aggregate[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved results to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
