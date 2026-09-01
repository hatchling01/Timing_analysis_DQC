#!/usr/bin/env python3
"""
phase1_02_time_sliced_rerun.py

Corrected rerun of only the time-sliced tenancy model from Phase 1.2.

Why this rerun exists
---------------------
The original Phase 1.2 implementation applied alternating module time slices
only to the combined victim-attacker execution. The attacker-only and
victim-only controls did not obey the same slice schedule, so baseline
subtraction incorrectly counted normal slice waiting as victim-induced delay.

This script fixes that issue by applying the same reserved alternating time
slices to:

1. attacker-only execution;
2. victim-only execution; and
3. combined victim-attacker execution.

The slice schedule exists independently of whether the other tenant is active.
Therefore, baseline subtraction removes ordinary scheduling delay and retains
only additional interference caused by concurrent tenancy.

Requirements
------------
Keep this file beside:

    phase1_01_job_module_allocation.py
    phase1_02_tenancy_models.py

Run
---

    python phase1_02_time_sliced_rerun.py

Outputs
-------

blackbox_window_results/
└── phase1_02_tenancy_models/
    └── time_sliced_corrected/
        ├── tenancy_model_trial_summary.csv
        ├── tenancy_model_summary.csv
        ├── tenancy_model_endpoint_wait_summary.csv
        ├── tenancy_model_resource_blocking_summary.csv
        ├── tenancy_model_resource_effect_summary.csv
        ├── tenancy_model_admission_summary.csv
        ├── tenancy_model_utilization_summary.csv
        └── tenancy_model_configuration_summary.csv
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import phase1_02_tenancy_models as p2


# =============================================================================
# Integrated run controls
# =============================================================================

OUTPUT_DIR = (
    Path("blackbox_window_results")
    / "phase1_02_tenancy_models"
    / "time_sliced_corrected"
)

RUN_QUICK_VALIDATION = False
TRIALS_PER_CONFIGURATION = 3
SAVE_REQUEST_LEVEL_RESULTS = False
MAX_CONFIGURATIONS = None


# =============================================================================
# Corrected request generation
# =============================================================================

def corrected_build_node_requests(
    *,
    jobs,
    allocations,
    tenancy_allocation,
    config,
    combined_execution: bool,
):
    """
    Build remote requests while enforcing the same time-slice schedule in
    attacker-only, victim-only, and combined executions.

    ``combined_execution`` remains in the signature for compatibility with the
    original Phase 1.2 code, but it intentionally does not control time-slice
    eligibility anymore.
    """

    del combined_execution

    requests = []
    request_id = 0

    shared_modules = set(
        tenancy_allocation.shared_modules
    )

    for job in jobs:
        allocation = allocations[
            job.tenant_id
        ]

        for event in job.logical_events:
            touched_partitions = {
                job.partition_of_qubit[qubit]
                for qubit in event.qubits
                if qubit
                in job.partition_of_qubit
            }

            if len(touched_partitions) < 2:
                continue

            for (
                left_partition,
                right_partition,
            ) in itertools.combinations(
                sorted(touched_partitions),
                2,
            ):
                source = (
                    allocation
                    .partition_to_module[
                        left_partition
                    ]
                )

                target = (
                    allocation
                    .partition_to_module[
                        right_partition
                    ]
                )

                if source == target:
                    continue

                release_time_ns = (
                    job.start_time_ns
                    + event.release_offset_ns
                )

                eligible_time_ns = (
                    release_time_ns
                )

                # Corrected behavior:
                # Apply reserved time slices in every control run.
                if (
                    config.tenancy_model
                    == "time_sliced_module_sharing"
                    and (
                        source in shared_modules
                        or target in shared_modules
                    )
                ):
                    eligible_time_ns = (
                        p2.next_owned_time(
                            release_time_ns,
                            job.role,
                        )
                    )

                requests.append(
                    p2.NodeRequest(
                        request_id=(
                            request_id
                        ),
                        tenant_id=(
                            job.tenant_id
                        ),
                        role=job.role,
                        logical_event_id=(
                            event.event_id
                        ),
                        release_time_ns=(
                            release_time_ns
                        ),
                        eligible_time_ns=(
                            eligible_time_ns
                        ),
                        source_module=source,
                        target_module=target,
                        switch_path=(
                            p2.physical_pair_path(
                                source,
                                target,
                            )
                        ),
                        resource_keys=(
                            p2.resource_keys_for_request(
                                tenant_id=(
                                    job.tenant_id
                                ),
                                source_module=(
                                    source
                                ),
                                target_module=(
                                    target
                                ),
                                logical_event_id=(
                                    event.event_id
                                ),
                                config=config,
                                allocation=(
                                    tenancy_allocation
                                ),
                            )
                        ),
                    )
                )

                request_id += 1

    return sorted(
        requests,
        key=lambda request: (
            request.release_time_ns,
            p2.role_priority(
                request.role
            ),
            request.tenant_id,
            request.logical_event_id,
            request.request_id,
        ),
    )


# =============================================================================
# Configure and run the original Phase 1.2 framework
# =============================================================================

def main() -> None:
    # Run only the corrected time-sliced model.
    p2.TENANCY_MODELS = [
        "time_sliced_module_sharing"
    ]

    # Save separately from the original Phase 1.2 results.
    p2.OUTPUT_DIR = OUTPUT_DIR

    # Integrated run settings.
    p2.RUN_QUICK_VALIDATION = (
        RUN_QUICK_VALIDATION
    )
    p2.TRIALS_PER_CONFIGURATION = (
        TRIALS_PER_CONFIGURATION
    )
    p2.SAVE_REQUEST_LEVEL_RESULTS = (
        SAVE_REQUEST_LEVEL_RESULTS
    )
    p2.MAX_CONFIGURATIONS = (
        MAX_CONFIGURATIONS
    )

    # Replace only the faulty request-generation function.
    p2.build_node_requests = (
        corrected_build_node_requests
    )

    print(
        "Phase 1.2 corrected rerun — "
        "time-sliced module sharing only"
    )
    print(
        "Time-slice eligibility is active "
        "in attacker-only, victim-only, "
        "and combined executions."
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    p2.main()


if __name__ == "__main__":
    main()
