# Quantum Network Timing Attacks and Side-Channel Analysis

## Abstract

This repository contains simulations and analyses of timing-based side-channel attacks on quantum networks and distributed quantum computing (DQC) architectures. Using the NetSquid quantum network simulation framework, we investigate how contention in shared entanglement services and cross-module communication in modular quantum systems can leak timing information, enabling attackers to infer victim workloads or circuit structures.

The project evaluates three DQC execution modes (monolithic, sequential modular, static distributed) across multiple quantum circuits (Bernstein-Vazirani, Deep Neural Network inference, Quantum Fourier Transform, SAT solver, and square root) and implements various attack strategies including periodic probing, bursty attacks, and synchronized schedules.

Key contributions include:
- Demonstration of timing leakage in Bell State Measurement (BSM) contention
- Quantification of cross-module communication overhead in modular architectures
- Evaluation of attack detection rates under different placement and scheduling strategies
- Comprehensive performance benchmarks across execution modes

## Project Structure

### Core Simulation Files
- `experiment_a.py`, `experiment_a2.py`, `experiment_a3_combined.py`: Basic timing leakage experiments in entanglement services
- `run_monolithic.py`: Monolithic execution simulation
- `run_sequential_modular.py`, `run_sequential_modular_v2.py`: Sequential modular execution
- `run_static_distributed.py`: Static distributed execution
- `run_attack_tier1_*.py`: Tier-1 attack simulations for different placements and probes

### Architecture Implementation
- `new_arch_baseline.py`: Baseline architecture definitions
- `new_arch_baseline_fivenode.py`: Five-module superconducting DQC architecture
- `new_arch_fivenode_traceadded.py`: Architecture with trace processing capabilities

### Quantum Circuits
- `bv_n19.qasm`: Bernstein-Vazirani algorithm (19 qubits)
- `dnn_n16.qasm`: Deep Neural Network inference (16 qubits)
- `qft_n18.qasm`: Quantum Fourier Transform (18 qubits)
- `sat_n11.qasm`: SAT solver (11 qubits)
- `square_root_n18.qasm`: Square root algorithm (18 qubits)
- `qaoa_nativegates_ibm_qiskit_opt3_*.qasm`: QAOA circuits with varying depths

### Results and Analysis
- `*_stats.json`: Performance statistics for different execution modes
- `*_results.json`: Attack simulation results
- `plot_*.py`: Plotting scripts for visualization
- `*.png`: Generated plots and figures
- `bv_results/`, `QFT_results/`, `square_root_n18_results/`: Circuit-specific P1 static attack outputs for BV, QFT, and square-root workloads
- `Disjoint_allocation/`: P1/disjoint-allocation sweeps, sequential best-attack runs, and QAOA-family distinguishability artifacts
- `Overlapped_allocation/`: P2/overlap attack sweeps, static overlap runs, and QAOA-family distinguishability artifacts
- `selecting_best_probe_probe3/`: Probe-selection burst sweeps comparing probe families on communication-heavy circuits

### Dependencies
- `netsquid_clean_env.yml`: Conda environment specification
- `netsquid_clean_pip_freeze.txt`: Python package requirements

## Architectures Evaluated

### Monolithic Execution
All qubits mapped to a single compute module. No cross-module communication, minimal timing leakage.

### Sequential Modular Execution
Circuit decomposed into stages executed sequentially across modules. Introduces cross-module entanglement operations with associated delays.

### Static Distributed Execution
Parallel execution across multiple modules with static qubit allocation. Highest cross-module traffic and potential for timing leakage.

## Attack Models

### Tier-1 Attacks
Attacker probes shared resources (hub or entanglement services) to detect victim activity through latency variations.

#### Placements
- **P1**: Victim uses modules 0-2, attacker uses 3-4 (no shared compute modules)
- **P2**: Partial overlap - victim and attacker share some modules

#### Probes
- **Probe 1**: CX chain operations
- **Probe 2**: Bursty entangling gates
- **Probe 3**: Light periodic probes

#### Schedules
- A1: Victim-only baseline
- A2: Always-on overlap
- A3: Front-loaded attacks
- A4: Back-loaded attacks
- A5: Periodic probing
- A6: Bursty synchronized
- A7: Saturation attacks

## Results

### Baseline Performance Comparison

| Execution Mode | Total Events | Local Events | Cross-Module Events | Avg Waiting Time (ns) | Max Waiting Time (ns) | Hub Makespan (ns) |
|----------------|--------------|--------------|---------------------|----------------------|----------------------|-------------------|
| Monolithic | 558 | 558 | 0 | 0 | 0 | 2790 |
| Sequential Modular | 558 | 310 | 248 | 5641 | 12210 | 15010 |
| Sequential Modular v2 | 558 | 310 | 248 | 145 | 420 | 32480 |
| Static Distributed | 558 | 310 | 248 | 5641 | 12210 | 15010 |

*Note: All results based on the same quantum circuit (likely BV or similar). Sequential v2 shows optimized scheduling with reduced waiting but increased makespan.*

### Attack Detection Results Across Circuits (Sequential v2, P2 Placement, A5 Schedule)

| Circuit | Victim Cross Fraction | Attacker Completed | Avg Waiting Time (ns) | Max Waiting Time (ns) | Waited Fraction |
|---------|-----------------------|-------------------|----------------------|----------------------|-----------------|
| BV (n19) | 0.211 | 4 | 112.5 | 225 | 0.75 |
| DNN (n16) | 0.035 | 15 | 0.0 | 0 | 0.0 |
| QFT (n18) | 0.271 | 44 | 1028.1 | 2550 | 0.86 |
| SAT (n11) | 0.442 | 9 | 290.6 | 525 | 0.89 |
| Square Root (n18) | 0.443 | 50 | 1478.4 | 3225 | 0.98 |

*Note: Higher cross-module operation fractions correlate with higher attacker waiting times and detection rates.*

### Tier-1 Attack Results (BV Circuit, P1 Placement, Static Distributed, Probe 1)

| Schedule | Attacker Completed | Avg Waiting Time (ns) | Max Waiting Time (ns) | Waited Fraction |
|----------|-------------------|----------------------|----------------------|-----------------|
| A1 (Victim Only) | 0 | 0 | 0 | 0.0 |
| A2 (Always On) | 80 | 3668.9 | 7530 | 0.99 |
| A3 (Front Loaded) | 80 | 3952.5 | 7905 | 0.99 |
| A4 (Back Loaded) | 80 | 3952.5 | 7905 | 0.99 |
| A5 (Periodic) | 40 | 1637.0 | 3565 | 0.98 |
| A6 (Bursty Sync) | 120 | 5725.2 | 11630 | 0.99 |
| A7 (Saturation) | 120 | 5757.2 | 11530 | 0.99 |

*Note: Results show significant timing leakage under all overlap schedules, with bursty and saturation attacks experiencing the highest delays.*

### Additional Result Families Present In The Repository

The repository contains several broader experiment families beyond the summary tables above. These are already generated on disk and can be reproduced from the included driver scripts.

#### P1 Disjoint-Allocation Probe-3 Sweeps
- `Disjoint_allocation/probe3_rate_sweep_*`: Probe-rate sweeps across BV, DNN, QFT, SAT, and square-root circuits
- `Disjoint_allocation/probe3_spacing_R1_*` and `probe3_spacing_R2_*`: Inter-probe spacing sweeps for two rate regimes
- `Disjoint_allocation/probe3_R1_uniform_short_*`, `..._medium_*`, `..._long_*`: Time-scale sweeps for light periodic probing
- `Disjoint_allocation/probe3_R1_uniform_reldur_*` and `..._absdur_*`: Relative- and absolute-duration attack window sweeps

These files extend the README's current BV-only example by showing how timing leakage changes as the attacker varies probe density, spacing, and observation window width across multiple workloads.

#### Disjoint Sequential Best-Attack Families
- `Disjoint_allocation/sequential_bestattack_*`: Best-attack runs for sequential modular execution
- `Disjoint_allocation/sequential_bestattack_v2_*`: Best-attack runs for the optimized sequential modular v2 workflow

Both sets are available for BV, DNN, QFT, SAT, and square-root workloads and provide per-run JSON plus job-count, makespan, and request-level plots.

#### P2 Overlapped-Allocation Sweep Families
- `Overlapped_allocation/static_overlap_p2_*`: Static-distributed overlap runs for all five benchmark circuits
- `Overlapped_allocation/overlap_p2_pattern_*`: Pattern/schedule sweeps under partial module overlap
- `Overlapped_allocation/overlap_p2_r1_timescale_*`: Probe-3 time-scale sweeps for overlapped placement
- `Overlapped_allocation/overlap_p2_ratesweep_*`: Probe-density sweeps at `P20`, `P50`, and `P100`

These artifacts complement the top-level `sequential_v2_overlap_p2_*` results by covering additional overlap patterns and static-distributed attack scenarios.

#### QAOA Family Distinguishability And Fingerprinting

The repository includes full QAOA-family timing-fingerprint studies for QAOA circuits `qaoa_nativegates_ibm_qiskit_opt3_5.qasm` through `..._15.qasm`.

- `Disjoint_allocation/qaoa_family_best_attack_results.json`, `..._summary.csv`, `..._fingerprints.csv`, `..._pairwise_distance.csv`, `..._pairwise_distance.png`, `..._request_metrics.png`: P1/static-distributed QAOA distinguishability outputs
- `Disjoint_allocation/qaoa_family_best_attack_sequential_v2_results.json` and companion CSV/PNG files: Sequential modular v2 QAOA distinguishability outputs
- `Overlapped_allocation/qaoa_family_best_attack_overlap_p2_results.json` and companion CSV/PNG files: P2/one-module-overlap QAOA distinguishability outputs

These files capture attack fingerprints, request-level metrics, and pairwise distance matrices that quantify how well timing observations separate different QAOA instances.

#### Probe-Selection Burst Sweeps
- `selecting_best_probe_probe3/burst_sweep_qft_n18_probe_{1,2,3}_*`
- `selecting_best_probe_probe3/burst_sweep_square_root_n18_probe_{1,2,3}_*`

These burst-sweep experiments compare CX-chain, bursty-entangling, and light-periodic probes on QFT and square-root workloads to help select an effective probe family for highly communication-heavy circuits.

#### Additional Circuit-Specific Static Attack Folders
- `QFT_results/`: P1 static attack plots and JSON outputs for QFT under probes 1, 2, and 3
- `square_root_n18_results/`: P1 static attack plots and JSON outputs for square-root, including aggregate `tier1_p1_static_*` plots and per-probe breakdowns

Together with `bv_results/`, these folders provide circuit-level drill-downs that are more detailed than the summary tables in this README.

### Timing Leakage in Entanglement Services

Experiment A demonstrates significant latency shifts when victim load is present:

- **Victim OFF**: Mean latency ~X ns
- **Victim ON**: Mean latency ~Y ns
- **KS Statistic**: Z (indicating distributional difference)

## Images and Visualizations

### Baseline Performance
![Local vs Cross-Module Events](plot_local_vs_cross.png)
*Comparison of local and cross-module events across execution modes*

![Total Events](plot_total_events.png)
*Total event counts by execution mode*

![Average Waiting Time](plot_avg_waiting_time.png)
*Average waiting times for hub requests*

![Max Waiting Time](plot_max_waiting_time.png)
*Worst-case waiting times for hub requests*

![Average Turnaround Time](plot_avg_turnaround_time.png)
*Average turnaround times*

![Hub Makespan](plot_hub_makespan.png)
*Hub makespan comparison*

![Hub Requests](plot_hub_requests.png)
*Completed hub requests*

![Nonzero Wait Requests](plot_nonzero_wait_requests.png)
*Fraction or count of requests that experienced nonzero waiting*

### Per-Module Analysis
![Monolithic](plot_per_module_monolithic.png)
*Event distribution in monolithic mode*

![Sequential Modular](plot_per_module_sequential_modular.png)
*Event distribution in sequential modular mode*

![Static Distributed](plot_per_module_static_distributed.png)
*Event distribution in static distributed mode*

### Sequential Stage Profiles
![Sequential Stage Profile](plot_sequential_stage_profile.png)
*Stage-wise event counts*

![Sequential Stage Wait Profile](plot_sequential_stage_wait_profile.png)
*Stage-wise waiting times*

### Attack Results (BV Circuit Examples)
![BV Job Counts](sequential_v2_overlap_p2_bv_n19_job_counts.png)
*Job completion counts for BV attacks*

![BV Job Makespan](sequential_v2_overlap_p2_bv_n19_job_makespan.png)
*Job makespan for BV attacks*

![BV Request Level](sequential_v2_overlap_p2_bv_n19_request_level.png)
*Request-level timing for BV attacks*

### Additional Attack Results
#### DNN Circuit
![DNN Job Counts](sequential_v2_overlap_p2_dnn_n16_job_counts.png)
![DNN Job Makespan](sequential_v2_overlap_p2_dnn_n16_job_makespan.png)
![DNN Request Level](sequential_v2_overlap_p2_dnn_n16_request_level.png)

#### QFT Circuit
![QFT Job Counts](sequential_v2_overlap_p2_qft_n18_job_counts.png)
![QFT Job Makespan](sequential_v2_overlap_p2_qft_n18_job_makespan.png)
![QFT Request Level](sequential_v2_overlap_p2_qft_n18_request_level.png)

#### SAT Circuit
![SAT Job Counts](sequential_v2_overlap_p2_sat_n11_job_counts.png)
![SAT Job Makespan](sequential_v2_overlap_p2_sat_n11_job_makespan.png)
![SAT Request Level](sequential_v2_overlap_p2_sat_n11_request_level.png)

#### Square Root Circuit
![Square Root Job Counts](sequential_v2_overlap_p2_square_root_n18_job_counts.png)
![Square Root Job Makespan](sequential_v2_overlap_p2_square_root_n18_job_makespan.png)
![Square Root Request Level](sequential_v2_overlap_p2_square_root_n18_request_level.png)

### Tier-1 Attack Plots (BV Circuit, Probe 1)
![P1 Static Probe 1 CX Chain Job Counts](bv_results/tier1_p1_static_probe_1_cx_chain_job_counts.png)
![P1 Static Probe 1 CX Chain Job Makespan](bv_results/tier1_p1_static_probe_1_cx_chain_job_makespan.png)
![P1 Static Probe 1 CX Chain Request Level](bv_results/tier1_p1_static_probe_1_cx_chain_request_level.png)

#### Additional Probes
![P1 Static Probe 2 Bursty Entangling Job Counts](bv_results/tier1_p1_static_probe_2_bursty_entangling_job_counts.png)
![P1 Static Probe 2 Bursty Entangling Job Makespan](bv_results/tier1_p1_static_probe_2_bursty_entangling_job_makespan.png)
![P1 Static Probe 2 Bursty Entangling Request Level](bv_results/tier1_p1_static_probe_2_bursty_entangling_request_level.png)

![P1 Static Probe 3 Light Periodic Job Counts](bv_results/tier1_p1_static_probe_3_light_periodic_job_counts.png)
![P1 Static Probe 3 Light Periodic Job Makespan](bv_results/tier1_p1_static_probe_3_light_periodic_job_makespan.png)
![P1 Static Probe 3 Light Periodic Request Level](bv_results/tier1_p1_static_probe_3_light_periodic_request_level.png)

### Additional Circuit-Specific Attack Plot Collections
`QFT_results/` and `square_root_n18_results/` contain the same probe-1, probe-2, and probe-3 plot families shown above for BV, along with their corresponding `*_results.json` files. The square-root folder also includes aggregate `tier1_p1_static_job_counts.png`, `tier1_p1_static_job_makespan.png`, `tier1_p1_static_job_level.png`, and `tier1_p1_static_results.json` outputs.

### QAOA Distinguishability Visualizations
![QAOA Static Pairwise Distance](Disjoint_allocation/qaoa_family_best_attack_pairwise_distance.png)
*Pairwise timing-distance matrix for the disjoint/static QAOA family study*

![QAOA Sequential v2 Pairwise Distance](Disjoint_allocation/qaoa_family_best_attack_sequential_v2_pairwise_distance.png)
*Pairwise timing-distance matrix for the sequential modular v2 QAOA family study*

![QAOA Overlap P2 Pairwise Distance](Overlapped_allocation/qaoa_family_best_attack_overlap_p2_pairwise_distance.png)
*Pairwise timing-distance matrix for the P2 overlapped-allocation QAOA family study*

### Sweep And Placement Study Outputs
- `Disjoint_allocation/` contains full JSON and PNG outputs for probe-rate, spacing, and time-scale sweeps across all benchmark circuits.
- `Overlapped_allocation/` contains static-overlap, pattern, time-scale, and rate-sweep outputs for all benchmark circuits.
- `selecting_best_probe_probe3/` contains burst-sweep comparisons used to choose between probe families on QFT and square-root workloads.

## Summary of Findings

### Key Insights

1. **Execution Mode Impact**: Monolithic execution eliminates cross-module communication entirely, providing the lowest timing leakage but potentially suboptimal resource utilization. Sequential modular v2 optimizes scheduling to reduce waiting times (145 ns vs 5641 ns in v1) at the cost of increased makespan (32480 ns vs 15010 ns). Static distributed shows identical performance to sequential v1 in this dataset.

2. **Circuit-Specific Leakage**: Timing leakage varies significantly by circuit structure. Circuits with low cross-module fractions (DNN: 3.5%) show minimal attacker waiting (0%), while high-cross circuits (SAT: 44.2%, Square Root: 44.3%) exhibit near-complete detection (89-98% waited fraction).

3. **Attack Schedule Effectiveness**: Different probing schedules yield varying detection strengths. Always-on overlap (A2) and bursty synchronized (A6) attacks experience the highest delays (3669-5757 ns avg waiting), while periodic probing (A5) provides moderate detection with lower resource usage.

4. **Placement Sensitivity**: P1 placement (disjoint modules) still allows significant timing leakage through shared hub contention, with all overlap schedules showing >98% waited requests. P2 placement amplifies this effect for circuits with cross-module operations.

5. **Probe Strategy Variations**: Different probe types (CX chains, bursty entangling, light periodic) show similar qualitative behavior but may differ in quantitative timing signatures and resource consumption.

### Implications for Quantum Security

- Modular DQC architectures inherently leak timing information through shared hub resources, regardless of module placement strategy
- Attackers can infer circuit structure and execution patterns via passive timing observation, with detection rates approaching 100% for communication-heavy circuits
- Mitigation strategies should focus on constant-time execution, noise injection, or fully distributed architectures without shared bottlenecks
- Architecture design must balance performance gains against security risks, particularly for sensitive quantum algorithms

### Future Work

- Implement additional attack tiers (active interference)
- Evaluate larger-scale architectures (16+ modules)
- Develop timing-oblivious scheduling algorithms
- Investigate quantum-specific countermeasures

## Setup and Usage

### Environment Setup
```bash
# Using conda
conda env create -f netsquid_clean_env.yml
conda activate netsquid-env

# Or using pip
pip install -r netsquid_clean_pip_freeze.txt
```

### Running Simulations
```bash
# Baseline performance
python run_monolithic.py
python run_sequential_modular_v2.py
python run_static_distributed.py

# Attack simulations
python run_attack_tier1_p1_static.py
python run_attack_tier1_p2_static_bestattack.py
python run_attack_tier1_p1_sequential_bestattack.py
python run_attack_tier1_p2_sequential_v2_bestattack.py
python run_attack_tier1_p1_static_probe3_ratesweep.py
python run_attack_tier1_p1_static_probe3_spacingsweep_R1.py
python run_attack_tier1_p1_static_probe3_spacingsweep_R2.py
python run_attack_tier1_p1_static_probe3_r1_uniform_short_timescale.py
python run_attack_tier1_p1_static_probe3_r1_uniform_medium_timescale.py
python run_attack_tier1_p1_static_probe3_r1_uniform_long_timescale.py
python run_attack_tier1_p1_static_probe3_r1_uniform_relativedurationsweep.py
python run_attack_tier1_p1_static_probe3_r1_uniform_absolutedurationsweep.py
python run_attack_tier1_p2_static_pattern_sweep.py
python run_attack_tier1_p2_static_r1_timescale_sweep.py
python run_attack_tier1_p2_static_probe3_rate_sweep.py
python qaoa_family_best_attack_overlap_p2.py

# Plotting
python plot_baseline_stats.py
```

### Generating Reports
Results are automatically saved as JSON files and plots as PNG images.

## References

- CCS 2026 Paper: See `CCS_2026 (1).pdf` for the full conference submission
- NetSquid Documentation: https://netsquid.org/
- Qiskit Documentation: https://qiskit.org/

## Authors

[Add author information if available]

## License

[Add license information if available]
