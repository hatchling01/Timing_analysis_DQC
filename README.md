# Quantum Network Timing Attacks and Side-Channel Analysis

## Abstract

This repository contains simulations and analyses of timing-based side-channel attacks on quantum networks and distributed quantum computing (DQC) architectures. Using the NetSquid quantum network simulation framework, we investigate how contention in shared entanglement services and cross-module communication in modular quantum systems can leak timing information, enabling attackers to infer victim workloads or circuit structures.

The project evaluates three DQC execution modes (monolithic, sequential modular, static distributed) across multiple quantum circuits (Bernstein-Vazirani, Deep Neural Network inference, Quantum Fourier Transform, SAT solver, and square root) and implements various attack strategies including periodic probing, bursty attacks, and synchronized schedules.

The `black_box_phase_1_results` branch is an extension of `master`. It preserves the original architecture and attack studies and adds a Phase 1 black-box evaluation in which the attacker learns only from its own probe completion times. The extension tests whether leakage survives realistic uncertainty and system choices: limited observation windows, unknown victim start times and placement, job/module allocation, tenancy isolation, communication-qubit allocation, remote-operation scheduling, and dynamic rerouting/remapping.

Key contributions include:
- Demonstration of timing leakage in Bell State Measurement (BSM) contention
- Quantification of cross-module communication overhead in modular architectures
- Evaluation of attack detection rates under different placement and scheduling strategies
- Comprehensive performance benchmarks across execution modes
- Black-box controls that separate untargeted probing from window-aligned probing
- System-level leakage analysis across allocation, tenancy, scheduling, rerouting, and incomplete attacker knowledge

## Project Structure

### Core Simulation Files
- `experiment_a.py`, `experiment_a2.py`, `experiment_a3_combined.py`: Basic timing leakage experiments in entanglement services
- `run_monolithic.py`: Monolithic execution simulation
- `run_sequential_modular.py`, `run_sequential_modular_v2.py`: Sequential modular execution
- `run_static_distributed.py`: Static distributed execution
- `run_attack_tier1_*.py`: Tier-1 attack simulations for different placements and probes
- `run_attack_tier1_p1_static_blackbox_*.py`: Black-box baseline, placement, hub-capacity, observation-window, probe, timing-uncertainty, and QAOA fingerprinting studies
- `phase1_01_*.py` through `phase1_06_*.py`: Phase 1 system-factor experiments and post-processing
- `phase1_06_finish_random_forest_standalone.py`, `phase1_06_random_forest_fast.py`: Complete the learned Phase 1.06 analysis from saved samples without rerunning simulations

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
- `blackbox_results/`: Untargeted full-run black-box control for five benchmark circuits, including raw attacker observations and victim ground truth
- `blackbox_window_results/`: Window-aligned black-box baselines, sensitivity sweeps, Phase 1.01-1.06 system studies, and QAOA fingerprint/noise-robustness results

### Software Requirements

- Python 3.11 or compatible
- Qiskit
- pandas
- NumPy
- Matplotlib
- scikit-learn for the optional learned classifiers

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

## Phase 1 Black-Box Extension

### Scope and Threat Model

Phase 1 asks what can be inferred when an attacker cannot inspect the victim's circuit trace, victim communication events, internal scheduler state, exact placement, or completion time. The attacker submits legal remote-operation probes and observes its own request behavior. The strict timing-only threat model uses release and completion timing; explicitly labeled allocator-aware analyses may additionally use failure, redirection, endpoint reuse, or assignment information. Victim-side traces are retained strictly as evaluator ground truth and are not attacker inputs.

The default black-box configuration uses static-distributed execution, P1 disjoint placement, a serialized hub (`hub_max_concurrent_transfers = 1`), and a light periodic probe. The controlled sweeps then relax the attacker's start-time/placement knowledge and vary the architectural policies that determine contention.

### Fixed Black-Box Attack Baseline

Unless a study explicitly changes a parameter, the calibrated baseline is:

- Probe 3, a low-contention periodic probe
- Probe sequence: `h(0), h(1), x(2), x(3), z(0), z(2), cx(0,2)`
- Remote-request offset within each round: 30 ns
- Inter-probe period: 420 ns
- Observation window: 20,000 ns
- Attacker self-contention: approximately zero in the selected baseline
- Controls: attacker-only, victim-only, and victim-plus-attacker executions
- Primary signal: request-level excess attacker turnaround time relative to attacker-only calibration

This configuration retains victim-dependent timing while keeping baseline self-contention negligible and average victim disruption in the low single-digit range.

### Architecture and Resource Model

The base architecture contains five quantum modules connected through a centralized remote-operation hub; the system-factor studies vary module count, hub capacity, links, switch paths, communication interfaces, alternate routes, and internal allocation policies. A cross-module gate becomes a remote-operation request that passes through arrival, queue admission, scheduling, hub/link/path acquisition, endpoint locking, service, completion, resource release, and queue re-evaluation.

Phase 1 distinguishes physical co-location from overlap on an exclusive resource. Relevant sharing classes include communication qubits, endpoint locks, local-routing paths, reset/readout/feedforward pipelines, links, switch paths, hub slots, allocation queues, and reconfiguration transitions. Two tenants can share a module without a timing channel when these communication and control resources are fully isolated.

### What Is in `blackbox_results/`

`blackbox_results/` contains the compact, untargeted full-run control experiment for five benchmark circuits. It is intentionally distinct from the window-aligned studies in `blackbox_window_results/`.

The directory contains 26 files under `blackbox_results/baseline/`:

- `blackbox_baseline_summary.csv`: one aggregate row per victim workload with the configuration, victim communication counts, attacker timing statistics, excess latency, contention fraction, and hub makespans.
- `<circuit>_blackbox_attacker_observations.csv`: the attacker's observable request-level timing trace. These are the measurements available to a black-box adversary.
- `<circuit>_blackbox_victim_ground_truth.csv`: evaluator-only victim activity used to score whether attacker delays coincide with victim traffic.
- `<circuit>_blackbox_summary.json`: a machine-readable per-circuit configuration and metric summary.
- `<circuit>_blackbox_excess_latency.png`: excess attacker latency across the run.
- `<circuit>_blackbox_timing_trace.png`: attacker timing overlaid with evaluator-only victim activity for interpretation.

The five circuit prefixes are `bv_n19`, `dnn_n16`, `qft_n18`, `sat_n11`, and `square_root_n18`. Under this control's fixed `D100` probe schedule (100 probe rounds, 420 ns effective cross-probe spacing, attacker start at 0 ns), all five workloads completed 100 attacker requests but recorded zero excess waiting and zero observed contention, even though the victims generated between 16 and 247 cross-module operations. This negative result is important: cross-module activity alone is not sufficient for an untargeted probe schedule to produce a signal. Temporal overlap with the victim's active communication window is the key condition tested by the next result family.

### What Is in `blackbox_window_results/`

`blackbox_window_results/` contains 3,527 generated artifacts and is the main Phase 1 result corpus. CSV files contain aggregate and request/trial-level data, JSON files preserve configurations and per-run metrics, PNG files visualize timing and system tradeoffs, and the largest request logs are compressed or stored with Git LFS.

The top-level result families are:

| Directory | Question answered |
|---|---|
| `baseline/` | Does a coarse, window-aligned black-box probe expose workload-dependent contention? |
| `observation_window/`, `window_start_estimation/`, `random_victim_start/` | How much timing uncertainty can the attack tolerate? |
| `probe_rate/`, `probe_type/`, `inter_probe_spacing/` | How do probe intensity, shape, and cadence trade signal strength against victim slowdown? |
| `placement/`, `hub_capacity/` | Does disjoint placement prevent leakage, and how does hub parallelism change it? |
| `phase1_01_job_module_allocation/` | How do admission and module-allocation policies create P1/P2-like placements and leakage? |
| `phase1_02_tenancy_models/` | Which isolation boundary—exclusive modules, spatial/time slicing, or communication-interface sharing—controls leakage? |
| `phase1_03_communication_qubit_allocation/` | How do communication-qubit allocation, reservation/reset, fairness, EPR prefetch, and failure behavior affect observability? |
| `phase1_04_remote_operation_schedulers/` | How do queue discipline, priority, lookahead, preemption, and prefetch alter latency and leakage? |
| `phase1_05_dynamic_rerouting_remapping/` | Can path changes and reconfiguration transients be detected or localized from timing? |
| `phase1_06_unknown_placement_robustness/` | Does detection survive unknown physical placement and policy knowledge? |
| `qaoa_circuit_fingerprinting/` | Can black-box timing distinguish QAOA circuits with 5-15 qubits? |
| `qaoa_circuit_noise_robustness/` | Does the QAOA timing signature persist under timestamp and scheduler noise? |

### Experiment 1.1: Job-to-Module Allocation

**Question:** Does leakage remain when jobs are dynamically allocated instead of manually fixed to P1 or P2?

`phase1_01_job_module_allocation.py` evaluates fixed P1/P2, uniform-random, load-balanced, communication-minimizing, communication-consolidating, endpoint-separation, and first-fit-admission policies. The main outputs are `job_module_allocation_trial_summary.csv`, `job_module_allocation_allocations.csv`, `job_module_allocation_sharing_summary.csv`, `job_module_allocation_policy_summary.csv`, `job_module_allocation_rejection_summary.csv`, and `job_module_allocation_knowledge_views.csv`.

Allocation changes the probability of exposure rather than the underlying channel. Communication consolidation, first fit, and fixed P2 tend to create endpoint overlap; load balancing, communication minimization, endpoint separation, and fixed P1 more often avoid it. Uniform random allocation produces both classes. Endpoint separation lowers exposure but can increase fragmentation and rejection. Across the policy summaries, accepted jobs retain a strong signal when they overlap on an exclusive endpoint or switch path, while hub-only sharing with sufficient capacity is negligible.

> **Experiment 1.1 conclusion:** Allocation determines how often exposure occurs; once tenants share an exclusive communication resource, the timing channel remains.

### Experiment 1.2: Tenancy and Module Sharing

**Question:** Which forms of multi-tenancy allow endpoint occupancy to become attacker-visible?

`phase1_02_tenancy_models.py` and `phase1_02_time_sliced_rerun.py` compare module-exclusive tenancy, spatial compute-qubit partitioning, shared and dedicated communication interfaces, time-sliced module sharing, and hybrid sharing of reset/readout/feedforward resources. Each configuration is evaluated with a four-slot hub and a `node_only` control so node-side leakage can be distinguished from global hub serialization. The main summary families cover trials, endpoint wait, resource blocking/effects, admission, utilization, and configuration; corrected time-sliced results are under `time_sliced_corrected/`.

Module-exclusive tenancy removes direct shared-module occupancy. Shared communication qubits, module-wide locks, shared local routing, and shared reset/control pipelines preserve measurable leakage, whereas a fully isolated shared module can produce zero leakage. Dedicated interfaces reduce leakage at the cost of admission flexibility. The corrected time-sliced rerun applies identical slice rules to attacker-only, victim-only, and combined controls: time slicing alone is not the cause, but shared internal resources during active slices still leak.

> **Experiment 1.2 conclusion:** Module sharing is not sufficient by itself; leakage requires overlap on an exclusive internal communication or control resource.

### Experiment 1.3: Dynamic Communication-Qubit Allocation

**Question:** Does dynamic communication-qubit assignment introduce, remove, or reshape the channel?

`phase1_03_communication_qubit_allocation.py` evaluates static assignment, first available, round robin, random allocation, per-tenant reservation, a shared global pool, priority, fair share, coherence-aware, and predicted-demand allocation. `phase1_03_postprocess.py` separates positive delay, negative speedup, failure-only leakage, combined observable changes, successful-run slowdown, physical utilization, EPR use/wastage, and timing-only versus learned detection.

One communication qubit per module creates strong pressure; two remove most endpoint-specific pressure, and four or eight eliminate ordinary end-to-end queue-delay leakage in the tested configurations. Shared pools improve utilization but expose queue and reuse behavior. Per-tenant reservation removes differential queue timing but can convert pressure into failures, idle capacity, and slowdown. Reset occupancy creates a reuse-delay channel, while EPR prefetching can appear as negative timing changes or speedups and introduces measurable wastage.

The threat-model boundary matters here: a timing-only attacker uses end-to-end request timing, while an allocator-aware attacker may use endpoint identity, reuse delay, failure, reassignment, redirection, or EPR state. Sufficient overprovisioning can remove the timing-only channel even when allocator-visible features retain a weaker signal.

> **Experiment 1.3 conclusion:** Capacity and allocation policy determine whether contention appears as delay, speedup, failure, redirection, reuse, or wasted capacity.

### Experiment 1.4: Remote-Operation Scheduling

**Question:** Is the timing fingerprint an artifact of a particular scheduler?

`phase1_04_remote_operation_schedulers.py` compares static circuit-layer scheduling, FCFS, dependency-ready, shortest-operation-first, longest-waiting-first, round-robin tenants, weighted fair queueing, priority, coherence-deadline, link-aware, endpoint-aware, and randomized arbitration. Outputs include policy/capacity/queue/tenancy/communication-qubit summaries, priority and decision-interval studies, lookahead, preemption, prefetch, fingerprint stability, phase visibility, non-ML and random-forest metrics, plus compressed request logs.

Leakage is scheduler-sensitive but not scheduler-specific. FCFS is among the most revealing policies; link-aware and endpoint-aware scheduling reduce timing magnitude. Fair, priority, dependency-ready, and randomized policies reshape rather than eliminate the fingerprint. Shallow queues convert contention into rejection, deep queues convert it into waiting, and increasing hub capacity alone is insufficient when links or endpoints remain narrow. Preemption and EPR prefetching can improve completion while increasing transition and phase-boundary visibility.

> **Experiment 1.4 conclusion:** Multiple realistic schedulers preserve measurable fingerprints, although the most useful attacker features depend on the policy.

### Experiment 1.5: Rerouting and Resource Remapping

**Question:** Does changing the path or endpoint during execution break the attack?

`phase1_05_dynamic_rerouting_remapping.py` evaluates static paths, per-job and per-operation selection, load- and failure-triggered rerouting, communication-qubit reassignment, hub migration, and victim-module migration. The output families cover mechanisms, decisions, frequency, path count, thresholds, switching and state-transfer cost, before/transient/after segments, change detection, path localization, fingerprint stability, events, and compressed attacker/request logs.

Load-triggered rerouting and hub migration retain much of the original signal. Communication-qubit reassignment produces a sharp transient; failure-triggered rerouting is especially visible during transition. Per-job and per-operation policies create persistent path-dependent fingerprints. Frequent switching reduces exact trace stability without removing aggregate leakage. Victim-module migration most strongly reduces steady-state exposure, but its transition and state-transfer overhead remain observable. Detecting a change is generally easier than localizing the new path.

> **Experiment 1.5 conclusion:** Remapping can reduce a stable path fingerprint, but reconfiguration itself becomes a timing channel.

### Experiment 1.6: Unknown Placement and Allocation State

**Question:** Can the attacker detect a victim without knowing either tenant's physical placement or the active allocator/scheduler policy?

`phase1_06_unknown_placement_robustness.py` evaluates exact placement knowledge, attacker-only placement knowledge, logical identifiers only, both mappings unknown, and known/unknown control policies. Probe strategies include fixed one-shot, candidate cycling, knowledge-guided one-shot, and adaptive explore/exploit. Result files cover physical and strategy trials, paired rules, knowledge/allocator/scheduler summaries, features/samples, victim-presence and sharing-class metrics, predictions, false positives, and probe selection.

The simple paired rule retains approximately 53.6%-55.8% detection across the evaluated strategies. The random-forest victim-presence classifier reaches approximately 73.9%-75.1% accuracy across knowledge levels. Placement-class accuracy is approximately 87.5%-88.6% with exact mappings and approximately 67.2%-69.4% when both mappings are unknown. Exact placement is therefore more important for identifying the sharing class than for detecting victim activity. Fixed one-shot probing has the best specificity and approximately the best overall learned accuracy; candidate cycling is the strongest simple non-ML alternative; adaptive probing improves sensitivity but increases false positives and disruption, and matches the oracle-selected probe in only about one-third of trials.

The optional learned analysis can be completed from existing `unknown_placement_samples.csv` without rerunning physical trials:

```bash
python phase1_06_finish_random_forest_standalone.py
python phase1_06_random_forest_fast.py
```

Both entry points write `unknown_placement_random_forest_metrics.csv` and `unknown_placement_random_forest_predictions.csv`.
`phase1_06_random_forest_fast.py` is the supplied compatibility filename and currently contains the same standalone implementation as `phase1_06_finish_random_forest_standalone.py`.

> **Experiment 1.6 conclusion:** Exact placement metadata is unnecessary for victim detection, but it improves sharing-class inference and targeted probe selection.

### Phase 1 Baseline Signal

Aligning the attacker to a coarse 20 µs observation window changes the result materially. The windowed baseline uses the same five workloads, P1 disjoint placement, a serialized hub, and a light periodic probe:

| Victim | Cross-module operations | Average excess probe latency | Contention-observed fraction |
|---|---:|---:|---:|
| BV n=19 | 16 | 88.5 ns | 10.4% |
| SAT n=11 | 42 | 672.3 ns | 29.2% |
| DNN n=16 | 72 | 678.5 ns | 52.1% |
| QFT n=18 | 217 | 14,144.0 ns | 97.9% |
| Square root n=18 | 247 | 17,767.5 ns | 100.0% |

The ordering is not perfectly determined by operation count—timing and burst structure also matter—but the two most communication-heavy workloads produce the strongest signals by a wide margin.

### Main Phase 1 Findings

1. **Temporal localization turns a null control into a strong signal.** The untargeted full-run control records no contention, while a coarse window aligned to victim activity exposes all five workloads. Expanding the observation window from 5 µs to 40 µs dilutes average contention from 73.3% to 43.5%, showing that irrelevant probes reduce signal concentration.

2. **Probe intensity increases both observability and interference.** Across the five-workload probe-rate sweep, moving from 0.25x to 4x raises average contention from 50.0% to 96.8% and average excess latency from 5.61 µs to 11.69 µs, but also raises average victim slowdown from 1.012x to 1.205x. Probe choice shows the same tradeoff: the bursty-entangling probe reaches 95.4% contention but causes 1.380x average slowdown, whereas the light-periodic probe yields 57.9% contention with 1.037x slowdown.

3. **Disjoint compute placement does not remove a serialized-hub side channel.** With hub capacity 1, P1 disjoint and P2 one-module-overlap configurations produce the same five-workload average signal (6.67 µs excess latency and 57.9% contention), because the hub is the shared bottleneck. Raising capacity to 2 eliminates the signal in the modeled P1 case, but P2 retains 1.63 µs average excess latency and 37.9% contention through the remaining shared resources.

4. **Isolation and control-plane policy are security parameters.** Module-exclusive tenancy produces zero modeled leakage, while time-sliced module sharing produces 253.6 ns unconditional mean leakage and a 70.7% detection probability. Remote-operation scheduling changes mean absolute attacker timing shifts from 7.7 ns (`link_aware`) to 320.5 ns (`first_come_first_served`); the static circuit-layer scheduler also has the highest mean latency, deadline-miss rate, and rejection rate in its policy summary.

5. **Dynamic reconfiguration leaves a timing fingerprint.** Per-operation path selection produces the largest mean absolute timing change (131.6 ns) and 100% path-change detection accuracy in the mechanism summary. Communication-qubit reassignment creates a 1.67x transient leakage amplification. These results indicate that rerouting can move or reshape leakage rather than simply remove it.

6. **Incomplete placement knowledge reduces precision but does not erase detection.** Across the Phase 1.06 knowledge conditions, paired detection remains approximately 53.6%-55.8% for the evaluated one-shot and adaptive strategies, with the first observable change appearing after roughly 23-24 probes on average. Adaptive exploration has the highest paired detection rate (55.8%) without requiring exact physical placement.

7. **The signal supports circuit fingerprinting and survives moderate noise.** QAOA circuits from 5 through 15 qubits retain distinct excess-latency structures. The noise study shows mean signals remain close to their clean values under the evaluated 5 ns and 20 ns timestamp/scheduler perturbations; 50 ns combined noise increases variance but does not uniformly collapse the workload ordering.

### Combined Causal Model

```text
Allocation policy
    -> physical tenant placement
    -> overlap on exclusive resources
    -> communication-qubit and endpoint allocation
    -> scheduler, queue, and routing behavior
    -> attacker-visible timing, failure, reuse, and transition signals
```

Leakage is strongest when tenants share communication qubits, endpoint locks, local routes, reset/readout/feedforward pipelines, links, switch paths, serialized hub service, allocation queues, or reconfiguration transitions. Multi-tenancy, module co-location, logical-qubit sharing, dynamic allocation, scheduling, or rerouting are not independently sufficient. The actual isolation boundary determines exposure.

Architecture also changes the *form* of the signal: queue delay, tail latency, failure/rejection, redirection, endpoint reuse, EPR-induced speedup, fairness periodicity, dependency bursts, phase boundaries, rerouting transients, or state-transfer overhead. A defense must therefore check all observables rather than only average waiting time.

### Security-Performance Tradeoffs

- Endpoint separation reduces overlap but increases fragmentation and rejection.
- Dedicated interfaces consume additional communication qubits and reduce admission flexibility.
- Reservations reduce differential queue timing but can strand capacity or expose failures.
- Deep queues improve admission while increasing waiting-time leakage; shallow queues expose rejection.
- Overprovisioning removes endpoint pressure but increases hardware cost and may leave link/hub bottlenecks.
- Link-aware and endpoint-aware schedulers reduce timing magnitude while changing throughput and fairness.
- EPR prefetching improves completion but exposes availability, wastage, and phase state.
- Module migration reduces steady-state exposure but adds transition and state-transfer leakage.

### Phase 1 Decision Gate

Phase 1 proceeds to the stronger attacker-capability study if at least one channel survives dynamic placement, at least two realistic scheduler policies, unknown victim placement, low attacker self-contention, and acceptable victim disruption.

| Criterion | Result | Evidence |
|---|---|---|
| Dynamic placement | **Passed** | Leakage survives when allocation creates overlap on an exclusive endpoint, link, path, or hub resource. |
| Multiple realistic schedulers | **Passed** | Signals remain under FCFS, round robin, dependency-ready, fair, priority, randomized, and other policies. |
| Unknown victim placement | **Passed** | Victim-presence accuracy remains about 74%-75%, compared with 50% chance. |
| Low attacker self-contention | **Passed** | Probe 3 at a 420 ns period has approximately zero attacker-only waiting in the calibrated baseline. |
| Acceptable victim disruption | **Passed** | The preferred light-periodic baseline averages 1.037x victim slowdown; more aggressive probes are reported separately. |

> **Decision:** Phase 1 passes the gate. Proceed to the stronger security study while preserving separate timing-only and allocator-aware threat models.

### Strongest Defensible Phase 1 Claim

> Dynamic placement, communication-qubit allocation, scheduler policy, and rerouting reshape the probability and manifestation of exposure but do not generally eliminate black-box timing leakage. The channel persists when tenants overlap on an exclusive communication resource, even under unknown placement and low-contention probing. Conversely, shared-module execution can be leakage-free when communication qubits, endpoint locks, routing paths, and reset/control pipelines are fully isolated.

### Known Limitations

1. Gate, communication, reset, and switching times are fixed or simplified.
2. Communication qubits, endpoints, links, switches, and reset resources are architectural abstractions.
3. EPR generation, prefetch, and expiration are simplified.
4. Physical noise, drift, and calibration variability remain limited to the modeled profiles.
5. Dependency and coherence-deadline models are simplified.
6. Allocator and scheduler policies are synthetic rather than vendor implementations.
7. Some sub-sweeps use limited random trials.
8. The attacker uses a fixed probe family with a small set of strategies.
9. Learned classifiers are trained on simulator-derived features.
10. No direct hardware validation has been performed.
11. Some learned attacks use allocator-visible features beyond end-to-end timing.
12. Reconfiguration localization is weaker than change detection.
13. Migration assumes state transfer is supported by the abstraction.

### Reproducibility Checklist

- Record the Git commit, Python version, package versions, configurations, and random seeds.
- Preserve all QASM workloads and representative compressed request logs.
- Apply identical conditions to attacker-only, victim-only, and combined controls.
- Separate successful, failed, rejected, redirected, and partial executions.
- Report positive delays, negative speedups, failure leakage, and combined observable changes separately.
- Separate timing-only, allocator-aware, placement-aware, and placement-unknown attackers.
- Condition conclusions on actual resource sharing rather than logical placement labels alone.
- Report victim slowdown only for successful victim executions.
- Distinguish physical utilization from aggregate resource demand.
- Retain corrected time-sliced outputs and the Phase 1.03 postprocessed summaries.

### Next Phase

Phase 1 establishes channel survival. The next phase should deepen attacker-capability and defense evaluation: victim-activity detection, shared-resource inference, communication-volume estimation, workload-family and exact-circuit classification, communication-phase reconstruction, adaptive probing, robustness across system uncertainty, and mitigation using the isolation boundaries identified here. The existing QAOA fingerprinting results provide an initial circuit-classification case study rather than closing this broader question.

## Results

The following sections summarize the original `master` result families. They remain part of this branch and provide the architectural and attack-model foundation extended by the Phase 1 black-box studies above.

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

The original Experiment A studies establish the basic shared-entanglement-service timing channel. Phase 1 extends that mechanism to a stricter measurement model: the attacker actively submits probes but infers victim behavior only from its own completion timing. Refer to `blackbox_results/baseline/` for the untargeted control and `blackbox_window_results/baseline/` for the window-aligned five-workload comparison.

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

1. **Execution mode establishes exposure, but timing alignment determines whether it is observable.** Monolithic execution avoids remote shared-resource traffic. Modular execution creates the opportunity for leakage, yet the zero-signal `blackbox_results/` control shows that a probe must overlap the relevant communication window to observe it.

2. **Communication structure controls signal strength.** In the windowed Phase 1 baseline, BV produces only 88.5 ns average excess latency, while QFT and square root produce 14.1 µs and 17.8 µs. Cross-module count is a strong indicator, with event timing and burstiness explaining differences among workloads with closer counts.

3. **Attack sensitivity and attacker-induced slowdown must be reported together.** Denser or burstier probes improve detection but perturb the victim more strongly. The light-periodic baseline is therefore a more conservative security measurement than the high-rate or bursty configurations.

4. **Placement is not an isolation guarantee when infrastructure remains shared.** P1 disjoint placement still leaks through a capacity-1 hub. In the modeled capacity-2 case, that P1 signal disappears, while P2 overlap retains leakage. Capacity and isolation boundaries matter jointly.

5. **Schedulers, allocators, tenancy models, and reconfiguration policies reshape the channel.** Phase 1.01-1.05 show that leakage is a property of the full resource-management stack, not only the circuit or physical topology. A mitigation can suppress one contention source while introducing failure, queueing, transition, or path-selection observables elsewhere.

6. **Timing supports activity detection and workload fingerprinting under imperfect knowledge.** Phase 1.06 retains moderate detection without exact physical placement, and the QAOA results show separable timing structure across related circuits under the evaluated noise profiles.

### Implications for Quantum Security

- Compute-module separation alone is insufficient if tenants still share serialized hubs, links, communication qubits, queues, or reset pipelines.
- The modeled attacker is active only in the sense that it submits ordinary probes; its inference is black-box and uses no victim trace or privileged scheduler state.
- Detection results must be interpreted with probe overhead: near-saturation signals can be easy to detect but less stealthy and more disruptive.
- Useful mitigations include increasing bottleneck capacity, enforcing module/interface isolation, reducing timing determinism, and designing schedulers that limit attacker-visible queue coupling.
- Mitigations require end-to-end evaluation because rerouting, reservation, or isolation can replace latency leakage with failure or transition leakage.

### Future Work

- Implement additional attack tiers (active interference)
- Evaluate larger-scale architectures (16+ modules)
- Develop timing-oblivious scheduling algorithms
- Investigate quantum-specific countermeasures

## Setup and Usage

### Environment Setup
```bash
python -m pip install qiskit pandas numpy matplotlib scikit-learn
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

# Phase 1 black-box baselines and sensitivity sweeps
python run_attack_tier1_p1_static_blackbox_baseline.py
python run_attack_tier1_p1_static_blackbox_window_baseline.py
python run_atack_tier1_p1_static_blackbox_observation_window_sweep.py
python run_atack_tier1_p1_static_blackbox_probe_rate_sweep.py
python run_atack_tier1_p1_static_blackbox_probe_type_sweep.py
python run_attack_tier1_p1_static_blackbox_p1_vs_p2_placement_sweep.py
python run_attack_tier1_p1_static_blackbox_hub_capacity_sweep.py
python run_attack_tier1_p1_static_blackbox_random_victim_start_sweep.py

# Phase 1 system-factor studies
python phase1_01_job_module_allocation.py
python phase1_02_tenancy_models.py
python phase1_03_communication_qubit_allocation.py
python phase1_03_postprocess.py
python phase1_04_remote_operation_schedulers.py
python phase1_05_dynamic_rerouting_remapping.py
python phase1_06_unknown_placement_robustness.py

# Optional Phase 1.06 learned analysis from saved samples
python phase1_06_finish_random_forest_standalone.py
python phase1_06_random_forest_fast.py

# Plotting
python plot_baseline_stats.py
```

### Generating Reports
The original experiments write JSON summaries and PNG plots. Phase 1 additionally writes aggregate/trial/request-level CSV files. `blackbox_results/` is the compact untargeted control corpus; `blackbox_window_results/` is the complete windowed and system-factor corpus. Several scripts can be computationally expensive and may overwrite result files with the same configuration, so preserve the committed artifacts before rerunning large sweeps.

## References

- CCS 2026 Paper: See `CCS_2026 (1).pdf` for the full conference submission
- NetSquid Documentation: https://netsquid.org/
- Qiskit Documentation: https://qiskit.org/

## Authors

[Add author information if available]

## License

[Add license information if available]
