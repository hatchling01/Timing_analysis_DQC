#!/usr/bin/env python3
"""
Updated Architecture M:
Local modular superconducting DQC with FIVE compute modules

- nodes: superconducting transmon compute modules
- interconnect: short-range cryogenic microwave / coax links
- topology: shared-star / hub modular architecture

Topology:
    module_0 \
    module_1  \
    module_2   >---- hub_0
    module_3  /
    module_4 /

This is an architecture-level model, not device physics.
It is designed to receive trace events extracted from QASM workflows.

Updated hub model:
- deterministic remote-transfer service
- finite concurrent transfer slots
- module-pair occupancy constraints
- request queue + admission control
- event-driven time advancement

Core rule:
- local event        -> execute inside the owning compute module
- cross-module event -> send remote-operation request to hub_0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# Trace event
# ============================================================

@dataclass
class TraceEvent:
    step: int
    op_name: str
    qubits: List[int]
    clbits: List[int]
    params: List
    placement_style: str

    # Distributed-trace info
    modules_touched: List[str] = field(default_factory=list)
    is_cross_module: bool = False

    # Sequential modular info
    stage: Optional[int] = None
    active_module: Optional[str] = None
    transfer_event: bool = False


# ============================================================
# Remote request
# ============================================================

@dataclass
class RemoteOperationRequest:
    request_id: int
    source_module: str
    target_modules: List[str]
    original_event: TraceEvent
    status: str = "pending"

    # Timing metadata
    arrival_time_ns: Optional[int] = None
    start_time_ns: Optional[int] = None
    end_time_ns: Optional[int] = None
    service_time_ns: Optional[int] = None

    # Derived metrics
    waiting_time_ns: Optional[int] = None
    turnaround_time_ns: Optional[int] = None

    def all_modules_involved(self) -> List[str]:
        return [self.source_module] + list(self.target_modules)


# ============================================================
# Link model
# ============================================================

@dataclass
class CryogenicLink:
    """
    Dedicated short-range cryogenic microwave/coax link
    between one compute module and the shared hub.
    """
    link_id: str
    src_node: str
    dst_node: str
    latency_ns: int = 10
    is_busy: bool = False

    def describe(self):
        return {
            "link_id": self.link_id,
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "latency_ns": self.latency_ns,
            "is_busy": self.is_busy,
            "link_type": "cryogenic_microwave_coax",
        }


# ============================================================
# Compute module model
# ============================================================

class ComputeModule:
    """
    Architectural execution unit.

    Holds:
    - local qubits
    - local execution log
    - remote request log
    - waiting remote ops

    Does:
    - execute local trace events
    - forward cross-module events to the hub
    """
    def __init__(self, module_id: str, local_qubits: List[int], hub_id: str):
        self.module_id = module_id
        self.local_qubits = sorted(local_qubits)
        self.hub_id = hub_id

        self.local_event_log: List[TraceEvent] = []
        self.remote_request_log: List[RemoteOperationRequest] = []
        self.waiting_remote_ops: List[int] = []

    def owns_all_qubits(self, qubits: List[int]) -> bool:
        return all(q in self.local_qubits for q in qubits)

    def execute_local_event(self, event: TraceEvent):
        self.local_event_log.append(event)

    def submit_remote_event(
        self,
        event: TraceEvent,
        request_id: int,
        target_modules: List[str],
    ) -> RemoteOperationRequest:
        req = RemoteOperationRequest(
            request_id=request_id,
            source_module=self.module_id,
            target_modules=target_modules,
            original_event=event,
            status="submitted_to_hub",
        )
        self.remote_request_log.append(req)
        self.waiting_remote_ops.append(request_id)
        return req

    def mark_remote_complete(self, request_id: int):
        if request_id in self.waiting_remote_ops:
            self.waiting_remote_ops.remove(request_id)

    def describe(self):
        return {
            "module_id": self.module_id,
            "node_type": "superconducting_transmon_compute_module",
            "local_qubits": self.local_qubits,
            "hub_id": self.hub_id,
            "num_local_events": len(self.local_event_log),
            "num_remote_requests": len(self.remote_request_log),
            "waiting_remote_ops": list(self.waiting_remote_ops),
        }


# ============================================================
# Updated hub model
# ============================================================

class SharedHub:
    """
    Shared superconducting interconnect-service hub.

    Realism added:
    - deterministic transfer service time
    - finite number of concurrent transfer slots
    - module occupancy constraints:
        a module cannot participate in more than one active remote transfer
    - queued admission with simple routing/resource logic
    - event-driven time advancement

    This remains architecture-level, not device-physics level.
    """
    def __init__(
        self,
        hub_id: str,
        max_concurrent_transfers: int = 2,
        setup_latency_ns: int = 20,
        transfer_latency_ns: int = 80,
    ):
        self.hub_id = hub_id
        self.max_concurrent_transfers = max_concurrent_transfers
        self.setup_latency_ns = setup_latency_ns
        self.transfer_latency_ns = transfer_latency_ns

        self.current_time_ns: int = 0

        self.pending_requests: List[RemoteOperationRequest] = []
        self.active_requests: List[RemoteOperationRequest] = []
        self.completed_requests: List[RemoteOperationRequest] = []

        self.busy_modules: set = set()

    def _deterministic_service_time(self, req: RemoteOperationRequest) -> int:
        # Simple deterministic semantics:
        # setup + source-link + target-link + transfer body
        num_targets = max(1, len(req.target_modules))
        return self.setup_latency_ns + self.transfer_latency_ns + 10 * num_targets

    def _can_start(self, req: RemoteOperationRequest) -> bool:
        if len(self.active_requests) >= self.max_concurrent_transfers:
            return False

        for mod in req.all_modules_involved():
            if mod in self.busy_modules:
                return False

        return True

    def _start_request(self, req: RemoteOperationRequest):
        req.status = "active_transfer"
        req.start_time_ns = self.current_time_ns
        req.service_time_ns = self._deterministic_service_time(req)
        req.end_time_ns = req.start_time_ns + req.service_time_ns
        req.waiting_time_ns = req.start_time_ns - req.arrival_time_ns

        self.active_requests.append(req)
        for mod in req.all_modules_involved():
            self.busy_modules.add(mod)

    def _try_admit_requests(self):
        """
        Admit as many queued requests as possible subject to:
        - max concurrent transfer slots
        - module occupancy constraints

        We scan the queue from front to back and start any admissible request.
        """
        if not self.pending_requests:
            return

        remaining = []
        for req in self.pending_requests:
            if self._can_start(req):
                self._start_request(req)
            else:
                remaining.append(req)

            if len(self.active_requests) >= self.max_concurrent_transfers:
                remaining.extend(
                    self.pending_requests[self.pending_requests.index(req)+1:]
                )
                break

        self.pending_requests = remaining

    def receive_request(self, req: RemoteOperationRequest):
        req.status = "queued_at_hub"
        req.arrival_time_ns = self.current_time_ns
        self.pending_requests.append(req)
        self._try_admit_requests()

    def advance_time(self, delta_ns: int = 1) -> List[RemoteOperationRequest]:
        """
        Advance hub time and complete any finished transfers.

        Returns:
            list of requests completed during this advance
        """
        if delta_ns < 0:
            raise ValueError("delta_ns must be non-negative")

        self.current_time_ns += delta_ns
        completed_now: List[RemoteOperationRequest] = []

        still_active = []
        for req in self.active_requests:
            if req.end_time_ns is not None and req.end_time_ns <= self.current_time_ns:
                req.status = "completed"
                req.turnaround_time_ns = req.end_time_ns - req.arrival_time_ns
                completed_now.append(req)

                for mod in req.all_modules_involved():
                    if mod in self.busy_modules:
                        self.busy_modules.remove(mod)

                self.completed_requests.append(req)
            else:
                still_active.append(req)

        self.active_requests = still_active
        self._try_admit_requests()

        return completed_now

    def drain_all(self) -> List[RemoteOperationRequest]:
        """
        Run until all pending and active requests are completed.
        """
        completed_all: List[RemoteOperationRequest] = []

        while self.pending_requests or self.active_requests:
            if self.active_requests:
                next_end = min(req.end_time_ns for req in self.active_requests if req.end_time_ns is not None)
                delta = max(1, next_end - self.current_time_ns)
            else:
                delta = 1

            completed_now = self.advance_time(delta_ns=delta)
            completed_all.extend(completed_now)

        return completed_all

    def describe(self):
        avg_wait = 0.0
        avg_turnaround = 0.0
        if self.completed_requests:
            waits = [r.waiting_time_ns for r in self.completed_requests if r.waiting_time_ns is not None]
            turns = [r.turnaround_time_ns for r in self.completed_requests if r.turnaround_time_ns is not None]
            avg_wait = sum(waits) / len(waits) if waits else 0.0
            avg_turnaround = sum(turns) / len(turns) if turns else 0.0

        return {
            "hub_id": self.hub_id,
            "node_type": "shared_superconducting_interconnect_hub",
            "current_time_ns": self.current_time_ns,
            "max_concurrent_transfers": self.max_concurrent_transfers,
            "setup_latency_ns": self.setup_latency_ns,
            "transfer_latency_ns": self.transfer_latency_ns,
            "pending_requests": len(self.pending_requests),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "busy_modules": sorted(self.busy_modules),
            "avg_waiting_time_ns": avg_wait,
            "avg_turnaround_time_ns": avg_turnaround,
        }


# ============================================================
# Full five-module architecture
# ============================================================

class FiveModuleLocalModularSuperconductingDQC:
    """
    Fixed baseline architecture M with five compute modules.

    Topology:
        module_i <-> hub_0, for i in {0,1,2,3,4}

    No direct module-to-module links.
    """
    def __init__(
        self,
        qubit_to_module: Dict[int, str],
        link_latency_ns: int = 10,
        hub_max_concurrent_transfers: int = 2,
        hub_setup_latency_ns: int = 20,
        hub_transfer_latency_ns: int = 80,
        event_tick_ns: int = 5,
    ):
        self.architecture_type = "local_modular_superconducting_dqc"
        self.node_type = "superconducting_transmon_compute_module"
        self.interconnect_type = "short_range_cryogenic_microwave_coax"
        self.topology_type = "shared_star_hub_five_modules"

        self.expected_modules = [f"module_{i}" for i in range(5)]
        self.hub = SharedHub(
            hub_id="hub_0",
            max_concurrent_transfers=hub_max_concurrent_transfers,
            setup_latency_ns=hub_setup_latency_ns,
            transfer_latency_ns=hub_transfer_latency_ns,
        )
        self.compute_modules: Dict[str, ComputeModule] = {}
        self.links: Dict[str, CryogenicLink] = {}
        self.qubit_to_module = dict(qubit_to_module)

        self.event_tick_ns = event_tick_ns
        self._request_counter = 0

        self._validate_mapping()
        self._build_modules_and_links(link_latency_ns)

    def _validate_mapping(self):
        unknown_modules = sorted(set(self.qubit_to_module.values()) - set(self.expected_modules))
        if unknown_modules:
            raise ValueError(
                f"Found unknown module names in qubit_to_module: {unknown_modules}. "
                f"Expected only: {self.expected_modules}"
            )

        present_modules = set(self.qubit_to_module.values())
        missing_modules = [m for m in self.expected_modules if m not in present_modules]
        if missing_modules:
            raise ValueError(
                f"Five-module baseline requires all five modules to appear in the mapping. "
                f"Missing: {missing_modules}"
            )

    def _build_modules_and_links(self, link_latency_ns: int):
        modules_to_qubits: Dict[str, List[int]] = {m: [] for m in self.expected_modules}
        for q, mod in self.qubit_to_module.items():
            modules_to_qubits[mod].append(q)

        for mod in self.expected_modules:
            self.compute_modules[mod] = ComputeModule(
                module_id=mod,
                local_qubits=modules_to_qubits[mod],
                hub_id=self.hub.hub_id,
            )

            link_id = f"{mod}_to_{self.hub.hub_id}"
            self.links[link_id] = CryogenicLink(
                link_id=link_id,
                src_node=mod,
                dst_node=self.hub.hub_id,
                latency_ns=link_latency_ns,
            )

    def next_request_id(self) -> int:
        rid = self._request_counter
        self._request_counter += 1
        return rid

    def route_trace_event(self, event: TraceEvent):
        """
        Core mapping rule:
        - local event: execute inside owning module
        - cross-module event: create hub-mediated remote request
        """
        if not event.qubits:
            return

        owning_modules = sorted({self.qubit_to_module[q] for q in event.qubits})

        # Local event
        if len(owning_modules) == 1:
            mod = owning_modules[0]
            self.compute_modules[mod].execute_local_event(event)
            return

        # Cross-module event
        source_module = event.active_module if event.active_module else owning_modules[0]
        if source_module not in self.compute_modules:
            source_module = owning_modules[0]

        target_modules = [m for m in owning_modules if m != source_module]

        request_id = self.next_request_id()
        req = self.compute_modules[source_module].submit_remote_event(
            event=event,
            request_id=request_id,
            target_modules=target_modules,
        )
        self.hub.receive_request(req)

    def advance_architecture_time(self, delta_ns: Optional[int] = None):
        """
        Advance global architecture time by stepping the hub and notifying
        source modules of any completed remote requests.
        """
        if delta_ns is None:
            delta_ns = self.event_tick_ns

        completed_now = self.hub.advance_time(delta_ns=delta_ns)
        for req in completed_now:
            self.compute_modules[req.source_module].mark_remote_complete(req.request_id)

        return completed_now

    def drain_hub(self):
        """
        Finish all pending/active remote transfers.
        """
        completed_all = self.hub.drain_all()
        for req in completed_all:
            self.compute_modules[req.source_module].mark_remote_complete(req.request_id)
        return completed_all

    def describe(self):
        print("=== Fixed Architecture M: Five-Module Baseline ===")
        print("architecture_type :", self.architecture_type)
        print("node_type         :", self.node_type)
        print("interconnect_type :", self.interconnect_type)
        print("topology_type     :", self.topology_type)
        print("event_tick_ns     :", self.event_tick_ns)

        print("\n=== Hub ===")
        print(self.hub.describe())

        print("\n=== Compute Modules ===")
        for mod_id in self.expected_modules:
            print(self.compute_modules[mod_id].describe())

        print("\n=== Links ===")
        for mod_id in self.expected_modules:
            link_id = f"{mod_id}_to_{self.hub.hub_id}"
            print(self.links[link_id].describe())

    def print_qubit_mapping(self):
        print("=== Qubit -> Module Mapping ===")
        for q in sorted(self.qubit_to_module):
            print(f"q{q} -> {self.qubit_to_module[q]}")

    def print_stick_diagram(self):
        print("\n=== Five-Module Stick Diagram ===")
        print(" module_0 \\")
        print(" module_1  \\")
        print(" module_2   >---- hub_0")
        print(" module_3  /")
        print(" module_4 /")
        print("\nEach spoke = short-range cryogenic microwave / coax link")


# ============================================================
# Helper to build a simple five-module mapping
# ============================================================

def build_five_module_qubit_mapping(num_qubits: int) -> Dict[int, str]:
    """
    Simple contiguous-block assignment across five modules.
    """
    if num_qubits < 5:
        raise ValueError("Need at least 5 qubits to populate all five modules.")

    mapping = {}
    block_size = (num_qubits + 5 - 1) // 5  # ceil division

    for q in range(num_qubits):
        module_id = min(q // block_size, 4)
        mapping[q] = f"module_{module_id}"

    # Ensure all five modules appear, even for awkward sizes
    present = set(mapping.values())
    expected = {f"module_{i}" for i in range(5)}
    if present != expected:
        mapping = {}
        for q in range(num_qubits):
            mapping[q] = f"module_{q % 5}"

    return mapping


# ============================================================
# Trace normalization + processing
# ============================================================

def normalize_trace_entry(entry: dict, trace_type: str) -> TraceEvent:
    """
    Convert one raw trace dictionary into a common TraceEvent object.

    Supported trace_type values:
    - "monolithic"
    - "static_distributed"
    - "sequential_modular"
    """
    if trace_type == "monolithic":
        return TraceEvent(
            step=entry["step"],
            op_name=entry["op_name"],
            qubits=entry["qubits"],
            clbits=entry.get("clbits", []),
            params=entry.get("params", []),
            placement_style=entry.get("placement_style", "monolithic_local"),
            modules_touched=[entry.get("module", "module_0")],
            is_cross_module=False,
            stage=None,
            active_module=entry.get("module", "module_0"),
            transfer_event=False,
        )

    elif trace_type == "static_distributed":
        return TraceEvent(
            step=entry["step"],
            op_name=entry["op_name"],
            qubits=entry["qubits"],
            clbits=entry.get("clbits", []),
            params=entry.get("params", []),
            placement_style=entry.get("placement_style", "static_distributed"),
            modules_touched=entry.get("modules_touched", entry.get("modules", [])),
            is_cross_module=entry.get("is_cross_module", entry.get("cross_module", False)),
            stage=None,
            active_module=None,
            transfer_event=entry.get("communication_event", False),
        )

    elif trace_type == "sequential_modular":
        return TraceEvent(
            step=entry["step"],
            op_name=entry["op_name"],
            qubits=entry["qubits"],
            clbits=entry.get("clbits", []),
            params=entry.get("params", []),
            placement_style=entry.get("placement_style", "sequential_modular"),
            modules_touched=entry.get("modules_touched", entry.get("modules", [])),
            is_cross_module=entry.get("is_cross_module", entry.get("cross_module", False)),
            stage=entry.get("stage", None),
            active_module=entry.get("active_module", None),
            transfer_event=entry.get("transfer_event", False),
        )

    else:
        raise ValueError(
            f"Unsupported trace_type='{trace_type}'. "
            f"Use 'monolithic', 'static_distributed', or 'sequential_modular'."
        )


def process_trace(
    arch,
    trace: list,
    trace_type: str,
    service_policy: str = "event_driven",
):
    """
    Process a full trace through the updated five-module architecture.

    service_policy:
    - "event_driven": route each event and advance architecture time by one event tick
    - "drain_end": route all events first, then drain hub at the end
    """
    if service_policy not in {"event_driven", "drain_end"}:
        raise ValueError("service_policy must be 'event_driven' or 'drain_end'")

    normalized_events = []

    for entry in trace:
        event = normalize_trace_entry(entry, trace_type)
        normalized_events.append(event)

        arch.route_trace_event(event)

        if service_policy == "event_driven":
            arch.advance_architecture_time()

    if service_policy == "drain_end":
        arch.drain_hub()
    else:
        # Even in event-driven mode, finish anything still in flight.
        arch.drain_hub()

    return normalized_events


def print_execution_summary(arch, normalized_events: list, trace_type: str):
    """
    Print a compact summary after processing a trace.
    """
    total_events = len(normalized_events)
    total_cross = sum(1 for e in normalized_events if e.is_cross_module)
    total_local = total_events - total_cross

    print("\n=== Execution Summary ===")
    print("trace_type            :", trace_type)
    print("total_events          :", total_events)
    print("local_events          :", total_local)
    print("cross_module_events   :", total_cross)

    print("\n=== Per-module Local Event Counts ===")
    for mod_id in arch.expected_modules:
        num_local = len(arch.compute_modules[mod_id].local_event_log)
        print(f"{mod_id}: {num_local}")

    print("\n=== Hub Summary ===")
    hub_info = arch.hub.describe()
    for k, v in hub_info.items():
        print(f"{k:22s}: {v}")

    if arch.hub.completed_requests:
        waits = [r.waiting_time_ns for r in arch.hub.completed_requests if r.waiting_time_ns is not None]
        turns = [r.turnaround_time_ns for r in arch.hub.completed_requests if r.turnaround_time_ns is not None]
        if waits:
            print("min_waiting_time_ns   :", min(waits))
            print("max_waiting_time_ns   :", max(waits))
        if turns:
            print("min_turnaround_ns     :", min(turns))
            print("max_turnaround_ns     :", max(turns))

    if trace_type == "sequential_modular":
        stage_counts = {}
        stage_cross_counts = {}
        for e in normalized_events:
            if e.stage is not None:
                stage_counts[e.stage] = stage_counts.get(e.stage, 0) + 1
                if e.is_cross_module:
                    stage_cross_counts[e.stage] = stage_cross_counts.get(e.stage, 0) + 1

        print("\n=== Sequential Stage Counts ===")
        for stage_id in sorted(stage_counts):
            cross_ct = stage_cross_counts.get(stage_id, 0)
            print(f"stage {stage_id}: total={stage_counts[stage_id]}, cross={cross_ct}")


def run_monolithic_trace(arch, trace):
    events = process_trace(
        arch=arch,
        trace=trace,
        trace_type="monolithic",
        service_policy="event_driven",
    )
    print_execution_summary(arch, events, "monolithic")
    return events


def run_static_distributed_trace(arch, trace):
    events = process_trace(
        arch=arch,
        trace=trace,
        trace_type="static_distributed",
        service_policy="event_driven",
    )
    print_execution_summary(arch, events, "static_distributed")
    return events


def run_sequential_modular_trace(arch, trace):
    events = process_trace(
        arch=arch,
        trace=trace,
        trace_type="sequential_modular",
        service_policy="event_driven",
    )
    print_execution_summary(arch, events, "sequential_modular")
    return events


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example 18-qubit mapping across five modules
    qubit_to_module = {
        0: "module_0",
        1: "module_0",
        2: "module_0",
        3: "module_0",
        4: "module_0",
        5: "module_0",
        6: "module_0",
        7: "module_0",
        8: "module_0",
        9: "module_1",
        10: "module_1",
        11: "module_1",
        12: "module_2",
        13: "module_2",
        14: "module_3",
        15: "module_3",
        16: "module_4",
        17: "module_4",
    }

    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=qubit_to_module,
        link_latency_ns=10,
        hub_max_concurrent_transfers=2,
        hub_setup_latency_ns=20,
        hub_transfer_latency_ns=80,
        event_tick_ns=5,
    )

    arch.describe()
    arch.print_qubit_mapping()
    arch.print_stick_diagram()

    # Example static trace with multiple cross-module requests
    static_trace = [
        {
            "step": 0,
            "op_name": "h",
            "qubits": [0],
            "clbits": [],
            "params": [],
            "placement_style": "static_distributed",
            "modules": ["module_0"],
            "cross_module": False,
        },
        {
            "step": 1,
            "op_name": "cx",
            "qubits": [0, 9],
            "clbits": [],
            "params": [],
            "placement_style": "static_distributed",
            "modules": ["module_0", "module_1"],
            "cross_module": True,
        },
        {
            "step": 2,
            "op_name": "cx",
            "qubits": [12, 14],
            "clbits": [],
            "params": [],
            "placement_style": "static_distributed",
            "modules": ["module_2", "module_3"],
            "cross_module": True,
        },
        {
            "step": 3,
            "op_name": "cx",
            "qubits": [16, 1],
            "clbits": [],
            "params": [],
            "placement_style": "static_distributed",
            "modules": ["module_4", "module_0"],
            "cross_module": True,
        },
    ]

    print("\n============= STATIC DISTRIBUTED RUN =============")
    run_static_distributed_trace(arch, static_trace)
