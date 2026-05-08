#!/usr/bin/env python3
"""
Baseline Architecture M:
Local modular superconducting DQC with FIVE compute modules

- nodes: superconducting transmon compute modules
- interconnect: short-range cryogenic microwave / coax links
- topology: shared-star / hub modular architecture

Topology:
    module_0 \
    module_1  \
    module_2   >---- hub_1
    module_3  /
    module_4 /

This is an architecture-level model, not device physics.
It is designed to receive trace events extracted from QASM workflows.

Core rule:
- local event        -> execute inside the owning compute module
- cross-module event -> send remote-operation request to hub_0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
# Hub model
# ============================================================

class SharedHub:
    """
    Shared interconnect-service node.

    Receives cross-module requests, stores them in a queue,
    and marks them complete when serviced.

    This is the architecture-level shared resource, not a BSM station.
    """
    def __init__(self, hub_id: str):
        self.hub_id = hub_id
        self.pending_requests: List[RemoteOperationRequest] = []
        self.active_requests: List[RemoteOperationRequest] = []
        self.completed_requests: List[RemoteOperationRequest] = []

    def receive_request(self, req: RemoteOperationRequest):
        req.status = "queued_at_hub"
        self.pending_requests.append(req)

    def service_next(self) -> Optional[RemoteOperationRequest]:
        """
        Minimal baseline policy:
        FIFO service of one request at a time.
        """
        if not self.pending_requests:
            return None

        req = self.pending_requests.pop(0)
        req.status = "servicing"
        self.active_requests.append(req)
        return req

    def complete_request(self, request_id: int) -> Optional[RemoteOperationRequest]:
        for i, req in enumerate(self.active_requests):
            if req.request_id == request_id:
                finished = self.active_requests.pop(i)
                finished.status = "completed"
                self.completed_requests.append(finished)
                return finished
        return None

    def describe(self):
        return {
            "hub_id": self.hub_id,
            "node_type": "shared_hub_service_node",
            "pending_requests": len(self.pending_requests),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
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
    def __init__(self, qubit_to_module: Dict[int, str], link_latency_ns: int = 10):
        self.architecture_type = "local_modular_superconducting_dqc"
        self.node_type = "superconducting_transmon_compute_module"
        self.interconnect_type = "short_range_cryogenic_microwave_coax"
        self.topology_type = "shared_star_hub_five_modules"

        self.expected_modules = [f"module_{i}" for i in range(5)]
        self.hub = SharedHub(hub_id="hub_0")
        self.compute_modules: Dict[str, ComputeModule] = {}
        self.links: Dict[str, CryogenicLink] = {}
        self.qubit_to_module = dict(qubit_to_module)

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

    def service_one_hub_request(self):
        """
        Minimal baseline 'run one remote service step':
        - FIFO pick next request
        - immediately mark complete
        """
        req = self.hub.service_next()
        if req is None:
            return None

        finished = self.hub.complete_request(req.request_id)

        if finished is not None:
            self.compute_modules[finished.source_module].mark_remote_complete(
                finished.request_id
            )

        return finished

    def describe(self):
        print("=== Fixed Architecture M: Five-Module Baseline ===")
        print("architecture_type :", self.architecture_type)
        print("node_type         :", self.node_type)
        print("interconnect_type :", self.interconnect_type)
        print("topology_type     :", self.topology_type)

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

    Example:
      num_qubits=10
      q0,q1   -> module_0
      q2,q3   -> module_1
      q4,q5   -> module_2
      q6,q7   -> module_3
      q8,q9   -> module_4
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
        # fallback rebalance
        mapping = {}
        for q in range(num_qubits):
            mapping[q] = f"module_{q % 5}"

    return mapping


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example: 10 qubits across five modules
    qubit_to_module = build_five_module_qubit_mapping(num_qubits=10)

    arch = FiveModuleLocalModularSuperconductingDQC(
        qubit_to_module=qubit_to_module,
        link_latency_ns=10,
    )

    arch.describe()
    arch.print_qubit_mapping()
    arch.print_stick_diagram()

    # Example local event
    local_event = TraceEvent(
        step=0,
        op_name="x",
        qubits=[0],
        clbits=[],
        params=[],
        placement_style="static_distributed",
        modules_touched=["module_0"],
        is_cross_module=False,
    )

    # Example cross-module event
    remote_event = TraceEvent(
        step=1,
        op_name="cx",
        qubits=[0, 8],
        clbits=[],
        params=[],
        placement_style="static_distributed",
        modules_touched=["module_0", "module_4"],
        is_cross_module=True,
    )

    print("\n=== Routing events ===")
    arch.route_trace_event(local_event)
    arch.route_trace_event(remote_event)

    print("\nAfter routing:")
    arch.describe()

    print("\n=== Service one hub request ===")
    finished = arch.service_one_hub_request()
    print("Finished request:", finished)

    print("\nAfter servicing:")
    arch.describe()
