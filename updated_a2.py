#!/usr/bin/env python3
"""
Experiment A2 (updated): memory-only contention
with explicit local modular topology.

Topology:
    A_mod ----\
               \
                >---- S_mod  (shared memory-limited entanglement service)
               /
    V_mod ----/

Interpretation:
- A_mod = attacker module
- V_mod = victim module
- S_mod = shared local interconnect / switch module
- Both tenants submit EPR-generation requests to the same shared service at S_mod
- Multiple requests may be active in parallel, up to shared memory capacity K
- There is NO separate serialized BSM stage in A2
"""

import sys
import math
import random
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List
from collections import deque

import numpy as np
import netsquid as ns
from netsquid.protocols import Protocol


# ============================================================
# Utilities
# ============================================================

def banner():
    print("Python:", sys.executable)
    print("NumPy :", np.__version__)
    print("NetSquid:", ns.__version__)


def sim_init(seed: int = 1):
    ns.sim_reset()
    ns.set_random_state(seed)
    random.seed(seed)
    np.random.seed(seed)


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x)
    y = np.sort(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    allv = np.sort(np.concatenate([x, y]))
    cx = np.searchsorted(x, allv, side="right") / len(x)
    cy = np.searchsorted(y, allv, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


# ============================================================
# Request / logs
# ============================================================

@dataclass
class EPRRequest:
    tenant: str
    req_id: int
    t_req: int
    src_module: str
    dst_module: str
    on_done: Callable[[str, int, int, int], None]
    # callback signature: (tenant, req_id, t_req, t_done)


@dataclass
class RunLog:
    attacker_latencies: List[int]
    attacker_done_times: List[int]
    attacker_req_times: List[int]

    total_completed: int = 0
    attacker_completed: int = 0
    victim_completed: int = 0

    attacker_paths: List[str] = None
    victim_paths: List[str] = None

    def __post_init__(self):
        if self.attacker_paths is None:
            self.attacker_paths = []
        if self.victim_paths is None:
            self.victim_paths = []


# ============================================================
# Local modular topology objects
# ============================================================

class Module:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.neighbors: List[str] = []

    def connect(self, other: "Module"):
        if other.name not in self.neighbors:
            self.neighbors.append(other.name)
        if self.name not in other.neighbors:
            other.neighbors.append(self.name)


class LocalModularTopology:
    """
    Explicit local modular topology for A2.

    Modules:
      - A_mod : attacker module
      - V_mod : victim module
      - S_mod : shared switch / service module

    Logical connectivity:
      A_mod <-> S_mod
      V_mod <-> S_mod
    """
    def __init__(self):
        self.modules: Dict[str, Module] = {}

        self.modules["A_mod"] = Module("A_mod", "attacker_module")
        self.modules["V_mod"] = Module("V_mod", "victim_module")
        self.modules["S_mod"] = Module("S_mod", "shared_switch_module")

        self.modules["A_mod"].connect(self.modules["S_mod"])
        self.modules["V_mod"].connect(self.modules["S_mod"])

    def describe(self):
        print("\n=== Local Modular Topology ===")
        for name, mod in self.modules.items():
            print(f"{name:>6s} | role={mod.role:24s} | neighbors={mod.neighbors}")


# ============================================================
# Shared memory-limited service at S_mod
# ============================================================

class MemoryPool:
    """
    Shared finite-capacity pool.

    In A2 this is an abstract shared memory/buffer resource
    limiting how many in-flight entanglement-generation processes
    may proceed concurrently.
    """
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.in_use = 0

    def available(self) -> bool:
        return self.in_use < self.capacity

    def allocate(self):
        if not self.available():
            raise RuntimeError("MemoryPool.allocate() called when full")
        self.in_use += 1

    def free(self):
        if self.in_use <= 0:
            raise RuntimeError("MemoryPool.free() called when empty")
        self.in_use -= 1


class EntanglementServerMemoryOnly(Protocol):
    """
    A2 service model:
    - Shared service hosted at S_mod
    - Requests wait for admission if memory pool is full
    - Once admitted, generation proceeds in parallel
    - No separate serialized BSM stage

    Total service time for an admitted request:
        generation attempts + heralding time
    """

    def __init__(
        self,
        name: str,
        host_module: str,
        mem_capacity: int = 4,
        p_success: float = 0.2,
        attempt_time_ns: int = 2_000,
        herald_time_ns: int = 500,
    ):
        super().__init__(name=name)
        self.host_module = host_module
        self.q: Deque[EPRRequest] = deque()
        self.mem = MemoryPool(mem_capacity)

        if not (0.0 < p_success <= 1.0):
            raise ValueError("p_success must be in (0, 1].")

        self.p_success = float(p_success)
        self.attempt_time_ns = int(attempt_time_ns)
        self.herald_time_ns = int(herald_time_ns)

    def submit(self, req: EPRRequest):
        self.q.append(req)

    def _sample_num_attempts(self) -> int:
        if self.p_success >= 1.0:
            return 1
        u = random.random()
        attempts = int(math.ceil(math.log(1.0 - u) / math.log(1.0 - self.p_success)))
        return max(attempts, 1)

    def _sample_generation_time(self) -> int:
        attempts = self._sample_num_attempts()
        return attempts * self.attempt_time_ns + self.herald_time_ns

    def run(self):
        while True:
            started_any = False

            # Admit as many queued requests as possible while memory is available
            while self.q and self.mem.available():
                req = self.q.popleft()
                self.mem.allocate()
                generation_time = self._sample_generation_time()

                sub = _GenerationSubprotocol(
                    name=f"gen_{req.tenant}_{req.req_id}",
                    server=self,
                    req=req,
                    generation_time_ns=generation_time,
                )
                sub.start()
                started_any = True

            if not started_any:
                yield self.await_timer(1)


class _GenerationSubprotocol(Protocol):
    """
    One in-flight generation process occupying one memory slot.
    """
    def __init__(
        self,
        name: str,
        server: EntanglementServerMemoryOnly,
        req: EPRRequest,
        generation_time_ns: int,
    ):
        super().__init__(name=name)
        self.server = server
        self.req = req
        self.generation_time_ns = int(generation_time_ns)

    def run(self):
        yield self.await_timer(self.generation_time_ns)

        # Completion frees shared memory
        self.server.mem.free()

        t_done = ns.sim_time()
        self.req.on_done(self.req.tenant, self.req.req_id, self.req.t_req, t_done)


# ============================================================
# Tenant requesters
# ============================================================

class PeriodicRequesterProtocol(Protocol):
    """
    Periodic requester running at a local module.

    For A2:
    - attacker issues requests from A_mod toward S_mod
    - victim issues requests from V_mod toward S_mod
    """

    def __init__(
        self,
        name: str,
        server: EntanglementServerMemoryOnly,
        tenant: str,
        src_module: str,
        dst_module: str,
        period_ns: int,
        start_ns: int,
        stop_ns: int,
        log: RunLog,
    ):
        super().__init__(name=name)
        self.server = server
        self.tenant = tenant
        self.src_module = src_module
        self.dst_module = dst_module
        self.period_ns = int(period_ns)
        self.start_ns = int(start_ns)
        self.stop_ns = int(stop_ns)
        self.log = log
        self._next_id = 0

    def run(self):
        now = ns.sim_time()
        if self.start_ns > now:
            yield self.await_timer(self.start_ns - now)

        while ns.sim_time() < self.stop_ns:
            self._submit_once()
            yield self.await_timer(self.period_ns)

    def _submit_once(self):
        req_id = self._next_id
        self._next_id += 1
        t_req = ns.sim_time()

        def on_done(tenant, req_id, t_req, t_done):
            self.log.total_completed += 1
            path_str = f"{self.src_module}->{self.dst_module}"

            if tenant == "attacker":
                self.log.attacker_completed += 1
                self.log.attacker_req_times.append(t_req)
                self.log.attacker_done_times.append(t_done)
                self.log.attacker_latencies.append(t_done - t_req)
                self.log.attacker_paths.append(path_str)
            else:
                self.log.victim_completed += 1
                self.log.victim_paths.append(path_str)

        self.server.submit(
            EPRRequest(
                tenant=self.tenant,
                req_id=req_id,
                t_req=t_req,
                src_module=self.src_module,
                dst_module=self.dst_module,
                on_done=on_done,
            )
        )


# ============================================================
# Experiment runner
# ============================================================

def build_topology():
    return LocalModularTopology()


def run_once(
    seed: int,
    victim_on: bool,
    attacker_period_ns: int,
    victim_period_ns: int,
    sim_duration_ns: int = 200_000,
    warmup_ns: int = 20_000,
    mem_capacity: int = 4,
    server_p_success: float = 0.2,
    attempt_time_ns: int = 2_000,
    herald_time_ns: int = 500,
):
    """
    Run one A2 simulation on the explicit local modular topology.

    Topology:
        A_mod -> S_mod   (attacker traffic)
        V_mod -> S_mod   (victim traffic)

    A2 semantics:
    - shared memory-limited service at S_mod
    - multiple requests may be active in parallel, up to mem_capacity
    - no separate serialized BSM stage
    """
    sim_init(seed)

    topo = build_topology()
    log = RunLog(attacker_latencies=[], attacker_done_times=[], attacker_req_times=[])

    srv = EntanglementServerMemoryOnly(
        name="S_mod_mem_service",
        host_module="S_mod",
        mem_capacity=mem_capacity,
        p_success=server_p_success,
        attempt_time_ns=attempt_time_ns,
        herald_time_ns=herald_time_ns,
    )
    srv.start()

    start = warmup_ns
    stop = sim_duration_ns

    attacker = PeriodicRequesterProtocol(
        name="attacker_prober",
        server=srv,
        tenant="attacker",
        src_module="A_mod",
        dst_module="S_mod",
        period_ns=attacker_period_ns,
        start_ns=start,
        stop_ns=stop,
        log=log,
    )
    attacker.start()

    if victim_on:
        victim = PeriodicRequesterProtocol(
            name="victim_load",
            server=srv,
            tenant="victim",
            src_module="V_mod",
            dst_module="S_mod",
            period_ns=victim_period_ns,
            start_ns=start,
            stop_ns=stop,
            log=log,
        )
        victim.start()

    ns.sim_run(sim_duration_ns)
    return topo, log


# ============================================================
# Reporting
# ============================================================

def summarize_log(tag: str, log: RunLog):
    arr = np.array(log.attacker_latencies, dtype=float)

    print(f"\n{tag}")
    print(" attacker_completed:", log.attacker_completed)
    print(" victim_completed  :", log.victim_completed)

    if len(arr) == 0:
        print(" no attacker samples")
        return

    print(" mean_latency_ns   :", float(arr.mean()))
    print(" median_latency_ns :", float(np.median(arr)))
    print(" p95_latency_ns    :", float(np.percentile(arr, 95)))
    print(" p99_latency_ns    :", float(np.percentile(arr, 99)))
    print(" cv_latency        :", float(arr.std() / (arr.mean() + 1e-12)))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    banner()

    attacker_period = 4_000
    victim_period = 6_000

    topo_off, log_off = run_once(
        seed=7,
        victim_on=False,
        attacker_period_ns=attacker_period,
        victim_period_ns=victim_period,
        mem_capacity=4,
    )

    topo_on, log_on = run_once(
        seed=7,
        victim_on=True,
        attacker_period_ns=attacker_period,
        victim_period_ns=victim_period,
        mem_capacity=4,
    )

    topo_off.describe()

    off = np.array(log_off.attacker_latencies, dtype=float)
    on = np.array(log_on.attacker_latencies, dtype=float)

    print("\nKS(OFF vs ON) =", ks_statistic(off, on))

    summarize_log("Victim OFF", log_off)
    summarize_log("Victim ON ", log_on)
