import sys
import numpy as np
import netsquid as ns

print("Python", sys.executable)
print("Numpy", np.__version__)
print("Netsquied", ns.__version__)
#!/usr/bin/env python3
"""
Experiment A (Variant 1): BSM-only contention timing leakage
- Two tenants share a single BSM-like entanglement service at a switch S.
- Attacker probes A–B; victim generates background load V1–V2.
- We measure attacker completion latency distribution shift.
"""

import sys
import math
import random
from dataclasses import dataclass
from typing import Callable, Deque, Optional
from collections import deque

from netsquid.protocols import Protocol
import numpy as np
import netsquid as ns
from pydynaa import Entity, EventType



def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x)
    y = np.sort(y)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # Empirical CDF comparison at pooled sample points
    allv = np.sort(np.concatenate([x, y]))
    cx = np.searchsorted(x, allv, side="right") / len(x)
    cy = np.searchsorted(y, allv, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


def banner():
    import numpy as _np
    print("Python:", sys.executable)
    print("NumPy:", _np.__version__)
    print("NetSquid:", ns.__version__)


def sim_init(seed: int = 1):
    """
    Reset NetSquid sim state and seed both:
      - NetSquid RNG
      - Python's RNG (for our own arrival processes)
      - NumPy RNG (if used later)
    """
    ns.sim_reset()
    ns.set_random_state(seed)
    random.seed(seed)
    np.random.seed(seed)

@dataclass
class EPRRequest:
    tenant: str              # "attacker" or "victim"
    req_id: int
    t_req: int               # simulation time when request was issued
    on_done: Callable[[str, int, int, int], None]
    # callback signature: (tenant, req_id, t_req, t_done)


class EntanglementServerProtocol(Protocol):
    """
    Single-capacity FIFO server implemented as a NetSquid Protocol.
    No pydynaa Entity._wait callbacks (avoids handler typing issues).
    """

    def __init__(
        self,
        name: str,
        p_success: float = 0.2,
        attempt_time_ns: int = 2_000,
        herald_time_ns: int = 500,
        bsm_time_ns: int = 300,
    ):
        super().__init__(name=name)
        self.q: Deque[EPRRequest] = deque()

        if not (0.0 < p_success <= 1.0):
            raise ValueError("p_success must be in (0, 1].")

        self.p_success = p_success
        self.attempt_time_ns = int(attempt_time_ns)
        self.herald_time_ns = int(herald_time_ns)
        self.bsm_time_ns = int(bsm_time_ns)

        self._is_serving = False

    def submit(self, req: EPRRequest):
        self.q.append(req)

    def _sample_service_time(self) -> int:
        # attempts ~ Geometric(p_success) on {1,2,3,...}
        if self.p_success >= 1.0:
            attempts = 1
        else:
            u = random.random()
            attempts = int(math.ceil(math.log(1.0 - u) / math.log(1.0 - self.p_success)))
            attempts = max(attempts, 1)

        return attempts * self.attempt_time_ns + self.bsm_time_ns + self.herald_time_ns

    def run(self):
        while True:
            # idle until at least one request exists
            while not self.q:
                # small sleep to yield to sim (event-driven alternative is possible,
                # but this is robust and fine for our use)
                yield self.await_timer(1)

            req = self.q.popleft()
            service_time = self._sample_service_time()

            # occupy the server for service_time
            yield self.await_timer(service_time)

            t_done = ns.sim_time()
            req.on_done(req.tenant, req.req_id, req.t_req, t_done)


if __name__ == "__main__":
    banner()
    sim_init(seed=7)

    srv = EntanglementServerProtocol("S", p_success=0.2)
    srv.start()
    print("Server constructed OK. sim_time:", ns.sim_time())

@dataclass
class RunLog:
    attacker_latencies: list
    attacker_done_times: list
    attacker_req_times: list

    # optional: system metrics
    total_completed: int = 0
    attacker_completed: int = 0
    victim_completed: int = 0


class PeriodicRequesterProtocol(Protocol):
    """
    Periodic requester as a NetSquid Protocol.
    This avoids pydynaa Entity._wait signature pitfalls.
    """

    def __init__(self, name: str, server: EntanglementServerProtocol, tenant: str,
                 period_ns: int, start_ns: int, stop_ns: int, log: RunLog):
        super().__init__(name=name)
        self.server = server
        self.tenant = tenant
        self.period_ns = int(period_ns)
        self.start_ns = int(start_ns)
        self.stop_ns = int(stop_ns)
        self.log = log
        self._next_id = 0

    def run(self):
        # wait until start time
        now = ns.sim_time()
        if self.start_ns > now:
            yield self.await_timer(self.start_ns - now)

        # issue requests periodically until stop time
        while ns.sim_time() < self.stop_ns:
            self._submit_once()
            yield self.await_timer(self.period_ns)

    def _submit_once(self):
        req_id = self._next_id
        self._next_id += 1
        t_req = ns.sim_time()

        def on_done(tenant, req_id, t_req, t_done):
            self.log.total_completed += 1
            if tenant == "attacker":
                self.log.attacker_completed += 1
                self.log.attacker_req_times.append(t_req)
                self.log.attacker_done_times.append(t_done)
                self.log.attacker_latencies.append(t_done - t_req)
            else:
                self.log.victim_completed += 1

        self.server.submit(EPRRequest(
            tenant=self.tenant,
            req_id=req_id,
            t_req=t_req,
            on_done=on_done
        ))


def run_once(
    seed: int,
    victim_on: bool,
    victim_period_ns: int,
    attacker_period_ns: int,
    sim_duration_ns: int = 200_000,
    warmup_ns: int = 20_000,
    server_p_success: float = 0.2,
):
    """
    Run a single simulation:
      - attacker always probes from t=warmup to end
      - victim either OFF or ON in same interval
    """
    sim_init(seed)

    log = RunLog(attacker_latencies=[], attacker_done_times=[], attacker_req_times=[])

    srv = EntanglementServerProtocol("S", p_success=server_p_success)
    srv.start()

    start = warmup_ns
    stop = sim_duration_ns

    # attacker probes
    attacker = PeriodicRequesterProtocol("attacker_prober", srv, "attacker",
                      period_ns=attacker_period_ns,
                      start_ns=start, stop_ns=stop, log=log)
    attacker.start()

    # victim load
    if victim_on:
        victim = PeriodicRequesterProtocol("victim_load", srv, "victim",
                          period_ns=victim_period_ns,
                          start_ns=start, stop_ns=stop, log=log)
        victim.start()

    ns.sim_run(sim_duration_ns)

    # return only steady-state samples (exclude first few completions if you want later)
    return log
def summarize_log(tag: str, log: RunLog):
    arr = np.array(log.attacker_latencies, dtype=float)
    print(f"\n{tag}")
    print(" attacker_completed:", log.attacker_completed)
    print(" victim_completed   :", log.victim_completed)
    if len(arr) == 0:
        print(" (no attacker samples)")
        return

    print(" mean_latency_ns    :", arr.mean())
    print(" median_latency_ns  :", np.median(arr))
    print(" p95_latency_ns     :", np.percentile(arr, 95))
    print(" p99_latency_ns     :", np.percentile(arr, 99))
    print(" cv_latency         :", arr.std() / (arr.mean() + 1e-12))  # variability normalized


if __name__ == "__main__":
    banner()

    # Choose rates so victim can actually cause queueing:
    # smaller period => higher load
    attacker_period = 4000   # ns
    victim_period   = 6000   # ns

    log_off = run_once(seed=7, victim_on=False,
                       victim_period_ns=victim_period,
                       attacker_period_ns=attacker_period)

    log_on  = run_once(seed=7, victim_on=True,
                       victim_period_ns=victim_period,
                       attacker_period_ns=attacker_period)

    def summarize(name, log):
        arr = np.array(log.attacker_latencies, dtype=float)
        if len(arr) == 0:
            print(name, "no attacker samples")
            return
        print(f"\n{name}")
        print(" attacker_completed:", log.attacker_completed)
        print(" victim_completed   :", log.victim_completed)
        print(" mean_latency_ns    :", arr.mean())
        print(" p95_latency_ns     :", np.percentile(arr, 95))
        print(" p99_latency_ns     :", np.percentile(arr, 99))

    off = np.array(log_off.attacker_latencies, dtype=float)
    on  = np.array(log_on.attacker_latencies, dtype=float)
    print("\nKS(OFF vs ON) =", ks_statistic(off, on))


    summarize("Victim OFF", log_off)
    summarize("Victim ON ", log_on)

