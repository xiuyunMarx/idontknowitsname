"""Host→device KV promotion without a request for sglang """
import time

import torch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams, MatchPrefixParams


def hicache_promote(self: Scheduler, token_ids, rid: str = "", wait: bool = False) -> None:
    """Scheduler-side. Load `token_ids`' host-resident prefix onto the device; a
    device-resident prefix is only MRU-touched. `wait` blocks until the copy lands
    (profiling only: it stalls the scheduler for the PCIe time)."""
    t0 = time.perf_counter()
    tc = getattr(self.tree_cache, "inner", self.tree_cache)   # SessionAwareCache wraps
    m = tc.match_prefix(MatchPrefixParams(key=tc._to_radix_key(token_ids)))
    loaded, started = 0, False
    if m.host_hit_length > 0:
        vals, _ = tc.init_load_back(InitLoadBackParams(
            last_host_node=m.last_host_node, host_hit_length=m.host_hit_length))
        loaded = len(vals)
        ldc = tc.cache_controller.layer_done_counter
        if loaded and ldc.events[(ldc.producer_index + 1) % ldc.num_counters].finish_event.query():
            idx = tc.ready_to_load_host_cache()
            started = idx >= 0
        if started:
            # No batch consumes this load through hicache_consumer_index, so order
            # every later kernel behind it: a request matching these blocks before
            # the copy finishes would otherwise read stale slots.
            fin = ldc.events[idx].finish_event
            fwd = getattr(self.tp_worker.model_runner, "forward_stream", None)
            if fwd is not None:
                fwd.wait_event(fin)
            torch.cuda.current_stream().wait_event(fin)
            if wait:
                fin.synchronize()
    print(f"[promote] {rid} tokens={len(token_ids)} device={len(m.device_indices)} "
          f"host={m.host_hit_length} loaded={loaded} started={int(started)} "
          f"ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)


Scheduler.hicache_promote = hicache_promote  # type: ignore[attr-defined]
