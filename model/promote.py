"""Scheduler-side RPCs for KV management: promotion and eviction priorities.

Needs the SGLang fork in ./sglang (branch kvflow-prefetch, installed
editable): `HiRadixCache.prefetch_prefix` runs a speculative host->device load
on HiCacheController's background promote thread and lowest-priority stream, so
the scheduler loop never issues or waits on the copies. Correctness stays
HiCache's own rule: the loaded nodes are locked and carry a `loading_event`
until the copy lands; a batch that admits them inherits the event as a
dependency of its producer and waits per layer (start_loading(deps)).

Present in the scheduler subprocess because `spawn` re-imports the parent
__main__, so `model.model` must import this module at module level.

Concurrency, KVFlow-style: one promotion pending or in flight (Deferred
otherwise; the caller retries). It is never refused because requests are queued.

Eviction steering (`kv_priority` RPC), KVFlow's scheme on SGLang's own
`TreeNode.priority`: four bands, lowest evicted first.
  RETIRED (-2)     a session's transient bytes once the session ended (PBKV's
                   lifecycle tier): no live session can reuse them
  TRANSIENT (-1)   bytes past a callsite's structural head once the request is
                   served: binding values, copied upstream replies, generated tokens
  0                anything unclassified (request-priority inserts 0..2 collapse here)
  PLAN_BASE + k    prefixes a planned job needs; k grows as the predicted use nears
Shared nodes take the max over sessions (KVFlow's min-steps, sign flipped).
"""
import time
from typing import Dict, List, Sequence, Tuple

from sglang.srt.managers.io_struct import RpcReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.evict_policy import EvictionStrategy

RETIRED = -2
TRANSIENT = -1
PLAN_BASE = 10


class Deferred(RuntimeError):
    """Refused for now: a promotion is already pending or in flight."""


class PlanStrategy(EvictionStrategy):
    """(band, LRU): retired first, transient next, unclassified, then planned (farthest use first)."""

    def get_priority(self, node) -> Tuple[int, float]:
        p = node.priority
        band = p if p < 0 else (0 if p < PLAN_BASE else p)
        return (band, node.last_access_time)


def hicache_promote(self: Scheduler, token_ids, rid: str = "", wait: bool = False) -> None:
    """Scheduler-side. Start loading `token_ids`' host-resident prefix onto the
    device in the background; a device-resident prefix is a no-op. `wait` blocks
    until the copy lands (profiling only)."""
    t0 = time.perf_counter()
    tc = getattr(self.tree_cache, "inner", self.tree_cache)   # SessionAwareCache wraps
    if tc.ongoing_promote:
        raise Deferred("deferred: a promotion is in flight")
    device, host, started, event = tc.prefetch_prefix(list(token_ids))
    if wait and event is not None:
        event.finish_event.synchronize()
    print(f"[promote] {rid} tokens={len(token_ids)} device={device} host={host} "
          f"started={started} ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)


def _band_tail(tc, ids: List[int], fixed_len: int, band: int) -> int:
    """Set `band` on the device nodes of `ids` lying wholly past `fixed_len`; a node
    straddling or preceding the head is left alone. Returns how many changed."""
    m = tc.match_prefix(MatchPrefixParams(key=tc._to_radix_key(ids)))
    node, end, n = m.last_device_node, len(m.device_indices), 0
    while node is not tc.root_node:
        start = end - len(node.key)
        if start < fixed_len:
            break
        node.priority = band
        n += 1
        node, end = node.parent, start
    return n


def kv_priority(self: Scheduler, demote: List[Tuple[List[int], int]],
                protect: List[Tuple[List[int], int]], rid: str = "",
                retire: Sequence[Tuple[List[int], int]] = ()) -> None:
    """Scheduler-side. `demote` / `retire`: (served prompt ids, fixed_len) pairs — nodes
    wholly past fixed_len become TRANSIENT / RETIRED. `protect`: (prefix ids, k) pairs —
    nodes on the prefix path get max(current, PLAN_BASE + k); it runs last, so a tail
    a live plan still needs is raised back. Protection from the previous call is
    undone first (restored to what it was), so the map always mirrors the plan."""
    t0 = time.perf_counter()
    tc = getattr(self.tree_cache, "inner", self.tree_cache)
    planned: Dict[int, tuple] = getattr(tc, "_planned", None)
    if planned is None:
        tc.eviction_strategy = PlanStrategy()
        planned = tc._planned = {}
    for node, prev in planned.values():
        if node.priority >= PLAN_BASE:
            node.priority = prev
    planned.clear()
    demoted = sum(_band_tail(tc, ids, fixed_len, TRANSIENT) for ids, fixed_len in demote)
    retired = sum(_band_tail(tc, ids, fixed_len, RETIRED) for ids, fixed_len in retire)
    for ids, k in protect:
        m = tc.match_prefix(MatchPrefixParams(key=tc._to_radix_key(ids)))
        node = m.last_host_node      # backed-up-but-evicted nodes count too
        while node is not tc.root_node:
            if id(node) not in planned:
                planned[id(node)] = (node, node.priority)
            node.priority = max(node.priority, PLAN_BASE + k)
            node = node.parent
    print(f"[priority] {rid} demote={len(demote)}/{demoted}n retire={len(retire)}/{retired}n "
          f"protect={len(protect)} nodes={len(planned)} ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)


_orig_rpc = Scheduler.handle_rpc_request


def handle_rpc_request(self: Scheduler, recv_req):
    """The stock handler ends every RPC with torch.distributed.barrier(), which
    syncs the scheduler's stream and costs a decode step per call. These RPCs are
    local to this scheduler, so they answer without the barrier; everything else
    takes the stock path."""
    if recv_req.method not in ("hicache_promote", "kv_priority"):
        return _orig_rpc(self, recv_req)
    try:
        getattr(self, recv_req.method)(**recv_req.parameters)
        return RpcReqOutput(True, "")
    except Deferred as e:
        return RpcReqOutput(False, str(e))
    except Exception as e:  # surfaced to the caller as a failed promotion
        return RpcReqOutput(False, f"failed: {e!r}")


Scheduler.hicache_promote = hicache_promote  # type: ignore[attr-defined]
Scheduler.kv_priority = kv_priority  # type: ignore[attr-defined]
Scheduler.handle_rpc_request = handle_rpc_request
