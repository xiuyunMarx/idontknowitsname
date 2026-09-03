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

Concurrency: up to MAX_INFLIGHT_PROMOTIONS promotions pending or in flight
(Deferred beyond that; the planner retries next tick). The fork queues them on one
low-priority stream, so more in flight means a deeper FIFO, not more PCIe
contention; each promotion's nodes are locked until its copy lands, so concurrent
ones cannot evict each other. Space: a promotion uses free device slots plus
RETIRED nodes it may reclaim, never active cache, and only if a real prefill
chunk stays free afterwards (prefetch_prefix reserve) — locked promotion slots
sit outside the scheduler's admission budget, so without the reserve concurrent
promotions can lock the whole pool and the next prefill dies with OOM.

Eviction steering (`kv_priority` RPC) on SGLang's own `TreeNode.priority`:
four bands on the device tier, lowest evicted first, LRU inside a band.
  RETIRED (-2)     a session's private bytes once the session ended (PBKV's
                   lifecycle tier): no live session can reuse them
  TRANSIENT (-1)   the one-off tail of a served prompt, past the head the flow
                   rules can rebuild: fresh binding values, generated tokens
  0                anything unclassified (request-priority inserts 0..2 collapse here)
  PLAN_BASE + s    prefixes a planned job needs; s is the planner's reuse score,
                   p(call) discounted by the predicted time to its arrival, summed
                   over every session whose plan covers the node (PBKV's
                   cross-workflow aggregation), so shared heads outrank private history
The host tier is not steered (HostStrategy): retired first, then plain LRU, so a
private prefix the device gave up stays reloadable until recency retires it.
"""
import time
from typing import Dict, List, Sequence, Tuple

from sglang.srt.managers.io_struct import RpcReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.evict_policy import EvictionStrategy
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

RETIRED = -2
TRANSIENT = -1
PLAN_BASE = 10
MAX_INFLIGHT_PROMOTIONS = 8   # concurrent sessions x a couple of predicted calls each


class Deferred(RuntimeError):
    """Refused for now: a promotion is already pending or in flight."""


class PlanStrategy(EvictionStrategy):
    """Device tier, (band, LRU): retired, transient, unclassified, then planned by rising score."""

    def get_priority(self, node) -> Tuple[int, float]:
        p = node.priority
        band = p if p < 0 else (0 if p < PLAN_BASE else p)
        return (band, node.last_access_time)


class HostStrategy(EvictionStrategy):
    """Host tier, (band, LRU): retired first, everything else by recency. The plan
    never reorders the host tier: it is the safety net every policy shares."""

    def get_priority(self, node) -> Tuple[int, float]:
        return (RETIRED if node.priority == RETIRED else 0, node.last_access_time)


def hicache_promote(self: Scheduler, token_ids, rid: str = "", wait: bool = False) -> None:
    """Scheduler-side. Start loading `token_ids`' host-resident prefix onto the
    device in the background; a device-resident prefix is a no-op. `wait` blocks
    until the copy lands (profiling only)."""
    t0 = time.perf_counter()
    tc = getattr(self.tree_cache, "inner", self.tree_cache)   # SessionAwareCache wraps
    if len(tc.ongoing_promote) >= MAX_INFLIGHT_PROMOTIONS:
        raise Deferred(f"deferred: {len(tc.ongoing_promote)} promotions in flight")
    # PBKV's conservative rule: a speculative load takes free space plus retired
    # cache only, and leaves one real prefill chunk free so the scheduler can never
    # find the pool fully locked (that was the c=2 OOM with 8 promotions in flight)
    reserve = self.chunked_prefill_size or self.max_prefill_tokens
    device, host, started, event = tc.prefetch_prefix(
        list(token_ids), reserve=reserve, evict_max_priority=RETIRED)
    free = tc.cache_controller.mem_pool_device_allocator.available_size()
    print(f"[promote] {rid} tokens={len(token_ids)} device={device} host={host} "
          f"started={started} free={free} reserve={reserve} "
          f"ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)
    if host > 0 and started == 0:
        # declined for space: keep the job queued so the planner retries once
        # retired cache or free slots appear, instead of believing it landed
        raise Deferred(f"deferred: no room for {host} host tokens (free={free}, reserve={reserve})")
    if wait and event is not None:
        event.finish_event.synchronize()


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
        tc.demand_load_partial = True   # demand loads: band-ordered eviction, partial load when quota-limited
        planned = tc._planned = {}
    for node, prev in planned.values():
        if node.priority >= PLAN_BASE:
            node.priority = prev
    planned.clear()
    demoted = sum(_band_tail(tc, ids, fixed_len, TRANSIENT) for ids, fixed_len in demote)
    retired = sum(_band_tail(tc, ids, fixed_len, RETIRED) for ids, fixed_len in retire)
    acc: Dict[int, int] = {}      # node -> summed score of every planned prefix that covers it
    for ids, k in protect:
        m = tc.match_prefix(MatchPrefixParams(key=tc._to_radix_key(ids)))
        node = m.last_host_node      # backed-up-but-evicted nodes count too
        while node is not tc.root_node:
            if id(node) not in planned:
                planned[id(node)] = (node, node.priority)
            acc[id(node)] = acc.get(id(node), 0) + k
            node = node.parent
    for nid, k in acc.items():
        node = planned[nid][0]
        node.priority = max(node.priority, PLAN_BASE + k)
    print(f"[priority] {rid} demote={len(demote)}/{demoted}n retire={len(retire)}/{retired}n "
          f"protect={len(protect)} nodes={len(planned)} ms={(time.perf_counter() - t0) * 1000:.1f}", flush=True)


_HOST = HostStrategy()
_orig_evict_host = HiRadixCache.evict_host


def _evict_host(self, num_tokens: int):
    """HiRadixCache.evict_host under HostStrategy; the device strategy is restored after."""
    saved = self.eviction_strategy
    self.eviction_strategy = _HOST
    try:
        return _orig_evict_host(self, num_tokens)
    finally:
        self.eviction_strategy = saved


HiRadixCache.evict_host = _evict_host   # type: ignore[assignment]

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
