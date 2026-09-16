"""CPU regressions using the actual HiRadix tree, eviction and promotion paths.

Only the device allocator and async copy are simulated; no GPU/server required.
"""
import asyncio
import time
import unittest
from functools import partial
from types import SimpleNamespace

import torch

from model.promote import (
    Deferred, PLAN_BASE, RETIRED, TRANSIENT, PlanStrategy,
    _band_tail, hicache_promote, kv_priority,
)
from server.kv_planner import Job, KVPlanner
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import (
    RadixKey, TreeNode, _key_match_page_size1, _key_match_paged, get_child_key,
)


class Allocator:
    def __init__(self):
        self.available = 0

    def available_size(self):
        return self.available

    def free(self, values):
        self.available += len(values)

    def promote(self, host, node_id):
        assert self.available >= len(host)
        self.available -= len(host)
        return torch.arange(len(host)), SimpleNamespace()


def cache(page_size=1):
    tc = HiRadixCache.__new__(HiRadixCache)
    tc.device, tc.disable, tc.is_eagle = "cpu", False, False
    tc.page_size = page_size
    tc.key_match_fn = (_key_match_page_size1 if page_size == 1 else
                       partial(_key_match_paged, page_size=page_size))
    tc.get_child_key_fn = partial(get_child_key, page_size=page_size)
    tc.root_node = TreeNode()
    tc.root_node.key = RadixKey([])
    tc.root_node.value = tc.root_node.host_value = torch.empty(0, dtype=torch.int64)
    tc.root_node.lock_ref = 1
    tc.evictable_leaves, tc.evictable_host_leaves = set(), set()
    tc.evictable_size_, tc.protected_size_ = 0, 0
    tc.ongoing_promote, tc._promote_deps = {}, []
    tc.eviction_strategy = PlanStrategy()
    alloc = Allocator()
    tc.cache_controller = SimpleNamespace(
        mem_pool_device_allocator=alloc, write_policy="write_through", promote=alloc.promote)
    tc._record_remove_event = lambda node: None
    tc.update_eviction_metrics = lambda *args: None
    return tc


def add(tc, ids, priority=0, host_only=False, locked=False):
    node = TreeNode(priority=priority)
    node.key, node.parent = RadixKey(ids), tc.root_node
    node.value = None if host_only else torch.arange(len(ids))
    node.host_value = torch.arange(len(ids)) if host_only else None
    node.lock_ref = int(locked)
    tc.root_node.children[tc.get_child_key_fn(node.key)] = node
    if not host_only:
        if locked:
            tc.protected_size_ += len(ids)
        else:
            tc.evictable_size_ += len(ids)
    tc._update_leaf_status(node)
    tc._update_host_leaf_status(node)
    return node


def scheduler(tc, reserve=4):
    return SimpleNamespace(tree_cache=tc, chunked_prefill_size=reserve, max_prefill_tokens=reserve)


class ReclamationTests(unittest.TestCase):
    def test_promotion_reclaims_retired_then_completed_transient(self):
        tc = cache()
        retired = add(tc, [10] * 4, RETIRED)
        transient = add(tc, [20] * 4, TRANSIENT)
        normal = add(tc, [30] * 4)
        protected = add(tc, [40] * 4, PLAN_BASE + 100)
        locked = add(tc, [50] * 4, TRANSIENT, locked=True)
        target = add(tc, [60] * 4, host_only=True)
        hicache_promote(scheduler(tc), [60] * 4, rid="cpu-transient")
        self.assertNotIn(retired, tc.root_node.children.values())
        self.assertNotIn(transient, tc.root_node.children.values())
        for node in (normal, protected, locked):
            self.assertIn(node, tc.root_node.children.values())
        self.assertFalse(target.evicted)
        self.assertEqual(tc.cache_controller.mem_pool_device_allocator.available_size(), 4)

    def test_promotion_does_not_evict_normal_protected_or_locked_cache(self):
        tc = cache()
        nodes = [add(tc, [10] * 4), add(tc, [20] * 4, PLAN_BASE + 100),
                 add(tc, [30] * 4, TRANSIENT, locked=True)]
        target = add(tc, [40] * 4, host_only=True)
        with self.assertRaises(Deferred):
            hicache_promote(scheduler(tc), [40] * 4)
        self.assertTrue(target.evicted)
        for node in nodes:
            self.assertIn(node, tc.root_node.children.values())

    def test_retirement_splits_static_head_from_private_history(self):
        tc = cache()
        ids = list(range(100))
        leaf = add(tc, ids)
        access = leaf.last_access_time
        planner = KVPlanner(object())
        planner.note_served("s", ids, 80, static_len=10)
        kv_priority(scheduler(tc), planner._demote, [])
        self.assertEqual(leaf.priority, TRANSIENT)
        planner.drop_session("s")
        kv_priority(scheduler(tc), [], [], retire=planner._retire)
        static = tc.root_node.children[tc.get_child_key_fn(RadixKey(ids))]
        self.assertEqual(len(static.key), 10)
        self.assertEqual(static.priority, 0)
        history = next(iter(static.children.values()))
        self.assertEqual(len(history.key), 70)
        self.assertEqual(history.priority, RETIRED)
        self.assertEqual(leaf.priority, RETIRED)
        self.assertEqual(leaf.last_access_time, access)

    def test_retirement_keeps_partial_static_page(self):
        tc = cache(page_size=4)
        leaf = add(tc, list(range(20)))
        _band_tail(tc, list(range(20)), 5, RETIRED)
        static = next(iter(tc.root_node.children.values()))
        self.assertEqual(len(static.key), 8)
        self.assertEqual(static.priority, 0)
        self.assertEqual(leaf.priority, RETIRED)
        self.assertEqual(len(leaf.key), 12)

    def test_live_session_protection_wins_over_another_sessions_retirement(self):
        tc = cache()
        ids = list(range(100))
        add(tc, ids)

        class Engine:
            async def set_kv_priority(self, demote, protect, rid, retire):
                kv_priority(scheduler(tc), demote, protect, rid, retire)

        planner = KVPlanner(Engine())
        planner.note_served("ended", ids, 80, static_len=10)
        now = time.monotonic()
        planner.submit("live", 1, [Job("j", "live", 1, "site", "hold", ids[:80],
                                       1.0, 80.0, now, now + 10, 0)])
        planner.drop_session("ended")
        asyncio.run(planner.push_priorities(now))
        static = next(iter(tc.root_node.children.values()))
        history = next(iter(static.children.values()))
        tail = next(iter(history.children.values()))
        self.assertGreaterEqual(static.priority, PLAN_BASE)
        self.assertGreaterEqual(history.priority, PLAN_BASE)
        self.assertEqual(tail.priority, RETIRED)
        planner.drop_session("live")
        asyncio.run(planner.push_priorities(now))
        self.assertEqual(static.priority, 0)
        self.assertEqual(history.priority, RETIRED)


if __name__ == "__main__":
    unittest.main()
