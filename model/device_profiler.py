"""Measure the device rates the planner schedules with.

Three numbers, all specific to this box + model + engine build:
  prefill_tps_idle    uncached prefill throughput on an otherwise idle engine
  prefill_tps_loaded  the same, while decode streams are running — the rate a
                      call's own prefill gets under load (planner release timing)
  promote_tps         host-tier KV loading back onto the device (a prefill of a
                      prompt whose blocks were evicted to host)

`profile_device(engine)` runs the measurements and stores the result on
`engine.device_profile`; the engine persists it in the model's profile JSON so a restart loads instead of re-measuring.
"""
import asyncio
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass
class DeviceProfile:
    prefill_tps_idle: float
    prefill_tps_loaded: float
    promote_tps: float
    loaded_decoders: int   # decode streams the loaded rate was measured against

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceProfile":
        return cls(prefill_tps_idle=float(d["prefill_tps_idle"]),
                   prefill_tps_loaded=float(d["prefill_tps_loaded"]),
                   promote_tps=float(d["promote_tps"]),
                   loaded_decoders=int(d["loaded_decoders"]))


async def _timed(engine, ids, priority: int):
    """One prefill-dominated request (a single generated token): elapsed s + meta."""
    meta = {}
    t0 = time.perf_counter()
    async for out in await engine.engine.async_generate(
            input_ids=ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
            rid=f"devprof-{uuid.uuid4().hex[:8]}", priority=priority, stream=True):
        meta = out.get("meta_info") or meta
    return time.perf_counter() - t0, meta


async def _prefill_idle(engine, priority: int, n: int = 6144, reps: int = 2) -> float:
    best = 0.0
    for _ in range(reps):
        elapsed, _ = await _timed(engine, engine._random_ids(n), priority)
        best = max(best, n / elapsed)
    return best


async def _prefill_loaded(engine, priority: int, n: int = 4096, decoders: int = 12) -> float:
    """Uncached prefill throughput while `decoders` decode streams run."""
    started = 0
    all_started = asyncio.Event()

    async def decode_worker(i: int) -> None:
        nonlocal started
        first = True
        async for _ in await engine.engine.async_generate(
                input_ids=engine._random_ids(24),
                sampling_params={"max_new_tokens": 512, "temperature": 0.0, "ignore_eos": True},
                rid=f"devprof-dec-{i}-{uuid.uuid4().hex[:8]}", priority=priority, stream=True):
            if first:
                first = False
                started += 1
                if started == decoders:
                    all_started.set()

    workers = [asyncio.create_task(decode_worker(i)) for i in range(decoders)]
    await all_started.wait()
    elapsed, _ = await _timed(engine, engine._random_ids(n), priority)
    await asyncio.gather(*workers)
    return n / elapsed


async def _promote(engine, priority: int, n: int = 16384):
    """Land a target, push it off the device with junk, retire the junk, promote the
    target back through the scheduler RPC (waiting for the copy), then request it: a
    device hit proves the promotion landed. None when it does not (no host tier, or
    the junk fell short)."""
    target = engine._random_ids(n)
    await _timed(engine, target, priority)
    junk = int(engine.ledger.device_cap_tokens() * 1.3)
    junk_prompts = []
    while junk > 0:
        step = min(8192, junk)
        ids = engine._random_ids(step)
        await _timed(engine, ids, priority)
        junk_prompts.append(ids)
        junk -= step
    # The junk fills the device tier in the unclassified band, which a promotion may not
    # reclaim, so it would be deferred for lack of room. Retire the junk (the band of an
    # ended session's leftovers) so the load-back evicts it, as in operation.
    await engine.set_kv_priority([], [], "devprof-retire", retire=[(ids, 0) for ids in junk_prompts])
    # The RPC returns before the background thread issues the copy, so its event cannot be
    # waited on here: time the RPC plus a request of the target, which waits on the loading
    # event and then runs one cached step, and take that step's own cost off with a second,
    # fully device-resident request.
    t0 = time.perf_counter()
    ok = await engine.promote(target, "devprof-promote", wait=True)
    _, meta = await _timed(engine, target, priority)
    elapsed = time.perf_counter() - t0
    device = int((meta.get("cached_tokens_details") or {}).get("device", 0))
    if not ok or device < n // 2:
        return None
    cached_step, _ = await _timed(engine, target, priority)
    return device / max(elapsed - cached_step, 1e-3)


async def profile_device(engine, loaded_decoders: int = 12) -> DeviceProfile:
    from model.model import PRIORITY_REAL  # deferred: model.model imports us
    idle = await _prefill_idle(engine, PRIORITY_REAL)
    loaded = await _prefill_loaded(engine, PRIORITY_REAL, decoders=loaded_decoders)
    promote = await _promote(engine, PRIORITY_REAL)
    if promote is None:
        promote = loaded  # no host tier to measure: repair means recompute
    prof = DeviceProfile(prefill_tps_idle=round(idle, 1),
                         prefill_tps_loaded=round(loaded, 1),
                         promote_tps=round(promote, 1),
                         loaded_decoders=loaded_decoders)
    print(f"[profile-device] prefill_idle={prof.prefill_tps_idle:.0f}tps "
          f"prefill_loaded={prof.prefill_tps_loaded:.0f}tps "
          f"promote={prof.promote_tps:.0f}tps", flush=True)
    engine.device_profile = prof
    return prof
