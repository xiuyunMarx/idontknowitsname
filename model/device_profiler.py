"""Measure the device rates the planner schedules with.

Two numbers, both specific to this box + model + engine build:
  prefill_tps_idle    uncached prefill throughput on an otherwise idle engine
  promote_tps         host-tier KV loading back onto the device (a prefill of a
                      prompt whose blocks were evicted to host)

`profile_device(engine)` runs the measurements and stores the result on
`engine.device_profile`; the engine persists it in the model's profile JSON so a
restart loads instead of re-measuring.
"""
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass
class DeviceProfile:
    prefill_tps_idle: float
    promote_tps: float

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceProfile":
        return cls(prefill_tps_idle=float(d["prefill_tps_idle"]), promote_tps=float(d["promote_tps"]))


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


async def _promote(engine, priority: int, n: int = 3072):
    """Land a target, push it off the device with junk, promote it back through the
    scheduler RPC (waiting for the copy), then request it: a device hit proves the
    promotion landed. None when it does not (no host tier, or the junk fell short)."""
    target = engine._random_ids(n)
    await _timed(engine, target, priority)
    junk = int(engine.ledger.device_cap_tokens() * 1.3)
    while junk > 0:
        step = min(8192, junk)
        await _timed(engine, engine._random_ids(step), priority)
        junk -= step
    t0 = time.perf_counter()
    ok = await engine.promote(target, "devprof-promote", wait=True)
    elapsed = time.perf_counter() - t0
    _, meta = await _timed(engine, target, priority)
    device = int((meta.get("cached_tokens_details") or {}).get("device", 0))
    if not ok or device < n // 2:
        return None
    return device / elapsed


async def profile_device(engine) -> DeviceProfile:
    from model.model import PRIORITY_REAL  # deferred: model.model imports us
    idle = await _prefill_idle(engine, PRIORITY_REAL)
    promote = await _promote(engine, PRIORITY_REAL)
    if promote is None:
        promote = idle  # no host tier to measure: repair means recompute
    prof = DeviceProfile(prefill_tps_idle=round(idle, 1), promote_tps=round(promote, 1))
    print(f"[profile-device] prefill_idle={prof.prefill_tps_idle:.0f}tps "
          f"promote={prof.promote_tps:.0f}tps", flush=True)
    engine.device_profile = prof
    return prof
