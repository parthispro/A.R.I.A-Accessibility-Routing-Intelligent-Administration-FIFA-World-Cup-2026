"""Unit tests for the rate limiters in app.main.

Covers three concerns the API layer depends on:
- Memory: the in-memory limiter prunes fully-refilled buckets so unique client
  IPs cannot grow the map without bound.
- Thread-safety: concurrent ``allow`` calls on one key hand out exactly
  ``capacity`` tokens (no lost updates, no deadlock).
- Distribution: the Redis-backed limiter and the startup selection
  (``_build_rate_limiter``) are wired correctly, with a graceful in-memory
  fallback when Redis is absent or unreachable.

The Redis tests use an in-process fake client, so they exercise the wrapper
(argument passing, namespacing, bool decoding, scoped reset) — not a live
Redis server or the Lua script itself.
"""

import sys
import threading
import time
import types

from app.main import (
    RATE_LIMIT_PER_MIN,
    RedisTokenBucketLimiter,
    TokenBucketLimiter,
    _build_rate_limiter,
)

# --------------------------------------------------------------- pruning

def test_prune_evicts_full_buckets_and_keeps_active_ones():
    limiter = TokenBucketLimiter(RATE_LIMIT_PER_MIN, 60.0, prune_threshold=10)
    # Reach past the threshold with a mix of full (idle) and drained (active).
    # Timestamps must be on the monotonic clock allow() reads, so the drained
    # buckets have effectively zero elapsed refill when the prune runs.
    with limiter._lock:
        now = time.monotonic()
        for i in range(8):
            limiter._buckets[f"full-{i}"] = (limiter.capacity, now)
        for i in range(6):
            limiter._buckets[f"empty-{i}"] = (0.0, now)

    # A brand-new key (not already present) triggers the prune.
    assert limiter.allow("brand-new-ip") is True

    remaining = set(limiter._buckets)
    assert not any(k.startswith("full-") for k in remaining), "full buckets evicted"
    assert sum(k.startswith("empty-") for k in remaining) == 6, "active buckets kept"


def test_prune_is_behaviourally_a_no_op_for_evicted_ips():
    # An evicted (fully-refilled) IP is indistinguishable from a never-seen one:
    # it starts fresh with a full bucket and is allowed.
    limiter = TokenBucketLimiter(RATE_LIMIT_PER_MIN, 60.0, prune_threshold=5)
    with limiter._lock:
        for i in range(10):
            limiter._buckets[f"full-{i}"] = (limiter.capacity, time.monotonic())
    limiter.allow("trigger")  # prunes the ten full buckets
    assert "full-0" not in limiter._buckets
    assert limiter.allow("full-0") is True  # served as a fresh full bucket


def test_no_prune_below_threshold():
    limiter = TokenBucketLimiter(RATE_LIMIT_PER_MIN, 60.0, prune_threshold=1000)
    for i in range(50):
        limiter.allow(f"ip-{i}")
    assert len(limiter._buckets) == 50  # nothing pruned under the threshold


# ----------------------------------------------------------- concurrency

def test_concurrent_allow_on_one_key_hands_out_exactly_capacity():
    # A long refill window makes refill negligible over the test's wall time,
    # so N racing threads should together consume exactly `capacity` tokens.
    capacity = 50
    limiter = TokenBucketLimiter(capacity, refill_seconds=10_000)
    attempts = 200
    barrier = threading.Barrier(attempts)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()  # release all threads together to maximize contention
        ok = limiter.allow("same-ip")
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(attempts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads), "no thread deadlocked"
    assert len(results) == attempts
    assert sum(results) == capacity, "exactly capacity tokens granted, no lost updates"


# --------------------------------------------------------- redis wrapper

class _FakeRedis:
    """Minimal stand-in for a redis-py client used by RedisTokenBucketLimiter.

    ``register_script`` returns a callable that faithfully simulates the token
    bucket in Python so the wrapper's behaviour (allow/deny, reset scope) can be
    asserted without a live Redis or a Lua interpreter.
    """

    def __init__(self):
        self.store: dict[str, dict[str, float]] = {}

    def register_script(self, _lua):
        def run(keys, args):
            cap, rate, now, _ttl = (float(a) for a in args)
            key = keys[0]
            bucket = self.store.get(key)
            tokens = bucket["tokens"] if bucket else cap
            ts = bucket["ts"] if bucket else now
            tokens = min(cap, tokens + max(0.0, now - ts) * rate)
            allowed = 0
            if tokens >= 1:
                tokens -= 1
                allowed = 1
            self.store[key] = {"tokens": tokens, "ts": now}
            return allowed

        return run

    def scan_iter(self, match=None):
        prefix = match.rstrip("*") if match else ""
        return [k for k in list(self.store) if k.startswith(prefix)]

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)


class _PingableFake(_FakeRedis):
    """_FakeRedis that also answers ping(), as _build_rate_limiter expects."""

    def ping(self):
        return True


def test_redis_limiter_allows_up_to_capacity_then_blocks():
    fake = _FakeRedis()
    limiter = RedisTokenBucketLimiter(fake, RATE_LIMIT_PER_MIN, 60.0)
    allowed = sum(limiter.allow("1.2.3.4") for _ in range(RATE_LIMIT_PER_MIN))
    assert allowed == RATE_LIMIT_PER_MIN
    assert limiter.allow("1.2.3.4") is False  # bucket now empty
    # Different client gets its own bucket.
    assert limiter.allow("5.6.7.8") is True


def test_redis_limiter_namespaces_keys_and_reset_is_scoped():
    fake = _FakeRedis()
    fake.store["unrelated:key"] = {"tokens": 1.0, "ts": 0.0}  # not ours
    limiter = RedisTokenBucketLimiter(fake, RATE_LIMIT_PER_MIN, 60.0)
    limiter.allow("9.9.9.9")
    assert any(k.startswith("accessmate:rl:") for k in fake.store)
    limiter.reset()
    assert "unrelated:key" in fake.store, "reset must not touch foreign keys"
    assert not any(k.startswith("accessmate:rl:") for k in fake.store)


def test_redis_reset_with_no_keys_never_calls_delete():
    # Redis DELETE with zero keys is an error, so reset() must skip the call
    # entirely when nothing matches the namespace.
    fake = _FakeRedis()
    calls = []
    fake.delete = lambda *keys: calls.append(keys)
    limiter = RedisTokenBucketLimiter(fake, RATE_LIMIT_PER_MIN, 60.0)
    limiter.reset()
    assert calls == []


# ---------------------------------------------------- startup selection

def test_build_rate_limiter_defaults_to_in_memory(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(_build_rate_limiter(), TokenBucketLimiter)


def test_build_rate_limiter_falls_back_when_redis_unreachable(monkeypatch):
    # A bogus URL (or a missing redis package) must degrade to in-memory, not crash.
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6390/0")
    assert isinstance(_build_rate_limiter(), TokenBucketLimiter)


def test_build_rate_limiter_uses_redis_when_available(monkeypatch):
    fake_module = types.SimpleNamespace(
        Redis=types.SimpleNamespace(from_url=lambda *a, **k: _PingableFake()),
    )
    monkeypatch.setitem(sys.modules, "redis", fake_module)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    limiter = _build_rate_limiter()
    assert isinstance(limiter, RedisTokenBucketLimiter)
    assert limiter.allow("1.1.1.1") is True
