"""FastAPI application layer for AccessMate.

Serves the accessible chat UI (static/) and a small JSON API. Security posture:
- Strict security headers on every response (CSP default-src 'self', nosniff,
  no-referrer, X-Frame-Options DENY) — set by middleware.
- Per-client-IP token-bucket rate limit on the chat endpoint (429 on burst).
  In-memory by default (bounded: idle buckets are pruned); set REDIS_URL to
  share the limit across replicas behind a load balancer.
- Input caps enforced by the Pydantic models in app.schemas (422 on violation).
- Stateless: chat content is never persisted and message bodies are never
  logged; history round-trips through the client.
- No CORS middleware / no wildcard origins: the UI is same-origin with the API.
- The Gemini API key is never returned or logged; /healthz reports only the
  live/offline mode.
"""

import json
import logging
import os
import secrets
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:  # optional dependency, imported lazily in _build_rate_limiter
    import redis

from app import assistant, data, tools
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Health,
    VenueList,
    VenueSummary,
)

logger = logging.getLogger("accessmate")

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

#: Chat requests allowed per client IP per minute (free-tier-friendly ceiling).
RATE_LIMIT_PER_MIN = 20

#: Prune idle in-memory buckets once the map grows past this many client keys.
_BUCKET_PRUNE_THRESHOLD = 1024

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
        "object-src 'none'; img-src 'self' data:; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    # Only honoured over HTTPS (ignored on plain HTTP), so safe in local dev.
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
}


class TokenBucketLimiter:
    """In-memory per-key token bucket. Thread-safe; monotonic-clock based.

    Memory is bounded: once the bucket map exceeds ``prune_threshold`` keys, a
    new-key request first evicts every fully-refilled bucket. A missing key
    defaults to a full bucket, so evicting a refilled one is behaviourally a
    no-op — it just stops unique client IPs from growing the map without limit.
    """

    def __init__(
        self,
        capacity: int,
        refill_seconds: float,
        prune_threshold: int = _BUCKET_PRUNE_THRESHOLD,
    ) -> None:
        """Create a bucket that refills to ``capacity`` every ``refill_seconds``."""
        self.capacity = float(capacity)
        self._refill_rate = capacity / refill_seconds  # tokens per second
        self._prune_threshold = prune_threshold
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Drop buckets that have refilled to capacity (caller holds the lock)."""
        full = [
            key
            for key, (tokens, last) in self._buckets.items()
            if tokens + (now - last) * self._refill_rate >= self.capacity
        ]
        for key in full:
            del self._buckets[key]

    def allow(self, key: str) -> bool:
        """Consume one token for ``key``; False when the bucket is empty."""
        now = time.monotonic()
        with self._lock:
            if key not in self._buckets and len(self._buckets) >= self._prune_threshold:
                self._prune(now)
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self._refill_rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True

    def reset(self) -> None:
        """Clear all buckets (used by tests)."""
        with self._lock:
            self._buckets.clear()


class RedisTokenBucketLimiter:
    """Distributed token bucket backed by Redis; atomic via a Lua script.

    Interface-compatible with :class:`TokenBucketLimiter` (``allow``/``reset``)
    but the ceiling is shared across every app replica, so a client cannot
    multiply its quota by spreading requests across containers behind a load
    balancer. Idle keys expire on a TTL, so Redis memory stays bounded without
    an explicit prune. Uses wall-clock time (``time.time``) rather than a
    monotonic clock because replicas need a shared, comparable time source
    (assumes NTP-synced hosts; sub-second skew is negligible for a per-minute
    bucket).
    """

    #: Atomic refill-and-consume. KEYS[1]=bucket; ARGV=capacity, rate, now, ttl.
    #: Returns 1 when a token was consumed, 0 when the bucket was empty.
    _LUA = """
    local cap  = tonumber(ARGV[1])
    local rate = tonumber(ARGV[2])
    local now  = tonumber(ARGV[3])
    local ttl  = tonumber(ARGV[4])
    local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
    local tokens = tonumber(b[1])
    local ts = tonumber(b[2])
    if tokens == nil then tokens = cap; ts = now end
    local elapsed = now - ts
    if elapsed < 0 then elapsed = 0 end
    tokens = math.min(cap, tokens + elapsed * rate)
    local allowed = 0
    if tokens >= 1 then tokens = tokens - 1; allowed = 1 end
    redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
    redis.call('EXPIRE', KEYS[1], ttl)
    return allowed
    """

    def __init__(
        self,
        client: "redis.Redis",
        capacity: int,
        refill_seconds: float,
        namespace: str = "accessmate:rl:",
    ) -> None:
        """Create a bucket that refills to ``capacity`` every ``refill_seconds``."""
        self.capacity = float(capacity)
        self._refill_rate = capacity / refill_seconds
        # Idle keys refill fully after refill_seconds; let them expire a bit later.
        self._ttl_seconds = int(refill_seconds) + 10
        self._client = client
        self._namespace = namespace
        self._script = client.register_script(self._LUA)

    def allow(self, key: str) -> bool:
        """Consume one token for ``key`` atomically across replicas."""
        allowed = self._script(
            keys=[self._namespace + key],
            args=[self.capacity, self._refill_rate, time.time(), self._ttl_seconds],
        )
        return bool(int(allowed))

    def reset(self) -> None:
        """Delete this limiter's keys only (used by tests); never FLUSHes."""
        keys = list(self._client.scan_iter(match=self._namespace + "*"))
        if keys:
            self._client.delete(*keys)


def _build_rate_limiter() -> TokenBucketLimiter | RedisTokenBucketLimiter:
    """Choose the rate limiter for this process.

    Uses Redis when ``REDIS_URL`` is set and reachable so the limit holds across
    replicas; otherwise the in-memory limiter. A missing or unreachable Redis
    (or the client library not installed) logs a warning and falls back rather
    than failing startup — the zero-infra default must always run.
    """
    url = os.environ.get("REDIS_URL")
    if not url:
        return TokenBucketLimiter(RATE_LIMIT_PER_MIN, 60.0)
    try:
        import redis  # noqa: PLC0415 — optional dependency, only needed for distributed mode

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
    except Exception as exc:  # noqa: BLE001 — ImportError, connection/auth error, etc.
        logger.warning(
            "REDIS_URL is set but Redis is unavailable (%s); "
            "falling back to in-memory rate limiting.",
            exc,
        )
        return TokenBucketLimiter(RATE_LIMIT_PER_MIN, 60.0)
    logger.info("Rate limiting via Redis (shared across replicas).")
    return RedisTokenBucketLimiter(client, RATE_LIMIT_PER_MIN, 60.0)


admin_security = HTTPBasic(auto_error=False)


def require_admin(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(admin_security)],
) -> None:
    """Require configured administrator credentials for operations endpoints.

    Credentials are read on every request so deployments can manage them through
    environment variables. The application never ships a default administrator
    password and uses constant-time comparisons to avoid timing leaks.
    """
    username = os.environ.get("ADMIN_USERNAME")
    password = os.environ.get("ADMIN_PASSWORD")
    if not username or not password:
        raise HTTPException(
            status_code=503,
            detail="Administrator access is not configured on this deployment.",
        )
    valid = (
        credentials is not None
        and secrets.compare_digest(credentials.username, username)
        and secrets.compare_digest(credentials.password, password)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Administrator credentials are required.",
            headers={"WWW-Authenticate": "Basic"},
        )


rate_limiter = _build_rate_limiter()

app = FastAPI(
    title="AccessMate API",
    description="Accessibility-first stadium copilot for the FIFA World Cup 2026.",
    version="1.0.0",
)
app.state.rate_limiter = rate_limiter


@app.middleware("http")
async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach strict security headers to every response."""
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    return response


def enforce_rate_limit(request: Request) -> None:
    """Reject the request with 429 when the client's bucket is empty."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment and try again.",
        )


@app.get("/healthz")
@app.get("/api/healthz")
def healthz() -> Health:
    """Liveness probe reporting whether the live LLM is available."""
    return Health(status="ok", llm="live" if assistant.api_key_configured() else "offline")


def _venue_list(venues: list[data.Venue]) -> VenueList:
    """Project full venue records down to the public summary fields."""
    return VenueList(
        venues=[
            VenueSummary(
                id=v["id"],
                name=v["name"],
                city=v["city"],
                country=v["country"],
                capacity=v["capacity"],
            )
            for v in venues
        ],
    )


@app.get("/api/venues")
def list_venues() -> VenueList:
    """All tournament venues (public summary fields only)."""
    return _venue_list(data.list_venues())


# Declared before /api/venues/{venue_id} so "search" is not read as a venue id.
@app.get("/api/venues/search")
def search_venues(
    q: Annotated[str, Query(min_length=1, max_length=64)],
) -> VenueList:
    """Venues whose name, FIFA/commercial name, city, or country matches ``q``.

    Case-insensitive substring match; an empty result is a 200 with an empty
    list, not an error. The query length is capped by validation (422 beyond).
    """
    return _venue_list(data.search_venues(q))


@app.get("/api/venues/{venue_id}")
def get_venue(venue_id: str) -> data.Venue:
    """Full record for one venue; 404 when the id is unknown."""
    venue = data.get_venue(venue_id)
    if venue is None:
        raise HTTPException(status_code=404, detail=f"Unknown venue id {venue_id!r}.")
    return venue


@app.get("/api/admin/venues/{venue_id}/overview")
def admin_overview(
    venue_id: str,
    _rate: Annotated[None, Depends(enforce_rate_limit)],
    _admin: Annotated[None, Depends(require_admin)],
) -> dict[str, object]:
    """Return a simulated seating, traffic, and resource snapshot for operations.

    This endpoint is read-only. Deployments must put `/api/admin/` routes behind
    administrator authentication before connecting real operational data.
    """
    overview = tools.get_admin_overview(venue_id)
    if "error" in overview:
        raise HTTPException(status_code=404, detail=overview["error"])
    return overview


@app.post("/api/chat")
def chat(body: ChatRequest, _rate: Annotated[None, Depends(enforce_rate_limit)]) -> ChatResponse:
    """Answer a chat message (live Gemini when keyed, deterministic offline otherwise).

    The message body is never logged or persisted; history is supplied by the
    client each turn.
    """
    reply = assistant.answer(
        body.message,
        profile=body.profile.model_dump(),
        history=[turn.model_dump() for turn in body.history],
    )
    return ChatResponse(reply=reply.text, mode=reply.mode, venue_id=body.profile.venue_id)


@app.post("/api/chat/stream")
def chat_stream(
    body: ChatRequest, _rate: Annotated[None, Depends(enforce_rate_limit)],
) -> StreamingResponse:
    """Answer a chat message as a stream of newline-delimited JSON (NDJSON) frames.

    Frames:
      {"type": "meta",  "mode": "live"|"offline", "venue_id": <id|null>}   (first)
      {"type": "delta", "text": "..."}                                     (0+)
      {"type": "error"}                                                    (on failure)

    Same guarantees as /api/chat: strict security headers (via middleware),
    per-IP rate limit, and statelessness — the body is never logged or persisted
    and history is supplied by the client each turn.
    """
    events = assistant.answer_stream(
        body.message,
        profile=body.profile.model_dump(),
        history=[turn.model_dump() for turn in body.history],
    )
    venue_id = body.profile.venue_id

    def _frames() -> Iterator[str]:
        """Render each assistant event as one NDJSON line."""
        try:
            for kind, payload in events:
                if kind == "meta":
                    frame = {"type": "meta", "mode": payload, "venue_id": venue_id}
                else:
                    frame = {"type": "delta", "text": payload}
                yield json.dumps(frame, ensure_ascii=False) + "\n"
        except Exception:  # noqa: BLE001 — never leak a stack trace into the response stream
            yield json.dumps({"type": "error"}) + "\n"

    return StreamingResponse(_frames(), media_type="application/x-ndjson")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page accessible chat UI."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


# Static assets (style.css, app.js) referenced by index.html as /static/*.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
