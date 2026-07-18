"""Tests for the FastAPI layer (app.main).

Runs entirely against the in-process app via TestClient; no network. Env keys
are cleared so /api/chat exercises the deterministic offline engine, and the
rate limiter is reset around every test.
"""

import base64

import pytest
from fastapi.testclient import TestClient

from app.assistant import AssistantReply
from app.main import RATE_LIMIT_PER_MIN, app, rate_limiter

VENUE = "new-york-new-jersey"


@pytest.fixture(autouse=True)
def _offline_env_and_reset(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("ADMIN_USERNAME", "test-admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def _admin_headers() -> dict[str, str]:
    token = base64.b64encode(b"test-admin:test-password").decode("ascii")
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def client():
    return TestClient(app)


# ---- Chat: happy path (offline) -------------------------------------------

def test_chat_happy_path_offline(client):
    resp = client.post(
        "/api/chat",
        json={
            "message": "wheelchair access?",
            "profile": {"language": "en", "needs": ["mobility"], "venue_id": VENUE},
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline"
    assert isinstance(body["reply"], str) and body["reply"].strip()
    assert body["venue_id"] == VENUE


def test_chat_without_venue_asks_to_pick_one(client):
    resp = client.post(
        "/api/chat",
        json={"message": "which gate is accessible?", "profile": {"language": "en"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline"
    assert body["reply"].strip()
    assert body["venue_id"] is None


def test_chat_live_mode_passthrough(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.assistant.answer",
        lambda *a, **k: AssistantReply(text="Live answer.", mode="live"),
    )
    resp = client.post("/api/chat", json={"message": "hi", "profile": {"venue_id": VENUE}})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "live"
    assert resp.json()["reply"] == "Live answer."


# ---- Chat: input validation (422) -----------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},                                   # empty
        {"message": "   "},                                # whitespace-only
        {"message": "x" * 2001},                           # too long
        {"message": "hi", "profile": {"needs": ["fly"]}},  # bad need enum
        {"message": "hi", "profile": {"language": "english"}},  # bad lang length
        {"message": "hi", "history": [{"role": "user", "text": "t"}] * 21},  # too many
        {"message": "hi", "history": [{"role": "bot", "text": "t"}]},        # bad role
        {"message": "hi", "junk": 1},                      # extra top-level field
    ],
)
def test_chat_rejects_bad_input_with_422(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


# ---- Venues ----------------------------------------------------------------

def test_list_venues(client):
    resp = client.get("/api/venues")
    assert resp.status_code == 200
    venues = resp.json()["venues"]
    assert len(venues) == 16
    for v in venues:
        assert {"id", "name", "city", "country", "capacity"} <= set(v)


def test_get_venue_by_id(client):
    resp = client.get(f"/api/venues/{VENUE}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "MetLife Stadium"


def test_admin_overview_has_monitoring_fields(client):
    resp = client.get(f"/api/admin/venues/{VENUE}/overview", headers=_admin_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["simulated"] is True
    assert {"summary", "seating", "gate_traffic", "resources", "alerts"} <= set(body)
    assert body["summary"]["capacity"] == 80663
    assert body["gate_traffic"]


def test_admin_overview_requires_credentials(client):
    assert client.get(f"/api/admin/venues/{VENUE}/overview").status_code == 401


def test_admin_overview_requires_deployment_configuration(client, monkeypatch):
    monkeypatch.delenv("ADMIN_USERNAME")
    assert client.get(
        f"/api/admin/venues/{VENUE}/overview",
        headers=_admin_headers(),
    ).status_code == 503


def test_admin_overview_unknown_venue_404(client):
    response = client.get(
        "/api/admin/venues/atlantis/overview",
        headers=_admin_headers(),
    )
    assert response.status_code == 404


def test_get_unknown_venue_404(client):
    resp = client.get("/api/venues/atlantis")
    assert resp.status_code == 404


# ---- Venue search ------------------------------------------------------------

def test_search_venues_by_city_case_insensitive(client):
    resp = client.get("/api/venues/search", params={"q": "MEXICO"})
    assert resp.status_code == 200
    venues = resp.json()["venues"]
    assert any(v["id"] == "mexico-city" for v in venues)
    for v in venues:
        assert {"id", "name", "city", "country", "capacity"} <= set(v)


def test_search_venues_no_match_returns_empty_list(client):
    resp = client.get("/api/venues/search", params={"q": "atlantis"})
    assert resp.status_code == 200
    assert resp.json() == {"venues": []}


def test_search_venues_validates_query(client):
    assert client.get("/api/venues/search").status_code == 422  # q required
    assert client.get("/api/venues/search", params={"q": ""}).status_code == 422
    too_long = "x" * 65
    assert client.get("/api/venues/search", params={"q": too_long}).status_code == 422


# ---- Health ----------------------------------------------------------------

def test_healthz_reports_offline_without_key(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "llm": "offline"}


def test_healthz_reports_live_with_key_but_never_echoes_it(client, monkeypatch):
    fake_key = "test-fake-key-never-real"  # not a real key format; just a sentinel
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    resp = client.get("/healthz")
    assert resp.json()["llm"] == "live"
    assert fake_key not in resp.text  # key value is never leaked in any response


# ---- Security headers ------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/api/venues", "/healthz"])
def test_security_headers_present(client, path):
    headers = client.get(path).headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_no_wildcard_cors(client):
    # No CORS middleware is installed; a normal response must not open origins.
    headers = client.get("/api/venues").headers
    assert headers.get("access-control-allow-origin") != "*"


# ---- Rate limiting ---------------------------------------------------------

def test_rate_limit_429_after_burst(client):
    payload = {"message": "hi", "profile": {"venue_id": VENUE}}
    for _ in range(RATE_LIMIT_PER_MIN):
        assert client.post("/api/chat", json=payload).status_code == 200
    # The next request over the ceiling is rejected.
    assert client.post("/api/chat", json=payload).status_code == 429


# ---- Static serving --------------------------------------------------------

def test_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "AccessMate" in resp.text


def test_static_assets_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
