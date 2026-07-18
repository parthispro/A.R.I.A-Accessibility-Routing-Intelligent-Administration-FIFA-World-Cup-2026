"""Full-stack integration tests: HTTP API -> assistant -> tools.

Exercises the whole request path in BOTH modes. The live path uses the fake
google-genai client from conftest (no network, no key). Offline coverage lives
in test_api.py; here we prove the live wiring end to end.
"""

from app.schemas import ChatRequest
from tests.conftest import FakeResponse


def test_live_roundtrip_through_api(
    client, patch_gemini, make_function_call, venue_id, monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Script: model calls a tool, then produces a grounded final answer.
    patch_gemini(
        [
            FakeResponse(
                function_calls=[
                    make_function_call(
                        "find_accessible_services",
                        {"venue_id": venue_id, "need": "mobility"},
                    ),
                ],
                model_turn={"role": "model"},  # appended verbatim; opaque to us
            ),
            FakeResponse(text="Wheelchair access is available on all levels."),
        ],
    )

    resp = client.post(
        "/api/chat",
        json={
            "message": "wheelchair access at MetLife?",
            "profile": {"language": "en", "needs": ["mobility"], "venue_id": venue_id},
            "history": [],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "live"
    assert body["reply"] == "Wheelchair access is available on all levels."
    assert body["venue_id"] == venue_id


def test_offline_roundtrip_through_api(client, venue_id, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    resp = client.post(
        "/api/chat",
        json={"message": "where is the nursing room?", "profile": {"venue_id": venue_id}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline"
    assert body["reply"].strip()


def test_chat_request_schema_defaults():
    """Schema is usable with only a message (profile/history default sensibly)."""
    req = ChatRequest(message="hi")
    assert req.profile.language == "en"
    assert req.profile.needs == []
    assert req.profile.venue_id is None
    assert req.history == []
