"""Tests for the streaming chat path (assistant.answer_stream + /api/chat/stream).

The google-genai client is mocked with a fake streaming client (from conftest);
no network or key is touched. Covers the streamed function-calling loop, the
blocked-response decline, offline fallback, and the NDJSON endpoint contract.
"""

import json

import pytest
from google.genai import types

from app import assistant
from tests.conftest import FakeChunk

VENUE = "new-york-new-jersey"


def _with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


# --- assistant.answer_stream --------------------------------------------------

def test_offline_stream_without_key(monkeypatch):
    _no_key(monkeypatch)
    events = list(
        assistant.answer_stream("wheelchair access?", {"venue_id": VENUE, "language": "en"}),
    )
    assert events[0] == ("meta", "offline")
    assert events[1][0] == "delta" and events[1][1]  # a real offline answer
    assert len(events) == 2


def test_live_direct_text_streams_incrementally(monkeypatch, patch_gemini_stream, text_chunk):
    _with_key(monkeypatch)
    patch_gemini_stream([[text_chunk("The opening "), text_chunk("match is 2026-06-11.")]])

    events = list(assistant.answer_stream("when is kickoff?", {"language": "en"}))

    assert events[0] == ("meta", "live")
    deltas = [payload for kind, payload in events if kind == "delta"]
    assert len(deltas) == 2  # streamed in pieces, not one blob
    assert "".join(deltas) == "The opening match is 2026-06-11."


def test_live_tool_round_then_streamed_answer(
    monkeypatch, patch_gemini_stream, text_chunk, call_chunk,
):
    _with_key(monkeypatch)
    client = patch_gemini_stream(
        [
            [call_chunk("get_venue_info", {"venue_id": VENUE})],  # turn 1: tool call
            [text_chunk("MetLife Stadium "), text_chunk("hosts the final.")],  # turn 2: answer
        ],
    )

    events = list(
        assistant.answer_stream("tell me about MetLife", {"venue_id": VENUE, "language": "en"}),
    )

    assert events[0] == ("meta", "live")
    assert "".join(p for k, p in events if k == "delta") == "MetLife Stadium hosts the final."

    # The second request's contents mirror the non-streaming loop:
    # [user_turn, model_turn(rebuilt from streamed parts), function-responses].
    second = client.contents_snapshots[1]
    assert len(second) == 3
    assert second[1].role == "model"  # verbatim tool-call turn preserved
    assert second[2].role == "user"
    assert len(second[2].parts) == 1  # one function response for the one call


def test_streamed_thought_parts_are_excluded_from_visible_text(
    monkeypatch, patch_gemini_stream,
):
    _with_key(monkeypatch)
    chunk = FakeChunk(
        parts=[
            types.Part(text="internal reasoning", thought=True),
            types.Part(text="Gate A is quietest."),
        ],
    )
    patch_gemini_stream([[chunk]])

    events = list(assistant.answer_stream("quietest gate?", {"language": "en"}))

    deltas = "".join(payload for kind, payload in events if kind == "delta")
    assert deltas == "Gate A is quietest."  # thought text never reaches the client


def test_live_blocked_no_text_yields_decline(monkeypatch, patch_gemini_stream, empty_chunk):
    _with_key(monkeypatch)
    patch_gemini_stream([[empty_chunk()]])  # no text, no calls (blocked / SAFETY)

    events = list(assistant.answer_stream("something blocked", {"language": "es"}))

    assert events[0] == ("meta", "live")
    assert events[-1] == ("delta", assistant._DECLINE["es"])  # localized decline


def test_auth_error_before_any_text_falls_back_offline(monkeypatch):
    _with_key(monkeypatch)
    from google.genai import errors

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                raise errors.ClientError(401, {"error": {"message": "bad key"}})

    monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: _RaisingClient())

    events = list(
        assistant.answer_stream("hi", {"venue_id": VENUE, "language": "en"}),
    )
    assert events[0] == ("meta", "offline")
    assert events[1][0] == "delta" and events[1][1]


def test_model_not_found_404_before_any_text_falls_back_offline(monkeypatch):
    _with_key(monkeypatch)
    from google.genai import errors

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                raise errors.ClientError(404, {"error": {"message": "model not found"}})

    monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: _RaisingClient())

    events = list(assistant.answer_stream("hi", {"language": "en"}))
    assert events[0] == ("meta", "offline")
    assert events[1][0] == "delta" and events[1][1]


def test_server_error_before_any_text_falls_back_offline(monkeypatch):
    _with_key(monkeypatch)
    from google.genai import errors

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                raise errors.ServerError(503, {"error": {"message": "unavailable"}})

    monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: _RaisingClient())

    events = list(assistant.answer_stream("hi", {"language": "en"}))
    assert events[0] == ("meta", "offline")


def test_streamed_tool_loop_stops_at_iteration_cap(
    monkeypatch, patch_gemini_stream, call_chunk,
):
    _with_key(monkeypatch)
    # Every scripted turn requests another tool call; the streamed loop must
    # stop at the cap and yield the decline (no visible text was produced).
    turns = [
        [call_chunk("get_venue_info", {"venue_id": VENUE})]
        for _ in range(assistant._MAX_TOOL_ITERATIONS)
    ]
    client = patch_gemini_stream(turns)

    events = list(assistant.answer_stream("loop forever", {"language": "en"}))

    assert len(client.contents_snapshots) == assistant._MAX_TOOL_ITERATIONS
    assert events[0] == ("meta", "live")
    assert events[-1] == ("delta", assistant._DECLINE["en"])


def test_stream_400_client_error_is_reraised(monkeypatch):
    _with_key(monkeypatch)
    from google.genai import errors

    class _RaisingClient:
        class models:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                raise errors.ClientError(400, {"error": {"message": "our bug"}})

    monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: _RaisingClient())

    # A 400 is our own malformed request — surface it, never mask it as offline.
    with pytest.raises(errors.ClientError):
        list(assistant.answer_stream("hi", {"language": "en"}))


def test_midstream_failure_after_first_delta_ends_stream_gracefully(
    monkeypatch, text_chunk,
):
    _with_key(monkeypatch)

    def _chunks():
        yield text_chunk("Partial ")
        raise ConnectionError("connection dropped mid-stream")

    class _Client:
        class models:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                return _chunks()

    monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: _Client())

    events = list(assistant.answer_stream("hi", {"language": "en"}))

    # The partial answer was already sent; the stream ends without raising.
    assert events == [("meta", "live"), ("delta", "Partial ")]


def test_empty_live_stream_yields_live_decline(monkeypatch):
    _with_key(monkeypatch)
    # Defensive branch: the live event generator ends without yielding anything.
    monkeypatch.setattr(
        "app.assistant._live_stream_events", lambda *a, **k: iter(()),
    )

    events = list(assistant.answer_stream("hi", {"language": "en"}))

    assert events == [("meta", "live"), ("delta", assistant._DECLINE["en"])]


# --- /api/chat/stream endpoint ------------------------------------------------

def test_endpoint_streams_ndjson_offline(client, monkeypatch):
    _no_key(monkeypatch)
    res = client.post(
        "/api/chat/stream",
        json={
            "message": "wheelchair access?",
            "profile": {"venue_id": VENUE, "language": "en", "needs": []},
            "history": [],
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-ndjson")
    # Security headers still applied to a streaming response by the middleware.
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in res.headers["Content-Security-Policy"]

    frames = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert frames[0] == {"type": "meta", "mode": "offline", "venue_id": VENUE}
    assert any(f["type"] == "delta" and f["text"] for f in frames)


def test_endpoint_streams_live_deltas(client, monkeypatch, patch_gemini_stream, text_chunk):
    _with_key(monkeypatch)
    patch_gemini_stream([[text_chunk("Gate A "), text_chunk("is quietest.")]])

    res = client.post(
        "/api/chat/stream",
        json={
            "message": "quietest gate?",
            "profile": {"venue_id": VENUE, "language": "en", "needs": []},
            "history": [],
        },
    )
    assert res.status_code == 200
    frames = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert frames[0]["type"] == "meta" and frames[0]["mode"] == "live"
    text = "".join(f["text"] for f in frames if f["type"] == "delta")
    assert text == "Gate A is quietest."


def test_endpoint_emits_error_frame_and_no_traceback_when_stream_raises(
    client, monkeypatch,
):
    _no_key(monkeypatch)

    def _broken_stream(*a, **k):
        yield ("meta", "offline")
        raise RuntimeError("boom-sentinel")

    monkeypatch.setattr("app.main.assistant.answer_stream", _broken_stream)

    res = client.post("/api/chat/stream", json={"message": "hi"})
    assert res.status_code == 200
    frames = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert frames[-1] == {"type": "error"}
    assert "boom-sentinel" not in res.text  # stack traces never leak downstream


def test_endpoint_rate_limited_returns_429(client, monkeypatch):
    _no_key(monkeypatch)
    payload = {
        "message": "hi",
        "profile": {"venue_id": VENUE, "language": "en", "needs": []},
        "history": [],
    }
    # The streaming endpoint shares the same per-IP limiter; exhaust the bucket
    # (20/min) and the next request must be rejected with 429.
    statuses = {client.post("/api/chat/stream", json=payload).status_code for _ in range(50)}
    assert 429 in statuses
