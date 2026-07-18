"""Shared pytest fixtures for the AccessMate test suite.

Provides a reset TestClient, a sample venue id, and a lightweight fake
google-genai client so tests can exercise the live path without a network or
a real API key. The offline path uses the real deterministic engine.
"""

import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app import assistant
from app.main import app, rate_limiter

VENUE_ID = "new-york-new-jersey"


@pytest.fixture(autouse=True)
def _reset_gemini_client():
    """Clear the assistant's cached Gemini client around every test.

    The client is now reused across requests (built once, lazily). Each test
    monkeypatches ``genai.Client`` with its own scripted fake, so without this
    reset an earlier test's cached fake would leak into later tests.
    """
    assistant._reset_client()
    yield
    assistant._reset_client()


@pytest.fixture
def venue_id() -> str:
    """Return a stable, real venue id (MetLife Stadium — hosts the final)."""
    return VENUE_ID


@pytest.fixture
def client():
    """Return a TestClient with the rate limiter reset before and after the test."""
    rate_limiter.reset()
    with TestClient(app) as test_client:
        yield test_client
    rate_limiter.reset()


# --- Fake google-genai client -------------------------------------------------

class _FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakeCandidate:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    """Mimics a google-genai GenerateContentResponse for tests."""

    def __init__(self, *, function_calls=None, text=None, model_turn=None):
        """Store the scripted function calls, text, and model turn content."""
        self.function_calls = function_calls or []
        self.text = text
        self.candidates = [_FakeCandidate(model_turn)] if model_turn else []


class FakeGeminiClient:
    """Returns a scripted sequence of FakeResponse objects."""

    def __init__(self, responses):
        """Store the scripted responses to pop one per ``generate_content`` call."""
        self._responses = list(responses)

        class _Models:
            def __init__(self, outer):
                self._outer = outer

            def generate_content(self, *, model, contents, config):
                return self._outer._responses.pop(0)

        self.models = _Models(self)


@pytest.fixture
def make_function_call():
    """Build the factory used to create a fake model function-call for scripted responses."""
    return _FakeCall


@pytest.fixture
def patch_gemini(monkeypatch):
    """Install a FakeGeminiClient built from the given scripted responses."""

    def _install(responses):
        client = FakeGeminiClient(responses)
        monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: client)
        return client

    return _install


# --- Fake streaming google-genai client ---------------------------------------
#
# The streaming path reads text and function calls off each chunk's
# candidates[0].content.parts, so a fake chunk only needs to expose real
# types.Part objects (text parts / function_call parts). Each generate_content_
# stream() call returns the next scripted list of chunks (one list per model turn).

class FakeChunk:
    """One streamed GenerateContentResponse chunk, built from real Parts."""

    def __init__(self, *, parts=None):
        """Wrap ``parts`` in a single fake candidate, mirroring a real streamed chunk."""
        content = types.Content(role="model", parts=parts) if parts is not None else None
        self.candidates = [_FakeCandidate(content)] if content is not None else []


class FakeStreamingClient:
    """Returns a scripted list of chunks per generate_content_stream call."""

    def __init__(self, turns):
        """Store the scripted per-turn chunk lists to pop one per streamed call."""
        self._turns = list(turns)  # list[list[FakeChunk]] — one inner list per turn
        self.contents_snapshots = []

        class _Models:
            def __init__(self, outer):
                self._outer = outer

            def generate_content_stream(self, *, model, contents, config):
                self._outer.contents_snapshots.append(list(contents))
                return iter(self._outer._turns.pop(0))

        self.models = _Models(self)


@pytest.fixture
def text_chunk():
    """Build the factory for a streamed chunk carrying a text delta."""
    return lambda text: FakeChunk(parts=[types.Part(text=text)])


@pytest.fixture
def call_chunk():
    """Build the factory for a streamed chunk carrying a single function call."""
    return lambda name, args: FakeChunk(
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


@pytest.fixture
def empty_chunk():
    """Build the factory for a streamed chunk with no visible text and no calls (blocked)."""
    return lambda: FakeChunk(parts=[])


@pytest.fixture
def patch_gemini_stream(monkeypatch):
    """Install a FakeStreamingClient built from scripted per-turn chunk lists."""

    def _install(turns):
        client = FakeStreamingClient(turns)
        monkeypatch.setattr("app.assistant.genai.Client", lambda *a, **k: client)
        return client

    return _install
