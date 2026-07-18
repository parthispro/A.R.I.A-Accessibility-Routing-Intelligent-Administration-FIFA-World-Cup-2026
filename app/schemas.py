"""Pydantic request/response models for the AccessMate API.

These caps (message length, history size, needs enum) are the API's first line
of input validation — oversized or malformed payloads are rejected by FastAPI
as 422 before any handler runs.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

#: The four accessibility needs a fan can declare (matches the UI checkboxes).
Need = Literal["mobility", "vision", "hearing", "sensory"]

#: A user-authored message: whitespace-trimmed, 1..2000 characters.
MessageStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]

#: A 2-letter language code (e.g. "en", "es", "fr", "ar").
LanguageStr = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=2, max_length=2),
]


class Profile(BaseModel):
    """The fan's declared context, sent with every chat turn."""

    model_config = {"extra": "ignore"}

    language: LanguageStr = "en"
    needs: Annotated[list[Need], Field(max_length=4)] = Field(default_factory=list)
    venue_id: str | None = Field(default=None, max_length=64)


class HistoryTurn(BaseModel):
    """One prior conversation turn, round-tripped from the client."""

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant"]
    text: Annotated[str, StringConstraints(max_length=2000)]


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``."""

    model_config = {"extra": "forbid"}

    message: MessageStr
    profile: Profile = Field(default_factory=Profile)
    history: Annotated[list[HistoryTurn], Field(max_length=20)] = Field(
        default_factory=list,
    )


class ChatResponse(BaseModel):
    """Response of ``POST /api/chat``."""

    reply: str
    mode: Literal["live", "offline"]
    venue_id: str | None = None


class VenueSummary(BaseModel):
    """Public venue fields for the list endpoint (no internal detail leaked)."""

    id: str
    name: str
    city: str
    country: str
    capacity: int


class VenueList(BaseModel):
    """Response of ``GET /api/venues``."""

    venues: list[VenueSummary]


class Health(BaseModel):
    """Response of ``GET /healthz`` — reports the LLM mode, never the key."""

    status: Literal["ok"]
    llm: Literal["live", "offline"]
