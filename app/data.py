"""Read-only access to the static venue dataset (data/venues.json).

The JSON file is loaded once per process and cached; all functions here are
pure lookups over that cached data.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "venues.json"

Venue = dict[str, Any]


@lru_cache(maxsize=1)
def load_venues() -> dict[str, Any]:
    """Load and cache the full dataset (tournament metadata + venues)."""
    with _DATA_PATH.open(encoding="utf-8") as f:
        dataset: dict[str, Any] = json.load(f)
        return dataset


def list_venues() -> list[Venue]:
    """Return all venue records."""
    venues: list[Venue] = load_venues()["venues"]
    return venues


def get_venue(venue_id: str) -> Venue | None:
    """Return the venue with the given id, or None if unknown."""
    return next((v for v in list_venues() if v["id"] == venue_id), None)


def search_venues(query: str) -> list[Venue]:
    """Case-insensitively match query against name, commercialName, fifaName, city, and country."""
    q = query.strip().lower()
    if not q:
        return []
    fields = ("name", "commercialName", "fifaName", "city", "country")
    return [v for v in list_venues() if any(q in v[field].lower() for field in fields)]
