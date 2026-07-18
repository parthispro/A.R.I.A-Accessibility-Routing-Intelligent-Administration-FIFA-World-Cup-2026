"""Sanity tests for the static venue dataset and the app.data access layer."""

from app.data import get_venue, list_venues, load_venues, search_venues

ACCESSIBILITY_KEYS = {
    "gates",
    "accessible_seating",
    "sensory_room",
    "assistive_listening",
    "vision_support",
    "elevators",
    "accessible_restrooms",
    "quiet_route_hint",
    "verified",
}

SERVICES_KEYS = {"water", "first_aid", "nursing_room", "prayer_room"}


def test_sixteen_venues():
    assert len(list_venues()) == 16


def test_venue_ids_unique():
    ids = [v["id"] for v in list_venues()]
    assert len(ids) == len(set(ids))


def test_tournament_matches_reference_existing_venues():
    data = load_venues()
    ids = {v["id"] for v in data["venues"]}
    assert data["tournament"]["openingMatch"]["venueId"] in ids
    assert data["tournament"]["final"]["venueId"] in ids


def test_every_venue_has_full_accessibility_key_set():
    for venue in list_venues():
        assert set(venue["accessibility"]) == ACCESSIBILITY_KEYS, venue["id"]


def test_every_venue_has_full_services_key_set():
    for venue in list_venues():
        assert set(venue["services"]) == SERVICES_KEYS, venue["id"]


def test_every_venue_has_at_least_one_accessible_gate():
    for venue in list_venues():
        gates = venue["accessibility"]["gates"]
        assert any(gate["accessible"] for gate in gates), venue["id"]


def test_get_venue_unknown_id_returns_none():
    assert get_venue("atlantis") is None


def test_search_venues_is_case_insensitive():
    metlife = search_venues("metlife")
    assert any(v["id"] == "new-york-new-jersey" for v in metlife)

    mexico = search_venues("MEXICO")
    assert any(v["id"] == "mexico-city" for v in mexico)


def test_search_venues_blank_query_returns_nothing():
    assert search_venues("") == []
    assert search_venues("   ") == []


def test_search_venues_no_match_returns_empty_list():
    assert search_venues("atlantis") == []
