"""Tests for app.tools: pure tool functions and the execute_tool dispatcher."""

import json

import pytest

from app import data, tools
from app.tools import (
    CONGESTION_LEVELS,
    execute_tool,
    find_accessible_services,
    get_admin_overview,
    get_live_status,
    get_venue_info,
    plan_visit,
)

VALID = "new-york-new-jersey"
UNKNOWN = "atlantis"


# ---------------------------------------------------------------- all tools

TOOL_CALLS = {
    "get_venue_info": lambda vid: get_venue_info(vid),
    "find_accessible_services": lambda vid: find_accessible_services(vid),
    "get_live_status": lambda vid: get_live_status(vid, hour=13),
    "plan_visit": lambda vid: plan_visit(vid, ["mobility"], hour=13),
}


@pytest.mark.parametrize("name", sorted(TOOL_CALLS))
def test_every_tool_works_for_valid_venue(name):
    result = TOOL_CALLS[name](VALID)
    assert isinstance(result, dict)
    assert "error" not in result
    json.dumps(result)  # JSON-serializable


@pytest.mark.parametrize("name", sorted(TOOL_CALLS))
def test_every_tool_returns_error_payload_for_unknown_venue(name):
    result = TOOL_CALLS[name](UNKNOWN)
    assert "error" in result
    assert UNKNOWN in result["error"]


# ------------------------------------------------------------ get_venue_info

def test_get_venue_info_basic_fields():
    info = get_venue_info("dallas")
    assert info["name"] == "AT&T Stadium"
    assert info["city"].startswith("Arlington")
    assert info["country"] == "USA"
    assert info["capacity_note"] == "approximate tournament capacity"
    assert [g["name"] for g in info["gates"]] == ["Entry A", "Entry E", "Entry K"]


def test_get_venue_info_matchday_hosting():
    opening = get_venue_info("mexico-city")["matchday"]
    assert opening["hosts_opening_match"] == "2026-06-11"
    assert "hosts_final" not in opening

    final = get_venue_info(VALID)["matchday"]
    assert final["hosts_final"] == "2026-07-19"
    assert "hosts_opening_match" not in final

    neither = get_venue_info("dallas")["matchday"]
    assert "hosts_opening_match" not in neither
    assert "hosts_final" not in neither
    assert len(neither["accessibility_ticket_types"]) == 3


def test_get_venue_info_does_not_expose_shared_cached_dicts():
    info = get_venue_info("dallas")
    info["gates"][0]["name"] = "MUTATED"
    assert data.get_venue("dallas")["accessibility"]["gates"][0]["name"] == "Entry A"


# -------------------------------------------------- find_accessible_services

NEED_FIELDS = {
    "mobility": {"gates", "elevators", "accessible_seating", "accessible_restrooms"},
    "vision": {"vision_support"},
    "hearing": {"assistive_listening"},
    "sensory": {"sensory_room", "quiet_route_hint"},
}


@pytest.mark.parametrize("need,fields", sorted(NEED_FIELDS.items()))
def test_find_accessible_services_filters_by_need(need, fields):
    result = find_accessible_services(VALID, need=need)
    assert result["need"] == need
    assert set(result["services"]) == fields
    assert "verified" in result


def test_find_accessible_services_general_includes_everything():
    result = find_accessible_services(VALID, need="general")
    assert set(result["services"]) >= {"gates", "sensory_room", "vision_support"}


def test_find_accessible_services_invalid_need_falls_back_to_general():
    result = find_accessible_services(VALID, need="teleportation")
    assert result["need"] == "general"
    assert "note" in result
    assert "error" not in result


def test_find_accessible_services_carries_verified_flag():
    assert find_accessible_services(VALID)["verified"] is True
    assert find_accessible_services("dallas")["verified"] is False


# ------------------------------------------------------------ get_live_status

def test_get_live_status_deterministic_for_fixed_venue_and_hour():
    first = get_live_status("seattle", hour=9)
    second = get_live_status("seattle", hour=9)
    assert first == second


def test_get_live_status_shape_and_enums():
    status = get_live_status("seattle", hour=9)
    assert status["simulated"] is True
    gate_names = {g["name"] for g in data.get_venue("seattle")["accessibility"]["gates"]}
    for entry in status["gate_congestion"]:
        assert entry["gate"] in gate_names
        assert entry["congestion"] in CONGESTION_LEVELS
    assert status["quiet_entrance"] in gate_names


def test_get_live_status_quiet_entrance_is_accessible_gate():
    accessible = {
        g["name"]
        for g in data.get_venue("seattle")["accessibility"]["gates"]
        if g["accessible"]
    }
    for hour in range(24):
        assert get_live_status("seattle", hour=hour)["quiet_entrance"] in accessible


def test_get_live_status_invalid_hour_falls_back_to_current_hour():
    status = get_live_status("seattle", hour="not-a-number")
    assert "error" not in status
    assert 0 <= status["hour_utc"] <= 23


def test_get_live_status_elevator_outage_keyed_to_gate_name():
    # Deterministic seed known to produce an outage.
    status = get_live_status("mexico-city", hour=1)
    outage = status["elevator_outage"]
    assert outage is not None
    gate_names = {
        g["name"] for g in data.get_venue("mexico-city")["accessibility"]["gates"]
    }
    assert outage["gate"] in gate_names
    # The quiet entrance avoids the outage gate when another option exists.
    assert status["quiet_entrance"] != outage["gate"]


def test_get_live_status_without_accessible_gates_degrades_cleanly(monkeypatch):
    # Defensive path for a (hypothetical) venue with no accessible gate: the
    # outage cannot be pinned to a gate and no quiet entrance exists, so both
    # degrade to None rather than raising.
    fake_venue = {
        "id": "no-access-venue",
        "accessibility": {
            "gates": [
                {"name": "Gate 1", "accessible": False, "notes": ""},
                {"name": "Gate 2", "accessible": False, "notes": ""},
            ],
        },
    }
    monkeypatch.setattr(tools.data, "get_venue", lambda venue_id: fake_venue)
    # Hour 5 makes this venue id's seed roll an elevator outage (see the seed
    # scheme in get_live_status), exercising the empty-accessible-names guard.
    status = tools.get_live_status("no-access-venue", hour=5)
    assert status["elevator_outage"] is None
    assert status["quiet_entrance"] is None


# --------------------------------------------------------- admin operations

def test_get_admin_overview_is_deterministic_and_labelled_simulated():
    first = get_admin_overview("mexico-city", hour=1)
    second = get_admin_overview("mexico-city", hour=1)
    assert first == second
    assert first["simulated"] is True
    assert first["summary"]["estimated_occupied_seats"] <= first["summary"]["capacity"]
    assert first["seating"]["estimated_accessible_available"] >= 0
    assert any(alert["title"].startswith("Elevator outage") for alert in first["alerts"])
    assert any(resource["status"] == "alert" for resource in first["resources"])


def test_get_admin_overview_handles_unknown_venue():
    assert "error" in get_admin_overview(UNKNOWN)


# ---------------------------------------------------------------- plan_visit

def test_plan_visit_structured_steps():
    # Hour 14 is a pinned seed with no elevator outage for VALID, so this also
    # covers the "no outage warning step" path (see the outage test below for
    # the complementary "with outage" path).
    plan = plan_visit(VALID, ["mobility", "sensory"], hour=14)
    actions = [step["action"] for step in plan["steps"]]
    assert actions[:3] == ["enter_via_gate", "arrive_early", "services_en_route"]
    assert actions.count("need_support") == 2
    assert "elevator_outage_warning" not in actions
    assert plan["needs"] == ["mobility", "sensory"]
    assert plan["simulated"] is True

    gate_step = plan["steps"][0]
    accessible = {
        g["name"]
        for g in data.get_venue(VALID)["accessibility"]["gates"]
        if g["accessible"]
    }
    assert gate_step["gate"] in accessible

    arrive_step = plan["steps"][1]
    assert arrive_step["minutes_before_kickoff"] >= 60

    services_step = plan["steps"][2]
    assert {"water", "first_aid", "nursing_room"} <= set(services_step)


def test_plan_visit_includes_elevator_outage_warning_step():
    # Deterministic seed known to produce an outage (see the live-status test).
    plan = plan_visit("mexico-city", ["mobility"], hour=1)
    actions = [step["action"] for step in plan["steps"]]
    assert "elevator_outage_warning" in actions
    warning = plan["steps"][actions.index("elevator_outage_warning")]
    assert warning["gate"]
    assert "out of service" in warning["note"]


def test_plan_visit_invalid_needs_fall_back_with_note():
    plan = plan_visit(VALID, ["flying", "mobility"], hour=13)
    assert plan["needs"] == ["mobility"]
    assert "note" in plan

    plan = plan_visit(VALID, ["flying"], hour=13)
    assert plan["needs"] == ["general"]
    assert "note" in plan


def test_plan_visit_deduplicates_repeated_needs():
    plan = plan_visit(VALID, ["mobility", "mobility"], hour=13)
    assert plan["needs"] == ["mobility"]
    assert "note" not in plan  # duplicates are not "unknown" needs


def test_plan_visit_accepts_needs_as_string_and_defaults():
    assert plan_visit(VALID, "mobility", hour=13)["needs"] == ["mobility"]
    assert plan_visit(VALID, hour=13)["needs"] == ["general"]


def test_plan_visit_echoes_language():
    assert plan_visit(VALID, ["general"], language="es", hour=13)["language"] == "es"


# --------------------------------------------------------------- execute_tool

def test_execute_tool_returns_valid_json_string():
    raw = execute_tool("get_venue_info", {"venue_id": "dallas"})
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert parsed == get_venue_info("dallas")


def test_execute_tool_unknown_tool_returns_error_json():
    parsed = json.loads(execute_tool("launch_rockets", {"venue_id": "dallas"}))
    assert "error" in parsed
    assert "launch_rockets" in parsed["error"]


def test_execute_tool_unknown_venue_returns_error_json():
    parsed = json.loads(execute_tool("get_venue_info", {"venue_id": UNKNOWN}))
    assert "error" in parsed


def test_execute_tool_missing_or_malformed_args_never_raise():
    assert "error" in json.loads(execute_tool("get_venue_info", {}))
    assert "error" in json.loads(execute_tool("get_venue_info", None))
    assert "error" in json.loads(execute_tool("get_venue_info", {"venue_id": 42}))


def test_execute_tool_drops_unexpected_args():
    parsed = json.loads(
        execute_tool("get_venue_info", {"venue_id": "dallas", "bogus": True}),
    )
    assert "error" not in parsed


def test_execute_tool_validates_need_enum_via_fallback():
    parsed = json.loads(
        execute_tool(
            "find_accessible_services", {"venue_id": "dallas", "need": "warp"},
        ),
    )
    assert parsed["need"] == "general"
    assert "note" in parsed


def test_execute_tool_matches_direct_call_for_pinned_hour():
    raw = execute_tool("get_live_status", {"venue_id": "seattle", "hour": 9})
    assert json.loads(raw) == get_live_status("seattle", hour=9)


def test_execute_tool_internal_failure_returns_error_json(monkeypatch):
    # Defensive path: even if a registered tool raises, the dispatcher must
    # return an error payload instead of propagating the exception upstream.
    def _boom(venue_id):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setitem(tools._TOOL_REGISTRY, "get_venue_info", (_boom, ("venue_id",)))
    parsed = json.loads(execute_tool("get_venue_info", {"venue_id": "dallas"}))
    assert "error" in parsed
    assert "get_venue_info" in parsed["error"]
