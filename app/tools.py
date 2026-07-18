"""Tool functions over the static venue dataset (app.data).

Each public function returns a compact, JSON-serializable dict and never
raises on bad input: unknown venues and invalid enum values produce friendly
error payloads or graceful fallbacks. ``execute_tool`` is the dispatcher used
by both the Gemini function-calling loop and the offline engine.

Results are always freshly built dicts — the shared, lru_cached dataset
returned by ``app.data`` is never mutated. ``get_live_status`` is a SIMULATED
operations feed: deterministic pseudo-random output seeded by venue id and
hour, marked ``"simulated": true``. No network access.
"""

import json
import random
from datetime import UTC, datetime
from typing import Any

from app import data

JSONDict = dict[str, Any]

VALID_NEEDS: frozenset[str] = frozenset(
    {"mobility", "vision", "hearing", "sensory", "general"},
)
CONGESTION_LEVELS: tuple[str, ...] = ("low", "moderate", "high")

#: Accessibility fields relevant to each declared need.
_NEED_FIELDS: dict[str, tuple[str, ...]] = {
    "mobility": ("gates", "elevators", "accessible_seating", "accessible_restrooms"),
    "vision": ("vision_support",),
    "hearing": ("assistive_listening",),
    "sensory": ("sensory_room", "quiet_route_hint"),
    "general": (
        "gates",
        "accessible_seating",
        "sensory_room",
        "assistive_listening",
        "vision_support",
        "elevators",
        "accessible_restrooms",
        "quiet_route_hint",
    ),
}

#: Chance (per venue-hour) that the simulated feed reports an elevator outage.
_ELEVATOR_OUTAGE_PROBABILITY = 0.15

#: Suggested arrival lead time (minutes before kickoff) by gate congestion.
_ARRIVAL_MINUTES = {"low": 60, "moderate": 75, "high": 90}

#: Congestion-level weights (low, moderate, high), indexed directly by UTC
#: hour (0-23). Models a matchday arrival curve rather than a flat
#: distribution: quiet overnight, building through the morning, peaking in the
#: pre-kickoff/kickoff window, and winding down late evening. ``hour`` is a
#: stand-in for "time relative to a typical matchday", not a specific venue's
#: local kickoff time.
_OVERNIGHT = (75, 20, 5)   # hours 0-5: near-empty concourses
_MORNING = (45, 40, 15)    # hours 6-10: gates opening, light arrivals
_MIDDAY = (20, 45, 35)     # hours 11-14: build-up toward kickoff
_PEAK = (10, 30, 60)       # hours 15-20: pre-kickoff / kickoff window
_WIND_DOWN = (35, 40, 25)  # hours 21-23: post-match egress

_HOUR_CONGESTION_WEIGHTS: tuple[tuple[int, int, int], ...] = (
    _OVERNIGHT, _OVERNIGHT, _OVERNIGHT, _OVERNIGHT, _OVERNIGHT, _OVERNIGHT,
    _MORNING, _MORNING, _MORNING, _MORNING, _MORNING,
    _MIDDAY, _MIDDAY, _MIDDAY, _MIDDAY,
    _PEAK, _PEAK, _PEAK, _PEAK, _PEAK, _PEAK,
    _WIND_DOWN, _WIND_DOWN, _WIND_DOWN,
)

# These figures model a dashboard demo rather than a real venue telemetry feed.
_CONGESTION_TRAFFIC: dict[str, tuple[int, int]] = {
    "low": (90, 4),
    "moderate": (240, 12),
    "high": (420, 24),
}
_CONGESTION_OCCUPANCY = {"low": 30, "moderate": 60, "high": 88}


def _venue_error(venue_id: object) -> JSONDict:
    """Friendly error payload for an unknown venue id."""
    examples = ", ".join(v["id"] for v in data.list_venues()[:3])
    return {
        "error": (
            f"Unknown venue id {venue_id!r}. "
            f"Use one of the 16 tournament venue ids, for example: {examples}."
        ),
    }


def _copy_gates(venue: data.Venue) -> list[JSONDict]:
    """Shallow-copy the venue's gate dicts so callers can't mutate cached data."""
    return [dict(gate) for gate in venue["accessibility"]["gates"]]


def _normalize_need(need: object) -> tuple[str, str | None]:
    """Return a valid need value plus an optional fallback note."""
    if isinstance(need, str) and need.strip().lower() in VALID_NEEDS:
        return need.strip().lower(), None
    note = (
        f"Unknown need {need!r}; showing general accessibility information. "
        f"Valid values: {', '.join(sorted(VALID_NEEDS))}."
    )
    return "general", note


def _normalize_needs(needs: object) -> tuple[list[str], str | None]:
    """Coerce ``needs`` into a list of valid need values (fallback: general)."""
    if isinstance(needs, str):
        needs = [needs]
    if not isinstance(needs, (list, tuple)) or not needs:
        return ["general"], None
    valid: list[str] = []
    invalid: list[Any] = []
    for raw in needs:
        need = raw.strip().lower() if isinstance(raw, str) else raw
        if need in VALID_NEEDS:
            if need not in valid:
                valid.append(need)
        else:
            invalid.append(raw)
    note = None
    if invalid:
        note = (
            f"Ignored unknown needs {invalid!r}. "
            f"Valid values: {', '.join(sorted(VALID_NEEDS))}."
        )
    if not valid:
        valid = ["general"]
    return valid, note


def get_venue_info(venue_id: str) -> JSONDict:
    """Return basic venue facts: names, location, capacity, gates, matchday basics."""
    venue = data.get_venue(venue_id)
    if venue is None:
        return _venue_error(venue_id)
    tournament = data.load_venues()["tournament"]
    matchday: JSONDict = {
        "accessibility_ticket_types": list(
            tournament["accessibility_tickets"]["types"],
        ),
        "companion_tickets": tournament["accessibility_tickets"]["companion_tickets"],
    }
    if tournament["openingMatch"]["venueId"] == venue_id:
        matchday["hosts_opening_match"] = tournament["openingMatch"]["date"]
    if tournament["final"]["venueId"] == venue_id:
        matchday["hosts_final"] = tournament["final"]["date"]
    return {
        "id": venue["id"],
        "name": venue["name"],
        "commercialName": venue["commercialName"],
        "fifaName": venue["fifaName"],
        "city": venue["city"],
        "country": venue["country"],
        "capacity": venue["capacity"],
        "capacity_note": "approximate tournament capacity",
        "gates": _copy_gates(venue),
        "matchday": matchday,
    }


def find_accessible_services(venue_id: str, need: str = "general") -> JSONDict:
    """Return accessibility services at a venue, filtered by declared need.

    ``need`` must be one of mobility/vision/hearing/sensory/general; anything
    else falls back to "general" with an explanatory note. The result carries
    the dataset's ``verified`` flag so answers can caveat unverified data.
    """
    venue = data.get_venue(venue_id)
    if venue is None:
        return _venue_error(venue_id)
    need, note = _normalize_need(need)
    accessibility = venue["accessibility"]
    services: JSONDict = {
        field: _copy_gates(venue) if field == "gates" else accessibility[field]
        for field in _NEED_FIELDS[need]
    }
    result: JSONDict = {
        "venue_id": venue["id"],
        "venue_name": venue["name"],
        "need": need,
        "services": services,
        "verified": accessibility["verified"],
    }
    if note:
        result["note"] = note
    return result


def _quiet_entrance(
    gate_congestion: list[JSONDict], outage_gate: str | None,
) -> str | None:
    """Least congested accessible gate, avoiding an elevator outage if possible."""
    ranked = sorted(
        (entry for entry in gate_congestion if entry["accessible"]),
        key=lambda entry: (
            entry["gate"] == outage_gate,
            CONGESTION_LEVELS.index(entry["congestion"]),
        ),
    )
    return ranked[0]["gate"] if ranked else None


def get_live_status(venue_id: str, hour: int | None = None) -> JSONDict:
    """Return the SIMULATED live operations feed for a venue.

    Deterministic pseudo-random output seeded by venue id + hour: per-gate
    congestion (low/moderate/high, weighted by ``_HOUR_CONGESTION_WEIGHTS`` so
    a matchday-peak hour is genuinely more likely to read "high" than 3am), a
    possible elevator outage keyed to an accessible gate name, and a
    quiet-entrance suggestion. ``hour`` defaults to the current UTC hour; pass
    a fixed value to pin the seed (tests).
    """
    venue = data.get_venue(venue_id)
    if venue is None:
        return _venue_error(venue_id)
    if hour is None:
        hour = datetime.now(UTC).hour
    else:
        try:
            hour = int(hour) % 24
        except (TypeError, ValueError):
            hour = datetime.now(UTC).hour
    rng = random.Random(f"{venue_id}-{hour}")  # noqa: S311 — deterministic simulation, not security
    weights = _HOUR_CONGESTION_WEIGHTS[hour]
    gates = venue["accessibility"]["gates"]
    gate_congestion = [
        {
            "gate": gate["name"],
            "accessible": gate["accessible"],
            "congestion": rng.choices(CONGESTION_LEVELS, weights=weights, k=1)[0],
        }
        for gate in gates
    ]
    outage_gate: str | None = None
    if rng.random() < _ELEVATOR_OUTAGE_PROBABILITY:
        accessible_names = [g["name"] for g in gates if g["accessible"]]
        if accessible_names:
            outage_gate = rng.choice(accessible_names)
    elevator_outage: JSONDict | None = None
    if outage_gate is not None:
        elevator_outage = {
            "gate": outage_gate,
            "note": (
                f"Elevator near {outage_gate} is out of service; "
                "staff can assist at the other accessible gates."
            ),
        }
    return {
        "simulated": True,
        "venue_id": venue["id"],
        "hour_utc": hour,
        "gate_congestion": gate_congestion,
        "elevator_outage": elevator_outage,
        "quiet_entrance": _quiet_entrance(gate_congestion, outage_gate),
    }


def get_admin_overview(venue_id: str, hour: int | None = None) -> JSONDict:
    """Return a simulated, read-only operations snapshot for one venue.

    The dataset has no turnstile, seat-sensor, or inventory integration, so
    estimated traffic, seating utilisation, and staff/equipment availability
    are derived from the deterministic live-status simulation. The explicit
    flags and notes keep this demo from claiming the values are live facts.
    """
    venue = data.get_venue(venue_id)
    if venue is None:
        return _venue_error(venue_id)

    status = get_live_status(venue_id, hour=hour)
    gate_traffic: list[JSONDict] = []
    occupancy_samples: list[int] = []
    for gate in status["gate_congestion"]:
        congestion = gate["congestion"]
        entrants, queue_minutes = _CONGESTION_TRAFFIC[congestion]
        if not gate["accessible"]:
            entrants = int(entrants * 1.15)
        gate_traffic.append(
            {
                "gate": gate["gate"],
                "accessible": gate["accessible"],
                "congestion": congestion,
                "estimated_entrants_per_15_min": entrants,
                "estimated_queue_minutes": queue_minutes,
            },
        )
        occupancy_samples.append(_CONGESTION_OCCUPANCY[congestion])

    occupancy_percent = round(sum(occupancy_samples) / len(occupancy_samples))
    capacity = venue["capacity"]
    estimated_occupied = round(capacity * occupancy_percent / 100)
    accessible_gates = sum(1 for gate in gate_traffic if gate["accessible"])
    accessible_allocation = max(50, round(capacity * 0.005))
    accessible_occupancy = min(100, occupancy_percent + 4)
    accessible_occupied = round(accessible_allocation * accessible_occupancy / 100)
    has_sensory_room = not venue["accessibility"]["sensory_room"].lower().startswith("none")
    outage = status["elevator_outage"]

    resources: list[JSONDict] = [
        {
            "resource": "Accessible entry teams",
            "status": "available",
            "detail": f"{accessible_gates} accessible gates supported in this snapshot.",
        },
        {
            "resource": "Wheelchair assistance",
            "status": "available" if occupancy_percent < 80 else "watch",
            "detail": (
                f"Estimated {max(2, accessible_gates * 3 - occupancy_percent // 30)} "
                "assist teams available."
            ),
        },
        {
            "resource": "Elevators",
            "status": "alert" if outage else "available",
            "detail": outage["note"] if outage else venue["accessibility"]["elevators"],
        },
        {
            "resource": "Sensory support",
            "status": "available" if has_sensory_room else "on_request",
            "detail": venue["accessibility"]["sensory_room"],
        },
        {
            "resource": "First aid",
            "status": "available",
            "detail": venue["services"]["first_aid"],
        },
    ]
    alerts: list[JSONDict] = []
    for gate in gate_traffic:
        if gate["congestion"] == "high":
            alerts.append(
                {
                    "severity": "high",
                    "title": f"High traffic at {gate['gate']}",
                    "detail": (
                        f"Estimated queue: {gate['estimated_queue_minutes']} minutes. "
                        "Consider directing guests to another accessible gate."
                    ),
                },
            )
    if outage:
        alerts.append(
            {
                "severity": "high",
                "title": f"Elevator outage near {outage['gate']}",
                "detail": outage["note"],
            },
        )
    if not venue["accessibility"]["verified"]:
        alerts.append(
            {
                "severity": "info",
                "title": "Venue data needs verification",
                "detail": (
                    "Accessibility and resource descriptions in this venue record are illustrative."
                ),
            },
        )
    if not alerts:
        alerts.append(
            {
                "severity": "info",
                "title": "No active operational alerts",
                "detail": "Continue monitoring the next simulated update.",
            },
        )

    return {
        "simulated": True,
        "venue_id": venue["id"],
        "venue_name": venue["name"],
        "hour_utc": status["hour_utc"],
        "summary": {
            "capacity": capacity,
            "estimated_occupancy_percent": occupancy_percent,
            "estimated_occupied_seats": estimated_occupied,
            "estimated_available_seats": capacity - estimated_occupied,
            "estimate_note": (
                "Derived from simulated gate traffic; not a ticketing or seat-sensor count."
            ),
        },
        "seating": {
            "accessible_seating_description": venue["accessibility"]["accessible_seating"],
            "estimated_accessible_allocation": accessible_allocation,
            "estimated_accessible_occupied": accessible_occupied,
            "estimated_accessible_available": accessible_allocation - accessible_occupied,
            "estimate_note": (
                "Demo planning estimate (0.5% of venue capacity), not an allocation record."
            ),
        },
        "gate_traffic": gate_traffic,
        "resources": resources,
        "alerts": alerts,
    }


def _quiet_gate_and_congestion(
    accessibility: JSONDict, status: JSONDict, gate_name: str | None,
) -> tuple[JSONDict | None, str]:
    """Look up the recommended gate's own record and its current congestion."""
    gate = next(
        (g for g in accessibility["gates"] if g["name"] == gate_name), None,
    )
    congestion = next(
        (
            entry["congestion"]
            for entry in status["gate_congestion"]
            if entry["gate"] == gate_name
        ),
        "low",
    )
    return gate, congestion


def _entry_step(gate_name: str | None, gate: JSONDict | None, congestion: str) -> JSONDict:
    """Build the 'enter via the recommended gate' step."""
    return {
        "action": "enter_via_gate",
        "gate": gate_name,
        "gate_notes": gate["notes"] if gate else "",
        "congestion": congestion,
        "reason": "least congested accessible gate right now (simulated live data)",
    }


def _arrival_step(congestion: str, *, has_special_need: bool) -> JSONDict:
    """Build the 'arrive early' step (extra buffer when a special need was declared)."""
    minutes = _ARRIVAL_MINUTES[congestion] + (15 if has_special_need else 0)
    return {"action": "arrive_early", "minutes_before_kickoff": minutes}


def _need_support_steps(valid_needs: list[str], accessibility: JSONDict) -> list[JSONDict]:
    """Build one 'need_support' step per declared accessibility need."""
    steps = []
    for need in valid_needs:
        tips = {
            field: accessibility[field]
            for field in _NEED_FIELDS[need]
            if field != "gates"
        }
        steps.append({"action": "need_support", "need": need, "tips": tips})
    return steps


def plan_visit(
    venue_id: str,
    needs: list[str] | None = None,
    language: str = "en",
    hour: int | None = None,
) -> JSONDict:
    """Structured step-by-step arrival plan for a venue visit.

    Combines the simulated live status (recommended quiet accessible gate),
    arrive-early advice, services en route, and need-specific tips. Returns
    structured steps, not prose — the LLM/offline layer renders language.
    """
    venue = data.get_venue(venue_id)
    if venue is None:
        return _venue_error(venue_id)
    valid_needs, note = _normalize_needs(needs)
    status = get_live_status(venue_id, hour=hour)
    accessibility = venue["accessibility"]
    services = venue["services"]

    gate_name = status["quiet_entrance"]
    gate, congestion = _quiet_gate_and_congestion(accessibility, status, gate_name)
    has_special_need = any(need != "general" for need in valid_needs)

    steps: list[JSONDict] = [
        _entry_step(gate_name, gate, congestion),
        _arrival_step(congestion, has_special_need=has_special_need),
        {
            "action": "services_en_route",
            "water": services["water"],
            "first_aid": services["first_aid"],
            "nursing_room": services["nursing_room"],
        },
        *_need_support_steps(valid_needs, accessibility),
    ]
    if status["elevator_outage"] is not None:
        steps.append(
            {
                "action": "elevator_outage_warning",
                "gate": status["elevator_outage"]["gate"],
                "note": status["elevator_outage"]["note"],
            },
        )
    for i, step in enumerate(steps, start=1):
        step["step"] = i

    result: JSONDict = {
        "venue_id": venue["id"],
        "venue_name": venue["name"],
        "language": language,
        "needs": valid_needs,
        "simulated": True,
        "verified": accessibility["verified"],
        "steps": steps,
    }
    if note:
        result["note"] = note
    return result


#: Dispatcher registry: tool name -> (function, accepted argument names).
_TOOL_REGISTRY: dict[str, tuple[Any, tuple[str, ...]]] = {
    "get_venue_info": (get_venue_info, ("venue_id",)),
    "find_accessible_services": (find_accessible_services, ("venue_id", "need")),
    "get_live_status": (get_live_status, ("venue_id", "hour")),
    "plan_visit": (plan_visit, ("venue_id", "needs", "language", "hour")),
}


def execute_tool(name: str, args: dict[str, Any] | None) -> str:
    """Dispatch a tool call by name and return the result as a JSON string.

    Never raises: unknown tool names, unknown venues, missing arguments, and
    unexpected failures all come back as ``{"error": ...}`` JSON strings.
    Unrecognized argument keys are silently dropped.
    """
    if not isinstance(name, str) or name not in _TOOL_REGISTRY:
        payload: JSONDict = {
            "error": (
                f"Unknown tool {name!r}. "
                f"Available tools: {', '.join(sorted(_TOOL_REGISTRY))}."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)
    func, accepted = _TOOL_REGISTRY[name]
    if not isinstance(args, dict):
        args = {}
    kwargs = {key: value for key, value in args.items() if key in accepted}
    venue_id = kwargs.get("venue_id")
    if not isinstance(venue_id, str) or data.get_venue(venue_id) is None:
        return json.dumps(_venue_error(venue_id), ensure_ascii=False)
    try:
        result = func(**kwargs)
    except Exception as exc:  # noqa: BLE001 — defensive: tools should never raise upstream
        result = {"error": f"Tool {name!r} failed on the given arguments: {exc}"}
    return json.dumps(result, ensure_ascii=False)
