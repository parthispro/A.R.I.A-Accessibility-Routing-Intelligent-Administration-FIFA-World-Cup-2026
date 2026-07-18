"""Tests for app.offline: deterministic keyword-routed fallback assistant."""

from app.offline import offline_answer


def profile(language="en", venue_id=None, needs=None):
    return {"language": language, "needs": needs or [], "venue_id": venue_id}


# ------------------------------------------------------------- accessibility

def test_english_accessibility_question_mentions_facilities():
    answer = offline_answer(
        "Is there wheelchair access for my seat?",
        profile("en", "new-york-new-jersey"),
    )
    assert "Accessibility at MetLife Stadium" in answer
    assert "Accessible gates" in answer
    assert "MetLife Gate" in answer
    assert "wheelchair" in answer.lower()


def test_sensory_question_routes_to_sensory_fields():
    answer = offline_answer(
        "Where is the quietest place for my autistic son?",
        profile("en", "los-angeles"),
    )
    assert "Sensory support" in answer
    assert "sensory" in answer.lower()


def test_unverified_venue_gets_caveat():
    answer = offline_answer("Is there a ramp?", profile("en", "dallas"))
    assert "not yet verified" in answer


# ---------------------------------------------------------------- languages

def test_spanish_nursing_room_question_answered_in_spanish():
    answer = offline_answer(
        "¿Dónde está el área de lactancia?", profile("es", "mexico-city"),
    )
    assert "lactancia" in answer
    assert "Ubicación" in answer
    assert "Nursing room near Puerta 1" in answer
    # Real Spanish, not prefixed English boilerplate.
    assert not answer.startswith("Yes")


def test_french_question_answered_in_french():
    answer = offline_answer(
        "Où sont les toilettes accessibles ?", profile("fr", "toronto"),
    )
    assert "Accessibilité" in answer
    assert "Toilettes accessibles" in answer
    assert "vérifiées" in answer  # unverified caveat, in French


def test_unknown_language_code_falls_back_to_english():
    answer = offline_answer("Is there wheelchair access?", profile("de", "dallas"))
    assert "Accessibility at AT&T Stadium" in answer


def test_language_detected_from_message_when_profile_has_none():
    # No profile language: the Spanish keyword match picks the es template.
    answer = offline_answer("hola", {})
    assert "Copa Mundial" in answer


def test_unmatched_message_without_profile_language_falls_back_to_english():
    answer = offline_answer("qwertyuiop", {})
    assert "I can help" in answer


# ------------------------------------------------------------------- arabic

def test_arabic_greeting_uses_arabic_template():
    # "Peace be upon you" — a greeting; must not fall back to English.
    answer = offline_answer("السلام عليكم", profile("ar", None))
    assert "A.R.I.A" in answer
    assert "FIFA" in answer
    assert "مرحبا" in answer  # Arabic "hello", proves the ar template was used


def test_arabic_wheelchair_question_routes_and_renders_in_arabic():
    # "Where is the route for the wheelchair?" -> accessibility, ar template.
    answer = offline_answer(
        "أين مسار الكرسي المتحرك؟", profile("ar", "new-york-new-jersey"),
    )
    assert "إمكانية الوصول في MetLife Stadium" in answer  # ar intro + venue fact
    assert "البوابات المتاحة" in answer                    # ar "accessible gates"
    assert "MetLife Gate" in answer                        # dataset gate name


def test_arabic_schedule_final_date():
    # "When is the final?" -> schedule, ar template, real date interpolated.
    answer = offline_answer("متى النهائي؟", profile("ar", "new-york-new-jersey"))
    assert "المباراة النهائية" in answer
    assert "2026-07-19" in answer


def test_arabic_food_water_localized():
    answer = offline_answer("أين الماء؟", profile("ar", "guadalajara"))
    assert "الماء في Estadio Akron" in answer


def test_arabic_unverified_venue_gets_caveat_in_arabic():
    answer = offline_answer("هل يوجد منحدر؟", profile("ar", "dallas"))
    assert "لم يتم التحقق" in answer  # "not yet verified", in Arabic


def test_arabic_no_venue_prompt_is_localized():
    answer = offline_answer("هل يوجد غرفة حسية؟", profile("ar", None))
    assert "من فضلك اختر" in answer      # "please choose", in Arabic
    assert "MetLife Stadium" in answer   # examples still listed


def test_arabic_clitic_prefixed_keyword_still_matches():
    # "...my child with autism" — the preposition/article glue onto the noun
    # ("بالتوحد"); the engine must still route to the sensory accessibility path.
    answer = offline_answer(
        "أريد مكاناً هادئاً لطفلي المصاب بالتوحد", profile("ar", "los-angeles"),
    )
    assert "الدعم الحسي" in answer  # "sensory support" field label, in Arabic


def test_arabic_profile_language_overrides_english_message():
    # English words route the intent, but the ar profile picks the template.
    answer = offline_answer("Is there wheelchair access?", profile("ar", "dallas"))
    assert "إمكانية الوصول في AT&T Stadium" in answer


def test_arabic_answers_are_deterministic():
    msg, prof = "أين المصعد؟", profile("ar", "los-angeles")
    assert offline_answer(msg, prof) == offline_answer(msg, prof)


# ------------------------------------------------------------ profile needs

def test_profile_need_used_when_message_has_no_need_keyword():
    answer = offline_answer(
        "Is the stadium accessible?",
        profile("en", "new-york-new-jersey", ["hearing"]),
    )
    assert "Assistive listening" in answer


def test_invalid_profile_need_falls_back_to_general():
    answer = offline_answer(
        "Is the stadium accessible?",
        profile("en", "new-york-new-jersey", ["flying"]),
    )
    assert "Accessible gates" in answer  # the general view shows the full set


# ----------------------------------------------------------------- services

def test_generic_services_question_lists_all_three_services():
    # A services-intent message with no specific service keyword falls back to
    # describing all three (nursing room, first aid, prayer space).
    from app import data
    from app.offline import _services_answer

    venue = data.get_venue("new-york-new-jersey")
    answer = _services_answer(venue, "services", "en")
    assert "nursing room" in answer
    assert "First aid" in answer
    assert "Prayer space" in answer


# --------------------------------------------------------------- navigation

def test_navigation_question_mentions_a_gate():
    answer = offline_answer(
        "Which gate should I use to get in?", profile("en", "seattle"),
    )
    assert "Recommended entrance" in answer
    assert "Northwest Gate" in answer or "Southeast Gate" in answer


def test_navigation_answer_warns_about_elevator_outage(monkeypatch):
    from app import tools

    real = tools.get_live_status
    # Pin the simulated feed to a seed known to produce an outage (hour 1).
    monkeypatch.setattr(
        "app.offline.tools.get_live_status",
        lambda venue_id, hour=None: real(venue_id, hour=1),
    )
    answer = offline_answer(
        "Which gate should I use?", profile("en", "mexico-city"),
    )
    assert "out of service" in answer


# ------------------------------------------------------------------ no venue

def test_no_venue_asks_user_to_pick_one_with_examples():
    answer = offline_answer("Is there a sensory room?", profile("en", None))
    assert "choose a stadium" in answer
    assert "Estadio Azteca" in answer
    assert "MetLife Stadium" in answer


def test_no_venue_prompt_is_localized():
    answer = offline_answer(
        "¿Hay rampa para silla de ruedas?", profile("es", None),
    )
    assert "elija primero un estadio" in answer


def test_unknown_venue_id_treated_as_no_venue():
    answer = offline_answer("Is there a ramp?", profile("en", "atlantis"))
    assert "choose a stadium" in answer


# ------------------------------------------------------------- other intents

def test_greeting():
    answer = offline_answer("Hello!", profile("en", None))
    assert "A.R.I.A" in answer


def test_fallback_help_for_unmatched_message():
    answer = offline_answer("asdfghjkl", profile("en", "dallas"))
    assert "I can help" in answer


def test_schedule_final_date():
    answer = offline_answer(
        "When is the final?", profile("en", "new-york-new-jersey"),
    )
    assert "2026-07-19" in answer


def test_schedule_opening_match_for_hosting_venue():
    answer = offline_answer(
        "When is the opening match?", profile("en", "mexico-city"),
    )
    assert "opening match" in answer
    assert "2026-06-11" in answer


def test_schedule_works_without_venue_in_french():
    answer = offline_answer("Quand a lieu la finale ?", profile("fr", None))
    assert "2026-06-11" in answer
    assert "2026-07-19" in answer
    assert "finale" in answer


def test_food_water_spanish():
    answer = offline_answer("¿Dónde hay agua?", profile("es", "guadalajara"))
    assert "Agua en Estadio Akron" in answer


# ------------------------------------------------------------- determinism

def test_offline_answers_are_deterministic():
    cases = [
        ("Is there wheelchair access?", profile("en", "dallas")),
        ("¿Dónde está el área de lactancia?", profile("es", "mexico-city")),
        ("Hello!", profile("en", None)),
    ]
    for message, prof in cases:
        assert offline_answer(message, prof) == offline_answer(message, prof)


def test_no_emoji_in_answers():
    answer = offline_answer(
        "Is there wheelchair access?", profile("en", "dallas"),
    )
    assert all(ord(ch) < 0x2600 for ch in answer)
