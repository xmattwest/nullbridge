"""Stimulus suite and deterministic firewall audit for Phase 6 (`phase6_inkling_biophysical_loop`).

Constructs 160 paired, zero-leakage mechanistic evaluation items across 4 regimes (40 each):
  1. `factual_2hop` (n = 40): 2-hop factual entity bridges (`E0 -> E1 -> E2`), where the prompt
     names `E0` (landmark, artwork, scientist, river, mountain, or currency), never mentions the
     intermediate bridge entity `E1` (country or creator), and queries the final target `E2`.
     Includes matched `foil_source_prompt` items whose true answer is `foil_answer`.
  2. `arithmetic_2step` (n = 40): 2-step exact integer expressions (`(a op1 b) op2 c`), verified
     programmatically so intermediate `hop1_bridge`, final `target_answer`, and `foil_answer`
     are pairwise distinct single-token integers in `[2, 98]`.
  3. `calendar_2step` (n = 40): 2-step modular sequence deductions across weekdays, months, and
     cardinal/alphabetic cycles, programmatically verified with distinct Hop-1 and Hop-2 states.
  4. `chat_shunt_triplets` (n = 40): Matched triplets rendering the exact same question as
     (a) `<|message_model|>` header-only (`effort = 0.7` and `effort = 0`),
     (b) `<|message_model|><|content_text|>` (1 token later), and
     (c) raw completion without chat formatting tokens, including canonical anchor items
     (`"New Delhi"`, `"7"`, `"Saturday"`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EM_DASH = "\u2014"

WEEKDAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class BiophysicalItem:
    item_id: str
    regime: str
    user_prompt: str
    raw_prompt: str
    hop1_bridge: str
    target_answer: str
    foil_answer: str
    foil_user_prompt: str
    foil_raw_prompt: str
    note: str = ""


@dataclass(frozen=True)
class ChatShuntTriplet:
    item_id: str
    domain: str
    user_question: str
    raw_completion_prompt: str
    hop1_bridge: str
    target_answer: str
    foil_answer: str


# 40 distinct (clue_phrase, hop1_country_or_bridge, hop2_capital_or_target) tuples.
# Notice `hop1` never appears inside `clue_phrase`, and `hop1 != hop2` for every row.
_FACTUAL_2HOP_SPECS: tuple[tuple[str, str, str], ...] = (
    ("the Taj Mahal", "India", "New Delhi"),
    ("the Eiffel Tower", "France", "Paris"),
    ("Mount Fuji", "Japan", "Tokyo"),
    ("the Great Pyramid of Giza", "Egypt", "Cairo"),
    ("the Colosseum", "Italy", "Rome"),
    ("the Brandenburg Gate", "Germany", "Berlin"),
    ("the Alhambra palace", "Spain", "Madrid"),
    ("the Parthenon", "Greece", "Athens"),
    ("Machu Picchu", "Peru", "Lima"),
    ("Wawel Castle", "Poland", "Warsaw"),
    ("Schonbrunn Palace", "Austria", "Vienna"),
    ("the Vigeland Sculpture Park", "Norway", "Oslo"),
    ("the Vasa Museum", "Sweden", "Stockholm"),
    ("the Forbidden City", "China", "Beijing"),
    ("Saint Basil's Cathedral", "Russia", "Moscow"),
    ("Stonehenge", "United Kingdom", "London"),
    ("Wat Arun", "Thailand", "Bangkok"),
    ("Hagia Sophia", "Turkey", "Ankara"),
    ("Maasai Mara National Reserve", "Kenya", "Nairobi"),
    ("Patagonia's Perito Moreno Glacier", "Argentina", "Buenos Aires"),
    ("the Atacama Desert's Mano del Desierto", "Chile", "Santiago"),
    ("Belem Tower", "Portugal", "Lisbon"),
    ("the Cliffs of Moher", "Ireland", "Dublin"),
    ("Tivoli Gardens", "Denmark", "Copenhagen"),
    ("Suomenlinna sea fortress", "Finland", "Helsinki"),
    ("Lake Balaton", "Hungary", "Budapest"),
    ("Persepolis", "Iran", "Tehran"),
    ("Gyeongbokgung Palace", "South Korea", "Seoul"),
    ("Ha Long Bay", "Vietnam", "Hanoi"),
    ("Banaue Rice Terraces", "Philippines", "Manila"),
    ("Borobudur temple", "Indonesia", "Jakarta"),
    ("Jemaa el-Fnaa marketplace", "Morocco", "Rabat"),
    ("Zuma Rock", "Nigeria", "Abuja"),
    ("Varadero Peninsula", "Cuba", "Havana"),
    ("Blue Mountains Peak", "Jamaica", "Kingston"),
    ("Pashupatinath Temple", "Nepal", "Kathmandu"),
    ("Sundarbans mangrove delta's Padma confluence", "Bangladesh", "Dhaka"),
    ("Mohenjo-daro ruins", "Pakistan", "Islamabad"),
    ("Cartagena's Castillo San Felipe", "Colombia", "Bogota"),
    ("Banff National Park", "Canada", "Ottawa"),
)


def build_factual_2hop_items() -> list[BiophysicalItem]:
    """Build 40 2-hop landmark -> country -> capital bridge items with matched foil prompts."""
    items: list[BiophysicalItem] = []
    n = len(_FACTUAL_2HOP_SPECS)
    for idx, (landmark, country, capital) in enumerate(_FACTUAL_2HOP_SPECS):
        foil_idx = (idx + 7) % n
        foil_landmark, _foil_country, foil_capital = _FACTUAL_2HOP_SPECS[foil_idx]
        user_p = (
            f"Name the national capital city of the country where {landmark} is located. "
            "Reply with the city name only, with no punctuation or extra words."
        )
        raw_p = f"Fact: The national capital city of the country where {landmark} is located is"
        foil_user_p = (
            f"Name the national capital city of the country where {foil_landmark} is located. "
            "Reply with the city name only, with no punctuation or extra words."
        )
        foil_raw_p = f"Fact: The national capital city of the country where {foil_landmark} is located is"
        items.append(
            BiophysicalItem(
                item_id=f"fact2h_{idx:02d}",
                regime="factual_2hop",
                user_prompt=user_p,
                raw_prompt=raw_p,
                hop1_bridge=country,
                target_answer=capital,
                foil_answer=foil_capital,
                foil_user_prompt=foil_user_p,
                foil_raw_prompt=foil_raw_p,
                note=f"{landmark} -> {country} -> {capital}",
            )
        )
    return items


# 40 deterministic 2-step arithmetic tuples: (expr_text, expected_hop1, expected_final)
_ARITH_2STEP_SPECS: tuple[tuple[str, int, int], ...] = (
    ("(3 * 2) + 1", 6, 7),
    ("(4 * 3) + 5", 12, 17),
    ("(6 * 4) - 5", 24, 19),
    ("(5 * 5) + 8", 25, 33),
    ("(7 * 3) + 4", 21, 25),
    ("(8 * 4) - 3", 32, 29),
    ("(9 * 3) + 6", 27, 33),
    ("(6 * 5) + 7", 30, 37),
    ("(7 * 4) - 5", 28, 23),
    ("(8 * 5) + 3", 40, 43),
    ("(9 * 4) + 5", 36, 41),
    ("(6 * 6) - 7", 36, 29),
    ("(7 * 5) + 6", 35, 41),
    ("(8 * 6) - 5", 48, 43),
    ("(9 * 5) + 4", 45, 49),
    ("(7 * 6) + 5", 42, 47),
    ("(8 * 7) - 3", 56, 53),
    ("(9 * 6) + 7", 54, 61),
    ("(6 * 8) + 5", 48, 53),
    ("(7 * 7) - 6", 49, 43),
    ("(14 + 9) * 2", 23, 46),
    ("(11 + 8) * 3", 19, 57),
    ("(15 - 6) * 4", 9, 36),
    ("(17 - 9) * 5", 8, 40),
    ("(13 + 6) * 2", 19, 38),
    ("(18 - 7) * 3", 11, 33),
    ("(12 + 5) * 4", 17, 68),
    ("(19 - 8) * 5", 11, 55),
    ("(16 + 7) * 2", 23, 46),
    ("(21 - 9) * 4", 12, 48),
    ("(24 + 8) - 9", 32, 23),
    ("(35 - 12) + 8", 23, 31),
    ("(27 + 15) - 6", 42, 36),
    ("(44 - 16) + 9", 28, 37),
    ("(19 + 14) - 7", 33, 26),
    ("(52 - 18) + 5", 34, 39),
    ("(29 + 17) - 8", 46, 38),
    ("(61 - 24) + 7", 37, 44),
    ("(33 + 18) - 9", 51, 42),
    ("(48 - 19) + 6", 29, 35),
)


def _eval_two_step_expr(expr: str) -> tuple[int, int]:
    """Programmatically verify `(a op1 b) op2 c` -> `(hop1, final)`."""
    left_part, rest = expr.split(")")
    inner = left_part.lstrip("(").strip()
    a_s, op1, b_s = inner.split()
    op2, c_s = rest.strip().split()
    a, b, c = int(a_s), int(b_s), int(c_s)
    if op1 == "+":
        m = a + b
    elif op1 == "-":
        m = a - b
    elif op1 == "*":
        m = a * b
    else:
        raise ValueError(f"Unsupported op1: {op1}")

    if op2 == "+":
        out = m + c
    elif op2 == "-":
        out = m - c
    elif op2 == "*":
        out = m * c
    else:
        raise ValueError(f"Unsupported op2: {op2}")
    return m, out


def build_arithmetic_2step_items() -> list[BiophysicalItem]:
    """Build 40 verified 2-step arithmetic items with distinct intermediate and foil integers."""
    items: list[BiophysicalItem] = []
    n = len(_ARITH_2STEP_SPECS)
    for idx, (expr, exp_m, exp_y) in enumerate(_ARITH_2STEP_SPECS):
        got_m, got_y = _eval_two_step_expr(expr)
        if (got_m, got_y) != (exp_m, exp_y):
            raise ValueError(f"Arithmetic spec mismatch on {expr}: {(got_m, got_y)} != {(exp_m, exp_y)}")
        # Pick a deterministic foil index whose final answer differs from both `exp_y` and `exp_m`
        foil_idx = (idx + 5) % n
        while _ARITH_2STEP_SPECS[foil_idx][2] in (exp_y, exp_m):
            foil_idx = (foil_idx + 1) % n
        foil_expr, _foil_m, foil_y = _ARITH_2STEP_SPECS[foil_idx]

        user_p = (
            f"Compute {expr}. "
            "Reply with the single integer only, with no punctuation or explanation."
        )
        raw_p = f"Expression: {expr} ="
        foil_user_p = (
            f"Compute {foil_expr}. "
            "Reply with the single integer only, with no punctuation or explanation."
        )
        foil_raw_p = f"Expression: {foil_expr} ="
        items.append(
            BiophysicalItem(
                item_id=f"arith2s_{idx:02d}",
                regime="arithmetic_2step",
                user_prompt=user_p,
                raw_prompt=raw_p,
                hop1_bridge=str(exp_m),
                target_answer=str(exp_y),
                foil_answer=str(foil_y),
                foil_user_prompt=foil_user_p,
                foil_raw_prompt=foil_raw_p,
                note=f"{expr} -> {exp_m} -> {exp_y}",
            )
        )
    return items


def build_calendar_2step_items() -> list[BiophysicalItem]:
    """Build 40 verified 2-step weekday and month sequence deduction items."""
    raw_specs: list[tuple[str, str, str, str]] = []

    # 20 weekday 2-step items (item #0: Wednesday + 1 day -> Thursday + 2 days -> Saturday)
    weekday_shifts: list[tuple[int, int, int]] = [
        (2, 1, 2),  # Wednesday -> Thursday -> Saturday
        (0, 1, 2),  # Monday -> Tuesday -> Thursday
        (1, 2, 1),  # Tuesday -> Thursday -> Friday
        (3, 1, 2),  # Thursday -> Friday -> Sunday
        (4, 1, 2),  # Friday -> Saturday -> Monday
        (5, 1, 2),  # Saturday -> Sunday -> Tuesday
        (6, 1, 2),  # Sunday -> Monday -> Wednesday
        (0, 2, 2),  # Monday -> Wednesday -> Friday
        (1, 1, 3),  # Tuesday -> Wednesday -> Saturday
        (2, 2, 2),  # Wednesday -> Friday -> Sunday
        (3, 2, 2),  # Thursday -> Saturday -> Monday
        (4, 2, 2),  # Friday -> Sunday -> Tuesday
        (5, 2, 2),  # Saturday -> Monday -> Wednesday
        (6, 2, 2),  # Sunday -> Tuesday -> Thursday
        (0, 3, 1),  # Monday -> Thursday -> Friday
        (1, 3, 2),  # Tuesday -> Friday -> Sunday
        (2, 3, 1),  # Wednesday -> Saturday -> Sunday
        (3, 3, 1),  # Thursday -> Sunday -> Monday
        (4, 3, 1),  # Friday -> Monday -> Tuesday
        (5, 3, 1),  # Saturday -> Tuesday -> Wednesday
    ]
    for start_i, d1, d2 in weekday_shifts:
        s_day = WEEKDAYS[start_i]
        hop1 = WEEKDAYS[(start_i + d1) % 7]
        hop2 = WEEKDAYS[(start_i + d1 + d2) % 7]
        d1_lbl = "1 day" if d1 == 1 else f"{d1} days"
        d2_lbl = "1 more day" if d2 == 1 else f"{d2} more days"
        u_txt = (
            f"Start on {s_day}, move forward {d1_lbl}, and then move forward {d2_lbl}. "
            "What day of the week do you land on? Reply with the single day name only."
        )
        r_txt = f"Calendar sequence: Start on {s_day}, advance {d1_lbl}, then advance {d2_lbl}. Final day:"
        raw_specs.append((u_txt, r_txt, hop1, hop2))

    # 20 month 2-step items
    month_shifts: list[tuple[int, int, int]] = [
        (0, 1, 2),   # Jan -> Feb -> Apr
        (1, 2, 2),   # Feb -> Apr -> Jun
        (2, 1, 3),   # Mar -> Apr -> Jul
        (3, 2, 2),   # Apr -> Jun -> Aug
        (4, 1, 2),   # May -> Jun -> Aug
        (5, 2, 3),   # Jun -> Aug -> Nov
        (6, 1, 2),   # Jul -> Aug -> Oct
        (7, 2, 2),   # Aug -> Oct -> Dec
        (8, 1, 2),   # Sep -> Oct -> Dec
        (9, 1, 2),   # Oct -> Nov -> Jan
        (10, 1, 2),  # Nov -> Dec -> Feb
        (11, 1, 2),  # Dec -> Jan -> Mar
        (0, 2, 3),   # Jan -> Mar -> Jun
        (2, 2, 3),   # Mar -> May -> Aug
        (4, 2, 3),   # May -> Jul -> Oct
        (6, 2, 3),   # Jul -> Sep -> Dec
        (8, 2, 3),   # Sep -> Nov -> Feb
        (9, 2, 3),   # Oct -> Dec -> Mar
        (10, 2, 3),  # Nov -> Jan -> Apr
        (11, 2, 3),  # Dec -> Feb -> May
    ]
    for start_m, m1, m2 in month_shifts:
        s_mon = MONTHS[start_m]
        hop1 = MONTHS[(start_m + m1) % 12]
        hop2 = MONTHS[(start_m + m1 + m2) % 12]
        m1_lbl = "1 month" if m1 == 1 else f"{m1} months"
        m2_lbl = "1 more month" if m2 == 1 else f"{m2} more months"
        u_txt = (
            f"Start in {s_mon}, move forward {m1_lbl}, and then move forward {m2_lbl}. "
            "What month of the year do you land in? Reply with the single month name only."
        )
        r_txt = f"Calendar sequence: Start in {s_mon}, advance {m1_lbl}, then advance {m2_lbl}. Final month:"
        raw_specs.append((u_txt, r_txt, hop1, hop2))

    items: list[BiophysicalItem] = []
    for idx, (u_txt, r_txt, hop1, hop2) in enumerate(raw_specs):
        # Match foil within the same half (weekdays 0..19, months 20..39)
        block_base = 0 if idx < 20 else 20
        foil_idx = block_base + ((idx - block_base + 3) % 20)
        while raw_specs[foil_idx][3] in (hop2, hop1):
            foil_idx = block_base + ((foil_idx - block_base + 1) % 20)
        foil_u, foil_r, _fh1, foil_ans = raw_specs[foil_idx]
        items.append(
            BiophysicalItem(
                item_id=f"cal2s_{idx:02d}",
                regime="calendar_2step",
                user_prompt=u_txt,
                raw_prompt=r_txt,
                hop1_bridge=hop1,
                target_answer=hop2,
                foil_answer=foil_ans,
                foil_user_prompt=foil_u,
                foil_raw_prompt=foil_r,
                note=f"{hop1} -> {hop2}",
            )
        )
    return items


def build_chat_shunt_triplets() -> list[ChatShuntTriplet]:
    """Build 40 matched chat `<|message_model|>` vs `<|content_text|>` vs raw completion triplets.

    Anchors items 0, 1, 2 to the canonical study probes (`"New Delhi"`, `"7"`, `"Saturday"`).
    """
    facts = build_factual_2hop_items()
    arith = build_arithmetic_2step_items()
    cals = build_calendar_2step_items()

    ordered_sources: list[tuple[str, BiophysicalItem]] = [
        ("factual_2hop", facts[0]),     # New Delhi
        ("arithmetic_2step", arith[0]), # 7
        ("calendar_2step", cals[0]),    # Saturday
    ]
    for i in range(1, 14):
        ordered_sources.append(("factual_2hop", facts[i]))
    for i in range(1, 13):
        ordered_sources.append(("arithmetic_2step", arith[i]))
    for i in range(1, 13):
        ordered_sources.append(("calendar_2step", cals[i]))

    assert len(ordered_sources) == 40
    triplets: list[ChatShuntTriplet] = []
    for idx, (dom, src) in enumerate(ordered_sources):
        triplets.append(
            ChatShuntTriplet(
                item_id=f"shunt_{idx:02d}",
                domain=dom,
                user_question=src.user_prompt,
                raw_completion_prompt=src.raw_prompt,
                hop1_bridge=src.hop1_bridge,
                target_answer=src.target_answer,
                foil_answer=src.foil_answer,
            )
        )
    return triplets


def build_phase6_stimulus_suite() -> dict[str, Any]:
    """Return all 160 paired items organized by task regime."""
    factual = build_factual_2hop_items()
    arithmetic = build_arithmetic_2step_items()
    calendar = build_calendar_2step_items()
    triplets = build_chat_shunt_triplets()
    reasoning_120 = [*factual, *arithmetic, *calendar]
    return {
        "factual_2hop": factual,
        "arithmetic_2step": arithmetic,
        "calendar_2step": calendar,
        "reasoning_120": reasoning_120,
        "chat_shunt_triplets": triplets,
    }


def audit_biophysical_stimuli(suite: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run deterministic structural, label, leakage, and zero-em-dash audit on Phase 6 stimuli."""
    if suite is None:
        suite = build_phase6_stimulus_suite()

    errors: list[str] = []
    counts = {
        "factual_2hop": len(suite["factual_2hop"]),
        "arithmetic_2step": len(suite["arithmetic_2step"]),
        "calendar_2step": len(suite["calendar_2step"]),
        "chat_shunt_triplets": len(suite["chat_shunt_triplets"]),
    }
    total_paired = sum(counts.values())
    if total_paired < 120 or any(v != 40 for v in counts.values()):
        errors.append(f"Expected 40 items in each of 4 regimes (160 total), got {counts}")

    seen_ids: set[str] = set()
    for reg in ("factual_2hop", "arithmetic_2step", "calendar_2step"):
        for it in suite[reg]:
            if it.item_id in seen_ids:
                errors.append(f"Duplicate item_id: {it.item_id}")
            seen_ids.add(it.item_id)
            if it.hop1_bridge == it.target_answer:
                errors.append(f"{it.item_id}: hop1_bridge == target_answer ({it.target_answer})")
            if it.foil_answer in (it.target_answer, it.hop1_bridge):
                errors.append(f"{it.item_id}: foil_answer collides with target/hop1 ({it.foil_answer})")
            # Ensure intermediate bridge entity is unspoken in factual_2hop prompts
            if reg == "factual_2hop" and it.hop1_bridge.lower() in it.user_prompt.lower():
                errors.append(f"{it.item_id}: unspoken hop1_bridge leaked in user_prompt ({it.hop1_bridge})")
            for field_txt in (
                it.user_prompt,
                it.raw_prompt,
                it.hop1_bridge,
                it.target_answer,
                it.foil_answer,
                it.foil_user_prompt,
                it.foil_raw_prompt,
                it.note,
            ):
                if EM_DASH in field_txt:
                    errors.append(f"{it.item_id}: em dash found in stimulus text")

    anchor_targets = [suite["chat_shunt_triplets"][i].target_answer for i in range(3)]
    if anchor_targets != ["New Delhi", "7", "Saturday"]:
        errors.append(f"Canonical anchor targets mismatch: {anchor_targets}")

    for tr in suite["chat_shunt_triplets"]:
        if tr.item_id in seen_ids:
            errors.append(f"Duplicate triplet item_id: {tr.item_id}")
        seen_ids.add(tr.item_id)
        for field_txt in (tr.user_question, tr.raw_completion_prompt, tr.target_answer, tr.hop1_bridge, tr.foil_answer):
            if EM_DASH in field_txt:
                errors.append(f"{tr.item_id}: em dash found in triplet text")

    return {
        "valid": len(errors) == 0,
        "total_paired_items": total_paired,
        "regime_counts": counts,
        "canonical_anchors": anchor_targets,
        "errors": errors,
    }
