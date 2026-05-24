from __future__ import annotations

import math
from typing import Any


SERVE_DIRECTION_DIGITS = {"0", "4", "5", "6"}
SERVE_FAULT_CODES = {"n", "w", "d", "x", "g", "e", "!"}
PLACEHOLDER_CODES = {"S", "R"}

FOREHAND_CODES = {"f", "r", "v", "o", "u", "l", "h", "j"}
BACKHAND_CODES = {"b", "s", "z", "p", "y", "m", "i", "k"}
OTHER_CODES = {"t", "q"}
RALLY_SHOT_CODES = FOREHAND_CODES | BACKHAND_CODES | OTHER_CODES

GROUNDSTROKE_CODES = {"f", "b", "r", "s"}
VOLLEY_CODES = {"v", "z", "o", "p", "j", "k", "h", "i"}
LOB_CODES = {"l", "m"}
DROP_CODES = {"u", "y"}

RALLY_ERROR_CODES = {"n", "w", "d", "x", "!", "e", "g"}
RALLY_ENDING_CODES = {
    "*": "winner",
    "@": "unforced_error",
    "#": "forced_error",
}
SHOT_MODIFIERS = {"+", ";", "^", "-", "="}


def clean_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_hand(value: Any, default: str = "R") -> str:
    text = clean_cell(value)
    if text is None:
        return default
    return text.upper()


def family_from_raw_code(raw_code: str) -> str:
    if raw_code in FOREHAND_CODES:
        return "FH"
    if raw_code in BACKHAND_CODES:
        return "BH"
    return "OTHER"


def spin_from_raw_code(raw_code: str) -> str:
    if raw_code in {"r", "s"}:
        return "SLICE"
    if raw_code in {"f", "b"}:
        return "DRIVE"
    return "NA"


def stroke_type_from_raw_code(raw_code: str) -> str:
    mapping = {
        "f": "forehand",
        "b": "backhand",
        "r": "forehand_slice",
        "s": "backhand_slice",
        "v": "forehand_volley",
        "z": "backhand_volley",
        "o": "forehand_overhead",
        "p": "backhand_overhead",
        "u": "forehand_drop",
        "y": "backhand_drop",
        "l": "forehand_lob",
        "m": "backhand_lob",
        "h": "forehand_half_volley",
        "i": "backhand_half_volley",
        "j": "forehand_swinging_volley",
        "k": "backhand_swinging_volley",
        "t": "trick_shot",
        "q": "unknown_shot",
    }
    return mapping.get(raw_code, "unknown")


def is_groundstroke_code(raw_code: str) -> bool:
    return raw_code in GROUNDSTROKE_CODES


def is_forehand_groundstroke(raw_code: str) -> bool:
    return raw_code == "f"


def direction_to_court_side(direction: int | None) -> str | None:
    if direction == 1:
        return "deuce"
    if direction == 2:
        return "middle"
    if direction == 3:
        return "ad"
    return None


def recipient_wing_from_direction(direction: int | None, recipient_hand: str | None) -> str | None:
    if direction not in {1, 3}:
        return None
    hand = normalize_hand(recipient_hand)
    if direction == 1:
        return "forehand" if hand == "R" else "backhand"
    return "backhand" if hand == "R" else "forehand"


def natural_court_side_for_stroke(hitter_hand: str | None, stroke_family: str) -> str | None:
    hand = normalize_hand(hitter_hand)
    if stroke_family == "FH":
        return "deuce" if hand == "R" else "ad"
    if stroke_family == "BH":
        return "ad" if hand == "R" else "deuce"
    return None


def infer_net_state(raw_code: str, position: str | None) -> str:
    if position == "net" or raw_code in VOLLEY_CODES:
        return "net"
    if position == "baseline" or raw_code in GROUNDSTROKE_CODES | LOB_CODES | DROP_CODES | OTHER_CODES:
        return "baseline"
    return "unknown"


def choose_active_string(first_serve: Any, second_serve: Any) -> tuple[str | None, int]:
    second = clean_cell(second_serve)
    if second is not None:
        return second, 2
    return clean_cell(first_serve), 1


def parse_serve_from_active_string(active_string: str | None) -> dict[str, Any]:
    text = clean_cell(active_string)
    if text is None or text in PLACEHOLDER_CODES:
        return {
            "serve_direction": None,
            "serve_number": None,
            "let_count": 0,
            "serve_and_volley": False,
            "suffix": "",
            "rally_start_index": 0,
            "outcome": "missing",
            "error_code": None,
        }

    i = 0
    let_count = 0
    while i < len(text) and text[i] == "c":
        let_count += 1
        i += 1

    if i >= len(text) or text[i] not in SERVE_DIRECTION_DIGITS:
        return {
            "serve_direction": None,
            "serve_number": None,
            "let_count": let_count,
            "serve_and_volley": False,
            "suffix": text[i:],
            "rally_start_index": i,
            "outcome": "unparsed",
            "error_code": None,
        }

    serve_direction = int(text[i])
    i += 1
    serve_and_volley = False
    suffix_chars: list[str] = []

    while i < len(text) and text[i] not in RALLY_SHOT_CODES:
        if text[i] == "+":
            serve_and_volley = True
        suffix_chars.append(text[i])
        i += 1

    suffix = "".join(suffix_chars)
    error_code = next((char for char in suffix if char in SERVE_FAULT_CODES), None)

    if i < len(text):
        outcome = "in_play"
    elif "*" in suffix:
        outcome = "ace"
    elif "#" in suffix:
        outcome = "forced_return_error"
    elif "@" in suffix:
        outcome = "unforced_return_error"
    elif error_code is not None:
        outcome = "serve_error"
    else:
        outcome = "unknown_terminal"

    return {
        "serve_direction": serve_direction,
        "let_count": let_count,
        "serve_and_volley": serve_and_volley,
        "suffix": suffix,
        "rally_start_index": i,
        "outcome": outcome,
        "error_code": error_code,
    }


def parse_rally(rally_str: str | None) -> list[dict[str, Any]]:
    text = clean_cell(rally_str)
    if text is None or text in PLACEHOLDER_CODES:
        return []

    serve_info = parse_serve_from_active_string(text)
    i = serve_info["rally_start_index"]
    shots: list[dict[str, Any]] = []

    while i < len(text):
        raw_code = text[i]

        if raw_code not in RALLY_SHOT_CODES:
            i += 1
            continue

        shot = {
            "raw_code": raw_code,
            "family": family_from_raw_code(raw_code),
            "spin": spin_from_raw_code(raw_code),
            "stroke_type": stroke_type_from_raw_code(raw_code),
            "dir": None,
            "depth": None,
            "approach": False,
            "net_cord": False,
            "stop_volley": False,
            "position": None,
            "ending": None,
            "error_type": None,
        }
        i += 1

        while i < len(text):
            char = text[i]

            if char == "+":
                shot["approach"] = True
                i += 1
                continue
            if char == ";":
                shot["net_cord"] = True
                i += 1
                continue
            if char == "^":
                shot["stop_volley"] = True
                i += 1
                continue
            if char == "-":
                shot["position"] = "net"
                i += 1
                continue
            if char == "=":
                shot["position"] = "baseline"
                i += 1
                continue
            if shot["dir"] is None and char in {"0", "1", "2", "3"}:
                shot["dir"] = int(char)
                i += 1
                continue
            if shot["depth"] is None and char in {"0", "7", "8", "9"}:
                shot["depth"] = int(char)
                i += 1
                continue
            if shot["error_type"] is None and char in RALLY_ERROR_CODES:
                shot["error_type"] = char
                i += 1
                continue
            if char in RALLY_ENDING_CODES:
                shot["ending"] = RALLY_ENDING_CODES[char]
                i += 1
                break
            break

        shots.append(shot)
        if shots[-1]["ending"] is not None:
            break

    return shots


def parse_point_row(row: dict[str, Any]) -> dict[str, Any]:
    active_string, serve_number = choose_active_string(row.get("1st"), row.get("2nd"))
    serve_info = parse_serve_from_active_string(active_string)
    serve_info["serve_number"] = serve_number
    rally_shots = parse_rally(active_string)

    return {
        "active_string": active_string,
        "serve_number": serve_number,
        "serve_info": serve_info,
        "rally_shots": rally_shots,
        "is_placeholder": active_string in PLACEHOLDER_CODES,
        "is_parseable": active_string is not None and serve_info["serve_direction"] is not None,
    }
