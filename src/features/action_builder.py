from __future__ import annotations

from src.parsing.mcp_parser import (
    direction_to_court_side,
    natural_court_side_for_stroke,
    recipient_wing_from_direction,
)


def infer_side_proxy(
    previous_direction: int | None,
    hitter_hand: str | None,
    current_family: str,
) -> tuple[str | None, str, str | None]:
    incoming_wing = recipient_wing_from_direction(previous_direction, hitter_hand)
    if incoming_wing is None:
        return None, "low", None

    if incoming_wing == "forehand" and current_family == "FH":
        return natural_court_side_for_stroke(hitter_hand, current_family), "high", incoming_wing
    if incoming_wing == "backhand" and current_family == "BH":
        return natural_court_side_for_stroke(hitter_hand, current_family), "high", incoming_wing
    return None, "low", incoming_wing


def classify_forehand_action(current_direction: int | None, side_proxy: str | None) -> str | None:
    if current_direction is None or side_proxy not in {"deuce", "ad"}:
        return None
    if current_direction == 2:
        return "BODY"
    target_side = direction_to_court_side(current_direction)
    if target_side is None:
        return None
    return "CC" if target_side == side_proxy else "DTL"

