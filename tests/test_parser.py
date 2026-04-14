from src.parsing.mcp_parser import parse_rally, parse_serve_from_active_string


def test_parse_rally_handles_terminal_error():
    parsed = parse_rally("5f27f27f3w@")
    assert len(parsed) == 3
    assert parsed[-1]["dir"] == 3
    assert parsed[-1]["error_type"] == "w"
    assert parsed[-1]["ending"] == "unforced_error"


def test_parse_rally_handles_modifiers_and_forced_error():
    parsed = parse_rally("4f28b+28b1v2d#")
    assert len(parsed) == 4
    assert parsed[1]["approach"] is True
    assert parsed[-1]["ending"] == "forced_error"
    assert parsed[-1]["error_type"] == "d"


def test_parse_serve_recognizes_serve_only_outcomes():
    assert parse_serve_from_active_string("4*")["outcome"] == "ace"
    assert parse_serve_from_active_string("4w")["outcome"] == "serve_error"
    assert parse_serve_from_active_string("6#")["outcome"] == "forced_return_error"

