"""Unit tests for the SOAP parsing and validation logic (no API calls)."""

import json

from soap_generator import SOAPNote, SOAPNoteGenerator


def test_to_markdown_contains_all_sections():
    note = SOAPNote("s", "o", "a", "p")
    markdown = note.to_markdown()
    for heading in ("# SOAP Note", "## Subjective", "## Objective", "## Assessment", "## Plan"):
        assert heading in markdown


def test_extract_json_from_fenced_response():
    raw = """```json
    {"subjective": "S", "objective": "O", "assessment": "A", "plan": "P"}
    ```"""
    data = SOAPNoteGenerator._extract_json(raw)
    assert data == {
        "subjective": "S",
        "objective": "O",
        "assessment": "A",
        "plan": "P",
    }


def test_extract_json_handles_trailing_commentary_and_braces_in_strings():
    raw = (
        'Here is the note: {"subjective": "Pain {sharp}", "objective": "O", '
        '"assessment": "A", "plan": "P"} Hope this helps.'
    )
    data = SOAPNoteGenerator._extract_json(raw)
    assert data is not None
    assert data["subjective"] == "Pain {sharp}"


def test_extract_json_returns_none_for_invalid_text():
    assert SOAPNoteGenerator._extract_json("no json here") is None


def test_validation_passes_for_clean_note():
    note = SOAPNote(
        subjective="- Pain on the outside of the right elbow for several months.",
        objective="- Tenderness over the lateral epicondyle. No swelling.",
        assessment="- Lateral epicondylitis.",
        plan="- Physical therapy and activity modification.",
    )
    assert SOAPNoteGenerator._validate_soap(note) == []


def test_validation_flags_objective_language_in_subjective():
    note = SOAPNote(
        subjective="- On examination there is tenderness over the lateral epicondyle.",
        objective="- Tenderness over the lateral epicondyle.",
        assessment="- Lateral epicondylitis.",
        plan="- Physical therapy.",
    )
    issues = SOAPNoteGenerator._validate_soap(note)
    assert any("Objective-style language in Subjective" in issue for issue in issues)


def test_validation_flags_subjective_language_in_objective():
    note = SOAPNote(
        subjective="- Pain in the right elbow.",
        objective="- Patient reports pain rated 4/10.",
        assessment="- Lateral epicondylitis.",
        plan="- Physical therapy.",
    )
    issues = SOAPNoteGenerator._validate_soap(note)
    assert any("Subjective-style language in Objective" in issue for issue in issues)


def test_note_round_trips_through_json():
    note = SOAPNote("S", "O", "A", "P")
    assert json.loads(note.to_json()) == note.to_dict()
