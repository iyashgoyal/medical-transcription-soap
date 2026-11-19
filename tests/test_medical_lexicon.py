"""Tests for the conservative medical-term normalisation layer."""

from medical_lexicon import normalize_medical_terms


def test_corrects_known_orthopaedic_terms():
    text = "How about when you pronate and supernate your forearms? Adducing the shoulder? Scolidal issues?"
    corrected, changes = normalize_medical_terms(text)
    assert "supinate" in corrected.lower()
    assert "adducting" in corrected.lower()
    assert "skeletal" in corrected.lower()
    assert len(changes) == 3


def test_preserves_capitalisation():
    corrected, _ = normalize_medical_terms("Supernate and SUPERNATION")
    assert "Supinate" in corrected
    assert "SUPINATION" in corrected


def test_leaves_unrelated_text_untouched():
    text = "The patient reports right elbow pain rated 4/10."
    corrected, changes = normalize_medical_terms(text)
    assert corrected == text
    assert changes == []


def test_word_boundaries_prevent_partial_replacements():
    corrected, _ = normalize_medical_terms("supersupernate")
    assert corrected == "supersupernate"


def test_corrects_contextual_phrase_and_symptom_terms():
    text = "Any coughing, shortness of breath or reason? Is the pain sharp, throbby, achy?"
    corrected, changes = normalize_medical_terms(text)
    assert "shortness of breath or wheezing" in corrected
    assert "throbbing" in corrected
    assert len(changes) == 2
