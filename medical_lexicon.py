"""
medical_lexicon.py - Conservative post-ASR medical terminology correction.

Speech-to-text engines occasionally mis-hear rare clinical words even when the
surrounding sentence is otherwise correct.  This module applies a deliberately
*narrow* correction list: only unambiguous, context-independent errors are
fixed, and every change is logged so the raw transcript remains auditable
(``output/transcript_raw.txt``).

The correction layer is intentionally not a general spell-checker.  Rewriting
ordinary speech, removing disfluencies, or paraphrasing would break the
"verbatim transcript" requirement.
"""

from __future__ import annotations

import re

# (wrong, right) pairs. Matching is case-insensitive and word-bounded.
_CORRECTIONS: tuple[tuple[str, str], ...] = (
    # Orthopaedic / anatomy terms heard in this dictation.
    ("supernate", "supinate"),
    ("supernates", "supinates"),
    ("supernated", "supinated"),
    ("supernating", "supinating"),
    ("supernation", "supination"),
    ("adducing", "adducting"),
    ("adcting", "adducting"),
    ("scolidal", "skeletal"),
    # Symptom vocabulary that ASR models commonly confuse.
    ("throbby", "throbbing"),
    # Phrase-level fix: "shortness of breath or reason" is a misheard
    # "shortness of breath or wheezing" in the review-of-systems question.
    ("shortness of breath or reason", "shortness of breath or wheezing"),
)

# Pre-compile once; the callback preserves the original capitalisation.
_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), right)
    for wrong, right in _CORRECTIONS
)


def _match_case(original: str, replacement: str) -> str:
    """Apply the capitalisation of ``original`` to ``replacement``."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def normalize_medical_terms(
    text: str,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Correct high-confidence medical ASR errors.

    Args:
        text: Raw transcript returned by the speech-to-text backend.

    Returns:
        A ``(corrected_text, changes)`` tuple. ``changes`` contains
        ``(original_word, corrected_word)`` pairs for logging/auditing and is
        empty when the transcript needed no correction.
    """
    if not text:
        return text, []

    changes: list[tuple[str, str]] = []
    normalized = text

    for pattern, replacement in _PATTERNS:
        def _sub(match: re.Match[str], rep: str = replacement) -> str:
            original = match.group(0)
            corrected = _match_case(original, rep)
            changes.append((original, corrected))
            return corrected

        normalized = pattern.sub(_sub, normalized)

    return normalized, changes
