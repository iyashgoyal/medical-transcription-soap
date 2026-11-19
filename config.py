"""
config.py - Central configuration for the Medical Transcription & SOAP Pipeline.

Every setting can be overridden with an environment variable, which keeps the
code deployable without edits while still providing sensible defaults for the
assignment's sample dictation.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "audio"

# SOAP_OUTPUT_DIR lets you write a run to another folder without touching the
# committed deliverables in output/ (useful for tests and repeats).
OUTPUT_DIR = Path(os.environ.get("SOAP_OUTPUT_DIR", PROJECT_ROOT / "output"))

DEFAULT_AUDIO_FILE = AUDIO_DIR / "sample_dictation.mp3"

TRANSCRIPT_OUTPUT = OUTPUT_DIR / "transcript.txt"
TRANSCRIPT_RAW_OUTPUT = OUTPUT_DIR / "transcript_raw.txt"
SOAP_NOTE_OUTPUT = OUTPUT_DIR / "soap_note.json"
SOAP_NOTE_MD_OUTPUT = OUTPUT_DIR / "soap_note.md"

# ──────────────────────────────────────────────────────────────
# Local Whisper (Speech-to-Text) settings
# ──────────────────────────────────────────────────────────────
# Model sizes: tiny, base, small, medium, large-v3.
# "small" is a good CPU compromise: noticeably better on medical vocabulary
# than "base" while remaining practical for 10-15 minute dictations.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")

# "cpu", "cuda", or "auto" (detect via CTranslate2).
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")

# Beam search width. Higher = more accurate, slower.
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

# Language hint. "en" avoids accidental language switching on short clips.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en") or None

# ──────────────────────────────────────────────────────────────
# Gemini (cloud fallback + SOAP structuring) settings
# ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_TRANSCRIPTION_MODEL = os.environ.get(
    "GEMINI_TRANSCRIPTION_MODEL", GEMINI_MODEL
)

# Low temperature keeps the note reproducible and limits creative additions.
GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))

# Retry policy for transient API/network failures.
API_MAX_RETRIES = int(os.environ.get("API_MAX_RETRIES", "3"))
