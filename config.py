"""
Configuration settings for the Medical Transcription & SOAP Note Pipeline.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "audio"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Default audio file
DEFAULT_AUDIO_FILE = AUDIO_DIR / "sample_dictation.mp3"

# Output files
TRANSCRIPT_OUTPUT = OUTPUT_DIR / "transcript.txt"
SOAP_NOTE_OUTPUT = OUTPUT_DIR / "soap_note.json"
SOAP_NOTE_MD_OUTPUT = OUTPUT_DIR / "soap_note.md"

# ──────────────────────────────────────────────────────────────
# Whisper (Local Speech-to-Text) Settings
# ──────────────────────────────────────────────────────────────
# Model sizes: tiny, base, small, medium, large-v3
# Larger models = better accuracy (esp. medical terms) but slower
WHISPER_MODEL_SIZE = "base"

# Computation settings
# "cpu" for CPU-only machines, "cuda" for NVIDIA GPU, "auto" for auto-detect
WHISPER_DEVICE = "auto"

# Beam size for decoding (higher = more accurate but slower)
WHISPER_BEAM_SIZE = 5

# Language (None = auto-detect, "en" = English)
WHISPER_LANGUAGE = "en"

# ──────────────────────────────────────────────────────────────
# Gemini API (Cloud LLM) Settings
# ──────────────────────────────────────────────────────────────
# API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Model for SOAP note generation
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Model for audio transcription (fallback if Whisper is unavailable)
GEMINI_TRANSCRIPTION_MODEL = "gemini-3.5-flash-lite"

# Temperature for SOAP note generation (lower = more deterministic)
GEMINI_TEMPERATURE = 0.2
