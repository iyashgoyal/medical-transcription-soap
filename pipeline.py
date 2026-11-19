"""
pipeline.py - Main Pipeline Orchestrator

End-to-end pipeline that:
1. Transcribes medical dictation audio → text
2. Generates a structured SOAP note from the transcript
3. Saves all outputs (transcript, JSON, Markdown)
"""

import logging
import sys
from pathlib import Path

from config import (
    DEFAULT_AUDIO_FILE,
    OUTPUT_DIR,
    TRANSCRIPT_OUTPUT,
    SOAP_NOTE_OUTPUT,
    SOAP_NOTE_MD_OUTPUT,
    WHISPER_MODEL_SIZE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TRANSCRIPTION_MODEL,
)
from transcriber import get_transcriber
from soap_generator import SOAPNoteGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    audio_path: str | Path | None = None,
    prefer_local_whisper: bool = True,
) -> None:
    """
    Execute the full medical transcription + SOAP note pipeline.

    Args:
        audio_path:          Path to the dictation audio file.
                             Defaults to audio/sample_dictation.mp3
        prefer_local_whisper: If True, try faster-whisper before Gemini.
    """
    audio_path = Path(audio_path or DEFAULT_AUDIO_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Medical Transcription & SOAP Note Pipeline")
    logger.info("=" * 60)
    logger.info("Audio file: %s", audio_path)
    logger.info("")

    # ── Step 1: Transcription ─────────────────────────────────
    logger.info("━━ STEP 1: Audio Transcription ━━")

    transcriber = get_transcriber(
        prefer_local=prefer_local_whisper,
        whisper_model=WHISPER_MODEL_SIZE,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_TRANSCRIPTION_MODEL,
    )
    logger.info("Backend: %s", transcriber.name)

    transcript = transcriber.transcribe(audio_path)

    # Save transcript
    TRANSCRIPT_OUTPUT.write_text(transcript, encoding="utf-8")
    logger.info("Transcript saved → %s", TRANSCRIPT_OUTPUT)
    logger.info("")

    # ── Step 2: SOAP Note Generation ──────────────────────────
    logger.info("━━ STEP 2: SOAP Note Generation ━━")

    generator = SOAPNoteGenerator(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
    )

    soap_note = generator.generate(transcript)

    # Save JSON output
    SOAP_NOTE_OUTPUT.write_text(soap_note.to_json(), encoding="utf-8")
    logger.info("SOAP note (JSON) saved → %s", SOAP_NOTE_OUTPUT)

    # Save Markdown output
    SOAP_NOTE_MD_OUTPUT.write_text(soap_note.to_markdown(), encoding="utf-8")
    logger.info("SOAP note (Markdown) saved → %s", SOAP_NOTE_MD_OUTPUT)
    logger.info("")

    # ── Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    print(soap_note.to_markdown())


if __name__ == "__main__":
    # Allow overriding the audio file via CLI argument
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(audio_path=audio)
