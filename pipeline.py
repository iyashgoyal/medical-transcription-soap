"""
pipeline.py - End-to-end medical transcription and SOAP note generation.

Usage:
    python pipeline.py
    python pipeline.py path/to/dictation.mp3
    python pipeline.py --backend gemini --output-dir runs/experiment-1

Steps:
    1. Transcribe the dictation audio (local Whisper or Gemini fallback).
    2. Apply conservative medical-term corrections; keep the raw ASR text.
    3. Generate a structured SOAP note and run cross-contamination checks.
    4. Write transcript.txt, transcript_raw.txt, soap_note.json,
       and soap_note.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import (
    API_MAX_RETRIES,
    DEFAULT_AUDIO_FILE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_TRANSCRIPTION_MODEL,
    OUTPUT_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)
from medical_lexicon import normalize_medical_terms
from soap_generator import SOAPNote, SOAPNoteGenerator
from transcriber import get_transcriber

logger = logging.getLogger("pipeline")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    audio_path: str | Path | None = None,
    *,
    backend: str = "auto",
    whisper_model: str = WHISPER_MODEL_SIZE,
    output_dir: str | Path | None = None,
    normalize_terms: bool = True,
) -> SOAPNote:
    """
    Run transcription, normalisation, and SOAP generation.

    Args:
        audio_path: Audio file. Defaults to audio/sample_dictation.mp3.
        backend: "auto", "local", or "gemini" (passed to get_transcriber).
        whisper_model: faster-whisper model size when the local backend is used.
        output_dir: Override the output folder (default: ./output).
        normalize_terms: Apply the conservative medical-term correction list.

    Returns:
        The generated SOAPNote.
    """
    audio_path = Path(audio_path or DEFAULT_AUDIO_FILE)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = out_dir / "transcript.txt"
    raw_transcript_path = out_dir / "transcript_raw.txt"
    soap_json_path = out_dir / "soap_note.json"
    soap_md_path = out_dir / "soap_note.md"

    logger.info("=" * 64)
    logger.info("Medical Transcription & SOAP Note Pipeline")
    logger.info("=" * 64)
    logger.info("Audio:      %s", audio_path)
    logger.info("Output dir: %s", out_dir)
    logger.info("Backend:    %s", backend)

    # ── Step 1: Transcription ─────────────────────────────────
    logger.info("")
    logger.info("STEP 1/2 - Audio transcription")

    transcriber = get_transcriber(
        backend=backend,
        whisper_model=whisper_model,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_TRANSCRIPTION_MODEL,
        whisper_device=WHISPER_DEVICE,
        whisper_beam_size=WHISPER_BEAM_SIZE,
        whisper_language=WHISPER_LANGUAGE,
        max_retries=API_MAX_RETRIES,
    )
    logger.info("Transcription backend: %s", transcriber.name)

    raw_transcript = transcriber.transcribe(audio_path)
    raw_transcript_path.write_text(raw_transcript, encoding="utf-8")
    logger.info("Raw transcript saved -> %s", raw_transcript_path)

    transcript = raw_transcript
    if normalize_terms:
        transcript, changes = normalize_medical_terms(raw_transcript)
        if changes:
            logger.info("Applied %d medical-term correction(s):", len(changes))
            for original, corrected in changes:
                logger.info("  %s -> %s", original, corrected)
        else:
            logger.info("No medical-term corrections were needed.")
    else:
        logger.info("Medical-term normalisation disabled.")

    transcript_path.write_text(transcript, encoding="utf-8")
    logger.info("Transcript saved -> %s", transcript_path)

    # ── Step 2: SOAP note ─────────────────────────────────────
    logger.info("")
    logger.info("STEP 2/2 - SOAP note generation")

    generator = SOAPNoteGenerator(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        max_retries=API_MAX_RETRIES,
    )
    note = generator.generate(transcript)

    soap_json_path.write_text(note.to_json(), encoding="utf-8")
    soap_md_path.write_text(note.to_markdown(), encoding="utf-8")
    logger.info("SOAP note (JSON) saved -> %s", soap_json_path)
    logger.info("SOAP note (Markdown) saved -> %s", soap_md_path)

    logger.info("")
    logger.info("=" * 64)
    logger.info("Pipeline complete")
    logger.info("=" * 64)

    print()
    print(note.to_markdown())
    return note


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe medical dictation and generate a SOAP note.",
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to the dictation audio (default: audio/sample_dictation.mp3)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "local", "gemini"),
        default="auto",
        help="Transcription backend. 'auto' prefers local Whisper, then Gemini.",
    )
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL_SIZE,
        help=f"faster-whisper model size (default: {WHISPER_MODEL_SIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for transcript and SOAP outputs (default: output/).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip the conservative medical-term correction step.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)

    if not GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY is not set. SOAP generation requires it. "
            "Set it before running: $env:GEMINI_API_KEY='...' (PowerShell)."
        )
        return 2

    try:
        run_pipeline(
            audio_path=args.audio,
            backend=args.backend,
            whisper_model=args.whisper_model,
            output_dir=args.output_dir,
            normalize_terms=not args.no_normalize,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
