"""
transcriber.py - Audio Transcription Module

Provides two transcription backends:
1. WhisperTranscriber  — Local, privacy-centric (OpenAI Whisper / faster-whisper)
2. GeminiTranscriber   — Cloud API fallback (Google Gemini multimodal)

The factory function `get_transcriber()` picks the best available backend.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Abstract Base
# ═══════════════════════════════════════════════════════════════

class BaseTranscriber(ABC):
    """Abstract base class for audio transcription."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (.mp3, .wav, etc.)

        Returns:
            The verbatim transcript as a string.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the transcription backend."""
        ...


# ═══════════════════════════════════════════════════════════════
# Backend 1 — Local Whisper (privacy-centric, no data leaves machine)
# ═══════════════════════════════════════════════════════════════

class WhisperTranscriber(BaseTranscriber):
    """
    Transcribes audio using faster-whisper (CTranslate2-optimised Whisper).

    Why faster-whisper over openai-whisper?
    • 4× faster on CPU, 2× faster on GPU
    • Lower memory footprint (INT8 quantisation)
    • Same accuracy as the original OpenAI Whisper models

    Privacy: All processing happens locally — no audio data is sent
    to any external service. This is critical for HIPAA-sensitive
    medical dictations.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        beam_size: int = 5,
        language: str | None = "en",
    ):
        from faster_whisper import WhisperModel

        # Resolve device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "int8" if device == "cpu" else "float16"

        logger.info(
            "Loading Whisper model '%s' on %s (compute_type=%s)",
            model_size, device, compute_type,
        )

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        self._beam_size = beam_size
        self._language = language
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"faster-whisper ({self._model_size})"

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe audio using the local Whisper model.

        Uses beam search decoding for better accuracy with
        medical terminology.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing '%s' with %s ...", audio_path.name, self.name)

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=True,          # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        logger.info(
            "Detected language: %s (probability: %.2f)",
            info.language, info.language_probability,
        )

        # Collect all segment texts
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())

        transcript = " ".join(transcript_parts)
        logger.info("Transcription complete — %d characters", len(transcript))
        return transcript


# ═══════════════════════════════════════════════════════════════
# Backend 2 — Gemini API (cloud fallback for hardware-constrained setups)
# ═══════════════════════════════════════════════════════════════

class GeminiTranscriber(BaseTranscriber):
    """
    Transcribes audio using Google Gemini's multimodal capabilities.

    Used as a fallback when local hardware cannot run Whisper efficiently.
    The Gemini model receives the raw audio and produces a verbatim
    transcript with strong medical terminology handling.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        from google import genai

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it as an environment variable."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"Gemini API ({self._model})"

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe audio by uploading to Gemini and prompting for
        verbatim transcription.
        """
        from google.genai import types

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Uploading '%s' to Gemini Files API...", audio_path.name)

        # Upload the audio file
        uploaded_file = self._client.files.upload(file=str(audio_path))
        logger.info("Upload complete: %s", uploaded_file.name)

        # Prompt for verbatim medical transcription
        prompt = (
            "Transcribe this medical dictation audio verbatim. "
            "Output ONLY the transcript text with no labels, headers, or formatting. "
            "Preserve all medical terminology exactly as spoken. "
            "Include filler words (um, uh) and natural speech patterns exactly as they occur. "
            "Do not add punctuation that changes meaning. "
            "Do not summarize or paraphrase — produce the complete word-for-word transcript."
        )

        logger.info("Requesting transcription from %s ...", self._model)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri=uploaded_file.uri,
                            mime_type="audio/mpeg",
                        )
                    ),
                    types.Part(text=prompt),
                ])
            ],
        )

        transcript = response.text.strip()
        logger.info("Transcription complete — %d characters", len(transcript))

        # Clean up the uploaded file
        try:
            self._client.files.delete(name=uploaded_file.name)
            logger.info("Cleaned up uploaded file: %s", uploaded_file.name)
        except Exception as e:
            logger.warning("Could not delete uploaded file: %s", e)

        return transcript


# ═══════════════════════════════════════════════════════════════
# Factory — auto-select the best available backend
# ═══════════════════════════════════════════════════════════════

def get_transcriber(
    prefer_local: bool = True,
    whisper_model: str = "base",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-3.5-flash-lite",
) -> BaseTranscriber:
    """
    Factory function that returns the best available transcriber.

    Priority:
    1. faster-whisper (local, privacy-centric) — if installed
    2. Gemini API (cloud fallback) — if API key is available

    Args:
        prefer_local:    Try local Whisper first (default True).
        whisper_model:   Whisper model size (tiny/base/small/medium/large-v3).
        gemini_api_key:  Google Gemini API key.
        gemini_model:    Gemini model for transcription.

    Returns:
        A BaseTranscriber instance.
    """
    if prefer_local:
        try:
            transcriber = WhisperTranscriber(model_size=whisper_model)
            logger.info("Using local transcriber: %s", transcriber.name)
            return transcriber
        except ImportError:
            logger.warning(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
        except Exception as e:
            logger.warning("Failed to initialise Whisper: %s", e)

    # Fallback to Gemini API
    if gemini_api_key:
        transcriber = GeminiTranscriber(
            api_key=gemini_api_key,
            model=gemini_model,
        )
        logger.info("Using cloud transcriber: %s", transcriber.name)
        return transcriber

    raise RuntimeError(
        "No transcription backend available. Either:\n"
        "  1. Install faster-whisper: pip install faster-whisper\n"
        "  2. Set GEMINI_API_KEY environment variable for Gemini API fallback"
    )
