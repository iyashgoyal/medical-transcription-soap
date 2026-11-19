"""
transcriber.py - Audio transcription module.

Two interchangeable speech-to-text backends:

1. ``WhisperTranscriber`` - local faster-whisper (privacy-preserving, preferred).
2. ``GeminiTranscriber``  - Gemini multimodal API (fallback for CPU-only or
   dependency-constrained environments).

``get_transcriber()`` selects a backend based on availability and an explicit
``backend`` argument ("auto", "local", or "gemini").
"""

from __future__ import annotations

import logging
import mimetypes
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def call_with_retries(
    operation: Callable[[], T],
    *,
    description: str,
    max_retries: int = 3,
    base_delay: float = 1.5,
) -> T:
    """
    Run ``operation`` with exponential backoff on transient failures.

    Used around network calls so a single rate-limit or timeout does not lose a
    13-minute transcription job.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - deliberately retry all API errors
            last_error = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description, attempt, max_retries, exc, delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"{description} failed after {max_retries} attempts") from last_error


def _audio_mime_type(path: Path) -> str:
    """Infer a Gemini-compatible MIME type from the file extension."""
    mime, _ = mimetypes.guess_type(path.name)
    if mime and mime.startswith("audio/"):
        return mime
    # Gemini accepts a small set of audio MIME types; MP3 is the safe default
    # for uncommon extensions such as .m4a on older Windows installs.
    return "audio/mpeg"


# ═══════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════

class BaseTranscriber(ABC):
    """Abstract base class for audio transcription backends."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> str:
        """Return a verbatim transcript for ``audio_path``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logs and reports."""


# ═══════════════════════════════════════════════════════════════
# Backend 1 - local faster-whisper
# ═══════════════════════════════════════════════════════════════

class WhisperTranscriber(BaseTranscriber):
    """
    Local transcription with faster-whisper (CTranslate2 Whisper).

    Why this is the preferred backend:
    * Audio never leaves the machine, which matters for PHI/HIPAA workflows.
    * CTranslate2 INT8 inference is roughly 4x faster than openai-whisper on
      CPU and uses less memory, with the same model weights and accuracy.
    * No per-minute API cost, no network dependency.
    """

    VALID_MODELS = {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3",
        "distil-large-v3",
    }

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        beam_size: int = 5,
        language: str | None = "en",
    ):
        from faster_whisper import WhisperModel

        if model_size not in self.VALID_MODELS:
            raise ValueError(
                f"Unsupported Whisper model '{model_size}'. "
                f"Choose from: {', '.join(sorted(self.VALID_MODELS))}"
            )

        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # pragma: no cover - depends on local hardware
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper '%s' on %s (compute_type=%s)...",
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
        self._device = device

    @property
    def name(self) -> str:
        return f"faster-whisper ({self._model_size}, {self._device})"

    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe ``audio_path`` with beam search and VAD.

        VAD removes long silences and reduces hallucinated text between turns.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing '%s' with %s...", audio_path.name, self.name)

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        logger.info(
            "Detected language: %s (probability %.2f)",
            info.language, info.language_probability,
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        )
        logger.info("Transcription complete - %d characters", len(transcript))
        return transcript


# ═══════════════════════════════════════════════════════════════
# Backend 2 - Gemini multimodal API
# ═══════════════════════════════════════════════════════════════

class GeminiTranscriber(BaseTranscriber):
    """
    Transcribe audio with Gemini's native audio understanding.

    The transcript is requested verbatim (including disfluencies and repeated
    words) so it can be compared fairly with a human ground-truth transcript.
    """

    TRANSCRIPTION_PROMPT = (
        "Transcribe this medical dictation audio verbatim. "
        "Output only the transcript text: no headings, labels, speaker names, "
        "timestamps, markdown, or commentary. "
        "Preserve every word exactly as spoken, including filler words (um, uh), "
        "false starts, and repetitions. "
        "Pay particular attention to medical terminology, anatomy, laterality "
        "(left/right), medication names, and numbers; spell them correctly. "
        "Do not summarize, paraphrase, clean up, or add information."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        max_retries: int = 3,
    ):
        from google import genai

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for the Gemini backend. "
                "Set it as an environment variable."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return f"Gemini API ({self._model})"

    def transcribe(self, audio_path: str | Path) -> str:
        from google.genai import types

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Uploading '%s' to the Gemini Files API...", audio_path.name)
        uploaded = call_with_retries(
            lambda: self._client.files.upload(file=str(audio_path)),
            description="Audio upload",
            max_retries=self._max_retries,
        )
        logger.info("Upload complete: %s", uploaded.name)

        # Files are processed asynchronously; wait until they are usable.
        for _ in range(60):
            state = getattr(getattr(uploaded, "state", None), "name", "")
            if state != "PROCESSING":
                break
            time.sleep(2)
            uploaded = self._client.files.get(name=uploaded.name)

        if getattr(getattr(uploaded, "state", None), "name", "") == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded audio file")

        try:
            logger.info("Requesting transcription from %s...", self._model)

            response = call_with_retries(
                lambda: self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part(
                                    file_data=types.FileData(
                                        file_uri=uploaded.uri,
                                        mime_type=_audio_mime_type(audio_path),
                                    )
                                ),
                                types.Part(text=self.TRANSCRIPTION_PROMPT),
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(temperature=0.0),
                ),
                description="Gemini transcription",
                max_retries=self._max_retries,
            )

            transcript = (response.text or "").strip()
            if not transcript:
                raise RuntimeError("Gemini returned an empty transcript")

            logger.info("Transcription complete - %d characters", len(transcript))
            return transcript
        finally:
            try:
                self._client.files.delete(name=uploaded.name)
                logger.info("Deleted uploaded file: %s", uploaded.name)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Could not delete uploaded file: %s", exc)


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

def get_transcriber(
    backend: str = "auto",
    whisper_model: str = "small",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-3.5-flash-lite",
    whisper_device: str = "auto",
    whisper_beam_size: int = 5,
    whisper_language: str | None = "en",
    max_retries: int = 3,
) -> BaseTranscriber:
    """
    Build a transcriber.

    Args:
        backend: "auto" (local first, cloud fallback), "local" (Whisper only),
            or "gemini" (cloud only).
        whisper_model: faster-whisper model size.
        gemini_api_key: key for the fallback/cloud backend.
        gemini_model: Gemini model used for transcription.
        whisper_device: "cpu", "cuda", or "auto".
        whisper_beam_size: beam width for Whisper decoding.
        whisper_language: language hint, or None for auto-detection.
        max_retries: retry count for cloud calls.

    Raises:
        ValueError: for an unknown backend name.
        RuntimeError: when the requested/available backend cannot start.
    """
    backend = backend.lower()
    if backend not in {"auto", "local", "gemini"}:
        raise ValueError("backend must be one of: auto, local, gemini")

    if backend in {"auto", "local"}:
        try:
            transcriber = WhisperTranscriber(
                model_size=whisper_model,
                device=whisper_device,
                beam_size=whisper_beam_size,
                language=whisper_language,
            )
            logger.info("Using local transcriber: %s", transcriber.name)
            return transcriber
        except ImportError:
            if backend == "local":
                raise RuntimeError(
                    "faster-whisper is not installed. "
                    "Install it with: pip install faster-whisper"
                ) from None
            logger.warning(
                "faster-whisper is not installed - falling back to Gemini. "
                "Install it with: pip install faster-whisper"
            )
        except Exception as exc:
            if backend == "local":
                raise RuntimeError(f"Could not start local Whisper: {exc}") from exc
            logger.warning("Local Whisper failed to start (%s) - using Gemini", exc)

    if gemini_api_key:
        transcriber = GeminiTranscriber(
            api_key=gemini_api_key,
            model=gemini_model,
            max_retries=max_retries,
        )
        logger.info("Using cloud transcriber: %s", transcriber.name)
        return transcriber

    raise RuntimeError(
        "No transcription backend is available. Either:\n"
        "  1. Install faster-whisper: pip install faster-whisper\n"
        "  2. Set GEMINI_API_KEY for the Gemini fallback"
    )
