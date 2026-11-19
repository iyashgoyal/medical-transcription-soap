"""
soap_generator.py - Convert a medical transcript into a structured SOAP note.

The generator uses Gemini with a constrained system prompt and JSON response
mode.  It also runs a lightweight cross-contamination check that flags
patient-reported language in the Objective section (and clinician-observed
language in Subjective).

Design notes
------------
* Temperature is low (default 0.2) to keep categorisation reproducible.
* The prompt forbids inventing findings; the transcript is the only source.
* Output is validated after parsing so malformed model output cannot silently
  produce an empty note.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("subjective", "objective", "assessment", "plan")


# ═══════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════

@dataclass
class SOAPNote:
    """Structured SOAP note with exactly the four standard sections."""

    subjective: str
    objective: str
    assessment: str
    plan: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        return (
            "# SOAP Note\n\n"
            "## Subjective\n"
            f"{self.subjective.strip()}\n\n"
            "## Objective\n"
            f"{self.objective.strip()}\n\n"
            "## Assessment\n"
            f"{self.assessment.strip()}\n\n"
            "## Plan\n"
            f"{self.plan.strip()}\n"
        )


# ═══════════════════════════════════════════════════════════════
# Prompt
# ═══════════════════════════════════════════════════════════════

SOAP_SYSTEM_PROMPT = """\
You are a clinical documentation assistant. Convert the supplied medical
dictation transcript into a concise, accurate SOAP note.

SOURCE-OF-TRUTH RULE
Use only information stated in the transcript. Never invent vital signs, lab
values, imaging results, medications, dosages, diagnoses, or follow-up
intervals. Preserve laterality (left/right), numbers, units, and time frames
exactly. If a section has no supporting information, write
"Not documented in the provided dictation." rather than guessing.

SECTION DEFINITIONS
1. Subjective (S): What the PATIENT reports - chief complaint, history of
   present illness, symptom quality/severity/onset/duration, aggravating and
   relieving factors, pertinent negatives, past medical/surgical history,
   medications, allergies, social and family history.
   Do NOT put clinician examination findings or test results here.

2. Objective (O): What the CLINICIAN observes, measures, or tests - general
   appearance, inspection, palpation, range-of-motion and strength testing,
   vital signs, imaging, and laboratory results. In a telehealth encounter
   where the clinician guides the patient through a self-examination, record
   the stated findings as examination findings (for example, "tenderness to
   palpation over the lateral epicondyle").
   Do NOT put patient complaints, feelings, or history here.

3. Assessment (A): The diagnosis or differential diagnosis supported by S and
   O. Give the medical term and, where helpful, the common name. Do not add
   diagnoses that the transcript does not support.

4. Plan (P): Treatment and follow-up - activity modification, medications,
   bracing/support, physical therapy, imaging decisions, patient education,
   prognosis, and follow-up. Include only what is stated or directly implied
   by the clinician in the transcript.

STYLE
- Concise clinical documentation style; abbreviations are acceptable.
- Use short bullet points for multiple findings.
- Retain clinically relevant negatives (for example, no numbness, no trauma).
- Do not add a preamble, disclaimer, or commentary outside the JSON.

OUTPUT FORMAT
Return only a valid JSON object with exactly these string keys:
"subjective", "objective", "assessment", "plan".

FORMATTING INSIDE EACH STRING
- Use newline-separated "- " bullet points, one clinical finding or action per
  bullet, not a dense paragraph.
- Add a short label where it helps scanning (for example "- Onset:",
  "- Aggravating factors:", "- Pertinent negatives:", "- Imaging:").
- Keep every bullet traceable to the transcript.
"""


# ═══════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════

class SOAPNoteGenerator:
    """Generate a validated SOAP note from a transcript using Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.2,
        max_retries: int = 3,
    ):
        from google import genai

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for SOAP generation. "
                "Set it as an environment variable."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self.last_validation_issues: list[str] = []

    @property
    def name(self) -> str:
        return f"Gemini ({self._model})"

    def generate(self, transcript: str) -> SOAPNote:
        """
        Generate and validate a SOAP note.

        Args:
            transcript: Verbatim transcript from Part A.

        Returns:
            A populated :class:`SOAPNote`.

        Raises:
            ValueError: if the transcript is empty or the model response cannot
                be parsed into the four required sections.
        """
        from google.genai import types

        from transcriber import call_with_retries

        if not transcript or not transcript.strip():
            raise ValueError("Cannot generate a SOAP note from an empty transcript")

        logger.info("Generating SOAP note with %s...", self.name)

        user_prompt = (
            "Medical dictation transcript:\n\n"
            f'"""\n{transcript.strip()}\n"""\n\n'
            "Create the SOAP note. Return only the JSON object."
        )

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SOAP_SYSTEM_PROMPT,
                    temperature=self._temperature,
                    response_mime_type="application/json",
                ),
            )

        response = call_with_retries(
            _call,
            description="SOAP generation",
            max_retries=self._max_retries,
        )

        raw_text = (response.text or "").strip()
        logger.debug("Raw model response:\n%s", raw_text)

        data = self._parse_json(raw_text)
        if not data:
            raise ValueError("Could not parse a JSON object from the model response")

        missing = [key for key in _REQUIRED_KEYS if not str(data.get(key, "")).strip()]
        if missing:
            raise ValueError(
                "Model response is missing required SOAP sections: "
                + ", ".join(missing)
            )

        note = SOAPNote(
            subjective=str(data["subjective"]).strip(),
            objective=str(data["objective"]).strip(),
            assessment=str(data["assessment"]).strip(),
            plan=str(data["plan"]).strip(),
        )

        self.last_validation_issues = self._validate_soap(note)
        if self.last_validation_issues:
            logger.warning("SOAP cross-contamination warnings:")
            for issue in self.last_validation_issues:
                logger.warning("  - %s", issue)
        else:
            logger.info("SOAP validation passed: no cross-contamination detected.")

        logger.info("SOAP note generation complete.")
        return note

    # ── Parsing helpers ──────────────────────────────────────

    @classmethod
    def _parse_json(cls, text: str) -> dict | None:
        """Parse JSON, tolerating markdown fences or surrounding prose."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Direct JSON parsing failed - attempting extraction")
            return cls._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """
        Extract the first balanced ``{...}`` object from noisy model output.

        This handles stray markdown fences and occasional trailing commentary
        that some models add despite JSON response mode.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")

        depth = 0
        start: int | None = None
        in_string = False
        escaped = False

        for index, char in enumerate(cleaned):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(cleaned[start : index + 1])
                    except json.JSONDecodeError:
                        start = None

        return None

    # ── Bonus: cross-contamination check ─────────────────────

    @staticmethod
    def _validate_soap(note: SOAPNote) -> list[str]:
        """
        Flag likely placement errors between Subjective and Objective.

        This is a heuristic audit, not a semantic parser: it produces warnings
        for a human reviewer rather than blocking the note.
        """
        issues: list[str] = []

        # Phrases that almost always indicate patient-reported content.
        subjective_markers = (
            "patient reports", "patient states", "patient says",
            "patient complains", "patient denies", "patient feels",
            "patient describes", "reports that", "states that",
            "endorses", "denies", "rates pain", "rates the pain",
            "pain is a", "pain score",
        )

        # Phrases that almost always indicate clinician-observed content.
        objective_markers = (
            "on exam", "on examination", "physical exam", "inspection reveals",
            "palpation", "tenderness", "range of motion", "strength testing",
            "vital signs", "blood pressure", "heart rate", "temperature",
            "x-ray", "xray", "mri", "ct scan", "ultrasound", "imaging shows",
            "labs show", "laboratory", "auscultation", "no erythema",
            "no swelling", "no deformity",
        )

        subjective_lower = note.subjective.lower()
        objective_lower = note.objective.lower()

        for marker in objective_markers:
            if marker in subjective_lower:
                issues.append(
                    f"Objective-style language in Subjective: '{marker}'"
                )

        for marker in subjective_markers:
            if marker in objective_lower:
                issues.append(
                    f"Subjective-style language in Objective: '{marker}'"
                )

        return issues
