"""
soap_generator.py - SOAP Note Generation Module

Takes a medical dictation transcript and structures it into a
SOAP (Subjective, Objective, Assessment, Plan) note using
Google Gemini's generative AI with carefully engineered prompts.

Features:
• Structured JSON output with schema enforcement
• Cross-contamination validation (S vs O separation)
• Medical terminology preservation
• Markdown-formatted output generation
"""

import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════

@dataclass
class SOAPNote:
    """Structured SOAP note with all four sections."""

    subjective: str
    objective: str
    assessment: str
    plan: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render as a formatted Markdown document."""
        return (
            "# SOAP Note\n\n"
            "## Subjective\n"
            f"{self.subjective}\n\n"
            "## Objective\n"
            f"{self.objective}\n\n"
            "## Assessment\n"
            f"{self.assessment}\n\n"
            "## Plan\n"
            f"{self.plan}\n"
        )


# ═══════════════════════════════════════════════════════════════
# Prompt Engineering
# ═══════════════════════════════════════════════════════════════

SOAP_SYSTEM_PROMPT = """\
You are a highly skilled medical scribe AI. Your job is to take a raw \
medical dictation transcript and produce a concise, accurate, and \
well-structured SOAP note.

RULES:
1. **Subjective (S):** Only include information the PATIENT reported — \
   symptoms, history, feelings, what makes it better/worse, duration, \
   severity (pain scale), medications tried, and relevant social/family history. \
   Do NOT include anything the doctor observed or measured.

2. **Objective (O):** Only include information the DOCTOR observed, measured, \
   or tested — physical exam findings, vital signs, inspection results, \
   palpation, range of motion tests, and any imaging or lab results. \
   Do NOT include patient-reported symptoms.

3. **Assessment (A):** Provide the diagnosis or differential diagnosis \
   based on the subjective and objective findings. Include the medical \
   term and any common name.

4. **Plan (P):** Include the treatment plan — activity modifications, \
   medications, therapies, follow-up instructions, referrals, and expected \
   prognosis/timeline.

FORMATTING:
- Use concise medical documentation style (abbreviations are OK)
- Use bullet points for multiple items within a section
- Preserve medical terminology exactly
- Include relevant negatives (pertinent negative findings)

OUTPUT:
Return ONLY a valid JSON object with keys: "subjective", "objective", \
"assessment", "plan". Each value is a string containing the section content.
"""


# ═══════════════════════════════════════════════════════════════
# SOAP Note Generator
# ═══════════════════════════════════════════════════════════════

class SOAPNoteGenerator:
    """
    Generates structured SOAP notes from medical transcripts
    using Google Gemini API with prompt engineering.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        from google import genai

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it as an environment variable."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, transcript: str) -> SOAPNote:
        """
        Generate a SOAP note from a medical dictation transcript.

        Args:
            transcript: The verbatim transcript text.

        Returns:
            A SOAPNote dataclass with S, O, A, P sections.
        """
        from google.genai import types

        logger.info("Generating SOAP note using %s ...", self._model)

        user_prompt = (
            f"Transcript of a medical dictation:\n\n"
            f'"""\n{transcript}\n"""\n\n'
            f"Generate a structured SOAP note from this transcript. "
            f"Return ONLY the JSON object — no markdown fences, no explanation."
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SOAP_SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        # Parse the JSON response
        raw_text = response.text.strip()
        logger.debug("Raw LLM response:\n%s", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # The model sometimes returns multiple JSON objects or
            # wraps JSON in markdown fences. Try to extract the first
            # valid JSON object.
            logger.warning("Direct JSON parse failed — attempting extraction")
            data = self._extract_json(raw_text)

        if not data:
            raise ValueError("Could not parse any JSON from the LLM response")

        soap_note = SOAPNote(
            subjective=data.get("subjective", ""),
            objective=data.get("objective", ""),
            assessment=data.get("assessment", ""),
            plan=data.get("plan", ""),
        )

        # Validate cross-contamination
        issues = self._validate_soap(soap_note)
        if issues:
            logger.warning("SOAP validation issues found:")
            for issue in issues:
                logger.warning("  ⚠ %s", issue)

        logger.info("SOAP note generation complete.")
        return soap_note

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """
        Extract the first valid JSON object from text that may contain
        multiple objects, markdown fences, or extra content.
        """
        import re

        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text)

        # Try to find and parse the first { ... } block
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = None
                        continue
        return None

    @staticmethod
    def _validate_soap(note: SOAPNote) -> list[str]:
        """
        BONUS: Cross-contamination check.

        Verifies that Subjective information (patient-reported) doesn't
        appear in the Objective section (doctor-measured), and vice versa.

        Returns a list of warning messages (empty if no issues).
        """
        issues = []

        # Subjective markers — patient-reported language
        subjective_markers = [
            "patient reports", "patient states", "patient says",
            "patient complains", "patient denies", "patient feels",
            "he reports", "she reports", "he states", "she states",
            "he says", "she says", "he denies", "she denies",
            "rates the pain", "rates pain", "describes",
            "has been feeling", "has noticed",
        ]

        # Objective markers — doctor-observed/measured language
        objective_markers = [
            "on exam", "on examination", "on inspection",
            "palpation reveals", "upon palpation",
            "range of motion", "rom ", "x-ray", "xray",
            "mri ", "ct scan", "lab results", "labs show",
            "vital signs", "blood pressure", "bp ", "heart rate",
            "temperature", "tenderness on", "swelling noted",
            "positive for", "negative for",
        ]

        # Check for objective language in the Subjective section
        subj_lower = note.subjective.lower()
        for marker in objective_markers:
            if marker in subj_lower:
                issues.append(
                    f"Possible objective data in Subjective: found '{marker}'"
                )

        # Check for subjective language in the Objective section
        obj_lower = note.objective.lower()
        for marker in subjective_markers:
            if marker in obj_lower:
                issues.append(
                    f"Possible subjective data in Objective: found '{marker}'"
                )

        return issues
