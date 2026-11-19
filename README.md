# Medical Transcription & SOAP Note Generation Pipeline

A production-grade, privacy-conscious pipeline to transcribe medical dictation audio and programmatically generate structured clinical **SOAP notes** (Subjective, Objective, Assessment, Plan).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Tool Selection Rationale](#tool-selection-rationale)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Usage](#usage)
  - [Option 1: Command Line (CLI)](#option-1-command-line-cli)
  - [Option 2: Jupyter Notebook](#option-2-jupyter-notebook)
- [Output Documentation](#output-documentation)
- [Clinical Evaluation & Prompt Engineering](#clinical-evaluation--prompt-engineering)
- [Bonus: Cross-Contamination Guard](#bonus-cross-contamination-guard)
- [Scaling Considerations](#scaling-considerations)
- [License](#license)

---

## Overview

Clinical documentation demands both verbatim fidelity for medical terms and strict adherence to structured documentation standards. This project implements an end-to-end automated workflow:

1. **Part A — Speech-to-Text (Transcriber):** Ingests raw medical audio dictation and produces a verbatim text transcript preserving complex clinical terminology, patient symptoms, and anatomical descriptions.
2. **Part B — Text-to-SOAP (Structuring Engine):** Extracts clinical entities, maps them strictly into standard SOAP components, and enforces JSON output with a post-processing cross-contamination validation layer.

---

## System Architecture

```
┌─────────────────────────────────┐
│     Medical Audio Dictation     │ (sample_dictation.mp3: ~13.75 min)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Speech-to-Text Layer        │
│  • Primary: faster-whisper      │ ◀── Local CPU/GPU, zero data leakage (HIPAA friendly)
│  • Fallback: Gemini Audio API   │ ◀── Cloud multimodal fallback
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Raw Verbatim Transcript      │ ──▶ [output/transcript.txt]
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    SOAP Structuring Engine      │
│  • System Prompt Constraints    │
│  • JSON Schema Enforcement      │
│  • Cross-Contamination Guard    │ ◀── Verifies S vs O isolation
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Structured Deliverables    │
│  • output/soap_note.json        │
│  • output/soap_note.md          │
└─────────────────────────────────┘
```

---

## Tool Selection Rationale

| Layer | Selected Tool | Rationale & Trade-offs |
|---|---|---|
| **Transcription (Primary)** | `faster-whisper` (CTranslate2) | **Privacy-First:** Audio stays strictly on-premises, critical for HIPAA compliance. CTranslate2 achieves up to **4× faster** CPU execution and lower VRAM usage than vanilla OpenAI Whisper via INT8/FP16 quantization. |
| **Transcription (Fallback)** | Google Gemini Multimodal API | Zero-setup fallback for environments without dedicated GPU/PyTorch compilation. Capable of ingesting long-form uncompressed audio with high medical phonetic accuracy. |
| **SOAP Structuring** | Google Gemini (3.5 / Flash) | Strong biomedical entity recognition, zero hallucination of measurements, native strict JSON output constraint (`response_mime_type="application/json"`), and deterministic sampling (`temperature=0.2`). |

---

## Repository Structure

```
medical-transcription-soap/
├── audio/
│   └── sample_dictation.mp3           # Input dictation audio file (~13.75 min)
├── output/
│   ├── transcript.txt                 # Verbatim transcription output
│   ├── soap_note.json                 # Structured SOAP note (JSON)
│   └── soap_note.md                   # Rendered clinical SOAP note (Markdown)
├── config.py                          # Global configuration and path declarations
├── transcriber.py                     # Dual-backend Speech-to-Text implementation
├── soap_generator.py                  # Prompt engineering & cross-contamination check
├── pipeline.py                        # Automated end-to-end execution runner
├── medical_transcription_soap.ipynb   # Interactive walkthrough notebook
├── requirements.txt                   # Production Python package requirements
├── .gitignore                         # Standard git excludes
└── README.md                          # Project documentation and guide
```

---

## Quick Start

### Prerequisites

- **Python 3.10+** (Tested on Python 3.11, 3.12, 3.13)
- An active **Google Gemini API Key** (for SOAP generation and multimodal fallback)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yashgoyal4321/medical-transcription-soap.git
   cd medical-transcription-soap
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Export your Gemini API Key in your terminal session or set it in your environment:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Windows Command Prompt
set GEMINI_API_KEY=your-api-key-here

# Linux / macOS
export GEMINI_API_KEY="your-api-key-here"
```

---

## Usage

### Option 1: Command Line (CLI)

Run the full end-to-end pipeline with the default sample dictation:

```bash
python pipeline.py
```

To specify an alternate audio file:
```bash
python pipeline.py "path/to/custom_dictation.mp3"
```

### Option 2: Jupyter Notebook

Launch the notebook to inspect intermediate outputs, step through audio transcription, and view the formatted Markdown note:

```bash
jupyter notebook medical_transcription_soap.ipynb
```

---

## Output Documentation

### 1. Raw Transcript (`output/transcript.txt`)
Contains the full verbatim transcription of the clinical encounter (patient with right elbow pain lasting several months, exacerbated by repetitive wrist motions at work, examined for range of motion, diagnosed with lateral epicondylitis).

### 2. Structured SOAP Note (`output/soap_note.json`)
```json
{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "..."
}
```

### 3. Clinical Markdown Note (`output/soap_note.md`)
Formatted clinical summary:
- **Subjective:** Chief complaint (lateral elbow pain), onset (several months ago), aggravating factors (wrist extension/flexion, repetitive line work), relieving factors (rest, ibuprofen), pain score (4/10), pertinent negatives.
- **Objective:** Physical exam inspection (no erythema, swelling, or deformity), palpation (point tenderness over lateral epicondyle, medial aspect non-tender), active/passive ROM tests.
- **Assessment:** Lateral epicondylitis (Tennis elbow).
- **Plan:** Conservative therapy, activity modification (ergonomic adjustments for repetitive forearm extensor use), counterforce brace/compression sleeve, physical therapy, NSAIDs/analgesics, follow-up timeline.

---

## Clinical Evaluation & Prompt Engineering

### Prompt Design Strategy
The structuring engine utilizes a strict system instruction tailored for clinical documentation:
1. **Explicit Inclusion/Exclusion Constraints:** Defining boundaries prevents subjective statements (e.g., patient complaints or self-reports) from leaking into physician-measured objective data.
2. **Pertinent Negatives Retention:** Capturing reported negatives (e.g., lack of fever, numbness, tingling, or radiating pain) is critical for clinical triage and medico-legal records.
3. **Low Temperature Sampling:** Set to `0.2` to ensure consistent, reproducible categorization.

---

## Bonus: Cross-Contamination Guard

A heuristic validation function (`SOAPNoteGenerator._validate_soap()`) analyzes the generated note:
- **Subjective-in-Objective Checker:** Flags patient-directed speech markers (`"patient reports"`, `"feels"`, `"complains of"`) erroneously placed under `Objective`.
- **Objective-in-Subjective Checker:** Flags clinical exam terminology (`"on exam"`, `"palpation reveals"`, `"blood pressure"`) placed under `Subjective`.

---

## Scaling Considerations

For enterprise deployment across hospital systems or telehealth providers:

1. **HIPAA & Local Processing:**
   - Host `faster-whisper-large-v3` on dedicated GPU clusters or on-premise Kubernetes pods.
   - Deploy local LLMs (e.g., **Llama-3-70B-Instruct** or **Mistral-Large**) via vLLM / Ollama for completely air-gapped SOAP extraction.
2. **Asynchronous Task Queuing:**
   - Decouple audio ingestion from note generation using **Celery** or **RabbitMQ** with Redis backends to handle concurrent provider dictations without HTTP timeout risks.
3. **EHR / EMR Integration:**
   - Map generated JSON directly to **HL7 FHIR** `DocumentReference` or `Composition` resources.
4. **Speaker Diarization:**
   - Integrate `pyannote.audio` to separate patient dialog from physician dictation in multi-party encounters.

---

## License

MIT License. See project files for details.
