# Daily Journaling Agent (Iteration 1 Baseline)

> **Scope & Intent:** This project is a simple demonstration system built to showcase core capabilities of an AI agentic system using `ADK-Python` and Google Cloud. It is **not** designed to support large-scale production workloads.

## Overview

The **Daily Journaling Agent** records verbatim daily journal entries, maintains a rolling observational analysis of the user's state (emotional states, personal challenges, and personal growth), answers questions grounded in past entries, provides pattern-based recommendations, and exports a structured summary for review by a medical professional (gated by explicit user confirmation).

### Repository Layout

```text
agentic_journal/
├── LICENSE                    # MIT License
├── README.md                  # Setup and usage guide
├── pyproject.toml             # Package dependencies and tool configuration
├── .env.example               # Example environment variables for Google Cloud
├── baseline_design.md         # Iteration 1 architecture and data model
├── journal_agent/             # ADK-Python agent package
│   ├── __init__.py            # Exports `agent` for ADK CLI discovery
│   ├── agent.py               # Defines `root_agent` and system instructions
│   ├── models.py              # Pydantic v2 schemas for entries, state, and exports
│   ├── storage.py             # Cloud Firestore and Cloud Storage persistence helpers
│   └── tools.py               # Five core ADK tool functions
├── scripts/
│   └── smoke_test_gcp.py      # Live end-to-end GCP smoke test
└── tests/
    └── test_tools.py          # Unit tests for tool validation and confirmation gates
```

## Prerequisites & Setup

1. **Create and activate a virtual environment:**
   ```bash
   uv venv --python python3.11 .venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

2. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth application-default login
   ```

3. **Configure environment variables:**
   Copy `.env.example` to `journal_agent/.env` (or export variables in your shell):
   ```bash
   cp .env.example journal_agent/.env
   ```
   Update `GOOGLE_CLOUD_PROJECT`, `FIRESTORE_DATABASE`, and `MEDICAL_EXPORT_GCS_BUCKET` in `journal_agent/.env`.

## Running the Agent

* **Interactive CLI:**
  ```bash
  adk run journal_agent
  ```

* **ADK Web UI:**
  ```bash
  adk web .
  ```

## Running Unit Tests

```bash
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
