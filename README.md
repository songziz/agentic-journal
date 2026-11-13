# Daily Journaling Agent (`ADK-Python` on Google Cloud)

> **Scope & Intent:** This project is a simple demonstration system built to showcase core capabilities of an AI agentic system using `ADK-Python` and Google Cloud, aligned with the 95-point **AgentOps Code Review Matrix**. It is **not** designed to support large-scale production workloads.

---

## 1. Architecture & Rubric Alignment (Iteration 2)

The **Daily Journaling Agent** records verbatim daily journal entries, maintains a rolling observational analysis of the user's state (**emotional states**, **personal challenges**, and **personal growth**), answers longitudinal questions grounded in past entries, provides pattern-based recommendations, and exports a structured clinician review report to Google Cloud Storage (gated by two-layer Human-in-the-Loop confirmation).

### Multi-Agent & Strategic Model Routing Hierarchy
* **`daily_journaling_agent` (Coordinator — `gemini-3.5-flash`)**
  * **`entry_intake_agent` (`gemini-3.5-flash`):** Low-latency verbatim entry persistence (`record_verbatim_journal_entry`) and background state synthesis (`synthesize_recent_user_state`).
  * **`insight_and_qa_agent` (`gemini-3.1-pro-preview`):** Longitudinal Q&A (`query_historical_journal_entries`) and pattern-grounded recommendations (`retrieve_pattern_recommendations`).
  * **`clinician_export_agent` (`gemini-3.1-pro-preview`):** Two-step Human-in-the-Loop preview and confirmed GCS export (`export_clinician_review_report`).

### Repository Structure

```text
agentic_journal/
├── LICENSE                          # MIT License
├── README.md                        # Setup, ADK CLI commands, and architecture guide
├── pyproject.toml                   # Package dependencies and linter/test config
├── .env.example                     # Example environment variables for Google Cloud
├── baseline_design.md               # Iteration 1 baseline architecture
├── iteration_2_design.md            # Iteration 2 full rubric-aligned design
├── evals/
│   └── golden_dataset.json          # Golden evaluation suite cases
├── terraform/                       # Infrastructure as Code (IaC)
│   ├── main.tf                      # GCP APIs, Firestore DB, GCS Bucket, Secret Manager
│   ├── variables.tf                 # Terraform input variables
│   └── outputs.tf                   # Terraform outputs
├── journal_agent/                   # ADK-Python agent package
│   ├── __init__.py                  # Exports `agent` for ADK CLI discovery
│   ├── agent.py                     # Coordinator, 3 Sub-Agents, Model Router, and ADK `App`
│   ├── guardrails.py                # Input/Output guardrails, HITL hooks, and ADK BasePlugin
│   ├── memory.py                    # Compaction config, Context Cache config, and Async memory tasks
│   ├── models.py                    # Frozen Pydantic v2 input/output JSON schemas
│   ├── observability.py             # Structured JSON logging, OpenTelemetry, and Cloud DLP PII redaction
│   ├── secrets.py                   # Google Cloud Secret Manager integration
│   ├── storage.py                   # Cloud Firestore and Cloud Storage persistence
│   └── tools.py                     # Schema-validated ADK tools with guided error recovery
├── scripts/
│   └── smoke_test_gcp.py            # Live end-to-end GCP verification script
└── tests/
    ├── test_tools.py                # Iteration 1 tool unit tests
    ├── test_iteration2_rubric.py    # Iteration 2 rubric unit tests (observability, HITL, DLP, secrets)
    └── test_eval_suite.py           # Automated golden dataset evaluation harness
```

---

## 2. Prerequisites & Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Provision Google Cloud Infrastructure via Terraform (Optional if already provisioned):**
   ```bash
   cd terraform
   terraform init
   terraform apply \
     -var="project_id=your-gcp-project-id" \
     -var="medical_export_bucket_name=your-medical-exports-bucket"
   cd ..
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example journal_agent/.env
   ```

---

## 3. Running with the Agent CLI (`adk`)

* **Interactive Terminal CLI:**
  ```bash
  .venv/bin/adk run journal_agent
  ```

* **ADK Web UI:**
  ```bash
  .venv/bin/adk web --port 8000 .
  ```

* **ADK API Server:**
  ```bash
  .venv/bin/adk api_server --port 8000 .
  ```

---

## 4. Testing, Linting & Live GCP Verification

* **Run Unit Tests & Golden Evaluation Suite:**
  ```bash
  .venv/bin/pytest -v
  ```

* **Run Ruff Linter:**
  ```bash
  .venv/bin/ruff check .
  ```

* **Run End-to-End Live GCP Smoke Test:**
  ```bash
  .venv/bin/python3 scripts/smoke_test_gcp.py \
    --project-id=your-gcp-project-id \
    --location=global \
    --bucket=your-medical-exports-bucket
  ```

---

## 5. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
