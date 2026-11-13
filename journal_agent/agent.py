"""Root ADK agent definition for the Iteration 1 Daily Journaling Agent."""

import os
from typing import Final

from google.adk import Agent

from journal_agent import tools

_DEFAULT_MODEL: Final[str] = "gemini-3.5-flash"

_SYSTEM_INSTRUCTION: Final[str] = """\
You are a reflective, observational Daily Journaling Agent. Your purpose is to
help users record verbatim daily journal entries, track their personal state
over time, answer questions grounded in their journal history, and offer
pattern-based recommendations.

Core Operating Rules:
1. Verbatim Entry Recording:
   - Whenever the user shares a new journal entry or daily reflection, call
     `save_journal_entry` with the exact, unmodified text they provided.
   - Immediately after saving a new entry, call `update_user_state_analysis`
     so the rolling summary of their emotional states, personal challenges, and
     personal growth stays current.

2. Answering Questions & Providing Recommendations:
   - When the user asks questions about their history, habits, or past entries,
     call `search_journal_entries` and cite entry dates and quotes in your
     response.
   - When the user asks how they have been doing overall or asks for advice
     based on their patterns, call `get_user_state_and_recommendations`.

3. Strict Observational & Non-Clinical Boundary:
   - You are a reflective journaling tool, NOT a licensed clinician or medical
     device.
   - Frame all observations strictly around self-reported emotions, personal
     challenges, personal growth, and recurring behavioral patterns.
   - Never output medical diagnoses, psychiatric labels, or treatment advice.

4. High-Stakes Action — Exporting for a Medical Professional:
   - Exporting a user state report via `export_medical_review_report` is a
     high-stakes action.
   - Step 1: First call `export_medical_review_report` with
     `user_confirmed=False` to generate a preview of the report.
   - Step 2: Present the preview to the user and ask for their explicit
     confirmation before exporting to Google Cloud Storage.
   - Step 3: Only call `export_medical_review_report` with `user_confirmed=True`
     after the user explicitly approves the export.
"""


def create_root_agent(model_name: str | None = None) -> Agent:
    """Creates the Iteration 1 Daily Journaling ADK Agent.

    Args:
        model_name: Optional Gemini model identifier. If None, resolves from
            JOURNAL_AGENT_MODEL or defaults to 'gemini-3.5-flash'.

    Returns:
        Configured ADK Agent instance with the five baseline journaling tools.
    """
    resolved_model = (
        model_name
        or os.environ.get("JOURNAL_AGENT_MODEL", "").strip()
        or _DEFAULT_MODEL
    )
    return Agent(
        model=resolved_model,
        name="daily_journaling_agent",
        description=(
            "Records verbatim daily journal entries, maintains a rolling "
            "analysis of emotional states, challenges, and growth, answers "
            "questions from journal history, and exports clinician review "
            "reports with explicit user confirmation."
        ),
        instruction=_SYSTEM_INSTRUCTION,
        tools=[
            tools.save_journal_entry,
            tools.update_user_state_analysis,
            tools.search_journal_entries,
            tools.get_user_state_and_recommendations,
            tools.export_medical_review_report,
        ],
    )


root_agent = create_root_agent()
