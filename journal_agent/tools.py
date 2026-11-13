"""ADK tool functions for the Daily Journaling Agent."""

import datetime
import logging
import os
import uuid
from typing import Final

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from journal_agent import models, storage

logger = logging.getLogger(__name__)

_DEFAULT_MODEL: Final[str] = "gemini-3.5-flash"
_DEFAULT_WINDOW_SIZE: Final[int] = 10
_MEDICAL_DISCLAIMER: Final[str] = (
    "This report is an observational summary of self-reported user journal "
    "entries and verbatim quotes. It does not constitute a clinical diagnosis, "
    "psychological assessment, or medical advice."
)
_SYNTHESIS_SYSTEM_PROMPT: Final[str] = (
    "You are an observational journaling analyst. Analyze the user's recent "
    "verbatim journal entries and produce a structured summary covering: "
    "(1) emotional_states, (2) personal_challenges, (3) personal_growth, "
    "(4) observed_patterns, and (5) 2-3 concrete, non-clinical recommendations. "
    "Ground all observations strictly in the provided journal entries, cite "
    "entry dates where relevant, and never make clinical or medical diagnoses."
)


def _create_genai_client() -> genai.Client:
    """Creates a Vertex AI backed Google GenAI client."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    return genai.Client(vertexai=True, project=project, location=location)


def save_journal_entry(
    raw_text: str,
    tags: list[str] | None = None,
    user_id: str = "default_user",
) -> dict[str, object]:
    """Saves the user's exact verbatim daily journal entry to Cloud Firestore.

    Call this tool whenever the user shares a new journal entry or reflection.
    Do not edit, paraphrase, or truncate the user's words before passing them in
    `raw_text`. After saving a new entry, call `update_user_state_analysis` to
    refresh the rolling analysis of the user's state.

    Args:
        raw_text: Exact verbatim journal text provided by the user.
        tags: Optional list of short topic or emotion tags (e.g., ['work',
            'sleep', 'gratitude']). If None, stores an empty tag list.
        user_id: Unique identifier of the user. Defaults to 'default_user'.

    Returns:
        Dictionary containing status, the saved journal entry metadata, and next
        step guidance, or guided error recovery instructions if saving fails.
    """
    try:
        record = storage.create_journal_entry(
            user_id=user_id,
            raw_text=raw_text,
            tags=tags or (),
        )
    except ValueError as err:
        logger.warning("Validation failed in save_journal_entry: %s", err)
        return {
            "status": "error",
            "error_type": "validation_error",
            "message": str(err),
            "recovery_instruction": (
                "Ask the user to provide non-empty journal text before calling "
                "save_journal_entry again."
            ),
        }
    except google_exceptions.GoogleAPICallError as err:
        logger.error("Firestore error in save_journal_entry: %s", err)
        return {
            "status": "error",
            "error_type": "firestore_error",
            "message": f"Failed to persist journal entry: {err}",
            "recovery_instruction": (
                "Verify GOOGLE_CLOUD_PROJECT and Firestore permissions, then "
                "inform the user that storage is temporarily unavailable."
            ),
        }

    logger.info("Saved journal entry %s for user %s", record.entry_id, record.user_id)
    return {
        "status": "saved",
        "entry": record.model_dump(),
        "next_step": (
            "Call update_user_state_analysis to refresh the rolling analysis "
            "of the user's emotional state, challenges, and growth."
        ),
    }


def update_user_state_analysis(
    user_id: str = "default_user",
    window_size: int = _DEFAULT_WINDOW_SIZE,
) -> dict[str, object]:
    """Updates the rolling analysis of user state from recent journal entries.

    Fetches the most recent `window_size` journal entries from Cloud Firestore,
    synthesizes an observational summary focused on personal challenges,
    personal growth, emotional states, and cross-entry patterns using Vertex AI
    Gemini with a strict JSON schema, and stores the updated snapshot in
    Firestore under `users/{user_id}/state_summaries/latest`.

    Args:
        user_id: Unique identifier of the user. Defaults to 'default_user'.
        window_size: Number of recent entries to analyze (bounded between 1 and
            30). Defaults to 10.

    Returns:
        Dictionary with status and the updated UserStateSummary fields, or
        guided recovery instructions if no entries exist or an API error occurs.
    """
    bounded_window = max(1, min(window_size, 30))
    try:
        recent_entries = storage.list_recent_journal_entries(
            user_id=user_id,
            limit=bounded_window,
        )
    except google_exceptions.GoogleAPICallError as err:
        logger.error("Firestore read error in update_user_state_analysis: %s", err)
        return {
            "status": "error",
            "error_type": "firestore_error",
            "message": f"Could not load recent journal entries: {err}",
            "recovery_instruction": (
                "Proceed by acknowledging the user's entry and explain that "
                "the state analysis could not be refreshed right now."
            ),
        }

    if not recent_entries:
        return {
            "status": "no_entries",
            "message": f"No journal entries found for user {user_id!r}.",
            "recovery_instruction": (
                "Ask the user to share their first journal entry and save it "
                "with save_journal_entry before running state analysis."
            ),
        }

    formatted_entries = "\n\n".join(
        f"[{entry.created_at}] (entry_id={entry.entry_id}, tags={entry.tags}):\n"
        f"{entry.raw_text}"
        for entry in reversed(recent_entries)
    )
    model_name = os.environ.get("JOURNAL_AGENT_MODEL", _DEFAULT_MODEL)

    try:
        genai_client = _create_genai_client()
        response = genai_client.models.generate_content(
            model=model_name,
            contents=(
                f"Analyze the following {len(recent_entries)} recent journal "
                f"entries for user {user_id!r}:\n\n{formatted_entries}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=_SYNTHESIS_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=models.UserStateSynthesis,
                temperature=0.2,
            ),
        )
        synthesis = models.UserStateSynthesis.model_validate_json(response.text or "{}")
        updated_summary = models.UserStateSummary(
            user_id=user_id.strip(),
            updated_at=datetime.datetime.now(datetime.UTC).isoformat(),
            entry_window_count=len(recent_entries),
            emotional_states=synthesis.emotional_states,
            personal_challenges=synthesis.personal_challenges,
            personal_growth=synthesis.personal_growth,
            observed_patterns=synthesis.observed_patterns,
            recommendations=synthesis.recommendations,
        )
        storage.save_user_state_summary(updated_summary)
    except Exception as err:
        logger.error("Failed to synthesize or save user state: %s", err)
        return {
            "status": "error",
            "error_type": "synthesis_error",
            "message": f"Failed to update user state analysis: {err}",
            "recovery_instruction": (
                "Inform the user that their verbatim journal entry is safely "
                "saved, and retry update_user_state_analysis later."
            ),
        }

    logger.info(
        "Updated state summary for user %s across %d entries",
        user_id,
        len(recent_entries),
    )
    return {
        "status": "updated",
        "state_summary": updated_summary.model_dump(),
    }


def search_journal_entries(
    query: str = "",
    start_date: str = "",
    end_date: str = "",
    limit: int = 10,
    user_id: str = "default_user",
) -> dict[str, object]:
    """Searches the user's past journal entries to answer questions about them.

    Use this tool when the user asks questions about their past experiences,
    feelings, habits, or events recorded in their journal.

    Args:
        query: Optional keyword or phrase to filter entries by content or tag.
            Pass an empty string to retrieve all entries in the date window.
        start_date: Optional inclusive ISO-8601 start date or timestamp
            (e.g., '2026-11-01').
        end_date: Optional inclusive ISO-8601 end date or timestamp
            (e.g., '2026-11-12T23:59:59Z').
        limit: Maximum number of matching entries to return. Defaults to 10.
        user_id: Unique identifier of the user. Defaults to 'default_user'.

    Returns:
        Dictionary containing the count and list of matching verbatim journal
        entries, or guided recovery instructions if retrieval fails.
    """
    try:
        matches = storage.search_entries(
            user_id=user_id,
            query_text=query,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    except google_exceptions.GoogleAPICallError as err:
        logger.error("Firestore error in search_journal_entries: %s", err)
        return {
            "status": "error",
            "error_type": "firestore_error",
            "message": f"Failed to query journal entries: {err}",
            "recovery_instruction": (
                "Explain that past journal entries could not be queried due to "
                "a database connection error."
            ),
        }

    return {
        "status": "success",
        "match_count": len(matches),
        "entries": [entry.model_dump() for entry in matches],
    }


def get_user_state_and_recommendations(
    user_id: str = "default_user",
) -> dict[str, object]:
    """Retrieves the latest user state analysis and pattern recommendations.

    Use this tool when the user asks how they have been doing overall, asks
    about recurring emotional or behavioral patterns, or asks for
    recommendations based on their journal history.

    Args:
        user_id: Unique identifier of the user. Defaults to 'default_user'.

    Returns:
        Dictionary containing the latest UserStateSummary and recommendations,
        or instructions to run `update_user_state_analysis` if none exists yet.
    """
    try:
        summary = storage.get_latest_user_state_summary(user_id=user_id)
    except google_exceptions.GoogleAPICallError as err:
        logger.error("Firestore error in get_user_state_and_recommendations: %s", err)
        return {
            "status": "error",
            "error_type": "firestore_error",
            "message": f"Failed to read user state summary: {err}",
            "recovery_instruction": (
                "Try calling search_journal_entries to inspect recent entries "
                "directly instead."
            ),
        }

    if summary is None:
        return {
            "status": "not_found",
            "message": f"No state analysis found yet for user {user_id!r}.",
            "recovery_instruction": (
                "Call update_user_state_analysis first to generate a state "
                "summary from the user's recent journal entries."
            ),
        }

    return {
        "status": "success",
        "state_summary": summary.model_dump(),
    }


def export_medical_review_report(
    user_confirmed: bool = False,
    clinician_notes_focus: str = "",
    entry_limit: int = 14,
    user_id: str = "default_user",
) -> dict[str, object]:
    """Exports a structured user state analysis for a medical professional.

    HIGH-STAKES ACTION: Exporting personal mental/emotional state summaries and
    verbatim journal excerpts for external review by a medical professional
    requires explicit user confirmation.
    - If `user_confirmed` is False, this tool does NOT export to Cloud Storage;
      instead, it returns a preview of the report and instructs you to ask the
      user for explicit confirmation.
    - Only pass `user_confirmed=True` after the user has explicitly approved the
      export in the conversation.

    Args:
        user_confirmed: Set to True ONLY after the user has reviewed the preview
            and explicitly confirmed exporting the report. Defaults to False.
        clinician_notes_focus: Optional note from the user highlighting specific
            questions or themes they want their clinician to review.
        entry_limit: Number of recent verbatim entries to attach as evidence.
            Defaults to 14.
        user_id: Unique identifier of the user. Defaults to 'default_user'.

    Returns:
        Dictionary containing either a report preview requiring confirmation or
        the exported Google Cloud Storage URI (`gs://...`).
    """
    try:
        summary = storage.get_latest_user_state_summary(user_id=user_id)
        recent_entries = storage.list_recent_journal_entries(
            user_id=user_id,
            limit=max(1, min(entry_limit, 30)),
        )
    except google_exceptions.GoogleAPICallError as err:
        logger.error("Firestore error in export_medical_review_report: %s", err)
        return {
            "status": "error",
            "error_type": "firestore_error",
            "message": f"Failed to retrieve user state or entries: {err}",
            "recovery_instruction": (
                "Inform the user that the export could not be prepared due to "
                "a Firestore read error."
            ),
        }

    if summary is None:
        return {
            "status": "missing_state_summary",
            "message": "No user state analysis exists to export.",
            "recovery_instruction": (
                "Call update_user_state_analysis first so there is an up-to-date "
                "state analysis before preparing a medical review export."
            ),
        }

    report = models.MedicalReviewReport(
        report_id=uuid.uuid4().hex[:12],
        user_id=user_id.strip(),
        generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
        disclaimer=_MEDICAL_DISCLAIMER,
        clinician_notes_focus=clinician_notes_focus.strip(),
        state_summary=summary,
        recent_entries=recent_entries,
    )

    if not user_confirmed:
        return {
            "status": "confirmation_required",
            "preview": {
                "user_id": report.user_id,
                "entries_included": len(report.recent_entries),
                "clinician_notes_focus": report.clinician_notes_focus,
                "emotional_states": summary.emotional_states,
                "personal_challenges": summary.personal_challenges,
                "personal_growth": summary.personal_growth,
                "observed_patterns": summary.observed_patterns,
                "disclaimer": report.disclaimer,
            },
            "recovery_instruction": (
                "Present this preview to the user and ask for their explicit "
                "confirmation before calling export_medical_review_report "
                "again with user_confirmed=True."
            ),
        }

    try:
        gcs_uri = storage.upload_medical_report_to_gcs(report=report)
    except (ValueError, google_exceptions.GoogleAPICallError) as err:
        logger.error("GCS upload failed in export_medical_review_report: %s", err)
        return {
            "status": "error",
            "error_type": "gcs_export_error",
            "message": str(err),
            "recovery_instruction": (
                "Verify that MEDICAL_EXPORT_GCS_BUCKET is set to a valid "
                "Cloud Storage bucket and that the service account has "
                "storage.objects.create permission."
            ),
        }

    logger.info(
        "Exported medical review report %s for user %s to %s",
        report.report_id,
        report.user_id,
        gcs_uri,
    )
    return {
        "status": "exported",
        "report_id": report.report_id,
        "gcs_uri": gcs_uri,
        "generated_at": report.generated_at,
        "disclaimer": report.disclaimer,
    }
