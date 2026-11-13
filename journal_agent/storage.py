"""Google Cloud Firestore and Cloud Storage persistence layer."""

import datetime
import os
import uuid
from collections.abc import Sequence
from typing import Final

from google.cloud import firestore, storage

from journal_agent import models

_USERS_COLLECTION: Final[str] = "users"
_ENTRIES_SUBCOLLECTION: Final[str] = "journal_entries"
_STATE_SUBCOLLECTION: Final[str] = "state_summaries"
_LATEST_STATE_DOC_ID: Final[str] = "latest"
_DEFAULT_FIRESTORE_DB: Final[str] = "(default)"


def get_firestore_client(
    project_id: str | None = None,
    database: str | None = None,
) -> firestore.Client:
    """Creates a Google Cloud Firestore client.

    Args:
        project_id: GCP project ID. If None, resolves from GOOGLE_CLOUD_PROJECT
            or Application Default Credentials.
        database: Firestore database name. If None, resolves from
            FIRESTORE_DATABASE or defaults to '(default)'.

    Returns:
        Configured Cloud Firestore client instance.
    """
    resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    resolved_db = (
        database or os.environ.get("FIRESTORE_DATABASE") or _DEFAULT_FIRESTORE_DB
    )
    return firestore.Client(project=resolved_project, database=resolved_db)


def get_storage_client(project_id: str | None = None) -> storage.Client:
    """Creates a Google Cloud Storage client.

    Args:
        project_id: GCP project ID. If None, resolves from GOOGLE_CLOUD_PROJECT
            or Application Default Credentials.

    Returns:
        Configured Google Cloud Storage client instance.
    """
    resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    return storage.Client(project=resolved_project)


def create_journal_entry(
    user_id: str,
    raw_text: str,
    tags: Sequence[str] = (),
    db_client: firestore.Client | None = None,
) -> models.JournalEntryRecord:
    """Persists a verbatim journal entry document to Cloud Firestore.

    Args:
        user_id: Identifier of the user authoring the entry.
        raw_text: Exact, unmodified journal text submitted by the user.
        tags: Optional sequence of topic or emotion tags.
        db_client: Optional pre-initialized Firestore client. If None, creates a
            default client.

    Returns:
        The stored JournalEntryRecord with generated entry_id and timestamp.

    Raises:
        ValueError: If raw_text or user_id is empty after stripping whitespace.
    """
    cleaned_user_id = user_id.strip()
    if not cleaned_user_id:
        raise ValueError("user_id must not be empty.")
    if not raw_text.strip():
        raise ValueError("raw_text must not be empty.")

    client = db_client if db_client is not None else get_firestore_client()
    entry_id = uuid.uuid4().hex
    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    record = models.JournalEntryRecord(
        entry_id=entry_id,
        user_id=cleaned_user_id,
        created_at=created_at,
        raw_text=raw_text,
        tags=[tag.strip() for tag in tags if tag.strip()],
    )

    doc_ref = (
        client.collection(_USERS_COLLECTION)
        .document(cleaned_user_id)
        .collection(_ENTRIES_SUBCOLLECTION)
        .document(entry_id)
    )
    doc_ref.set(record.model_dump())
    return record


def list_recent_journal_entries(
    user_id: str,
    limit: int = 10,
    db_client: firestore.Client | None = None,
) -> list[models.JournalEntryRecord]:
    """Fetches the most recent journal entries for a user ordered newest first.

    Args:
        user_id: Identifier of the user.
        limit: Maximum number of recent entries to retrieve.
        db_client: Optional pre-initialized Firestore client. If None, creates a
            default client.

    Returns:
        List of JournalEntryRecord instances ordered from newest to oldest.
    """
    client = db_client if db_client is not None else get_firestore_client()
    capped_limit = max(1, min(limit, 50))
    entries_ref = (
        client.collection(_USERS_COLLECTION)
        .document(user_id.strip())
        .collection(_ENTRIES_SUBCOLLECTION)
    )
    query = entries_ref.order_by(
        "created_at", direction=firestore.Query.DESCENDING
    ).limit(capped_limit)
    return [
        models.JournalEntryRecord.model_validate(snapshot.to_dict())
        for snapshot in query.stream()
        if snapshot.exists
    ]


def search_entries(
    user_id: str,
    query_text: str = "",
    start_date: str = "",
    end_date: str = "",
    limit: int = 10,
    db_client: firestore.Client | None = None,
) -> list[models.JournalEntryRecord]:
    """Searches a user's journal entries by date range and optional substring.

    Args:
        user_id: Identifier of the user.
        query_text: Optional case-insensitive keyword or phrase to match in
            raw_text or tags.
        start_date: Optional inclusive ISO-8601 date/timestamp lower bound.
        end_date: Optional inclusive ISO-8601 date/timestamp upper bound.
        limit: Maximum number of matching entries to return.
        db_client: Optional pre-initialized Firestore client. If None, creates a
            default client.

    Returns:
        Matching JournalEntryRecord instances ordered from newest to oldest.
    """
    recent_records = list_recent_journal_entries(
        user_id=user_id,
        limit=max(limit * 3, 30),
        db_client=db_client,
    )
    normalized_query = query_text.strip().lower()
    normalized_start = start_date.strip()
    normalized_end = end_date.strip()

    matched_records: list[models.JournalEntryRecord] = []
    for record in recent_records:
        if normalized_start and record.created_at < normalized_start:
            continue
        if normalized_end and record.created_at > normalized_end:
            continue
        if normalized_query:
            haystack = " ".join([record.raw_text, *record.tags]).lower()
            if normalized_query not in haystack:
                continue
        matched_records.append(record)
        if len(matched_records) >= limit:
            break

    return matched_records


def save_user_state_summary(
    summary: models.UserStateSummary,
    db_client: firestore.Client | None = None,
) -> models.UserStateSummary:
    """Saves the latest rolling user state summary in Cloud Firestore.

    Args:
        summary: Validated UserStateSummary model to store under
            users/{user_id}/state_summaries/latest.
        db_client: Optional pre-initialized Firestore client. If None, creates a
            default client.

    Returns:
        The persisted UserStateSummary instance.
    """
    client = db_client if db_client is not None else get_firestore_client()
    doc_ref = (
        client.collection(_USERS_COLLECTION)
        .document(summary.user_id)
        .collection(_STATE_SUBCOLLECTION)
        .document(_LATEST_STATE_DOC_ID)
    )
    doc_ref.set(summary.model_dump())
    return summary


def get_latest_user_state_summary(
    user_id: str,
    db_client: firestore.Client | None = None,
) -> models.UserStateSummary | None:
    """Retrieves the latest user state analysis from Cloud Firestore.

    Args:
        user_id: Identifier of the user.
        db_client: Optional pre-initialized Firestore client. If None, creates a
            default client.

    Returns:
        The latest UserStateSummary, or None if no state summary exists yet for
        the user.
    """
    client = db_client if db_client is not None else get_firestore_client()
    snapshot = (
        client.collection(_USERS_COLLECTION)
        .document(user_id.strip())
        .collection(_STATE_SUBCOLLECTION)
        .document(_LATEST_STATE_DOC_ID)
        .get()
    )
    if not snapshot.exists:
        return None
    data = snapshot.to_dict()
    if not data:
        return None
    return models.UserStateSummary.model_validate(data)


def format_medical_report_markdown(report: models.MedicalReviewReport) -> str:
    """Formats a MedicalReviewReport into a readable Markdown document.

    Args:
        report: Validated MedicalReviewReport instance.

    Returns:
        Formatted Markdown string suitable for clinician review.
    """
    summary = report.state_summary
    recommendations_lines = "\n".join(f"- {item}" for item in summary.recommendations)
    excerpts_lines = "\n\n".join(
        f"### Entry `{entry.entry_id}` ({entry.created_at})\n> {entry.raw_text}"
        for entry in report.recent_entries
    )
    focus_section = (
        f"**User-Requested Review Focus:** {report.clinician_notes_focus}\n\n"
        if report.clinician_notes_focus
        else ""
    )

    return (
        f"# Observational Journal Summary for Healthcare Professional Review\n\n"
        f"**Report ID:** `{report.report_id}`  \n"
        f"**User ID:** `{report.user_id}`  \n"
        f"**Generated At (UTC):** `{report.generated_at}`  \n"
        f"**Entries Analyzed:** {summary.entry_window_count}\n\n"
        f"> **Disclaimer:** {report.disclaimer}\n\n"
        f"{focus_section}"
        f"## 1. Emotional States (Self-Reported Trends)\n"
        f"{summary.emotional_states}\n\n"
        f"## 2. Personal Challenges & Stressors\n"
        f"{summary.personal_challenges}\n\n"
        f"## 3. Personal Growth & Positive Coping\n"
        f"{summary.personal_growth}\n\n"
        f"## 4. Observed Behavioral & Emotional Patterns\n"
        f"{summary.observed_patterns}\n\n"
        f"## 5. Non-Clinical Self-Care Recommendations Provided\n"
        f"{recommendations_lines or '- None recorded.'}\n\n"
        f"## 6. Verbatim Dated Journal Excerpts\n"
        f"{excerpts_lines or 'No verbatim entries available.'}\n"
    )


def upload_medical_report_to_gcs(
    report: models.MedicalReviewReport,
    bucket_name: str | None = None,
    gcs_client: storage.Client | None = None,
) -> str:
    """Uploads a formatted medical review report to Google Cloud Storage.

    Args:
        report: Validated MedicalReviewReport to upload.
        bucket_name: Target GCS bucket name. If None, resolves from
            MEDICAL_EXPORT_GCS_BUCKET.
        gcs_client: Optional pre-initialized Cloud Storage client. If None,
            creates a default client.

    Returns:
        The `gs://` URI of the uploaded Markdown report object.

    Raises:
        ValueError: If no GCS bucket name is provided or configured in
            MEDICAL_EXPORT_GCS_BUCKET.
    """
    resolved_bucket = (
        bucket_name or os.environ.get("MEDICAL_EXPORT_GCS_BUCKET", "").strip()
    )
    if not resolved_bucket:
        raise ValueError(
            "MEDICAL_EXPORT_GCS_BUCKET is not set. Configure a Cloud Storage "
            "bucket name before exporting a medical review report."
        )

    client = gcs_client if gcs_client is not None else get_storage_client()
    bucket = client.bucket(resolved_bucket)
    object_path = (
        f"medical_exports/{report.user_id}/{report.generated_at}_{report.report_id}.md"
    )
    blob = bucket.blob(object_path)
    markdown_body = format_medical_report_markdown(report)
    blob.upload_from_string(
        markdown_body,
        content_type="text/markdown; charset=utf-8",
    )
    return f"gs://{resolved_bucket}/{object_path}"
