"""Unit tests for the Iteration 1 Daily Journaling Agent tools and storage."""

from unittest import mock

import pytest

from journal_agent import models, storage, tools


def _sample_entry(
    entry_id: str = "entry_1",
    raw_text: str = "Felt anxious about the deadline, but finished the draft.",
    tags: list[str] | None = None,
) -> models.JournalEntryRecord:
    return models.JournalEntryRecord(
        entry_id=entry_id,
        user_id="user_123",
        created_at="2026-11-12T10:00:00+00:00",
        raw_text=raw_text,
        tags=tags if tags is not None else ["work", "deadline"],
    )


def _sample_summary() -> models.UserStateSummary:
    return models.UserStateSummary(
        user_id="user_123",
        updated_at="2026-11-12T10:05:00+00:00",
        entry_window_count=3,
        emotional_states="Initial anxiety around deadlines followed by relief.",
        personal_challenges="Time pressure on project milestones.",
        personal_growth="Completing drafts despite initial hesitation.",
        observed_patterns="Morning writing sessions reduce afternoon stress.",
        recommendations=[
            "Block 30 minutes each morning for focused drafting.",
            "Note one completed task at the end of each workday.",
        ],
    )


class TestSaveJournalEntry:
    @mock.patch.object(storage, "create_journal_entry", autospec=True)
    def test_saves_verbatim_entry_successfully(
        self, mock_create_entry: mock.MagicMock
    ) -> None:
        expected_record = _sample_entry()
        mock_create_entry.return_value = expected_record

        result = tools.save_journal_entry(
            raw_text=expected_record.raw_text,
            tags=["work", "deadline"],
            user_id="user_123",
        )

        assert result["status"] == "saved"
        assert result["entry"] == expected_record.model_dump()
        mock_create_entry.assert_called_once_with(
            user_id="user_123",
            raw_text=expected_record.raw_text,
            tags=["work", "deadline"],
        )

    @pytest.mark.parametrize(
        ("raw_text", "user_id"),
        [
            pytest.param("   ", "user_123", id="empty_raw_text"),
            pytest.param("Valid journal entry.", "   ", id="empty_user_id"),
        ],
    )
    def test_returns_guided_error_on_invalid_input(
        self, raw_text: str, user_id: str
    ) -> None:
        result = tools.save_journal_entry(raw_text=raw_text, user_id=user_id)

        assert result["status"] == "error"
        assert result["error_type"] == "validation_error"
        assert "recovery_instruction" in result


class TestSearchAndRecommendations:
    @mock.patch.object(storage, "search_entries", autospec=True)
    def test_search_journal_entries_returns_matches(
        self, mock_search: mock.MagicMock
    ) -> None:
        entry = _sample_entry()
        mock_search.return_value = [entry]

        result = tools.search_journal_entries(
            query="deadline",
            user_id="user_123",
        )

        assert result["status"] == "success"
        assert result["match_count"] == 1
        assert result["entries"] == [entry.model_dump()]

    @mock.patch.object(storage, "get_latest_user_state_summary", autospec=True)
    def test_get_user_state_returns_not_found_when_missing(
        self, mock_get_summary: mock.MagicMock
    ) -> None:
        mock_get_summary.return_value = None

        result = tools.get_user_state_and_recommendations(user_id="user_123")

        assert result["status"] == "not_found"
        assert "update_user_state_analysis" in str(result["recovery_instruction"])


class TestExportMedicalReviewReport:
    @mock.patch.object(storage, "upload_medical_report_to_gcs", autospec=True)
    @mock.patch.object(storage, "list_recent_journal_entries", autospec=True)
    @mock.patch.object(storage, "get_latest_user_state_summary", autospec=True)
    def test_requires_explicit_confirmation_before_uploading(
        self,
        mock_get_summary: mock.MagicMock,
        mock_list_entries: mock.MagicMock,
        mock_upload_gcs: mock.MagicMock,
    ) -> None:
        mock_get_summary.return_value = _sample_summary()
        mock_list_entries.return_value = [_sample_entry()]

        result = tools.export_medical_review_report(
            user_confirmed=False,
            clinician_notes_focus="Sleep and work stress",
            user_id="user_123",
        )

        assert result["status"] == "confirmation_required"
        assert "preview" in result
        mock_upload_gcs.assert_not_called()

    @mock.patch.object(storage, "upload_medical_report_to_gcs", autospec=True)
    @mock.patch.object(storage, "list_recent_journal_entries", autospec=True)
    @mock.patch.object(storage, "get_latest_user_state_summary", autospec=True)
    def test_uploads_to_gcs_when_user_confirmed(
        self,
        mock_get_summary: mock.MagicMock,
        mock_list_entries: mock.MagicMock,
        mock_upload_gcs: mock.MagicMock,
    ) -> None:
        mock_get_summary.return_value = _sample_summary()
        mock_list_entries.return_value = [_sample_entry()]
        mock_upload_gcs.return_value = (
            "gs://test-bucket/medical_exports/user_123/report.md"
        )

        result = tools.export_medical_review_report(
            user_confirmed=True,
            clinician_notes_focus="Sleep and work stress",
            user_id="user_123",
        )

        assert result["status"] == "exported"
        assert (
            result["gcs_uri"] == "gs://test-bucket/medical_exports/user_123/report.md"
        )
        mock_upload_gcs.assert_called_once()
