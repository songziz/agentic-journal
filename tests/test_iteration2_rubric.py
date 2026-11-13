"""Unit tests verifying Iteration 2 AgentOps rubric capabilities."""

import asyncio
import json
import logging
from unittest import mock

import pytest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.cloud import dlp_v2, secretmanager

from journal_agent import (
    agent,
    guardrails,
    memory,
    observability,
    secrets,
    storage,
    tools,
)


class TestObservabilityAndPiiRedaction:
    @pytest.mark.parametrize(
        ("raw_input", "expected_placeholder"),
        [
            pytest.param(
                "Contact me at jane.doe@example.com about my therapy.",
                "[REDACTED_EMAIL]",
                id="email_redaction",
            ),
            pytest.param(
                "My number is (415) 555-2671 if needed.",
                "[REDACTED_PHONE]",
                id="phone_redaction",
            ),
            pytest.param(
                "SSN 123-45-6789 should never be stored.",
                "[REDACTED_SSN]",
                id="ssn_redaction",
            ),
        ],
    )
    def test_redact_pii_text_scrubs_sensitive_tokens(
        self, raw_input: str, expected_placeholder: str
    ) -> None:
        scrubbed = observability.redact_pii_text(raw_input)
        assert expected_placeholder in scrubbed
        assert "jane.doe@example.com" not in scrubbed
        assert "555-2671" not in scrubbed
        assert "123-45-6789" not in scrubbed

    def test_cloud_dlp_deidentification_integration(self) -> None:
        mock_dlp = mock.create_autospec(dlp_v2.DlpServiceClient, instance=True)
        fake_response = mock.MagicMock()
        fake_response.item.value = "Met with [PERSON_NAME] at [REDACTED_EMAIL]."
        mock_dlp.deidentify_content.return_value = fake_response

        result = observability.redact_pii_text(
            "Met with Alice Smith at alice@example.com.",
            use_cloud_dlp=True,
            project_id="test-gcp-project",
            dlp_client=mock_dlp,
        )

        assert result == "Met with [PERSON_NAME] at [REDACTED_EMAIL]."
        mock_dlp.deidentify_content.assert_called_once()

    def test_structured_json_formatter_and_intent_outcome_capture(self) -> None:
        formatter = observability.StructuredJsonFormatter()
        record = logging.LogRecord(
            name="journal_agent.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="User email test@example.com invoked tool",
            args=(),
            exc_info=None,
        )
        record.event_type = "tool_intent"
        record.structured_data = {
            "intended_tool": "record_verbatim_journal_entry",
            "intended_args": {"raw_text": "Reach me at 415-555-9999"},
        }

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["event_type"] == "tool_intent"
        assert "[REDACTED_EMAIL]" in parsed["message"]
        assert (
            parsed["data"]["intended_args"]["raw_text"]
            == "Reach me at [REDACTED_PHONE]"
        )

    def test_opentelemetry_spans_recorded_during_tool_call(self) -> None:
        observability.clear_recorded_spans()
        with mock.patch.object(
            storage, "search_entries", autospec=True, return_value=[]
        ):
            tools.query_historical_journal_entries(
                query="sleep", user_id="user_otel"
            )

        spans = observability.get_recorded_spans()
        span_names = [getattr(s, "name", "") for s in spans]
        assert "query_historical_journal_entries" in span_names


class TestContextMemoryAndAsyncConsolidation:
    def test_compact_conversation_history_condenses_old_turns(self) -> None:
        turns = [
            f"Turn {i}: I wrote a long reflection with email user{i}@example.com "
            + ("detail " * 40)
            for i in range(10)
        ]

        compacted = memory.compact_conversation_history(
            turns, max_tokens=150, retain_recent=3
        )

        assert len(compacted) == 4
        assert compacted[0].startswith("[Compacted Summary of 7 Earlier Turns]:")
        assert "[REDACTED_EMAIL]" in compacted[0]
        assert compacted[1:] == turns[-3:]

    def test_async_state_consolidation_runs_in_background(self) -> None:
        def fake_sync(user_id: str, window_size: int) -> dict[str, object]:
            return {
                "status": "updated",
                "user_id": user_id,
                "window_size": window_size,
            }

        result = asyncio.run(
            memory.run_async_state_consolidation(
                sync_fn=fake_sync,
                user_id="async_user",
                window_size=5,
            )
        )
        assert result["status"] == "updated"
        assert result["user_id"] == "async_user"


class TestGuardrailsHitlAndMultiAgentRouting:
    def test_output_guardrail_blocks_clinical_diagnosis_claims(self) -> None:
        unsafe_output = (
            "Based on your entries, I diagnose you with major depressive "
            "disorder."
        )
        rewritten = guardrails.evaluate_output_guardrails(unsafe_output)
        assert rewritten is not None
        assert "cannot make clinical diagnoses" in rewritten

    def test_hitl_hook_blocks_confirmed_export_without_prior_preview(
        self,
    ) -> None:
        mock_tool = mock.create_autospec(BaseTool, instance=True)
        mock_tool.name = "export_clinician_review_report"
        mock_ctx = mock.create_autospec(ToolContext, instance=True)
        mock_ctx.state = {}

        blocked = guardrails.before_tool_hook(
            tool=mock_tool,
            args={"user_confirmed": True, "user_id": "user_1"},
            tool_context=mock_ctx,
        )

        assert blocked is not None
        assert blocked["status"] == "hitl_confirmation_blocked"

        guardrails.after_tool_hook(
            tool=mock_tool,
            args={"user_confirmed": False, "user_id": "user_1"},
            tool_context=mock_ctx,
            tool_response={"status": "confirmation_required"},
        )
        assert mock_ctx.state["medical_export_preview_ready"] is True

        allowed = guardrails.before_tool_hook(
            tool=mock_tool,
            args={"user_confirmed": True, "user_id": "user_1"},
            tool_context=mock_ctx,
        )
        assert allowed is None

    def test_multi_agent_hierarchy_and_app_configuration(self) -> None:
        root = agent.create_root_agent()
        sub_names = [sub.name for sub in root.sub_agents]

        assert sub_names == [
            "entry_intake_agent",
            "insight_and_qa_agent",
            "clinician_export_agent",
        ]
        assert agent.app.events_compaction_config is not None
        assert agent.app.context_cache_config is not None
        assert len(agent.app.plugins) == 1


class TestSecretManagerIntegration:
    def test_resolves_bucket_from_secret_manager_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MEDICAL_EXPORT_GCS_BUCKET", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-gcp-project")

        mock_sm = mock.create_autospec(
            secretmanager.SecretManagerServiceClient, instance=True
        )
        fake_resp = mock.MagicMock()
        fake_resp.payload.data = b"test-gcp-project-journal-bucket"
        mock_sm.access_secret_version.return_value = fake_resp

        bucket = secrets.resolve_export_bucket_name(sm_client=mock_sm)

        assert bucket == "test-gcp-project-journal-bucket"
        mock_sm.access_secret_version.assert_called_once()
