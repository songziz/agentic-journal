"""Automated evaluation harness against evals/golden_dataset.json."""

import json
from pathlib import Path
from unittest import mock

import pytest

from journal_agent import agent, guardrails, models, storage, tools

_GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "evals" / "golden_dataset.json"
)


def _load_golden_cases() -> list[dict[str, object]]:
    with _GOLDEN_DATASET_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["cases"])


class TestGoldenEvaluationSuite:
    def test_golden_dataset_has_all_required_categories(self) -> None:
        cases = _load_golden_cases()
        categories = {str(case["category"]) for case in cases}

        assert {
            "entry_intake",
            "insight_and_qa",
            "clinician_export_hitl",
            "security_guardrail",
            "clinical_boundary_guardrail",
        }.issubset(categories)

    def test_strategic_model_routing_matches_golden_expectations(self) -> None:
        cases = _load_golden_cases()
        root = agent.create_root_agent()
        sub_agent_by_name = {sub.name: sub for sub in root.sub_agents}

        for case in cases:
            expected_sub = case.get("expected_sub_agent")
            expected_model = case.get("expected_model_tier")
            if expected_sub and expected_model:
                matched_agent = sub_agent_by_name[str(expected_sub)]
                assert matched_agent.model == expected_model

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(c, id=str(c["case_id"]))
            for c in _load_golden_cases()
            if c.get("expected_guardrail_triggered")
        ],
    )
    def test_input_guardrails_block_adversarial_and_clinical_cases(
        self, case: dict[str, object]
    ) -> None:
        prompt = str(case["user_prompt"])
        expected_substring = str(case["expected_substring"])

        refusal = guardrails.inspect_input_guardrails(prompt)

        assert refusal is not None
        assert expected_substring in refusal

    @mock.patch.object(storage, "list_recent_journal_entries", autospec=True)
    @mock.patch.object(storage, "get_latest_user_state_summary", autospec=True)
    def test_clinician_export_golden_case_enforces_hitl_preview(
        self,
        mock_get_summary: mock.MagicMock,
        mock_list_entries: mock.MagicMock,
    ) -> None:
        mock_get_summary.return_value = models.UserStateSummary(
            user_id="eval_user",
            updated_at="2026-11-13T12:00:00+00:00",
            entry_window_count=2,
            emotional_states="Steady focus after morning walks.",
            personal_challenges="Balancing deadlines.",
            personal_growth="Consistent reflection habit.",
            observed_patterns="Exercise improves afternoon concentration.",
            recommendations=["Continue 20-minute walks after lunch."],
        )
        mock_list_entries.return_value = []

        export_cases = [
            c
            for c in _load_golden_cases()
            if c["case_id"] == "high_stakes_clinician_export_requires_preview_first"
        ]
        assert len(export_cases) == 1
        case = export_cases[0]

        outcome = tools.export_clinician_review_report(
            user_confirmed=bool(case["expected_initial_confirmed_flag"]),
            user_id="eval_user",
        )

        assert outcome["status"] == case["expected_initial_status"]
        assert "preview" in outcome
