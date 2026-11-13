"""Multi-agent orchestration, strategic model routing, and ADK App definition."""

import os
from typing import Final

from google.adk import Agent
from google.adk.apps import App, ResumabilityConfig

from journal_agent import guardrails, memory, tools

_DEFAULT_FAST_MODEL: Final[str] = "gemini-3.5-flash"
_DEFAULT_PRO_MODEL: Final[str] = "gemini-3.1-pro-preview"

_FAST_TASK_TYPES: Final[frozenset[str]] = frozenset(
    {"intake", "recording", "routing", "coordinator", "fast"}
)
_PRO_TASK_TYPES: Final[frozenset[str]] = frozenset(
    {"analysis", "insight", "qa", "recommendations", "export", "reasoning"}
)

_CONSTITUTION_HEADER: Final[str] = """\
Agent Constitution & Behavioral Boundaries:
- Persona: Empathetic, objective, observational journaling assistant.
- Strict Non-Clinical Scope: Never diagnose medical, psychological, or
  psychiatric conditions, and never prescribe treatments or medications.
  Describe only self-reported observations, emotional trends, personal
  challenges, personal growth, and verbatim journal quotes with entry dates.
- Verbatim Integrity: Never alter, rewrite, or summarize user text before
  passing it to `record_verbatim_journal_entry`.
- Human-in-the-Loop (HITL) Safety Gate: Never execute a confirmed clinician
  report export (`user_confirmed=True`) until a preview (`user_confirmed=False`)
  has been presented to the user and explicitly approved in the conversation.
"""

_COORDINATOR_INSTRUCTION: Final[str] = f"""\
{_CONSTITUTION_HEADER}

You are the `journal_coordinator_agent`. You coordinate the user's journaling
experience either by invoking the appropriate tool directly or by delegating to
your specialized sub-agents:
1. `entry_intake_agent` (Fast Model): Records verbatim daily journal entries
   (`record_verbatim_journal_entry`) and refreshes the rolling state summary
   (`synthesize_recent_user_state`).
2. `insight_and_qa_agent` (Pro Reasoning Model): Answers questions about the
   user's history (`query_historical_journal_entries`) and provides
   pattern-grounded recommendations (`retrieve_pattern_recommendations`).
3. `clinician_export_agent` (Pro Reasoning Model): Manages the two-step
   Human-in-the-Loop preview and confirmed export of observational summaries for
   a medical professional (`export_clinician_review_report`).
"""

_INTAKE_INSTRUCTION: Final[str] = f"""\
{_CONSTITUTION_HEADER}

You are the `entry_intake_agent`. When the user shares a daily reflection or
journal entry:
1. Call `record_verbatim_journal_entry` with the exact text the user provided.
2. Call `synthesize_recent_user_state` so the rolling summary of emotional
   states, personal challenges, and personal growth is updated.
3. Acknowledge the entry warmly and briefly highlight any key themes recorded.
"""

_INSIGHT_INSTRUCTION: Final[str] = f"""\
{_CONSTITUTION_HEADER}

You are the `insight_and_qa_agent`. You specialize in longitudinal reflection,
answering user questions from their journal history, and offering actionable,
non-clinical recommendations based on recurring patterns:
1. Call `query_historical_journal_entries` to retrieve dated verbatim entries
   relevant to the user's question.
2. Call `retrieve_pattern_recommendations` (and `synthesize_recent_user_state`
   if no summary exists yet) to ground your insights and 2-3 recommendations in
   observed cross-entry correlations.
"""

_EXPORT_INSTRUCTION: Final[str] = f"""\
{_CONSTITUTION_HEADER}

You are the `clinician_export_agent`. You handle the high-stakes action of
exporting an observational journal summary for a medical professional:
1. First call `export_clinician_review_report` with `user_confirmed=False` to
   generate a structured preview.
2. Present the preview (emotional states, personal challenges, personal growth,
   observed patterns, and non-clinical disclaimer) to the user and ask for their
   explicit confirmation.
3. Only after the user explicitly confirms, call
   `export_clinician_review_report` with `user_confirmed=True` and share the
   resulting `gs://` URI.
"""


def select_model_for_task(task_type: str) -> str:
    """Routes a task type to the optimal Gemini model tier (Flash vs. Pro).

    Args:
        task_type: Task category identifier (e.g., 'intake', 'coordinator',
            'insight', 'export', 'reasoning').

    Returns:
        Model identifier string ('gemini-3.5-flash' for fast tasks or
        'gemini-3.1-pro-preview' for complex reasoning/planning tasks).
    """
    normalized = task_type.strip().lower()
    fast_model = (
        os.environ.get("JOURNAL_AGENT_MODEL", "").strip()
        or _DEFAULT_FAST_MODEL
    )
    pro_model = (
        os.environ.get("JOURNAL_AGENT_PRO_MODEL", "").strip()
        or _DEFAULT_PRO_MODEL
    )
    if normalized in _PRO_TASK_TYPES:
        return pro_model
    if normalized in _FAST_TASK_TYPES:
        return fast_model
    return fast_model


def create_sub_agents(
    fast_model: str | None = None,
    pro_model: str | None = None,
) -> tuple[Agent, Agent, Agent]:
    """Creates the three specialized sub-agents with strategic model routing.

    Args:
        fast_model: Optional override for the Flash tier model name.
        pro_model: Optional override for the Pro reasoning tier model name.

    Returns:
        Tuple of `(entry_intake_agent, insight_and_qa_agent,
        clinician_export_agent)`.
    """
    resolved_fast = fast_model or select_model_for_task("intake")
    resolved_pro = pro_model or select_model_for_task("reasoning")

    entry_intake_agent = Agent(
        model=resolved_fast,
        name="entry_intake_agent",
        description=(
            "Fast specialist that records verbatim daily journal entries to "
            "Firestore and updates the rolling user state analysis."
        ),
        instruction=_INTAKE_INSTRUCTION,
        tools=[
            tools.record_verbatim_journal_entry,
            tools.synthesize_recent_user_state,
        ],
        before_model_callback=guardrails.before_model_guardrail,
        after_model_callback=guardrails.after_model_guardrail,
        before_tool_callback=guardrails.before_tool_hook,
        after_tool_callback=guardrails.after_tool_hook,
        on_tool_error_callback=guardrails.on_tool_error_hook,
    )

    insight_and_qa_agent = Agent(
        model=resolved_pro,
        name="insight_and_qa_agent",
        description=(
            "Pro reasoning specialist that answers questions about the user "
            "from historical journal entries and provides pattern-based "
            "recommendations."
        ),
        instruction=_INSIGHT_INSTRUCTION,
        tools=[
            tools.query_historical_journal_entries,
            tools.retrieve_pattern_recommendations,
            tools.synthesize_recent_user_state,
        ],
        before_model_callback=guardrails.before_model_guardrail,
        after_model_callback=guardrails.after_model_guardrail,
        before_tool_callback=guardrails.before_tool_hook,
        after_tool_callback=guardrails.after_tool_hook,
        on_tool_error_callback=guardrails.on_tool_error_hook,
    )

    clinician_export_agent = Agent(
        model=resolved_pro,
        name="clinician_export_agent",
        description=(
            "High-stakes specialist that prepares previews and executes "
            "human-confirmed exports of user state analyses for a medical "
            "professional."
        ),
        instruction=_EXPORT_INSTRUCTION,
        tools=[
            tools.export_clinician_review_report,
            tools.retrieve_pattern_recommendations,
            tools.synthesize_recent_user_state,
        ],
        before_model_callback=guardrails.before_model_guardrail,
        after_model_callback=guardrails.after_model_guardrail,
        before_tool_callback=guardrails.before_tool_hook,
        after_tool_callback=guardrails.after_tool_hook,
        on_tool_error_callback=guardrails.on_tool_error_hook,
    )

    return entry_intake_agent, insight_and_qa_agent, clinician_export_agent


def create_root_agent(
    model_name: str | None = None,
    pro_model_name: str | None = None,
) -> Agent:
    """Creates the Coordinator Root Agent with specialized sub-agents and tools.

    Args:
        model_name: Optional override for the Coordinator and Flash sub-agent
            model. If None, resolves via `select_model_for_task('coordinator')`.
        pro_model_name: Optional override for the Pro reasoning sub-agents. If
            None, resolves via `select_model_for_task('reasoning')`.

    Returns:
        Configured Coordinator `Agent` instance.
    """
    resolved_fast = model_name or select_model_for_task("coordinator")
    resolved_pro = pro_model_name or select_model_for_task("reasoning")
    intake_agent, insight_agent, export_agent = create_sub_agents(
        fast_model=resolved_fast,
        pro_model=resolved_pro,
    )

    return Agent(
        model=resolved_fast,
        name="daily_journaling_agent",
        description=(
            "Coordinator agent for recording verbatim daily journal entries, "
            "analyzing emotional states, challenges, and growth, answering "
            "questions from journal history, and exporting confirmed clinician "
            "review reports."
        ),
        instruction=_COORDINATOR_INSTRUCTION,
        sub_agents=[intake_agent, insight_agent, export_agent],
        tools=[
            tools.record_verbatim_journal_entry,
            tools.synthesize_recent_user_state,
            tools.query_historical_journal_entries,
            tools.retrieve_pattern_recommendations,
            tools.export_clinician_review_report,
        ],
        before_model_callback=guardrails.before_model_guardrail,
        after_model_callback=guardrails.after_model_guardrail,
        before_tool_callback=guardrails.before_tool_hook,
        after_tool_callback=guardrails.after_tool_hook,
        on_tool_error_callback=guardrails.on_tool_error_hook,
    )


root_agent = create_root_agent()

app = App(
    name="journal_agent",
    root_agent=root_agent,
    plugins=[guardrails.JournalSafetyAndObservabilityPlugin()],
    events_compaction_config=memory.build_events_compaction_config(),
    context_cache_config=memory.build_context_cache_config(),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
