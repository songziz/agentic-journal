"""Pydantic data models and strict JSON schemas for the Daily Journaling Agent."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JournalEntryInput(BaseModel):
    """Strict input schema validating arguments to record a journal entry.

    Attributes:
        raw_text: Exact verbatim journal text submitted by the user.
        tags: Optional list of topic or emotion tags.
        user_id: Identifier of the user authoring the entry.
    """

    model_config = ConfigDict(frozen=True)

    raw_text: str = Field(
        min_length=1,
        description="Exact verbatim journal text submitted by the user.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional list of topic or emotion tags.",
    )
    user_id: str = Field(
        default="default_user",
        min_length=1,
        description="Unique identifier of the user.",
    )

    @field_validator("raw_text", "user_id")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty or whitespace only.")
        return value if value != cleaned and " " in value else cleaned


class JournalSearchInput(BaseModel):
    """Strict input schema validating parameters for searching journal entries.

    Attributes:
        query: Optional keyword or phrase to filter entries.
        start_date: Optional inclusive ISO-8601 lower date bound.
        end_date: Optional inclusive ISO-8601 upper date bound.
        limit: Maximum number of matching entries to return.
        user_id: Unique identifier of the user.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        default="",
        description="Optional case-insensitive keyword or phrase filter.",
    )
    start_date: str = Field(
        default="",
        description="Optional inclusive ISO-8601 start date or timestamp.",
    )
    end_date: str = Field(
        default="",
        description="Optional inclusive ISO-8601 end date or timestamp.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of entries to return (1 to 50).",
    )
    user_id: str = Field(
        default="default_user",
        min_length=1,
        description="Unique identifier of the user.",
    )


class ClinicianExportInput(BaseModel):
    """Strict input schema validating parameters for clinician report exports.

    Attributes:
        user_confirmed: Whether the user explicitly approved the export preview.
        clinician_notes_focus: Optional focus note for the reviewing clinician.
        entry_limit: Number of recent verbatim entries to include.
        user_id: Unique identifier of the user.
    """

    model_config = ConfigDict(frozen=True)

    user_confirmed: bool = Field(
        default=False,
        description="True only after the user approves the report preview.",
    )
    clinician_notes_focus: str = Field(
        default="",
        description="Optional user-requested focus area for the clinician.",
    )
    entry_limit: int = Field(
        default=14,
        ge=1,
        le=30,
        description="Number of recent verbatim entries to include (1 to 30).",
    )
    user_id: str = Field(
        default="default_user",
        min_length=1,
        description="Unique identifier of the user.",
    )


class JournalEntryRecord(BaseModel):
    """Represents an immutable verbatim journal entry stored in Cloud Firestore.

    Attributes:
        entry_id: Unique identifier of the journal entry document.
        user_id: Identifier of the user who authored the entry.
        created_at: ISO-8601 UTC timestamp when the entry was recorded.
        raw_text: Exact, unmodified journal text submitted by the user.
        tags: Optional list of topic or emotion tags associated with the entry.
    """

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(description="Unique identifier of the journal entry.")
    user_id: str = Field(description="User identifier owning this entry.")
    created_at: str = Field(description="ISO-8601 UTC timestamp of creation.")
    raw_text: str = Field(description="Exact verbatim journal text from the user.")
    tags: list[str] = Field(
        default_factory=list,
        description="Optional topic or emotion tags attached to the entry.",
    )


class UserStateSynthesis(BaseModel):
    """Schema constraining LLM synthesis of recent journal entries.

    Attributes:
        emotional_states: Observational summary of recent moods and shifts.
        personal_challenges: Recurring stressors, obstacles, or unresolved themes.
        personal_growth: Observed positive habits, milestones, and coping progress.
        observed_patterns: Cross-entry behavioral or emotional correlations.
        recommendations: Concrete, non-clinical suggestions grounded in patterns.
    """

    model_config = ConfigDict(frozen=True)

    emotional_states: str = Field(
        description=(
            "Observational summary of prevailing moods, emotional shifts, and "
            "self-reported triggers with entry date references."
        )
    )
    personal_challenges: str = Field(
        description=(
            "Recurring personal challenges, stressors, or obstacles reported "
            "across recent entries."
        )
    )
    personal_growth: str = Field(
        description=(
            "Positive habits, resilience, milestones, and personal growth "
            "observed across recent entries."
        )
    )
    observed_patterns: str = Field(
        description=(
            "Recurring correlations between activities, environments, and "
            "emotional states across entries."
        )
    )
    recommendations: list[str] = Field(
        description=(
            "2 to 3 concrete, actionable, non-clinical recommendations "
            "grounded in the observed patterns."
        )
    )


class UserStateSummary(BaseModel):
    """Represents the rolling user state analysis stored in Cloud Firestore.

    Attributes:
        user_id: Identifier of the user.
        updated_at: ISO-8601 UTC timestamp when the summary was last refreshed.
        entry_window_count: Number of recent entries included in this synthesis.
        emotional_states: Summary of prevailing moods, shifts, and triggers.
        personal_challenges: Recurring stressors, obstacles, or unresolved themes.
        personal_growth: Positive habits, milestones, and coping progress.
        observed_patterns: Cross-entry behavioral or emotional correlations.
        recommendations: Actionable, non-clinical suggestions based on patterns.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(description="User identifier.")
    updated_at: str = Field(description="ISO-8601 UTC timestamp of the analysis.")
    entry_window_count: int = Field(
        description="Number of recent journal entries analyzed."
    )
    emotional_states: str = Field(
        description="Observational summary of prevailing emotional states."
    )
    personal_challenges: str = Field(
        description="Summary of recurring personal challenges."
    )
    personal_growth: str = Field(
        description="Summary of observed personal growth and progress."
    )
    observed_patterns: str = Field(
        description="Cross-entry behavioral and emotional patterns."
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Non-clinical recommendations grounded in observed patterns.",
    )


class MedicalReviewReport(BaseModel):
    """Structured observational export prepared for a medical professional.

    Attributes:
        report_id: Unique identifier of the generated report.
        user_id: Identifier of the user whose journal data is summarized.
        generated_at: ISO-8601 UTC timestamp when the report was generated.
        disclaimer: Mandatory non-diagnostic observational disclaimer.
        clinician_notes_focus: Optional user-requested focus area for the review.
        state_summary: Latest synthesized user state snapshot.
        recent_entries: Verbatim dated journal entries included as evidence.
    """

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(description="Unique identifier of the export report.")
    user_id: str = Field(description="User identifier.")
    generated_at: str = Field(description="ISO-8601 UTC timestamp of export.")
    disclaimer: str = Field(
        description=(
            "Explicit disclaimer stating that this report contains self-reported "
            "observational summaries and verbatim journal excerpts, not clinical "
            "diagnoses or medical advice."
        )
    )
    clinician_notes_focus: str = Field(
        default="",
        description="Optional user-specified focus area for the clinician.",
    )
    state_summary: UserStateSummary = Field(
        description="Rolling user state synthesis at the time of export."
    )
    recent_entries: list[JournalEntryRecord] = Field(
        default_factory=list,
        description="Verbatim recent journal entries included for reference.",
    )


class SessionCheckpoint(BaseModel):
    """Persistent conversational session checkpoint stored in Cloud Firestore.

    Attributes:
        session_id: Unique session identifier.
        user_id: Identifier of the user.
        updated_at: ISO-8601 UTC timestamp of the checkpoint.
        compacted_summary: Summarized context of earlier conversation turns.
        recent_turn_count: Number of turns retained in the active window.
        medical_export_preview_ready: Whether a clinician export preview was
            presented and awaits human confirmation.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(description="Unique session identifier.")
    user_id: str = Field(description="User identifier.")
    updated_at: str = Field(description="ISO-8601 UTC timestamp.")
    compacted_summary: str = Field(
        default="",
        description="Compacted summary of older conversational turns.",
    )
    recent_turn_count: int = Field(
        default=0,
        ge=0,
        description="Count of recent turns retained in context.",
    )
    medical_export_preview_ready: bool = Field(
        default=False,
        description="Whether a medical export preview has been shown.",
    )
