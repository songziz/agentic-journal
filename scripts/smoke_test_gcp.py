"""End-to-end live GCP smoke test for the Iteration 1 Daily Journaling Agent."""

import argparse
import asyncio
import os
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from journal_agent import agent, tools


async def _run_agent_turn(user_id: str, prompt: str) -> str:
    """Runs a single conversational turn through the ADK root_agent."""
    runner = InMemoryRunner(
        app_name="agentic_journal_smoke_test",
        agent=agent.create_root_agent(),
    )
    session = await runner.session_service.create_session(
        app_name="agentic_journal_smoke_test",
        user_id=user_id,
    )
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )
    replies: list[str] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    replies.append(part.text)
    return "\n".join(replies)


def main() -> int:
    """Executes the live GCP smoke test across Firestore, Vertex AI, and GCS."""
    parser = argparse.ArgumentParser(
        description="Run live GCP smoke tests for the Daily Journaling Agent."
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        help="GCP project ID with Vertex AI, Firestore, and GCS enabled.",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        help="Vertex AI region (default: global).",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MEDICAL_EXPORT_GCS_BUCKET", ""),
        help="GCS bucket name for medical review exports.",
    )
    parser.add_argument(
        "--user-id",
        default="smoke_test_user",
        help="Test user ID namespace in Firestore (default: smoke_test_user).",
    )
    args = parser.parse_args()

    if not args.project_id:
        print(
            "ERROR: Pass --project-id or set GOOGLE_CLOUD_PROJECT.",
            file=sys.stderr,
        )
        return 1

    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "TRUE"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    if args.bucket:
        os.environ["MEDICAL_EXPORT_GCS_BUCKET"] = args.bucket

    print(
        "=== 1. Saving verbatim journal entry to Firestore "
        f"(project={args.project_id}) ==="
    )
    save_res = tools.save_journal_entry(
        raw_text=(
            "Today I felt overwhelmed by two overlapping project deadlines, "
            "but taking a 20-minute walk after lunch helped me focus and "
            "finish the design doc."
        ),
        tags=["work", "deadlines", "walk"],
        user_id=args.user_id,
    )
    print(save_res)
    if save_res.get("status") != "saved":
        return 1

    print("\n=== 2. Synthesizing rolling user state via Vertex AI Gemini ===")
    state_res = tools.update_user_state_analysis(
        user_id=args.user_id,
        window_size=5,
    )
    print(state_res)
    if state_res.get("status") != "updated":
        return 1

    print(
        "\n=== 3. Searching entries & fetching pattern recommendations ==="
    )
    search_res = tools.search_journal_entries(
        query="walk",
        user_id=args.user_id,
    )
    print("Search result count:", search_res.get("match_count"))
    rec_res = tools.get_user_state_and_recommendations(user_id=args.user_id)
    print("Recommendations:", rec_res)

    print(
        "\n=== 4. Testing medical export confirmation gate (user_confirmed=False) ==="
    )
    preview_res = tools.export_medical_review_report(
        user_confirmed=False,
        clinician_notes_focus="Work deadline stress and coping strategies",
        user_id=args.user_id,
    )
    print("Preview status:", preview_res.get("status"))

    if args.bucket:
        print(
            f"\n=== 5. Exporting confirmed medical report to gs://{args.bucket} ==="
        )
        export_res = tools.export_medical_review_report(
            user_confirmed=True,
            clinician_notes_focus="Work deadline stress and coping strategies",
            user_id=args.user_id,
        )
        print(export_res)
        if export_res.get("status") != "exported":
            return 1
    else:
        print(
            "\n=== 5. Skipping live GCS upload "
            "(no --bucket / MEDICAL_EXPORT_GCS_BUCKET provided) ==="
        )

    print("\n=== 6. Running end-to-end ADK Agent turn ===")
    reply = asyncio.run(
        _run_agent_turn(
            user_id=args.user_id,
            prompt=(
                f"My user_id is '{args.user_id}'. Based on my journal entries, "
                "what challenges have I faced recently and what do you recommend?"
            ),
        )
    )
    print("Agent reply:\n", reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
