"""Context compaction, Vertex AI context caching, and async memory operations."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Final

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig

from journal_agent import observability

logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_THRESHOLD: Final[int] = 4000
_DEFAULT_COMPACTION_INTERVAL: Final[int] = 5
_DEFAULT_OVERLAP_SIZE: Final[int] = 2
_DEFAULT_RETENTION_SIZE: Final[int] = 10
_CHARS_PER_TOKEN_ESTIMATE: Final[int] = 4

_BACKGROUND_TASKS: set[asyncio.Task[object]] = set()


def build_events_compaction_config(
    compaction_interval: int = _DEFAULT_COMPACTION_INTERVAL,
    overlap_size: int = _DEFAULT_OVERLAP_SIZE,
    token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
    event_retention_size: int = _DEFAULT_RETENTION_SIZE,
) -> EventsCompactionConfig:
    """Builds the ADK EventsCompactionConfig to prevent conversation bloat.

    Args:
        compaction_interval: Number of turns between automatic compactions.
        overlap_size: Number of overlapping events kept across compactions.
        token_threshold: Token count threshold triggering history compaction.
        event_retention_size: Number of recent raw events retained uncompacted.

    Returns:
        Configured ADK EventsCompactionConfig instance.
    """
    return EventsCompactionConfig(
        compaction_interval=compaction_interval,
        overlap_size=overlap_size,
        token_threshold=token_threshold,
        event_retention_size=event_retention_size,
    )


def build_context_cache_config(
    cache_intervals: int = 10,
    ttl_seconds: int = 1800,
    min_tokens: int = 1024,
) -> ContextCacheConfig:
    """Builds the Vertex AI ContextCacheConfig for system instructions and tools.

    Args:
        cache_intervals: Maximum invocation intervals before refreshing cache.
        ttl_seconds: Time-to-live of the Vertex AI context cache in seconds.
        min_tokens: Minimum token count required before creating a cache entry.

    Returns:
        Configured ADK ContextCacheConfig instance.
    """
    return ContextCacheConfig(
        cache_intervals=cache_intervals,
        ttl_seconds=ttl_seconds,
        min_tokens=min_tokens,
    )


def compact_conversation_history(
    messages: Sequence[str],
    max_tokens: int = _DEFAULT_TOKEN_THRESHOLD,
    retain_recent: int = 4,
) -> list[str]:
    """Compacts a sequence of conversation messages using a sliding token window.

    When estimated tokens exceed `max_tokens`, older turns outside the
    `retain_recent` window are condensed into a single PII-scrubbed summary
    entry so context bloat is bounded.

    Args:
        messages: Ordered sequence of conversation turn strings.
        max_tokens: Maximum estimated token budget for the conversation history.
        retain_recent: Number of most recent messages to keep verbatim.

    Returns:
        Compacted list of messages respecting the token window.
    """
    if not messages:
        return []

    estimated_tokens = sum(len(m) for m in messages) // _CHARS_PER_TOKEN_ESTIMATE
    if estimated_tokens <= max_tokens or len(messages) <= retain_recent:
        return list(messages)

    older_turns = messages[:-retain_recent]
    recent_turns = list(messages[-retain_recent:])

    condensed_snippets = [
        observability.redact_pii_text(turn.strip()[:160])
        for turn in older_turns
        if turn.strip()
    ]
    summary_header = (
        f"[Compacted Summary of {len(older_turns)} Earlier Turns]: "
        + " | ".join(condensed_snippets)
    )
    max_summary_chars = max(200, (max_tokens // 2) * _CHARS_PER_TOKEN_ESTIMATE)
    return [summary_header[:max_summary_chars], *recent_turns]


async def run_async_state_consolidation(
    sync_fn: Callable[[str, int], dict[str, object]],
    user_id: str,
    window_size: int = 10,
) -> dict[str, object]:
    """Executes expensive state synthesis asynchronously in a background thread.

    Args:
        sync_fn: Synchronous state synthesis callable accepting `(user_id,
            window_size)`.
        user_id: Identifier of the user whose journal state is being consolidated.
        window_size: Number of recent entries to include in the synthesis.

    Returns:
        Result dictionary produced by `sync_fn`.
    """
    tracer = observability.configure_tracing()
    with tracer.start_as_current_span("async_memory_consolidation") as span:
        span.set_attribute("journal.user_id", user_id)
        span.set_attribute("journal.window_size", window_size)
        result = await asyncio.to_thread(sync_fn, user_id, window_size)
        span.set_attribute("journal.status", str(result.get("status", "")))
        return result


def schedule_background_state_consolidation(
    sync_fn: Callable[[str, int], dict[str, object]],
    user_id: str,
    window_size: int = 10,
) -> bool:
    """Schedules non-blocking background memory consolidation if an event loop runs.

    Args:
        sync_fn: Synchronous state synthesis callable accepting `(user_id,
            window_size)`.
        user_id: Identifier of the user whose journal state is being consolidated.
        window_size: Number of recent entries to include in the synthesis.

    Returns:
        True if a background asyncio.Task was scheduled on a running event loop,
        or False if no event loop is currently running in this thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    task = loop.create_task(
        run_async_state_consolidation(
            sync_fn=sync_fn,
            user_id=user_id,
            window_size=window_size,
        )
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    logger.info(
        "Scheduled async memory consolidation task for user %s",
        user_id,
        extra={
            "event_type": "async_memory_scheduled",
            "structured_data": {
                "user_id": user_id,
                "window_size": window_size,
            },
        },
    )
    return True
