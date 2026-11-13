"""Structured JSON logging, OpenTelemetry tracing, and PII redaction pipelines."""

import datetime
import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Final

from google.api_core import exceptions as google_exceptions
from google.cloud import dlp_v2
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
)
_SSN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)
_DLP_INFO_TYPES: Final[tuple[str, ...]] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SOCIAL_SECURITY_NUMBER",
    "PERSON_NAME",
)

_IN_MEMORY_SPAN_EXPORTER: Final[InMemorySpanExporter] = InMemorySpanExporter()
_TRACING_INITIALIZED: bool = False


def redact_pii_text(
    text: str,
    use_cloud_dlp: bool = False,
    project_id: str | None = None,
    dlp_client: dlp_v2.DlpServiceClient | None = None,
) -> str:
    """Scrubs sensitive PII from text using regex rules and optional Cloud DLP.

    Args:
        text: Input string that may contain sensitive personal identifiers.
        use_cloud_dlp: Whether to invoke the Google Cloud DLP API in addition
            to deterministic regex scrubbing.
        project_id: GCP project ID for Cloud DLP. If None, resolves from
            GOOGLE_CLOUD_PROJECT.
        dlp_client: Optional pre-initialized DlpServiceClient. If None, creates
            a default client when `use_cloud_dlp` is True.

    Returns:
        Scrubbed string with PII tokens replaced by redaction placeholders.
    """
    if not text:
        return text

    scrubbed = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    scrubbed = _SSN_PATTERN.sub("[REDACTED_SSN]", scrubbed)
    scrubbed = _PHONE_PATTERN.sub("[REDACTED_PHONE]", scrubbed)

    resolved_project = (
        project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    if not use_cloud_dlp or not resolved_project:
        return scrubbed

    client = dlp_client if dlp_client is not None else dlp_v2.DlpServiceClient()
    parent = f"projects/{resolved_project}/locations/global"
    inspect_config = {
        "info_types": [{"name": name} for name in _DLP_INFO_TYPES],
    }
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "primitive_transformation": {
                        "replace_with_info_type_config": {}
                    }
                }
            ]
        }
    }
    try:
        response = client.deidentify_content(
            request={
                "parent": parent,
                "inspect_config": inspect_config,
                "deidentify_config": deidentify_config,
                "item": {"value": scrubbed},
            }
        )
        if response.item and response.item.value:
            return response.item.value
    except google_exceptions.GoogleAPICallError:
        return scrubbed

    return scrubbed


def redact_pii_payload(payload: object) -> object:
    """Recursively scrubs PII from strings inside dicts, lists, and tuples.

    Args:
        payload: Arbitrary serializable data structure to scrub before logging
            or storage.

    Returns:
        Copy of `payload` with all string values scrubbed via `redact_pii_text`.
    """
    if isinstance(payload, str):
        return redact_pii_text(payload)
    if isinstance(payload, Mapping):
        return {
            str(key): redact_pii_payload(val) for key, val in payload.items()
        }
    if isinstance(payload, list):
        return [redact_pii_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_pii_payload(item) for item in payload)
    return payload


class StructuredJsonFormatter(logging.Formatter):
    """Logging formatter that emits single-line structured JSON records."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats a LogRecord into a PII-scrubbed JSON string.

        Args:
            record: Standard library LogRecord instance.

        Returns:
            Serialized JSON string with timestamp, severity, trace context, and
            structured payload fields.
        """
        current_span = trace.get_current_span()
        span_ctx = current_span.get_span_context() if current_span else None
        trace_id = (
            f"{span_ctx.trace_id:032x}"
            if span_ctx and span_ctx.is_valid
            else ""
        )
        span_id = (
            f"{span_ctx.span_id:016x}"
            if span_ctx and span_ctx.is_valid
            else ""
        )

        event_type = getattr(record, "event_type", "application_log")
        structured_data = getattr(record, "structured_data", {})

        log_object: dict[str, object] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "event_type": event_type,
            "message": redact_pii_text(record.getMessage()),
            "trace_id": trace_id,
            "span_id": span_id,
            "data": redact_pii_payload(structured_data),
        }
        return json.dumps(log_object, default=str)


def configure_structured_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures the package logger to emit structured JSON log records.

    Args:
        level: Minimum logging level to emit. Defaults to logging.INFO.

    Returns:
        Configured `journal_agent` root logger.
    """
    pkg_logger = logging.getLogger("journal_agent")
    pkg_logger.setLevel(level)
    has_json_handler = any(
        isinstance(h.formatter, StructuredJsonFormatter)
        for h in pkg_logger.handlers
    )
    if not has_json_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        pkg_logger.addHandler(handler)
    return pkg_logger


def configure_tracing(
    service_name: str = "agentic-journal",
    enable_cloud_trace: bool | None = None,
    project_id: str | None = None,
) -> trace.Tracer:
    """Initializes OpenTelemetry distributed tracing with optional Cloud Trace.

    Args:
        service_name: Logical service name for emitted spans.
        enable_cloud_trace: Whether to attach the Google Cloud Trace exporter.
            If None, resolves from ENABLE_CLOUD_TRACE environment variable.
        project_id: GCP project ID for Cloud Trace. If None, resolves from
            GOOGLE_CLOUD_PROJECT.

    Returns:
        Configured OpenTelemetry Tracer instance.
    """
    global _TRACING_INITIALIZED
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

    if not _TRACING_INITIALIZED:
        provider.add_span_processor(
            SimpleSpanProcessor(_IN_MEMORY_SPAN_EXPORTER)
        )
        should_export_cloud = (
            enable_cloud_trace
            if enable_cloud_trace is not None
            else os.environ.get("ENABLE_CLOUD_TRACE", "").lower()
            in {"1", "true", "yes"}
        )
        resolved_project = (
            project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        )
        if should_export_cloud and resolved_project:
            try:
                from opentelemetry.exporter.cloud_trace import (
                    CloudTraceSpanExporter,
                )

                cloud_exporter = CloudTraceSpanExporter(
                    project_id=resolved_project
                )
                provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
            except Exception as err:
                logging.getLogger(__name__).warning(
                    "CloudTraceSpanExporter not attached: %s", err
                )
        _TRACING_INITIALIZED = True

    return trace.get_tracer(service_name)


def get_recorded_spans() -> tuple[object, ...]:
    """Returns spans captured by the in-memory OpenTelemetry span exporter.

    Returns:
        Tuple of finished ReadableSpan objects recorded in the current process.
    """
    return _IN_MEMORY_SPAN_EXPORTER.get_finished_spans()


def clear_recorded_spans() -> None:
    """Clears the in-memory OpenTelemetry span buffer."""
    _IN_MEMORY_SPAN_EXPORTER.clear()


def log_tool_intent(
    logger: logging.Logger,
    tool_name: str,
    tool_args: Mapping[str, object],
) -> None:
    """Records the agent's intended tool call before execution.

    Args:
        logger: Logger configured with StructuredJsonFormatter.
        tool_name: Name of the tool the agent intends to invoke.
        tool_args: Arguments passed to the tool (automatically PII-scrubbed).
    """
    logger.info(
        "Tool intent captured for %s",
        tool_name,
        extra={
            "event_type": "tool_intent",
            "structured_data": {
                "intended_tool": tool_name,
                "intended_args": dict(tool_args),
            },
        },
    )


def log_tool_outcome(
    logger: logging.Logger,
    tool_name: str,
    tool_args: Mapping[str, object],
    outcome: Mapping[str, object],
) -> None:
    """Records the actual outcome of a tool call after execution.

    Args:
        logger: Logger configured with StructuredJsonFormatter.
        tool_name: Name of the executed tool.
        tool_args: Arguments passed to the tool (automatically PII-scrubbed).
        outcome: Result dictionary returned by the tool.
    """
    logger.info(
        "Tool outcome captured for %s (status=%s)",
        tool_name,
        outcome.get("status", "unknown"),
        extra={
            "event_type": "tool_outcome",
            "structured_data": {
                "executed_tool": tool_name,
                "tool_args": dict(tool_args),
                "actual_status": outcome.get("status", "unknown"),
                "outcome": dict(outcome),
            },
        },
    )
