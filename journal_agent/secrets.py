"""Secure secret management backed by Google Cloud Secret Manager."""

import logging
import os
from typing import Final

from google.api_core import exceptions as google_exceptions
from google.cloud import secretmanager

logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_BUCKET_SECRET_ID: Final[str] = "journal-agent-export-bucket"


def get_secret_manager_client() -> secretmanager.SecretManagerServiceClient:
    """Creates a Google Cloud Secret Manager client instance.

    Returns:
        Configured SecretManagerServiceClient instance.
    """
    return secretmanager.SecretManagerServiceClient()


def access_secret_version(
    secret_id: str,
    project_id: str | None = None,
    version_id: str = "latest",
    sm_client: secretmanager.SecretManagerServiceClient | None = None,
) -> str | None:
    """Fetches a secret payload from Google Cloud Secret Manager.

    Args:
        secret_id: Identifier of the secret in Secret Manager.
        project_id: GCP project ID. If None, resolves from GOOGLE_CLOUD_PROJECT.
        version_id: Secret version to access. Defaults to 'latest'.
        sm_client: Optional pre-initialized SecretManagerServiceClient. If None,
            creates a default client.

    Returns:
        The UTF-8 decoded secret string, or None if the project is unset or the
        secret cannot be accessed.
    """
    resolved_project = (
        project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    )
    if not resolved_project or not secret_id.strip():
        return None

    client = (
        sm_client if sm_client is not None else get_secret_manager_client()
    )
    secret_Path = (
        f"projects/{resolved_project}/secrets/{secret_id.strip()}/"
        f"versions/{version_id}"
    )
    try:
        response = client.access_secret_version(request={"name": secret_Path})
    except google_exceptions.GoogleAPICallError as err:
        logger.warning(
            "Could not access secret %s in project %s: %s",
            secret_id,
            resolved_project,
            err,
        )
        return None

    return response.payload.data.decode("utf-8").strip()


def resolve_export_bucket_name(
    explicit_bucket: str | None = None,
    project_id: str | None = None,
    sm_client: secretmanager.SecretManagerServiceClient | None = None,
) -> str:
    """Resolves the medical export GCS bucket from argument, env, or Secret Manager.

    Args:
        explicit_bucket: Optional bucket name passed directly by the caller.
        project_id: Optional GCP project ID for Secret Manager lookup. If None,
            resolves from GOOGLE_CLOUD_PROJECT.
        sm_client: Optional pre-initialized SecretManagerServiceClient. If None,
            creates a default client when querying Secret Manager.

    Returns:
        Resolved Cloud Storage bucket name, or an empty string if unconfigured.
    """
    if explicit_bucket and explicit_bucket.strip():
        return explicit_bucket.strip()

    env_bucket = os.environ.get("MEDICAL_EXPORT_GCS_BUCKET", "").strip()
    if env_bucket:
        return env_bucket

    secret_id = os.environ.get(
        "MEDICAL_EXPORT_BUCKET_SECRET_ID", _DEFAULT_EXPORT_BUCKET_SECRET_ID
    )
    secret_val = access_secret_version(
        secret_id=secret_id,
        project_id=project_id,
        sm_client=sm_client,
    )
    return secret_val or ""
