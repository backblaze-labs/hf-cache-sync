"""S3-compatible object storage backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

from hf_cache_sync import __version__
from hf_cache_sync.config import AppConfig

B2_USER_AGENT = "b2ai-hfcache"
DEFAULT_USER_AGENT = "hf-cache-sync"

# S3 / B2 codes that mean "the object does not exist" rather than a real failure.
# Bucket-level errors (NoSuchBucket) deliberately stay loud — they're config errors,
# not missing-key signals.
NOT_FOUND_CODES = {"404", "NoSuchKey"}

# Codes the --fallback path should treat as "remote unreachable, use HF hub instead."
# Auth / config errors deliberately stay loud so users actually fix them.
TRANSIENT_CODES = {
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
    "503",
    "500",
}

# Codes that mean "bad config or bad creds" — these must always surface, never
# be papered over by fallback.
AUTH_CODES = {
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "AccessDenied",
    "AllAccessDisabled",
    "401",
    "403",
}


class StorageError(RuntimeError):
    """Human-readable wrapper around boto3 errors.

    Carries the original ``ClientError`` / ``EndpointConnectionError`` as
    ``__cause__`` so debugging output is preserved while the user-facing
    message stays plain English.

    Attributes:
        code: Best-effort S3 error code or sentinel like ``"NetworkUnreachable"``.
        transient: True iff retrying or falling back to another source could plausibly succeed.
        auth_failure: True for credential / permission errors.
    """

    def __init__(
        self, message: str, *, code: str = "", transient: bool = False, auth_failure: bool = False
    ):
        super().__init__(message)
        self.code = code
        self.transient = transient
        self.auth_failure = auth_failure


def _is_backblaze_endpoint(endpoint: str) -> bool:
    """Check if the endpoint is a Backblaze B2 S3-compatible endpoint."""
    if not endpoint:
        return False
    host = (urlparse(endpoint).hostname or "").lower().rstrip(".")
    return host == "backblazeb2.com" or host.endswith(".backblazeb2.com")


def get_user_agent(endpoint: str) -> str:
    """Return the appropriate user-agent string for the storage endpoint."""
    if _is_backblaze_endpoint(endpoint):
        return f"{B2_USER_AGENT}/{__version__}"
    return f"{DEFAULT_USER_AGENT}/{__version__}"


def is_not_found(err: ClientError) -> bool:
    """True iff the ClientError is a 404 / NoSuchKey class of error."""
    code = err.response.get("Error", {}).get("Code", "")
    return code in NOT_FOUND_CODES


def _humanize_client_error(err: ClientError, *, bucket: str, endpoint: str) -> StorageError:
    """Translate a boto3 ClientError into an actionable StorageError."""
    error = err.response.get("Error", {})
    code = error.get("Code", "")
    message = error.get("Message", str(err))
    auth = code in AUTH_CODES
    transient = code in TRANSIENT_CODES

    if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch"):
        msg = (
            "S3 credentials rejected. Check `storage.access_key` / `storage.secret_key` "
            "in your config, or the B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY "
            "(or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) env vars."
        )
    elif code == "AccessDenied":
        msg = (
            "Access denied. Your application key may not have permission for this "
            f"operation on bucket '{bucket}'. Check the key's bucket scope and "
            "capabilities."
        )
    elif code == "NoSuchBucket":
        msg = (
            f"Bucket '{bucket}' not found. Verify the bucket exists and that the "
            f"endpoint ({endpoint or 'AWS default'}) and region match."
        )
    elif code in ("PermanentRedirect", "AuthorizationHeaderMalformed"):
        msg = (
            f"Bucket '{bucket}' is not in the configured region. Update "
            "`storage.region` and `storage.endpoint` to match the bucket."
        )
    elif transient:
        msg = (
            f"Remote storage unavailable ({code or 'transient error'}). "
            "Retry or fall back to the HF hub."
        )
    else:
        msg = f"Storage error [{code or 'unknown'}]: {message}"

    return StorageError(msg, code=code, transient=transient, auth_failure=auth)


def _humanize_endpoint_error(err: EndpointConnectionError, *, endpoint: str) -> StorageError:
    msg = (
        f"Could not reach endpoint {endpoint or '(default)'}. "
        "Check the URL, your network connectivity, and any proxy/VPN settings."
    )
    return StorageError(msg, code="EndpointConnectionError", transient=True)


def _humanize_no_credentials_error(err: Exception) -> StorageError:
    """Wrap boto's NoCredentialsError / PartialCredentialsError as a StorageError.

    Boto raises this when no credential source is available *anywhere* (config,
    env, IAM, etc.). The credential check in ``doctor`` already surfaces this
    explicitly, so the wrapper exists mainly to keep callers from crashing on
    an unhandled exception type.
    """
    msg = (
        "No credentials available. Set storage.access_key/secret_key in your "
        "config OR export B2_APPLICATION_KEY_ID/B2_APPLICATION_KEY (or "
        "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)."
    )
    return StorageError(msg, code="NoCredentialsError", auth_failure=True)


@contextmanager
def _wrap_errors(bucket: str, endpoint: str) -> Iterator[None]:
    """Translate boto3 errors raised within the block into StorageError."""
    try:
        yield
    except ClientError as e:
        raise _humanize_client_error(e, bucket=bucket, endpoint=endpoint) from e
    except EndpointConnectionError as e:
        raise _humanize_endpoint_error(e, endpoint=endpoint) from e
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise _humanize_no_credentials_error(e) from e


class StorageBackend:
    def __init__(self, config: AppConfig, *, workers: int = 8):
        """Build a backend sized for ``workers`` concurrent operations.

        ``workers`` controls ``max_pool_connections`` on the underlying
        boto3 client. With boto3's default of 10, a thread pool of e.g. 16
        contends on the connection pool and emits urllib3 "Connection pool
        is full" warnings while requests stall. Sizing the pool to the
        worker count keeps every thread first-class.
        """
        self.config = config
        self.workers = workers
        self.bucket = config.storage.bucket
        self.prefix = config.remote_prefix
        self.endpoint = config.storage.endpoint
        self._client: S3Client | None = None

    @property
    def client(self) -> S3Client:
        if self._client is None:
            user_agent = get_user_agent(self.config.storage.endpoint)
            # Floor at boto3's default (10) so commands with low concurrency
            # (e.g. doctor, diff) keep their existing pool size.
            pool_size = max(self.workers, 10)
            boto_config = BotoConfig(
                user_agent_extra=user_agent,
                # Standard mode covers throttling + transient 5xx with sensible backoff.
                retries={"max_attempts": 5, "mode": "standard"},
                max_pool_connections=pool_size,
            )
            kwargs: dict = {
                "service_name": "s3",
                "region_name": self.config.storage.region or None,
                "config": boto_config,
            }
            if self.config.storage.endpoint:
                kwargs["endpoint_url"] = self.config.storage.endpoint
            if self.config.storage.access_key:
                kwargs["aws_access_key_id"] = self.config.storage.access_key
            if self.config.storage.secret_key:
                kwargs["aws_secret_access_key"] = self.config.storage.secret_key
            self._client = boto3.client(**kwargs)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as e:
            if is_not_found(e):
                return False
            raise _humanize_client_error(e, bucket=self.bucket, endpoint=self.endpoint) from e
        except EndpointConnectionError as e:
            raise _humanize_endpoint_error(e, endpoint=self.endpoint) from e
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise _humanize_no_credentials_error(e) from e

    def upload_file(self, local_path: Path, key: str) -> None:
        with _wrap_errors(self.bucket, self.endpoint):
            self.client.upload_file(str(local_path), self.bucket, self._key(key))

    def upload_bytes(self, data: bytes, key: str) -> None:
        with _wrap_errors(self.bucket, self.endpoint):
            self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with _wrap_errors(self.bucket, self.endpoint):
            self.client.download_file(self.bucket, self._key(key), str(local_path))

    def download_bytes(self, key: str) -> bytes:
        with _wrap_errors(self.bucket, self.endpoint):
            resp = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
            return resp["Body"].read()

    def delete(self, key: str) -> None:
        with _wrap_errors(self.bucket, self.endpoint):
            self.client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix)
        keys: list[str] = []
        with _wrap_errors(self.bucket, self.endpoint):
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    k = obj["Key"]
                    if self.prefix and k.startswith(self.prefix):
                        k = k[len(self.prefix) :]
                    keys.append(k)
        return keys

    def head_bucket(self) -> None:
        """Verify the bucket is reachable and accessible. Raises StorageError otherwise."""
        with _wrap_errors(self.bucket, self.endpoint):
            self.client.head_bucket(Bucket=self.bucket)
