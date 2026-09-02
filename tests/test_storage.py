"""Tests for storage backend user-agent, error handling, and humanized errors."""

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

from hf_cache_sync.config import AppConfig, StorageConfig
from hf_cache_sync.storage import (
    B2_USER_AGENT,
    DEFAULT_USER_AGENT,
    StorageBackend,
    StorageError,
    _humanize_client_error,
    _humanize_endpoint_error,
    _humanize_no_credentials_error,
    _is_backblaze_endpoint,
    _wrap_errors,
    get_user_agent,
)


def test_is_backblaze_endpoint():
    assert _is_backblaze_endpoint("https://s3.us-west-000.backblazeb2.com") is True
    assert _is_backblaze_endpoint("https://s3.eu-central-003.backblazeb2.com") is True
    assert _is_backblaze_endpoint("https://backblazeb2.com") is True
    assert _is_backblaze_endpoint("https://S3.US-WEST-000.BACKBLAZEB2.COM.") is True
    assert _is_backblaze_endpoint("https://s3.amazonaws.com") is False
    assert _is_backblaze_endpoint("https://minio.local:9000") is False
    assert _is_backblaze_endpoint("https://evil-backblazeb2.com") is False
    assert _is_backblaze_endpoint("https://evil.example/backblazeb2.com") is False
    assert _is_backblaze_endpoint("") is False


def test_get_user_agent_b2():
    ua = get_user_agent("https://s3.us-west-000.backblazeb2.com")
    assert ua.startswith(B2_USER_AGENT)


def test_get_user_agent_other():
    ua = get_user_agent("https://s3.amazonaws.com")
    assert ua.startswith(DEFAULT_USER_AGENT)


def _make_client_error(code: str, message: str = "boom") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="HeadObject",
    )


@pytest.mark.parametrize(
    "code,must_be_auth,must_be_transient,expected_keyword",
    [
        ("InvalidAccessKeyId", True, False, "credentials rejected"),
        ("SignatureDoesNotMatch", True, False, "credentials rejected"),
        ("AccessDenied", True, False, "Access denied"),
        ("NoSuchBucket", False, False, "not found"),
        ("PermanentRedirect", False, False, "not in the configured region"),
        ("ServiceUnavailable", False, True, "unavailable"),
        ("SlowDown", False, True, "unavailable"),
    ],
)
def test_humanize_client_error_codes(code, must_be_auth, must_be_transient, expected_keyword):
    err = _humanize_client_error(_make_client_error(code), bucket="b", endpoint="https://e")
    assert isinstance(err, StorageError)
    assert err.code == code
    assert err.auth_failure is must_be_auth
    assert err.transient is must_be_transient
    assert expected_keyword.lower() in str(err).lower()


def test_humanize_client_error_unknown_code_includes_message():
    err = _humanize_client_error(
        _make_client_error("WeirdCode", "weird message"),
        bucket="b",
        endpoint="https://e",
    )
    assert "WeirdCode" in str(err)
    assert "weird message" in str(err)


def test_humanize_endpoint_error():
    inner = EndpointConnectionError(endpoint_url="https://nope.example.com")
    err = _humanize_endpoint_error(inner, endpoint="https://nope.example.com")
    assert err.code == "EndpointConnectionError"
    assert err.transient is True
    assert "Could not reach" in str(err)


def test_storage_error_chain_preserves_cause():
    """The original boto error must be available via __cause__ for debugging."""
    inner = _make_client_error("AccessDenied")
    try:
        raise _humanize_client_error(inner, bucket="b", endpoint="") from inner
    except StorageError as caught:
        assert caught.__cause__ is inner


def test_humanize_no_credentials_error():
    """NoCredentialsError must surface as an actionable auth StorageError."""
    err = _humanize_no_credentials_error(NoCredentialsError())
    assert isinstance(err, StorageError)
    assert err.code == "NoCredentialsError"
    assert err.auth_failure is True
    assert "B2_APPLICATION_KEY_ID" in str(err)


@pytest.mark.parametrize(
    "exc_cls",
    [NoCredentialsError, PartialCredentialsError],
)
def test_wrap_errors_translates_credential_errors(exc_cls):
    """Crashes from boto's credential resolver must turn into StorageError."""
    if exc_cls is PartialCredentialsError:
        inner = exc_cls(provider="env", cred_var="AWS_SECRET_ACCESS_KEY")
    else:
        inner = exc_cls()
    with pytest.raises(StorageError) as excinfo, _wrap_errors("b", "https://e"):
        raise inner
    assert excinfo.value.auth_failure is True
    assert excinfo.value.__cause__ is inner


def _backend(workers: int) -> StorageBackend:
    return StorageBackend(AppConfig(storage=StorageConfig(bucket="b", region="r")), workers=workers)


@pytest.mark.parametrize(
    "workers,expected_pool",
    [
        (1, 10),  # below boto3 default — keep boto3's default of 10
        (8, 10),  # CLI default — unchanged from prior behavior
        (10, 10),  # exactly the floor
        (16, 16),  # above default — pool grows to fit
        (32, 32),  # high concurrency
    ],
)
def test_max_pool_connections_scales_with_workers(workers, expected_pool):
    """boto3's default pool of 10 stalls thread pools larger than that.
    Backend must size max_pool_connections >= workers (with a 10 floor)."""
    backend = _backend(workers)
    assert backend.client.meta.config.max_pool_connections == expected_pool


def test_default_workers_keeps_boto_default_pool():
    """Backends constructed without an explicit workers (doctor, diff)
    should keep boto3's default pool size to avoid surprising behavior."""
    backend = StorageBackend(AppConfig(storage=StorageConfig(bucket="b", region="r")))
    assert backend.client.meta.config.max_pool_connections == 10
