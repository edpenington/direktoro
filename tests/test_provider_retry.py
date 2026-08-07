"""create_message_with_retry: transient provider errors retry on the
backoff schedule; non-transient errors propagate immediately."""

import pytest

from direktoro.providers import (
    RETRY_BACKOFF_SECONDS,
    ProviderError,
    ProviderRateLimitError,
    ProviderRetryableError,
    create_message_with_retry,
)


class _FlakyAdapter:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def create_message(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"ok": True, "kwargs": kwargs}


def test_transient_errors_retry_then_succeed():
    adapter = _FlakyAdapter([
        ProviderRetryableError("overloaded", status_code=529),
        ProviderRateLimitError("429"),
    ])
    slept = []
    out = create_message_with_retry(
        adapter, _sleep=slept.append, model="m")
    assert out["ok"] is True
    assert adapter.calls == 3
    assert slept == list(RETRY_BACKOFF_SECONDS[:2])


def test_exhausted_retries_reraise():
    adapter = _FlakyAdapter(
        [ProviderRetryableError("still down", status_code=529)]
        * (len(RETRY_BACKOFF_SECONDS) + 1))
    slept = []
    with pytest.raises(ProviderRetryableError):
        create_message_with_retry(adapter, _sleep=slept.append)
    assert adapter.calls == len(RETRY_BACKOFF_SECONDS) + 1
    assert slept == list(RETRY_BACKOFF_SECONDS)


def test_non_transient_error_propagates_immediately():
    adapter = _FlakyAdapter([ProviderError("bad request")])
    slept = []
    with pytest.raises(ProviderError):
        create_message_with_retry(adapter, _sleep=slept.append)
    assert adapter.calls == 1
    assert slept == []


def test_on_retry_called_once_per_retried_failure():
    errors = [
        ProviderRetryableError("overloaded", status_code=529),
        ProviderRateLimitError("429"),
    ]
    adapter = _FlakyAdapter(list(errors))
    seen = []
    out = create_message_with_retry(
        adapter, _sleep=lambda _: None,
        on_retry=lambda attempt, delay, err: seen.append(
            (attempt, delay, err)),
        model="m")
    assert out["ok"] is True
    assert [(a, d) for a, d, _ in seen] == [
        (0, RETRY_BACKOFF_SECONDS[0]), (1, RETRY_BACKOFF_SECONDS[1])]
    assert [e for _, _, e in seen] == errors


def test_on_retry_not_called_on_final_reraise():
    adapter = _FlakyAdapter(
        [ProviderRateLimitError("429")] * (len(RETRY_BACKOFF_SECONDS) + 1))
    seen = []
    with pytest.raises(ProviderRateLimitError):
        create_message_with_retry(
            adapter, _sleep=lambda _: None,
            on_retry=lambda *a: seen.append(a))
    assert len(seen) == len(RETRY_BACKOFF_SECONDS)
