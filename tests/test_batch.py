"""Tests for the minimal Anthropic Message Batches client (direktoro.batch).

Hermetic and SDK-free: every batch client is a hand-written stub, no network is
touched (the conftest guard would fail the suite if it were), and no
``anthropic`` SDK object is constructed. Covers the submit/poll/fetch/map round
trip, the loud per-request failure path (errored/expired/canceled and missing
ids, all captured — never one masking the rest), and the normalisation into the
live NormalisedResponse shape.

Costing a batch is not this module's job and is not tested here: a batch is
billed at the caller's batch rates, which the caller feeds to
``direktoro.cost.cost_from_rates`` (see tests/test_cost.py).
"""

import importlib
import sys
from types import SimpleNamespace as NS

import pytest

import direktoro.batch as batch
from direktoro.batch import (
    BatchResultError,
    build_batch_client,
    map_batch_results,
    normalise_batch_message,
    run_message_batch,
)
from direktoro.providers import (
    AnthropicAdapter, MissingAPIKey, NormalisedResponse,
    resolved_decoding_params)
from direktoro.registry import PROVIDER_ANTHROPIC
from direktoro.routing import call_identity_fields, canonical_json


# --------------------------------------------------------------------------- #
# stubs: a stand-in anthropic client exposing messages.batches.{create,retrieve,
# results}, plus batch message / result shapes
# --------------------------------------------------------------------------- #

def _usage(**kw):
    base = dict(input_tokens=0, output_tokens=0,
                cache_read_input_tokens=0, cache_creation_input_tokens=0)
    base.update(kw)
    return NS(**base)


def _block(text):
    return NS(type="text", text=text)


def _message(text="ok", *, usage=None, stop_reason="end_turn",
             model="claude-opus-4-8", content=None):
    return NS(content=content if content is not None else [_block(text)],
              usage=usage if usage is not None else _usage(),
              stop_reason=stop_reason, model=model)


def _succeeded(custom_id, message):
    return NS(custom_id=custom_id, result=NS(type="succeeded", message=message))


def _failed(custom_id, rtype, *, error=None):
    return NS(custom_id=custom_id, result=NS(type=rtype, error=error))


class _StubBatches:
    def __init__(self, *, results, statuses=("ended",)):
        self._results = results
        self._statuses = list(statuses)
        self.created_requests = None
        self.retrieve_calls = 0

    def create(self, *, requests):
        self.created_requests = requests
        return NS(id="batch_test")

    def retrieve(self, batch_id):
        i = min(self.retrieve_calls, len(self._statuses) - 1)
        status = self._statuses[i]
        self.retrieve_calls += 1
        return NS(id=batch_id, processing_status=status)

    def results(self, batch_id):
        return iter(self._results)


class _StubClient:
    def __init__(self, **kw):
        self.messages = NS(batches=_StubBatches(**kw))


def _requests(*ids):
    return [{"custom_id": cid,
             "params": {"model": "claude-opus-4-8", "max_tokens": 8,
                        "system": [{"type": "text", "text": "s"}],
                        "messages": [{"role": "user", "content": cid}]}}
            for cid in ids]


# --------------------------------------------------------------------------- #
# submit / poll / fetch / map — the happy path
# --------------------------------------------------------------------------- #

def test_run_message_batch_maps_results_by_custom_id():
    # Results come back OUT OF ORDER relative to the submitted requests.
    results = [
        _succeeded("b", _message("beta", model="m")),
        _succeeded("a", _message("alpha", model="m")),
    ]
    client = _StubClient(results=results)
    responses = run_message_batch(client, _requests("a", "b"),
                                  sleep=lambda *_: None, poll_interval=0)

    assert set(responses) == {"a", "b"}
    assert responses["a"].content[0].text == "alpha"   # keyed by id, not position
    assert responses["b"].content[0].text == "beta"
    assert isinstance(responses["a"], NormalisedResponse)
    # The submitted requests reached the client verbatim.
    assert [r["custom_id"] for r in client.messages.batches.created_requests] \
        == ["a", "b"]


def test_poll_waits_until_the_batch_ends():
    slept = []
    client = _StubClient(
        results=[_succeeded("a", _message())],
        statuses=("in_progress", "in_progress", "ended"))
    run_message_batch(client, _requests("a"),
                      sleep=slept.append, poll_interval=7)
    # Polled three times (two not-ended, then ended); slept once per not-ended.
    assert client.messages.batches.retrieve_calls == 3
    assert slept == [7, 7]


def test_poll_max_polls_bounds_a_wedged_batch():
    client = _StubClient(results=[], statuses=("in_progress",))
    with pytest.raises(BatchResultError, match="still processing"):
        batch.poll_batch(client, "batch_x", sleep=lambda *_: None,
                         poll_interval=0, max_polls=3)


# --------------------------------------------------------------------------- #
# loud per-request failure — captured per id, never a silent drop
# --------------------------------------------------------------------------- #

def test_failed_results_raise_capturing_every_failing_id():
    results = [
        _succeeded("ok", _message()),
        _failed("boom", "errored",
                error=NS(type="invalid_request", message="bad shape")),
        _failed("gone", "expired"),
        _failed("stopped", "canceled"),
    ]
    client = _StubClient(results=results)
    with pytest.raises(BatchResultError) as exc:
        run_message_batch(client, _requests("ok", "boom", "gone", "stopped"),
                          sleep=lambda *_: None, poll_interval=0)
    errors = exc.value.errors
    # All three failures captured, not just the first — and the succeeded id is
    # not silently returned as a partial result.
    assert set(errors) == {"boom", "gone", "stopped"}
    assert "errored" in errors["boom"] and "bad shape" in errors["boom"]
    assert errors["gone"] == "expired"
    assert errors["stopped"] == "canceled"


def test_a_missing_result_is_a_loud_failure():
    # Submitted "a" and "b", but the batch only returned "a".
    client = _StubClient(results=[_succeeded("a", _message())])
    with pytest.raises(BatchResultError) as exc:
        run_message_batch(client, _requests("a", "b"),
                          sleep=lambda *_: None, poll_interval=0)
    assert exc.value.errors == {"b": "missing"}


def test_map_batch_results_without_expected_ids_skips_completeness_check():
    # No expected_ids: only actual failures are raised on, and a short result
    # set is accepted — without the submitted ids there is nothing to compare
    # against, so completeness becomes the caller's question.
    out = map_batch_results([_succeeded("a", _message("x"))])
    assert set(out) == {"a"} and out["a"].content[0].text == "x"


# --------------------------------------------------------------------------- #
# normalisation into the live NormalisedResponse / NormalisedUsage shape
# --------------------------------------------------------------------------- #

def test_normalise_batch_message_matches_the_live_shape():
    msg = _message(
        "answered", stop_reason="end_turn", model="claude-opus-4-8",
        usage=_usage(input_tokens=100, output_tokens=20,
                     cache_read_input_tokens=50, cache_creation_input_tokens=1000))
    resp = normalise_batch_message(msg, raw_request={"model": "claude-opus-4-8"})

    assert resp.content[0].text == "answered"
    assert resp.provider == PROVIDER_ANTHROPIC
    assert resp.stop_reason == "end_turn"
    assert resp.resolved_model == "claude-opus-4-8"
    assert resp.raw_request == {"model": "claude-opus-4-8"}
    # Usage carries the four Anthropic-normalised counts through unchanged.
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 20
    assert resp.usage.cache_read_input_tokens == 50
    assert resp.usage.cache_creation_input_tokens == 1000


def test_map_batch_results_accepts_dict_shaped_results():
    # SDK objects are attribute-access; model_dump()/fixtures are dict-access.
    dict_result = {
        "custom_id": "d",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": "from a dict"}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "stop_reason": "end_turn",
                "model": "claude-opus-4-8",
            },
        },
    }
    out = map_batch_results([dict_result])
    assert out["d"].content[0]["text"] == "from a dict"
    assert out["d"].usage.input_tokens == 7
    assert out["d"].provider == PROVIDER_ANTHROPIC


# --------------------------------------------------------------------------- #
# client construction + SDK-free import
# --------------------------------------------------------------------------- #

def test_build_batch_client_missing_key_raises_before_any_sdk_import():
    with pytest.raises(MissingAPIKey, match="ANTHROPIC_API_KEY"):
        build_batch_client(env={})  # no key -> raises before importing anthropic


# --------------------------------------------------------------------------- #
# cache-write TTL split — the total alone cannot be priced
# --------------------------------------------------------------------------- #

def test_the_cache_write_ttl_split_carries_through():
    # 5-minute writes bill at 1.25x base input, 1-hour writes at 2x, and
    # `cache_creation_input_tokens` is their sum. A batch response that carried
    # only the sum would be priced at whichever single rate the caller guessed —
    # costing an hour-TTL write at the five-minute rate gives 1.25/2 of the true
    # charge, silently.
    msg = _message(usage=_usage(
        input_tokens=10, output_tokens=2,
        cache_creation_input_tokens=1400,
        cache_creation=NS(ephemeral_5m_input_tokens=400,
                          ephemeral_1h_input_tokens=1000)))
    resp = normalise_batch_message(msg)

    assert resp.usage.cache_creation_input_tokens == 1400
    assert resp.usage.cache_creation_5m_input_tokens == 400
    assert resp.usage.cache_creation_1h_input_tokens == 1000
    assert (resp.usage.cache_creation_5m_input_tokens
            + resp.usage.cache_creation_1h_input_tokens
            == resp.usage.cache_creation_input_tokens)


def test_an_absent_cache_creation_split_leaves_both_tiers_zero():
    # The nested split is a value a message need not carry (and the dict-shaped
    # fixtures below never do). Its absence must not crash and must not invent a
    # tier: zero means "no split reported", and the total still stands.
    resp = normalise_batch_message(
        _message(usage=_usage(cache_creation_input_tokens=900)))
    assert resp.usage.cache_creation_input_tokens == 900
    assert resp.usage.cache_creation_5m_input_tokens == 0
    assert resp.usage.cache_creation_1h_input_tokens == 0


def test_the_split_is_read_from_a_dict_shaped_message_too():
    out = map_batch_results([{
        "custom_id": "d",
        "result": {"type": "succeeded", "message": {
            "content": [], "stop_reason": "end_turn", "model": "m",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 30,
                      "cache_creation": {"ephemeral_5m_input_tokens": 10,
                                         "ephemeral_1h_input_tokens": 20}}}},
    }])
    assert out["d"].usage.cache_creation_5m_input_tokens == 10
    assert out["d"].usage.cache_creation_1h_input_tokens == 20


# --------------------------------------------------------------------------- #
# call identity — a batch-served call and the identical live call must not look
# like different calls
# --------------------------------------------------------------------------- #

_LIVE_MODEL = "claude-sonnet-4-6"   # accepts a temperature, so the block has
                                    # more in it than the output cap alone


def _batch_params(model=_LIVE_MODEL, **overrides):
    """A canonical batch request built the way a caller should build one: the
    decoding block comes from `resolved_decoding_params`, the same function the
    live adapter uses, so the two paths send the same parameters."""
    decoding = resolved_decoding_params(model, sampling={"temperature": 0.0}, max_tokens=512)
    params = {"model": model,
              "system": [{"type": "text", "text": "s"}],
              "messages": [{"role": "user", "content": "hi"}],
              **decoding}
    params.update(overrides)
    return params


class _LiveStream:
    def __init__(self, resp):
        self._resp = resp
        self.text_stream = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._resp


class _LiveClient:
    """The streaming surface `AnthropicAdapter` reaches for, so the same message
    can be put through the live path and the batch path and compared."""

    def __init__(self, resp):
        self.messages = NS(stream=lambda **kw: _LiveStream(resp))


def _live_response(message, params):
    return AnthropicAdapter(_LiveClient(message)).create_message(
        model=params["model"], system=params["system"],
        messages=params["messages"], max_tokens=params["max_tokens"],
        sampling={"temperature": params.get("temperature")})


def test_a_batch_response_matches_a_live_one_field_for_field():
    # The docstring on normalise_batch_message promises a batch response is
    # indistinguishable from a live one above the provider layer. Put the SAME
    # message through both paths and hold every field of that promise.
    params = _batch_params()
    message = _message("answer", model=_LIVE_MODEL,
                       usage=_usage(input_tokens=100, output_tokens=20,
                                    cache_read_input_tokens=5))
    live = _live_response(message, params)
    batched = run_message_batch(
        _StubClient(results=[_succeeded("a", message)]),
        [{"custom_id": "a", "params": params}],
        sleep=lambda *_: None, poll_interval=0)["a"]

    for field in ("provider", "base_url", "stop_reason", "resolved_model",
                  "wire_request", "decoding_params", "generation_id",
                  "served_provider", "reported_cost"):
        assert getattr(batched, field) == getattr(live, field), field
    assert batched.usage == live.usage
    assert batched.content == live.content


def test_decoding_params_reach_the_response_from_the_submitted_request():
    params = _batch_params()
    responses = run_message_batch(
        _StubClient(results=[_succeeded("a", _message(model=_LIVE_MODEL))]),
        [{"custom_id": "a", "params": params}],
        sleep=lambda *_: None, poll_interval=0)

    assert responses["a"].decoding_params == {"max_tokens": 512,
                                              "temperature": 0.0}


def test_the_call_identity_block_is_the_same_for_batch_and_live():
    # This is the consequence that matters: `direktoro.routing` documents
    # folding `response.decoding_params` into `call_identity_fields`, and that
    # block is what makes two runs at different caps or efforts comparable. An
    # empty block gives a batch-served call a DIFFERENT identity from the
    # identical live call, which quietly breaks that comparison.
    params = _batch_params()
    message = _message(model=_LIVE_MODEL)
    live = _live_response(message, params)
    batched = run_message_batch(
        _StubClient(results=[_succeeded("a", message)]),
        [{"custom_id": "a", "params": params}],
        sleep=lambda *_: None, poll_interval=0)["a"]

    def identity(resp):
        return canonical_json(call_identity_fields(
            _LIVE_MODEL, decoding_params=resp.decoding_params))

    assert identity(batched) == identity(live)
    # And the block is not vacuously equal: an empty one really would differ.
    assert identity(batched) != canonical_json(
        call_identity_fields(_LIVE_MODEL, decoding_params={}))


def test_per_request_decoding_params_are_not_shared_across_ids():
    a = _batch_params(max_tokens=64)
    b = _batch_params(max_tokens=4096)
    responses = run_message_batch(
        _StubClient(results=[_succeeded("a", _message(model=_LIVE_MODEL)),
                             _succeeded("b", _message(model=_LIVE_MODEL))]),
        [{"custom_id": "a", "params": a}, {"custom_id": "b", "params": b}],
        sleep=lambda *_: None, poll_interval=0)

    assert responses["a"].decoding_params["max_tokens"] == 64
    assert responses["b"].decoding_params["max_tokens"] == 4096


def test_mapping_results_without_decoding_params_still_works():
    # Called directly with raw results and nothing else, there is no submitted
    # request to read, so the block is empty rather than wrong.
    out = map_batch_results([_succeeded("a", _message())])
    assert out["a"].decoding_params == {}


# --------------------------------------------------------------------------- #
# a duplicate custom_id is a loud failure, not a silent overwrite
# --------------------------------------------------------------------------- #

def test_a_duplicate_custom_id_raises_rather_than_keeping_the_last():
    # One slot per id: a repeat would otherwise overwrite the first response and
    # return a full-looking map holding an arbitrary pick between two answers,
    # with nothing to tell the caller it happened.
    results = [_succeeded("a", _message("first")),
               _succeeded("a", _message("second"))]
    with pytest.raises(BatchResultError) as exc:
        map_batch_results(results)
    assert set(exc.value.errors) == {"a"}
    assert "2 results" in exc.value.errors["a"]


def test_a_duplicate_does_not_hide_other_failures():
    results = [_succeeded("a", _message()),
               _succeeded("a", _message()),
               _failed("boom", "errored")]
    with pytest.raises(BatchResultError) as exc:
        map_batch_results(results)
    assert set(exc.value.errors) == {"a", "boom"}


def test_a_duplicate_is_caught_through_run_message_batch():
    client = _StubClient(results=[_succeeded("a", _message()),
                                  _succeeded("a", _message())])
    with pytest.raises(BatchResultError) as exc:
        run_message_batch(client, _requests("a"),
                          sleep=lambda *_: None, poll_interval=0)
    assert "cannot be determined" in exc.value.errors["a"]


def test_distinct_ids_are_unaffected():
    out = map_batch_results([_succeeded("a", _message("x")),
                             _succeeded("b", _message("y"))])
    assert {k: v.content[0].text for k, v in out.items()} == {"a": "x", "b": "y"}


def test_importing_batch_needs_no_sdk(monkeypatch):
    # Drop cached copies and poison the SDKs; re-importing direktoro.batch must
    # still succeed, because the anthropic import lives inside build_batch_client.
    for name in list(sys.modules):
        if (name == "direktoro.batch"
                or name == "anthropic" or name.startswith("anthropic.")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    mod = importlib.import_module("direktoro.batch")
    assert hasattr(mod, "run_message_batch")
