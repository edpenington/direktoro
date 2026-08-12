"""Minimal Anthropic Message Batches client, scoped to bulk independent calls.

A large set of mutually independent requests is the Batch API's workload: no
request needs another's answer, latency does not matter, and the provider bills
a batch at a discount to its live rates. This module is that client and nothing
more — it is not a general SDK wrapper. It:

  * submits a batch of canonical (Anthropic-shaped) requests, each tagged with a
    caller-chosen ``custom_id``;
  * polls the batch until it ends;
  * fetches the per-request results and maps them back BY ``custom_id`` (Batch
    results arrive in arbitrary order — never rely on position);
  * normalises each *succeeded* result into the same
    :class:`~direktoro.providers.NormalisedResponse` /
    :class:`~direktoro.providers.NormalisedUsage` shape the live
    :class:`~direktoro.providers.AnthropicAdapter` returns, so everything above
    the provider layer consumes a batch response exactly like a live one.

Loud per-request failure
------------------------
An ``errored`` / ``expired`` / ``canceled`` per-request result, a submitted
``custom_id`` that never came back at all, or one that came back more than once,
is a loud failure, never a silent drop or a silent overwrite.
:func:`map_batch_results` collects *every* such id (so one bad request does not
hide the others) and raises :class:`BatchResultError` carrying the full
``{custom_id: reason}`` map. A batch with any failed request yields no partial
result set.

Cost
----
A batch is billed at its own rates, and this module states none of them: what a
batch costs is arithmetic over the caller's rate card
(:func:`direktoro.cost.cost_from_rates`), which the caller feeds its batch rates
rather than its live ones. Nothing here has to be kept in step with a provider's
pricing page.

SDK-free import
---------------
The ``anthropic`` SDK is imported lazily inside :func:`build_batch_client` and is
never needed when the caller injects its own client (the normal case, and how the
tests run). Importing this module therefore never imports ``anthropic`` — the
package-wide "``import direktoro`` succeeds without the SDKs" property (asserted
in ``tests/test_public_api.py``) holds for the batch surface too.
"""

from __future__ import annotations

import time as _time

from direktoro.providers import MissingAPIKey, NormalisedResponse, NormalisedUsage
from direktoro.registry import PROVIDER_ANTHROPIC, SAMPLING_PARAMS
from direktoro.wire_log import response_to_dict

# Default gap between batch-status polls, in seconds. Batches usually finish
# well within an hour; a leisurely poll keeps the request count low. Tests inject
# their own sleep and a status stub that ends immediately, so this value never
# actually delays the suite.
BATCH_POLL_INTERVAL_SECONDS = 30

# Terminal batch processing status: results are ready to fetch.
_ENDED = "ended"

# Per-request result outcome that carries a usable message.
_SUCCEEDED = "succeeded"


class BatchResultError(RuntimeError):
    """One or more ``custom_id``\\ s did not yield exactly one usable response.

    Raised by :func:`map_batch_results` after EVERY failing ``custom_id`` has
    been collected, so a single bad request never masks the rest. ``errors`` is
    the ``{custom_id: reason}`` map, and the reason is one of:

    * the provider's per-request result type (``errored`` / ``expired`` /
      ``canceled``, with the error message where the SDK provides one);
    * ``"missing"`` — a submitted id that produced no result at all;
    * a duplication reason — the id came back more than once, so no single
      response can be said to be its answer.

    All three are the same failure seen from different sides: the batch cannot
    be turned into one response per submitted id. A batch with any of them
    yields no partial responses.
    """

    def __init__(self, errors: dict):
        self.errors = dict(errors)
        detail = "; ".join(f"{cid}: {reason}"
                           for cid, reason in sorted(self.errors.items()))
        super().__init__(
            f"{len(self.errors)} batch request(s) did not succeed: {detail}")


# ---------------------------------------------------------------------------
# Normalisation (batch success -> the live NormalisedResponse shape)
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    """Read ``key`` from a batch message whether it is an SDK object or a dict.

    The SDK returns typed objects (attribute access); hand-written fixtures and
    ``model_dump()`` output are dicts (item access). Supporting both keeps the
    tests hermetic without constructing SDK types.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalise_batch_message(message, *, raw_request=None, decoding_params=None):
    """Normalise one succeeded batch message into a :class:`NormalisedResponse`.

    Mirrors :meth:`AnthropicAdapter._normalise` field-for-field so a batch
    response is indistinguishable from a live one above the provider layer:
    ``content`` is the list of Anthropic-shaped blocks, ``usage`` splits cached
    reads out of full-price input (never double-counted) and carries the
    cache-write TTL split the live path carries, and provider / stop reason /
    resolved model carry through.

    Three fields are worth naming, because "field-for-field" is a promise and
    not all three come from the message:

    * ``base_url`` is ``None`` because that is what a LIVE Anthropic call
      records too — every Anthropic registry entry pins no base URL. An
      equality, not a placeholder.
    * ``raw_request`` and ``decoding_params`` describe the request rather than
      the response, so they can only come from the caller that submitted it.
      :func:`run_message_batch` threads ``decoding_params`` through from the
      submitted params; a caller mapping raw results by hand supplies whichever
      it holds, and what it omits stays ``None`` / ``{}``. ``wire_request``
      mirrors the live path's equality — the canonical request IS the
      Anthropic wire — so the supplied ``raw_request`` rides under both names,
      and an audit path reads ``wire_request`` on a batch response exactly as
      it does on a live one. Both names ALIAS the caller's own params dict
      rather than snapshotting it: a caller that mutates a shared params
      template between submissions rewrites what its earlier records say was
      sent. Write the audit entry before reusing the dict, or pass a copy.

    ``decoding_params`` matters beyond the audit trail: a consumer folding it
    into :func:`direktoro.routing.call_identity_fields` — which that function
    documents as the way to key runs by what was actually sent — gets a
    DIFFERENT identity block for an empty one. That is the block that makes two
    runs at different caps or efforts comparable, so leaving it empty would make
    a batch-served call and the identical live call look like different calls.
    Pass the decoding subset of the submitted request (see
    :data:`_DECODING_KEYS`), which is what the live adapter records.
    """
    u = _get(message, "usage")
    # The two cache-write TTL tiers bill at different multiples of the base
    # input rate and `cache_creation_input_tokens` is their sum, so the split is
    # carried alongside it exactly as the live adapter carries it — a batch
    # response that dropped it would be the cheaper one to price wrongly. Read
    # defensively: `usage.cache_creation` is a nested value a message need not
    # carry, and its absence leaves both tiers at zero, which is what "no split
    # reported" means (see `NormalisedUsage`).
    creation = _get(u, "cache_creation", None)
    usage = NormalisedUsage(
        input_tokens=_get(u, "input_tokens", 0) or 0,
        output_tokens=_get(u, "output_tokens", 0) or 0,
        cache_read_input_tokens=_get(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=_get(u, "cache_creation_input_tokens", 0) or 0,
        cache_creation_5m_input_tokens=(
            _get(creation, "ephemeral_5m_input_tokens", 0) or 0),
        cache_creation_1h_input_tokens=(
            _get(creation, "ephemeral_1h_input_tokens", 0) or 0),
    )
    return NormalisedResponse(
        content=list(_get(message, "content", None) or []),
        usage=usage,
        resolved_model=_get(message, "model", None),
        stop_reason=_get(message, "stop_reason", None),
        provider=PROVIDER_ANTHROPIC,
        # Not a placeholder: an Anthropic id pins no base URL, so the live
        # adapter records None here too.
        base_url=None,
        raw_request=raw_request,
        raw_response=response_to_dict(message),
        # Same equality as the live adapter: the canonical request is the
        # Anthropic wire request, so it rides under both names.
        wire_request=raw_request,
        decoding_params=decoding_params if decoding_params is not None else {},
    )


# The Anthropic request keys that `providers.resolved_decoding_params` emits,
# and therefore the ones that make up a response's `decoding_params` on this
# provider: the output cap, the sampling controls the model accepts, and the
# thinking / effort pair when a call asks for them. The sampling names are
# DERIVED from `registry.SAMPLING_PARAMS` — the one list of the controls this
# layer sends — so a control added there is carried into batch call identity
# too, rather than dropped from it by a name this tuple never learned.
_DECODING_KEYS = ("max_tokens", *SAMPLING_PARAMS, "thinking", "output_config")


def _submitted_decoding_params(params):
    """The ``decoding_params`` block for a submitted batch request.

    Read off the request as SUBMITTED rather than re-derived from the model id.
    Re-deriving would make this function refuse a batch it can perfectly well
    read back — an id outside the registry, or a shape the resolver declines to
    emit — and would answer "what would we send today" when the honest question
    is "what did this batch send". A key the request did not carry is absent
    from the block, and a key carried as ``None`` is dropped too, because
    ``resolved_decoding_params`` never emits a None decoding value and a block
    that did would not match the live one it is meant to equal.
    """
    return {key: value for key in _DECODING_KEYS
            if (value := _get(params, key)) is not None}


def _result_error_reason(result):
    """A short reason string for a non-succeeded per-request result."""
    rtype = _get(result, "type", "unknown")
    error = _get(result, "error", None)
    if error is not None:
        etype = _get(error, "type", None)
        message = _get(error, "message", None)
        parts = [str(p) for p in (etype, message) if p]
        if parts:
            return f"{rtype} ({': '.join(parts)})"
    return str(rtype)


# ---------------------------------------------------------------------------
# Submit / poll / fetch / map
# ---------------------------------------------------------------------------

def submit_batch(client, requests):
    """Create a Message Batch and return its id.

    ``requests`` is a list of ``{"custom_id": str, "params": dict}`` entries,
    where ``params`` is a canonical (Anthropic-shaped) Messages request — the
    same ``model`` / ``max_tokens`` / ``system`` / ``messages`` a live call
    sends. ``client`` is an ``anthropic.Anthropic`` (or a stub exposing
    ``messages.batches.create``); this module never builds one implicitly, so a
    caller with no key and no client cannot accidentally spend.
    """
    created = client.messages.batches.create(requests=list(requests))
    return _get(created, "id")


def poll_batch(client, batch_id, *, sleep=None, poll_interval=None,
               max_polls=None):
    """Poll ``batch_id`` until its ``processing_status`` is ``"ended"``.

    Returns the final batch object. ``sleep`` (default :func:`time.sleep`) and
    ``poll_interval`` (default :data:`BATCH_POLL_INTERVAL_SECONDS`) are injectable
    so the suite drives this with no real delay. ``max_polls`` bounds the loop
    (``None`` = unbounded, the real-run default); exceeding it raises rather than
    spinning forever against a wedged batch.
    """
    sleep = sleep or _time.sleep
    interval = (poll_interval if poll_interval is not None
                else BATCH_POLL_INTERVAL_SECONDS)
    polls = 0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if _get(batch, "processing_status") == _ENDED:
            return batch
        polls += 1
        if max_polls is not None and polls >= max_polls:
            raise BatchResultError(
                {batch_id: f"still processing after {polls} poll(s)"})
        sleep(interval)


def fetch_batch_results(client, batch_id):
    """Return the per-request results for ``batch_id`` as a list.

    Thin wrapper over ``client.messages.batches.results`` (a streaming iterator);
    materialising it to a list lets :func:`map_batch_results` make two passes
    (collect successes, then raise on the full failure set)."""
    return list(client.messages.batches.results(batch_id))


def map_batch_results(raw_results, *, expected_ids=None, decoding_params=None,
                      raw_requests=None):
    """Map raw per-request results to ``{custom_id: NormalisedResponse}``.

    Each succeeded result is normalised (see :func:`normalise_batch_message`) and
    keyed by its ``custom_id``. Any ``errored`` / ``expired`` / ``canceled``
    result — and, when ``expected_ids`` is given, any submitted id with no result
    at all — is recorded, and after ALL results are processed a single
    :class:`BatchResultError` is raised carrying every failing id. So one bad
    request neither hides the others nor is silently dropped, and a batch with
    any failure returns no partial map.

    A ``custom_id`` returned MORE than once is a failure of the same kind and is
    raised the same way. The map has one slot per id, so a repeat would
    otherwise overwrite the first response and hand back a full-looking map in
    which one entry is an arbitrary pick between two answers — with no way for a
    caller to know it happened. Which of the two is right is not this function's
    call to make, so it makes none.

    ``decoding_params`` is an optional ``{custom_id: dict}`` mapping of the
    decoding block each request was submitted with, threaded onto that request's
    response so a batch-served call and the identical live call carry the same
    call identity (see :func:`normalise_batch_message`). ``raw_requests`` is
    the same shape for the submitted request itself, threaded through as
    ``raw_request`` / ``wire_request`` so the audit fields match the live
    path's too. :func:`run_message_batch` builds both from the submitted
    requests; an id with no entry gets an empty block / ``None``.
    """
    decoding_by_id = decoding_params or {}
    requests_by_id = raw_requests or {}
    responses: dict = {}
    errors: dict = {}
    seen: dict = {}
    for item in raw_results:
        custom_id = _get(item, "custom_id")
        seen[custom_id] = seen.get(custom_id, 0) + 1
        result = _get(item, "result")
        if _get(result, "type") == _SUCCEEDED:
            responses[custom_id] = normalise_batch_message(
                _get(result, "message"),
                raw_request=requests_by_id.get(custom_id),
                decoding_params=decoding_by_id.get(custom_id))
        else:
            errors[custom_id] = _result_error_reason(result)

    # After the loop, so every duplicate is counted before any is reported, and
    # so a duplicated id whose first result also failed is described by the
    # duplication rather than by that failure: with two results in play the
    # per-request reason no longer identifies which request it came from.
    for custom_id, count in seen.items():
        if count > 1:
            errors[custom_id] = (
                f"{count} results returned for one custom_id; an id identifies "
                f"exactly one request, so which response belongs to it cannot "
                f"be determined")

    if expected_ids is not None:
        returned = set(responses) | set(errors)
        for cid in expected_ids:
            if cid not in returned:
                errors[cid] = "missing"

    if errors:
        raise BatchResultError(errors)
    return responses


def run_message_batch(client, requests, *, sleep=None, poll_interval=None,
                      max_polls=None):
    """Submit ``requests``, wait for the batch to end, and return the responses.

    Orchestrates :func:`submit_batch` -> :func:`poll_batch` ->
    :func:`fetch_batch_results` -> :func:`map_batch_results`, checking every
    submitted ``custom_id`` came back. Returns ``{custom_id: NormalisedResponse}``
    for the whole batch, or raises :class:`BatchResultError` if any request
    failed, went missing, or came back twice. The polling knobs pass straight
    through to :func:`poll_batch`.

    Because this function holds the submitted requests, it is where the decoding
    params each response should carry are read off (see
    :func:`_submitted_decoding_params`) and threaded through, so a response
    returned from here has the same call-identity block the live adapter would
    have given it.
    """
    expected_ids = [_get(r, "custom_id") for r in requests]
    decoding_params = {_get(r, "custom_id"):
                       _submitted_decoding_params(_get(r, "params"))
                       for r in requests}
    raw_requests = {_get(r, "custom_id"): _get(r, "params")
                    for r in requests}
    batch_id = submit_batch(client, requests)
    poll_batch(client, batch_id, sleep=sleep, poll_interval=poll_interval,
               max_polls=max_polls)
    raw_results = fetch_batch_results(client, batch_id)
    return map_batch_results(raw_results, expected_ids=expected_ids,
                             decoding_params=decoding_params,
                             raw_requests=raw_requests)


# ---------------------------------------------------------------------------
# Client construction (lazy SDK import)
# ---------------------------------------------------------------------------

def build_batch_client(*, env=None, api_key_env="ANTHROPIC_API_KEY"):
    """Construct an ``anthropic.Anthropic`` client for batch submission.

    The ``anthropic`` SDK is imported here, lazily, so importing this module
    needs no SDK. Reads the key from ``api_key_env`` (default the Anthropic key)
    in ``env`` (the process environment by default) and raises
    :class:`~direktoro.providers.MissingAPIKey` before any network call when it
    is unset. SDK-level retries are disabled (``max_retries=0``) so retry policy
    is owned above, not doubled. Callers that already hold a client (tests inject
    a stub) never reach here.
    """
    import os

    environ = os.environ if env is None else env
    api_key = environ.get(api_key_env, "")
    if not api_key:
        raise MissingAPIKey(
            f"environment variable {api_key_env} is not set; it is needed to "
            f"submit an Anthropic message batch.")
    import anthropic

    return anthropic.Anthropic(api_key=api_key, max_retries=0)
