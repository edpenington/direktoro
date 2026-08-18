"""Provider adapters: one thin translation layer per LLM provider.

Callers speak THIS PACKAGE'S canonical request/response format, and the
adapters translate it to and from each provider's wire. The canonical format
is direktoro's contract with its consumers: system as a string or a list of
text blocks, messages as content-block lists (`text` / `image` / `tool_use` /
`tool_result`), tool definitions as `{name, description, input_schema}`, and
decoding params named `max_tokens` plus whatever sampling controls the caller
specified. Its block vocabulary coincides byte-for-byte with the Anthropic
wire — a deliberate choice that makes one adapter's translation the identity —
but the contract is this package's: nothing above the adapter sees a provider
wire format, which is what lets a caller store, resume, replay or render a
conversation in one shape regardless of who served it.

One rule about tool-call OUTPUT that no wire enforces uniformly, stated here
because a consumer cannot discover it from the request format: a BOOLEAN
FIELD IN A TOOL CALL'S INPUT MUST NOT ARRIVE NULL. The Anthropic wire
rejects a null boolean server-side; the OpenAI-family tools this package
emits do not (the Responses translation sets `strict: False`, and the Chat
Completions tool shape carries no strict field at all), so a null boolean
passes those wires silently. A consumer validating tool-call output
therefore enforces the rule itself rather than relying on the wire to catch
it.

An adapter translates the canonical request to its provider's wire format,
makes the call, and translates the response back into a `NormalisedResponse`
whose `.content` is a list of canonical block objects and whose `.usage`
is normalised token counts. One piece of calling code therefore reads every
provider's response, and every ADAPTER response carries the `wire_request`
that was actually sent — the batch mapper threads it through when handed
the submitted requests — ready for `direktoro.wire_log.redact_wire_request`
and the caller's audit log.

Adapters do not retry: `create_message` makes one call and, on a provider API
error, raises a normalised `ProviderError` (or a retryable subclass). A caller
that wants retry loops around `create_message` and catches the normalised
exceptions — `create_message_with_retry` is the ready-made loop — and a caller
that does not lets them propagate. Nothing retries underneath either: a client
`build_adapter` constructs is built with the SDK's own retry disabled, so the
retry policy a caller sees is the whole of it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from direktoro.errors import ProviderError
from direktoro.registry import (
    EFFORT_LEVELS, PROVIDER_ANTHROPIC, SAMPLING_PARAMS, THINKING_ADAPTIVE,
    THINKING_BUDGET,
    THINKING_DISABLED, THINKING_DISPLAYS, THINKING_MODES,
    WIRE_CHAT_COMPLETIONS, WIRE_RESPONSES, model_info)
from direktoro.routing import (
    ProviderRouteMismatch, assert_served_upstream, provider_object)
from direktoro.wire_log import response_to_dict


# ---------------------------------------------------------------------------
# Normalised exceptions
#
# The base is imported from `direktoro.errors`, a leaf module `direktoro.routing`
# can import too, so `ProviderRouteMismatch` shares this tree without an import
# cycle (see that module). Importing it here keeps
# `direktoro.providers.ProviderError` resolving for every consumer that already
# names it.
# ---------------------------------------------------------------------------

class ProviderRateLimitError(ProviderError):
    """Rate-limited (HTTP 429). Retryable with backoff."""


class ProviderRetryableError(ProviderError):
    """A transient provider failure worth retrying, other than a rate limit.

    `status_code` carries the HTTP status when the failure had one — a 5xx
    server error, including Anthropic's 529 overloaded — and stays None when
    there was no status to record: a connection that could not be established
    never reached one, and a failure reported INSIDE the body of a 200 (the
    Responses `server_error` of `_translate_responses_error`) does not have one
    to give. All of them are retryable for the same reason: nothing was served,
    so trying again cannot pay twice for one answer.
    """

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


# Backoff schedule for transient provider errors (rate limits, 5xx including
# Anthropic 529 overloaded, and connections that never established). Four
# retries, about 40 seconds of total waiting; a provider outage longer than that
# fails the run loudly.
RETRY_BACKOFF_SECONDS = (2, 5, 11, 23)


def create_message_with_retry(adapter, *, _sleep=None, on_retry=None,
                              **kwargs):
    """Call `adapter.create_message`, retrying transient provider errors.

    Retries `ProviderRateLimitError` and `ProviderRetryableError` on
    the RETRY_BACKOFF_SECONDS schedule, then re-raises. Non-transient
    `ProviderError`s propagate immediately.

    Which provider failure lands in which bucket is decided by the translators
    (`_translate_anthropic_error`, `_translate_openai_error`): rate limits, 5xx
    responses and failed connections are retryable, because none of them was
    served; a TIMEOUT deliberately is not, because a timed-out request may
    already have been served and billed while its response was lost, and this
    loop would then pay for the same answer several times over and record one.
    A caller whose calls are cheap or idempotent can catch the plain
    `ProviderError` a timeout raises and retry on its own terms.

    A failed attempt raises inside this function, so it never reaches whatever
    audit log the caller writes after a successful call. A caller that wants
    the retries recorded passes `on_retry(attempt, delay_seconds, error)`,
    invoked once per retried failure, and logs from there. TWO PROPERTIES OF
    IT A LOG IS READ AGAINST, so that reading does not have to be inferred
    from the tests that pin them: `attempt` is 0-INDEXED, so the first failure
    reports attempt 0 alongside the first delay; and it is NOT CALLED ON THE
    FINAL RE-RAISE, so the number of events is the number of rungs actually
    used. A call that exhausts the schedule emits len(RETRY_BACKOFF_SECONDS)
    events and then raises, and a call that recovers emits one event per rung
    it needed — which is what separates recovery from exhaustion in a log that
    holds only these events and the outcome.

    This is one backoff loop shared across a whole call site. A caller running
    calls concurrently and wanting per-call backoff state should keep its own
    loop instead and catch the same two exception types.
    """
    import time as _time
    sleep = _sleep or _time.sleep
    for attempt, delay in enumerate((*RETRY_BACKOFF_SECONDS, None)):
        try:
            return adapter.create_message(**kwargs)
        except (ProviderRateLimitError, ProviderRetryableError) as e:
            if delay is None:
                raise
            if on_retry is not None:
                on_retry(attempt, delay, e)
            sleep(delay)


# ---------------------------------------------------------------------------
# Normalised response
# ---------------------------------------------------------------------------

@dataclass
class NormalisedUsage:
    """Token usage in Anthropic-normalised semantics.

    `input_tokens` counts ONLY full-price (cache-miss) input; cached reads are
    reported separately in `cache_read_input_tokens` so pricing never
    double-counts them. `cache_creation_input_tokens` is Anthropic's cache
    write count and is zero for providers with no separate cache-write charge
    (OpenAI, GLM).

    CACHE WRITES ARE NOT ONE RATE, AND THE TOTAL ALONE CANNOT BE PRICED.
    Anthropic writes a cache entry at one of two time-to-live tiers and bills
    them differently: a 5-minute write at 1.25x the model's base input rate, a
    1-hour write at 2x. `cache_creation_input_tokens` is their SUM, so pricing
    it at a single rate is right only when every write in the call happened to
    be at that tier — an hour's worth of writes costed at the five-minute rate
    comes out at 1.25/2 of the real charge, nearly two fifths low, and silently.
    `cache_creation_5m_input_tokens` and `cache_creation_1h_input_tokens` carry
    the split, and a caller pricing cache writes should multiply each by its own
    rate rather than reach for the total. They sum to
    `cache_creation_input_tokens` on a response that reports the split, and are
    both zero where none is reported — every non-Anthropic provider, and any
    Anthropic response that omits the nested counts — in which case the total is
    the only figure there is and pricing it needs the caller to know which tier
    it asked for.

    ADDING A FIELD: APPEND IT AT THE END.
    A new field goes after the last one, with a default, and nowhere else. This
    record is public and constructed POSITIONALLY by anything downstream that
    synthesises a usage figure, so a field inserted in the middle rebinds every
    positional argument after it, silently and without a TypeError: an output
    count lands in `cache_read_input_tokens`, a cache read lands in a cache
    write. The consequence is a cost, and the direction is unbounded. Appending
    is the only edit that cannot do that, and a default is what keeps existing
    constructions valid.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_creation_5m_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0


@dataclass
class NormalisedResponse:
    """One API response, normalised.

    `content` is a list of Anthropic-shaped block objects (each exposes `.type`
    and, per type, `.id` / `.name` / `.input` or `.text`), so code that walks
    the blocks — pulling out tool calls, or reading the first block's text —
    works unchanged across providers.

    `raw_request` is the canonical request as sent, in canonical vocabulary,
    for the caller's audit log: a sampling control the model refuses is
    absent here exactly as it is absent from the wire, while a value the wire
    respells or supplies from the registry (the output cap's wire key, a
    registry-defaulted reasoning level) appears in `wire_request` and
    `decoding_params`, the wire-exact records — not here.
    `wire_request` is the provider's actual wire request as sent — on the
    Anthropic path it is the canonical request byte-for-byte (the canonical
    format IS that wire), and it is recorded under both names so an audit
    path reads ONE field whoever served the call. `raw_response` is the
    provider's response as a plain dict (`wire_log.response_to_dict`), never
    an SDK object.
    `decoding_params` records the decoding parameters actually sent after
    quirks were applied (for example sampling controls omitted for models
    that refuse them), so a stored record of the call is self-contained.

    The routing fields are populated only for gateway-served (OpenRouter)
    responses and stay None for direct calls, which have no gateway equivalent:

      - `generation_id`: OpenRouter's per-call generation id (the completion's
        top-level `id`, `gen-...`), an external receipt a caller can look the
        call up by at the gateway.
      - `served_provider`: the upstream that actually served the call
        (OpenRouter's top-level `provider` attribution), checked against the
        Route's pin before the response is returned.
      - `reported_cost`: the USD cost the gateway itself reports for the call
        (OpenRouter `usage.cost`) — a record of what was CHARGED, not an
        estimate of it, and the authoritative figure for a routed call. Direct
        providers report no such figure and leave this None; costing a direct
        call is arithmetic over the caller's own rates
        (`direktoro.cost.cost_from_rates`).
    """

    content: list
    usage: NormalisedUsage
    resolved_model: Any = None
    stop_reason: Any = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    raw_request: Optional[dict] = None
    raw_response: Any = None
    wire_request: Optional[dict] = None
    decoding_params: dict = field(default_factory=dict)
    generation_id: Optional[str] = None
    served_provider: Optional[str] = None
    reported_cost: Optional[float] = None


# ---------------------------------------------------------------------------
# Thinking / reasoning-effort request spec
# ---------------------------------------------------------------------------

class ThinkingUnsupported(ValueError):
    """A requested thinking / effort shape the model's endpoint would reject.

    Raised BEFORE the request is sent, from `resolved_decoding_params`, whenever
    the registry knows the shape is wrong for that model: `budget_tokens` on a
    family that removed it, an effort level the model does not have, disabled
    thinking above the effort ceiling that allows it, a model that has not
    declared its thinking surface at all. The failure this prevents is a 400 on
    a paid call, so it is deliberately loud and deliberately early. A
    `ValueError` subclass, so a caller already guarding registry lookups with
    `except ValueError` catches it unchanged.
    """


@dataclass(frozen=True)
class Thinking:
    """What a caller asks for on ONE call: a thinking mode and/or an effort.

    Both fields are optional and each maps to a distinct wire parameter, so a
    caller can set the depth without touching the mode or vice versa:

      - `mode`: `"adaptive"` (the model decides when and how much to think),
        `"disabled"` (no thinking), or `"budget"` (a fixed thinking-token
        budget — pre-4.6 endpoints only; the 4.7+ families return 400 for it).
        Emitted as Anthropic's `thinking` parameter.
      - `effort`: one of `EFFORT_LEVELS`. Emitted as Anthropic's
        `output_config: {"effort": ...}`. Independent of `mode`: effort governs
        how much the model spends overall, thinking governs whether it reasons
        first.
      - `budget_tokens`: required with `mode="budget"`, ignored otherwise. Must
        be at least the endpoint's minimum (1024) and STRICTLY LESS than the
        call's `max_tokens`.
      - `display`: `"summarized"` to get readable thinking text back, or
        `"omitted"` for empty thinking blocks. Valid with either ON-mode
        (`"adaptive"` and `"budget"`); invalid with `mode="disabled"`, where
        there is nothing to display. The default is per-model, not per-field:
        `"omitted"` from the 4.7 generation on, `"summarized"` before it. The
        setting changes what comes BACK, never what is billed.

    NOTE ON DISPLAY AND IDENTITY. `display` is folded into call identity like
    every other emitted parameter, so turning it on to read a run's reasoning
    changes the fingerprint and invalidates fingerprint-keyed cached results.
    That is deliberate — this layer's rule is to fingerprint exactly what is
    sent, and a request that carries `display` IS a different request — but it
    makes a debug flag a spend decision. Decide it once for a whole set of calls
    rather than toggling it between them.

    Frozen and hashable, so a spec is a constant a config can carry. Passing it
    is entirely optional: a call that omits it sends no thinking parameters at
    all and leaves each model's own default in force.

    NOTE ON CAPS. `max_tokens` caps thinking AND response text together on every
    Claude model, so raising effort or enabling thinking on a call whose cap was
    sized without it buys less answer, not more. direktoro deliberately does not
    adjust the cap for you — silently resizing a caller's cap would be a spend
    decision made behind their back — but it will refuse a `budget_tokens` that
    cannot fit, which is the one case where the arithmetic is unambiguous.
    """

    mode: Optional[str] = None
    effort: Optional[str] = None
    budget_tokens: Optional[int] = None
    display: Optional[str] = None

    def __post_init__(self):
        if self.mode is None and self.effort is None:
            raise ValueError(
                "Thinking() sets neither mode nor effort, so it would emit "
                "nothing; pass None instead of an empty spec, or name what you "
                "want.")
        if self.mode is not None and self.mode not in THINKING_MODES:
            raise ValueError(
                f"Thinking.mode {self.mode!r} is not a known mode; known modes "
                f"are {list(THINKING_MODES)}")
        if self.effort is not None and self.effort not in EFFORT_LEVELS:
            raise ValueError(
                f"Thinking.effort {self.effort!r} is not a known level; known "
                f"levels are {list(EFFORT_LEVELS)}")
        if self.budget_tokens is not None:
            if self.mode != THINKING_BUDGET:
                raise ValueError(
                    "Thinking.budget_tokens is only meaningful with "
                    f"mode={THINKING_BUDGET!r}; got mode={self.mode!r}")
            if not isinstance(self.budget_tokens, int) or \
                    self.budget_tokens < 1:
                raise ValueError(
                    "Thinking.budget_tokens must be a positive int, got "
                    f"{self.budget_tokens!r}")
        if self.display is not None:
            if self.display not in THINKING_DISPLAYS:
                raise ValueError(
                    f"Thinking.display {self.display!r} is not a known "
                    f"setting; known settings are {list(THINKING_DISPLAYS)}")
            # `display` rides the thinking block, so it needs a mode that emits
            # one, and it is invalid with disabled thinking (nothing to
            # display). It IS valid with budget mode: Anthropic documents
            # `display` alongside `type: "enabled"` as well as
            # `type: "adaptive"`, and the 4.6-generation and pre-4.6 endpoints
            # accept that pair, so requiring adaptive would refuse a live shape.
            if self.mode == THINKING_DISABLED:
                raise ValueError(
                    "Thinking.display is invalid with mode='disabled': "
                    "disabled thinking produces no thinking block, so there is "
                    "nothing to display. Drop the display, or ask for "
                    f"mode={THINKING_ADAPTIVE!r} / {THINKING_BUDGET!r}.")
            if self.mode is None:
                raise ValueError(
                    "Thinking.display needs a thinking mode to ride on; an "
                    "effort-only spec emits no thinking block, so the display "
                    f"would be silently dropped. Name mode={THINKING_ADAPTIVE!r}"
                    f" or mode={THINKING_BUDGET!r} alongside it.")


# The thinking fields a per-role decoding block in an application config may
# carry, mapped config-key -> `Thinking` field.
_CONFIG_THINKING_KEYS = {
    "thinking_mode": "mode",
    "thinking_effort": "effort",
    "thinking_budget_tokens": "budget_tokens",
    "thinking_display": "display",
}


def split_decoding_config(mapping):
    """Split one role's decoding block from an application config.

    The block is the caller's own YAML/JSON mapping, carried opaquely: the
    sampling controls (`SAMPLING_PARAMS`) and the thinking fields
    (`thinking_mode`, `thinking_effort`, `thinking_budget_tokens`,
    `thinking_display`) side by side, so the application never needs to know
    which key is which — this function is what knows.

    Returns `(sampling, thinking)`, exactly the two arguments
    `resolved_decoding_params` takes: a dict of the named sampling controls
    or None when none were named, and a `Thinking` spec or None. An unknown
    key raises ValueError naming what is accepted, so a misspelled control
    fails at config load instead of being silently dropped. Value validation
    happens where it always happens — the `Thinking` constructor for the
    spec's shape, `resolved_decoding_params` for what the model's endpoint
    accepts — so this function adds no second opinion on either. A key set
    to null reads as unspecified, same as the resolver's own convention, so
    a config may carry a fixed key set and leave values empty.
    """
    if mapping is None:
        return None, None
    if not isinstance(mapping, dict):
        raise ValueError(
            f"a decoding block must be a mapping of parameter names to "
            f"values, got {type(mapping).__name__}.")
    # key=str so a malformed block with a non-string key still raises THIS
    # error, naming the key, rather than a TypeError from sorting.
    unknown = sorted(set(mapping) - set(SAMPLING_PARAMS)
                     - set(_CONFIG_THINKING_KEYS), key=str)
    if unknown:
        raise ValueError(
            f"unknown decoding key(s) {unknown}; a decoding block accepts "
            f"the sampling controls {list(SAMPLING_PARAMS)} and the thinking "
            f"fields {sorted(_CONFIG_THINKING_KEYS)}.")
    sampling = {}
    for k in SAMPLING_PARAMS:
        v = mapping.get(k)
        if v is None:
            # Null reads as unspecified, sampling and thinking keys alike.
            continue
        # YAML spells one intent two ways (`0` and `0.0`), and the resolved
        # value folds into call identity byte-for-byte, so the float-valued
        # controls are normalised here: two configs that mean the same call
        # must not fingerprint apart. top_k is integral and is left alone.
        if k in ("temperature", "top_p") and isinstance(v, (int, float)) \
                and not isinstance(v, bool):
            v = float(v)
        sampling[k] = v
    spec = {attr: mapping[key]
            for key, attr in _CONFIG_THINKING_KEYS.items()
            if mapping.get(key) is not None}
    thinking = Thinking(**spec) if spec else None
    return (sampling or None), thinking


def _checked_display(model, support, display):
    """`display` if `model` accepts it, else refuse before the call is billed.

    `thinking.display` is a per-model capability like the modes and the effort
    levels: it postdates the oldest ids this registry still carries, and an
    endpoint that does not know the field rejects the request. Declaring it on
    `ThinkingSupport.displays` is what lets the seam say "not on this model"
    instead of emitting a field on the strength of a generation guess.
    """
    if display is None:
        return None
    if display not in support.displays:
        accepted = list(support.displays)
        raise ThinkingUnsupported(
            f"model {model!r} does not accept thinking.display {display!r}"
            + (f"; it accepts {accepted}." if accepted else
               "; its registry entry declares no display settings, so this "
               "layer will not send one.")
            + " Sending it would return a 400.")
    return display


def _thinking_params(model, info, thinking, *, max_tokens):
    """The wire fragment for `thinking`, or `{}` when none was asked for.

    Validates the request against the model's registry `ThinkingSupport` and
    raises `ThinkingUnsupported` for anything the endpoint would reject, so the
    400 never happens on a paid call. The keys depend on the wire: Anthropic
    gets `thinking` / `output_config`, which merge straight into the Anthropic
    request; the OpenAI-family wires resolve to a single `reasoning_effort`
    level (see `_openai_family_thinking_params`), which the Responses branch
    re-spells as `reasoning: {"effort": ...}`.

    One case is satisfied rather than refused: asking to DISABLE thinking on a
    model that does not accept an off-switch but also does not think unless
    asked emits nothing, because omitting the parameter already gives exactly
    the requested behaviour. The caller gets what it asked for and the
    identity block records the same absence a `thinking=None` call would, which
    is honest: the two calls are byte-identical on the wire and behave
    identically.
    """
    if thinking is None:
        return {}

    support = info.thinking
    if support is None:
        raise ThinkingUnsupported(
            f"model {model!r} declares no thinking support in the registry, so "
            f"direktoro will not guess a shape its endpoint accepts. Add a "
            f"`thinking=ThinkingSupport(...)` to its registry entry, recording "
            f"the modes and effort levels the endpoint was verified to take.")

    if info.provider != PROVIDER_ANTHROPIC:
        return _openai_family_thinking_params(model, info, support, thinking)

    params = {}

    if thinking.effort is not None:
        if thinking.effort not in support.efforts:
            accepted = list(support.efforts)
            raise ThinkingUnsupported(
                f"model {model!r} does not accept reasoning effort "
                f"{thinking.effort!r}"
                + (f"; it accepts {accepted}." if accepted else
                   "; it has no effort parameter at all (pre-4.6 models error "
                   "on one).")
                + " Sending it would return a 400.")
        params["output_config"] = {"effort": thinking.effort}

    mode = thinking.mode
    if mode is None:
        return params

    if mode == THINKING_ADAPTIVE:
        if THINKING_ADAPTIVE not in support.modes:
            raise ThinkingUnsupported(
                f"model {model!r} does not accept adaptive thinking "
                f"(`{{'type': 'adaptive'}}`); it accepts "
                f"{list(support.modes)}.")
        block = {"type": "adaptive"}
        display = _checked_display(model, support, thinking.display)
        if display is not None:
            block["display"] = display
        params["thinking"] = block

    elif mode == THINKING_DISABLED:
        if THINKING_DISABLED not in support.modes:
            if not support.default_on:
                # Omitting the parameter already means "no thinking" here, so
                # the request is satisfied by emitting nothing. See the
                # docstring: same wire, same behaviour, same identity.
                return params
            raise ThinkingUnsupported(
                f"model {model!r} thinks by default and does not accept "
                f"`{{'type': 'disabled'}}`, so thinking cannot be turned off "
                f"on it; it accepts {list(support.modes)}.")
        effective = thinking.effort or support.default_effort
        cap = support.disabled_max_effort
        if cap is not None and effective is not None and \
                EFFORT_LEVELS.index(effective) > EFFORT_LEVELS.index(cap):
            raise ThinkingUnsupported(
                f"model {model!r} accepts `thinking={{'type': 'disabled'}}` "
                f"only at effort {cap!r} or below, and this call is at "
                f"{effective!r}"
                + ("" if thinking.effort is not None else
                   " (the level in force when `effort` is omitted)")
                + ". The pair returns a 400. Lower the effort, or leave "
                  "thinking enabled.")
        params["thinking"] = {"type": "disabled"}

    else:  # THINKING_BUDGET
        if THINKING_BUDGET not in support.modes:
            raise ThinkingUnsupported(
                f"model {model!r} REJECTS `budget_tokens` with a 400 (the "
                f"Claude 4.7-generation and later removed the fixed thinking "
                f"budget). Ask for Thinking(mode='adaptive', effort=...) "
                f"instead; it accepts {list(support.modes)}.")
        budget = thinking.budget_tokens
        if budget is None:
            raise ThinkingUnsupported(
                f"Thinking(mode='budget') for model {model!r} needs "
                f"budget_tokens; without one there is no budget to send.")
        if budget < support.budget_min:
            raise ThinkingUnsupported(
                f"budget_tokens {budget} is below the endpoint minimum "
                f"{support.budget_min} for model {model!r}; the API rejects it.")
        if max_tokens is None:
            # The one arithmetic this layer calls unambiguous is `budget <
            # max_tokens`, and with no cap there is nothing to check. Refusing
            # here rather than skipping the guard keeps the error on the thing
            # that is actually missing: a request carrying a budget and no cap
            # is rejected by the API on `max_tokens`, which names the wrong
            # parameter and sends the caller looking in the wrong place.
            raise ThinkingUnsupported(
                f"Thinking(mode='budget') for model {model!r} needs a "
                f"max_tokens to size the budget against: budget_tokens must be "
                f"strictly less than max_tokens (the cap covers thinking AND "
                f"response text), and max_tokens is required by the API in any "
                f"case. Pass the call's cap.")
        if budget >= max_tokens:
            raise ThinkingUnsupported(
                f"budget_tokens {budget} must be strictly less than max_tokens "
                f"{max_tokens} for model {model!r} (the cap covers thinking "
                f"AND response text); the API rejects it. Raise max_tokens or "
                f"lower the budget.")
        block = {"type": "enabled", "budget_tokens": budget}
        display = _checked_display(model, support, thinking.display)
        if display is not None:
            block["display"] = display
        params["thinking"] = block

    return params


def _openai_family_thinking_params(model, info, support, thinking):
    """The OpenAI-family rendering of a thinking spec: one reasoning level.

    These wires carry a single reasoning control — `reasoning_effort` on Chat
    Completions, `reasoning: {"effort": ...}` on Responses — so a spec
    resolves to at most one level string, returned here under the Chat
    Completions spelling:

      - a named EFFORT is validated against the entry's declared levels and
        sent as itself;
      - mode "disabled" is sent as the level "none" — the off-switch the
        endpoints declaring the disabled mode were probed to honour (zero
        reasoning tokens; live 2026-08-12) — and cannot be combined with a
        named effort, since one wire key cannot carry two levels;
      - mode "adaptive" alone emits nothing: on an endpoint that reasons by
        default it IS the omitted-state behaviour, and emitting a level for
        it would fold a value into call identity that nobody chose.

    `budget_tokens` and `display` are Anthropic wire concepts with no
    rendering on these wires, so a spec carrying either is refused rather
    than partly honoured.
    """
    if thinking.budget_tokens is not None:
        raise ThinkingUnsupported(
            f"model {model!r} is served by {info.provider!r}, whose wire has "
            f"no `budget_tokens`: a fixed thinking budget cannot be rendered "
            f"for it. Name an effort instead, or drop the spec.")
    if thinking.display is not None:
        raise ThinkingUnsupported(
            f"model {model!r} is served by {info.provider!r}, whose wire has "
            f"no `thinking.display`: the setting cannot be rendered for it.")

    params = {}
    if thinking.effort is not None:
        if thinking.effort not in support.efforts:
            accepted = list(support.efforts)
            raise ThinkingUnsupported(
                f"model {model!r} does not accept reasoning effort "
                f"{thinking.effort!r}"
                + (f"; it accepts {accepted}." if accepted else
                   "; its endpoint takes no reasoning-effort parameter at "
                   "all.")
                + " The call would fail after it was billed.")
        params["reasoning_effort"] = thinking.effort

    mode = thinking.mode
    if mode is None:
        return params

    if mode == THINKING_ADAPTIVE:
        if THINKING_ADAPTIVE not in support.modes:
            raise ThinkingUnsupported(
                f"model {model!r} does not accept adaptive thinking; it "
                f"accepts {list(support.modes)}.")
        return params

    if mode == THINKING_DISABLED:
        # A wire property, so it is checked before any entry property: this
        # wire carries ONE reasoning level, and disabling rides it, so the
        # contradictory pair is refused whether or not the entry declares an
        # off-switch — an entry-first check would silently drop both halves
        # on an entry that declares none.
        if thinking.effort is not None:
            raise ThinkingUnsupported(
                f"Thinking(mode='disabled', effort={thinking.effort!r}) "
                f"cannot be rendered for model {model!r}: this wire carries "
                f"ONE reasoning level, disabling rides it as \"none\", and a "
                f"second level cannot be sent alongside. Drop one of the two.")
        if THINKING_DISABLED not in support.modes:
            if support.default_on:
                raise ThinkingUnsupported(
                    f"model {model!r} reasons by default and its endpoint "
                    f"refuses the off-switch, so thinking cannot be turned "
                    f"off on it; it accepts {list(support.modes)}.")
            if (info.quirks or {}).get("reasoning_effort"):
                raise ThinkingUnsupported(
                    f"model {model!r} sends its registry-default reasoning "
                    f"level on every call and its endpoint declares no "
                    f"off-switch, so a disabled request cannot be honoured "
                    f"by omission — the default would run anyway.")
            # Omitting the parameter already means "no reasoning" here, so
            # the request is satisfied by emitting nothing. See the
            # docstring: same wire, same behaviour, same identity.
            return {}
        return {"reasoning_effort": "none"}

    # THINKING_BUDGET (a budget WITH tokens was already refused above).
    raise ThinkingUnsupported(
        f"model {model!r} is served by {info.provider!r}, whose wire has no "
        f"fixed thinking budget; Thinking(mode='budget') cannot be rendered "
        f"for it.")


# The `thinking.type` values that mean thinking is ACTIVE for this request.
# `disabled` is not one of them, and an effort-only spec emits no thinking key
# at all.
_THINKING_ON_TYPES = ("adaptive", "enabled")

# The window `top_p` is documented to be allowed in on a request that also
# turns thinking on, inclusive at both ends (Anthropic's thinking
# documentation, read 2026-08-01, quoted in `_refuse_sampling_with_thinking`).
_TOP_P_THINKING_WINDOW = (0.95, 1.0)

# The controls the documentation names as incompatible with active thinking on
# the 4.6-generation-and-earlier endpoints. Stated, not inferred by excluding
# `top_p` from whatever `sending` happens to carry: a control added to
# `SAMPLING_PARAMS` later is an unestablished case, and an unestablished case
# is sent for the endpoint to answer, never refused by elimination.
_THINKING_INCOMPATIBLE_SAMPLING = ("temperature", "top_k")


def _top_p_rides_with_thinking(value):
    """Whether `value` is inside the documented `top_p`-with-thinking window.

    The documentation says "allowed at values between 0.95 and 1"; this reads
    that as the closed interval — both endpoints in — which is this layer's
    reading of the sentence, not a separately established endpoint fact. A
    non-numeric value is outside the window: it is a numeric range, so a
    value that cannot be compared against it is not in it, and the refusal
    that follows names the range the endpoint documents.
    """
    low, high = _TOP_P_THINKING_WINDOW
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return low <= value <= high


def _refuse_sampling_with_thinking(model, sending, thinking_params):
    """Refuse the sampling controls a call that turns thinking on cannot carry.

    Anthropic's thinking documentation (read 2026-08-01): on the families whose
    entries declare a `rejects_sampling` set — Opus 5, Opus 4.8, Opus 4.7 and
    Sonnet 5 — a non-default `temperature` / `top_p` / `top_k` is a 400 on
    EVERY request, and those models never reach here with one. "On older
    models, the restriction applies only while thinking is on: `temperature`
    and `top_k` are incompatible with thinking, and `top_p` is allowed at
    values between 0.95 and 1."

    So on an active-thinking request this refuses `temperature` and `top_k`,
    and refuses `top_p` only outside `_TOP_P_THINKING_WINDOW`. An in-window
    `top_p` goes to the wire alongside the thinking block, because that is the
    pair the endpoint documents as legal and refusing it would refuse a call
    that works.

    That per-CALL interaction is the one shape `ThinkingSupport` cannot express,
    because it is not a property of the model alone: Sonnet 4.6 and Haiku 4.5
    both accept a temperature, and both reject it once the same request asks
    them to think. It is not a property of the parameter alone either, which is
    why it lives here rather than beside `rejects_sampling`: that field answers
    "does this endpoint take this control", a question with one answer per
    (model, param), while this answers "may these two be sent together", which
    has an answer only once both halves of the request are known. Resolving the
    two independently would emit `{"temperature": 0.0, "thinking": {"type":
    "enabled", ...}}` — a 400 on a paid call, from the layer whose stated job is
    to refuse those.

    `sending` is the sampling params this call would actually send, AFTER the
    model's declared refusals have been applied: a param the endpoint drops
    anyway is not a conflict, and naming it here would refuse a call that would
    have worked.

    REFUSED, not silently dropped, unlike a declared refusal. There the model
    never takes the parameter, so omitting it is the only honest rendering.
    Here the caller has asked for two things that cannot both hold and either
    could be the one it cares about — dropping the sampling control would change
    the sampling distribution of a scientific run behind its back, and dropping
    the thinking would change the reasoning. Naming both outs and stopping is
    the only choice this layer is entitled to make.
    """
    if not sending:
        return
    block = thinking_params.get("thinking")
    if not block or block.get("type") not in _THINKING_ON_TYPES:
        return
    low, high = _TOP_P_THINKING_WINDOW
    incompatible = sorted(name for name in sending
                          if name in _THINKING_INCOMPATIBLE_SAMPLING)
    if incompatible:
        named = ", ".join(f"`{k}`" for k in incompatible)
        raise ThinkingUnsupported(
            f"model {model!r} accepts {named}, but not on a request that "
            f"also turns thinking on ({block['type']!r}): on the "
            f"4.6-generation and earlier endpoints `temperature` and `top_k` "
            f"are incompatible with active thinking and the pair returns a "
            f"400. Choose one — drop the sampling params from this call to "
            f"think, or ask for Thinking(mode='disabled') / no thinking spec "
            f"to keep sampling control. (`top_p` is the one control that "
            f"rides alongside thinking here, at {low} to {high}. The 4.7+ "
            f"models settle it the other way: they refuse the sampling "
            f"controls outright, so this layer already omits them.)")
    top_p = sending.get("top_p")
    if top_p is not None and not _top_p_rides_with_thinking(top_p):
        raise ThinkingUnsupported(
            f"top_p={top_p!r} cannot be sent to model {model!r} on a request "
            f"that also turns thinking on ({block['type']!r}): the "
            f"4.6-generation and earlier endpoints allow `top_p` with active "
            f"thinking only between {low} and {high} inclusive, and return a "
            f"400 outside that window. Move it into the window, or ask for "
            f"Thinking(mode='disabled') / no thinking spec to sample as you "
            f"like.")


def _refuse_out_of_band_sampling(model, info, sending):
    """Refuse a sampling value outside the model's documented range.

    `Model.sampling_bands` records, per param, the (low, high) range the
    endpoint's reference documents. A value outside it fails at the endpoint —
    after the reviewer-stage run ahead of it has already been billed, if the
    caller exercises roles in sequence — so it is refused here, before any
    spend, by the one layer that knows which model the value is bound for.
    A param with no declared band passes through: nothing was established,
    and the endpoint's own answer settles it.
    """
    for name in SAMPLING_PARAMS:
        if name not in sending:
            continue
        band = info.sampling_bands.get(name)
        if band is None:
            continue
        value = sending[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name}={value!r} for model {model!r} is not a number; the "
                f"endpoint documents a numeric range {band[0]} to {band[1]}.")
        if not band[0] <= value <= band[1]:
            raise ValueError(
                f"{name}={value} is outside the range {band[0]} to {band[1]} "
                f"documented for model {model!r} (see its registry entry); "
                f"the endpoint refuses it, so it is refused here before the "
                f"call is billed.")


# The floor for `_refuse_starving_cap`: DIREKTORO'S OWN POLICY FIGURE, not an
# endpoint fact — no vendor documents a minimum viable cap for a reasoning
# call, and asserting one as theirs would outrun the evidence. 2048 is small
# enough that any deliberately sized cap clears it (Anthropic's migration
# guidance starts `max_tokens` at 64000 for the top efforts, thirty times
# higher) and large enough to catch a cap that is wrong by construction. The
# guard exists to catch those, never to size caps.
_THINKING_CAP_FLOOR = 2048


def _reason_the_call_thinks(info, thinking):
    """Why this call runs reasoning, or None when it does not.

    Returns a short cause phrase for `_refuse_starving_cap`'s message: an
    explicit on-mode; a named effort on a wire whose single reasoning dial
    turning-on IS the effort key; or the entry's declared `default_on` with
    nothing disabling it. An entry with no declared `ThinkingSupport` returns
    None — nothing was established, so there is nothing to guard.
    """
    support = info.thinking
    if support is None:
        return None
    if thinking is not None:
        if thinking.mode == THINKING_DISABLED:
            return None
        if thinking.mode in (THINKING_ADAPTIVE, THINKING_BUDGET):
            return f"the call asks for {thinking.mode!r} thinking"
        if thinking.effort is not None and \
                info.provider != PROVIDER_ANTHROPIC:
            return ("the call names a reasoning effort, which turns "
                    "reasoning on for this wire")
    if support.default_on:
        return ("thinking is on by default for this model and the call "
                "does not disable it")
    return None


def _refuse_starving_cap(model, info, thinking, *, max_tokens):
    """Refuse a cap a thinking call cannot answer within.

    The one refusal in this layer aimed at a request the endpoint would
    ACCEPT. `max_tokens` caps thinking plus response text together, so a
    reasoning call under a starving cap spends the cap on reasoning and
    truncates or empties the answer — a paid call whose failure is silent,
    and which a sequential caller repeats once per field. Refusing it here
    turns a whole wasted run into one loud error before any spend.

    The floor is `_THINKING_CAP_FLOOR`, this package's own policy figure
    (see its comment); the message says whose number it is. A budget-mode
    spec that STATES its `budget_tokens` is exempt: the caller did its own
    arithmetic, and `budget_tokens < max_tokens` is already enforced where
    the budget is validated.
    """
    if max_tokens is None:
        return
    if thinking is not None and thinking.mode == THINKING_BUDGET \
            and thinking.budget_tokens is not None:
        return
    cause = _reason_the_call_thinks(info, thinking)
    if cause is None:
        return
    if max_tokens < _THINKING_CAP_FLOOR:
        raise ThinkingUnsupported(
            f"max_tokens {max_tokens} cannot fit a reasoning call on model "
            f"{model!r}: {cause}, the cap covers thinking AND response text "
            f"together, and below {_THINKING_CAP_FLOOR} — direktoro's own "
            f"floor, not the endpoint's — the response is truncated or "
            f"empty while still billed. Raise the cap, or disable thinking "
            f"on the call.")


# ---------------------------------------------------------------------------
# Resolved decoding params (single source of truth)
# ---------------------------------------------------------------------------

def resolved_decoding_params(model, *, max_tokens, sampling=None,
                             thinking=None):
    """The decoding params the adapter will actually send for `model`.

    One source of truth shared by the adapters (so the wire request matches)
    and the recorded call identity (so a fingerprint folds in exactly what is
    sent, no more and no less). The output-token cap rides under the wire's
    key: `max_output_tokens` for the OpenAI Responses API, `max_tokens` for
    Chat Completions and Anthropic.

    `sampling` is what the CALLER specified, as a mapping of names from
    `SAMPLING_PARAMS` to values — `{"temperature": 0.0}`, `{"temperature": 0.2,
    "top_p": 0.9}`, or None/`{}` for "specified nothing". A name is emitted only
    when the caller gave it AND the model does not declare it refused
    (`Model.rejects_sampling`). The four cases are deliberately distinct:

      - specified, accepted    -> sent, and folded into call identity.
      - specified, refused     -> absent from BOTH, so editing it never moves
                                  the identity of a model that would not take
                                  it. A caller that wants to tell its operator
                                  the value was inert compares what it
                                  specified against what this returns.
      - unspecified, accepted  -> absent from both, and the PROVIDER's own
                                  default applies. That default is the
                                  provider's business and may change; this
                                  layer neither pins nor records it, because
                                  inventing a value here would claim a fact
                                  nobody established.
      - unspecified, refused   -> absent from both, and nothing to report.

    A specified, accepted value is additionally checked against the model's
    documented range (`Model.sampling_bands`) and refused with a ValueError
    when it falls outside — the endpoint would reject it, so it fails here
    before the call is billed rather than after.

    A `reasoning_effort` quirk is the level in force when the caller
    specifies nothing, included under the wire's key (`reasoning={"effort":
    ...}` for Responses, `reasoning_effort` for Chat Completions). A
    caller-chosen level — a named effort, or "none" carrying a disabled mode —
    takes its place on the wire and in identity.

    `thinking` is an optional per-call `Thinking` spec (mode and/or reasoning
    effort). It defaults to None, which emits nothing and leaves each model's
    own default thinking behaviour in force. When supplied it is validated
    against the model's registry `ThinkingSupport` and rendered for the
    model's wire — Anthropic's `thinking` / `output_config` keys, or the
    single OpenAI-family reasoning level, where disabling rides as "none";
    a shape the endpoint would reject raises `ThinkingUnsupported` here,
    before any spend. A call that will think — an explicit on-mode, or a
    default-on model left undisabled — is also refused when `max_tokens`
    cannot fit the endpoint's minimum thinking budget plus any answer (see
    `_refuse_starving_cap`): the endpoint would accept it, and then spend the
    whole cap on reasoning. Because this
    function is the single source of truth for BOTH the wire request and the
    decoding-params block folded into call identity, a chosen mode or effort
    automatically becomes part of that identity: two runs differing only in
    reasoning effort produce different `decoding_params`, hence a different
    `direktoro.routing.call_identity_fields` block, hence different fingerprints
    in whatever a caller composes on top.

    A model's DEFAULT thinking behaviour is deliberately NOT emitted here.
    Claude Opus 5 and Sonnet 5 think when the parameter is omitted, but that is
    a property of the model id, which identity already carries; emitting it
    would fold the same fact in twice and move every identity block for it.
    Read it explicitly via `registry.thinking_support(model).default_on`.
    """
    info = model_info(model)
    quirks = info.quirks or {}
    # What the caller specified, minus what this endpoint refuses, in
    # SAMPLING_PARAMS order so the emitted dict does not depend on the caller's
    # key order or on set iteration. A None value reads as unspecified, so a
    # caller may carry a fixed set of keys and leave the ones it has no opinion
    # on empty.
    asked = dict(sampling or {})
    unknown = sorted(set(asked) - set(SAMPLING_PARAMS), key=str)
    if unknown:
        raise ValueError(
            f"unknown sampling parameter(s) {unknown}; this layer emits "
            f"{list(SAMPLING_PARAMS)}. A name it does not know would be "
            f"silently dropped from both the wire and the recorded identity.")
    sending = {name: asked[name] for name in SAMPLING_PARAMS
               if asked.get(name) is not None
               and name not in info.rejects_sampling}
    _refuse_out_of_band_sampling(model, info, sending)
    thinking_params = _thinking_params(
        model, info, thinking, max_tokens=max_tokens)
    _refuse_starving_cap(model, info, thinking, max_tokens=max_tokens)

    if info.provider == PROVIDER_ANTHROPIC:
        dec = {"max_tokens": max_tokens}
        # Sampling x thinking is a per-CALL interaction, not a per-model one,
        # so it is checked here rather than in `_thinking_params` — and against
        # what would actually be SENT, so a param the endpoint drops anyway
        # never refuses a call that would have worked.
        _refuse_sampling_with_thinking(model, sending, thinking_params)
        dec.update(sending)
        dec.update(thinking_params)
        return dec

    # The `reasoning_effort` quirk is the level in force when the caller says
    # nothing; a level the caller chose (a named effort, or "none" for a
    # disabled mode) takes its place on the wire and in identity.
    effort = thinking_params.get("reasoning_effort") \
        or quirks.get("reasoning_effort")

    if info.wire_api == WIRE_CHAT_COMPLETIONS:
        dec = {"max_tokens": max_tokens}
        if effort:
            dec["reasoning_effort"] = effort
        dec.update(sending)
        return dec

    # OpenAI Responses. The Responses API has no `top_k` parameter at all, so
    # a caller's top_k on an entry that does not refuse it cannot be sent —
    # refused here as a wire fact rather than crashing in the SDK call or
    # riding to the endpoint as a kwarg nothing reads.
    if "top_k" in sending:
        raise ValueError(
            f"model {model!r} is served on the OpenAI Responses wire, which "
            f"has no `top_k` parameter; the value cannot be sent. Drop "
            f"`top_k` from the ask for this model.")
    dec = {"max_output_tokens": max_tokens}
    if effort:
        dec["reasoning"] = {"effort": effort}
    dec.update(sending)
    return dec


def _sent_sampling(decoding):
    """The sampling controls a resolved decoding block actually sends.

    Only the output cap and the reasoning level are respelled per wire; a
    sampling control that reaches a wire does so under its `SAMPLING_PARAMS`
    name (a control a wire cannot spell at all — `top_k` on Responses — is
    refused by the resolver, never resolved), so the subset is read by name
    and needs no translation. This is what the canonical request's SAMPLING
    records: a control the model refuses is absent from `raw_request` exactly
    as it is absent from the wire and from `decoding_params`. Recording the
    caller's raw ask instead would make one audit field disagree with the
    other two on the one question they exist to answer. (`raw_request` stays
    canonical-vocabulary throughout — a registry-defaulted reasoning level or
    an Anthropic thinking block appears in `wire_request` and
    `decoding_params`, the wire-exact records, not here.)
    """
    return {name: decoding[name] for name in SAMPLING_PARAMS
            if name in decoding}


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """Adapter for Anthropic (Claude).

    The canonical request already IS the Anthropic wire request, so this
    adapter only applies the model's decoding quirks (Opus 4.7+ reject
    the sampling controls), streams the call, and normalises the SDK response
    object.
    """

    provider = PROVIDER_ANTHROPIC

    def __init__(self, client, *, base_url=None):
        self._client = client
        self.base_url = base_url

    def create_message(self, *, model, system, messages, max_tokens,
                       tools=None, tool_choice=None, sampling=None,
                       thinking=None):
        # Opus 4.7+ refuse the sampling controls; the resolver omits a refused
        # one so the wire and the recorded decoding params reflect exactly what
        # was sent. Its keys (max_tokens, the accepted sampling controls, and `thinking` /
        # `output_config` when a Thinking spec was passed) are already Anthropic
        # wire keys, so they merge straight into the wire request. `thinking`
        # defaults to None: omit it and the request carries no thinking
        # parameters at all, leaving the model's own default in force.
        decoding = resolved_decoding_params(
            model, sampling=sampling, max_tokens=max_tokens,
            thinking=thinking)
        wire = {
            "model": model,
            "system": system,
            "messages": messages,
            **decoding,
        }
        if tools is not None:
            wire["tools"] = tools
        if tool_choice is not None:
            wire["tool_choice"] = tool_choice

        try:
            with self._client.messages.stream(**wire) as stream:
                for _ in stream.text_stream:
                    pass
                response = stream.get_final_message()
        except Exception as exc:
            raise _translate_anthropic_error(exc)

        return self._normalise(response, wire, decoding)

    def _normalise(self, response, wire, decoding):
        u = getattr(response, "usage", None)
        # The 5-minute and 1-hour cache writes bill at different multiples of
        # the base input rate, so the per-tier split is carried alongside the
        # total: reporting only `cache_creation_input_tokens` leaves a caller
        # with a number it cannot price without guessing which tier produced it
        # (see NormalisedUsage). Read defensively — `usage.cache_creation` is a
        # nested object that a response need not carry, and `getattr` on a
        # missing attribute or on None yields the zero that means "no split
        # reported", which is exactly what the two counters mean.
        creation = getattr(u, "cache_creation", None)
        usage = NormalisedUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=(
                getattr(u, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=(
                getattr(u, "cache_creation_input_tokens", 0) or 0),
            cache_creation_5m_input_tokens=(
                getattr(creation, "ephemeral_5m_input_tokens", 0) or 0),
            cache_creation_1h_input_tokens=(
                getattr(creation, "ephemeral_1h_input_tokens", 0) or 0),
        )
        return NormalisedResponse(
            content=list(getattr(response, "content", None) or []),
            usage=usage,
            resolved_model=getattr(response, "model", None),
            stop_reason=getattr(response, "stop_reason", None),
            provider=self.provider,
            base_url=self.base_url,
            raw_request=wire,
            # The canonical request IS this wire, so the same dict rides under
            # both names and an audit path reads `wire_request` whoever served
            # the call.
            raw_response=response_to_dict(response),
            wire_request=wire,
            decoding_params=decoding,
        )


def _translate_anthropic_error(exc):
    """Map an anthropic SDK exception to a normalised ProviderError.

    Imported lazily AND guarded, exactly as `_translate_openai_error` is. This
    package supports being installed without its provider SDKs, with the caller
    injecting a client it built itself; on that shape `import anthropic` raises,
    and an unguarded import here would replace every provider failure with a
    ModuleNotFoundError — which `except ProviderError` does not catch and
    `create_message_with_retry` does not retry. With the SDK unavailable there
    is nothing to classify against, so the original exception is returned
    unchanged and reaches the caller intact.

    RETRYABLE (`ProviderRateLimitError` / `ProviderRetryableError`): rate
    limits, 5xx responses including Anthropic's 529 overloaded, and
    `APIConnectionError`. A connection that never established cannot have been
    served, so retrying it cannot be charged twice.

    NOT RETRYABLE, DELIBERATELY: `APITimeoutError`. A timeout is the one
    transient failure where the request may already have been served and billed
    while the response was lost, so an automatic retry can pay for the same
    answer several times over and record one — a run whose recorded cost is
    quietly below what was charged, which is the failure this layer exists to
    prevent. It becomes a plain `ProviderError` and stops the call loudly; a
    caller that knows its calls are cheap can catch that and retry itself. Note
    the ORDER below is load-bearing rather than incidental: `APITimeoutError`
    subclasses `APIConnectionError` in this SDK, so it is tested first, and
    swapping the two clauses would silently make every timeout retryable.

    Unknown exceptions are returned unchanged so genuinely unexpected errors
    still surface with their original type and traceback.
    """
    try:
        import anthropic
    except ImportError:
        return exc

    if isinstance(exc, anthropic.RateLimitError):
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        code = getattr(exc, "status_code", 0) or 0
        if 500 <= code < 600:
            return ProviderRetryableError(str(exc), status_code=code)
        return ProviderError(str(exc))
    # Timeout before connection: the narrower class first, or the retryable
    # connection clause below would swallow it. See the docstring.
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderError(str(exc))
    if isinstance(exc, anthropic.APIConnectionError):
        # No status to record: the request never reached one.
        return ProviderRetryableError(str(exc))
    if isinstance(exc, anthropic.APIError):
        return ProviderError(str(exc))
    return exc


# ---------------------------------------------------------------------------
# OpenAI adapter (serves OpenAI direct and the OpenRouter gateway)
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Adapter for OpenAI-family endpoints, across two wire protocols.

    OpenAI's own models speak the Responses API (`instructions` + `input`
    items, `function` tools, `function_call` / `function_call_output` items,
    `input_image` blocks, `max_output_tokens`). Gateway-served GLM / Qwen use
    the OpenAI-compatible Chat Completions surface (OpenRouter, upstream of
    which those models' native Responses endpoint 404s), whose shape differs
    (`messages`, `tools[].function`, assistant `tool_calls`, `role:"tool"`
    results, `image_url` content parts, `max_tokens`). The per-model `wire_api`
    in the registry selects the protocol; either way the adapter translates the
    canonical Anthropic-shaped request to the wire format and the response back
    to Anthropic-shaped content blocks + normalised usage. When the model's
    entry carries a `Route` (gateway-served), the Chat Completions path also
    emits the OpenRouter provider object, captures the generation id / served
    upstream / reported cost, and enforces the pin (see `create_message`). The
    `openai` package is imported lazily, so a caller that uses only Anthropic
    models never needs it installed.
    """

    def __init__(self, client, *, provider="openai", base_url=None):
        self._client = client
        self.provider = provider
        self.base_url = base_url

    def create_message(self, *, model, system, messages, max_tokens,
                       tools=None, tool_choice=None, sampling=None,
                       thinking=None):
        # `thinking` is threaded to the resolver, which renders it for this
        # wire — a named effort as the single reasoning level, a disabled
        # mode as "none" — or refuses a shape the entry's declared surface
        # does not take. See `_openai_family_thinking_params`.
        info = model_info(model)
        canonical = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools is not None:
            canonical["tools"] = tools
        if tool_choice is not None:
            canonical["tool_choice"] = tool_choice

        if info.wire_api == WIRE_CHAT_COMPLETIONS:
            wire, decoding = _to_chat_completions_wire(
                model=model, system=system, messages=messages, tools=tools,
                tool_choice=tool_choice, max_tokens=max_tokens,
                sampling=sampling, thinking=thinking)
            canonical.update(_sent_sampling(decoding))
            route = info.route
            if route is not None:
                # Gateway-served: emit OpenRouter's provider routing object and
                # request the response-reported cost. `extra_body` merges these
                # at the top level of the request body, exactly where OpenRouter
                # reads them; recorded in `wire` for the audit trail. MERGED
                # rather than assigned, because the translation puts body-only
                # decoding params there too (`top_k`), and assigning would drop
                # a sampling control the caller asked for while leaving it in
                # the recorded identity.
                extra_body = wire.setdefault("extra_body", {})
                extra_body["provider"] = provider_object(route)
                extra_body["usage"] = {"include": True}
            try:
                response = self._client.chat.completions.create(**wire)
            except Exception as exc:
                raise _translate_openai_error(exc)
            # One converter for every path (`wire_log.response_to_dict`), so
            # raw_response is a plain dict on this adapter exactly as on the
            # Anthropic one, whatever object the client returned.
            raw = response_to_dict(response)
            # BEFORE ANYTHING READS THIS AS A RESPONSE: a gateway may report an
            # upstream failure in the body of a 200 rather than as an HTTP
            # status, and a body carrying `error` is not a response at all.
            # Classified here, a transient one reaches the retry ladder as the
            # rate limit or 5xx it is; left to the reader below, it becomes an
            # empty completion, and on a routed call the pin assertion speaks
            # first and refuses it — non-retryably — for an attribution an
            # error body was never going to carry. See `_translate_error_body`.
            failure = _translate_error_body(raw)
            if failure is not None:
                raise failure
            normalised = self._from_chat_wire(raw, canonical=canonical,
                                              wire=wire, decoding=decoding)
            if route is not None:
                self._attach_routing(normalised, raw, route)
            return normalised

        wire, decoding = _to_openai_wire(
            model=model, system=system, messages=messages, tools=tools,
            tool_choice=tool_choice, max_tokens=max_tokens,
            sampling=sampling, thinking=thinking)
        canonical.update(_sent_sampling(decoding))

        try:
            response = self._client.responses.create(**wire)
        except Exception as exc:
            raise _translate_openai_error(exc)

        raw = response_to_dict(response)
        # As on the Chat Completions path above, before anything reads this as
        # a response: a Responses object that failed says so in `status` and
        # `error`, neither of which the reader below refuses on, and an empty
        # answer under an ordinary stop reason is the quieter of the two ways
        # this can go wrong. See `_translate_responses_error`.
        failure = _translate_responses_error(raw)
        if failure is not None:
            raise failure
        return self._from_wire(raw, canonical=canonical, wire=wire,
                               decoding=decoding)

    def _from_chat_wire(self, raw, *, canonical, wire, decoding):
        """Translate a Chat Completions `.model_dump()` dict to a
        NormalisedResponse.

        `raw` is a plain dict so this is exercised offline against hand-written
        response fixtures, no SDK object required.
        """
        choices = raw.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        finish_reason = first.get("finish_reason")

        content = []
        # Text first, so a text-only response surfaces as content[0].text,
        # matching the Anthropic block ordering a caller reads against.
        for text in _chat_message_texts(message.get("content")):
            content.append(_TextBlock(text))
        has_tool_call = False
        for tc in message.get("tool_calls") or []:
            if (tc.get("type") or "function") != "function":
                continue
            has_tool_call = True
            content.append(_chat_tool_use_block(tc))

        u = raw.get("usage") or {}
        prompt_tokens = u.get("prompt_tokens", 0) or 0
        details = u.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        # Chat Completions reports total prompt tokens in `prompt_tokens`
        # (cached included); subtract the cached count so normalised
        # `input_tokens` is full-price only and pricing never double-counts.
        # A host may omit `prompt_tokens_details`, in which case cached is 0
        # (conservative full price).
        full_input = max(0, prompt_tokens - cached)
        usage = NormalisedUsage(
            input_tokens=full_input,
            output_tokens=u.get("completion_tokens", 0) or 0,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,
        )

        return NormalisedResponse(
            content=content,
            usage=usage,
            resolved_model=raw.get("model"),
            stop_reason=_chat_stop_reason(finish_reason, has_tool_call),
            provider=self.provider,
            base_url=self.base_url,
            raw_request=canonical,
            raw_response=raw,
            wire_request=wire,
            decoding_params=decoding,
        )

    def _attach_routing(self, normalised, raw, route):
        """Capture routing provenance and enforce the pin on a routed response.

        Reads OpenRouter's generation id (`id`), served-provider attribution
        (`provider`), and response-reported cost (`usage.cost`) off the raw
        completion, RUNS THE PIN ASSERTION (raises `ProviderRouteMismatch` if
        the served upstream is not the pinned one, or is absent) so nothing that
        did not go where it declared is returned, and threads each of the three
        fields onto the NormalisedResponse as it holds. `raw` is a plain dict,
        so this is exercised offline against hand-authored routed-response
        fixtures.

        EVERY REFUSAL HERE IS ABOUT A CALL THE GATEWAY ALREADY BILLED, so each
        one carries the response it refuses on the exception's `response`
        attribute rather than discarding it: the tokens were spent whatever this
        layer thinks of the receipt, and a consumer that cannot see them cannot
        ledger them. Each field is threaded on as soon as it holds, so the
        attached response carries the provenance established before the refusal
        — a missing cost still arrives with its generation id and served
        upstream, which is what a ledger entry for the spend needs.
        """
        served = raw.get("provider")
        # Enforce the pin before returning anything: a mismatch raises here and
        # the response is refused rather than handed back, but it rides on the
        # exception because it was billed.
        try:
            assert_served_upstream(route, served)
        except ProviderRouteMismatch as mismatch:
            mismatch.response = normalised
            raise
        normalised.served_provider = served
        usage = raw.get("usage") or {}
        generation_id = raw.get("id")
        # A routed response without its audit receipt (generation id) or the
        # cost the gateway charged for it is as unrecordable as one served by
        # the wrong upstream: the gateway's figure is the only record of what
        # this call cost, so a silent None would leave the call costless with no
        # receipt. Loud, like the pin.
        if generation_id is None:
            refusal = ProviderError(
                "routed response carried no generation id ('id'); the call "
                "cannot be recorded with an audit receipt, so it is refused.")
            refusal.response = normalised
            raise refusal
        normalised.generation_id = generation_id
        reported_cost = usage.get("cost")
        if reported_cost is None:
            refusal = ProviderError(
                "routed response carried no usage.cost (was usage:{include:"
                "true} honoured?); the gateway's own figure is the record of "
                "what this call was charged, so a costless response is refused "
                "rather than recorded at $0.")
            refusal.response = normalised
            raise refusal
        normalised.reported_cost = reported_cost

    def _from_wire(self, raw, *, canonical, wire, decoding):
        """Translate a Responses `.model_dump()` dict to a NormalisedResponse.

        `raw` is a plain dict so this is exercised offline against hand-written
        response fixtures, no SDK object required.
        """
        content = []
        has_tool_call = False
        has_refusal = False
        for item in raw.get("output") or []:
            itype = item.get("type")
            if itype == "function_call":
                has_tool_call = True
                content.append(_openai_tool_use_block(item))
            elif itype == "message":
                for part in item.get("content") or []:
                    ptype = part.get("type")
                    if ptype in _TEXT_PART_TYPES:
                        content.append(_TextBlock(part.get("text", "")))
                    elif ptype == "refusal":
                        # A refusal IS the model's answer to this call, so it is
                        # surfaced as a text block — the same shape a caller
                        # already walks — and the stop reason becomes "refusal"
                        # below. Dropping the part instead leaves an empty
                        # `.content` under a stop reason saying the turn simply
                        # ended, which reads downstream as "the model produced
                        # no answer" and points a reader at the prompt or at a
                        # decoding bug rather than at the refusal that actually
                        # happened. The Responses part carries its text under
                        # `refusal`; `text` is accepted as a fallback so a host
                        # that reuses the ordinary key is still read rather than
                        # silently emptied.
                        has_refusal = True
                        content.append(_TextBlock(
                            part.get("refusal") or part.get("text", "")))
            # reasoning and other item types carry no assistant-visible
            # content here and are skipped.

        u = raw.get("usage") or {}
        input_tokens = u.get("input_tokens", 0) or 0
        details = u.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        # OpenAI reports total prompt tokens in `input_tokens` (cached
        # included); subtract the cached count so the normalised `input_tokens`
        # is full-price only and pricing never double-counts. GLM may omit the
        # cached field, in which case cached is 0 (conservative full price).
        full_input = max(0, input_tokens - cached)
        usage = NormalisedUsage(
            input_tokens=full_input,
            output_tokens=u.get("output_tokens", 0) or 0,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,
        )

        return NormalisedResponse(
            content=content,
            usage=usage,
            resolved_model=raw.get("model"),
            stop_reason=_openai_stop_reason(raw, has_tool_call,
                                            has_refusal=has_refusal),
            provider=self.provider,
            base_url=self.base_url,
            raw_request=canonical,
            raw_response=raw,
            wire_request=wire,
            decoding_params=decoding,
        )


# The content-part types that carry assistant text on the OpenAI-family wires:
# `output_text` is the Responses spelling, `text` the Chat Completions one. Both
# are accepted on BOTH paths, and from one constant, so the two translators
# cannot drift apart on what counts as an answer. A host that replies in the
# other dialect's part shape is read rather than treated as having said nothing,
# which is the outcome that matters: the output tokens are billed either way,
# and a response silently emptied here is indistinguishable downstream from a
# model that declined to answer.
_TEXT_PART_TYPES = ("output_text", "text")


class _TextBlock:
    """An Anthropic-shaped text block synthesised from an OpenAI response."""

    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUseBlock:
    """An Anthropic-shaped tool_use block synthesised from an OpenAI
    function_call output item."""

    type = "tool_use"

    def __init__(self, id, name, input):
        self.id = id
        self.name = name
        self.input = input


def _openai_tool_use_block(item):
    import json
    args = item.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args else {}
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(args, dict):
        parsed = args
    else:
        parsed = {}
    return _ToolUseBlock(
        id=item.get("call_id") or item.get("id"),
        name=item.get("name"),
        input=parsed,
    )


def _openai_stop_reason(raw, has_tool_call, *, has_refusal=False):
    """Map a Responses completion to the canonical stop reason.

    An `incomplete` status is the output cap when `incomplete_details.reason`
    says so and a refusal otherwise. The statuses that carry no answer at all
    (`failed` and the rest) never arrive here: `_translate_responses_error`
    refuses them before the response is read, which is why this maps only the
    two a caller can be handed. A message carrying a `refusal` part is a
    refusal too, even on a `completed` status: the endpoint completed normally
    and the model declined, and recording that as `end_turn` would file a
    declined call as a finished one. Refusal outranks `tool_use` — a response
    that both declined and called a tool has still declined, and that is the
    fact a reader needs first.
    """
    status = raw.get("status")
    if status == "incomplete":
        reason = (raw.get("incomplete_details") or {}).get("reason")
        return "max_tokens" if reason == "max_output_tokens" else "refusal"
    if has_refusal:
        return "refusal"
    if has_tool_call:
        return "tool_use"
    return "end_turn"


def _chat_message_texts(content):
    """The assistant text carried by a Chat Completions `message.content`.

    The classic shape is a plain string. An OpenAI-compatible host may instead
    send a parts LIST — the shape the Responses API uses — and a reader that
    accepts only the string yields an empty response for a call whose output
    tokens were billed all the same, with nothing to distinguish it downstream
    from a model that said nothing. Both shapes are read, and the list is walked
    for `_TEXT_PART_TYPES` parts exactly as `_from_wire` walks a Responses
    message, so the two wires agree on what counts as assistant text.

    Returns a list because a parts list may carry several text parts, and the
    caller wraps each in its own block, preserving the order the host sent.
    Empty strings are skipped, matching the string path: a block carrying no
    text is not an answer, and emitting one would make a genuinely empty
    response look answered.
    """
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []
    texts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in _TEXT_PART_TYPES:
            text = part.get("text", "")
            if text:
                texts.append(text)
    return texts


def _chat_tool_use_block(tc):
    """Anthropic-shaped tool_use block from a Chat Completions tool_call.

    The tool_call carries its arguments as a JSON string under
    `function.arguments`; parse it to the dict the canonical `tool_use` block
    carries (an empty or malformed string yields `{}`, so a truncated call
    surfaces as an empty input rather than crashing the caller's loop)."""
    import json
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args else {}
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(args, dict):
        parsed = args
    else:
        parsed = {}
    return _ToolUseBlock(id=tc.get("id"), name=fn.get("name"), input=parsed)


def _chat_stop_reason(finish_reason, has_tool_call):
    """Map a Chat Completions `finish_reason` to the canonical stop reason.

    `length` is the output-cap truncation; `content_filter` is the host's safety
    filter stopping the response, which is a refusal; `tool_calls` (or any
    assistant message that actually carried a tool call) maps to tool_use;
    `stop` and anything else map to end_turn.

    `content_filter` maps to `"refusal"` — the value the Responses path already
    produces for a blocked call, so one canonical vocabulary covers both wires.
    Falling through to `end_turn` would record a filtered call as a completed
    turn, and a caller reading an empty `.content` under `end_turn` concludes
    the model produced no answer and goes looking at the prompt, when the filter
    is what stopped it. Checked ahead of the tool-call clause: a filtered
    response has been filtered whatever else it contains.
    """
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter":
        return "refusal"
    if finish_reason == "tool_calls" or has_tool_call:
        return "tool_use"
    return "end_turn"


def _to_chat_completions_wire(*, model, system, messages, tools, tool_choice,
                              max_tokens, sampling=None, thinking=None):
    """Translate a canonical Anthropic-shaped request to Chat Completions wire
    kwargs (the gateway-served GLM / Qwen path).

    Returns `(wire, decoding)` where `decoding` is the subset of decoding
    params actually sent (from `resolved_decoding_params`, which a caller
    records for provenance and folds into call identity). Its keys
    (`max_tokens`, `reasoning_effort`, the sampling controls when accepted) are
    Chat Completions wire keys, so they merge straight into the wire request —
    all but `top_k`, which goes under `extra_body` (see below).
    Pure: no network, no SDK, so it is unit-tested directly. The output-token
    cap rides under `max_tokens` (the classic Chat Completions key), which the
    OpenRouter-compatible surface documents, so the newer OpenAI-only
    `max_completion_tokens` is deliberately not used here. The caller adds the
    OpenRouter provider object under `extra_body` when the model is routed,
    merging with whatever is already there.
    `thinking` is threaded to the resolver only so a spec aimed at a
    non-Anthropic model is refused there rather than silently dropped here.
    """
    decoding = resolved_decoding_params(
        model, sampling=sampling, max_tokens=max_tokens,
        thinking=thinking)
    wire = {
        "model": model,
        "messages": _messages_to_chat(system, messages),
        **decoding,
    }
    # `top_k` is not a Chat Completions PARAMETER: the openai SDK's
    # `chat.completions.create` names every parameter it takes and defines no
    # `**kwargs`, so a top-level `top_k=` is a TypeError raised in the SDK
    # before the request goes anywhere. It IS a body field the OpenRouter
    # surface reads, and `extra_body` is the SDK's channel for exactly that —
    # the one the routed provider object and usage flag already ride. So the
    # value is still sent, and still identity-bearing: `decoding` keeps it
    # unchanged, so `decoding_params`, the canonical request and the
    # fingerprint all record it as the sampling control it is. Only its
    # position in the request moves.
    if "top_k" in wire:
        wire.setdefault("extra_body", {})["top_k"] = wire.pop("top_k")
    if tools:
        wire["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]
    if tool_choice is not None:
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "auto":
            wire["tool_choice"] = "auto"
        else:
            wire["tool_choice"] = tool_choice

    return wire, decoding


def _messages_to_chat(system, messages):
    """Translate the canonical system + Anthropic messages to a Chat
    Completions `messages` list.

    The system text becomes the first `system` message. Each canonical message
    becomes one or more chat messages: an assistant turn's text and tool_use
    blocks collapse into a single assistant message (`content` + `tool_calls`),
    and each tool_result becomes its own `role:"tool"` message keyed by
    `tool_call_id`. User text and image blocks become the content-parts array
    (`text` / `image_url` data URIs) of one user message.
    """
    out = []
    instructions = _system_to_instructions(system)
    if instructions:
        out.append({"role": "system", "content": instructions})
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        out.extend(_blocks_to_chat_messages(role, content or []))
    return out


def _blocks_to_chat_messages(role, blocks):
    import json

    if role == "assistant":
        # Text + tool_use blocks collapse into one assistant message.
        text_parts = []
        tool_calls = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}),
                                                ensure_ascii=False),
                    },
                })
        text = "\n".join(t for t in text_parts if t)
        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    # User (or any non-assistant) role: text/image parts buffer into one
    # message; each tool_result flushes the buffer and emits its own tool
    # message so ordering is preserved.
    messages = []
    parts = []

    def flush():
        if parts:
            messages.append({"role": role, "content": list(parts)})
            parts.clear()

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source") or {}
            media = src.get("media_type", "image/png")
            data = src.get("data", "")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{data}"},
            })
        elif btype == "tool_result":
            flush()
            payload = block.get("content")
            output = (payload if isinstance(payload, str)
                      else json.dumps(payload, ensure_ascii=False))
            messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id"),
                "content": output,
            })
    flush()
    return messages


def _to_openai_wire(*, model, system, messages, tools, tool_choice,
                    max_tokens, sampling=None, thinking=None):
    """Translate a canonical Anthropic-shaped request to Responses wire kwargs.

    Returns `(wire, decoding)` where `decoding` is the subset of decoding
    params actually sent (from `resolved_decoding_params`, which a caller
    records for provenance and folds into call identity). Its keys
    (`max_output_tokens`, `reasoning`, the sampling controls when accepted) are already
    Responses wire keys, so they merge straight into the wire request. Pure: no
    network, no SDK, so it is unit-tested directly. `thinking` is threaded to
    the resolver only so a spec aimed at a non-Anthropic model is refused there
    rather than silently dropped here.
    """
    decoding = resolved_decoding_params(
        model, sampling=sampling, max_tokens=max_tokens,
        thinking=thinking)
    wire = {
        "model": model,
        "input": _messages_to_input(messages),
        **decoding,
    }
    instructions = _system_to_instructions(system)
    if instructions:
        wire["instructions"] = instructions
    if tools:
        wire["tools"] = [
            {
                "type": "function",
                "name": t.get("name"),
                "description": t.get("description", ""),
                # Non-strict: strict mode demands that every nested property be
                # marked required, which a caller's schema is under no
                # obligation to do, and enabling it would turn a perfectly valid
                # schema into a failed call. The schema body is otherwise passed
                # through unchanged on every wire, so the same tool definition
                # goes out whoever serves the call.
                "strict": False,
                "parameters": t.get("input_schema", {}),
            }
            for t in tools
        ]
    if tool_choice is not None:
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "auto":
            wire["tool_choice"] = "auto"
        else:
            wire["tool_choice"] = tool_choice

    return wire, decoding


def _system_to_instructions(system):
    """Concatenate the canonical system text blocks into a single
    Responses `instructions` string. cache_control markers are dropped
    (OpenAI/GLM caching is automatic)."""
    if isinstance(system, str):
        return system
    parts = []
    for block in system or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n\n".join(parts)


def _messages_to_input(messages):
    """Translate canonical Anthropic messages to a Responses `input` list.

    Text and image blocks become a message item with `input_text` /
    `input_image` (or `output_text` for assistant text). A tool_use block
    becomes a top-level `function_call` item and a tool_result block a
    top-level `function_call_output` item, both correlated by call id, exactly
    as the Responses API expects (they are not nested inside a message).
    """
    items = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            text_type = "output_text" if role == "assistant" else "input_text"
            items.append({
                "role": role,
                "content": [{"type": text_type, "text": content}],
            })
            continue
        items.extend(_blocks_to_input_items(role, content or []))
    return items


def _blocks_to_input_items(role, blocks):
    import json

    items = []
    buffer = []
    text_type = "output_text" if role == "assistant" else "input_text"

    def flush():
        if buffer:
            items.append({"role": role, "content": list(buffer)})
            buffer.clear()

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            buffer.append({"type": text_type, "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source") or {}
            media = src.get("media_type", "image/png")
            data = src.get("data", "")
            buffer.append({
                "type": "input_image",
                "image_url": f"data:{media};base64,{data}",
            })
        elif btype == "tool_use":
            flush()
            items.append({
                "type": "function_call",
                "call_id": block.get("id"),
                "name": block.get("name"),
                "arguments": json.dumps(block.get("input", {}),
                                        ensure_ascii=False),
            })
        elif btype == "tool_result":
            flush()
            payload = block.get("content")
            output = (payload if isinstance(payload, str)
                      else json.dumps(payload, ensure_ascii=False))
            items.append({
                "type": "function_call_output",
                "call_id": block.get("tool_use_id"),
                "output": output,
            })
    flush()
    return items


# HTTP 429 means two unrelated things on the OpenAI wire and only one of them
# is worth waiting for. Throttling clears on its own, which is what the ladder
# is for. A spent account does not: no rung brings the balance back, and every
# rung climbed is time an operator spends not being told. The status cannot
# separate them — the evidence is `type` / `code` in the body, and these are
# the values that mean the account, not the rate.
_OPENAI_SPENT_ACCOUNT_CODES = frozenset({
    # The `type` OpenAI sends with both codes below.
    "insufficient_quota",
    # Recorded 2026-08-18 on a prepaid balance at zero: four calls walked the
    # full (2, 5, 11, 23) ladder and exhausted it, ~41 seconds each, to arrive
    # at what the first response already said.
    "credit_balance_exhausted",
    # Documented sibling, not observed here: a configured spend cap rather than
    # an empty balance. Same shape of fact — a limit a wait does not move.
    "billing_hard_limit_reached",
})


def _openai_error_codes(exc):
    """Every `type` / `code` string an openai exception carries.

    The SDK's concrete client unwraps the documented `{"error": {...}}`
    envelope before constructing the exception, so `exc.code` and `exc.type`
    normally hold what is wanted. The raw body is read too, in BOTH the
    unwrapped and the enveloped shape, because this package supports a caller
    injecting a client it built itself — including one whose error
    construction is the SDK base class's, which does not unwrap. Falling back
    to a retry because the body was nested one layer further down is the
    failure worth spending four lines to avoid.
    """
    found = {getattr(exc, "code", None), getattr(exc, "type", None)}
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        for layer in (body, body.get("error")):
            if isinstance(layer, dict):
                found |= {layer.get("code"), layer.get("type")}
    return {value for value in found if isinstance(value, str)}


def _translate_openai_error(exc):
    """Map an openai SDK exception to a normalised ProviderError.

    Imported lazily and guarded: with the openai package unavailable there is
    nothing to classify against, so the original exception is returned unchanged
    and reaches the caller intact. `_translate_anthropic_error` mirrors this;
    the two classify the same failure into the same normalised class, which is
    what lets a caller's retry loop stay provider-independent.

    RETRYABLE (`ProviderRateLimitError` / `ProviderRetryableError`): rate
    limits, 5xx responses, and `APIConnectionError`. A connection that never
    established cannot have been served, so retrying it cannot be charged twice.

    NOT RETRYABLE, DELIBERATELY: a 429 whose body says the ACCOUNT is spent
    rather than the rate exceeded. OpenAI overloads that status for two
    unrelated conditions, and classifying on the status alone cannot tell them
    apart: throttling is the most retryable thing there is, an exhausted credit
    balance the least. Waiting out the ladder cannot clear a balance, so all it
    buys is ~41 seconds per call of an operator not being told what the first
    response already said — and on a long batch every subsequent call pays it
    again, turning a fast failure into a slow one. The narrow set of codes that
    mean the account is `_OPENAI_SPENT_ACCOUNT_CODES`; everything else about a
    429 stays retryable, because retrying a rate limit is the right default and
    this exemption is not an invitation to widen it. Only this translator
    carries the clause, because this is the wire the overload was observed on;
    a spent account arriving as a 4xx other than 429 is already non-retryable
    on both.

    NOT RETRYABLE, DELIBERATELY: `APITimeoutError`. A timeout is the one
    transient failure where the request may already have been served and billed
    while the response was lost, so an automatic retry can pay for the same
    answer several times over and record one — a run whose recorded cost is
    quietly below what was charged. It becomes a plain `ProviderError` and stops
    the call loudly; a caller that knows its calls are cheap can catch that and
    retry itself. The ORDER below is load-bearing: `APITimeoutError` subclasses
    `APIConnectionError` in this SDK too, so it is tested first, and swapping
    the two clauses would silently make every timeout retryable.
    """
    try:
        import openai
    except ImportError:
        return exc

    if isinstance(exc, openai.RateLimitError):
        # A 429 the ladder cannot clear. See the docstring and
        # `_OPENAI_SPENT_ACCOUNT_CODES`.
        if _openai_error_codes(exc) & _OPENAI_SPENT_ACCOUNT_CODES:
            return ProviderError(str(exc))
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, openai.APIStatusError):
        code = getattr(exc, "status_code", 0) or 0
        if 500 <= code < 600:
            return ProviderRetryableError(str(exc), status_code=code)
        return ProviderError(str(exc))
    # Timeout before connection: the narrower class first, or the retryable
    # connection clause below would swallow it. See the docstring.
    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(str(exc))
    if isinstance(exc, openai.APIConnectionError):
        # No status to record: the request never reached one.
        return ProviderRetryableError(str(exc))
    if isinstance(exc, openai.APIError):
        return ProviderError(str(exc))
    return exc


def _translate_error_body(raw):
    """Map an OpenAI-compatible error BODY to a normalised ProviderError, or
    return None when the body is a response.

    A gateway may report an upstream failure as HTTP 200 WITH `error` IN THE
    BODY rather than as an HTTP status. OpenRouter does this for origin
    timeouts and shed load, sending `{"error": {"code": 504, "message": "error
    code: 524"}}` with every other top-level field null. Nothing raised in the
    SDK, so `_translate_openai_error` never sees it, and `_from_chat_wire`
    reads `choices` / `usage` / `finish_reason` and never `error` — so the body
    becomes a well-formed response with empty content and zero tokens. What
    happens next is worse than the emptiness: on a routed call the pin
    assertion is the first thing to read that body, finds no attribution
    because the body carries nothing at all, and refuses it as
    `ProviderRouteMismatch` — NOT retryable, by design and rightly, since a
    genuine pin violation is a correctness failure. A transient blip the
    backoff ladder would have survived is then reported to the caller as a
    broken pin and its answer is lost for good. On a non-routed call
    `_attach_routing` does not run at all and the empty completion is simply
    returned as a success.

    A status in the body is classified EXACTLY as `_translate_openai_error`
    classifies the same status arriving as an HTTP response — 429 a rate limit,
    5xx retryable, anything else a plain ProviderError — because where the
    gateway chose to put the status says nothing about whether the failure is
    transient. That includes the 504/524 pair: the deliberate non-retry of
    `APITimeoutError` up there is about a response LOST IN TRANSIT, which may
    have been served and billed while the caller was not looking, and an error
    body is the opposite case — it IS the gateway's answer, arriving with no
    generation id and no usage, so nothing was served that a retry could pay
    for twice. A `code` that is not an HTTP status (an OpenAI-style string such
    as "insufficient_quota") is not guessed at: it stops the call loudly with
    the gateway's own message.

    Raised INSTEAD of a response, so the exception carries no `response`: an
    error body is a call that "reached the provider and got an error back",
    which `ProviderError` defines as having no billed material to carry, and
    the null `usage` in it says the same thing.

    `raw` is a plain dict, so this is exercised offline against hand-authored
    error bodies, no SDK object required.
    """
    error = raw.get("error") if isinstance(raw, dict) else None
    if not error:
        return None
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        # `code` is where OpenRouter puts the HTTP status; `status` is accepted
        # too, for a host that names the same thing the other way. Tested for
        # None rather than for presence, since a host that carries both sends
        # the one it does not use as null rather than omitting it.
        code = error.get("code")
        if code is None:
            code = error.get("status")
    else:
        # A host that sends a bare string where the documented shape is an
        # object. There is no status to classify, so it stops the call loudly.
        message, code = str(error).strip(), None
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = None

    # The gateway's own words, quoted: it bounds text this layer did not write
    # (and swallows the trailing newline OpenRouter puts on it). A body whose
    # message is empty falls back to the whole `error` object, which is then the
    # most a reader can be given.
    detail = ("provider reported a failure in the response body rather than as "
              f"an HTTP status (code {code!r}): {message or error!r}")
    if status == 429:
        return ProviderRateLimitError(detail)
    if status is not None and 500 <= status < 600:
        return ProviderRetryableError(detail, status_code=status)
    return ProviderError(detail)


# The two Responses statuses `_from_wire` can read as an answer: `completed`,
# and `incomplete` (the output cap or a filter, which `_openai_stop_reason`
# renders). The rest of the documented vocabulary — `failed`, `cancelled`,
# `queued`, `in_progress` — describes a call that produced no answer to read.
_RESPONSES_READABLE_STATUSES = frozenset({"completed", "incomplete"})

# The Responses error codes worth another attempt, out of a vocabulary that is
# otherwise about the request itself (`invalid_prompt`, the `invalid_image` /
# `image_too_large` family) and would fail identically four more times.
_RESPONSES_RATE_LIMIT_CODE = "rate_limit_exceeded"
_RESPONSES_RETRYABLE_CODE = "server_error"


def _translate_responses_error(raw):
    """Map a Responses object that did not complete to a normalised
    ProviderError, or return None when the response is a response.

    THE RESPONSES SIBLING OF `_translate_error_body`, and the same failure in a
    different vocabulary: the HTTP call succeeded and the failure is a field in
    the body. A Responses object carries `status` — `completed`, `failed`,
    `cancelled`, `queued`, `in_progress`, `incomplete` — and an `error` object
    when it failed. `_from_wire` reads `output`, `usage`, `model`, `status` and
    `incomplete_details`, and NEVER `error`; `_openai_stop_reason` renders
    `incomplete` and treats every other status as an ordinary end of turn. So a
    `status: "failed"` response normalises to empty content under `end_turn`,
    which reads downstream as a model that finished and said nothing.

    THAT IS THE QUIETEST FAILURE IN THIS FILE — quieter than the gateway error
    body, which at least raises something. An empty answer under a stop reason
    saying the turn ended normally is indistinguishable from a model that
    declined to elaborate, so a caller that treats "nothing further" as a valid
    outcome records the call as having happened. Nothing is left for an audit
    log to catch it by.

    The codes are a documented string enum, not HTTP statuses, so they are
    mapped by name rather than by the arithmetic `_translate_error_body` uses:
    `rate_limit_exceeded` is a rate limit and `server_error` is retryable
    (with no `status_code`, because the failing call never reached one — the
    HTTP response was a 200 carrying this object). Every other documented code
    is about the request as sent — an invalid prompt, an image too large or in
    the wrong format — and would fail identically on every retry, so it stops
    the call loudly instead. A status that is not readable but carries no error
    object (a `cancelled` or still-`queued` response) is refused on the status
    alone: there is no answer in it either.

    An ABSENT status is left alone rather than refused. `_openai_stop_reason`
    already treats it as an ordinary turn, translation fixtures omit it, and
    this is not the place to start requiring a field that was never required.

    Raised INSTEAD of a response, so no billed material rides on it — as with
    the error body, and for the same reason: a response that failed was not
    served, and its `usage` says so.

    `raw` is a plain dict, so this is exercised offline against hand-authored
    responses, no SDK object required.
    """
    if not isinstance(raw, dict):
        return None
    error = raw.get("error")
    status = raw.get("status")
    unreadable = (status is not None
                  and status not in _RESPONSES_READABLE_STATUSES)
    # An `error` beside a READABLE status is decisive only when there is no
    # answer next to it. This is where the two translators part company, and
    # the wire is the reason: `error` is a documented field of every Responses
    # object, null on success, so its presence is not by itself proof that the
    # body is not a response — where on Chat Completions `error` is no part of
    # the schema at all, and a body carrying one is not a completion. A host
    # that leaves a stale error on a response that completed WITH output is
    # therefore read as having answered, because discarding a billed answer
    # over a contradictory field is the one outcome worth avoiding more than a
    # loud refusal. Nothing in the failure this guards against is lost to it: a
    # `failed` object carries no output and no readable status either, so the
    # first test takes it whatever the second would have said.
    if not unreadable and (not error or raw.get("output")):
        return None
    if not error:
        return ProviderError(
            f"the Responses call carries status {status!r} and no answer to "
            "read; it is refused rather than returned as an empty completion.")

    if isinstance(error, dict):
        code = error.get("code")
        message = str(error.get("message") or "").strip()
    else:
        # A host that sends a bare string where the documented shape is an
        # object: no code to classify, so it stops the call loudly.
        code, message = None, str(error).strip()
    # The provider's own words, quoted — see `_translate_error_body`.
    detail = (f"the Responses call did not complete (status {status!r}, error "
              f"code {code!r}): {message or error!r}")
    if code == _RESPONSES_RATE_LIMIT_CODE:
        return ProviderRateLimitError(detail)
    if code == _RESPONSES_RETRYABLE_CODE:
        return ProviderRetryableError(detail)
    return ProviderError(detail)


# ---------------------------------------------------------------------------
# Forced-named-tool normalisation
# ---------------------------------------------------------------------------

def tool_choice_named(model_id, tool_name):
    """The tool_choice value that forces `tool_name`, in the model's wire shape.

    The adapters normalise only tool_choice `{"type": "auto"}` to each
    provider's "auto" vocabulary; a forced-specific-tool tool_choice is passed
    through unchanged (see `_to_openai_wire` / `_to_chat_completions_wire` and
    `AnthropicAdapter`). So forcing one named tool means shaping the value per
    wire protocol here: Anthropic `tool_use`, the OpenAI Responses `function`,
    and the Chat Completions nested `function`. All three shapes are
    live-validated against their endpoints by `direktoro.cli`'s smoke run.

    A model whose registry entry states `forced_tool_choice=False` (each such
    entry records the probe that established it) cannot be forced at all, so
    this returns the wire's "auto" form (`{"type": "auto"}`, which every
    adapter normalises to its provider's auto vocabulary) instead of a forced
    value. A
    call site that does `tool_choice_named(model, name)` and passes the result
    to `adapter.create_message` therefore degrades to auto with no code change
    of its own; it consults `supports_forced_tool_choice` separately to arm the
    bounded retry a non-forcing model needs, because an auto call MAY return no
    tool call at all.

    Even for a forcing model a forced tool choice is a strong request, not a
    guarantee: a live call has been observed returning no tool call under a
    forced choice, which is the same tool-free response that retry handles.
    """
    info = model_info(model_id)
    if not info.forced_tool_choice:
        # Non-forcing endpoint: emit the canonical auto value; the adapters map
        # `{"type": "auto"}` to each wire's native auto token (Anthropic passes
        # it through; the OpenAI Responses / Chat Completions paths fold it to
        # the "auto" string). One return covers every wire.
        return {"type": "auto"}
    if info.provider == PROVIDER_ANTHROPIC:
        return {"type": "tool", "name": tool_name}
    if info.wire_api == WIRE_RESPONSES:
        return {"type": "function", "name": tool_name}
    return {"type": "function", "function": {"name": tool_name}}


def extract_tool_call(response, tool_name):
    """Return `(tool_input, error)` for the single `tool_name` call in `response`.

    The tool input is the `.input` of the one `tool_use` content block named
    `tool_name`. Returns `(None, message)` when the model called a different
    tool, no tool at all, or `tool_name` more than once: a caller forcing a
    single named tool wants the double call caught, not silently reduced to the
    first.

    This reads the call off the response and nothing more. Whether the input is
    VALID — required fields, value ranges, cross-field rules — is the caller's
    question, because only the caller knows what its tool schema means.
    """
    matches = []
    tool_names = []
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        name = getattr(block, "name", None)
        tool_names.append(name)
        if name == tool_name:
            matches.append(getattr(block, "input", None))
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, (f"model called {tool_name!r} {len(matches)} times; "
                      f"exactly one call was expected")
    if tool_names:
        return None, f"model called {tool_names} but not {tool_name!r}"
    return None, (
        f"model returned no tool call (stop_reason="
        f"{getattr(response, 'stop_reason', None)!r})")


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

class MissingAPIKey(RuntimeError):
    """A model's provider API-key environment variable is unset.

    Raised by `build_adapter` before any network call, so a run that cannot
    authenticate fails at startup rather than part-way through, after it has
    already spent on the work it did manage to do.
    """


def build_adapter(model_id, *, env=None, client=None, max_retries=None):
    """Construct the provider adapter for `model_id`.

    The model id determines the provider, base URL, and API-key env var, so
    routing needs no extra config: an Anthropic id builds an `AnthropicAdapter`;
    an OpenAI id, or a gateway-routed id (OpenRouter-served GLM / Qwen, whose
    entry carries a `Route`), builds an `OpenAIAdapter` keyed by its own base
    URL. A routed adapter emits the Route's provider object and enforces the pin
    per response (see `OpenAIAdapter.create_message`); the id's registry entry,
    not this function, decides that.

    Two construction seams, so a caller never has to build `AnthropicAdapter` /
    `OpenAIAdapter` by hand:

      - `client`: an already-built provider SDK client to wrap. When given, key
        resolution is skipped entirely (no env read, no `MissingAPIKey`, no SDK
        import) and the client is wrapped in the adapter its id selects — this
        is how a parallel fan-out shares ONE client across many calls, and how
        tests inject a stub client with no key and no SDK installed. The client
        must speak the wire the id's provider expects (an `anthropic.Anthropic`
        for an Anthropic id, an `openai.OpenAI` for an OpenAI-family / routed
        id); this function does not check that.
      - `max_retries`: forwarded to the SDK client constructor when this
        function builds the client itself (the no-`client` path). A client built
        here is built with `max_retries=0` UNLESS the caller names a value, so
        retry policy is owned above, not doubled — the same rule
        `direktoro.batch.build_batch_client` states for the client it builds.
        The SDK default is 2, and both SDKs' own retry predicate retries a
        request TIMEOUT: that is the one failure the error translators here
        deliberately refuse to retry, because a timed-out request may already
        have been served and billed while its response was lost, so the SDK
        default silently pays for one answer up to three times and returns one.
        Naming a value hands retry back to the SDK on purpose; an injected
        `client` is untouched, and passing `max_retries` with one is a loud
        `ValueError` because the value could reach no constructor.

    The `anthropic` and `openai` SDKs are imported lazily and ONLY on the
    build-the-client path, so importing this module needs neither, a run that
    uses only Anthropic models never needs `openai` installed, and an injected
    client needs no SDK importable at all. Raises `MissingAPIKey` when no client
    is injected and the key env var is unset. `env` overrides the environment
    (the process environment by default), for tests.
    """
    info = model_info(model_id)
    if client is not None:
        if max_retries is not None:
            raise ValueError(
                "build_adapter: max_retries cannot be combined with an injected "
                "client; the client is already built, so the value would reach "
                "no constructor. Set max_retries on the client you build, or "
                "drop the client so build_adapter builds one.")
        if info.provider == PROVIDER_ANTHROPIC:
            return AnthropicAdapter(client, base_url=info.base_url)
        return OpenAIAdapter(
            client, provider=info.provider, base_url=info.base_url)

    environ = os.environ if env is None else env
    api_key = environ.get(info.api_key_env, "")
    if not api_key:
        raise MissingAPIKey(
            f"environment variable {info.api_key_env} is not set; it is needed "
            f"for model {model_id!r}.")
    # A client built here disables the SDK's own retry unless the caller names a
    # value: this layer's retry policy is `create_message_with_retry` and the
    # error translation under it, and an SDK retrying beneath that doubles the
    # loop and hides the attempts from the caller's audit log entirely.
    retry_kwargs = {"max_retries": 0 if max_retries is None else max_retries}
    if info.provider == PROVIDER_ANTHROPIC:
        import anthropic
        built = anthropic.Anthropic(api_key=api_key, **retry_kwargs)
        return AnthropicAdapter(built, base_url=info.base_url)
    import openai
    built = openai.OpenAI(
        api_key=api_key, base_url=info.base_url, **retry_kwargs)
    return OpenAIAdapter(built, provider=info.provider, base_url=info.base_url)
