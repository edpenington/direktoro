"""Model registry: the single source of truth for every model this package can
reach, across providers.

Each model id maps to a `Model` record carrying its provider, API base URL,
API-key environment variable, capability flags, and a `quirks` dict for
provider-specific decoding rules (today, the OpenAI reasoning-effort level).
Which sampling controls an endpoint refuses is its own declaration,
`Model.rejects_sampling`.

What this registry knows is how to REACH a model and what that model can do.
Unit prices live in `direktoro.prices`, a dated table recording each vendor's
published rate with the page it was read from and the date it was read. Cost
arithmetic is `direktoro.cost.cost_from_rates`, over rates the caller supplies
— a table entry, or its own — and a gateway-reported charge is read off the
response as the fact it is (`NormalisedResponse.reported_cost`).

Model identity is pinned through the id itself, but "pinned" means different
things per provider, and the difference decides what is citation-grade.

Anthropic ids are pinned snapshots in BOTH of their forms. Pre-4.6 models carry
a dated id (`claude-opus-4-5-20251101`, `claude-haiku-4-5-20251001`); from the
4.6 generation on the ids are dateless (`claude-opus-5`, `claude-opus-4-8`,
`claude-sonnet-4-6`) and the dateless id IS the snapshot, not a pointer onto
one. Anthropic states it does not update the weights or configuration of an
existing model id — an updated model ships under a NEW id. A dateless Anthropic
id is therefore citation-grade exactly as written, with no dated form to wait
for and nothing to repoint to. (The convenience aliases Anthropic accepts for
PRE-4.6 models — bare `claude-haiku-4-5` resolving to the newest dated snapshot
— ARE repointable pointers. That is why this registry keys those models by
their dated id and leaves the alias unregistered.)

The residual identity risk for Anthropic is narrower than repointing: weights
are fixed per id, but the surrounding serving infrastructure (request router,
safety classifiers, sampling logic) can change, and Anthropic notes such updates
occasionally produce minor observable behaviour differences under an unchanged
id. Neither the id nor the response distinguishes that, so it is an accepted,
documented residual rather than something a fingerprint can capture.

The OpenRouter GLM / Qwen slugs are the genuinely rolling case: the slug is the
only id available, and the upstream serving it (or its quantization) can change
under that slug. For those, the provider-reported model string comes back on
every response (`NormalisedResponse.resolved_model`) for the caller to record
alongside the id it asked for, and a routed entry adds a second, external anchor
— the served-upstream attribution and generation id captured from the gateway
(see `direktoro.routing`).

This registry is what a caller validates a requested model against before it
spends anything, and what the provider adapters read to resolve
provider/base_url/key/route. An id with no entry here is unknown, and every
lookup rejects it loudly rather than guessing. Membership is not the whole
question, though: a RETIRED id is a member and still resolves, so a caller
accepting a new run asks `is_retired` as well (see `Model.retired`).

`thinking` records what thinking / reasoning-effort shapes a model's endpoint
actually accepts, so `direktoro.providers.resolved_decoding_params` can REFUSE
to emit a shape the model would reject with a 400 rather than discovering it on
a paid call (see `ThinkingSupport`).

HOW A FACT GETS INTO THIS TABLE
-------------------------------
Every capability flag and every `quirks` entry rests on one of exactly two kinds
of evidence, and which kind it is depends on which half of the table the entry
is in. The distinction matters to anyone citing this file, so it is stated
rather than blurred:

  - DIRECT entries (Anthropic, OpenAI) are read from the vendor's own PUBLISHED
    MODEL REFERENCE — the model and deprecation tables, the migration guide, the
    thinking documentation, the API reference — and the comment beside the
    value names the date that reference was read (the reads currently in the
    table are from 2026-07-31, 2026-08-01 and 2026-08-12). These are
    documentation facts. Nothing in this half of the table was established by
    calling an endpoint and watching it accept or reject a parameter.
  - ROUTED entries (OpenRouter) are read from LIVE ENDPOINT PROBES: GET
    /api/v1/models and /endpoints for the served upstream, quantization and
    supported_parameters, then a real plain / tool / vision call against the
    pinned endpoint. The comment beside the value names what was probed and
    when. These are the observations in this file; the direct half has none.
    One narrow exception: a routed entry's `sampling_bands` describe the
    GATEWAY's own documented request surface — the range OpenRouter itself
    accepts — because a continuous range is not a thing a probe can establish;
    the band's comment says so.

Either way the evidence travels with the value: the comment beside it is the
evidence, not decoration, so it moves with the value and is rewritten only when
the value is re-measured or the reference is re-read.

A value left at its FIELD DEFAULT is not a record of anything and should not be
read as one. `supports_images` is the case that matters: the five routed entries
set it explicitly because a probe actually sent an image, while all eleven
direct entries simply take the default True, which stands on the published model
reference like the rest of their row and on no probe at all.
`forced_tool_choice` closes that gap the other way: it has no working default
at all, so an unstated value is a construction error rather than a silent
claim, and every entry carries the flag with its basis beside it.

A family constant (`_NO_SAMPLING`, `_EFFORTS_4_7`, `_THINK_OPUS_4_7`, ...) is shared
between entries because the published reference states the fact per FAMILY, and
the constant's own comment carries that citation and its read date. Sharing one
is therefore not inference from a sibling: it is a single documented fact
recorded once instead of transcribed five times. What is forbidden is copying a
value across because the entry above it happens to set the same flag, with no
source of its own — an unsourced flag is indistinguishable from a sourced one
once it is in the table, which is what makes that expensive.

The practical consequence: a flag that differs from its neighbours is not an
inconsistency to be tidied away. Families are not uniform — two upstreams
serving one slug can disagree about whether a forced tool_choice routes, and a
provider can drop the sampling controls in a point release while everything else
about the model stays put. Changing such a flag back means re-probing the
endpoint or re-reading the reference and rewriting the evidence, never reasoning
from the shape of the table.

The standing limitation, stated once here rather than implied: this is
hand-verified and nothing re-checks it on a schedule. A model whose behaviour
changes without an announcement is wrong here until somebody looks. The dates
are what make that gap visible.

A dated provider fact — a published retirement date, a scheduled change — is
DATA. Record it, and let a test assert what is recorded. A test written to go
red when a date arrives converts an anticipated change into a build break on a
day nobody chose, and it fails whether or not anything is actually wrong.
Assert the fact; do not set an alarm.

DELETING AN ENTRY
-----------------
Delete an entry only if nothing has ever run on it. Otherwise set
`retired=True`. This table is also the provenance record for runs that already
happened: an id that served a run must keep resolving, or that run's stored
metadata stops being interpretable. `retired=True` keeps every lookup working
and is refused only where a NEW run is being accepted, so it costs nothing to
prefer.

"Where a new run is being accepted" is the caller's own boundary, not something
this module can enforce for it, so the flag is exported rather than merely
honoured internally: `is_retired(model_id)` is the predicate and
`known_models(include_retired=False)` is the startable list. `direktoro`'s own
CLI applies exactly those at config load. A library consumer that never asks
will find every accessor here resolving a retired id quite happily, which is the
design — provenance must not break — and is precisely why the question has to be
askable.
"""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

from direktoro.routing import GATEWAY_OPENROUTER, Route


# Environment variables that hold each provider's API key. Three keys cover
# every entry: Anthropic and OpenAI direct, everything else through OpenRouter.
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# Provider API base URLs. Anthropic uses the SDK default (None), so no base
# URL is pinned for it. OpenAI and OpenRouter are pinned explicitly. Everything
# non-direct rides OpenRouter's OpenAI-compatible Chat Completions surface, so
# the same OpenAI adapter serves it, keyed by the OpenRouter base URL and
# OPENROUTER_API_KEY, with a `route` pinning the upstream provider.
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Provider identifiers.
# Every entry carries one of these three; there is no fourth. A non-OpenAI host
# speaking the OpenAI wire format reaches this package only through the gateway,
# so it is PROVIDER_OPENROUTER — a separate "OpenAI-compatible, direct" provider
# class would be a public promise about something the table does not contain.
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_OPENROUTER = "openrouter"        # OpenAI-compat wire via the gateway

# Wire protocols an OpenAI-family model speaks. OpenAI's own models use the
# Responses API; gateway-served GLM / Qwen use the OpenAI-compatible Chat
# Completions surface (their native Responses endpoint 404s, so a single adapter
# must branch on this). Anthropic models never reach the OpenAI adapter, so
# their wire_api is unused.
WIRE_RESPONSES = "responses"
WIRE_CHAT_COMPLETIONS = "chat_completions"

# Reasoning-effort levels, ASCENDING — the vocabulary a caller picks from,
# validated per entry against `ThinkingSupport.efforts`. The Anthropic wire
# spells a level `output_config: {"effort": ...}`; the OpenAI families spell
# the same idea `reasoning_effort` (Chat Completions) / `reasoning:
# {"effort": ...}` (Responses). Order matters: it is how
# `ThinkingSupport.disabled_max_effort` is compared.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The thinking modes a caller can ask a model for (see providers.Thinking).
#   - adaptive: the model decides when and how much to think. The only on-mode
#     on the 4.6-generation-and-later Claude models.
#   - disabled: no thinking.
#   - budget:   a fixed thinking-token budget. Pre-4.6 models only — the 4.7+
#               families REJECT `budget_tokens` with a 400.
THINKING_ADAPTIVE = "adaptive"
THINKING_DISABLED = "disabled"
THINKING_BUDGET = "budget"
THINKING_MODES = (THINKING_ADAPTIVE, THINKING_DISABLED, THINKING_BUDGET)

# Thinking-block display settings (Anthropic `thinking.display`): whether a
# thinking block comes back carrying readable reasoning text or an empty string.
# Billing is identical under either setting — `display` changes what you SEE,
# not what you pay.
#
# The DEFAULT differs by generation and the FIELD does not, which is easy to
# conflate. Anthropic's thinking documentation, read
# 2026-08-01: "omitted" is the default on Fable 5, Mythos 5, Opus 5, Sonnet 5,
# Opus 4.8 and Opus 4.7; "summarized" is the default on Opus 4.6, Sonnet 4.6 and
# earlier models. But the field itself is accepted on both generations, and in
# BOTH on-modes — "`display` works in both modes: set it alongside
# `type: "adaptive"` or `type: "enabled"`" — the docs' own Sonnet 4.6 quickstart
# sets `display: "summarized"`. It is invalid with `{"type": "disabled"}` alone,
# where there is nothing to display. Which model accepts which setting is
# therefore DECLARED per entry on `ThinkingSupport.displays`, not inferred from
# the generation the default belongs to.
THINKING_DISPLAY_SUMMARIZED = "summarized"
THINKING_DISPLAY_OMITTED = "omitted"
THINKING_DISPLAYS = (THINKING_DISPLAY_SUMMARIZED, THINKING_DISPLAY_OMITTED)


@dataclass(frozen=True)
class ThinkingSupport:
    """What thinking / effort shapes ONE model's endpoint accepts.

    The registry is the only place that knows a model's family, so it is the
    only place that can tell a caller "that shape returns a 400" BEFORE the
    request is billed. `direktoro.providers.resolved_decoding_params` reads this
    record and refuses to emit a shape the model rejects, which is the whole
    point: a 400 on a paid call is the failure this record exists to prevent.

      - `modes`: which of `THINKING_MODES` the endpoint accepts as an explicit
        `thinking` parameter.
      - `efforts`: the accepted `output_config.effort` levels, a subset of
        `EFFORT_LEVELS`. Empty means the endpoint has no effort parameter at
        all (pre-4.6 models error on it).
      - `default_on`: True when OMITTING the `thinking` parameter still runs
        thinking. This is the trap the seam exists for — Claude Opus 5 and
        Sonnet 5 think by default, Opus 4.8 / 4.7 do not, and `max_tokens` caps
        thinking PLUS response text either way, so a cap sized on a
        non-thinking model buys less answer on a thinking one. Consumers read
        it via `thinking_support(model).default_on` to size caps honestly.
      - `default_effort`: the effort level in force when `output_config.effort`
        is omitted (Anthropic: "high"), or None when no single level
        corresponds to the omitted state — an endpoint with no effort
        parameter at all, or a gateway-served entry whose omitted-state
        behaviour is dynamic rather than a level (the routed records below).
        Required alongside `disabled_max_effort`, whose cap is evaluated at
        the omitted-state level for a request that disables thinking without
        naming an effort.
      - `disabled_max_effort`: the HIGHEST effort at which `{"type":
        "disabled"}` is accepted; None means no cap. Claude Opus 5 caps this at
        "high" — pairing disabled thinking with `xhigh` or `max` returns a 400,
        and the check is per REQUEST, so a later call that raises effort while
        thinking is still disabled is rejected even though earlier calls in the
        same conversation succeeded.
      - `displays`: the accepted `thinking.display` settings, a subset of
        `THINKING_DISPLAYS`. Empty means the endpoint takes no `display` field
        and asking for one is refused. Modelled per entry for the same reason
        as `modes` and `efforts`: `display` is generation-scoped in principle
        (it postdates the oldest ids this registry carries), so the seam must
        be able to say "not on this model" rather than emit a field it has no
        evidence the endpoint accepts. Every DECLARED entry today takes both
        settings — see the THINKING_DISPLAYS comment for the documentation.
      - `budget_min`: the minimum `budget_tokens` on a budget-mode endpoint
        (1024 on every model that still takes one). A budget must also be
        strictly less than `max_tokens`.

    A model whose entry leaves `thinking` as None is UNDECLARED, not
    unsupported: the seam refuses any explicit thinking spec for it and says so,
    rather than guessing a shape. Adding this field is additive for the seam's
    TRANSLATIONS — nothing declares it except the entries whose behaviour was
    verified — but declaring `default_on=True` also arms the starving-cap
    guard (`providers._refuse_starving_cap`), so a spec-less call that
    resolved before the declaration can be refused after it. That is the
    point of declaring, and it is said here so nobody reads "additive" as
    "changes no call's admissibility".
    """

    modes: tuple = ()
    efforts: tuple = ()
    default_on: bool = False
    default_effort: Optional[str] = None
    disabled_max_effort: Optional[str] = None
    displays: tuple = ()
    budget_min: int = 1024

    def __post_init__(self):
        for mode in self.modes:
            if mode not in THINKING_MODES:
                raise ValueError(
                    f"ThinkingSupport.modes: unknown mode {mode!r}; "
                    f"known modes are {list(THINKING_MODES)}")
        for effort in self.efforts:
            if effort not in EFFORT_LEVELS:
                raise ValueError(
                    f"ThinkingSupport.efforts: unknown effort {effort!r}; "
                    f"known levels are {list(EFFORT_LEVELS)}")
        if self.default_effort is not None:
            if self.default_effort not in self.efforts:
                raise ValueError(
                    f"ThinkingSupport.default_effort {self.default_effort!r} "
                    f"must be one of the declared efforts {list(self.efforts)}")
        if self.disabled_max_effort is not None:
            if THINKING_DISABLED not in self.modes:
                raise ValueError(
                    "ThinkingSupport.disabled_max_effort is meaningless "
                    "without THINKING_DISABLED in modes")
            if self.disabled_max_effort not in self.efforts:
                raise ValueError(
                    f"ThinkingSupport.disabled_max_effort "
                    f"{self.disabled_max_effort!r} must be one of the declared "
                    f"efforts {list(self.efforts)}")
            if self.default_effort is None:
                raise ValueError(
                    "ThinkingSupport.disabled_max_effort needs default_effort: "
                    "a request that disables thinking without naming an effort "
                    "is evaluated at the level in force when `effort` is "
                    "omitted, which must therefore be stated.")
        for display in self.displays:
            if display not in THINKING_DISPLAYS:
                raise ValueError(
                    f"ThinkingSupport.displays: unknown setting {display!r}; "
                    f"known settings are {list(THINKING_DISPLAYS)}")
        if self.displays and not (
                THINKING_ADAPTIVE in self.modes or THINKING_BUDGET in self.modes):
            raise ValueError(
                "ThinkingSupport.displays is meaningless without an on-mode "
                f"({THINKING_ADAPTIVE!r} or {THINKING_BUDGET!r}) in modes: "
                "`display` is invalid with disabled thinking, so a model that "
                "can only disable has nothing to display.")
        if self.budget_min < 1:
            raise ValueError("ThinkingSupport.budget_min must be positive")


@dataclass(frozen=True)
class Model:
    """One model's provider, routing, capabilities, and quirks.

    TWO KINDS OF DECODING PARAMETER
    -------------------------------
    Everything this record says about decoding falls into one of two classes,
    and they are not interchangeable:

      - CALLER-SPECIFIED. The caller chose it, so this record's job is to say
        whether the endpoint takes it. `rejects_sampling` and `thinking` are
        the two declarations; `resolved_decoding_params` honours a specified
        value the endpoint accepts, and for one it does not, either omits it
        (sampling) or refuses the call (thinking). An unspecified one is
        absent from the wire and from call identity, and the provider's own
        default applies — a fact about the provider, not about this table.
      - REGISTRY-DEFAULTED. This record supplies it when the caller says
        nothing: `quirks["reasoning_effort"]` is the only member. It is sent
        on a spec-less call to the entries that declare it and folds into
        call identity like any other sent param; a caller-chosen level,
        named through a `Thinking` spec and validated against the entry's
        declared `efforts`, takes its place on the wire and in identity.

    A caller-facing consumer reports an inert CALLER-SPECIFIED param (one it
    asked for that the endpoint will not take) and says nothing about either an
    unspecified one or a registry-defaulted one. Keeping the classes apart is
    what makes that reporting possible.

    `quirks` carries provider-specific decoding rules read by the adapters:

      - `reasoning_effort`: an OpenAI reasoning-effort level ("low" | "medium"
        | "high") the OpenAI adapter passes as `reasoning={"effort": ...}` on
        the Responses wire, or as `reasoning_effort` on the Chat Completions
        wire, when the call carries no `Thinking` spec of its own.
        REGISTRY-DEFAULTED, per the split above: a caller names its own level
        through the thinking seam, validated against the entry's declared
        `efforts`, and that level replaces this one on the wire and in
        identity.

    `wire_api` selects the OpenAI-family wire protocol (`WIRE_RESPONSES` or
    `WIRE_CHAT_COMPLETIONS`); it is read only by the OpenAI adapter and ignored
    for Anthropic models.

    `supports_images` declares whether the model's API accepts image content
    blocks. It is True for every vision-capable model and False for a text-only
    endpoint (one that rejects a message whose content carries an image part).
    A caller reads it via `model_supports_images` to decide whether to send
    image parts at all, rather than discovering the rejection on a paid call.
    Every shipped entry is currently vision-capable, and the flag's default is
    True — so the five routed entries state it because a probe sent an image,
    while the eleven direct entries inherit it from the published model
    reference (see HOW A FACT GETS INTO THIS TABLE).

    `forced_tool_choice` declares whether the model's endpoint honours a
    FORCED / named / "required" tool_choice (one that names a specific tool the
    model MUST call). False means only tool_choice "auto" routes, under which
    the model MAY decline to call the tool. The flag has NO default: whether an
    endpoint honours forcing was either established or it was not, and a
    default in either direction asserts the unestablished — a wrong True is a
    404 on a paid call, a wrong False silently makes the auto + retry path the
    normal one. Every entry states it, with its basis beside the value per HOW
    A FACT GETS INTO THIS TABLE.
    `tool_choice_named` reads this flag and emits the wire's "auto" form
    for a non-forcing model, so a call site that forces a named tool degrades to
    auto without changing its code; a caller arms its own bounded validate/retry
    loop off `supports_forced_tool_choice` (see `direktoro.providers`). It is
    NOT a call-identity input: the tool_choice mode follows from the model id,
    which identity already carries, so recording it is provenance, not identity.

    `rejects_sampling` names the sampling controls (`SAMPLING_PARAMS`:
    `temperature`, `top_p`, `top_k`) this model's endpoint REFUSES. Empty — the
    default — means it takes whatever it is sent, which is also what an entry
    says when nobody has established otherwise: this field claims refusals, never
    acceptances, so an undeclared param is sent rather than guessed at.

    It is one declaration covering what were two separate shapes of the same
    fact: a reasoning model that rejects only some controls, and a model whose
    provider dropped them all. Both are now a set of names, and a new sampling
    control is a name in that set rather than a new field. Today's values:
    Opus 4.7+ and Sonnet 5 refuse all three (the Claude model reference lists
    them as removed for that family); the GPT-5.x reasoning entries refuse
    `temperature`, which is what their documentation states and the only one
    established for them; Gemini 3.6 Flash refuses `temperature` and `top_p`,
    which is what its Vertex `supported_parameters` omits.

    `resolved_decoding_params` omits a refused param, so the omission is honest —
    the param is absent from BOTH the wire request and the recorded decoding
    params folded into call identity, never stripped from the wire while still
    fingerprinted. Editing a caller's temperature therefore never moves the
    identity of a model that would refuse it. The live 404 (`require_parameters`
    refuses an endpoint that would silently drop a sent param) stays the loud
    backstop should one ever leak through. Read via the
    `rejected_sampling_params` accessor (mirrors `supports_forced_tool_choice`);
    like it, the declaration reaches call identity only through
    `resolved_decoding_params` dropping the param, never on its own.

    `sampling_bands` declares, per sampling param, the (low, high) value range
    its reference documents as accepted — the model's own reference for a
    direct entry, its gateway's request surface for a routed one (see HOW A
    FACT GETS INTO THIS TABLE) — so a value outside it CAN be refused before
    the call is billed rather than 400ing on the first paid request. An
    absent band is an honest absence — no range has been established, and the
    value is sent for the endpoint's own answer to settle — and a declared
    band never claims everything inside it succeeds; the endpoint's own
    rejection stays the loud backstop for an in-band value it happens to
    dislike. A band for a param the entry `rejects_sampling` is refused at
    import: the endpoint cannot both refuse a param outright and accept a
    range of it. Read via the `sampling_band` accessor; handed out as a
    read-only mapping, because a recorded fact is not something a reader can
    edit in place.

    `retired` marks an id the provider has withdrawn: the entry is kept so past
    runs still resolve against it, but it must never start a NEW run (a live
    call would 404). It defaults False, so flagging one is a deliberate act.

    NOTHING IN THIS PACKAGE REFUSES A RETIRED ID ON YOUR BEHALF. No lookup here
    reads the flag, and none may: `model_info`, `build_adapter`,
    `resolved_decoding_params`, `tool_choice_named` and `call_identity_fields`
    all have to keep resolving a withdrawn id or the provenance of a run that
    already happened stops being interpretable. `known_models()` lists retired
    ids for the same reason. So the gate is a thing a caller applies, at
    whatever point it decides a new run is starting, and this module's job is to
    make that question askable rather than to pick the moment:
    `is_retired(model_id)` is the predicate and
    `known_models(include_retired=False)` is the startable list. direktoro's own
    CLI applies them in `resolve_models` at config load, so its failure is loud
    and up front rather than a 404 part-way through a run; a library consumer
    that applies neither will reach the provider and get the 404.

    It is not part of call identity: identity carries the model id string, not
    this struct, so setting the flag never moves a caller's fingerprint.

    `route` marks a gateway-served (OpenRouter) entry: the adapter emits its
    `Route` as OpenRouter's provider object, captures the generation id, served
    upstream and reported cost, and runs the pin assertion on every response.
    `route=None` is a direct entry (Anthropic / OpenAI), whose response carries
    no cost figure of its own.

    `thinking` is the model's `ThinkingSupport` capability record, read by
    `direktoro.providers.resolved_decoding_params` so a thinking / effort shape
    the endpoint would reject is refused before it is sent, not after it is
    billed. None means UNDECLARED, and the seam then refuses any explicit spec
    and says so rather than guessing a shape for an endpoint nobody has
    verified. It reaches call identity only through `resolved_decoding_params`
    emitting a chosen mode or effort into `decoding_params`, exactly like
    `temperature`, never on its own.

    ADDING A FIELD: APPEND IT AT THE END.
    A new field goes after the LAST field (today `sampling_bands`), with a
    default, and nowhere else. This
    record is constructed POSITIONALLY — every entry in `MODEL_REGISTRY` passes
    provider / base_url / api_key_env positionally, this package's own tests
    construct `Model(...)` the same way, and so does anything downstream that
    builds a synthetic entry. Inserting a field in the middle rebinds every one
    of those positional arguments to the attribute after it, silently and
    without a TypeError: a base URL lands in `api_key_env`, a key env lands in
    `quirks`. Appending is the only edit that cannot do that, and a default is
    what keeps existing constructions positionally valid.
    `forced_tool_choice` deliberately trades the second half of that away:
    its None default is a sentinel the constructor refuses, so every
    construction — registry entry or synthetic — must state the flag. A new
    field must not repeat that trade without the same grade of justification,
    because each sentinel breaks every existing downstream construction once.
    `tests/test_cost.py::TestModelFieldOrder` pins the order so the rule fails
    loudly rather than being remembered.
    """

    provider: str
    base_url: Optional[str]
    api_key_env: str
    quirks: dict = field(default_factory=dict)
    wire_api: str = WIRE_RESPONSES
    supports_images: bool = True
    # None is a sentinel, not a value: construction refuses it in
    # __post_init__. The field keeps a "default" only so the positional
    # append-only rule holds (see ADDING A FIELD below).
    forced_tool_choice: Optional[bool] = None
    rejects_sampling: frozenset = frozenset()
    retired: bool = False
    route: Optional[Route] = None
    thinking: Optional[ThinkingSupport] = None
    sampling_bands: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        # Routing (provider-object emission, pin assertion, reported-cost
        # capture) is implemented only on the Chat Completions path; a routed
        # entry on any other wire would silently bypass ALL of it, so the
        # combination is refused at import.
        if self.route is not None and self.wire_api != WIRE_CHAT_COMPLETIONS:
            raise ValueError(
                "a routed model entry must use wire_api="
                "WIRE_CHAT_COMPLETIONS: the OpenRouter provider object, "
                "pin assertion, and reported-cost capture exist only on "
                "that path.")
        # `rejects_sampling` occupies the slot a boolean flag once did, and the
        # record is reachable positionally, so a value of the wrong shape is
        # refused at import rather than read as a truthy set of no names. An
        # unknown name is refused for the opposite reason: it would silently
        # declare a refusal of something nothing ever sends, which reads as a
        # documented fact while doing nothing.
        if not isinstance(self.rejects_sampling, (frozenset, set)):
            raise TypeError(
                f"rejects_sampling must be a set of sampling parameter names "
                f"from {list(SAMPLING_PARAMS)}, got "
                f"{type(self.rejects_sampling).__name__}.")
        unknown = sorted(set(self.rejects_sampling) - set(SAMPLING_PARAMS))
        if unknown:
            raise ValueError(
                f"rejects_sampling names {unknown}, which are not sampling "
                f"parameters; it accepts {list(SAMPLING_PARAMS)}.")
        # Whether an endpoint honours a forced tool_choice was either
        # established or it was not, and a default in either direction would
        # assert the unestablished — so there is no working default and every
        # entry states the flag, with its basis in the entry's comment. The
        # SHAPE is checked too: the record is reachable positionally, and a
        # truthy non-bool ("yes", "false") read as True would force a named
        # tool on an endpoint that 404s one — the paid failure the statement
        # requirement exists to prevent.
        if not isinstance(self.forced_tool_choice, bool):
            raise ValueError(
                "Model.forced_tool_choice must be stated as a bool: True "
                "(the endpoint honours a forced / named tool_choice) or "
                'False (only "auto" routes). There is no default; record the '
                "basis — vendor documentation or a dated probe — beside the "
                f"value. Got {self.forced_tool_choice!r}.")
        # Same positional exposure, same shape rule as `rejects_sampling`:
        # the wrong container is refused at import, not read for whatever
        # `.items()` it happens to lack.
        if not isinstance(self.sampling_bands, (dict, MappingProxyType)):
            raise TypeError(
                f"sampling_bands must be a dict of sampling parameter names "
                f"to (low, high) tuples, got "
                f"{type(self.sampling_bands).__name__}.")
        for param, band in self.sampling_bands.items():
            if param not in SAMPLING_PARAMS:
                raise ValueError(
                    f"sampling_bands names {param!r}, which is not a sampling "
                    f"parameter; it accepts {list(SAMPLING_PARAMS)}.")
            if param in self.rejects_sampling:
                raise ValueError(
                    f"sampling_bands declares a band for {param!r}, which "
                    f"rejects_sampling says the endpoint refuses outright; "
                    f"one of the two declarations is wrong. (A synthetic "
                    f"entry built with dataclasses.replace must adjust both "
                    f"fields together.)")
            # NaN never compares, so a NaN bound would satisfy `low <= high`
            # vacuously and then fail EVERY later range test; an infinite
            # bound declares a fact while claiming nothing. Both are refused
            # as malformed, not recorded.
            if (not isinstance(band, tuple) or len(band) != 2
                    or not all(isinstance(b, (int, float))
                               and not isinstance(b, bool)
                               and math.isfinite(b) for b in band)
                    or band[0] > band[1]):
                raise ValueError(
                    f"sampling_bands[{param!r}] must be a (low, high) tuple "
                    f"of finite numbers with low <= high, got {band!r}.")
        # A band is a recorded documented fact, so it is not handed out
        # mutable: the mapping is copied and frozen here, and every reader
        # gets the same read-only view. (`rejects_sampling` gets this for
        # free from being a frozenset; a mapping needs it done by hand.)
        object.__setattr__(self, "sampling_bands",
                           MappingProxyType(dict(self.sampling_bands)))


# The sampling controls a caller may specify, and the only names
# `Model.rejects_sampling` accepts. Ordered, and emitted in this order, so the
# decoding params folded into call identity do not depend on set iteration.
SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

# Opus 4.7 and later (4.7, 4.8, 5) and Sonnet 5 reject the sampling controls: a
# non-default temperature/top_p/top_k returns a 400. Claude model reference,
# read 2026-08-01, which lists all three as removed for this family — one
# documented family fact, recorded once and shared by the entries it covers
# rather than transcribed per row.
_NO_SAMPLING = frozenset(SAMPLING_PARAMS)

# The GPT-5.x reasoning entries. Their documentation establishes that they
# reject `temperature`; it says nothing either way about the other two, so
# neither is named. `rejects_sampling` claims refusals only, so an unestablished
# one is sent and the endpoint's own answer settles it.
_NO_TEMPERATURE = frozenset({"temperature"})

# Anthropic documents temperature "0.0 to 1.0" and publishes no numeric range
# for top_p or top_k (Messages API reference, read 2026-08-12) — one
# documented API-wide fact, recorded once and shared by the live entries that
# still take sampling, rather than transcribed per row.
_ANTHROPIC_SAMPLING_BANDS = {"temperature": (0.0, 1.0)}

# OpenRouter documents temperature 0.0 to 2.0 and top_p 0.0 to 1.0 for its
# request surface; top_k is documented one-sided ("0 or above"), which a
# (low, high) band cannot state, so none is declared (gateway API reference,
# read 2026-08-12). One documented gateway fact, shared by the routed entries
# that take sampling — see the SAMPLING BANDS note in the routed section.
_OPENROUTER_SURFACE_BANDS = {"temperature": (0.0, 2.0), "top_p": (0.0, 1.0)}

# ---- Thinking / effort capability presets --------------------------------
# Anthropic model reference + migration guide, verified 2026-07-31. These are
# what makes the seam able to refuse a 400-producing shape BEFORE it is sent.
# The effort ladder gained `xhigh` with Opus 4.7, so the 4.6 generation stops at
# high/max. `budget_tokens` is REMOVED (400) from Opus 4.7 onward and on Sonnet
# 5; it survives as a deprecated escape hatch on the 4.6 generation and is the
# only way to think at all on pre-4.6 models.
_EFFORTS_4_7 = ("low", "medium", "high", "xhigh", "max")
_EFFORTS_4_6 = ("low", "medium", "high", "max")

# Every Anthropic entry whose thinking surface is declared below accepts both
# `thinking.display` settings, in whichever on-mode it supports; only the
# DEFAULT differs by generation, and a default is not a capability (see the
# THINKING_DISPLAYS comment above). Naming the tuple rather than repeating it
# keeps a future entry that does NOT take the field an explicit, visible
# `displays=()` rather than a silent omission.
_DISPLAYS_BOTH = THINKING_DISPLAYS

# Claude Opus 5. Thinking is ON when the `thinking` parameter is omitted
# (unlike Opus 4.8/4.7) — the trap this seam exists to close, because
# `max_tokens` caps thinking plus response together. `{"type": "disabled"}` is
# accepted ONLY at effort `high` or below; pairing it with xhigh/max is a 400.
_THINK_OPUS_5 = ThinkingSupport(
    modes=(THINKING_ADAPTIVE, THINKING_DISABLED), efforts=_EFFORTS_4_7,
    default_on=True, default_effort="high", disabled_max_effort="high",
    displays=_DISPLAYS_BOTH)

# Claude Opus 4.8 / 4.7. Adaptive is the only on-mode and it is NOT on by
# default: omitting `thinking` runs without thinking. `disabled` is accepted at
# any effort.
_THINK_OPUS_4_7 = ThinkingSupport(
    modes=(THINKING_ADAPTIVE, THINKING_DISABLED), efforts=_EFFORTS_4_7,
    default_on=False, default_effort="high", displays=_DISPLAYS_BOTH)

# Claude Sonnet 5. Adaptive is the only on-mode and it RUNS when `thinking` is
# omitted (Sonnet 4.6 did not); `disabled` is accepted at any effort. Sonnet 5
# is the first Sonnet-tier model with `xhigh`.
_THINK_SONNET_5 = ThinkingSupport(
    modes=(THINKING_ADAPTIVE, THINKING_DISABLED), efforts=_EFFORTS_4_7,
    default_on=True, default_effort="high", displays=_DISPLAYS_BOTH)

# Claude Sonnet 4.6. Adaptive is recommended but not the default (omitting
# `thinking` runs without thinking); `budget_tokens` still functions as the
# deprecated transitional escape hatch; no `xhigh`.
_THINK_SONNET_4_6 = ThinkingSupport(
    modes=(THINKING_ADAPTIVE, THINKING_DISABLED, THINKING_BUDGET),
    efforts=_EFFORTS_4_6, default_on=False, default_effort="high",
    displays=_DISPLAYS_BOTH)

# Pre-4.6 models (Haiku 4.5): thinking only via
# `{"type": "enabled", "budget_tokens": N}` with N >= 1024 and N < max_tokens.
# The `effort` parameter ERRORS on these models, so no levels are declared and
# asking for one is refused.
_THINK_BUDGET_ONLY = ThinkingSupport(
    modes=(THINKING_BUDGET,), efforts=(), default_on=False,
    displays=_DISPLAYS_BOTH)

# ---- Routed-entry thinking surfaces (live probes 2026-08-12) ---------------
# Probed through the production pins (provider object, require_parameters,
# zdr / data_collection as declared): one plain call, then one call per
# `reasoning_effort` value in {low, medium, high, xhigh, max, minimal, none}.
# "Accepted" means the pinned endpoint routed the call (200) under
# require_parameters, which refuses an endpoint that does not support a sent
# parameter.

# The GLM vision pair (probed per entry, same Z.AI host): reasoning runs on a
# plain call, so default_on; all five ladder levels are accepted; and
# `reasoning_effort: "none"` returns zero reasoning tokens — the off-switch
# that THINKING_DISABLED renders to on this wire. No level is documented or
# observable as the omitted-state default, so default_effort stays None.
# THE PROBE FOUND NO EVIDENCE THE LEVEL BOUNDS VOLUME: on one reasoning-heavy
# prompt at cap 8192 (glm-4.6v, 3 samples per level), "low" consumed
# 5292-8191 reasoning tokens and "max" 5902-8191, with samples at BOTH levels
# censored at the cap — too small and too truncated to establish a per-level
# bound in either direction. Declaring a level buys the wire fact; treat any
# spend expectation attached to it as unmeasured.
_THINK_GLM_VISION = ThinkingSupport(
    modes=(THINKING_ADAPTIVE, THINKING_DISABLED),
    efforts=("low", "medium", "high", "xhigh", "max"),
    default_on=True)

# Gemini 3.6 Flash at the pinned Vertex flex endpoint: reasoning runs on a
# plain call; all five ladder levels are accepted at the wire, which is what
# `efforts` declares. Two honesty notes on what acceptance does NOT establish:
# per-level reasoning volumes were non-monotone at one sample per level
# (low 91, medium 100, high 130, xhigh 120, max 109 tokens); and OpenRouter's
# reference maps effort onto Google's thinkingLevel with "xhigh" folding to
# "high" upstream (gateway docs read 2026-08-12) and no stated mapping for
# "max" — so two declared levels can denote one served behaviour while
# fingerprinting as what was sent. `reasoning_effort: "none"` is REFUSED —
# 400 "Reasoning is mandatory for this endpoint and cannot be disabled" — so
# there is no disabled mode. ("minimal" was also accepted and returned zero
# reasoning tokens, but it is not a ladder level and nothing here emits it.)
_THINK_GEMINI_FLASH = ThinkingSupport(
    modes=(THINKING_ADAPTIVE,),
    efforts=("low", "medium", "high", "xhigh", "max"),
    default_on=True)

# Qwen3-VL 235B Instruct: an instruct, non-reasoning endpoint. Every
# reasoning_effort value 404s under require_parameters ("no endpoints found
# that can handle the requested parameters") and a plain call produces no
# reasoning tokens. Declared EMPTY rather than left undeclared — the absence
# of a thinking surface is a probed fact here, not a gap — and each field is
# stated rather than defaulted, because a value left at its field default
# records nothing (module docstring). budget_min keeps its inert default:
# there is no reasoning parameter for it to be a minimum of.
_THINK_QWEN_INSTRUCT = ThinkingSupport(modes=(), efforts=(), default_on=False)


MODEL_REGISTRY = {
    # ---- Anthropic (Claude) -------------------------------------------------
    # Snapshot note (docs verified 2026-07-31): the dateless ids below
    # (claude-opus-5, claude-opus-4-8, claude-opus-4-7, claude-sonnet-5,
    # claude-sonnet-4-6) are the 4.6-generation naming scheme, in which the
    # dateless id IS the pinned snapshot — NOT a rolling alias awaiting a dated
    # form. Anthropic does not update the weights or configuration of an
    # existing id; a new model gets a new id. These are citation-grade as
    # written, so there is nothing to repoint. The pre-4.6 models below keep
    # their dated ids because for THOSE the bare alias really is a repointable
    # pointer. The routed GLM / Qwen slugs remain the genuinely rolling case
    # (see the module docstring).
    #
    # FORCED TOOL_CHOICE (every LIVE Anthropic entry below): the Messages API
    # documents `tool_choice: {"type": "tool", "name": ...}` — "the model will
    # use the specified tool" — with no per-model carve-out (API reference,
    # read 2026-08-12), so each live entry states forced_tool_choice=True on
    # that documentation. The three RETIRED entries state True on a weaker
    # basis, said plainly: the same tool_choice documentation applied to them
    # while they were live and predates their retirements, but a withdrawn
    # endpoint cannot be re-verified, and the current reference no longer
    # describes it. The flag must still be stated (there is no default), the
    # new-run gate keeps it unreachable, and the weaker evidence class is the
    # honest price of that combination.
    #
    # Opus 5. Context 1M (default and maximum), max output 128K. Rejects
    # temperature/top_p/top_k like the rest of the 4.7+ family (non-default
    # values return 400). THINKING: adaptive is ON when the `thinking` param is
    # omitted (unlike 4.8/4.7, where omitting it means no thinking), and
    # max_tokens caps thinking PLUS response text, so a cap tuned on 4.8 buys
    # less answer here; `{"type": "disabled"}` is accepted only at effort `high`
    # or below (400 at xhigh/max); `budget_tokens` is a 400. All five effort
    # levels. Thinking tokens are billed by the provider as output tokens, so a
    # caller counting them needs no separate counter. Limits and behaviour
    # verified against the published model and deprecation tables 2026-07-31.
    "claude-opus-5": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True,
        rejects_sampling=_NO_SAMPLING, thinking=_THINK_OPUS_5),
    # Opus 4.8. Context 1M, max output 128K. Adaptive thinking is the only
    # on-mode and is OFF when the `thinking` param is omitted. Verified
    # 2026-07-31.
    "claude-opus-4-8": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True,
        rejects_sampling=_NO_SAMPLING, thinking=_THINK_OPUS_4_7),
    # Opus 4.7. Context 1M, max output 128K. Same thinking surface as 4.8 — one
    # documented family fact shared via `_THINK_OPUS_4_7` (Anthropic model
    # reference + migration guide, read 2026-07-31), not a value copied across
    # from the neighbouring row.
    "claude-opus-4-7": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True,
        rejects_sampling=_NO_SAMPLING, thinking=_THINK_OPUS_4_7),
    # Sonnet 5. Context 1M, max output 128K. Dateless 4.6-generation id, i.e. a
    # pinned snapshot (see the snapshot note above), so it is citation-grade as
    # written. Rejects all three sampling controls like the Opus 4.7+ family (the
    # Claude model reference lists Sonnet 5's temperature/top_p/top_k as
    # removed -> 400), so it carries _NO_SAMPLING; supports images (default).
    # THINKING: adaptive runs when the param is omitted (Sonnet 4.6 does not),
    # `disabled` is accepted at any effort, `budget_tokens` is a 400, and it is
    # the first Sonnet with `xhigh`.
    "claude-sonnet-5": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True,
        rejects_sampling=_NO_SAMPLING, thinking=_THINK_SONNET_5),
    # Sonnet 4.6. Context 1M, max output 128K. Takes sampling params (no
    # refusals). Adaptive thinking is OFF when the param is omitted;
    # `budget_tokens` still functions here as the deprecated escape hatch; the
    # effort ladder stops at `max` (no `xhigh` before Opus 4.7).
    "claude-sonnet-4-6": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True,
        thinking=_THINK_SONNET_4_6,
        sampling_bands=_ANTHROPIC_SAMPLING_BANDS),
    # Haiku 4.5. Context 200K, max output 64K — the only current Anthropic entry
    # that is not 1M/128K. Pre-4.6 generation: thinking only via
    # `budget_tokens`, and the `effort` parameter errors, so no levels are
    # declared. Keyed by its DATED id because for pre-4.6 models the bare alias
    # is a repointable pointer (see the module docstring).
    "claude-haiku-4-5-20251001": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True,
        thinking=_THINK_BUDGET_ONLY,
        sampling_bands=_ANTHROPIC_SAMPLING_BANDS),
    # The three entries below are RETIRED on the Anthropic API (deprecation
    # table verified 2026-07-31): a live call fails, so they must never start a
    # new run. They are kept, flagged rather than deleted, because the registry
    # is also the provenance table for runs that ALREADY happened — deleting
    # them would leave the model id a past run recorded unresolvable.
    # `retired=True` keeps every lookup working and is refused only where a
    # caller applies the gate (`is_retired` / `known_models(include_retired=
    # False)`; direktoro's own CLI applies it at config load). Prefer the flag
    # over deletion for anything that may have run.
    # They deliberately declare NO `thinking` support: the CLI's new-run gate
    # already refuses them, so no new call can carry a thinking spec, and
    # leaving the field undeclared means the seam refuses rather than asserting
    # capability facts about a withdrawn endpoint nobody can re-verify.
    # Sonnet 4 (legacy): retired 2026-06-15, replaced by claude-sonnet-5.
    "claude-sonnet-4-20250514": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True, retired=True),
    # Claude 3.5 Sonnet (legacy): retired 2025-10-28, replaced by
    # claude-sonnet-5.
    "claude-3-5-sonnet-20241022": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True, retired=True),
    # Opus 4 (legacy): retired 2026-06-15, replaced by claude-opus-5.
    "claude-opus-4-20250514": Model(
        PROVIDER_ANTHROPIC, None, ANTHROPIC_KEY_ENV,
        forced_tool_choice=True, retired=True),

    # ---- OpenAI (GPT) -------------------------------------------------------
    # FORCED TOOL_CHOICE (both GPT-5.6 entries): OpenAI's function-calling
    # reference documents forcing a named function via tool_choice, from the
    # published-reference reads of 2026-07-31 / 2026-08-01 that populated
    # these rows. No sampling band is declared: the range of the params these
    # entries still accept was not re-read.
    #
    # GPT-5.6 flagship. Reasoning model: rejects temperature, defaults to medium
    # reasoning effort.
    "gpt-5.6-sol": Model(
        PROVIDER_OPENAI, OPENAI_BASE_URL, OPENAI_KEY_ENV,
        quirks={"reasoning_effort": "medium"},
        forced_tool_choice=True,
        rejects_sampling=_NO_TEMPERATURE,
        wire_api=WIRE_RESPONSES),
    # GPT-5.6 mid-tier. Same reasoning-model surface as the flagship.
    "gpt-5.6-terra": Model(
        PROVIDER_OPENAI, OPENAI_BASE_URL, OPENAI_KEY_ENV,
        quirks={"reasoning_effort": "medium"},
        forced_tool_choice=True,
        rejects_sampling=_NO_TEMPERATURE,
        wire_api=WIRE_RESPONSES),

    # ---- Routed via OpenRouter ---------------------------------------------
    # SAMPLING BANDS (`_OPENROUTER_SURFACE_BANDS`, the routed entries that
    # take sampling): these describe the OpenRouter request surface itself —
    # not any upstream's narrower taste, which no probe of a continuous range
    # could establish. What the gateway itself would do with an out-of-range
    # value (refuse, clamp, forward) was not probed; the band exists so a
    # caller can refuse such a value before any spend, and an upstream
    # rejection of an in-band value stays the loud backstop.
    #
    # GLM / Qwen traffic routes through OpenRouter's OpenAI-compatible Chat
    # Completions surface, pinned to a named upstream and fingerprinted. The
    # registry id IS the OpenRouter model slug verbatim (id-as-identity). A
    # routed call's cost comes back on the response (OpenRouter usage.cost,
    # requested via usage.include) and is recorded as reported. Slugs + upstream
    # provider + quant + vision/tools support were confirmed live 2026-07-23
    # (GET /api/v1/models and /endpoints, then a plain + tool + vision probe
    # each); the served-provider attribution field is the completion's top-level
    # `provider` (a display name, e.g. "Z.AI").
    #
    # GLM 5V Turbo: OpenRouter serves this only from Z.AI itself (single fp8
    # endpoint), so upstream is pinned to z-ai and the quant to fp8 — a coarser
    # served variant is refused. Plain completion, vision, and prompt caching
    # confirmed live 2026-07-23. TOOL-CALLING CAVEAT (both GLM vision endpoints,
    # confirmed live): the Z.AI-hosted endpoint rejects a FORCED / "required"
    # tool_choice with a 404 (no endpoint supports that tool_choice value),
    # independent of require_parameters; only tool_choice "auto" routes, under
    # which the model MAY decline to call the tool. A caller that forces a named
    # tool must fall back to "auto" plus its own retry for these two models; it
    # reads `supports_forced_tool_choice` to know that up front.
    "z-ai/glm-5v-turbo": Model(
        PROVIDER_OPENROUTER, OPENROUTER_BASE_URL, OPENROUTER_KEY_ENV,
        wire_api=WIRE_CHAT_COMPLETIONS, supports_images=True,
        # A FORCED / named tool_choice 404s on this Z.AI-hosted endpoint
        # through OpenRouter ("no endpoint supports that tool_choice value",
        # confirmed live 2026-07-23), so this entry runs tool_choice "auto" and
        # a caller arms its own bounded validate/retry loop (see the
        # TOOL-CALLING CAVEAT above).
        forced_tool_choice=False,
        sampling_bands=_OPENROUTER_SURFACE_BANDS,
        thinking=_THINK_GLM_VISION,
        route=Route(gateway=GATEWAY_OPENROUTER, upstream=("z-ai",),
                    quantizations=("fp8",))),
    # GLM 4.6V: served by both Z.AI (fp8) and Novita (bf16); pinned to Z.AI
    # (first-party) at fp8 so provenance is unambiguous. Vision + tools
    # confirmed live.
    "z-ai/glm-4.6v": Model(
        PROVIDER_OPENROUTER, OPENROUTER_BASE_URL, OPENROUTER_KEY_ENV,
        wire_api=WIRE_CHAT_COMPLETIONS, supports_images=True,
        # Same forced-tool_choice 404 as glm-5v-turbo (both GLM vision
        # endpoints, confirmed live 2026-07-23): runs "auto" + retry.
        forced_tool_choice=False,
        sampling_bands=_OPENROUTER_SURFACE_BANDS,
        thinking=_THINK_GLM_VISION,
        route=Route(gateway=GATEWAY_OPENROUTER, upstream=("z-ai",),
                    quantizations=("fp8",))),
    # Qwen3-VL flagship (235B-A22B Instruct), pinned to Venice + Parasail at fp8.
    # All confirmed live 2026-07-23:
    #   1. OpenRouter does NOT host the proprietary `qwen3-vl-plus` commercial
    #      tier — it carries only the open-weight Qwen3-VL variants (8B /
    #      30B-A3B / 32B / 235B-A22B), of which this 235B is the flagship.
    #   2. The first-party upstream (Alibaba) offers NO request-level ZDR through
    #      OpenRouter (alibaba + zdr -> 404 no-endpoints), so under this entry's
    #      privacy pin the model cannot be served first-party.
    #   3. Two upstreams are pinned rather than one because pinned-upstream churn
    #      is real and fast: an upstream serving this slug at discovery was
    #      deranked within hours. Venice and Parasail are the healthy fp8
    #      endpoints, both verified live under zdr+deny with forced tool calling
    #      AND vision (Venice's image validator rejects tiny probe images but
    #      accepts realistic page-sized renders). A caller that cares about the
    #      quantization should declare the pinned fp8 level in its own
    #      provenance record.
    #      READ THIS ALONGSIDE THE MiMo ENTRY BELOW, WHICH RECORDS THE OPPOSITE
    #      PARASAIL RESULT ON THE SAME DATE. Both probes are per-(host, model),
    #      not per-host: forced tool calling was probed for THIS slug on Parasail
    #      and routed, and probed for `xiaomi/mimo-v2.5` on Parasail and 404ed.
    #      A host serves each model through its own deployment, so "Parasail
    #      honours a forced tool_choice" is not a fact either probe establishes
    #      and neither entry claims it. The two records are not in conflict, and
    #      neither flag may be changed to match the other without re-probing the
    #      pair (host, model) it actually describes.
    "qwen/qwen3-vl-235b-a22b-instruct": Model(
        PROVIDER_OPENROUTER, OPENROUTER_BASE_URL, OPENROUTER_KEY_ENV,
        wire_api=WIRE_CHAT_COMPLETIONS, supports_images=True,
        # Forced named tool_choice returned the named call on both pinned
        # hosts, live 2026-07-23 (point 3 above; the per-(host, model) probe
        # discipline is spelled out there).
        forced_tool_choice=True,
        sampling_bands=_OPENROUTER_SURFACE_BANDS,
        thinking=_THINK_QWEN_INSTRUCT,
        route=Route(gateway=GATEWAY_OPENROUTER, upstream=("venice", "parasail"),
                    quantizations=("fp8",))),
    # Xiaomi MiMo v2.5, pinned to Parasail + Venice at fp8. Vision + prompt
    # caching verified live on both hosts under zdr+deny 2026-07-23.
    # HOST PIN (live /endpoints 2026-07-23): DigitalOcean also serves this slug
    # but reports quantization "unknown", so the declared-fp8 pin (which refuses
    # a coarser served variant) excludes it. Xiaomi's own first-party endpoint is
    # fp8 but is a NON-ZDR path (like Alibaba for Qwen), so it is excluded under
    # this entry's privacy pin; Parasail + Venice are the ZDR-capable third
    # parties.
    # FORCED-TOOL-CHOICE (live probes 2026-07-23, the decisive findings): Venice
    # HONOURS a forced / named tool_choice (returned the named tool call under
    # zdr+deny+fp8). Parasail does NOT: a forced named tool_choice 404s ("No
    # endpoints found", independent of require_parameters) — the same failure
    # mode as the Z.AI-hosted GLM endpoints — while tool_choice "auto" routes to
    # Parasail cleanly. To keep behaviour UNIFORM across the two-host pin (rather
    # than drop to a single Venice-only host and lose the two-upstream churn
    # resilience), this entry runs tool_choice "auto" and a caller arms its own
    # bounded validate/retry loop off `supports_forced_tool_choice` — exactly
    # the GLM degrade path — so `forced_tool_choice=False`.
    # This Parasail result is for THIS slug. The Qwen entry above, probed the
    # same day against the same host, records forced tool calling working there;
    # a host serves each model through its own deployment, so the two findings
    # are per-(host, model) and describe different things rather than
    # contradicting each other. See the matching note on the Qwen entry.
    # COLOUR-FIDELITY CAVEAT (live probe 2026-07-23): MiMo mis-described a solid
    # blue 512px square as "grayscale", a figure-reading fidelity risk a caller
    # should measure on its own material before trusting this model with
    # images whose colour carries meaning.
    "xiaomi/mimo-v2.5": Model(
        PROVIDER_OPENROUTER, OPENROUTER_BASE_URL, OPENROUTER_KEY_ENV,
        wire_api=WIRE_CHAT_COMPLETIONS, supports_images=True,
        # Parasail 404s a forced named tool_choice (like the Z.AI GLM endpoints);
        # Venice honours it. Runs "auto" + retry to keep the two-host pin uniform
        # (live-confirmed 2026-07-23; see the FORCED-TOOL-CHOICE note above).
        forced_tool_choice=False,
        sampling_bands=_OPENROUTER_SURFACE_BANDS,
        # THINKING SURFACE: UNDECLARED, with the partial probe findings
        # recorded (2026-08-12): a plain call reasons, and reasoning_effort
        # low / xhigh / minimal were accepted with "none" returning zero
        # reasoning tokens — but Parasail rate-limited the probes for
        # medium / high / max, so the ladder is unestablished and a partial
        # `efforts` tuple would read as refusals nobody observed. Left None
        # until the remaining levels are probed.
        route=Route(gateway=GATEWAY_OPENROUTER,
                    upstream=("parasail", "venice"),
                    quantizations=("fp8",))),
    # Google Gemini 3.6 Flash, pinned to Google Vertex at the FLEX service tier.
    # VERTEX TIER PINNING: OpenRouter exposes Vertex's
    # service tiers as distinct endpoint tags, and 3.6 lists three of them —
    # google-vertex/global, .../global/flex and .../global/priority. Flex is
    # pinned by the full tag in `provider.order`
    # (`order=["google-vertex/global/flex"]` routes to flex and 404s if flex
    # cannot serve, allow_fallbacks False), so the tier a call is served at is
    # declared rather than chosen by the gateway.
    # THE TIER PIN IS REQUEST-SIDE ONLY, and is the one Route field the response
    # cannot confirm. A completion's served-provider attribution names the
    # PROVIDER ("Google") and never the tier, so `assert_served_upstream` folds
    # this order token to its slug head and checks only that Vertex served the
    # call — `.../global`, `.../global/flex` and `.../global/priority` are
    # indistinguishable to it. Nothing else in this package compares the tier
    # against anything either: `reported_cost` is captured and threaded, never
    # checked. So an identity block for this entry says "flex" on the strength
    # of the token that was SENT, which is weaker than every other asserted
    # field here, and is recorded as such rather than read as a confirmation.
    # NO SAMPLING PARAMS: the 3.6 Vertex
    # endpoints' supported_parameters list include_reasoning, max_tokens,
    # reasoning, reasoning_effort, response_format, seed, stop,
    # structured_outputs, tool_choice, tools — and NONE of temperature, top_p or
    # top_k (Google dropped sampling controls on 3.6), so this entry names all
    # three in rejects_sampling and resolved_decoding_params omits them from
    # both wire and fingerprint.
    # LIVE PROBES (2026-07-24): the full pin
    # (order=["google-vertex/global/flex"], allow_fallbacks False, require_
    # parameters True, data_collection deny, zdr True) PLUS temperature:0.0 ->
    # 404 "No endpoints found matching your data policy" (the loud backstop, as
    # designed — require_parameters refuses the endpoint that would drop the sent
    # temperature). The SAME pin WITHOUT temperature, with a forced named
    # tool_choice and a 512px image -> SUCCESS: provider attribution "Google",
    # generation id gen-1784894825-..., usage.cost present, finish_reason
    # "tool_calls", correct tool call identifying the image (vision works). So
    # 3.6 HONOURS a forced named tool_choice (forced_tool_choice=True, unlike
    # GLM/MiMo), and the flex tier tag folds to the served attribution for the
    # pin assertion (served "Google" -> googlevertex alias -> google, pin
    # passes), which confirms Vertex and not the tier — see THE TIER PIN IS
    # REQUEST-SIDE ONLY above.
    # Architecture: input text/image/video/file/audio, context 1,048,576.
    # SINGLE-PROVIDER DEPENDENCY: only Vertex is ZDR-capable — Google AI Studio
    # (google-ai-studio*) is NOT ZDR — so there is no second upstream to pin, a
    # churn and single-point-of-failure risk a consumer should declare.
    # quantizations=() : Vertex is proprietary and reports quant "unknown", so a
    # quant filter would wrongly exclude it.
    "google/gemini-3.6-flash": Model(
        PROVIDER_OPENROUTER, OPENROUTER_BASE_URL, OPENROUTER_KEY_ENV,
        wire_api=WIRE_CHAT_COMPLETIONS, supports_images=True,
        # Forced named tool_choice verified live at the flex endpoint 2026-07-24.
        forced_tool_choice=True,
        # The 3.6 Vertex endpoints list no sampling control at all — temperature,
        # top_p and top_k are absent from the same /endpoints supported_parameters
        # read (live 2026-07-24) transcribed above, which is one enumeration and
        # settles all three together. Omit them honestly from wire and
        # fingerprint, and let the require_parameters 404 be the loud backstop.
        rejects_sampling=frozenset({"temperature", "top_p", "top_k"}),
        thinking=_THINK_GEMINI_FLASH,
        route=Route(gateway=GATEWAY_OPENROUTER,
                    upstream=("google-vertex/global/flex",),
                    quantizations=())),
}


def known_models(*, include_retired=True):
    """Sorted list of known model ids (the registry keys).

    `include_retired` chooses which of the registry's two jobs you are asking
    about, and they want different lists:

      - True (the default) is the PROVENANCE list — every id this registry can
        resolve, retired ones included, because a run that already happened on a
        withdrawn id must still resolve and still be citable. It is exactly
        `sorted(MODEL_REGISTRY)`.
      - False is the STARTABLE list — the ids a NEW run may name. Use this
        wherever a config is validated or a model is offered for selection: a
        retired id would resolve perfectly here and then 404 at the provider.

    The default is the wider list deliberately. Narrowing it would silently drop
    ids from every consumer's lookup on upgrade, which is the failure mode the
    retired flag exists to avoid; asking for the narrower one is a deliberate
    act, exactly like flagging an entry retired in the first place.
    """
    if include_retired:
        return sorted(MODEL_REGISTRY)
    return sorted(model_id for model_id, info in MODEL_REGISTRY.items()
                  if not info.retired)


def is_known_model(model):
    """True when `model` has a registry entry.

    Membership only: this is True for a RETIRED id too, because a retired entry
    is still a resolvable entry. Ask `is_retired` as well before starting a new
    run on one.
    """
    return model in MODEL_REGISTRY


def is_retired(model):
    """Whether `model` has been withdrawn by its provider.

    The new-run gate, as a predicate a library consumer can actually apply.
    Nothing in this package refuses a retired id on a caller's behalf — every
    lookup and every adapter keeps resolving one, because the registry is also
    the provenance record for runs that already happened (see `Model.retired`).
    So the question has to be askable, and this is where it is asked: True means
    the id resolves for provenance but a live call would 404, so it must not
    start a new run.

    Consult it at the point a new run is accepted — validating a config,
    populating a model picker — which is what direktoro's own CLI does in
    `resolve_models`. `known_models(include_retired=False)` is the same fact as
    a list. Raises ValueError for an unknown id, like `model_info`, so an id
    that is not in the table at all fails loudly rather than reporting a
    reassuring False.
    """
    return model_info(model).retired


def model_info(model):
    """Return the `Model` record for `model`.

    Raises ValueError for an unknown id. Unknown models must fail loudly rather
    than be silently mis-routed.

    The error lists the startable ids and the retired ones SEPARATELY. A single
    "known models" list would recommend a withdrawn id as a fix for a typo, and
    the caller would then get a 404 from the provider instead of an error from
    here — so the retired ids are named (they are legitimate answers when the
    question is about a past run) but marked as unable to start a new one.

    Resolves a retired id normally: this function is on the provenance path, and
    the retirement gate is `is_retired` / `known_models(include_retired=False)`,
    applied by the caller where it accepts a new run.
    """
    try:
        return MODEL_REGISTRY[model]
    except KeyError:
        pass
    live = known_models(include_retired=False)
    retired = sorted(set(MODEL_REGISTRY) - set(live))
    message = (f"unknown model {model!r}: no registry entry. Known models "
               f"available for a new run: {', '.join(live)}.")
    if retired:
        message += (f" Also known but RETIRED, resolvable only so past runs "
                    f"stay citable and not startable: {', '.join(retired)}.")
    raise ValueError(message)


def model_supports_images(model):
    """Whether `model` accepts image content blocks on its API.

    True for a vision-capable model; False for a text-only endpoint (one that
    rejects a message whose content carries an image part). Every shipped entry
    is vision-capable today, so this returns True throughout — read it anyway
    rather than assuming, since that is what makes adding a text-only entry
    safe. Raises ValueError for an unknown id, like `model_info`, so a text-only
    degradation is never silently guessed from a missing entry.
    """
    return model_info(model).supports_images


def supports_forced_tool_choice(model) -> bool:
    """Whether `model`'s endpoint honours a FORCED / named tool_choice.

    True for every direct endpoint (Anthropic, OpenAI) and for two routed
    entries, the Qwen flagship and Gemini 3.6 Flash. False for three routed
    entries: the two GLM vision endpoints, whose Z.AI host 404s a forced
    tool_choice through OpenRouter, and MiMo, one of whose two pinned hosts does
    the same for that slug (both confirmed live 2026-07-23; the direct entries'
    values come from the vendors' published tool-use documentation rather than a
    probe — see HOW A FACT GETS INTO THIS TABLE).
    A consumer that forces a named tool consults this to decide whether to arm
    its bounded validate/retry loop: for a False model it sends tool_choice
    "auto" (which `tool_choice_named` already emits) and retries a tool-free
    response with a firm nudge before failing loudly; for a True model it forces
    the named tool and never runs that degrade path. Raises ValueError for an
    unknown id, like `model_info`, so the capability is never guessed from a
    missing entry. Mirrors `model_supports_images` (the field lives on the
    `Model` record too, as `model_info(model).forced_tool_choice`; this is the
    named accessor consumers use).
    """
    return model_info(model).forced_tool_choice


def rejected_sampling_params(model):
    """The sampling controls `model`'s endpoint refuses, as a frozenset.

    Empty for an endpoint that takes what it is sent, and empty equally for one
    nobody has established a refusal for: this reports declared refusals, never
    acceptances. Today: Opus 4.7+ and Sonnet 5 refuse all of
    `temperature`/`top_p`/`top_k`; the GPT-5.x reasoning entries refuse
    `temperature`; google/gemini-3.6-flash refuses `temperature` and `top_p`
    (its Vertex endpoints list neither in supported_parameters — confirmed live
    2026-07-24).

    The decoding resolver (`resolved_decoding_params`) consults the field this
    reads and OMITS a refused param from both the wire request and the
    fingerprint's decoding_params block, so a caller's temperature against a
    model that refuses it is honestly absent rather than silently
    stripped-but-fingerprinted. Raises ValueError for an unknown id, like
    `model_info`, so the capability is never guessed from a missing entry.
    Mirrors `supports_forced_tool_choice` (the field lives on the `Model`
    record too, as `model_info(model).rejects_sampling`; this is the named
    accessor).
    """
    return frozenset(model_info(model).rejects_sampling)


def sampling_band(model, param):
    """The (low, high) range `model`'s reference documents for `param`, or None.

    None means no range has been established for that param on this entry —
    the value is sent and the endpoint's own answer settles it — never that
    every value is accepted. A declared band claims only that a value OUTSIDE
    it can be refused before the call is billed; an in-band value the endpoint
    dislikes still fails at the endpoint, which stays the loud backstop.
    Raises ValueError for an unknown id, like `model_info`. Mirrors
    `rejected_sampling_params` (the field lives on the `Model` record too, as
    `model_info(model).sampling_bands`; this is the named accessor).
    """
    return model_info(model).sampling_bands.get(param)


def thinking_support(model):
    """Return `model`'s `ThinkingSupport`, or None when it declares none.

    The capability record the thinking / effort seam validates against: which
    thinking modes and effort levels the endpoint accepts, and — the field
    consumers most need — `default_on`, whether OMITTING the `thinking`
    parameter still runs thinking. Claude Opus 5 and Sonnet 5 think by default
    and `max_tokens` caps thinking plus response text together, so a cap sized
    on a non-thinking model is under-provisioned once it is pointed at one of
    those; a consumer reads this to size caps deliberately instead of inheriting
    a default it never chose.

    None means the entry has not declared its thinking surface — 10 of the 16
    entries today: every non-Anthropic entry, plus the three retired ids. It
    does NOT mean "no
    thinking": it means direktoro will refuse to emit a thinking shape for that
    model rather than guess one. Raises ValueError for an unknown id, like
    `model_info`. Mirrors `supports_forced_tool_choice` /
    `rejected_sampling_params` (the field lives on the `Model` record too, as
    `model_info(model).thinking`; this is the named accessor).
    """
    return model_info(model).thinking

