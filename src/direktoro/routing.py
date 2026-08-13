"""Routing & provenance for gateway-served (OpenRouter) models.

Anthropic and OpenAI are called directly; everything else routes through
OpenRouter, pinned to a named upstream provider and fingerprinted. This module
owns three things:

  - the `Route` record a routed registry entry carries (which gateway, which
    upstream provider, what fallback / quantization / privacy discipline);
  - the canonical serialisations consumers fold into their own fingerprints
    (`fingerprint_fields` for the Route alone, `call_identity_fields` for the
    whole provider-call identity block — model id + provider + base_url +
    Route + wire-keyed decoding params);
  - the request-side provider object emitted to OpenRouter and the response-side
    pin assertion (`assert_served_upstream`) that RAISES `ProviderRouteMismatch`
    when a routed call did not go where it declared, so nothing partial is
    ledgered.

direktoro owns the wire dialect; consumers should not. So the decoding params
folded into `call_identity_fields` are keyed under the WIRE's parameter name
(`reasoning` for the Responses API, `reasoning_effort` for Chat Completions),
which is exactly what `direktoro.providers.resolved_decoding_params` already
produces — that function stays the single source of truth for wire-keying, and
`call_identity_fields` embeds its output canonically rather than re-deriving it.

This module performs no network and imports no SDK; the registry lookup it needs
for `call_identity_fields` is imported lazily inside that function so importing
`direktoro.routing` never pulls the registry (and so registry entries can hold a
`Route` without an import cycle). The one package import it does make is
`direktoro.errors`, a leaf module holding the base class its refusal shares with
every other provider failure — importable from here precisely because it imports
nothing itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from direktoro.errors import ProviderError


# The only gateway today. Recorded on every Route for provenance and so a
# future second gateway can be told apart in a fingerprint.
GATEWAY_OPENROUTER = "openrouter"


@dataclass(frozen=True)
class Route:
    """How a gateway-served model is pinned.

    Emitted to OpenRouter as its `provider` routing object on every routed
    request (see `provider_object`), and folded into consumer fingerprints as
    canonical config (see `fingerprint_fields`). Every field tightens the route;
    none loosens it.

      - `gateway`: the gateway slug, "openrouter" today.
      - `upstream`: the OpenRouter `provider.order` — the upstream provider
        slug(s) allowed to serve this model, most-preferred first. A one-tuple
        pins a single upstream. A token may also be a fully-qualified endpoint
        tag naming a region or service tier ("google-vertex/global/flex"); that
        narrows what the GATEWAY will route to, but only the provider half of it
        can be checked against the response (see `assert_served_upstream`).
      - `allow_fallbacks`: False means a pinned upstream that cannot serve the
        request FAILS rather than silently rerouting to another provider (whose
        quantization, privacy policy, or served weights we did not vet).
      - `quantizations`: an allow-list of acceptable quantization levels
        (OpenRouter's `provider.quantizations`); empty means no quantization
        filter (used when the pinned upstream reports an unknown quant, which a
        non-empty filter would wrongly exclude). Naming the vetted level refuses
        a coarser served variant.
      - `require_parameters`: True refuses an endpoint that would silently drop
        a parameter we sent (tools, tool_choice, temperature, ...).
      - `data_collection`: "deny" refuses endpoints that may retain/train on the
        request.
      - `zdr`: request-level zero-data-retention enforcement. ORs with the
        account-wide setting — it can only tighten, never loosen — so emitting
        it on every routed request is belt-and-braces against account drift.
    """

    gateway: str
    upstream: tuple[str, ...]
    allow_fallbacks: bool = False
    quantizations: tuple[str, ...] = ()
    require_parameters: bool = True
    data_collection: str = "deny"
    zdr: bool = True


class ProviderRouteMismatch(ProviderError):
    """A routed call did not go where its Route declared.

    Raised when the served-upstream attribution on a routed response does not
    match the pinned `Route.upstream` (or is absent, so the pin cannot be
    verified at all). A routed run whose provenance cannot be confirmed stops
    loudly rather than ledgering an unverifiable receipt — the same discipline
    that refuses a routed response carrying no cost figure instead of recording
    it as zero.

    A `ProviderError` like every other provider failure, so a caller guarding a
    call with `except ProviderError` catches a broken pin too rather than having
    it escape as an unrelated error type.

    THE CALL WAS SERVED AND BILLED: when the adapter raises this, the gateway
    has already charged for the response being refused. `response` carries that
    billed material — the `NormalisedResponse` as it stood when the pin failed,
    its usage intact — so a consumer can record what was spent alongside the
    refusal instead of losing the figures with the response. It is None when
    `assert_served_upstream` is called directly on an attribution a caller
    checked itself, where there is no response to carry (see
    `direktoro.errors.ProviderError`).
    """


# ---------------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------------

def fingerprint_fields(route: Route) -> dict:
    """Canonical, deterministic serialisation of a `Route`.

    Returns a JSON-serialisable dict with a fixed key set and only primitive /
    list values (tuples become lists), so `json.dumps(..., sort_keys=True)` is
    byte-stable across runs and processes. Consumers fold this into their own
    fingerprints; ownership of the composed fingerprint stays with them.
    """
    return {
        "gateway": route.gateway,
        "upstream": list(route.upstream),
        "allow_fallbacks": route.allow_fallbacks,
        "quantizations": list(route.quantizations),
        "require_parameters": route.require_parameters,
        "data_collection": route.data_collection,
        "zdr": route.zdr,
    }


def _canonical(value):
    """Recursively canonicalise a JSON-ish value for byte-stable hashing.

    Dicts are rebuilt with sorted keys, sequences (list/tuple) mapped
    element-wise, tuples flattened to lists. Primitives pass through. This makes
    the mapping's serialisation independent of caller dict ordering.
    """
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def call_identity_fields(model_id, *, route=None, decoding_params=None) -> dict:
    """The canonical provider-call identity block for `model_id`.

    Everything about a call that this layer decides — which model, which
    provider, which base URL, which route, which decoding params under which
    wire's key — belongs in one block, produced here. A caller composes it with
    its own prompt / template / tool hashes and keeps ownership of the composed
    fingerprint; it should NOT also fold provider or base_url in itself, or the
    same fact is counted twice. Returns a JSON-serialisable ordered mapping:

      - `model`: the registry id (id-as-identity).
      - `provider`: the registry provider string — one of exactly three,
        "anthropic", "openai" or "openrouter" (see `direktoro.registry`).
      - `base_url`: the pinned API base URL (None for the Anthropic SDK default).
      - `route`: `fingerprint_fields(...)` of the model's REGISTRY route when
        the model is routed; omitted for a direct model (so a direct model's
        identity block is unchanged by the existence of routing). The registry
        is authoritative: when the `route` arg is None it is taken from
        `model_info(model_id).route`, so a routed model's pin is ALWAYS in its
        identity and a re-pin re-fingerprints even if a caller forgets to
        thread the route through. A caller-supplied route that disagrees with
        the registry raises: the identity block must be self-consistent with
        the model it names.
      - `decoding_params`: the wire-keyed decoding params, canonicalised, when
        supplied. Pass the output of `resolved_decoding_params` — that function
        keys them under the wire's name (`reasoning` for Responses,
        `reasoning_effort` for Chat Completions), so the wire dialect lives in
        one place and a consumer never writes a wire key itself.

    Byte-stable: every value is primitive / list / key-sorted dict, so
    `json.dumps(call_identity_fields(...), sort_keys=True)` is identical across
    runs. Validates `model_id` against the registry (unknown ids fail loudly),
    reusing the same loud-fail as every other lookup.
    """
    from direktoro.registry import model_info

    info = model_info(model_id)
    if route is None:
        route = info.route
    elif info.route is not None and route != info.route:
        raise ValueError(
            f"call_identity_fields: supplied route disagrees with the registry "
            f"route for {model_id!r}; the identity block must describe the "
            f"model as registered (registry: {info.route!r}, got: {route!r}).")
    fields = {
        "model": model_id,
        "provider": info.provider,
        "base_url": info.base_url,
    }
    if route is not None:
        fields["route"] = fingerprint_fields(route)
    if decoding_params is not None:
        fields["decoding_params"] = _canonical(decoding_params)
    return fields


def canonical_json(fields: dict) -> str:
    """Byte-stable JSON for an identity / fingerprint mapping.

    Convenience for consumers and tests: `sort_keys` + compact separators so the
    string is identical across runs regardless of insertion order. Consumers may
    hash this directly or compose it with their own inputs first.
    """
    return json.dumps(fields, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# Request side: the OpenRouter provider object
# ---------------------------------------------------------------------------

def provider_object(route: Route) -> dict:
    """The OpenRouter `provider` routing object for a `Route`.

    Emitted on every routed request (the adapter puts it under `extra_body`
    alongside `usage: {"include": true}`). `order` pins the upstream(s);
    `allow_fallbacks` False makes a pin that cannot serve fail; `quantizations`
    is included only when the Route names a level (an empty filter would wrongly
    exclude an upstream reporting an unknown quant). `require_parameters`,
    `data_collection`, and `zdr` are always emitted — they can only tighten.
    """
    obj = {
        "order": list(route.upstream),
        "allow_fallbacks": route.allow_fallbacks,
        "require_parameters": route.require_parameters,
        "data_collection": route.data_collection,
        "zdr": route.zdr,
    }
    if route.quantizations:
        obj["quantizations"] = list(route.quantizations)
    return obj


# ---------------------------------------------------------------------------
# Response side: the pin assertion
# ---------------------------------------------------------------------------

# Provider slugs whose OpenRouter routing slug does not fold to their served
# display-name attribution, mapped slug-key -> display-key. Most upstreams fold
# to the same key from either vocabulary (slug "z-ai" and display "Z.AI" both ->
# "zai"), so this table is empty for them. Google Vertex is the exception: its
# routing slug is "google-vertex" (-> "googlevertex") but a Vertex-served
# completion attributes the provider as the display name "Google" (-> "google"),
# confirmed live 2026-07-23. (Google AI Studio needs no alias: its slug
# "google-ai-studio" and display "Google AI Studio" both fold to
# "googleaistudio".)
_PROVIDER_ALIASES = {
    "googlevertex": "google",
}


def _normalise_provider_name(name) -> str:
    """Fold a provider slug / display name to a comparison key.

    OpenRouter's `provider.order` takes slugs ("z-ai", "alibaba") but a
    completion's served-provider attribution comes back as a display name
    ("Z.AI", "Alibaba"). Lowercasing and stripping every non-alphanumeric
    character folds both to the same key ("zai", "alibaba") so the pin can be
    compared across the two vocabularies.

    An order token can also be a fully-qualified endpoint tag that appends a
    region/tier/quant path to the provider slug ("google-vertex/global/flex",
    the Vertex flex-tier pin). The fold takes the slug head before the first "/"
    (bare slugs and display names have none and are unaffected), then applies
    `_PROVIDER_ALIASES` for the handful of slugs whose routing key differs from
    their served display key.

    WHAT THAT COSTS, STATED PLAINLY: everything after the head is discarded, so
    a fully-qualified endpoint tag is enforced REQUEST-SIDE ONLY. The served
    attribution names the provider and not the tier — a flex-served and a
    priority-served completion are both attributed "Google" — so
    `assert_served_upstream` cannot tell the pinned tier from any other, and
    nothing else in this package checks it either (`reported_cost` is captured
    and threaded, never compared to anything). A stored identity that says
    "flex" is therefore reporting the token that was SENT, which is a weaker
    claim than the rest of the Route, every other field of which the response
    does confirm.
    """
    if not isinstance(name, str):
        return ""
    head = name.split("/", 1)[0]
    key = re.sub(r"[^a-z0-9]", "", head.lower())
    return _PROVIDER_ALIASES.get(key, key)


def assert_served_upstream(route: Route, served_provider) -> None:
    """Raise `ProviderRouteMismatch` unless `served_provider` matches the pin.

    Runs on every routed response before it is returned. The served provider
    (OpenRouter's top-level `provider` attribution) must fold to one of the
    Route's `upstream` slugs. An absent / empty attribution is itself a mismatch:
    a routed call whose destination cannot be verified stops rather than
    ledgering an unverifiable receipt (the pin's whole purpose).

    This checks the PROVIDER and nothing finer. Where an upstream token is a
    fully-qualified endpoint tag ("google-vertex/global/flex"), the region/tier
    path is dropped by the fold and is not verified here or anywhere else — see
    `_normalise_provider_name`.
    """
    served_key = _normalise_provider_name(served_provider)
    if not served_key:
        raise ProviderRouteMismatch(
            "routed response carried no served-provider attribution "
            f"(got {served_provider!r}); the pin to {list(route.upstream)!r} "
            "cannot be verified, so the call is refused rather than ledgered "
            "as an unverifiable receipt.")
    allowed = {_normalise_provider_name(u) for u in route.upstream}
    if served_key not in allowed:
        raise ProviderRouteMismatch(
            f"routed call was served by {served_provider!r} but the Route pins "
            f"upstream to {list(route.upstream)!r}. allow_fallbacks is "
            f"{route.allow_fallbacks}; a call that did not go where it declared "
            "is refused so nothing partial is ledgered.")
