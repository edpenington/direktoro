"""Hermetic unit tests for direktoro.routing.

No network, no SDK: the Route record, its canonical serialisations, the
OpenRouter provider object, and the pin assertion are all pure. Byte-stability
is asserted directly, since consumers fold these into fingerprints that must not
drift for reasons other than a real routing change.
"""

import json

import pytest

from direktoro.routing import (
    GATEWAY_OPENROUTER,
    ProviderRouteMismatch,
    Route,
    _normalise_provider_name,
    assert_served_upstream,
    call_identity_fields,
    canonical_json,
    fingerprint_fields,
    provider_object,
)


def _route(**over):
    base = dict(gateway=GATEWAY_OPENROUTER, upstream=("z-ai",))
    base.update(over)
    return Route(**base)


class TestRouteDefaults:
    def test_defaults_are_fail_closed_and_privacy_tight(self):
        # Every default tightens the route. A caller that names only a gateway
        # and an upstream still gets no silent reroute, no parameter-dropping
        # endpoint, no data collection, and request-level ZDR.
        r = _route()
        assert r.allow_fallbacks is False
        assert r.require_parameters is True
        assert r.data_collection == "deny"
        assert r.zdr is True
        assert r.quantizations == ()

    def test_is_frozen(self):
        with pytest.raises(Exception):
            _route().upstream = ("other",)  # frozen dataclass


class TestFingerprintFields:
    def test_shape_and_json_types(self):
        r = _route(upstream=("z-ai", "novita"), quantizations=("fp8",))
        fp = fingerprint_fields(r)
        assert fp == {
            "gateway": "openrouter",
            "upstream": ["z-ai", "novita"],       # tuples -> lists for JSON
            "allow_fallbacks": False,
            "quantizations": ["fp8"],
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        }
        # JSON-serialisable and byte-stable.
        json.dumps(fp)

    def test_byte_stable_across_calls(self):
        r = _route(quantizations=("fp8",))
        a = json.dumps(fingerprint_fields(r), sort_keys=True)
        b = json.dumps(fingerprint_fields(r), sort_keys=True)
        assert a == b

    def test_a_field_change_moves_the_fingerprint(self):
        base = canonical_json(fingerprint_fields(_route()))
        # A different upstream is a real routing change: the fingerprint moves.
        moved = canonical_json(fingerprint_fields(_route(upstream=("novita",))))
        assert base != moved
        # zdr flipping likewise moves it (privacy discipline is fingerprinted).
        assert base != canonical_json(fingerprint_fields(_route(zdr=False)))


class TestCallIdentityFields:
    def test_direct_model_has_no_route_block(self):
        fields = call_identity_fields("claude-opus-4-8")
        assert fields["model"] == "claude-opus-4-8"
        assert fields["provider"] == "anthropic"
        assert fields["base_url"] is None
        assert "route" not in fields          # direct: unchanged by routing
        assert "decoding_params" not in fields

    def test_routed_model_folds_the_route(self):
        r = _route(quantizations=("fp8",))
        fields = call_identity_fields("z-ai/glm-4.6v", route=r)
        assert fields["provider"] == "openrouter"
        assert fields["base_url"] == "https://openrouter.ai/api/v1"
        assert fields["route"] == fingerprint_fields(r)

    def test_decoding_params_embedded_and_canonicalised(self):
        # Pass the wire-keyed output of resolved_decoding_params; the block
        # embeds it, key-sorted, so insertion order cannot move the fingerprint.
        # (Route omitted: the registry route is authoritative and supplied
        # automatically — a synthetic route for a registered model raises.)
        a = call_identity_fields(
            "z-ai/glm-4.6v",
            decoding_params={"temperature": 0.0, "max_tokens": 8})
        b = call_identity_fields(
            "z-ai/glm-4.6v",
            decoding_params={"max_tokens": 8, "temperature": 0.0})
        assert canonical_json(a) == canonical_json(b)
        assert a["decoding_params"] == {"max_tokens": 8, "temperature": 0.0}

    def test_byte_stable_across_calls(self):
        one = canonical_json(call_identity_fields(
            "z-ai/glm-4.6v", decoding_params={"max_tokens": 8}))
        two = canonical_json(call_identity_fields(
            "z-ai/glm-4.6v", decoding_params={"max_tokens": 8}))
        assert one == two

    def test_unknown_model_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown model"):
            call_identity_fields("not-a-model")

    def test_the_identity_block_key_set_is_pinned(self):
        """The block's SHAPE is a compatibility surface, and this is where it is
        held. A consumer hashes `canonical_json` of this mapping into provenance
        it then publishes; the run is over by the time anyone notices, so a key
        added, removed or renamed here moves every number already published and
        cannot be recomputed. The README calls that a breaking change — this
        makes the promise fail loudly instead of relying on it being remembered.

        Both forms are pinned, because they are two different shapes: a direct
        model omits `route` entirely (so routing's existence never touched a
        direct model's identity), and `decoding_params` appears only when the
        caller supplies one.
        """
        assert list(call_identity_fields("claude-opus-4-8")) == [
            "model", "provider", "base_url"]
        assert list(call_identity_fields("gpt-5.6-sol",
                                         decoding_params={"max_tokens": 8})) == [
            "model", "provider", "base_url", "decoding_params"]
        assert list(call_identity_fields("z-ai/glm-4.6v")) == [
            "model", "provider", "base_url", "route"]
        assert list(call_identity_fields(
            "z-ai/glm-4.6v", decoding_params={"max_tokens": 8})) == [
            "model", "provider", "base_url", "route", "decoding_params"]
        # And the nested route block, whose key set is the other half of the
        # same promise (its values are pinned by TestFingerprintFields).
        assert list(call_identity_fields("z-ai/glm-4.6v")["route"]) == [
            "gateway", "upstream", "allow_fallbacks", "quantizations",
            "require_parameters", "data_collection", "zdr"]

    def test_gemini_36_omission_flows_into_identity(self):
        # The sampling-params seam end-to-end: a consumer resolves decoding params
        # for google/gemini-3.6-flash WITH a temperature, but the model takes no
        # sampling controls, so resolved_decoding_params drops it — and because the
        # identity block folds exactly that resolved dict, temperature is absent
        # from decoding_params (honest omission, never stripped-but-fingerprinted).
        from direktoro.providers import resolved_decoding_params

        resolved = resolved_decoding_params(
            "google/gemini-3.6-flash", sampling={"temperature": 0.0}, max_tokens=8)
        assert "temperature" not in resolved
        fields = call_identity_fields(
            "google/gemini-3.6-flash", decoding_params=resolved)
        assert "temperature" not in fields["decoding_params"]
        assert fields["decoding_params"] == {"max_tokens": 8}
        # And the fingerprint is identical whether or not the config carried a
        # temperature: the config temperature moves nothing for this model.
        no_temp = call_identity_fields(
            "google/gemini-3.6-flash",
            decoding_params=resolved_decoding_params(
                "google/gemini-3.6-flash", max_tokens=8))
        assert canonical_json(fields) == canonical_json(no_temp)


class TestProviderObject:
    def test_full_shape_with_quant(self):
        obj = provider_object(_route(upstream=("z-ai",),
                                     quantizations=("fp8",)))
        assert obj == {
            "order": ["z-ai"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "quantizations": ["fp8"],
        }

    def test_quantizations_omitted_when_empty(self):
        # An empty filter would wrongly exclude an upstream reporting an unknown
        # quant (e.g. Alibaba), so the key is omitted entirely.
        obj = provider_object(_route(upstream=("alibaba",)))
        assert "quantizations" not in obj
        assert obj["order"] == ["alibaba"]


class TestNormaliseProviderName:
    @pytest.mark.parametrize("slug, name", [
        ("z-ai", "Z.AI"),
        ("alibaba", "Alibaba"),
        ("deepinfra", "DeepInfra"),
        ("novita", "Novita"),
    ])
    def test_slug_and_display_name_fold_together(self, slug, name):
        assert _normalise_provider_name(slug) == _normalise_provider_name(name)


class TestVertexTierFold:
    """The Vertex flex-tier pin: the order token is a fully-qualified endpoint
    tag `google-vertex/global/flex`, but a Vertex-served completion attributes
    the provider as the display name `Google`. The fold must reconcile the two
    so the pin assertion passes, and must not disturb the bare-slug entries."""

    def test_vertex_tag_folds_to_served_google(self):
        # The pinned order token and the served display name fold to one key.
        assert (_normalise_provider_name("google-vertex/global/flex")
                == _normalise_provider_name("Google"))
        # Every tier variant folds to that same key: the served attribution
        # names the PROVIDER and never the tier, so there is nothing finer for
        # the fold to compare against.
        assert (_normalise_provider_name("google-vertex/global/priority")
                == _normalise_provider_name("Google"))

    def test_flex_pin_assertion_passes_on_google_attribution(self):
        route = _route(upstream=("google-vertex/global/flex",),
                       quantizations=())
        assert_served_upstream(route, "Google")   # no raise

    def test_ai_studio_does_not_fold_to_vertex(self):
        # Google AI Studio (the non-ZDR path) stays distinct from Vertex "Google".
        assert (_normalise_provider_name("google-ai-studio")
                == _normalise_provider_name("Google AI Studio"))
        assert (_normalise_provider_name("google-ai-studio")
                != _normalise_provider_name("Google"))

    def test_bare_slugs_unaffected_by_the_head_split(self):
        # A bare slug has no "/" and no alias entry, so neither the
        # head-before-"/" split nor the alias table changes how it folds: it
        # still meets its display name on the same key.
        for slug, display in (("z-ai", "Z.AI"), ("venice", "Venice"),
                              ("parasail", "Parasail")):
            assert (_normalise_provider_name(slug)
                    == _normalise_provider_name(display))

    def test_the_tier_half_of_the_tag_is_enforced_request_side_only(self):
        # The limitation, asserted rather than described, because a Route field
        # that reads like the others but is checked less than they are is worth
        # being unable to forget. The order token narrows what the GATEWAY will
        # route to; the response cannot confirm it, because the attribution is
        # a provider display name with no tier in it. So a pin on flex accepts
        # a `Google` attribution that could equally have come from the standard
        # or priority tier, and nothing else in this package compares the tier
        # to anything either — `reported_cost` is captured and threaded, never
        # checked. An identity block saying "flex" is reporting the token that
        # was SENT.
        flex = _route(upstream=("google-vertex/global/flex",),
                      quantizations=())
        priority = _route(upstream=("google-vertex/global/priority",),
                          quantizations=())
        # One and the same attribution satisfies either pin: the assertion
        # cannot tell the tiers apart.
        assert_served_upstream(flex, "Google")
        assert_served_upstream(priority, "Google")
        # And it is genuinely comparing only the provider head.
        assert (_normalise_provider_name("google-vertex/global/flex")
                == _normalise_provider_name("google-vertex/global/priority"))
        # What it DOES still catch is the provider being wrong, which is the
        # part of the pin the response can speak to.
        with pytest.raises(ProviderRouteMismatch):
            assert_served_upstream(flex, "Google AI Studio")

    def test_the_unverified_tier_is_still_carried_verbatim_into_identity(self):
        # The other half of the same fact: the tag is recorded exactly as
        # pinned, so a consumer's fingerprint distinguishes a flex run from a
        # priority one even though the response never confirmed either. That is
        # a record of what was REQUESTED, which is what makes it worth saying
        # out loud in the docs rather than leaving it to look like an assertion.
        flex = fingerprint_fields(
            _route(upstream=("google-vertex/global/flex",)))
        priority = fingerprint_fields(
            _route(upstream=("google-vertex/global/priority",)))
        assert flex["upstream"] == ["google-vertex/global/flex"]
        assert canonical_json(flex) != canonical_json(priority)


class TestAssertServedUpstream:
    def test_match_passes(self):
        # Served display name folds to the pinned slug: no raise.
        assert_served_upstream(_route(upstream=("z-ai",)), "Z.AI")

    def test_mismatch_raises(self):
        with pytest.raises(ProviderRouteMismatch, match="Novita"):
            assert_served_upstream(_route(upstream=("z-ai",)), "Novita")

    def test_absent_attribution_raises(self):
        for absent in (None, "", 123):
            with pytest.raises(ProviderRouteMismatch, match="attribution"):
                assert_served_upstream(_route(upstream=("z-ai",)), absent)

    def test_any_of_multiple_upstreams_matches(self):
        route = _route(upstream=("z-ai", "novita"))
        assert_served_upstream(route, "Novita")   # second choice still valid


class TestIdentityRouteAuthority:
    """The registry, not the caller, is authoritative for a routed model's pin
    in the identity block: with no route argument the pin comes from the
    registry, so a re-pin re-fingerprints even when a caller forgets to thread
    the route through, and a disagreeing caller route raises rather than
    producing a block that describes a model differently from how it is
    registered."""

    def test_routed_model_gets_registry_route_by_default(self):
        fields = call_identity_fields("z-ai/glm-4.6v")
        assert fields["route"]["upstream"] == ["z-ai"]
        assert fields["route"]["zdr"] is True

    def test_default_equals_explicit_registry_route(self):
        from direktoro.registry import model_info
        info = model_info("z-ai/glm-4.6v")
        assert (call_identity_fields("z-ai/glm-4.6v")
                == call_identity_fields("z-ai/glm-4.6v", route=info.route))

    def test_repin_moves_the_block_even_without_threading(self, monkeypatch):
        import dataclasses
        from direktoro import registry
        entry = registry.MODEL_REGISTRY["z-ai/glm-4.6v"]
        repinned = dataclasses.replace(
            entry, route=Route(gateway=GATEWAY_OPENROUTER,
                               upstream=("novita",)))
        monkeypatch.setitem(registry.MODEL_REGISTRY, "z-ai/glm-4.6v", repinned)
        assert (call_identity_fields("z-ai/glm-4.6v")["route"]["upstream"]
                == ["novita"])

    def test_disagreeing_route_raises(self):
        with pytest.raises(ValueError, match="disagrees with the registry"):
            call_identity_fields(
                "z-ai/glm-4.6v",
                route=Route(gateway=GATEWAY_OPENROUTER, upstream=("novita",)))

    def test_direct_model_still_has_no_route_key(self):
        assert "route" not in call_identity_fields("claude-opus-4-8")


class TestRoutedWireInvariant:
    """A routed entry on a non-Chat-Completions wire would silently bypass the
    provider object, the pin assertion and the reported cost — every guarantee
    routing exists for — so construction must fail loudly at import."""

    def test_routed_responses_wire_rejected(self):
        from direktoro.registry import Model, WIRE_RESPONSES
        with pytest.raises(ValueError, match="WIRE_CHAT_COMPLETIONS"):
            Model("openrouter", "https://openrouter.ai/api/v1",
                  "OPENROUTER_API_KEY", wire_api=WIRE_RESPONSES,
                  route=_route())
