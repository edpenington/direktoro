"""The thinking / reasoning-effort seam.

Three properties, in the order they matter:

  1. **Optional.** A call that omits `thinking` sends no thinking parameters at
     all: `resolved_decoding_params`, the Anthropic wire request and the
     call-identity block carry exactly the keys they would if this seam did not
     exist, and each model's own default stays in force. A caller adopts the
     seam per call site, or never.
  2. **The registry refuses a shape the model rejects, BEFORE the call.** A 400
     on a paid call is the failure this exists to prevent, so every known-bad
     pairing (a `budget_tokens` on a 4.7+ family, an effort level a model does
     not have, disabled thinking above Claude Opus 5's effort ceiling, a model
     whose thinking surface is undeclared) raises `ThinkingUnsupported` from
     the resolver with no client, no key and no network involved.
  3. **Effort is part of call identity.** A caller fingerprints its call
     configuration so a run is reproducible; two runs differing only in
     reasoning effort must fingerprint differently. That holds without any
     argument added to `call_identity_fields`, because effort reaches it
     through `resolved_decoding_params`, the single source of truth for both
     the wire request and the recorded decoding params.
"""

import pytest

from direktoro import (
    EFFORT_LEVELS, THINKING_ADAPTIVE, THINKING_BUDGET, THINKING_DISABLED,
    Thinking, ThinkingSupport, ThinkingUnsupported, call_identity_fields,
    canonical_json, model_info, resolved_decoding_params, thinking_support)
from direktoro.providers import AnthropicAdapter


OPUS_5 = "claude-opus-5"
OPUS_4_8 = "claude-opus-4-8"
SONNET_5 = "claude-sonnet-5"
SONNET_4_6 = "claude-sonnet-4-6"
HAIKU_4_5 = "claude-haiku-4-5-20251001"
GLM = "z-ai/glm-5v-turbo"
GPT = "gpt-5.6-sol"


# ---------------------------------------------------------------------------
# 1. Additive: omitting the spec changes nothing
# ---------------------------------------------------------------------------

class TestOmittingThinkingChangesNothing:
    """The seam is opt-in. A call site that passes only `temperature` and
    `max_tokens` must emit exactly those and nothing else."""

    @pytest.mark.parametrize("model", [
        OPUS_5, OPUS_4_8, SONNET_5, SONNET_4_6, HAIKU_4_5, GPT, GLM])
    def test_default_emits_no_thinking_keys(self, model):
        dec = resolved_decoding_params(model, sampling={"temperature": 0.0}, max_tokens=4096)
        assert "thinking" not in dec
        assert "output_config" not in dec

    def test_anthropic_default_is_exactly_the_two_decoding_keys(self):
        # Opus 5 rejects temperature (_NO_TEMP), so the cap is the whole of it.
        assert resolved_decoding_params(
            OPUS_5, sampling={"temperature": 0.0}, max_tokens=4096) == {"max_tokens": 4096}
        # A model that takes temperature still gets exactly the two keys.
        assert resolved_decoding_params(
            SONNET_4_6, sampling={"temperature": 0.3}, max_tokens=4096) == {
                "max_tokens": 4096, "temperature": 0.3}

    def test_explicit_none_matches_the_omitted_call(self):
        assert (resolved_decoding_params(OPUS_5,
                                         max_tokens=4096, thinking=None)
                == resolved_decoding_params(OPUS_5,
                                            max_tokens=4096))

    def test_adapter_wire_request_is_unchanged_when_omitted(self):
        client = _StubAnthropicClient()
        AnthropicAdapter(client).create_message(
            model=OPUS_5, system=[], messages=[], max_tokens=4096)
        assert "thinking" not in client.wire
        assert "output_config" not in client.wire


# ---------------------------------------------------------------------------
# 2a. What the seam emits when a shape IS accepted
# ---------------------------------------------------------------------------

class TestEmittedShapes:
    """The current Claude API shapes. `{"type": "adaptive"}` on the 4.7+
    families; `{"type": "enabled", "budget_tokens": N}` only on pre-4.6."""

    def test_adaptive_is_the_current_shape(self):
        dec = resolved_decoding_params(
            OPUS_5, max_tokens=4096,
            thinking=Thinking(mode=THINKING_ADAPTIVE))
        assert dec["thinking"] == {"type": "adaptive"}
        # Emphatically NOT the pre-4.6 shape, which 400s on this family.
        assert "budget_tokens" not in dec["thinking"]

    def test_effort_rides_output_config(self):
        dec = resolved_decoding_params(
            OPUS_5, max_tokens=4096,
            thinking=Thinking(mode=THINKING_ADAPTIVE, effort="xhigh"))
        assert dec["output_config"] == {"effort": "xhigh"}
        assert dec["thinking"] == {"type": "adaptive"}

    def test_effort_alone_needs_no_mode(self):
        dec = resolved_decoding_params(
            SONNET_5, max_tokens=4096,
            thinking=Thinking(effort="low"))
        assert dec["output_config"] == {"effort": "low"}
        assert "thinking" not in dec

    def test_display_opts_into_summarized_thinking(self):
        dec = resolved_decoding_params(
            OPUS_5, max_tokens=4096,
            thinking=Thinking(mode=THINKING_ADAPTIVE, display="summarized"))
        assert dec["thinking"] == {"type": "adaptive",
                                   "display": "summarized"}

    def test_display_rides_the_budget_block_too(self):
        # Anthropic documents `display` in BOTH on-modes: "set it alongside
        # `type: "adaptive"` or `type: "enabled"`". Requiring adaptive would
        # refuse a shape the pre-4.6 endpoints accept.
        dec = resolved_decoding_params(
            HAIKU_4_5, max_tokens=8192,
            thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=2048,
                              display="summarized"))
        assert dec["thinking"] == {"type": "enabled", "budget_tokens": 2048,
                                   "display": "summarized"}

    def test_disabled_shape(self):
        dec = resolved_decoding_params(
            OPUS_4_8, max_tokens=4096,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert dec["thinking"] == {"type": "disabled"}

    def test_budget_shape_on_a_pre_4_6_model(self):
        # temperature=None deliberately: a temperature alongside active thinking
        # is refused on this generation (see TestSamplingParamsAndThinking), and
        # asserting only the thinking block would let the 400-producing pair
        # through unnoticed.
        dec = resolved_decoding_params(
            HAIKU_4_5, max_tokens=8192,
            thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=2048))
        assert dec["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "temperature" not in dec

    def test_anthropic_adapter_puts_them_on_the_wire(self):
        client = _StubAnthropicClient()
        AnthropicAdapter(client).create_message(
            model=OPUS_5, system=[], messages=[], max_tokens=4096,
            thinking=Thinking(mode=THINKING_ADAPTIVE, effort="high"))
        assert client.wire["thinking"] == {"type": "adaptive"}
        assert client.wire["output_config"] == {"effort": "high"}
        # `thinking` and `output_config` are Anthropic top-level request
        # parameters, so they merge into the wire request, not into `messages`.
        assert client.wire["model"] == OPUS_5

    def test_adapter_records_them_in_decoding_params_provenance(self):
        client = _StubAnthropicClient()
        response = AnthropicAdapter(client).create_message(
            model=OPUS_5, system=[], messages=[], max_tokens=4096,
            thinking=Thinking(mode=THINKING_ADAPTIVE, effort="high"))
        assert response.decoding_params["thinking"] == {"type": "adaptive"}
        assert response.decoding_params["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# 2b. What the seam REFUSES, and why
# ---------------------------------------------------------------------------

class TestRefusesShapesTheModelWouldReject:
    """Each of these is a 400 the seam declines to pay for. The registry knows
    the family, so the refusal happens in the resolver — no client, no key, no
    network, no spend."""

    @pytest.mark.parametrize("model", [OPUS_5, OPUS_4_8, SONNET_5])
    def test_budget_tokens_refused_on_the_4_7_plus_families(self, model):
        with pytest.raises(ThinkingUnsupported, match="budget_tokens"):
            resolved_decoding_params(
                model, max_tokens=8192,
                thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=2048))

    def test_disabled_thinking_refused_above_opus_5_effort_ceiling(self):
        for effort in ("xhigh", "max"):
            with pytest.raises(ThinkingUnsupported, match="disabled"):
                resolved_decoding_params(
                    OPUS_5, max_tokens=8192,
                    thinking=Thinking(mode=THINKING_DISABLED, effort=effort))

    def test_disabled_thinking_allowed_at_or_below_high_on_opus_5(self):
        for effort in ("low", "medium", "high"):
            dec = resolved_decoding_params(
                OPUS_5, max_tokens=8192,
                thinking=Thinking(mode=THINKING_DISABLED, effort=effort))
            assert dec["thinking"] == {"type": "disabled"}

    def test_disabled_without_an_effort_uses_the_default_effort(self):
        # Anthropic's default effort is `high`, which is exactly the ceiling,
        # so a bare disable is accepted rather than refused on a guess.
        dec = resolved_decoding_params(
            OPUS_5, max_tokens=8192,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert dec["thinking"] == {"type": "disabled"}
        assert thinking_support(OPUS_5).default_effort == "high"

    def test_xhigh_refused_on_the_4_6_generation(self):
        # `xhigh` arrived with Opus 4.7; Sonnet 4.6's ladder stops at max.
        with pytest.raises(ThinkingUnsupported, match="xhigh"):
            resolved_decoding_params(
                SONNET_4_6, max_tokens=8192,
                thinking=Thinking(effort="xhigh"))

    def test_effort_refused_on_a_model_with_no_effort_parameter(self):
        with pytest.raises(ThinkingUnsupported,
                           match="no effort parameter at all"):
            resolved_decoding_params(
                HAIKU_4_5, max_tokens=8192,
                thinking=Thinking(effort="high"))

    def test_adaptive_refused_on_a_pre_4_6_model(self):
        with pytest.raises(ThinkingUnsupported, match="adaptive"):
            resolved_decoding_params(
                HAIKU_4_5, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))

    def test_budget_below_the_endpoint_minimum_refused(self):
        with pytest.raises(ThinkingUnsupported, match="minimum"):
            resolved_decoding_params(
                HAIKU_4_5, max_tokens=8192,
                thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=512))

    def test_budget_not_below_max_tokens_refused(self):
        # The cap covers thinking AND response text, so a budget at or above it
        # leaves no room for an answer and the API rejects it.
        with pytest.raises(ThinkingUnsupported, match="strictly less"):
            resolved_decoding_params(
                HAIKU_4_5, max_tokens=2048,
                thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=2048))

    def test_budget_with_no_max_tokens_refused(self):
        # Skipping the `budget < max_tokens` guard when there is no cap would
        # emit {"max_tokens": None, "thinking": {"budget_tokens": 999999}}. The
        # API rejects that on max_tokens, so the caller gets an error naming the
        # wrong parameter; refusing here names the one that is missing.
        with pytest.raises(ThinkingUnsupported, match="needs a max_tokens"):
            resolved_decoding_params(
                HAIKU_4_5, max_tokens=None,
                thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=999999))

    def test_display_refused_on_a_model_that_declares_none(self, monkeypatch):
        # Every DECLARED entry accepts both display settings today, so the
        # per-model gate is exercised against a synthetic entry — the shape a
        # future model without the field would take.
        from direktoro.registry import MODEL_REGISTRY, Model, ThinkingSupport
        monkeypatch.setitem(
            MODEL_REGISTRY, "synthetic-no-display",
            Model("anthropic", None, "ANTHROPIC_API_KEY",
                  forced_tool_choice=True,
                  thinking=ThinkingSupport(
                      modes=(THINKING_ADAPTIVE,), efforts=("high",),
                      default_effort="high", displays=())))
        # The mode itself is fine …
        assert resolved_decoding_params(
            "synthetic-no-display", max_tokens=4096,
            thinking=Thinking(mode=THINKING_ADAPTIVE))["thinking"] == \
            {"type": "adaptive"}
        # … it is the display that is refused, before any spend.
        with pytest.raises(ThinkingUnsupported, match="thinking.display"):
            resolved_decoding_params(
                "synthetic-no-display", max_tokens=4096,
                thinking=Thinking(mode=THINKING_ADAPTIVE,
                                  display="summarized"))

    def test_display_refused_when_the_model_declares_only_the_other_setting(
            self, monkeypatch):
        from direktoro.registry import MODEL_REGISTRY, Model, ThinkingSupport
        monkeypatch.setitem(
            MODEL_REGISTRY, "synthetic-omitted-only",
            Model("anthropic", None, "ANTHROPIC_API_KEY",
                  forced_tool_choice=True,
                  thinking=ThinkingSupport(
                      modes=(THINKING_ADAPTIVE,), efforts=("high",),
                      default_effort="high", displays=("omitted",))))
        with pytest.raises(ThinkingUnsupported, match="it accepts \\['omitted'\\]"):
            resolved_decoding_params(
                "synthetic-omitted-only", max_tokens=4096,
                thinking=Thinking(mode=THINKING_ADAPTIVE,
                                  display="summarized"))

    @pytest.mark.parametrize("model", [GPT, GLM])
    def test_non_anthropic_models_refuse_the_spec(self, model):
        # The seam emits Anthropic wire keys. Reasoning effort for the OpenAI
        # families rides the registry's `reasoning_effort` quirk, and no routed
        # entry has had its accepted levels live-verified, so refusing is the
        # honest answer rather than translating on a guess.
        with pytest.raises(ThinkingUnsupported):
            resolved_decoding_params(
                model, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))

    def test_undeclared_thinking_support_refuses_rather_than_guesses(self):
        # The retired ids deliberately declare nothing: a caller applying the
        # retirement gate (`is_retired`) refuses them for new runs anyway, and
        # nobody can re-verify a withdrawn endpoint to declare a surface for it.
        assert thinking_support("claude-opus-4-20250514") is None
        with pytest.raises(ThinkingUnsupported, match="declares no thinking"):
            resolved_decoding_params(
                "claude-opus-4-20250514", max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))

    def test_refusal_is_a_value_error_subclass(self):
        # Consumers that already guard registry lookups with `except
        # ValueError` catch the refusal without a code change.
        assert issubclass(ThinkingUnsupported, ValueError)

    def test_disable_on_a_model_that_never_thinks_is_satisfied_by_omission(
            self):
        # Haiku 4.5 does not think unless a budget is given, so "disabled" is
        # already true of the omitted-parameter call. The request is honoured
        # by emitting nothing: same wire, same behaviour, same identity as a
        # `thinking=None` call — no shape is guessed and nothing is refused.
        plain = resolved_decoding_params(
            HAIKU_4_5, sampling={"temperature": 0.0}, max_tokens=4096)
        disabled = resolved_decoding_params(
            HAIKU_4_5, sampling={"temperature": 0.0}, max_tokens=4096,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert disabled == plain
        assert thinking_support(HAIKU_4_5).default_on is False


class TestSamplingParamsAndThinking:
    """The one constraint `ThinkingSupport` cannot express, because it is not a
    property of the model alone.

    Anthropic's thinking documentation: on the 4.7-generation-and-later models a
    non-default temperature/top_p/top_k is a 400 on every request (the
    `no_temperature` quirk handles those). "On older models, the restriction
    applies only while thinking is on: `temperature` and `top_k` are
    incompatible with thinking". Sonnet 4.6 and Haiku 4.5 are exactly the two
    live entries without the quirk, so they accept a temperature — until the
    same request also asks them to think, at which point resolving the two
    independently would put a 400-producing pair on the wire."""

    THINKING_ON = [
        (SONNET_4_6, Thinking(mode=THINKING_ADAPTIVE)),
        (SONNET_4_6, Thinking(mode=THINKING_ADAPTIVE, effort="high")),
        (SONNET_4_6, Thinking(mode=THINKING_BUDGET, budget_tokens=2048)),
        (HAIKU_4_5, Thinking(mode=THINKING_BUDGET, budget_tokens=2048)),
    ]

    @pytest.mark.parametrize("model,spec", THINKING_ON)
    def test_temperature_with_active_thinking_is_refused(self, model, spec):
        with pytest.raises(ThinkingUnsupported, match="also turns thinking on"):
            resolved_decoding_params(model, sampling={"temperature": 0.3}, max_tokens=8192,
                                     thinking=spec)

    @pytest.mark.parametrize("model,spec", THINKING_ON)
    def test_the_same_call_without_a_temperature_is_fine(self, model, spec):
        dec = resolved_decoding_params(model,
                                       max_tokens=8192, thinking=spec)
        assert "thinking" in dec
        assert "temperature" not in dec

    def test_temperature_survives_when_thinking_is_off(self):
        # Disabled thinking and an effort-only spec both leave sampling alone:
        # the restriction is on ACTIVE thinking, not on the presence of a spec.
        for spec in (None, Thinking(mode=THINKING_DISABLED),
                     Thinking(effort="high")):
            dec = resolved_decoding_params(SONNET_4_6, sampling={"temperature": 0.3},
                                           max_tokens=8192, thinking=spec)
            assert dec["temperature"] == 0.3

    def test_the_4_7_family_is_unaffected(self):
        # They reject temperature outright, so the resolver already omits it and
        # there is no pair to refuse — this check must not turn their calls into
        # errors.
        for model in (OPUS_5, OPUS_4_8, SONNET_5):
            dec = resolved_decoding_params(
                model, sampling={"temperature": 0.3}, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))
            assert "temperature" not in dec
            assert dec["thinking"] == {"type": "adaptive"}

    def test_the_refusal_names_both_ways_out(self):
        with pytest.raises(ThinkingUnsupported) as excinfo:
            resolved_decoding_params(
                SONNET_4_6, sampling={"temperature": 0.3}, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))
        message = str(excinfo.value)
        assert "`temperature`" in message
        assert "drop the sampling params" in message
        assert "disabled" in message

    def test_a_temperature_is_never_silently_dropped(self):
        # The alternative fix — omit the temperature and send the thinking —
        # would change the sampling distribution of a scientific run behind the
        # caller's back. Refusing is the deliberate choice; this test fails if
        # someone "helpfully" makes it silent.
        try:
            dec = resolved_decoding_params(
                SONNET_4_6, sampling={"temperature": 0.3}, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))
        except ThinkingUnsupported:
            return
        pytest.fail(
            f"expected a refusal, got a silently altered request: {dec!r}")


# ---------------------------------------------------------------------------
# 3. Call identity
# ---------------------------------------------------------------------------

class TestEffortIsPartOfCallIdentity:
    """`call_identity_fields` is the provider-call identity block a consumer
    folds into its own fingerprint. Effort and thinking reach it through
    `decoding_params` — the same channel `temperature` already uses — so the
    block needs no argument of its own for them, and a call naming no spec
    produces exactly the fields it would without the seam."""

    def _identity(self, model, thinking=None):
        return canonical_json(call_identity_fields(
            model,
            decoding_params=resolved_decoding_params(
                model, max_tokens=8192,
                thinking=thinking)))

    def test_two_efforts_fingerprint_differently(self):
        high = self._identity(
            OPUS_5, Thinking(mode=THINKING_ADAPTIVE, effort="high"))
        max_ = self._identity(
            OPUS_5, Thinking(mode=THINKING_ADAPTIVE, effort="max"))
        assert high != max_

    def test_two_thinking_modes_fingerprint_differently(self):
        adaptive = self._identity(OPUS_5, Thinking(mode=THINKING_ADAPTIVE))
        disabled = self._identity(OPUS_5, Thinking(mode=THINKING_DISABLED))
        assert adaptive != disabled

    def test_asking_for_thinking_differs_from_not_asking(self):
        assert self._identity(OPUS_5) != self._identity(
            OPUS_5, Thinking(mode=THINKING_ADAPTIVE))

    def test_display_moves_the_fingerprint_deliberately(self):
        # `display` changes what comes back, not what the model does or what it
        # bills — so folding it into identity means turning it on to READ a
        # run's reasoning invalidates every fingerprint-keyed cached result.
        # That is the intended trade (this layer fingerprints exactly what is
        # sent, and a request carrying `display` is a different request), but it
        # makes a debug flag a spend decision, which the `Thinking` docstring
        # says out loud. Asserted so the choice stays visible rather than
        # incidental.
        plain = self._identity(OPUS_5, Thinking(mode=THINKING_ADAPTIVE))
        shown = self._identity(
            OPUS_5, Thinking(mode=THINKING_ADAPTIVE, display="summarized"))
        assert plain != shown

    def test_identity_is_byte_stable_for_a_fixed_spec(self):
        spec = Thinking(mode=THINKING_ADAPTIVE, effort="high",
                        display="summarized")
        assert self._identity(OPUS_5, spec) == self._identity(OPUS_5, spec)

    def test_identity_unchanged_for_every_model_when_no_spec_is_given(self):
        # The guarantee that lets a caller adopt the seam at one call site
        # without moving the fingerprints of the others: a spec-free call's
        # block carries only the decoding keys it would carry anyway.
        block = call_identity_fields(
            OPUS_5, decoding_params=resolved_decoding_params(
                OPUS_5, max_tokens=8192))
        assert block["decoding_params"] == {"max_tokens": 8192}

    def test_model_default_thinking_is_not_folded_into_identity(self):
        # Opus 5 thinks when the parameter is omitted, but that is a property of
        # the model id, which identity already carries. Folding it in would
        # count the same fact twice.
        assert thinking_support(OPUS_5).default_on is True
        block = call_identity_fields(
            OPUS_5, decoding_params=resolved_decoding_params(
                OPUS_5, max_tokens=8192))
        assert "thinking" not in block["decoding_params"]


# ---------------------------------------------------------------------------
# The request spec's own vocabulary
# ---------------------------------------------------------------------------

class TestThinkingSpecValidation:
    """Caller mistakes (as opposed to model incapability) fail at construction,
    so a typo never reaches a model lookup."""

    def test_empty_spec_refused(self):
        with pytest.raises(ValueError, match="neither mode nor effort"):
            Thinking()

    def test_unknown_mode_refused(self):
        with pytest.raises(ValueError, match="not a known mode"):
            Thinking(mode="extended")

    def test_unknown_effort_refused(self):
        with pytest.raises(ValueError, match="not a known level"):
            Thinking(effort="ultra")

    def test_budget_tokens_needs_budget_mode(self):
        with pytest.raises(ValueError, match="only meaningful"):
            Thinking(mode=THINKING_ADAPTIVE, budget_tokens=2048)

    def test_display_is_invalid_with_disabled_thinking(self):
        # The documented restriction: `display` is invalid with
        # `thinking.type: "disabled"` — there is nothing to display. It is NOT
        # restricted to adaptive (see test_display_rides_the_budget_block_too).
        with pytest.raises(ValueError, match="invalid with mode='disabled'"):
            Thinking(mode=THINKING_DISABLED, display="summarized")

    def test_display_needs_a_mode_to_ride_on(self):
        # An effort-only spec emits no thinking block, so a display on it would
        # be silently dropped rather than sent.
        with pytest.raises(ValueError, match="needs a thinking mode"):
            Thinking(effort="high", display="summarized")

    def test_unknown_display_refused(self):
        with pytest.raises(ValueError, match="not a known"):
            Thinking(mode=THINKING_ADAPTIVE, display="verbose")

    def test_spec_is_frozen_and_hashable(self):
        spec = Thinking(mode=THINKING_ADAPTIVE, effort="high")
        assert hash(spec) == hash(Thinking(mode=THINKING_ADAPTIVE,
                                           effort="high"))
        with pytest.raises(Exception):
            spec.effort = "low"


class TestThinkingSupportValidation:
    """A malformed capability record must fail at import, not mis-permit a
    shape at request time."""

    def test_unknown_mode_refused(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ThinkingSupport(modes=("extended",))

    def test_unknown_effort_refused(self):
        with pytest.raises(ValueError, match="unknown effort"):
            ThinkingSupport(efforts=("ultra",), default_effort="ultra")

    def test_efforts_require_a_default_effort(self):
        with pytest.raises(ValueError, match="no default_effort"):
            ThinkingSupport(efforts=("low", "high"))

    def test_default_effort_must_be_declared(self):
        with pytest.raises(ValueError, match="must be one of the declared"):
            ThinkingSupport(efforts=("low",), default_effort="high")

    def test_disabled_ceiling_needs_the_disabled_mode(self):
        with pytest.raises(ValueError, match="meaningless without"):
            ThinkingSupport(modes=(THINKING_ADAPTIVE,), efforts=("high",),
                            default_effort="high", disabled_max_effort="high")

    def test_disabled_ceiling_must_be_a_declared_effort(self):
        with pytest.raises(ValueError, match="must be one of the declared"):
            ThinkingSupport(modes=(THINKING_DISABLED,), efforts=("low",),
                            default_effort="low", disabled_max_effort="high")

    def test_unknown_display_refused(self):
        with pytest.raises(ValueError, match="unknown setting"):
            ThinkingSupport(modes=(THINKING_ADAPTIVE,), displays=("verbose",))

    def test_displays_need_an_on_mode(self):
        # A model that can only DISABLE thinking has nothing to display.
        with pytest.raises(ValueError, match="meaningless without an on-mode"):
            ThinkingSupport(modes=(THINKING_DISABLED,),
                            displays=("summarized",))

    def test_displays_ride_either_on_mode(self):
        # Both are legal declarations: `display` works alongside
        # `type: "adaptive"` and `type: "enabled"` alike.
        assert ThinkingSupport(modes=(THINKING_ADAPTIVE,),
                               displays=("summarized",)).displays
        assert ThinkingSupport(modes=(THINKING_BUDGET,),
                               displays=("summarized",)).displays


# ---------------------------------------------------------------------------
# The registry's declared surface, against the published model reference
# ---------------------------------------------------------------------------

class TestRegistryThinkingDeclarations:
    """The capability table is the thing that makes the refusals right, so it
    is asserted directly. Anthropic model reference + migration guide, verified
    2026-07-31."""

    def test_opus_5_thinks_by_default(self):
        support = thinking_support(OPUS_5)
        assert support.default_on is True
        assert support.disabled_max_effort == "high"
        assert set(support.efforts) == set(EFFORT_LEVELS)
        assert THINKING_BUDGET not in support.modes

    def test_sonnet_5_thinks_by_default(self):
        support = thinking_support(SONNET_5)
        assert support.default_on is True
        assert support.disabled_max_effort is None
        assert set(support.efforts) == set(EFFORT_LEVELS)
        assert THINKING_BUDGET not in support.modes

    def test_opus_4_8_does_not_think_by_default(self):
        # The difference from Opus 5 that makes "a caller gets behaviour it
        # never chose" a real hazard rather than a hypothetical: the same
        # spec-free call means thinking on one model and not on the other.
        support = thinking_support(OPUS_4_8)
        assert support.default_on is False
        assert THINKING_ADAPTIVE in support.modes
        assert THINKING_BUDGET not in support.modes

    def test_sonnet_4_6_keeps_the_deprecated_budget_escape_hatch(self):
        support = thinking_support(SONNET_4_6)
        assert THINKING_BUDGET in support.modes
        assert "xhigh" not in support.efforts

    def test_haiku_4_5_is_budget_only_with_no_effort(self):
        support = thinking_support(HAIKU_4_5)
        assert support.modes == (THINKING_BUDGET,)
        assert support.efforts == ()
        assert support.default_effort is None

    def test_every_current_anthropic_entry_declares_its_surface(self):
        undeclared = [
            model_id for model_id, info in _anthropic_entries()
            if not info.retired and info.thinking is None]
        assert undeclared == []

    def test_every_declared_entry_states_its_display_settings(self):
        # `display` is accepted on both generations and in both on-modes; only
        # the DEFAULT is generation-scoped ("omitted" from 4.7 on, "summarized"
        # before it), and a default is not a capability. An entry that leaves
        # `displays` empty refuses the field, so this asserts the declaration is
        # deliberate rather than forgotten.
        from direktoro import THINKING_DISPLAYS
        for model_id, info in _anthropic_entries():
            if info.retired:
                continue
            assert set(info.thinking.displays) == set(THINKING_DISPLAYS), \
                model_id

    def test_display_is_accepted_on_every_live_entry_in_its_own_on_mode(self):
        # The end-to-end check behind the declaration: the emitted block carries
        # the display, whichever on-mode the model has.
        for model_id in (OPUS_5, OPUS_4_8, "claude-opus-4-7", SONNET_5,
                         SONNET_4_6):
            dec = resolved_decoding_params(
                model_id, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE,
                                  display="summarized"))
            assert dec["thinking"]["display"] == "summarized", model_id
        dec = resolved_decoding_params(
            HAIKU_4_5, max_tokens=8192,
            thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=2048,
                              display="summarized"))
        assert dec["thinking"]["display"] == "summarized"

    def test_a_declared_surface_is_the_minority_of_the_table(self):
        # What the README and `thinking_support` both say, asserted rather than
        # counted by hand: a thinking surface is declared on the live Anthropic
        # entries and NOWHERE else, so most of the table has none. Undeclared
        # does not mean "cannot think" — it means direktoro refuses to emit a
        # thinking shape for that model rather than guessing one — and stating
        # it the other way round would over-claim what has been verified.
        from direktoro import MODEL_REGISTRY, PROVIDER_ANTHROPIC

        declared = sorted(model_id for model_id, info in MODEL_REGISTRY.items()
                          if info.thinking is not None)
        assert declared == sorted(
            model_id for model_id, info in MODEL_REGISTRY.items()
            if info.provider == PROVIDER_ANTHROPIC and not info.retired)
        assert len(declared) < len(MODEL_REGISTRY) - len(declared), (
            "a declared thinking surface is documented as the minority of the "
            "table; if that has changed, the README says so too.")

    def test_no_non_anthropic_entry_declares_a_surface(self):
        # Declaring one would let the seam emit Anthropic keys onto a wire that
        # does not have them; the provider gate in `_thinking_params` is the
        # other half of that guard.
        from direktoro import MODEL_REGISTRY, PROVIDER_ANTHROPIC
        declared = [model_id for model_id, info in MODEL_REGISTRY.items()
                    if info.provider != PROVIDER_ANTHROPIC
                    and info.thinking is not None]
        assert declared == []


def _anthropic_entries():
    from direktoro import MODEL_REGISTRY, PROVIDER_ANTHROPIC
    return [(model_id, info) for model_id, info in MODEL_REGISTRY.items()
            if info.provider == PROVIDER_ANTHROPIC]


# ---------------------------------------------------------------------------
# Stub client (no SDK, no network)
# ---------------------------------------------------------------------------

class _StubAnthropicClient:
    """Captures the wire request the AnthropicAdapter builds.

    Mirrors just enough of `anthropic.Anthropic`: `messages.stream(**wire)` used
    as a context manager, yielding an object with a `text_stream` and a
    `get_final_message()`."""

    def __init__(self):
        self.wire = None
        self.messages = self

    def stream(self, **wire):
        self.wire = wire
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(())

    def get_final_message(self):
        return _StubMessage()


class _StubMessage:
    content = []
    model = OPUS_5
    stop_reason = "end_turn"
    usage = None
