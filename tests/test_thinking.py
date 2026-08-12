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

    def test_an_undeclared_openai_entry_still_refuses_the_spec(self):
        # The GPT entries declare no thinking surface (their accepted levels
        # have not been read from a current reference), so a spec aimed at one
        # is refused with the fact rather than rendered on a guess; the
        # registry `reasoning_effort` quirk keeps carrying their omitted-state
        # default.
        with pytest.raises(ThinkingUnsupported, match="declares no thinking"):
            resolved_decoding_params(
                GPT, max_tokens=8192,
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


class TestOpenAIFamilyRendering:
    """A thinking spec renders onto the OpenAI-family wires as one reasoning
    level: a named effort as itself, a disabled mode as "none" (the probed
    off-switch), and adaptive-alone as nothing at all — the omitted-state
    behaviour of a default-on endpoint, with no invented value in identity."""

    GLM_46V = "z-ai/glm-4.6v"
    QWEN = "qwen/qwen3-vl-235b-a22b-instruct"
    GEMINI = "google/gemini-3.6-flash"

    def test_a_named_effort_rides_the_chat_wire(self):
        dec = resolved_decoding_params(
            self.GLM_46V, max_tokens=8192,
            thinking=Thinking(mode=THINKING_ADAPTIVE, effort="high"))
        assert dec == {"max_tokens": 8192, "reasoning_effort": "high"}

    def test_adaptive_alone_emits_nothing(self):
        # Reasoning on a default-on endpoint IS the omitted-state behaviour;
        # emitting a level for it would fold a value nobody chose into
        # identity.
        plain = resolved_decoding_params(self.GLM_46V, max_tokens=8192)
        adaptive = resolved_decoding_params(
            self.GLM_46V, max_tokens=8192,
            thinking=Thinking(mode=THINKING_ADAPTIVE))
        assert adaptive == plain == {"max_tokens": 8192}

    def test_disabled_rides_as_none(self):
        dec = resolved_decoding_params(
            self.GLM_46V, max_tokens=512,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert dec == {"max_tokens": 512, "reasoning_effort": "none"}

    def test_disabled_with_an_effort_is_refused(self):
        # One wire key cannot carry two levels.
        with pytest.raises(ThinkingUnsupported, match="ONE reasoning level"):
            resolved_decoding_params(
                self.GLM_46V, max_tokens=8192,
                thinking=Thinking(mode=THINKING_DISABLED, effort="low"))

    def test_budget_and_display_have_no_rendering(self):
        with pytest.raises(ThinkingUnsupported, match="budget_tokens"):
            resolved_decoding_params(
                self.GLM_46V, max_tokens=8192,
                thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=2048))
        with pytest.raises(ThinkingUnsupported, match="display"):
            resolved_decoding_params(
                self.GLM_46V, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE,
                                  display="summarized"))

    def test_an_endpoint_with_no_reasoning_parameter_refuses_a_level(self):
        # Qwen's EMPTY efforts tuple is a probed fact (every level 404s under
        # require_parameters), so naming one is refused with that fact.
        with pytest.raises(ThinkingUnsupported,
                           match="no reasoning-effort parameter"):
            resolved_decoding_params(
                self.QWEN, max_tokens=4096,
                thinking=Thinking(effort="high"))

    def test_disable_on_a_non_reasoning_endpoint_is_satisfied_by_omission(
            self):
        # Qwen does not reason unless asked (it cannot be asked), so
        # "disabled" is already true of the plain call.
        plain = resolved_decoding_params(self.QWEN, max_tokens=4096)
        disabled = resolved_decoding_params(
            self.QWEN, max_tokens=4096,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert disabled == plain

    def test_mandatory_reasoning_cannot_be_disabled(self):
        # Gemini's probe answered 400 "Reasoning is mandatory for this
        # endpoint"; the registry says so and the seam refuses before spend.
        with pytest.raises(ThinkingUnsupported, match="reasons by default"):
            resolved_decoding_params(
                self.GEMINI, max_tokens=4096,
                thinking=Thinking(mode=THINKING_DISABLED))

    def test_a_caller_level_replaces_the_registry_default(self):
        # The GPT quirk is the omitted-state default: sent when the caller
        # says nothing. (A caller cannot yet name a level for GPT — its
        # surface is undeclared — so the replacement is asserted on identity
        # shape via the chat wire instead.)
        spec_less = resolved_decoding_params("gpt-5.6-sol", max_tokens=4096)
        assert spec_less["reasoning"] == {"effort": "medium"}
        chosen = resolved_decoding_params(
            self.GLM_46V, max_tokens=8192,
            thinking=Thinking(mode=THINKING_ADAPTIVE, effort="low"))
        assert chosen["reasoning_effort"] == "low"

    def test_effort_reaches_call_identity_on_the_chat_wire(self):
        low = canonical_json(call_identity_fields(
            self.GLM_46V, decoding_params=resolved_decoding_params(
                self.GLM_46V, max_tokens=8192,
                thinking=Thinking(effort="low"))))
        high = canonical_json(call_identity_fields(
            self.GLM_46V, decoding_params=resolved_decoding_params(
                self.GLM_46V, max_tokens=8192,
                thinking=Thinking(effort="high"))))
        assert low != high

    def test_a_caller_level_replaces_the_quirk_on_one_entry(self, monkeypatch):
        # The headline claim of the REGISTRY-DEFAULTED split, exercised on a
        # single synthetic entry carrying BOTH a quirk and a declared surface
        # (no live entry has both yet): the caller's level rides the wire and
        # the quirk's does not.
        from direktoro.registry import MODEL_REGISTRY, Model, ThinkingSupport
        monkeypatch.setitem(
            MODEL_REGISTRY, "synthetic-quirk-and-surface",
            Model("openrouter", "https://openrouter.ai/api/v1",
                  "OPENROUTER_API_KEY", wire_api="chat_completions",
                  quirks={"reasoning_effort": "medium"},
                  forced_tool_choice=True,
                  thinking=ThinkingSupport(
                      modes=(THINKING_ADAPTIVE,),
                      efforts=("low", "medium", "high"), default_on=False)))
        spec_less = resolved_decoding_params(
            "synthetic-quirk-and-surface", max_tokens=4096)
        assert spec_less["reasoning_effort"] == "medium"
        chosen = resolved_decoding_params(
            "synthetic-quirk-and-surface", max_tokens=4096,
            thinking=Thinking(effort="low"))
        assert chosen["reasoning_effort"] == "low"

    def test_disable_is_refused_when_a_quirk_would_run_anyway(
            self, monkeypatch):
        # On an entry with a registry-default level, no off-switch and no
        # default-on reasoning, a disabled request cannot be honoured by
        # omission — the quirk would run regardless — so it is refused
        # rather than silently overridden.
        from direktoro.registry import MODEL_REGISTRY, Model, ThinkingSupport
        monkeypatch.setitem(
            MODEL_REGISTRY, "synthetic-quirk-no-off",
            Model("openrouter", "https://openrouter.ai/api/v1",
                  "OPENROUTER_API_KEY", wire_api="chat_completions",
                  quirks={"reasoning_effort": "medium"},
                  forced_tool_choice=True,
                  thinking=ThinkingSupport(
                      modes=(THINKING_ADAPTIVE,),
                      efforts=("low", "medium", "high"), default_on=False)))
        with pytest.raises(ThinkingUnsupported,
                           match="registry-default reasoning"):
            resolved_decoding_params(
                "synthetic-quirk-no-off", max_tokens=4096,
                thinking=Thinking(mode=THINKING_DISABLED))
        # And the contradictory two-part spec is refused as a wire property,
        # not silently dropped, even though this entry declares no disabled
        # mode.
        with pytest.raises(ThinkingUnsupported, match="ONE reasoning level"):
            resolved_decoding_params(
                "synthetic-quirk-no-off", max_tokens=4096,
                thinking=Thinking(mode=THINKING_DISABLED, effort="low"))

    def test_a_caller_level_re_spells_for_the_responses_wire(
            self, monkeypatch):
        # No live Responses-wire entry declares a surface yet; the branch is
        # pinned on a synthetic one so it cannot rot unexercised.
        from direktoro.registry import MODEL_REGISTRY, Model, ThinkingSupport
        monkeypatch.setitem(
            MODEL_REGISTRY, "synthetic-responses-surface",
            Model("openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
                  wire_api="responses", forced_tool_choice=True,
                  thinking=ThinkingSupport(
                      modes=(THINKING_ADAPTIVE,),
                      efforts=("low", "medium", "high"), default_on=False)))
        dec = resolved_decoding_params(
            "synthetic-responses-surface", max_tokens=4096,
            thinking=Thinking(effort="high"))
        assert dec["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in dec


class TestSplitDecodingConfig:
    """One role's decoding block from an application config splits into the
    (sampling, thinking) pair the resolver takes, without the application
    knowing which key is which."""

    def test_a_mixed_block_splits(self):
        from direktoro import split_decoding_config
        sampling, thinking = split_decoding_config(
            {"temperature": 0.0, "top_p": 0.9,
             "thinking_mode": "adaptive", "thinking_effort": "high"})
        assert sampling == {"temperature": 0.0, "top_p": 0.9}
        assert thinking == Thinking(mode=THINKING_ADAPTIVE, effort="high")

    def test_sampling_only_and_thinking_only(self):
        from direktoro import split_decoding_config
        assert split_decoding_config({"temperature": 1.0}) == \
            ({"temperature": 1.0}, None)
        sampling, thinking = split_decoding_config(
            {"thinking_effort": "low"})
        assert sampling is None
        assert thinking == Thinking(effort="low")

    def test_absent_and_empty_blocks_are_nothing(self):
        from direktoro import split_decoding_config
        assert split_decoding_config(None) == (None, None)
        assert split_decoding_config({}) == (None, None)

    def test_a_null_value_reads_as_unspecified(self):
        # A config may carry a fixed key set and leave values empty; the
        # resolver's own convention, honoured here so the two agree —
        # sampling and thinking keys ALIKE, so a null temperature is never
        # reported to an operator as a value that was specified and inert.
        from direktoro import split_decoding_config
        sampling, thinking = split_decoding_config(
            {"temperature": 0.0, "thinking_mode": None})
        assert sampling == {"temperature": 0.0}
        assert thinking is None
        assert split_decoding_config(
            {"temperature": None, "top_p": None,
             "thinking_mode": None}) == (None, None)

    def test_integer_spellings_of_float_controls_are_normalised(self):
        # YAML spells one intent two ways (`0` and `0.0`) and the resolved
        # value folds into call identity byte-for-byte, so the float
        # controls normalise; two configs meaning the same call fingerprint
        # together. top_k is integral and is left alone.
        from direktoro import split_decoding_config
        a, _ = split_decoding_config({"temperature": 0, "top_p": 1})
        b, _ = split_decoding_config({"temperature": 0.0, "top_p": 1.0})
        assert a == b == {"temperature": 0.0, "top_p": 1.0}
        assert all(isinstance(v, float) for v in a.values())
        c, _ = split_decoding_config({"top_k": 5})
        assert c == {"top_k": 5} and isinstance(c["top_k"], int)

    def test_a_non_string_key_still_gets_the_naming_error(self):
        # Raw YAML can produce non-string keys; they must land in THIS error,
        # named, not in a TypeError from sorting mixed types.
        from direktoro import split_decoding_config
        with pytest.raises(ValueError, match="unknown decoding key"):
            split_decoding_config({1: 0.5, "temprature": 0.2})

    def test_an_unknown_key_fails_at_config_load(self):
        from direktoro import split_decoding_config
        with pytest.raises(ValueError, match="unknown decoding key"):
            split_decoding_config({"temprature": 0.0})
        with pytest.raises(ValueError, match="must be a mapping"):
            split_decoding_config(0.7)

    def test_a_bad_thinking_value_fails_with_the_spec_error(self):
        # Value validation is the Thinking constructor's, not a second
        # opinion here.
        from direktoro import split_decoding_config
        with pytest.raises(ValueError, match="not a known level"):
            split_decoding_config({"thinking_effort": "extreme"})

    def test_the_split_feeds_the_resolver_end_to_end(self):
        from direktoro import split_decoding_config
        sampling, thinking = split_decoding_config(
            {"thinking_effort": "high"})
        dec = resolved_decoding_params(
            "z-ai/glm-4.6v", max_tokens=8192,
            sampling=sampling, thinking=thinking)
        assert dec == {"max_tokens": 8192, "reasoning_effort": "high"}


class TestStarvingCapRefused:
    """A reasoning call under a cap below direktoro's own policy floor is
    refused before spend — the endpoint would accept it and burn the whole
    cap on reasoning. The floor is this package's number, not a vendor's,
    and the refusal says so."""

    def test_the_floor_is_named_as_policy_not_endpoint_fact(self):
        with pytest.raises(ThinkingUnsupported,
                           match="direktoro's own floor, not the endpoint's"):
            resolved_decoding_params(OPUS_5, max_tokens=1024)

    def test_an_explicit_budget_is_exempt(self):
        # The caller stated its own arithmetic: budget < max_tokens is
        # enforced where the budget is validated, and the policy floor does
        # not second-guess a stated pairing the API accepts.
        dec = resolved_decoding_params(
            HAIKU_4_5, max_tokens=2000,
            thinking=Thinking(mode=THINKING_BUDGET, budget_tokens=1024))
        assert dec["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert dec["max_tokens"] == 2000

    def test_a_named_effort_arms_the_guard_on_the_openai_wires(self):
        # On these wires the effort key IS the reasoning switch, so naming a
        # level under a starving cap is refused even though the entry's
        # default_on already covers the spec-less case.
        with pytest.raises(ThinkingUnsupported,
                           match="names a reasoning effort"):
            resolved_decoding_params(
                "z-ai/glm-4.6v", max_tokens=1024,
                thinking=Thinking(effort="high"))

    def test_the_refusal_names_the_actual_cause(self):
        # A spec-less call on a default-on model is refused FOR the default,
        # and the message says so rather than blaming a spec nobody passed.
        with pytest.raises(ThinkingUnsupported, match="on by default"):
            resolved_decoding_params(OPUS_5, max_tokens=1024)
        with pytest.raises(ThinkingUnsupported, match="asks for 'adaptive'"):
            resolved_decoding_params(
                OPUS_4_8, max_tokens=1024,
                thinking=Thinking(mode=THINKING_ADAPTIVE))

    def test_default_on_model_with_a_small_cap_is_refused(self):
        with pytest.raises(ThinkingUnsupported, match="cannot fit"):
            resolved_decoding_params(OPUS_5, max_tokens=1024)

    def test_disabling_thinking_lifts_the_refusal(self):
        dec = resolved_decoding_params(
            OPUS_5, max_tokens=1024,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert dec["max_tokens"] == 1024

    def test_a_default_off_model_passes_with_a_small_cap(self):
        dec = resolved_decoding_params(OPUS_4_8, max_tokens=1024)
        assert dec == {"max_tokens": 1024}

    def test_explicitly_enabling_thinking_arms_the_guard(self):
        with pytest.raises(ThinkingUnsupported, match="cannot fit"):
            resolved_decoding_params(
                OPUS_4_8, max_tokens=1024,
                thinking=Thinking(mode=THINKING_ADAPTIVE))

    def test_the_guard_covers_default_on_routed_entries(self):
        with pytest.raises(ThinkingUnsupported, match="cannot fit"):
            resolved_decoding_params("z-ai/glm-4.6v", max_tokens=1024)
        dec = resolved_decoding_params(
            "z-ai/glm-4.6v", max_tokens=1024,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert dec == {"max_tokens": 1024, "reasoning_effort": "none"}

    def test_an_undeclared_surface_is_not_guarded(self):
        # Nothing established, nothing to guard: the GPT entries carry no
        # ThinkingSupport, so a small cap passes through to the endpoint.
        dec = resolved_decoding_params("gpt-5.6-sol", max_tokens=64)
        assert dec["max_output_tokens"] == 64

    def test_no_cap_no_guard(self):
        dec = resolved_decoding_params(OPUS_5, max_tokens=None)
        assert dec["max_tokens"] is None


class TestSamplingParamsAndThinking:
    """The one constraint `ThinkingSupport` cannot express, because it is not a
    property of the model alone.

    Anthropic's thinking documentation: on the 4.7-generation-and-later models a
    non-default temperature/top_p/top_k is a 400 on every request (those entries
    declare it in `rejects_sampling`). "On older models, the restriction applies
    only while thinking is on: `temperature` and `top_k` are incompatible with
    thinking, and `top_p` is allowed at values between 0.95 and 1." Sonnet 4.6
    and Haiku 4.5 are exactly the two live entries that declare no refusal, so
    they accept a temperature — until the same request also asks them to think,
    at which point resolving the two independently would put a 400-producing
    pair on the wire. `top_p` inside its window is the one pair the
    documentation allows, and it goes through."""

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

    def test_an_in_window_top_p_rides_alongside_thinking(self):
        # The documentation allows `top_p` between 0.95 and 1 with thinking on,
        # so that pair goes to the wire: refusing it would refuse a call the
        # endpoint serves.
        dec = resolved_decoding_params(
            SONNET_4_6, sampling={"top_p": 0.97}, max_tokens=8192,
            thinking=Thinking(mode=THINKING_ADAPTIVE))
        assert dec["top_p"] == 0.97
        assert dec["thinking"] == {"type": "adaptive"}

    def test_the_window_is_inclusive_at_both_ends(self):
        for value in (0.95, 1.0):
            dec = resolved_decoding_params(
                SONNET_4_6, sampling={"top_p": value}, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))
            assert dec["top_p"] == value

    def test_an_out_of_window_top_p_is_refused_naming_the_window(self):
        with pytest.raises(ThinkingUnsupported) as excinfo:
            resolved_decoding_params(
                SONNET_4_6, sampling={"top_p": 0.5}, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))
        message = str(excinfo.value)
        assert "0.95" in message and "1.0" in message
        assert "top_p" in message

    def test_a_temperature_is_still_refused_beside_an_allowed_top_p(self):
        # `top_p` being allowed changes nothing for the other two controls.
        with pytest.raises(ThinkingUnsupported, match="`temperature`"):
            resolved_decoding_params(
                SONNET_4_6, sampling={"temperature": 0.0, "top_p": 0.97},
                max_tokens=8192, thinking=Thinking(mode=THINKING_ADAPTIVE))

    def test_top_p_outside_the_window_is_fine_without_thinking(self):
        # The window is a property of the PAIR: with thinking off, `top_p` is
        # whatever the model's own band allows.
        dec = resolved_decoding_params(
            SONNET_4_6, sampling={"top_p": 0.5}, max_tokens=8192,
            thinking=Thinking(mode=THINKING_DISABLED))
        assert dec["top_p"] == 0.5

    def test_top_k_is_refused_beside_active_thinking(self):
        # The other named-incompatible control; previously only temperature
        # exercised that branch.
        with pytest.raises(ThinkingUnsupported, match="`top_k`"):
            resolved_decoding_params(
                SONNET_4_6, sampling={"top_k": 40}, max_tokens=8192,
                thinking=Thinking(mode=THINKING_ADAPTIVE))


class TestResponsesWireHasNoTopK:
    """The Responses API spells no `top_k` at all, so a caller's top_k on an
    entry that does not refuse it is a wire fact, refused before the SDK call
    rather than crashing inside it or riding as a kwarg nothing reads."""

    def test_top_k_is_refused_as_a_wire_fact(self):
        with pytest.raises(ValueError, match="no `top_k` parameter"):
            resolved_decoding_params(
                GPT, sampling={"top_k": 40}, max_tokens=4096)

    def test_the_other_controls_still_flow(self):
        dec = resolved_decoding_params(
            GPT, sampling={"top_p": 0.9}, max_tokens=4096)
        assert dec["top_p"] == 0.9


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

    def test_efforts_without_a_default_effort_are_valid(self):
        # None means no single level corresponds to the omitted state — the
        # shape of the routed records, whose omitted-state behaviour is
        # dynamic. Declaring the accepted levels does not require pretending
        # one of them is the default.
        support = ThinkingSupport(efforts=("low", "high"))
        assert support.default_effort is None

    def test_disabled_ceiling_needs_a_default_effort(self):
        # The cap is evaluated at the omitted-state level for a request that
        # disables thinking without naming an effort, so an entry declaring
        # the cap must state that level.
        with pytest.raises(ValueError, match="needs default_effort"):
            ThinkingSupport(
                modes=(THINKING_ADAPTIVE, THINKING_DISABLED),
                efforts=("low", "high"), disabled_max_effort="high")

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

    def test_the_undeclared_surfaces_are_pinned(self):
        # A surface is declared exactly where evidence exists: the live
        # Anthropic entries (published reference) and four routed entries
        # (live probes 2026-08-12) — including Qwen's, whose EMPTY surface is
        # a probed fact. Undeclared means direktoro refuses a spec rather
        # than guessing: the three retired ids (unverifiable), the GPT
        # entries (no current reference read for their levels), and MiMo
        # (probe rate-limited part-way; see its entry comment).
        from direktoro import MODEL_REGISTRY

        undeclared = sorted(model_id for model_id, info in
                            MODEL_REGISTRY.items() if info.thinking is None)
        assert undeclared == sorted([
            "claude-3-5-sonnet-20241022", "claude-opus-4-20250514",
            "claude-sonnet-4-20250514", "gpt-5.6-sol", "gpt-5.6-terra",
            "xiaomi/mimo-v2.5"])

    def test_routed_probed_surfaces_say_what_the_probes_saw(self):
        # The GLM pair: reasons by default, all five ladder levels route, and
        # the wire off-switch works — and no Anthropic-only concept (display,
        # budget) is declared for a wire that has none.
        glm = thinking_support("z-ai/glm-4.6v")
        assert glm.default_on is True
        assert set(glm.efforts) == set(EFFORT_LEVELS)
        assert THINKING_DISABLED in glm.modes
        assert glm.displays == ()
        assert glm.default_effort is None
        assert thinking_support("z-ai/glm-5v-turbo") == glm
        # Gemini 3.6 Flash: reasons by default and CANNOT be disabled (the
        # probe's 400: "Reasoning is mandatory for this endpoint").
        gem = thinking_support("google/gemini-3.6-flash")
        assert gem.default_on is True
        assert THINKING_DISABLED not in gem.modes
        assert set(gem.efforts) == set(EFFORT_LEVELS)
        # Qwen instruct: no reasoning surface at all, declared as a fact.
        qwen = thinking_support("qwen/qwen3-vl-235b-a22b-instruct")
        assert qwen.modes == ()
        assert qwen.efforts == ()
        assert qwen.default_on is False


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
