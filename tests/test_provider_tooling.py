"""Tool-calling helpers and adapter construction: `tool_choice_named`,
`extract_tool_call`, and `build_adapter`.

All offline. The first two are pure, and `build_adapter` reads an injected env
(never the process environment or the network), so the missing-key path and the
client-construction path both run without a real key.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from direktoro.providers import (
    AnthropicAdapter,
    MissingAPIKey,
    OpenAIAdapter,
    build_adapter,
    extract_tool_call,
    tool_choice_named,
)
from direktoro.registry import (
    PROVIDER_OPENROUTER, model_info, supports_forced_tool_choice,
    supports_sampling_params)


class TestToolChoiceNamed:
    def test_anthropic_shape(self):
        assert tool_choice_named("claude-haiku-4-5-20251001", "record_answer") \
            == {"type": "tool", "name": "record_answer"}

    def test_openai_responses_shape(self):
        assert tool_choice_named("gpt-5.6-sol", "record_answer") \
            == {"type": "function", "name": "record_answer"}

    def test_chat_completions_shape(self):
        # A FORCING Chat Completions model rides the nested `function` shape.
        # The routed Qwen entry is the one to assert it with: the GLM entries
        # speak the same wire but cannot be forced at all, so they exercise the
        # degrade path instead (see TestForcedToolChoiceDegrade).
        assert tool_choice_named(
            "qwen/qwen3-vl-235b-a22b-instruct", "record_answer") \
            == {"type": "function", "function": {"name": "record_answer"}}


class TestForcedToolChoiceFlag:
    def test_default_is_true(self):
        # Every direct entry and the routed Qwen flagship force a named tool.
        for m in ("claude-haiku-4-5-20251001", "claude-opus-4-8",
                  "gpt-5.6-sol", "gpt-5.6-terra",
                  "qwen/qwen3-vl-235b-a22b-instruct"):
            assert model_info(m).forced_tool_choice is True, m

    def test_glm_vision_endpoints_cannot_force(self):
        # The two GLM vision endpoints 404 a forced tool_choice on Z.AI's host
        # (live 2026-07-23), so their entries set forced_tool_choice=False.
        for m in ("z-ai/glm-5v-turbo", "z-ai/glm-4.6v"):
            assert model_info(m).forced_tool_choice is False, m


class TestSupportsForcedToolChoicePredicate:
    def test_true_for_forcing_models(self):
        for m in ("claude-haiku-4-5-20251001", "gpt-5.6-sol",
                  "qwen/qwen3-vl-235b-a22b-instruct"):
            assert supports_forced_tool_choice(m) is True, m

    def test_false_for_glm_vision(self):
        for m in ("z-ai/glm-5v-turbo", "z-ai/glm-4.6v"):
            assert supports_forced_tool_choice(m) is False, m

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            supports_forced_tool_choice("totally-made-up-model-9000")


class TestSupportsSamplingParamsPredicate:
    """The sampling-params capability seam, mirroring the forced-tool-choice
    predicate. True for every model that lists the sampling controls; False only
    for google/gemini-3.6-flash, whose Vertex endpoints list neither temperature
    nor top_p."""

    def test_true_for_normal_models(self):
        # Anthropic (even a no_temperature reasoning model still counts as
        # sampling-capable at the registry level — the quirk is the finer gate),
        # OpenAI, and the other routed models all take sampling params.
        for m in ("claude-haiku-4-5-20251001", "claude-opus-4-8", "gpt-5.6-sol",
                  "qwen/qwen3-vl-235b-a22b-instruct", "xiaomi/mimo-v2.5",
                  "z-ai/glm-4.6v"):
            assert supports_sampling_params(m) is True, m
            assert model_info(m).supports_sampling_params is True, m

    def test_false_for_gemini_36(self):
        assert supports_sampling_params("google/gemini-3.6-flash") is False
        assert model_info(
            "google/gemini-3.6-flash").supports_sampling_params is False

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            supports_sampling_params("totally-made-up-model-9000")


class TestForcedToolChoiceDegrade:
    def test_glm_degrades_to_auto_per_wire(self):
        # tool_choice_named returns the canonical auto value for a non-forcing
        # model, so a call-site that forces a named tool degrades to auto with
        # no code change. Both GLM vision endpoints degrade.
        for m in ("z-ai/glm-5v-turbo", "z-ai/glm-4.6v"):
            assert tool_choice_named(m, "record_answer") == {"type": "auto"}, m

    def test_forcing_models_still_force(self):
        # The degrade is scoped to non-forcing models; forcing models are
        # unchanged across all three wire shapes.
        assert tool_choice_named("claude-haiku-4-5-20251001", "x") \
            == {"type": "tool", "name": "x"}
        assert tool_choice_named("gpt-5.6-sol", "x") \
            == {"type": "function", "name": "x"}
        assert tool_choice_named("qwen/qwen3-vl-235b-a22b-instruct", "x") \
            == {"type": "function", "function": {"name": "x"}}


def _resp(blocks, stop_reason="tool_use"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


def _tool_use(name, input, id="tu1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


class TestExtractToolCall:
    def test_single_named_call_returns_input(self):
        answer = {"answer": "Paris"}
        got, err = extract_tool_call(
            _resp([_tool_use("record_answer", answer)]), "record_answer")
        assert err is None
        assert got == answer

    def test_no_tool_call_is_an_error(self):
        got, err = extract_tool_call(
            _resp([SimpleNamespace(type="text", text="hi")], "end_turn"),
            "record_answer")
        assert got is None
        assert "no tool call" in err

    def test_wrong_tool_name_is_an_error(self):
        got, err = extract_tool_call(
            _resp([_tool_use("other", {})]), "record_answer")
        assert got is None
        assert "record_answer" in err

    def test_double_call_is_an_error(self):
        got, err = extract_tool_call(
            _resp([_tool_use("record_answer", {}, "a"),
                   _tool_use("record_answer", {}, "b")]), "record_answer")
        assert got is None
        assert "2 times" in err


class TestBuildAdapter:
    def test_missing_key_raises(self):
        with pytest.raises(MissingAPIKey) as e:
            build_adapter("claude-haiku-4-5-20251001", env={})
        assert "ANTHROPIC_API_KEY" in str(e.value)

    def test_anthropic_adapter_built_from_injected_env(self):
        # The other SDK-dependent test: `build_adapter` constructs a real
        # `anthropic.Anthropic` client (no network — the SDK does not dial on
        # construction). Skipped without the SDK so the suite runs anywhere
        # `import direktoro` does; the injected-client test below covers the
        # same seam with no SDK at all.
        pytest.importorskip(
            "anthropic", reason="constructs a real anthropic.Anthropic client")
        adapter = build_adapter(
            "claude-haiku-4-5-20251001", env={"ANTHROPIC_API_KEY": "sk-test"})
        assert isinstance(adapter, AnthropicAdapter)
        # Anthropic uses the SDK default base URL (None in the registry).
        assert adapter.base_url is None

    # ---- The injected-client and max_retries construction seams -----------

    def test_injected_client_skips_key_resolution_anthropic(self):
        # An injected client wraps directly: no env, no key, no SDK needed.
        stub = object()
        adapter = build_adapter("claude-haiku-4-5-20251001", client=stub)
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter._client is stub
        assert adapter.base_url is None

    def test_injected_client_skips_key_resolution_openai_family(self):
        # A routed (GLM) id builds an OpenAIAdapter around the injected client,
        # keyed by its own provider and base URL from the registry.
        stub = object()
        info = model_info("z-ai/glm-4.6v")
        adapter = build_adapter("z-ai/glm-4.6v", client=stub)
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter._client is stub
        assert adapter.provider == PROVIDER_OPENROUTER
        assert adapter.base_url == info.base_url

    def test_injected_client_needs_no_key_even_with_empty_env(self):
        # The whole point for the fan-out / stubs: no MissingAPIKey when a
        # client is supplied, even with an empty env.
        adapter = build_adapter(
            "claude-haiku-4-5-20251001", client=object(), env={})
        assert isinstance(adapter, AnthropicAdapter)

    def test_client_plus_max_retries_is_a_loud_error(self):
        with pytest.raises(ValueError) as e:
            build_adapter(
                "claude-haiku-4-5-20251001", client=object(), max_retries=0)
        assert "max_retries" in str(e.value)

    def test_max_retries_reaches_anthropic_client(self, monkeypatch):
        # max_retries forwards to the SDK constructor on the build-the-client
        # path. A fake `anthropic` module captures the kwargs, so the test is
        # hermetic and works whether or not the real SDK is installed.
        captured = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("anthropic")
        fake.Anthropic = FakeAnthropic
        monkeypatch.setitem(sys.modules, "anthropic", fake)

        adapter = build_adapter(
            "claude-haiku-4-5-20251001",
            env={"ANTHROPIC_API_KEY": "sk-test"}, max_retries=0)
        assert isinstance(adapter, AnthropicAdapter)
        assert captured["api_key"] == "sk-test"
        assert captured["max_retries"] == 0

    def test_max_retries_reaches_openai_client(self, monkeypatch):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("openai")
        fake.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake)

        info = model_info("z-ai/glm-4.6v")
        adapter = build_adapter(
            "z-ai/glm-4.6v",
            env={"OPENROUTER_API_KEY": "sk-or"}, max_retries=0)
        assert isinstance(adapter, OpenAIAdapter)
        assert captured["api_key"] == "sk-or"
        assert captured["base_url"] == info.base_url
        assert captured["max_retries"] == 0

    def test_no_max_retries_leaves_sdk_default_untouched(self, monkeypatch):
        # Default path forwards NO max_retries kwarg, so the SDK default stands.
        captured = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("anthropic")
        fake.Anthropic = FakeAnthropic
        monkeypatch.setitem(sys.modules, "anthropic", fake)

        build_adapter(
            "claude-haiku-4-5-20251001", env={"ANTHROPIC_API_KEY": "sk-test"})
        assert "max_retries" not in captured
