"""Offline tests for the provider adapters: request translation out to each wire
format, response translation back to the canonical shape, and the routing
provenance captured on a gateway-served call.

No network, no API key. The OpenAI-family paths run against hand-authored
request/response fixtures (the translators are pure, and a `.model_dump()` dict
is all the adapter reads), and the Anthropic adapter against a stubbed streaming
client.
"""

import base64
import json
import sys
from types import SimpleNamespace

import pytest

from direktoro.providers import (
    AnthropicAdapter,
    NormalisedResponse,
    NormalisedUsage,
    OpenAIAdapter,
    ProviderAccountError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRetryableError,
    RETRY_BACKOFF_SECONDS,
    _messages_to_chat,
    _messages_to_input,
    _system_to_instructions,
    _to_chat_completions_wire,
    _to_openai_wire,
    _translate_anthropic_error,
    _translate_openai_error,
    create_message_with_retry,
)
from direktoro.registry import (
    MODEL_REGISTRY, OPENAI_BASE_URL, OPENROUTER_BASE_URL,
    WIRE_CHAT_COMPLETIONS, model_info)
from direktoro.routing import ProviderRouteMismatch


PNG = b"\x89PNG\r\n\x1a\nfake-bytes"
PNG_B64 = base64.b64encode(PNG).decode("ascii")


# ---------------------------------------------------------------------------
# OpenAI request translation (canonical Anthropic shape -> Responses wire)
# ---------------------------------------------------------------------------

class TestOpenAIRequestTranslation:
    def test_system_blocks_become_instructions(self):
        system = [
            {"type": "text", "text": "Line A",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Line B"},
        ]
        assert _system_to_instructions(system) == "Line A\n\nLine B"

    def test_tools_become_function_definitions(self):
        wire, _ = _to_openai_wire(
            model="gpt-5.6-sol", system="S", messages=[],
            tools=[{"name": "record_answer", "description": "d",
                    "input_schema": {"type": "object", "properties": {}}}],
            tool_choice={"type": "auto"}, max_tokens=4096,
            sampling={"temperature": 0.0})
        assert wire["tools"][0] == {
            "type": "function",
            "name": "record_answer",
            "description": "d",
            "strict": False,
            "parameters": {"type": "object", "properties": {}},
        }
        assert wire["tool_choice"] == "auto"
        assert wire["max_output_tokens"] == 4096
        assert wire["instructions"] == "S"

    def test_gpt_omits_temperature_adds_reasoning(self):
        # The cap clears _THINKING_CAP_FLOOR deliberately: the GPT entries
        # declare `default_on=True` (they reason unless told not to, per their
        # reference), so the starving-cap guard is ARMED for them and a cap
        # below the floor is refused — see the sibling test below.
        wire, decoding = _to_openai_wire(
            model="gpt-5.6-sol", system="S", messages=[], tools=None,
            tool_choice=None, max_tokens=4096, sampling={"temperature": 0.0})
        assert "temperature" not in wire
        assert wire["reasoning"] == {"effort": "medium"}
        assert decoding == {"max_output_tokens": 4096,
                            "reasoning": {"effort": "medium"}}

    def test_gpt_starving_cap_is_refused_before_the_wire_is_built(self):
        # The behaviour change that came with declaring the GPT thinking
        # surface: these models reason on a spec-less call, `max_output_tokens`
        # caps reasoning plus answer together, so a cap that cannot fit both is
        # refused here rather than billed and truncated.
        from direktoro.providers import ThinkingUnsupported

        with pytest.raises(ThinkingUnsupported, match="cannot fit a reasoning"):
            _to_openai_wire(
                model="gpt-5.6-sol", system="S", messages=[], tools=None,
                tool_choice=None, max_tokens=100, sampling=None)

    def test_message_blocks_translate_to_input_items(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": PNG_B64}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling a tool"},
                {"type": "tool_use", "id": "tu1", "name": "record_answer",
                 "input": {"answer": {"a": 1}}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": "{\"status\": \"ok\"}"},
            ]},
        ]
        items = _messages_to_input(messages)

        # user text + image become one message item.
        assert items[0]["role"] == "user"
        assert items[0]["content"][0] == {"type": "input_text",
                                          "text": "look at this"}
        assert items[0]["content"][1] == {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{PNG_B64}"}

        # assistant text becomes an output_text message; the tool_use becomes
        # a separate function_call item (arguments is a JSON string).
        assert items[1] == {"role": "assistant",
                            "content": [{"type": "output_text",
                                         "text": "calling a tool"}]}
        assert items[2]["type"] == "function_call"
        assert items[2]["call_id"] == "tu1"
        assert items[2]["name"] == "record_answer"
        assert json.loads(items[2]["arguments"]) == {"answer": {"a": 1}}

        # tool_result becomes a top-level function_call_output item.
        assert items[3] == {"type": "function_call_output",
                            "call_id": "tu1", "output": "{\"status\": \"ok\"}"}


# ---------------------------------------------------------------------------
# OpenAI response translation (Responses wire -> NormalisedResponse)
# ---------------------------------------------------------------------------

def _openai_adapter(provider="openai", base_url=OPENAI_BASE_URL):
    return OpenAIAdapter(client=None, provider=provider, base_url=base_url)


class TestOpenAIResponseTranslation:
    def test_tool_call_and_usage(self):
        raw = {
            "model": "gpt-5.6-sol-2026-07-09",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "call_1",
                 "name": "record_answer", "arguments": "{\"answer\": {\"x\": 1}}"},
            ],
            "usage": {"input_tokens": 1000, "output_tokens": 50,
                      "input_tokens_details": {"cached_tokens": 200}},
        }
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert len(resp.content) == 1
        block = resp.content[0]
        assert block.type == "tool_use"
        assert block.id == "call_1"
        assert block.name == "record_answer"
        assert block.input == {"answer": {"x": 1}}
        assert resp.stop_reason == "tool_use"
        assert resp.resolved_model == "gpt-5.6-sol-2026-07-09"
        # cached tokens are subtracted from full-price input.
        assert resp.usage.input_tokens == 800
        assert resp.usage.cache_read_input_tokens == 200
        assert resp.usage.cache_creation_input_tokens == 0
        assert resp.usage.output_tokens == 50

    def test_text_message_first_block_has_text(self):
        # A caller reading a text-only answer takes response.content[0].text, so
        # a message item must surface as a text block in first position.
        raw = {
            "model": "z-ai/glm-4.6v",
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "{\"ok\": true}"}]}],
            "usage": {"input_tokens": 500, "output_tokens": 10},
        }
        resp = _openai_adapter("openrouter", OPENROUTER_BASE_URL)._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.content[0].type == "text"
        assert resp.content[0].text == "{\"ok\": true}"
        # No cached-token field reported: cache read is zero (full price).
        assert resp.usage.input_tokens == 500
        assert resp.usage.cache_read_input_tokens == 0

    def test_incomplete_maps_to_max_tokens(self):
        raw = {"model": "gpt-5.6-sol", "status": "incomplete",
               "incomplete_details": {"reason": "max_output_tokens"},
               "output": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# OpenAI adapter end-to-end against a fake Responses client
# ---------------------------------------------------------------------------

class _FakeResponses:
    def __init__(self, raw, sink):
        self._raw = raw
        self._sink = sink

    def create(self, **kwargs):
        self._sink["wire"] = kwargs
        return SimpleNamespace(model_dump=lambda: self._raw)


class _FakeOpenAIClient:
    def __init__(self, raw, sink):
        self.responses = _FakeResponses(raw, sink)


class TestOpenAIAdapterRoundTrip:
    def test_gpt_call_round_trip(self):
        raw = {
            "model": "gpt-5.6-sol-2026",
            "status": "completed",
            "output": [{"type": "function_call", "call_id": "c1",
                        "name": "record_answer", "arguments": "{}"}],
            "usage": {"input_tokens": 300, "output_tokens": 20,
                      "input_tokens_details": {"cached_tokens": 100}},
        }
        sink = {}
        adapter = OpenAIAdapter(
            _FakeOpenAIClient(raw, sink), provider="openai",
            base_url=OPENAI_BASE_URL)
        resp = adapter.create_message(
            model="gpt-5.6-sol",
            system=[{"type": "text", "text": "SYS"}],
            messages=[{"role": "user",
                       "content": [{"type": "text", "text": "hi"}]}],
            tools=[{"name": "record_answer", "description": "d",
                    "input_schema": {"type": "object"}}],
            tool_choice={"type": "auto"},
            max_tokens=4096, sampling={"temperature": 0.0})

        # Wire request went out in Responses shape, temperature omitted.
        assert sink["wire"]["max_output_tokens"] == 4096
        assert sink["wire"]["instructions"] == "SYS"
        assert "temperature" not in sink["wire"]
        # Response normalised.
        assert resp.content[0].name == "record_answer"
        assert resp.resolved_model == "gpt-5.6-sol-2026"
        assert resp.usage.input_tokens == 200
        assert resp.usage.cache_read_input_tokens == 100
        assert resp.provider == "openai"
        # Canonical request preserved for the audit log; wire request kept too.
        assert resp.raw_request["model"] == "gpt-5.6-sol"
        assert resp.wire_request["model"] == "gpt-5.6-sol"
        assert resp.decoding_params["reasoning"] == {"effort": "medium"}

    def test_a_refused_control_is_absent_from_every_recorded_request(self):
        # `raw_request` is the canonical request AS SENT, so a control the model
        # refuses is missing from it exactly as it is missing from the wire and
        # from decoding_params. Recording the caller's raw ask instead would
        # leave one audit field claiming a temperature was set on a call that
        # never carried one.
        raw = {
            "model": "gpt-5.6-sol-2026",
            "status": "completed",
            "output": [{"type": "function_call", "call_id": "c1",
                        "name": "record_answer", "arguments": "{}"}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        sink = {}
        adapter = OpenAIAdapter(
            _FakeOpenAIClient(raw, sink), provider="openai",
            base_url=OPENAI_BASE_URL)
        resp = adapter.create_message(
            model="gpt-5.6-sol",
            system=[{"type": "text", "text": "SYS"}],
            messages=[{"role": "user",
                       "content": [{"type": "text", "text": "hi"}]}],
            max_tokens=4096, sampling={"temperature": 0.0})

        assert "temperature" not in resp.raw_request
        assert "temperature" not in resp.wire_request
        assert "temperature" not in resp.decoding_params
        assert "temperature" not in sink["wire"]


# ---------------------------------------------------------------------------
# Chat Completions request translation (GLM via Z.ai's OpenAI-compat endpoint)
# ---------------------------------------------------------------------------

class TestChatCompletionsRequestTranslation:
    def test_system_becomes_first_message(self):
        wire, _ = _to_chat_completions_wire(
            model="z-ai/glm-4.6v", system=[{"type": "text", "text": "SYS"}],
            messages=[], tools=None, tool_choice=None, max_tokens=4096,
            sampling={"temperature": 0.0})
        assert wire["messages"][0] == {"role": "system", "content": "SYS"}

    def test_output_cap_is_max_tokens_not_max_output_tokens(self):
        # The Chat Completions surface uses the classic `max_tokens` key.
        wire, decoding = _to_chat_completions_wire(
            model="z-ai/glm-4.6v", system="S", messages=[], tools=None,
            tool_choice=None, max_tokens=4096, sampling={"temperature": 0.0})
        assert wire["max_tokens"] == 4096
        assert "max_output_tokens" not in wire
        assert decoding == {"max_tokens": 4096, "temperature": 0.0}

    def test_tools_become_function_wrapper_shape(self):
        wire, _ = _to_chat_completions_wire(
            model="z-ai/glm-4.6v", system="S", messages=[],
            tools=[{"name": "record_answer", "description": "d",
                    "input_schema": {"type": "object", "properties": {}}}],
            tool_choice={"type": "auto"}, max_tokens=4096, sampling={"temperature": 0.0})
        assert wire["tools"][0] == {
            "type": "function",
            "function": {
                "name": "record_answer",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        assert wire["tool_choice"] == "auto"

    def test_glm_keeps_temperature_no_reasoning(self):
        wire, decoding = _to_chat_completions_wire(
            model="z-ai/glm-4.6v", system="S", messages=[], tools=None,
            tool_choice=None, max_tokens=4096, sampling={"temperature": 0.0})
        assert wire["temperature"] == 0.0
        assert "reasoning_effort" not in wire
        assert decoding == {"max_tokens": 4096, "temperature": 0.0}

    def test_gemini_36_omits_sampling_from_wire_and_decoding(self):
        # google/gemini-3.6-flash names every sampling control in
        # `rejects_sampling` (its Vertex endpoints list none of them), so a call
        # that specifies them sends none: they are absent from the wire request
        # AND from the recorded decoding params (honest omission — the same dict
        # feeds the fingerprint).
        wire, decoding = _to_chat_completions_wire(
            model="google/gemini-3.6-flash", system="S", messages=[], tools=None,
            tool_choice=None, max_tokens=4096,
            sampling={"temperature": 0.0, "top_p": 0.9, "top_k": 40})
        for name in ("temperature", "top_p", "top_k"):
            assert name not in wire, name
            assert name not in decoding, name
        assert decoding == {"max_tokens": 4096}

    def test_top_k_rides_extra_body_not_a_wire_kwarg(self):
        # `top_k` is a body field this surface reads but NOT a parameter the
        # openai SDK's `create` names, so a top-level `top_k=` is a TypeError
        # before the request leaves. It goes under `extra_body`, the SDK's
        # channel for body fields it does not name — and stays in `decoding`,
        # so it is still sent and still identity-bearing.
        wire, decoding = _to_chat_completions_wire(
            model="z-ai/glm-4.6v", system="S", messages=[], tools=None,
            tool_choice=None, max_tokens=4096,
            sampling={"temperature": 0.0, "top_k": 40})
        assert "top_k" not in wire
        assert wire["extra_body"] == {"top_k": 40}
        assert wire["temperature"] == 0.0      # a named parameter stays put
        assert decoding == {"max_tokens": 4096, "temperature": 0.0,
                            "top_k": 40}

    def test_no_top_k_means_no_extra_body_at_all(self):
        # An empty `extra_body` on a non-routed request would be a body key
        # nothing asked for.
        wire, _ = _to_chat_completions_wire(
            model="z-ai/glm-4.6v", system="S", messages=[], tools=None,
            tool_choice=None, max_tokens=4096, sampling={"temperature": 0.0})
        assert "extra_body" not in wire

    def test_tool_use_and_result_round_trip_through_messages(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": PNG_B64}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "calling a tool"},
                {"type": "tool_use", "id": "tu1", "name": "record_answer",
                 "input": {"answer": {"a": 1}}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1",
                 "content": "{\"status\": \"ok\"}"},
            ]},
        ]
        out = _messages_to_chat("S", messages)

        # system message first, then the user text+image parts.
        assert out[0] == {"role": "system", "content": "S"}
        assert out[1]["role"] == "user"
        assert out[1]["content"][0] == {"type": "text", "text": "look at this"}
        assert out[1]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}

        # assistant text + tool_use collapse into one message with tool_calls.
        assert out[2]["role"] == "assistant"
        assert out[2]["content"] == "calling a tool"
        call = out[2]["tool_calls"][0]
        assert call["id"] == "tu1"
        assert call["type"] == "function"
        assert call["function"]["name"] == "record_answer"
        assert json.loads(call["function"]["arguments"]) == {"answer": {"a": 1}}

        # tool_result becomes a role:tool message keyed by tool_call_id.
        assert out[3] == {"role": "tool", "tool_call_id": "tu1",
                          "content": "{\"status\": \"ok\"}"}


# ---------------------------------------------------------------------------
# Chat Completions adapter end-to-end against a fake chat client
# ---------------------------------------------------------------------------

_UNSENT = object()


class _FakeChatCompletions:
    """`client.chat.completions.create`, stubbed WITH THE REAL SIGNATURE.

    The openai SDK's `create` is keyword-only, names every parameter it takes,
    and defines no `**kwargs`, so a kwarg it does not name is a TypeError raised
    in the SDK before any request goes out. A `**kwargs` stub accepts every one
    of those and reports a happy wire, which makes an unsendable parameter
    invisible until a live, billable run — so this one names what the adapter
    can send (the decoding params it resolves for this wire, the tool
    parameters, and `extra_body`, mirroring `openai.OpenAI().chat.completions
    .create`) and accepts nothing else.

    `sink["wire"]` records only the parameters actually passed, so a test can
    assert a control's ABSENCE from the request as well as its value."""

    def __init__(self, raw, sink):
        self._raw = raw
        self._sink = sink

    def create(self, *, model, messages, max_tokens=_UNSENT,
               temperature=_UNSENT, top_p=_UNSENT, reasoning_effort=_UNSENT,
               tools=_UNSENT, tool_choice=_UNSENT, extra_body=_UNSENT):
        passed = {
            "model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p,
            "reasoning_effort": reasoning_effort, "tools": tools,
            "tool_choice": tool_choice, "extra_body": extra_body,
        }
        self._sink["wire"] = {name: value for name, value in passed.items()
                              if value is not _UNSENT}
        return SimpleNamespace(model_dump=lambda: self._raw)


class _FakeChatClient:
    """Mirrors the OpenAI SDK surface the adapter reaches for Chat
    Completions: `client.chat.completions.create(**wire)`."""

    def __init__(self, raw, sink):
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(raw, sink))


def _routed_chat_response(model, served_provider, *, cached=None,
                          cost=0.0001, text=None, finish_reason=None):
    """A routed OpenRouter chat completion fixture: a top-level generation `id`,
    `provider` attribution, and `usage.cost`, plus either a forced tool call
    (default) or a plain text message. `served_provider` is the display name
    OpenRouter attributes the call to (checked against the Route pin)."""
    usage = {"prompt_tokens": 1000, "completion_tokens": 50,
             "total_tokens": 1050, "cost": cost}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    if text is not None:
        message = {"role": "assistant", "content": text}
        finish = finish_reason or "stop"
    else:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "record_answer",
                         "arguments": "{\"answer\": {\"x\": 1}}"}}]}
        finish = finish_reason or "tool_calls"
    return {
        "id": "gen-test-123",
        "model": model,
        "provider": served_provider,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": usage,
    }


def _routed_adapter(raw, sink):
    """An OpenAIAdapter wired for the gateway, as build_adapter constructs it
    for a routed entry (provider "openrouter", the OpenRouter base URL)."""
    return OpenAIAdapter(_FakeChatClient(raw, sink), provider="openrouter",
                         base_url=OPENROUTER_BASE_URL)


class TestTheChatStubMatchesTheInstalledSDK:
    """The stub above only means anything if its signature is the real one.

    Everything the Chat Completions path is tested against goes through
    `_FakeChatCompletions.create`, so a stub that accepts more than the SDK does
    is a seam that hides unsendable parameters until a live run pays for the
    discovery. These hold it to the SDK that is actually installed — the same
    check the dependency floors get in tests/test_public_api.py, for the same
    reason: a claim about the SDK is worth making against the SDK."""

    def _sdk_parameters(self):
        openai = pytest.importorskip("openai")
        import inspect

        client = openai.OpenAI(api_key="not-a-real-key")
        return inspect.signature(client.chat.completions.create).parameters

    def test_the_stub_names_nothing_the_sdk_does_not(self):
        import inspect

        sdk = self._sdk_parameters()
        stub = inspect.signature(_FakeChatCompletions.create).parameters
        extra = [name for name in stub
                 if name not in ("self",) and name not in sdk]
        assert extra == [], (
            f"the stub accepts {extra}, which the installed openai SDK's "
            f"chat.completions.create does not; a test passing through it "
            f"proves nothing about what can be sent.")

    def test_the_sdk_takes_no_top_k_but_does_take_extra_body(self):
        # The wire fact the translation is built on: `top_k` is not a parameter
        # of this endpoint's method (and it takes no **kwargs, so passing one is
        # a TypeError), while `extra_body` is the SDK's own channel for body
        # fields it does not name.
        sdk = self._sdk_parameters()
        assert "top_k" not in sdk
        assert not any(p.kind is p.VAR_KEYWORD for p in sdk.values())
        assert "extra_body" in sdk


class TestRoutedChatAdapter:
    """A routed (OpenRouter) chat completion: the adapter emits the provider
    routing object + usage.include, captures the generation id / served upstream
    / reported cost, and runs the pin assertion before returning."""

    def test_tool_call_emits_provider_object_and_captures_provenance(self):
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI", cached=200,
                                  cost=0.00042), sink)
        resp = adapter.create_message(
            model="z-ai/glm-4.6v",
            system=[{"type": "text", "text": "SYS"}],
            messages=[{"role": "user",
                       "content": [{"type": "text", "text": "hi"}]}],
            tools=[{"name": "record_answer", "description": "d",
                    "input_schema": {"type": "object"}}],
            tool_choice={"type": "auto"}, max_tokens=4096, sampling={"temperature": 0.0})

        # OpenRouter's provider routing object + usage.include ride extra_body.
        extra = sink["wire"]["extra_body"]
        assert extra["provider"]["order"] == ["z-ai"]
        assert extra["provider"]["allow_fallbacks"] is False
        assert extra["provider"]["require_parameters"] is True
        assert extra["provider"]["data_collection"] == "deny"
        assert extra["provider"]["zdr"] is True
        assert extra["provider"]["quantizations"] == ["fp8"]
        assert extra["usage"] == {"include": True}
        # The chat body itself is unchanged.
        assert sink["wire"]["max_tokens"] == 4096
        assert sink["wire"]["messages"][0] == {"role": "system",
                                               "content": "SYS"}
        # Response normalised: tool_use block + cached-token accounting.
        block = resp.content[0]
        assert block.type == "tool_use"
        assert block.name == "record_answer"
        assert resp.stop_reason == "tool_use"
        assert resp.usage.input_tokens == 800       # 1000 - 200 cached
        assert resp.usage.cache_read_input_tokens == 200
        # Routing provenance captured; cost taken from the response.
        assert resp.provider == "openrouter"
        assert resp.served_provider == "Z.AI"
        assert resp.generation_id == "gen-test-123"
        assert resp.reported_cost == pytest.approx(0.00042)
        # And a control this model DOES take is recorded on the canonical
        # request: `raw_request` is what was sent, not a blanket omission.
        assert resp.raw_request["temperature"] == 0.0

    def test_top_k_reaches_the_call_under_extra_body(self):
        # End to end against the stub whose signature is the SDK's: a routed
        # model that accepts top_k produces a call carrying it in the request
        # BODY, beside the routing extras rather than instead of them, and no
        # `top_k` kwarg — which is the form the SDK would reject.
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI"), sink)
        resp = adapter.create_message(
            model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0, "top_k": 40})

        assert "top_k" not in sink["wire"]
        extra = sink["wire"]["extra_body"]
        assert extra["top_k"] == 40
        # Merged with the routing extras, which still went out intact.
        assert extra["provider"]["order"] == ["z-ai"]
        assert extra["usage"] == {"include": True}
        # Sent means recorded: identity and the canonical request both carry it.
        assert resp.decoding_params["top_k"] == 40
        assert resp.raw_request["top_k"] == 40
        assert resp.wire_request["extra_body"]["top_k"] == 40

    def test_every_routed_model_that_takes_top_k_can_send_it(self):
        # Asserted over the table rather than one id, so an entry added without
        # `top_k` in its rejects_sampling is covered on arrival: whatever the
        # resolver keeps has to be sendable through this adapter, and the stub's
        # signature is what makes "sendable" mean the SDK's own answer.
        routed = [model_id for model_id, info in MODEL_REGISTRY.items()
                  if info.route is not None
                  and info.wire_api == WIRE_CHAT_COMPLETIONS
                  and "top_k" not in info.rejects_sampling]
        assert routed, "no routed entry accepts top_k"
        for model_id in routed:
            sink = {}
            adapter = _routed_adapter(
                _routed_chat_response(
                    model_id, model_info(model_id).route.upstream[0]), sink)
            adapter.create_message(
                model=model_id, system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"top_k": 40})
            assert sink["wire"]["extra_body"]["top_k"] == 40, model_id

    def test_uncached_usage_full_price(self):
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI", cached=None), sink)
        resp = adapter.create_message(
            model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        # No prompt_tokens_details: cached is zero, all input at full price.
        assert resp.usage.input_tokens == 1000
        assert resp.usage.cache_read_input_tokens == 0

    def test_text_response_first_block_has_text(self):
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI",
                                  text="{\"ok\": true}"), sink)
        resp = adapter.create_message(
            model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert resp.content[0].type == "text"
        assert resp.content[0].text == "{\"ok\": true}"
        assert resp.stop_reason == "end_turn"
        assert resp.served_provider == "Z.AI"

    def test_length_finish_maps_to_max_tokens(self):
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI", text="...",
                                  finish_reason="length"), sink)
        resp = adapter.create_message(
            model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert resp.stop_reason == "max_tokens"


class TestRoutedPinAssertion:
    """The pin assertion runs on every routed response before it is returned: a
    call served by an upstream other than the pinned one, or with no attribution
    to verify, raises ProviderRouteMismatch so nothing partial is returned."""

    def test_wrong_upstream_raises(self):
        sink = {}
        # z-ai/glm-4.6v pins upstream to z-ai; a Novita attribution mismatches.
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Novita"), sink)
        with pytest.raises(ProviderRouteMismatch, match="Novita"):
            adapter.create_message(
                model="z-ai/glm-4.6v", system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})

    def test_absent_attribution_raises(self):
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", None), sink)
        with pytest.raises(ProviderRouteMismatch, match="attribution"):
            adapter.create_message(
                model="z-ai/glm-4.6v", system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})


class TestRoutedImageAndTool:
    """Per routed model: an image part plus a tool schema serialise into the
    Chat Completions wire, the tool call round-trips, and the served upstream +
    reported cost are captured. Offline against fixtures (no gateway)."""

    @pytest.mark.parametrize(
        "model, served, order",
        [
            ("z-ai/glm-5v-turbo", "Z.AI", ["z-ai"]),
            # Served by the SECOND allowed upstream — proves the pin assertion
            # accepts any member of the declared set, not just the first.
            ("qwen/qwen3-vl-235b-a22b-instruct", "Parasail",
             ["venice", "parasail"]),
        ],
    )
    def test_image_and_tool_serialise_and_round_trip(
            self, model, served, order):
        info = model_info(model)
        assert info.base_url == OPENROUTER_BASE_URL  # registry routing sanity
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response(model, served, cost=0.001), sink)
        resp = adapter.create_message(
            model=model,
            system=[{"type": "text", "text": "SYS"}],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "see figure"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": PNG_B64}},
            ]}],
            tools=[{"name": "record_answer", "description": "d",
                    "input_schema": {"type": "object", "properties": {}}}],
            tool_choice={"type": "auto"}, max_tokens=4096, sampling={"temperature": 0.0})

        wire = sink["wire"]
        # The image serialised as a Chat Completions image_url data URI.
        user_parts = wire["messages"][1]["content"]
        assert user_parts[0] == {"type": "text", "text": "see figure"}
        assert user_parts[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}
        assert wire["tool_choice"] == "auto"
        assert wire["model"] == model
        # The Route's upstream pin went out on the request.
        assert wire["extra_body"]["provider"]["order"] == order
        # The tool call round-trips; provenance captured.
        assert resp.content[0].type == "tool_use"
        assert resp.content[0].input == {"answer": {"x": 1}}
        assert resp.stop_reason == "tool_use"
        assert resp.provider == "openrouter"
        assert resp.served_provider == served
        assert resp.reported_cost == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Anthropic adapter against a stubbed streaming client
# ---------------------------------------------------------------------------

class _AnthStream:
    def __init__(self, resp):
        self._resp = resp
        self.text_stream = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._resp


class _AnthClient:
    def __init__(self, resp, sink):
        self._resp = resp
        self._sink = sink
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self._sink["wire"] = kwargs
        return _AnthStream(self._resp)


def _anth_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2,
                              cache_read_input_tokens=3,
                              cache_creation_input_tokens=1),
        model="claude-opus-4-8-20260601",
        stop_reason="end_turn")


class TestAnthropicAdapter:
    def test_opus_omits_temperature(self):
        sink = {}
        adapter = AnthropicAdapter(_AnthClient(_anth_response(), sink))
        resp = adapter.create_message(
            model="claude-opus-4-8",
            system=[{"type": "text", "text": "S"}],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert "temperature" not in sink["wire"]
        assert resp.decoding_params == {"max_tokens": 4096}
        assert resp.resolved_model == "claude-opus-4-8-20260601"
        # Usage passes through unchanged (Anthropic semantics are canonical).
        assert resp.usage.input_tokens == 10
        assert resp.usage.cache_read_input_tokens == 3
        assert resp.usage.cache_creation_input_tokens == 1
        # The canonical request IS the Anthropic wire, recorded under both
        # names so an audit path reads wire_request whoever served the call.
        assert resp.wire_request is resp.raw_request

    def test_sonnet_sends_temperature(self):
        sink = {}
        adapter = AnthropicAdapter(_AnthClient(_anth_response(), sink))
        resp = adapter.create_message(
            model="claude-sonnet-4-6", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert sink["wire"]["temperature"] == 0.0
        assert resp.decoding_params == {"max_tokens": 4096, "temperature": 0.0}

    def test_error_is_translated(self):
        # One of the two tests that genuinely needs the Anthropic SDK
        # installed: it constructs a real `anthropic.RateLimitError` to check
        # the translation. SKIPPED rather than failed when the SDK is absent,
        # so the suite stays runnable wherever `import direktoro` is — which is
        # the whole point of the lazy imports, and is what you get by unpacking
        # the sdist and installing nothing but pytest.
        anthropic = pytest.importorskip(
            "anthropic",
            reason="constructs a real anthropic.RateLimitError")

        class _BoomClient:
            def __init__(self):
                self.messages = SimpleNamespace(stream=self._boom)

            def _boom(self, **kwargs):
                raise anthropic.RateLimitError(
                    "429", response=SimpleNamespace(
                        status_code=429, headers={}, request=None),
                    body=None)

        adapter = AnthropicAdapter(_BoomClient())
        with pytest.raises(ProviderRateLimitError):
            adapter.create_message(
                model="claude-sonnet-4-6", system="S",
                messages=[], max_tokens=10, sampling={"temperature": 0.0})


class TestRoutedReceiptGuards:
    """A routed response missing its audit receipt (generation id) or its
    authoritative cost (usage.cost) is refused loudly, like a pin mismatch.
    Returning it with those fields None would record a call that cost nothing
    and cannot be looked up at the gateway, which is worse than not recording
    it at all."""

    def test_missing_cost_raises(self):
        sink = {}
        adapter = _routed_adapter(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI", cost=None), sink)
        with pytest.raises(ProviderError, match="usage.cost"):
            adapter.create_message(
                model="z-ai/glm-4.6v", system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})

    def test_missing_generation_id_raises(self):
        sink = {}
        raw = _routed_chat_response("z-ai/glm-4.6v", "Z.AI")
        raw["id"] = None
        adapter = _routed_adapter(raw, sink)
        with pytest.raises(ProviderError, match="generation id"):
            adapter.create_message(
                model="z-ai/glm-4.6v", system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})


class TestARefusedRoutedResponseIsStillBilled:
    """Every routing refusal happens AFTER the gateway served and billed the
    call, so each carries the response it refuses on `exception.response`.

    The tokens were spent whatever this layer thinks of the receipt. Raising the
    bare exception discards the only record of that spend, leaving a consumer
    with a refusal it cannot ledger — so the `NormalisedResponse` as it stood,
    usage intact and routing fields as far as they got, rides on the
    exception."""

    def _refusal(self, raw, expected=ProviderError):
        with pytest.raises(expected) as caught:
            _routed_adapter(raw, {}).create_message(
                model="z-ai/glm-4.6v", system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})
        return caught.value

    def test_a_pin_mismatch_carries_the_billed_response(self):
        error = self._refusal(
            _routed_chat_response("z-ai/glm-4.6v", "Novita", cached=200),
            ProviderRouteMismatch)
        assert error.response is not None
        # The spend is readable: full-price input, cache reads, output.
        assert error.response.usage.input_tokens == 800
        assert error.response.usage.cache_read_input_tokens == 200
        assert error.response.usage.output_tokens == 50
        # Routing fields got no further than the check that failed.
        assert error.response.served_provider is None
        assert error.response.generation_id is None

    def test_a_missing_receipt_carries_the_billed_response(self):
        raw = _routed_chat_response("z-ai/glm-4.6v", "Z.AI")
        raw["id"] = None
        error = self._refusal(raw)
        assert error.response.usage.input_tokens == 1000
        assert error.response.usage.output_tokens == 50
        # The pin held before the receipt failed, so that much is on the record.
        assert error.response.served_provider == "Z.AI"

    def test_a_missing_cost_carries_the_billed_response_and_its_receipt(self):
        # The most useful of the three: the call is unpriceable from the
        # gateway's own figure, so the tokens AND the generation id are what a
        # consumer needs to ledger it (or to look the charge up later).
        error = self._refusal(
            _routed_chat_response("z-ai/glm-4.6v", "Z.AI", cost=None))
        assert error.response.usage.input_tokens == 1000
        assert error.response.generation_id == "gen-test-123"
        assert error.response.served_provider == "Z.AI"
        assert error.response.reported_cost is None

    def test_a_pin_mismatch_is_caught_as_a_provider_failure(self):
        # ProviderRouteMismatch is a ProviderError, so one `except
        # ProviderError` around a call catches a broken pin along with every
        # other provider failure instead of letting it escape unhandled.
        with pytest.raises(ProviderError):
            _routed_adapter(
                _routed_chat_response("z-ai/glm-4.6v", "Novita"), {}
            ).create_message(
                model="z-ai/glm-4.6v", system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})

    def test_an_error_with_no_response_carries_none(self):
        # The attribute is not a promise that there IS billed material: a
        # failure raised INSTEAD of a response has none, and reads as None.
        assert ProviderError("nothing was served").response is None


# ---------------------------------------------------------------------------
# A gateway failure delivered in the body of a 200
# ---------------------------------------------------------------------------

def _error_body(code, message):
    """A gateway error body: HTTP 200, every response field null, `error` set.

    The shape below is the one OpenRouter actually sent (recorded 2026-08-18 on
    google/gemini-3.7-flash) — the null fields matter as much as the error,
    because they are why the body reads as a well-formed empty response to
    everything downstream of it."""
    return {"id": None, "choices": None, "created": None, "model": None,
            "object": None, "moderation": None, "service_tier": None,
            "system_fingerprint": None, "usage": None,
            "error": {"message": message, "code": code}}


class _SequencedChatCompletions(_FakeChatCompletions):
    """The stub above — its real signature included, since every call still goes
    through it — answering a different body per call, so a retry can be watched
    landing on the second one."""

    def __init__(self, raws, sink):
        super().__init__(raws[0], sink)
        self._raws = list(raws)

    def create(self, **wire):
        if self._raws:
            self._raw = self._raws.pop(0)
        return super().create(**wire)


class TestAGatewayErrorBodyIsNotAResponse:
    """OpenRouter reports some upstream failures as HTTP 200 with `error` in the
    body. The SDK does not raise, so the transport translator never sees it, and
    the body carries no content, no usage, and no attribution.

    Read as a response it is a transient failure wearing the mask of a permanent
    one: the pin assertion is the first thing to reach it, finds no served
    provider — because the body has nothing at all in it — and refuses the call
    as ProviderRouteMismatch, which is deliberately NOT retryable. A 524 the
    backoff ladder would very likely have survived is then reported as a broken
    pin, and a consumer that trusts the label degrades the answer for good. The
    body is classified before anything reads it as a response, so each failure
    arrives as the class it actually is."""

    def _refused(self, raw, expected, *, model="z-ai/glm-4.6v"):
        with pytest.raises(expected) as caught:
            _routed_adapter(raw, {}).create_message(
                model=model, system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})
        return caught.value

    def test_an_origin_timeout_is_retryable_not_a_pin_violation(self):
        # The observed case, verbatim: Cloudflare 524 relayed as code 504.
        error = self._refused(_error_body(504, "error code: 524\n"),
                              ProviderRetryableError)
        assert error.status_code == 504
        assert "error code: 524" in str(error)
        # Nothing here is about routing, so nothing here says so.
        assert not isinstance(error, ProviderRouteMismatch)
        assert "pin" not in str(error)

    def test_a_rate_limit_in_the_body_is_a_rate_limit(self):
        # The sharpest case: the single most retryable condition there is,
        # arriving on the path that used to call it a routing violation.
        error = self._refused(
            _error_body(429, "Provider returned error"), ProviderRateLimitError)
        assert "Provider returned error" in str(error)

    def test_a_non_transient_code_stops_the_call_with_the_gateway_message(self):
        error = self._refused(_error_body(400, "invalid model id"),
                              ProviderError)
        assert type(error) is ProviderError
        assert "invalid model id" in str(error)

    def test_a_code_that_is_not_a_status_is_not_guessed_at(self):
        # OpenAI-style string codes carry no HTTP status to classify. Retrying
        # on a guess would pay for the same refusal four more times.
        error = self._refused(_error_body("insufficient_quota", "no credit"),
                              ProviderError)
        assert type(error) is ProviderError
        assert "no credit" in str(error)

    def test_a_string_where_the_object_belongs_is_still_read(self):
        raw = _error_body(500, "x")
        raw["error"] = "upstream exploded"
        error = self._refused(raw, ProviderError)
        assert "upstream exploded" in str(error)

    def test_the_status_is_read_where_a_host_names_it_that_way(self):
        # An OpenAI-compatible host may carry the HTTP status as `status` and
        # send `code` as null rather than omitting it.
        raw = _error_body(None, "upstream unavailable")
        raw["error"]["status"] = 503
        error = self._refused(raw, ProviderRetryableError)
        assert error.status_code == 503

    def test_an_error_beside_content_is_still_refused_on_this_wire(self):
        # THE ONE DELIBERATE RISK HERE, pinned so it stays deliberate. `error`
        # is no part of the Chat Completions schema, so a body carrying one is
        # not a completion and is refused whatever else it holds — where the
        # Responses translator, whose `error` IS a schema field, reads an
        # answer beside it as an answer. If a gateway ever pairs real content
        # with an error, this refusal discards a billed answer and the rule
        # should change; 58 error bodies recorded against this path (2026-08-18,
        # OpenRouter -> Vertex) carry `choices: null` and nothing else, so it
        # has not fired, and the evidence for the rule is that corpus rather
        # than a guess.
        raw = _error_body(503, "upstream unavailable")
        raw["choices"] = [{"message": {"role": "assistant", "content": "half"},
                           "finish_reason": "error"}]
        error = self._refused(raw, ProviderRetryableError)
        assert error.status_code == 503

    def test_the_refusal_carries_no_billed_response(self):
        # Unlike every routing refusal, which is about a call the gateway
        # already billed: this one is a call that reached the provider and got
        # an error back, and `usage` is null in the body to prove it. There is
        # nothing to ledger, so nothing rides on the exception.
        error = self._refused(_error_body(504, "error code: 524\n"),
                              ProviderRetryableError)
        assert error.response is None

    def test_a_transient_body_reaches_the_retry_ladder(self):
        # The whole point, end to end: the class is what the ladder reads, so
        # the call that used to be refused now waits and gets its answer.
        sink = {}
        client = _FakeChatClient(None, sink)
        client.chat.completions = _SequencedChatCompletions(
            [_error_body(504, "error code: 524\n"),
             _routed_chat_response("z-ai/glm-4.6v", "Z.AI",
                                   text="{\"ok\": true}")], sink)
        adapter = OpenAIAdapter(client, provider="openrouter",
                                base_url=OPENROUTER_BASE_URL)
        slept = []
        resp = create_message_with_retry(
            adapter, _sleep=slept.append, model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})

        assert resp.content[0].text == "{\"ok\": true}"
        assert resp.served_provider == "Z.AI"
        assert slept == [RETRY_BACKOFF_SECONDS[0]]

    def test_a_response_that_is_a_response_is_untouched(self):
        # The check must not fire on an ordinary completion, including one whose
        # body carries an explicit `error: null`.
        raw = _routed_chat_response("z-ai/glm-4.6v", "Z.AI", text="{\"ok\": 1}")
        raw["error"] = None
        resp = _routed_adapter(raw, {}).create_message(
            model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert resp.content[0].text == "{\"ok\": 1}"
        assert resp.served_provider == "Z.AI"

    def test_an_unrouted_call_is_not_left_with_a_silent_empty_completion(
            self, monkeypatch):
        # `_attach_routing` is skipped when the entry carries no Route, so on
        # that path an error body raised NOTHING before: it returned a
        # successful-looking response with no content and zero tokens, which
        # reads downstream as "the model produced no answer". The check runs
        # before that branch, so both paths refuse it. No registry entry pairs
        # Chat Completions with no Route today; this holds the behaviour for the
        # one that does.
        import dataclasses

        unrouted = dataclasses.replace(model_info("z-ai/glm-4.6v"), route=None)
        monkeypatch.setattr("direktoro.providers.model_info",
                            lambda model_id: unrouted)
        # The patch has to be doing something, or this passes on the routed
        # path and proves nothing: an unattributed response that the pin WOULD
        # have refused comes back normally, which is the unrouted path and only
        # the unrouted path.
        unattributed = _routed_chat_response("z-ai/glm-4.6v", None, text="hi")
        assert _routed_adapter(unattributed, {}).create_message(
            model="z-ai/glm-4.6v", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0}).content[0].text

        error = self._refused(_error_body(504, "error code: 524\n"),
                              ProviderRetryableError)
        assert error.status_code == 504


# ---------------------------------------------------------------------------
# A Responses call that did not complete
# ---------------------------------------------------------------------------

def _failed_response(code, message, *, status="failed"):
    """A Responses object that failed: a status other than `completed`, an
    `error`, and — the part that does the damage — an empty `output` that reads
    as a finished turn with nothing in it."""
    return {"id": "resp_test", "object": "response", "status": status,
            "model": "gpt-5.6-terra-2026", "output": [], "usage": None,
            "incomplete_details": None,
            "error": {"code": code, "message": message}}


class _SequencedResponses(_FakeResponses):
    """The Responses stub answering a different object per call, so a retry can
    be watched landing on the second one."""

    def __init__(self, raws, sink):
        super().__init__(raws[0], sink)
        self._raws = list(raws)

    def create(self, **wire):
        if self._raws:
            self._raw = self._raws.pop(0)
        return super().create(**wire)


class TestAFailedResponsesCallIsNotAnEmptyAnswer:
    """The Responses path has the same failure-in-the-body shape as the gateway
    error body above, and its silence is worse.

    `_from_wire` reads `output`, `usage`, `model`, `status` and
    `incomplete_details` and never `error`, so a `status: "failed"` object
    normalises to empty content under `end_turn` — a call that reads downstream
    as a model that finished and said nothing. Nothing raises, nothing is
    logged, and a consumer for which "nothing further to add" is a legitimate
    answer records the call as having happened. These hold the two statuses that
    carry an answer apart from every status that does not."""

    def _refused(self, raw, expected, *, model="gpt-5.6-terra"):
        adapter = OpenAIAdapter(_FakeOpenAIClient(raw, {}), provider="openai",
                                base_url=OPENAI_BASE_URL)
        with pytest.raises(expected) as caught:
            adapter.create_message(
                model=model, system="S",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4096, sampling={"temperature": 0.0})
        return caught.value

    def test_a_server_error_is_retryable(self):
        error = self._refused(_failed_response("server_error", "The model "
                                               "failed to generate a response"),
                              ProviderRetryableError)
        # No HTTP status to record: the failing call never reached one, because
        # the transport answered 200 and carried this object.
        assert error.status_code is None
        assert "failed to generate" in str(error)

    def test_a_rate_limit_is_a_rate_limit(self):
        error = self._refused(
            _failed_response("rate_limit_exceeded", "Rate limit reached"),
            ProviderRateLimitError)
        assert "Rate limit reached" in str(error)

    def test_a_request_side_code_is_not_retried(self):
        # `invalid_prompt` and the image-validation family are about the request
        # as sent. Four more attempts would fail identically and cost the wait.
        error = self._refused(
            _failed_response("invalid_prompt", "Your prompt was flagged"),
            ProviderError)
        assert type(error) is ProviderError
        assert "invalid_prompt" in str(error)

    def test_a_status_with_no_answer_is_refused_on_the_status_alone(self):
        # `cancelled` (and a still-`queued` object) carries no error to read and
        # no answer either.
        raw = _failed_response("server_error", "x", status="cancelled")
        del raw["error"]
        error = self._refused(raw, ProviderError)
        assert "cancelled" in str(error)

    def test_the_refusal_carries_no_billed_response(self):
        error = self._refused(_failed_response("server_error", "boom"),
                              ProviderRetryableError)
        assert error.response is None

    def test_an_answer_beside_a_stale_error_is_still_an_answer(self):
        # `error` is a documented field of EVERY Responses object, null on
        # success, so its presence alone does not mean the body is not a
        # response — unlike the Chat Completions error body, where `error` is
        # no part of the schema. A host that completes with output and leaves
        # a non-null error behind has still answered, and discarding a billed
        # answer over a contradictory field is worse than any refusal.
        raw = {"model": "gpt-5.6-terra-2026", "status": "completed",
               "error": {"code": "server_error", "message": "stale"},
               "output": [{"type": "message", "content": [
                   {"type": "output_text", "text": "answered"}]}],
               "usage": {"input_tokens": 5, "output_tokens": 1}}
        adapter = OpenAIAdapter(_FakeOpenAIClient(raw, {}), provider="openai",
                                base_url=OPENAI_BASE_URL)
        resp = adapter.create_message(
            model="gpt-5.6-terra", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert resp.content[0].text == "answered"

    def test_an_error_with_nothing_beside_it_is_refused_on_any_status(self):
        # The other half of the same rule: an error and NO output is the silent
        # empty answer this whole class exists to stop, whatever the status
        # claims.
        raw = {"model": "gpt-5.6-terra-2026", "status": "completed",
               "error": {"code": "server_error", "message": "nothing came"},
               "output": [], "usage": None}
        error = self._refused(raw, ProviderRetryableError)
        assert "nothing came" in str(error)

    def test_an_unreadable_status_outranks_an_answer_beside_it(self):
        # A `failed` object carrying output is contradictory; the status is the
        # authoritative field on this wire, so it decides.
        raw = _failed_response("server_error", "failed mid-flight")
        raw["output"] = [{"type": "message", "content": [
            {"type": "output_text", "text": "partial"}]}]
        error = self._refused(raw, ProviderRetryableError)
        assert "failed mid-flight" in str(error)

    def test_a_transient_failure_reaches_the_retry_ladder(self):
        # End to end on the leg a reviewer runs on: the failed object is
        # classified, the ladder waits, and the retry returns the answer that
        # would otherwise have been recorded as an empty one.
        sink = {}
        client = _FakeOpenAIClient(None, sink)
        client.responses = _SequencedResponses(
            [_failed_response("server_error", "The model failed to generate a "
                              "response"),
             {"model": "gpt-5.6-terra-2026", "status": "completed",
              "output": [{"type": "message", "content": [
                  {"type": "output_text", "text": "{\"verdict\": \"ok\"}"}]}],
              "usage": {"input_tokens": 500, "output_tokens": 12}}], sink)
        adapter = OpenAIAdapter(client, provider="openai",
                                base_url=OPENAI_BASE_URL)
        slept = []
        resp = create_message_with_retry(
            adapter, _sleep=slept.append, model="gpt-5.6-terra", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})

        assert resp.content[0].text == "{\"verdict\": \"ok\"}"
        assert resp.stop_reason == "end_turn"
        assert slept == [RETRY_BACKOFF_SECONDS[0]]

    @pytest.mark.parametrize("status", ["completed", "incomplete"])
    def test_the_two_statuses_that_carry_an_answer_are_untouched(self, status):
        raw = {"model": "gpt-5.6-terra-2026", "status": status,
               "error": None, "incomplete_details": {"reason":
                                                     "max_output_tokens"},
               "output": [{"type": "message", "content": [
                   {"type": "output_text", "text": "{\"verdict\": \"ok\"}"}]}],
               "usage": {"input_tokens": 500, "output_tokens": 12}}
        adapter = OpenAIAdapter(_FakeOpenAIClient(raw, {}), provider="openai",
                                base_url=OPENAI_BASE_URL)
        resp = adapter.create_message(
            model="gpt-5.6-terra", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert resp.content[0].text == "{\"verdict\": \"ok\"}"

    def test_a_response_with_no_status_at_all_is_left_alone(self):
        # `_openai_stop_reason` has always read an absent status as an ordinary
        # turn, and a host that omits the field is not failing. This check is
        # about statuses that SAY something, not about requiring one.
        raw = {"model": "gpt-5.6-terra-2026",
               "output": [{"type": "message", "content": [
                   {"type": "output_text", "text": "answered"}]}],
               "usage": {"input_tokens": 5, "output_tokens": 1}}
        adapter = OpenAIAdapter(_FakeOpenAIClient(raw, {}), provider="openai",
                                base_url=OPENAI_BASE_URL)
        resp = adapter.create_message(
            model="gpt-5.6-terra", system="S",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096, sampling={"temperature": 0.0})
        assert resp.content[0].text == "answered"
        assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Anthropic cache-write TTL split
# ---------------------------------------------------------------------------

def _anth_usage(**kw):
    """An SDK-shaped usage object. `cache_creation` is the nested per-TTL split
    Anthropic reports alongside the cache-write total; pass it as a mapping and
    it becomes the attribute-access object the SDK returns."""
    split = kw.pop("cache_creation", None)
    base = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=0,
                cache_creation_input_tokens=0)
    base.update(kw)
    return SimpleNamespace(
        cache_creation=None if split is None else SimpleNamespace(**split),
        **base)


class TestAnthropicCacheWriteTTLSplit:
    """The two cache-write tiers bill at different multiples of the base input
    rate (5-minute at 1.25x, 1-hour at 2x), so the response must carry the SPLIT
    and not only the sum. Costing an hour-TTL write at the five-minute rate
    yields 1.25/2 of the true charge — a total that is quietly too low, which is
    the one failure this layer must never produce."""

    def test_the_split_is_carried_alongside_the_total(self):
        sink = {}
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            usage=_anth_usage(
                input_tokens=10, output_tokens=2,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=1400,
                cache_creation={"ephemeral_5m_input_tokens": 400,
                                "ephemeral_1h_input_tokens": 1000}),
            model="claude-opus-4-8-20260601", stop_reason="end_turn")
        adapter = AnthropicAdapter(_AnthClient(response, sink))
        resp = adapter.create_message(
            model="claude-opus-4-8", system="S",
            messages=[{"role": "user", "content": "hi"}], max_tokens=4096)

        assert resp.usage.cache_creation_input_tokens == 1400
        assert resp.usage.cache_creation_5m_input_tokens == 400
        assert resp.usage.cache_creation_1h_input_tokens == 1000
        # The two tiers account for the total: a caller pricing them separately
        # prices every cache-write token exactly once.
        assert (resp.usage.cache_creation_5m_input_tokens
                + resp.usage.cache_creation_1h_input_tokens
                == resp.usage.cache_creation_input_tokens)

    def test_an_absent_split_leaves_both_tiers_zero(self):
        # `usage.cache_creation` is a nested object a response need not carry.
        # Its absence must not crash and must not invent a tier: both counters
        # read zero, meaning "no split reported", and the total still stands.
        sink = {}
        response = SimpleNamespace(
            content=[], usage=_anth_usage(cache_creation_input_tokens=900),
            model="claude-opus-4-8", stop_reason="end_turn")
        adapter = AnthropicAdapter(_AnthClient(response, sink))
        resp = adapter.create_message(
            model="claude-opus-4-8", system="S", messages=[], max_tokens=10)

        assert resp.usage.cache_creation_input_tokens == 900
        assert resp.usage.cache_creation_5m_input_tokens == 0
        assert resp.usage.cache_creation_1h_input_tokens == 0

    def test_the_new_counters_are_appended_last(self):
        # NormalisedUsage is public and constructed POSITIONALLY downstream, so
        # a field inserted mid-record silently rebinds every argument after it
        # (an output count landing in a cache counter, and a wrong cost with no
        # error). Pin the order so that edit fails here instead.
        import dataclasses

        assert [f.name for f in dataclasses.fields(NormalisedUsage)] == [
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cache_creation_5m_input_tokens",
            "cache_creation_1h_input_tokens",
        ]
        # Positional construction binds by position, so a caller passing only
        # the first four counters still gets those four and the two TTL counters
        # default to zero, rather than one of them absorbing an argument.
        usage = NormalisedUsage(100, 20, 5, 7)
        assert usage.cache_read_input_tokens == 5
        assert usage.cache_creation_input_tokens == 7
        assert usage.cache_creation_5m_input_tokens == 0


# ---------------------------------------------------------------------------
# Content that must not be silently lost, and refusals that must not be
# recorded as ordinary end-of-turn
# ---------------------------------------------------------------------------

def _chat_completion(message, *, finish_reason="stop"):
    """A minimal Chat Completions `.model_dump()` fixture (no gateway routing,
    so `_from_chat_wire` can be exercised on its own)."""
    return {
        "model": "z-ai/glm-4.6v",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def _compat_adapter():
    return OpenAIAdapter(client=None, provider="openai_compat", base_url=None)


class TestContentIsNeverSilentlyLost:
    """Three paths where a response could come back empty or mislabelled while
    its output tokens were billed all the same. Each one ends downstream in the
    same wrong conclusion — "the model produced no answer" — which sends a
    reader to the prompt instead of to the filter, the refusal, or the content
    shape that actually stopped the call."""

    def test_chat_content_as_a_parts_list_is_read(self):
        # An OpenAI-compatible host may send `message.content` as the parts LIST
        # the Responses API uses rather than a plain string. Reading only the
        # string yields an empty response for a billed call.
        raw = _chat_completion({
            "role": "assistant",
            "content": [{"type": "text", "text": "{\"ok\": true}"}]})
        resp = _compat_adapter()._from_chat_wire(
            raw, canonical={}, wire={}, decoding={})

        assert [b.type for b in resp.content] == ["text"]
        assert resp.content[0].text == "{\"ok\": true}"
        assert resp.stop_reason == "end_turn"

    def test_chat_parts_list_keeps_order_and_skips_empty_parts(self):
        raw = _chat_completion({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": ""},          # nothing to say
                {"type": "output_text", "text": "second"},  # other dialect
                {"type": "image_url", "image_url": {"url": "x"}},  # not text
            ]})
        resp = _compat_adapter()._from_chat_wire(
            raw, canonical={}, wire={}, decoding={})

        assert [b.text for b in resp.content] == ["first", "second"]

    def test_chat_string_content_still_works(self):
        raw = _chat_completion({"role": "assistant", "content": "plain"})
        resp = _compat_adapter()._from_chat_wire(
            raw, canonical={}, wire={}, decoding={})
        assert [b.text for b in resp.content] == ["plain"]

    def test_content_filter_finish_is_a_refusal_not_end_turn(self):
        raw = _chat_completion(
            {"role": "assistant", "content": None},
            finish_reason="content_filter")
        resp = _compat_adapter()._from_chat_wire(
            raw, canonical={}, wire={}, decoding={})

        assert resp.stop_reason == "refusal"
        # And the canonical vocabulary is shared with the Responses path, so a
        # caller branches on one value whichever wire served the call.
        assert resp.stop_reason != "end_turn"

    def test_content_filter_outranks_a_tool_call(self):
        # A filtered response has been filtered whatever else it carried.
        raw = _chat_completion(
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "record_answer", "arguments": "{}"}}]},
            finish_reason="content_filter")
        resp = _compat_adapter()._from_chat_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.stop_reason == "refusal"

    def test_responses_refusal_part_is_surfaced_with_its_text(self):
        # A Responses `refusal` part IS the model's answer. Dropped, the caller
        # sees empty content under a stop reason saying the turn simply ended.
        raw = {
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "I cannot help with that."}]}],
            "usage": {"input_tokens": 40, "output_tokens": 8},
        }
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})

        assert resp.stop_reason == "refusal"
        assert [b.type for b in resp.content] == ["text"]
        assert resp.content[0].text == "I cannot help with that."

    def test_responses_refusal_outranks_a_tool_call(self):
        raw = {
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [
                {"type": "function_call", "call_id": "c1",
                 "name": "record_answer", "arguments": "{}"},
                {"type": "message", "content": [
                    {"type": "refusal", "refusal": "no"}]},
            ],
            "usage": {"input_tokens": 40, "output_tokens": 8},
        }
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.stop_reason == "refusal"

    def test_incomplete_for_any_other_reason_is_a_refusal(self):
        # `incomplete` with a reason other than the output cap is the endpoint
        # stopping the response for its own reasons — a refusal, not a turn that
        # finished. `test_incomplete_maps_to_max_tokens` above pins the other
        # arm; without this one the whole branch could collapse to end_turn and
        # the suite would not notice.
        raw = {"model": "gpt-5.6-sol", "status": "incomplete",
               "incomplete_details": {"reason": "content_filter"},
               "output": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.stop_reason == "refusal"

    def test_incomplete_with_no_details_is_a_refusal(self):
        raw = {"model": "gpt-5.6-sol", "status": "incomplete",
               "output": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.stop_reason == "refusal"

    def test_a_plain_completed_text_response_is_still_end_turn(self):
        # The refusal handling must not relabel ordinary answers.
        raw = {
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "here you go"}]}],
            "usage": {"input_tokens": 40, "output_tokens": 8},
        }
        resp = _openai_adapter()._from_wire(
            raw, canonical={}, wire={}, decoding={})
        assert resp.stop_reason == "end_turn"

    def test_extract_tool_call_names_the_refusal_not_the_absence(self):
        # The point of the stop reason: `extract_tool_call` reports it, so a
        # refused call reads as refused rather than as a model that said
        # nothing for no stated reason.
        from direktoro.providers import extract_tool_call

        raw = _chat_completion(
            {"role": "assistant", "content": None},
            finish_reason="content_filter")
        resp = _compat_adapter()._from_chat_wire(
            raw, canonical={}, wire={}, decoding={})
        _, error = extract_tool_call(resp, "record_answer")
        assert "refusal" in error


# ---------------------------------------------------------------------------
# SDK exception -> normalised exception, for BOTH translators
# ---------------------------------------------------------------------------
# The 5xx -> retryable mapping is the whole reason `create_message_with_retry`
# does anything during a provider outage, so both translators are held to the
# same table rather than one standing in for the other: they must classify the
# same failure into the same normalised class, and that symmetry is what lets a
# caller's retry loop stay provider-independent.

# Exactly the tuple `create_message_with_retry` catches, so "retryable" here
# means "this loop will actually retry it" and not merely "the class name says
# so".
_RETRYABLE = (ProviderRateLimitError, ProviderRetryableError)


def _status_error(sdk, cls_name, code):
    cls = getattr(sdk, cls_name)
    return cls(f"{code} from the provider", response=SimpleNamespace(
        status_code=code, headers={}, request=None), body=None)


def _translator_cases(sdk):
    """(label, sdk exception, expected direktoro class, expected retryable)."""
    return [
        ("rate limit",
         _status_error(sdk, "RateLimitError", 429),
         ProviderRateLimitError, True),
        ("5xx server error",
         _status_error(sdk, "APIStatusError", 500),
         ProviderRetryableError, True),
        ("529 overloaded",
         _status_error(sdk, "APIStatusError", 529),
         ProviderRetryableError, True),
        ("4xx bad request",
         _status_error(sdk, "APIStatusError", 400),
         ProviderError, False),
        # A connection that never established cannot have been served, so a
        # retry cannot be charged twice.
        ("connection error",
         sdk.APIConnectionError(message="connection refused", request=None),
         ProviderRetryableError, True),
        # A timeout MAY already have been served and billed while the response
        # was lost, so retrying risks paying twice and recording once. Deliberate
        # and stated; note APITimeoutError subclasses APIConnectionError, so this
        # row is also what pins the clause ORDER in the translator.
        ("timeout", sdk.APITimeoutError(request=None), ProviderError, False),
    ]


def _sdk(name):
    """The provider SDK, or SKIP. These tables construct REAL SDK exception
    objects, because a hand-rolled stand-in cannot show that the translator's
    isinstance clauses are in the right order against the real class hierarchy —
    which is precisely what keeps timeouts out of the retryable bucket."""
    return pytest.importorskip(
        name, reason=f"constructs real {name} SDK exceptions")


class TestAnthropicErrorTranslation:
    def test_table(self):
        sdk = _sdk("anthropic")
        for label, exc, expected_cls, retryable in _translator_cases(sdk):
            out = _translate_anthropic_error(exc)
            assert type(out) is expected_cls, label
            assert isinstance(out, _RETRYABLE) is retryable, label

    def test_5xx_records_its_status_code(self):
        out = _translate_anthropic_error(
            _status_error(_sdk("anthropic"), "APIStatusError", 529))
        assert out.status_code == 529

    def test_connection_error_records_no_status(self):
        # It never reached one; None is the honest value, not a stand-in.
        out = _translate_anthropic_error(
            _sdk("anthropic").APIConnectionError(message="boom", request=None))
        assert out.status_code is None

    def test_a_non_sdk_exception_passes_through_unchanged(self):
        _sdk("anthropic")
        original = ValueError("something else entirely")
        assert _translate_anthropic_error(original) is original


class TestOpenAIErrorTranslation:
    def test_table(self):
        sdk = _sdk("openai")
        for label, exc, expected_cls, retryable in _translator_cases(sdk):
            out = _translate_openai_error(exc)
            assert type(out) is expected_cls, label
            assert isinstance(out, _RETRYABLE) is retryable, label

    def test_5xx_records_its_status_code(self):
        out = _translate_openai_error(
            _status_error(_sdk("openai"), "APIStatusError", 503))
        assert out.status_code == 503

    def test_connection_error_records_no_status(self):
        out = _translate_openai_error(
            _sdk("openai").APIConnectionError(message="boom", request=None))
        assert out.status_code is None

    def test_a_non_sdk_exception_passes_through_unchanged(self):
        _sdk("openai")
        original = ValueError("something else entirely")
        assert _translate_openai_error(original) is original


class TestA429ThatNoWaitCanClear:
    """OpenAI overloads HTTP 429 for two unrelated conditions: throttling, the
    most retryable failure there is, and a spent account, the least.

    Classifying on the status alone climbs the whole ladder for a credit
    balance no wait restores. Recorded 2026-08-18 on a live review leg: four
    calls walked (2, 5, 11, 23) to exhaustion, ~41 seconds each, to arrive at
    what the first response had already said — and on a longer batch every
    subsequent call pays that again, which turns a fast failure into a slow
    one. The distinguishing evidence is never the status; it is `type` /
    `code` in the body."""

    # The body as recorded, verbatim.
    SPENT = {"message": "You have no credits remaining. Add credits to "
                        "continue using the API at "
                        "https://platform.openai.com/settings/organization/"
                        "billing/.",
             "type": "insufficient_quota", "param": None,
             "code": "credit_balance_exhausted"}

    def _rate_limit(self, body):
        sdk = _sdk("openai")
        return sdk.RateLimitError("Error code: 429", body=body,
                                  response=SimpleNamespace(
                                      status_code=429, headers={},
                                      request=None))

    def test_a_spent_account_is_not_a_rate_limit(self):
        out = _translate_openai_error(self._rate_limit(self.SPENT))
        assert not isinstance(out, ProviderRateLimitError)

    def test_a_spent_account_says_it_is_about_the_account(self):
        # The second axis. This clause has to establish that the account is at
        # fault in order to decide not to retry, and flattening that into the
        # base class throws the finding away: a caller would then see the same
        # type for a spent balance and for a malformed request, which want
        # opposite handling — one is resumable once someone tops up, the other
        # is a fault a resume hits again forever. By the time the exception
        # reaches a caller the body that distinguished them is a string.
        out = _translate_openai_error(self._rate_limit(self.SPENT))
        assert type(out) is ProviderAccountError

    def test_it_is_caught_as_a_provider_failure_like_any_other(self):
        # Backward compatible by construction: a consumer that never opts in
        # keeps catching it with the `except ProviderError` it already has.
        assert isinstance(
            _translate_openai_error(self._rate_limit(self.SPENT)),
            ProviderError)

    @pytest.mark.parametrize("cls_name,status", [("AuthenticationError", 401),
                                                 ("PermissionDeniedError",
                                                  403)])
    @pytest.mark.parametrize("sdk_name,translate", [
        ("openai", _translate_openai_error),
        ("anthropic", _translate_anthropic_error)])
    def test_credentials_are_about_the_account_on_both_wires(
            self, sdk_name, translate, cls_name, status):
        # A 401 and a 403 mean the same thing on every wire, unlike the 429
        # overload, so both translators carry these. A pause path that worked
        # for one provider's credentials and not another's would be holding a
        # distinction this package invented rather than one the providers draw.
        out = translate(_status_error(_sdk(sdk_name), cls_name, status))
        assert type(out) is ProviderAccountError

    def test_the_clause_is_ordered_before_the_general_status_branch(self):
        # AuthenticationError and PermissionDeniedError both subclass
        # APIStatusError, so moving the general clause above them would flatten
        # every credential failure into the base class again — silently, and
        # with every test but this one still passing.
        sdk = _sdk("openai")
        assert issubclass(sdk.AuthenticationError, sdk.APIStatusError)
        assert issubclass(sdk.PermissionDeniedError, sdk.APIStatusError)

    def test_a_malformed_request_is_not_about_the_account(self):
        # The distinction earns its keep only if the other side holds: a 400
        # is about the request and must NOT arrive as an account failure, or a
        # consumer pausing on this class would pause on a config fault.
        out = _translate_openai_error(
            _status_error(_sdk("openai"), "APIStatusError", 400))
        assert type(out) is ProviderError
        assert not isinstance(out, ProviderAccountError)

    def test_an_ordinary_429_still_retries(self):
        # The default is unchanged and must stay unchanged: this exemption is
        # narrow, and a throttled call is exactly what the ladder is for.
        out = _translate_openai_error(self._rate_limit(
            {"message": "Rate limit reached for gpt-5.6-terra",
             "type": "rate_limit_error", "code": "rate_limit_exceeded"}))
        assert type(out) is ProviderRateLimitError

    def test_a_429_with_no_body_at_all_still_retries(self):
        out = _translate_openai_error(self._rate_limit(None))
        assert type(out) is ProviderRateLimitError

    @pytest.mark.parametrize("code", ["insufficient_quota",
                                      "credit_balance_exhausted",
                                      "billing_hard_limit_reached"])
    def test_each_spent_account_code_is_refused(self, code):
        out = _translate_openai_error(
            self._rate_limit({"message": "no credit", "code": code}))
        assert type(out) is ProviderAccountError

    def test_the_envelope_is_read_where_a_client_did_not_unwrap_it(self):
        # The SDK's concrete client unwraps `{"error": {...}}` before building
        # the exception, but this package supports an injected client, and the
        # base class does not unwrap. Reading only the top level would fall
        # back to a retry on the nested shape.
        out = _translate_openai_error(
            self._rate_limit({"error": dict(self.SPENT)}))
        assert type(out) is ProviderAccountError

    def test_the_ladder_does_not_climb_for_it(self):
        # The point of the classification, end to end: no sleep, one attempt.
        class _Spent:
            def __init__(self, exc):
                self.exc = exc
                self.calls = 0

            def create_message(self, **kwargs):
                self.calls += 1
                raise _translate_openai_error(self.exc)

        adapter = _Spent(self._rate_limit(self.SPENT))
        slept = []
        with pytest.raises(ProviderError):
            create_message_with_retry(adapter, _sleep=slept.append)
        assert adapter.calls == 1
        assert slept == []


class TestTranslatorsSurviveAnAbsentSDK:
    """This package supports being installed without its provider SDKs, with the
    caller injecting the client. On that shape an unguarded `import` inside a
    translator replaces every provider failure with a ModuleNotFoundError, which
    `except ProviderError` does not catch and `create_message_with_retry` does
    not retry — the original error is destroyed on the one install shape that
    most needs it intact."""

    @staticmethod
    def _poison(monkeypatch, name):
        for mod in list(sys.modules):
            if mod == name or mod.startswith(name + "."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setitem(sys.modules, name, None)  # `import name` -> ImportError

    def test_anthropic_translator_returns_the_original(self, monkeypatch):
        self._poison(monkeypatch, "anthropic")
        original = RuntimeError("the real provider failure")
        assert _translate_anthropic_error(original) is original

    def test_openai_translator_returns_the_original(self, monkeypatch):
        self._poison(monkeypatch, "openai")
        original = RuntimeError("the real provider failure")
        assert _translate_openai_error(original) is original

    def test_the_adapter_propagates_the_original_failure(self, monkeypatch):
        # End to end: with no SDK importable, the failure a stubbed client
        # raises must reach the caller as itself.
        self._poison(monkeypatch, "anthropic")
        boom = RuntimeError("upstream said no")

        class _BoomClient:
            def __init__(self):
                self.messages = SimpleNamespace(stream=self._boom)

            def _boom(self, **kwargs):
                raise boom

        with pytest.raises(RuntimeError) as exc:
            AnthropicAdapter(_BoomClient()).create_message(
                model="claude-sonnet-4-6", system="S", messages=[],
                max_tokens=10)
        assert exc.value is boom


class TestRetryLoopActsOnTheTranslatedClass:
    """The classification above is only worth anything because
    `create_message_with_retry` acts on it. These hold the join."""

    def test_a_translated_5xx_is_retried(self):
        sdk = _sdk("anthropic")
        failures = [_translate_anthropic_error(
            _status_error(sdk, "APIStatusError", 529))] * 2

        class _Adapter:
            calls = 0

            def create_message(self, **kwargs):
                type(self).calls += 1
                if failures:
                    raise failures.pop(0)
                return "ok"

        slept = []
        adapter = _Adapter()
        assert create_message_with_retry(adapter, _sleep=slept.append) == "ok"
        assert _Adapter.calls == 3
        assert slept == list(RETRY_BACKOFF_SECONDS[:2])

    def test_a_translated_timeout_is_not_retried(self):
        sdk = _sdk("anthropic")
        translated = _translate_anthropic_error(sdk.APITimeoutError(request=None))

        class _Adapter:
            calls = 0

            def create_message(self, **kwargs):
                type(self).calls += 1
                raise translated

        slept = []
        with pytest.raises(ProviderError):
            create_message_with_retry(_Adapter(), _sleep=slept.append)
        assert _Adapter.calls == 1
        assert slept == []


# ---------------------------------------------------------------------------
# The Anthropic stream must be drained before the final message is taken
# ---------------------------------------------------------------------------

class _DrainCheckingStream:
    """A streaming context manager that yields real chunks and refuses to hand
    over the final message until they have all been consumed.

    The SDK helper accumulates the final message FROM the stream, so a caller
    that reaches `get_final_message()` without draining `text_stream` is asking
    for a message that has not finished arriving. The stubs elsewhere in this
    file expose an empty `text_stream`, which is why they cannot see the drain
    loop at all — an empty iterator is drained whether or not anything iterates
    it."""

    def __init__(self, resp, chunks=("par", "tial", " text")):
        self._resp = resp
        self._chunks = list(chunks)
        self.consumed = []
        self.text_stream = self._emit()

    def _emit(self):
        for chunk in self._chunks:
            self.consumed.append(chunk)
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if self.consumed != self._chunks:
            raise AssertionError(
                "get_final_message() was reached with text_stream undrained "
                f"({self.consumed!r} of {self._chunks!r} consumed); the final "
                "message is accumulated from the stream, so it is not complete "
                "yet.")
        return self._resp


class TestAnthropicStreamIsDrained:
    def test_the_stream_is_consumed_before_the_final_message_is_taken(self):
        response = _anth_response()
        stream = _DrainCheckingStream(response)
        sink = {}

        class _Client:
            def __init__(self):
                self.messages = SimpleNamespace(stream=self._stream)

            def _stream(self, **kwargs):
                sink["wire"] = kwargs
                return stream

        resp = AnthropicAdapter(_Client()).create_message(
            model="claude-opus-4-8", system="S",
            messages=[{"role": "user", "content": "hi"}], max_tokens=4096)

        assert stream.consumed == ["par", "tial", " text"]
        # raw_response is a plain dict (wire_log.response_to_dict), never the
        # SDK object, so an audit log can serialise it without SDK knowledge.
        assert resp.raw_response["content"] == [{"type": "text", "text": "hi"}]
        assert resp.content[0].text == "hi"
