"""direktoro.wire_log: redaction and serialisation for a consumer's audit log.

Pure and hermetic: no SDK, no network. The property under test is that the
ONLY thing redaction touches is inline base64 image bytes — every other block
passes through verbatim, because its content is what an audit log is for —
and that each wire's image shape is recognised.
"""

import base64
import hashlib

from direktoro import (
    redact_messages, redact_system, redact_wire_request, response_to_dict)

_PNG = base64.b64encode(b"not-really-a-png").decode()
_SHA = hashlib.sha256(b"not-really-a-png").hexdigest()


def _anthropic_image():
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": _PNG}}


def _stub():
    return {"type": "image_ref", "media_type": "image/png",
            "sha256": _SHA, "byte_length": len(b"not-really-a-png")}


class TestRedactMessages:
    def test_canonical_image_block_becomes_a_ref(self):
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": "look:"},
                                 _anthropic_image()]}]
        out = redact_messages(messages)
        assert out[0]["content"] == [{"type": "text", "text": "look:"},
                                     _stub()]

    def test_text_tool_use_and_tool_result_pass_verbatim(self):
        content = [
            {"type": "text", "text": "t"},
            {"type": "tool_use", "id": "x", "name": "record",
             "input": {"a": 1}},
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
        ]
        out = redact_messages([{"role": "assistant", "content": content}])
        assert out[0]["content"] == content

    def test_string_content_and_non_dict_items_pass_through(self):
        messages = [{"role": "user", "content": "plain"}, "not-a-message"]
        assert redact_messages(messages) == messages
        assert redact_messages("not-a-list") == "not-a-list"

    def test_undecodable_base64_becomes_an_error_stub_not_a_crash(self):
        bad = {"type": "image",
               "source": {"type": "base64", "media_type": "image/png",
                          "data": "!!not-base64!!"}}
        out = redact_messages([{"role": "user", "content": [bad]}])
        assert out[0]["content"][0]["_decode_error"] is True


class TestRedactSystem:
    def test_block_list_system_is_walked(self):
        out = redact_system([{"type": "text", "text": "S"},
                             _anthropic_image()])
        assert out == [{"type": "text", "text": "S"}, _stub()]

    def test_string_system_passes_through(self):
        assert redact_system("just text") == "just text"


class TestRedactWireRequest:
    def test_chat_completions_image_url_part(self):
        wire = {"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_PNG}"}}]}]}
        out = redact_wire_request(wire)
        assert out["messages"][0]["content"] == [_stub()]

    def test_responses_input_image_part(self):
        wire = {"model": "m", "input": [
            {"role": "user", "content": [
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{_PNG}"}]}]}
        out = redact_wire_request(wire)
        assert out["input"][0]["content"] == [_stub()]

    def test_canonical_wire_system_and_messages_are_both_walked(self):
        # The Anthropic wire IS the canonical request, so wire_request
        # carries `system` alongside `messages`; both are covered.
        wire = {"model": "m",
                "system": [{"type": "text", "text": "S"},
                           _anthropic_image()],
                "messages": [{"role": "user",
                              "content": [_anthropic_image()]}]}
        out = redact_wire_request(wire)
        assert out["system"][1] == _stub()
        assert out["messages"][0]["content"] == [_stub()]

    def test_a_plain_image_url_is_a_small_reference_and_stays(self):
        wire = {"messages": [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "https://example.invalid/fig1.png"}}]}]}
        assert redact_wire_request(wire) == wire
        # The Responses spelling of the same thing: a plain string URL.
        wire = {"input": [
            {"role": "user", "content": [
                {"type": "input_image",
                 "image_url": "https://example.invalid/fig1.png"}]}]}
        assert redact_wire_request(wire) == wire

    def test_odd_image_shapes_pass_through_unredacted(self):
        # A non-base64 Anthropic source and a bare-string image_url value are
        # not inline bytes; neither is stubbed and neither crashes the walk.
        blocks = [
            {"type": "image", "source": {"type": "url",
                                         "url": "https://example.invalid/f"}},
            {"type": "image_url", "image_url": "not-a-dict"},
        ]
        out = redact_messages([{"role": "user", "content": blocks}])
        assert out[0]["content"] == blocks

    def test_a_wire_with_both_keys_has_both_redacted(self):
        # No wire this package emits carries both `input` and `messages`, but
        # the function is public and a merged or hand-built wire can; a
        # redactor that stopped at the first key would fail OPEN on the
        # second.
        wire = {"input": [{"role": "user",
                           "content": [_anthropic_image()]}],
                "messages": [{"role": "user",
                              "content": [_anthropic_image()]}]}
        out = redact_wire_request(wire)
        assert out["input"][0]["content"] == [_stub()]
        assert out["messages"][0]["content"] == [_stub()]

    def test_non_message_items_pass_through(self):
        wire = {"input": [
            {"type": "function_call_output", "call_id": "c", "output": "ok"},
            {"role": "user", "content": "plain string"}]}
        assert redact_wire_request(wire) == wire

    def test_the_input_is_never_mutated(self):
        wire = {"messages": [{"role": "user",
                              "content": [_anthropic_image()]}]}
        redact_wire_request(wire)
        assert wire["messages"][0]["content"][0]["type"] == "image"


class TestResponseToDict:
    def test_dict_and_none_pass_through(self):
        assert response_to_dict(None) is None
        d = {"id": "resp_1"}
        assert response_to_dict(d) is d

    def test_model_dump_is_preferred(self):
        class _R:
            def model_dump(self):
                return {"id": "m1"}
        assert response_to_dict(_R()) == {"id": "m1"}

    def test_attribute_walk_covers_an_sdk_less_object(self):
        from types import SimpleNamespace
        # SimpleNamespace has no model_dump/dict, so the walk applies.
        r = SimpleNamespace(
            id="x", model="m", stop_reason="end_turn", stop_sequence=None,
            role="assistant", type="message",
            usage=SimpleNamespace(input_tokens=1),
            content=[SimpleNamespace(type="text", text="hi")])
        out = response_to_dict(r)
        assert out["model"] == "m"
        assert out["content"] == [{"type": "text", "text": "hi"}]

    def test_the_v1_dict_branch_is_taken(self):
        class _V1:
            def dict(self):
                return {"id": "v1"}
        assert response_to_dict(_V1()) == {"id": "v1"}

    def test_json_mode_is_preferred_when_the_dump_takes_it(self):
        # mode="json" renders leaf types as JSON natives so the result
        # survives a consumer's plain json.dumps; a dump that does not take
        # the keyword falls back to the plain call.
        class _R:
            def model_dump(self, mode=None):
                return {"mode": mode}
        assert response_to_dict(_R()) == {"mode": "json"}

        class _Old:
            def model_dump(self):
                return {"id": "old"}
        assert response_to_dict(_Old()) == {"id": "old"}

    def test_never_raises(self):
        class _Hostile:
            def model_dump(self):
                raise RuntimeError("boom")
        out = response_to_dict(_Hostile())
        assert "_serialisation_error" in out

    def test_an_unreadable_response_is_distinguishable_from_an_empty_one(
            self):
        # An object the walk cannot faithfully represent (no `.content`)
        # lands in the error stub, never in `content: []` — "the model
        # emitted nothing" and "we could not read the response" must stay
        # distinguishable in the record.
        from types import SimpleNamespace
        r = SimpleNamespace(id="x", model="m", stop_reason=None,
                            stop_sequence=None, role=None, type=None,
                            usage=None)
        out = response_to_dict(r)
        assert "_serialisation_error" in out
        assert "content" not in out
