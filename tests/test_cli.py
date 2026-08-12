"""Tests for the consolidated live-smoke harness (`direktoro.cli`).

Hermetic like the rest of the suite: the live path is exercised against a
stubbed adapter injected by monkeypatching `build_adapter`, and the missing-key
path against an environment with every provider key cleared. The network guard
in conftest.py backstops all of it.
"""

from types import SimpleNamespace

import pytest

from direktoro import MODEL_REGISTRY, cli
from direktoro.providers import (
    MissingAPIKey, NormalisedResponse, NormalisedUsage)


def _clear_provider_keys(monkeypatch):
    for info in MODEL_REGISTRY.values():
        monkeypatch.delenv(info.api_key_env, raising=False)


def _tool_use_block(name, tool_input):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input)


def _response(blocks, *, usage=None, stop_reason="tool_use"):
    return NormalisedResponse(
        content=blocks, usage=usage or NormalisedUsage(
            input_tokens=100, output_tokens=20),
        stop_reason=stop_reason)


class _StubAdapter:
    def __init__(self, response):
        self._response = response
        self.requests = []

    def create_message(self, **kwargs):
        self.requests.append(kwargs)
        return self._response


# Stands in for the parsed argparse namespace, which always carries every flag
# it defines — `--temperature` included, whether or not the caller gave one.
_ARGS = SimpleNamespace(max_tokens=2048, temperature=0.0)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

class TestSelection:
    def test_one_model_per_key_env(self):
        selected = cli.select_default_models()
        envs = {model_info.api_key_env for model_info in
                MODEL_REGISTRY.values()}
        assert len(selected) == len(envs)
        labels = [label for label, _ in selected]
        assert len(labels) == len(set(labels))

    def test_every_selected_model_is_the_named_representative(self):
        # The model a --live run bills is named, not derived. No ordering of
        # this registry carries meaning, so a derived choice would move
        # silently the moment an entry was added.
        for _label, model_id in cli.select_default_models():
            env = MODEL_REGISTRY[model_id].api_key_env
            assert cli._SMOKE_REPRESENTATIVES[env] == model_id

    def test_every_named_representative_is_live_and_registered(self):
        for env, model_id in cli._SMOKE_REPRESENTATIVES.items():
            assert model_id in MODEL_REGISTRY, f"{env}: {model_id} is not a model"
            assert not MODEL_REGISTRY[model_id].retired, (
                f"{env}: {model_id} is retired and would 404 on a live call")
            assert MODEL_REGISTRY[model_id].api_key_env == env

    def test_a_provider_with_no_named_representative_fails_loudly(
            self, monkeypatch):
        # Adding a provider must force the choice to be made; a silent
        # fallback would put a billable run on a model nobody chose.
        trimmed = {k: v for k, v in cli._SMOKE_REPRESENTATIVES.items()
                   if k != "ANTHROPIC_API_KEY"}
        monkeypatch.setattr(cli, "_SMOKE_REPRESENTATIVES", trimmed)
        with pytest.raises(ValueError, match="no smoke representative"):
            cli.select_default_models()

    def test_a_representative_naming_an_unknown_model_fails_loudly(
            self, monkeypatch):
        broken = dict(cli._SMOKE_REPRESENTATIVES,
                      OPENAI_API_KEY="not-a-model")
        monkeypatch.setattr(cli, "_SMOKE_REPRESENTATIVES", broken)
        with pytest.raises(ValueError, match="not in the registry"):
            cli.select_default_models()

    def test_selection_is_deterministic(self):
        assert cli.select_default_models() == cli.select_default_models()

    def test_models_override_resolves_and_validates(self):
        selected = cli.resolve_models("claude-opus-4-8, gpt-5.6-terra")
        assert [m for _, m in selected] == [
            "claude-opus-4-8", "gpt-5.6-terra"]
        assert [lbl for lbl, _ in selected] == ["anthropic", "openai"]

    def test_unknown_model_id_fails_loudly(self):
        with pytest.raises(ValueError):
            cli.resolve_models("claude-opus-4-8,not-a-model")

    def test_every_live_registry_id_is_nameable(self):
        # --models is validated against the registry, so an entry the table
        # carries but the gate refuses would be unreachable from the CLI while
        # looking perfectly available in `known_models`. Asserted over the whole
        # table rather than a chosen id, so a new entry is covered on arrival.
        for model_id, info in MODEL_REGISTRY.items():
            if info.retired:
                continue
            assert cli.resolve_models(model_id) == [
                (cli.provider_label(model_id), model_id)], model_id


class TestRetiredGate:
    """`registry.Model.retired` is enforced at the CLI's new-run acceptance
    gate, and only there: a withdrawn id must be refused at config load rather
    than 404 mid-run."""

    RETIRED_ID = "claude-sonnet-4-20250514"

    def test_retired_id_is_refused_for_a_new_run(self):
        with pytest.raises(ValueError, match="retired"):
            cli.resolve_models(self.RETIRED_ID)

    def test_retired_id_is_refused_alongside_live_ones(self):
        # The whole selection fails; a retired id is never silently dropped
        # from a matrix that otherwise looks fine.
        with pytest.raises(ValueError, match="retired"):
            cli.resolve_models(f"claude-opus-5,{self.RETIRED_ID}")

    def test_retired_id_still_resolves_outside_the_gate(self):
        # The gate is the ONLY place the flag bites: provenance for a run that
        # already happened must keep working.
        assert MODEL_REGISTRY[self.RETIRED_ID].retired is True
        assert cli.provider_label(self.RETIRED_ID) == "anthropic"

    def test_retired_entries_are_never_auto_selected(self):
        selected = {model_id for _, model_id in cli.select_default_models()}
        for model_id in selected:
            assert MODEL_REGISTRY[model_id].retired is False, model_id

    def test_empty_models_override_fails_loudly(self):
        with pytest.raises(ValueError):
            cli.resolve_models(" , ,")


# ---------------------------------------------------------------------------
# Canonical request
# ---------------------------------------------------------------------------

class TestBuildRequest:
    def test_request_is_anthropic_shaped(self):
        request = cli.build_request(
            "claude-opus-4-8", max_tokens=2048, sampling={"temperature": 0.0})
        assert request["system"][0]["type"] == "text"
        assert request["messages"][0]["role"] == "user"
        assert request["tools"] == [cli.RECORD_ANSWER_TOOL]
        assert request["tool_choice"] == {
            "type": "tool", "name": cli.TOOL_NAME}

    def test_tool_choice_shaped_per_wire(self):
        responses_choice = cli.build_request(
            "gpt-5.6-terra", max_tokens=1, sampling={"temperature": 0.0})["tool_choice"]
        assert responses_choice == {
            "type": "function", "name": cli.TOOL_NAME}
        # A FORCING Chat Completions model (routed Qwen) rides the nested
        # `function` shape.
        chat_choice = cli.build_request(
            "qwen/qwen3-vl-235b-a22b-instruct",
            max_tokens=1, sampling={"temperature": 0.0})["tool_choice"]
        assert chat_choice == {
            "type": "function", "function": {"name": cli.TOOL_NAME}}

    def test_a_non_forcing_model_degrades_to_auto(self):
        # A model whose endpoint 404s a forced named tool_choice gets
        # tool_choice "auto" instead, so the request routes at all — and the
        # model MAY then decline to call the tool, which is the cost of the
        # degrade. Driven off the registry flag rather than a hardcoded list,
        # so it follows the table.
        non_forcing = [model_id for model_id, info in MODEL_REGISTRY.items()
                       if not info.forced_tool_choice]
        assert non_forcing, "no non-forcing entry to exercise the degrade"
        for m in non_forcing:
            choice = cli.build_request(
                m, max_tokens=1, sampling={"temperature": 0.0})["tool_choice"]
            assert choice == {"type": "auto"}, m


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_is_default_and_prints_every_model(self, capsys):
        assert cli.main([]) == 0
        out = capsys.readouterr().out
        for _, model_id in cli.select_default_models():
            assert model_id in out

    def test_dry_run_needs_no_keys(self, capsys, monkeypatch):
        _clear_provider_keys(monkeypatch)
        assert cli.main(["--dry-run"]) == 0

    def test_usage_error_exits_2(self, capsys):
        assert cli.main(["--models", "not-a-model"]) == 2
        assert "error:" in capsys.readouterr().err


class TestThinkingIsReachableFromTheFlags:
    """`--thinking` has to be usable on the models that have a thinking surface,
    and a temperature is what stops it.

    On the 4.6-generation and earlier endpoints a temperature and active
    thinking cannot both be sent — the pair is a 400, so
    `resolved_decoding_params` refuses it up front — and those are exactly the
    entries that accept a temperature at all. A `--temperature` that could not
    be left unset would therefore make `--thinking` refusable on every model it
    can apply to, and `--thinking-budget` (whose only eligible model is the
    pre-4.6 Haiku 4.5) impossible to send at all: a flag the program offers and
    can never accept. So the temperature is omitted unless asked for, and these
    tests hold it that way."""

    BUDGET_MODEL = "claude-haiku-4-5-20251001"
    ADAPTIVE_MODEL = "claude-sonnet-4-6"

    def test_temperature_is_unset_by_default(self):
        args = cli.build_arg_parser().parse_args([])
        assert args.temperature is None

    def test_a_temperature_is_accepted_when_asked_for(self):
        args = cli.build_arg_parser().parse_args(["--temperature", "0.4"])
        assert args.temperature == pytest.approx(0.4)

    def test_default_run_sends_no_temperature(self, capsys):
        assert cli.main(["--models", self.ADAPTIVE_MODEL]) == 0
        assert '"temperature"' not in capsys.readouterr().out

    def test_thinking_budget_can_actually_be_sent(self, capsys):
        # The flag the CLI advertises for pre-4.6 models, exercised end to end
        # on the only pre-4.6 entry. Exit 0 means nothing was refused.
        assert cli.main([
            "--models", self.BUDGET_MODEL, "--thinking", "budget",
            "--thinking-budget", "2048"]) == 0
        out = capsys.readouterr().out
        assert '"budget_tokens": 2048' in out or '"budget_tokens":2048' in out

    def test_adaptive_thinking_can_actually_be_sent(self, capsys):
        assert cli.main([
            "--models", self.ADAPTIVE_MODEL, "--thinking", "adaptive"]) == 0
        assert "REFUSED" not in capsys.readouterr().out

    def test_asking_for_both_is_refused_loudly(self, capsys):
        # The refusal is deliberate: a caller that names a
        # temperature AND active thinking has asked for two things that cannot
        # both hold, and silently dropping either would change a run behind its
        # back. Non-zero exit, and the model is reported rather than the matrix
        # aborted.
        assert cli.main([
            "--models", self.ADAPTIVE_MODEL, "--thinking", "adaptive",
            "--temperature", "0.0"]) == 1
        assert "REFUSED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Live run (stubbed)
# ---------------------------------------------------------------------------

class TestLiveRun:
    def test_missing_keys_recorded_not_raised(self, capsys, monkeypatch):
        _clear_provider_keys(monkeypatch)
        assert cli.main(["--live"]) == 1
        out = capsys.readouterr().out
        assert out.count("MissingAPIKey") == len(cli.select_default_models())

    def test_happy_path_row(self, monkeypatch):
        response = _response(
            [_tool_use_block(cli.TOOL_NAME, {"answer": "Paris"})])
        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _StubAdapter(response))
        row = cli.call_one("anthropic", "claude-opus-4-8", _ARGS)
        assert row["ok"] is True
        assert row["answer"] == "Paris"
        assert row["error"] is None and row["violations"] is None

    def test_a_direct_row_reports_no_cost(self, monkeypatch):
        # A direct provider reports no cost figure and the harness has no rate
        # card to invent one from, so the cell stays empty. An empty cell is an
        # absence of a reported figure; a 0.0 would read as "this call was
        # free", which is the claim that must never be made by accident.
        response = _response(
            [_tool_use_block(cli.TOOL_NAME, {"answer": "Paris"})])
        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _StubAdapter(response))
        row = cli.call_one("anthropic", "claude-opus-4-8", _ARGS)
        assert row["ok"] is True
        assert row["cost"] is None

    def test_wrong_tool_is_a_violation(self, monkeypatch):
        response = _response(
            [_tool_use_block("other_tool", {"answer": "Paris"})])
        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _StubAdapter(response))
        row = cli.call_one("anthropic", "claude-opus-4-8", _ARGS)
        assert row["ok"] is False
        assert row["violations"]

    def test_empty_answer_is_a_violation(self, monkeypatch):
        response = _response([_tool_use_block(cli.TOOL_NAME, {"answer": ""})])
        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _StubAdapter(response))
        row = cli.call_one("anthropic", "claude-opus-4-8", _ARGS)
        assert row["ok"] is False
        assert row["violations"] == ["answer is missing or empty"]

    def test_routed_row_uses_response_reported_cost(self, monkeypatch):
        # A routed call comes back with the cost the gateway charged
        # (OpenRouter usage.cost); the row records that figure as reported,
        # plus the generation id and served upstream captured off the response.
        response = _response(
            [_tool_use_block(cli.TOOL_NAME, {"answer": "Paris"})])
        response.reported_cost = 0.000123
        response.generation_id = "gen-abc123"
        response.served_provider = "Z.AI"
        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _StubAdapter(response))
        row = cli.call_one("openrouter", "z-ai/glm-4.6v", _ARGS)
        assert row["ok"] is True
        assert row["cost"] == pytest.approx(0.000123)
        assert row["generation_id"] == "gen-abc123"
        assert row["served_provider"] == "Z.AI"

    def test_routed_missing_reported_cost_is_captured(self, monkeypatch):
        # A routed response with no reported cost is a plumbing fault (did
        # usage.include reach the gateway?); captured in the row, not raised.
        response = _response(
            [_tool_use_block(cli.TOOL_NAME, {"answer": "Paris"})])
        response.reported_cost = None
        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _StubAdapter(response))
        row = cli.call_one("openrouter", "z-ai/glm-4.6v", _ARGS)
        assert row["ok"] is False
        assert row["error"] and "reported cost" in row["error"].lower()

    def test_provider_error_is_captured_not_raised(self, monkeypatch):
        class _ExplodingAdapter:
            def create_message(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            cli, "build_adapter", lambda model_id: _ExplodingAdapter())
        row = cli.call_one("anthropic", "claude-opus-4-8", _ARGS)
        assert row["ok"] is False
        assert "boom" in row["error"]

    def test_missing_key_row(self, monkeypatch):
        def _raise(model_id):
            raise MissingAPIKey("ANTHROPIC_API_KEY is not set")

        monkeypatch.setattr(cli, "build_adapter", _raise)
        row = cli.call_one("anthropic", "claude-opus-4-8", _ARGS)
        assert row["ok"] is False
        assert "MissingAPIKey" in row["error"]
